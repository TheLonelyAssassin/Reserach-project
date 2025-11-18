import os, time, random, argparse, warnings, csv,torch
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module=r"albumentations\.check_version")
from pathlib import Path
from aiaf import AIAF_Tiny
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from iresnet_arcface import iresnet100
from attack_wrappers import MaskOverlay, FastAttackMixer,PGDAttack,CWAWrapper,AdvPatchWrapper

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def make_embed_pgd(arc, eps=8/255, alpha=(8/255)/4, iters=5):
    """PGD attack in embedding space (label-free)."""
    def attack(x):
        x0 = x.detach()
        z0 = l2n(arc(to_arcface_range(x0))).detach()
        xa = x0.clone()
        for _ in range(iters):
            xa.requires_grad_(True)
            z = l2n(arc(to_arcface_range(xa)))
            loss = torch.cosine_similarity(z, z0, dim=1).mean()
            g = torch.autograd.grad(loss, xa, retain_graph=False, create_graph=False)[0]
            xa = xa - alpha * g.sign()
            xa = torch.min(torch.max(xa, x0 - eps), x0 + eps)
            xa = xa.clamp(0, 1).detach()
        return xa
    return attack


def make_embed_cw(arc, eps=8/255, iters=30, step=None, c=1e-2):
    """C&W attack in embedding space (label-free)."""
    if step is None:
        step = eps / 5
    def attack(x):
        x0 = x.detach()
        z0 = l2n(arc(to_arcface_range(x0))).detach()
        xa = x0.clone()
        for _ in range(iters):
            xa.requires_grad_(True)
            z = l2n(arc(to_arcface_range(xa)))
            l_cos = torch.cosine_similarity(z, z0, dim=1).mean()
            l_l2 = ((xa - x0).flatten(1).norm(p=2, dim=1) ** 2).mean()
            loss = l_cos + c * l_l2
            g = torch.autograd.grad(loss, xa, retain_graph=False, create_graph=False)[0]
            xa = xa - step * g.sign()
            xa = torch.min(torch.max(xa, x0 - eps), x0 + eps)
            xa = xa.clamp(0, 1).detach()
        return xa
    return attack


def make_patch_overlay(frac=0.15, seed=None):
    """Random noise patch overlay (label-free)."""
    def attack(x):
        B, C, H, W = x.shape
        k = max(1, int(round(frac * min(H, W))))
        xa = x.clone()
        for b in range(B):
            top = random.randrange(0, max(1, H - k + 1))
            left = random.randrange(0, max(1, W - k + 1))
            noise = torch.rand((C, k, k), device=x.device, dtype=x.dtype)
            xa[b, :, top:top+k, left:left+k] = noise
        return xa.clamp(0, 1)
    return attack


def make_mask_overlay(mask_path, alpha=0.5):
    """Mask overlay (label-free)."""
    cache = {"mask": None}
    def attack(x):
        nonlocal cache
        if cache["mask"] is None:
            m = torchvision.io.read_image(mask_path).float() / 255.0
            if m.shape[0] == 1:
                m = m.repeat(3, 1, 1)
            cache["mask"] = m
        m = cache["mask"].to(x.device, dtype=x.dtype)
        if m.shape[-2:] != x.shape[-2:]:
            m_resized = F.interpolate(m.unsqueeze(0), x.shape[-2:], mode="bilinear", align_corners=False)[0]
        else:
            m_resized = m
        base = torch.zeros_like(x)
        xa = (1 - alpha * m_resized.unsqueeze(0)) * x + (alpha * m_resized.unsqueeze(0)) * base
        return xa.clamp(0, 1)
    return attack


# =============== EVALUATION FUNCTIONS ===============
@torch.no_grad()
def evaluate_embedding_consistency(aiaf, arc, loader, device, max_batches=12):
    """Evaluate embedding consistency (no classification needed)."""
    aiaf.eval()
    arc.eval()
    
    cos_sum = 0.0
    batches = 0
    
    for xb, _ in loader:
        if batches >= max_batches:
            break
        batches += 1
        
        xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
        
        emb_clean = l2n(arc(to_arcface_range(xb)))
        
        recon, *_ = aiaf(xb)
        recon = recon.clamp(0, 1)
        if recon.shape[-2:] != (112, 112):
            recon = F.interpolate(recon, (112, 112), mode="bilinear", align_corners=False)
        emb_def = l2n(arc(to_arcface_range(recon)))
        
        cos = F.cosine_similarity(emb_clean, emb_def, dim=1).mean()
        cos_sum += cos.item()
    
    return cos_sum / max(batches, 1)


def _tar_at_far(fpr, tpr, far=1e-2):
    if far <= fpr[0]:
        return float(tpr[0])
    if far >= fpr[-1]:
        return float(tpr[-1])
    i = np.searchsorted(fpr, far)
    x0, x1 = fpr[i-1], fpr[i]
    y0, y1 = tpr[i-1], tpr[i]
    w = (far - x0) / max(1e-12, (x1 - x0))
    return float(y0 + w * (y1 - y0))


def _eer(fpr, tpr):
    fnr = 1 - tpr
    i = np.argmin(np.abs(fpr - fnr))
    return float((fpr[i] + fnr[i]) / 2)


def _metrics_from_pairs(sims, labels):
    if len(labels) == 0:
        return {'auc': 0.5, 'eer': 1.0, 'tar@1e-2': 0.0, 'tar@1e-3': 0.0}
    fpr, tpr, _ = roc_curve(labels, sims)
    return {
        'auc': float(auc(fpr, tpr)),
        'eer': _eer(fpr, tpr),
        'tar@1e-2': _tar_at_far(fpr, tpr, 1e-2),
        'tar@1e-3': _tar_at_far(fpr, tpr, 1e-3),
    }


def eval_adversarial_openset(aiaf, arc, val_loader, device, dev_ids, mask_path,
                              pgd_eps=16/255, pgd_alpha=(16/255)/4, pgd_iters=40,
                              cw_eps=16/255, cw_iters=100, cw_c=1e-2,
                              patch_frac=0.25, mask_alpha=0.7, max_batches=None):
    """Evaluate adversarial robustness on open-set (unseen IDs)."""
    aiaf.eval()
    arc.eval()
    
    # Define attacks FIRST
    atk = {
        "pgd_embed": PGDAttack(arc, eps=pgd_eps, alpha=pgd_alpha, iters=pgd_iters),
        "cw_embed": CWAWrapper(arc, eps=cw_eps, iters=cw_iters, c=cw_c),
        "patch_olay": AdvPatchWrapper(frac=patch_frac),
        "mask_olay": MaskOverlay(mask_path, alpha=mask_alpha),
    }
    
    embs = {}
    batches_processed = 0
    
    # SINGLE loop with limit
    for xb, yb in val_loader:
        if max_batches is not None and batches_processed >= max_batches:
            break
        batches_processed += 1
        
        xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        
        dev_ids_tensor = torch.tensor(list(dev_ids), dtype=torch.long, device=device)
        mask = torch.isin(yb, dev_ids_tensor)
        if not mask.any():
            continue
        
        xb_dev = xb[mask]
        yb_dev = yb[mask]
        
        with torch.no_grad():
            z_clean = l2n(arc(to_arcface_range(xb_dev)))
        
        for zc, y in zip(z_clean, yb_dev):
            y = int(y.item())
            if y not in embs:
                embs[y] = {'clean': [], 'adv': {}}
            embs[y]['clean'].append(zc.detach().cpu())
        
        for name, fn in atk.items():
            with torch.enable_grad():
                xa = fn(xb_dev)
            
            with torch.no_grad():
                recon_adv, *_ = aiaf(xa)
                recon_adv = recon_adv.clamp(0, 1)
                if recon_adv.shape[-2:] != (112, 112):
                    recon_adv = F.interpolate(recon_adv, (112, 112), mode="bilinear", align_corners=False)
                z_adv = l2n(arc(to_arcface_range(recon_adv)))
            
            for za, y in zip(z_adv, yb_dev):
                y = int(y.item())
                if name not in embs[y]['adv']:
                    embs[y]['adv'][name] = []
                embs[y]['adv'][name].append(za.detach().cpu())
    
    print(f"[INFO] Adversarial eval: processed {batches_processed} batches")
    out = {}
    for name in atk.keys():
        sims, labels = [], []
        ids = [k for k, v in embs.items() if name in v['adv'] and len(v['adv'][name]) >= 2 and len(v['clean']) >= 2]
        for k in ids:
            c, a = embs[k]['clean'], embs[k]['adv'][name]
            m = min(len(c), len(a))
            for i in range(m - 1):
                sims.append(float((c[i] * a[i + 1]).sum().item()))
                labels.append(1)
        ids2 = [k for k, v in embs.items() if name in v['adv'] and len(v['adv'][name]) >= 1 and len(v['clean']) >= 1]
        for i in range(min(len(ids2), 2000)):
            a_id = ids2[i % len(ids2)]
            b_id = ids2[(i * 31 + 11) % len(ids2)]
            if a_id == b_id:
                continue
            sims.append(float((embs[a_id]['clean'][0] * embs[b_id]['adv'][name][0]).sum().item()))
            labels.append(0)
        cos_preserve = []
        for k in ids2:
            clean_list = embs[k]['clean']
            adv_list = embs[k]['adv'][name]
            for c, a in zip(clean_list, adv_list):
                cos_preserve.append(float((c * a).sum().item()))
        
        out[name] = _metrics_from_pairs(np.array(sims, np.float32), np.array(labels, np.int32))
        out[name]['cos_mean'] = float(np.mean(sims)) if len(sims) else 0.0
        out[name]['cos_preserve'] = float(np.mean(cos_preserve)) if len(cos_preserve) else 0.0 
    return out
#  UTILS
def safe_img(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    return x.clamp(0.0, 1.0)


def is_finite_tensor(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item()


def get_memory_usage():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0


def to_arcface_range(x):
    return x * 2 - 1


def l2n(z, eps=1e-9):
    return z / (z.norm(dim=1, keepdim=True) + eps)


#  DATASET 
class ImgFolderNoTF(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.samples = []
        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            d = os.path.join(root, c)
            for fn in os.listdir(d):
                if fn.lower().endswith(('.jpg','.jpeg','.png','.webp','.bmp')):
                    self.samples.append((os.path.join(d, fn), self.class_to_idx[c]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        x = torchvision.io.read_image(p).float() / 255.0
        if x.shape[1:] != (112, 112):
            raise ValueError(f"Expected 112x112 image, got {tuple(x.shape[1:])} at {p}")
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        return x, y


def split_dataset_by_id(root, val_id_frac=0.2, seed=42):
    ds = ImgFolderNoTF(root)
    classes = sorted(list(ds.class_to_idx.values()))
    rng = random.Random(seed)
    rng.shuffle(classes)
    n_val = max(1, int(round(len(classes) * val_id_frac)))
    val_ids = set(classes[:n_val])
    tr_idx, va_idx = [], []
    for i, (_, y) in enumerate(ds.samples):
        (va_idx if y in val_ids else tr_idx).append(i)
    tr = Subset(ds, tr_idx)
    va = Subset(ds, va_idx)
    return tr, va, len(ds.class_to_idx), val_ids
#  ATTACK SCHEDULE 
def attack_schedule(epoch: int, total_epochs: int, warmup_complete: bool):
    if not warmup_complete:
        return {}, 1.0, 12.0, 0.0
    t = epoch / max(1, (total_epochs - 1))
    start_cw, start_patch = 0.08, 0.12
    enabled = ["pgd", "mask"]
    if t >= start_cw:
        enabled.append("cw")
    if t >= start_patch:
        enabled.append("patch")
    w = {k: 1.0 / len(enabled) for k in enabled}
    p_clean = max(0.20, 0.60 - 0.70 * t)
    alpha_id = 20.0 
    beta_adv = 1.5 * t
    return w, float(p_clean), float(alpha_id), float(beta_adv)

def create_stratified_random_schedule(ranges, n_batches, n_bins=5):
    batches_per_bin = n_batches // n_bins
    remainder = n_batches % n_bins
    schedule = []
    
    for bin_idx in range(n_bins):
        pgd_eps_low = ranges['pgd']['eps'][0] + bin_idx * (ranges['pgd']['eps'][1] - ranges['pgd']['eps'][0]) / n_bins
        pgd_eps_high = ranges['pgd']['eps'][0] + (bin_idx + 1) * (ranges['pgd']['eps'][1] - ranges['pgd']['eps'][0]) / n_bins
        
        cw_iter_low = ranges['cw']['iters'][0] + bin_idx * (ranges['cw']['iters'][1] - ranges['cw']['iters'][0]) // n_bins
        cw_iter_high = ranges['cw']['iters'][0] + (bin_idx + 1) * (ranges['cw']['iters'][1] - ranges['cw']['iters'][0]) // n_bins
        
        patch_low = ranges['patch']['frac'][0] + bin_idx * (ranges['patch']['frac'][1] - ranges['patch']['frac'][0]) / n_bins
        patch_high = ranges['patch']['frac'][0] + (bin_idx + 1) * (ranges['patch']['frac'][1] - ranges['patch']['frac'][0]) / n_bins
        
        mask_low = ranges['mask']['alpha'][0] + bin_idx * (ranges['mask']['alpha'][1] - ranges['mask']['alpha'][0]) / n_bins
        mask_high = ranges['mask']['alpha'][0] + (bin_idx + 1) * (ranges['mask']['alpha'][1] - ranges['mask']['alpha'][0]) / n_bins
        
        n_samples = batches_per_bin + (1 if bin_idx < remainder else 0)
        
        for _ in range(n_samples):
            pgd_eps = random.uniform(pgd_eps_low, pgd_eps_high)
            schedule.append({
                'pgd': {'eps': pgd_eps, 'alpha': pgd_eps / 4.0, 'iters': random.randint(ranges['pgd']['iters'][0], ranges['pgd']['iters'][1])},
                'cw': {'iters': random.randint(max(50, int(cw_iter_low)), int(cw_iter_high)), 'conf': random.uniform(ranges['cw']['conf'][0], ranges['cw']['conf'][1]), 'c': random.uniform(ranges['cw']['c'][0], ranges['cw']['c'][1])},
                'patch': {'frac': random.uniform(patch_low, patch_high), 'iters': random.randint(ranges['patch']['iters'][0], ranges['patch']['iters'][1])},
                'mask': {'alpha': random.uniform(mask_low, mask_high)}
            })
    
    random.shuffle(schedule)
    return schedule


def _make_attack_plan(n_batches: int, p_clean: float, weights: dict, seed: int | None = None):
    names = list(weights.keys())
    total_w = sum(float(weights[k]) for k in names) or 1.0
    probs = [float(weights[k]) / total_w for k in names]
    n_clean = int(round(p_clean * n_batches))
    n_adv = max(0, n_batches - n_clean)
    counts = [int(round(p * n_adv)) for p in probs]
    diff = n_adv - sum(counts)
    if diff != 0:
        sign = 1 if diff > 0 else -1
        for i in range(abs(diff)):
            counts[i % len(counts)] += sign
    plan = ["clean"] * n_clean
    for name, c in zip(names, counts):
        plan += [name] * max(0, c)
    rng = random.Random(seed) if seed is not None else random
    rng.shuffle(plan)
    if len(plan) < n_batches:
        plan += ["clean"] * (n_batches - len(plan))
    elif len(plan) > n_batches:
        plan = plan[:n_batches]
    return plan


def _make_frozen_pairs(va_subset, heldout_ids, pairs_per_id=150, pos_frac=0.5, seed=1337):
    ds = va_subset.dataset if hasattr(va_subset, 'dataset') else va_subset
    indices = getattr(va_subset, 'indices', list(range(len(va_subset))))
    val_local_to_label = []
    for global_i in indices:
        y = ds.samples[global_i][1]
        val_local_to_label.append(y)

    held_ids = sorted([i for i in set(val_local_to_label) if i in heldout_ids])
    id2locals = {i: [] for i in held_ids}
    for li, y in enumerate(val_local_to_label):
        if y in id2locals:
            id2locals[y].append(li)

    rng = random.Random(seed)
    pos_per_id = int(round(pairs_per_id * pos_frac))
    neg_per_id = pairs_per_id - pos_per_id

    pairs = []
    for cid in held_ids:
        locs = id2locals[cid]
        if not locs:
            continue
        if len(locs) >= 2 and pos_per_id > 0:
            for _ in range(pos_per_id):
                a, b = rng.sample(locs, 2) if len(locs) >= 2 else (locs[0], locs[0])
                if a == b and len(locs) > 1:
                    b = (b + 1) % len(locs)
                pairs.append({'i': int(a), 'j': int(b), 'y': 1})
        for _ in range(max(0, neg_per_id)):
            a = rng.choice(locs)
            other_ids = [j for j in held_ids if j != cid and len(id2locals[j]) > 0]
            if not other_ids:
                continue
            cid2 = rng.choice(other_ids)
            b = rng.choice(id2locals[cid2])
            pairs.append({'i': int(a), 'j': int(b), 'y': 0})

    meta = {'pairs_per_id': pairs_per_id, 'pos_frac': pos_frac, 'seed': seed, 'num_ids': len(held_ids), 'total_pairs': len(pairs)}
    return {'pairs': pairs, 'ids': held_ids, 'meta': meta}


def eval_frozen_pairs(aiaf, arc, val_loader, device, frozen_pairs):
    aiaf.eval()
    arc.eval()
    
    pairs = frozen_pairs['pairs']
    needed = sorted({int(p['i']) for p in pairs} | {int(p['j']) for p in pairs})
    
    @torch.no_grad()
    def _embed_clean(x):
        return l2n(arc(to_arcface_range(x)))
    
    @torch.no_grad()
    def _embed_def(x):
        recon, *_ = aiaf(x)
        recon = recon.clamp(0, 1)
        if recon.shape[-2:] != (112, 112):
            recon = F.interpolate(recon, (112, 112), mode="bilinear", align_corners=False)
        return l2n(arc(to_arcface_range(recon)))
    
    base_ds = val_loader.dataset
    mini_ds = Subset(base_ds, needed)
    mini_loader = DataLoader(mini_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
    
    zc, zd = {}, {}
    cursor = 0
    with torch.no_grad():
        for xb, _ in mini_loader:
            xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
            z_clean = _embed_clean(xb)
            z_def = _embed_def(xb)
            b = xb.size(0)
            for k in range(b):
                li = needed[cursor + k]
                zc[li] = z_clean[k].detach().cpu()
                zd[li] = z_def[k].detach().cpu()
            cursor += b
    
    def _cos(a, b):
        return float((a * b).sum().item())
    
    sims_cd, y_cd = [], []
    for p in pairs:
        i = int(p['i'])
        j = int(p['j'])
        y = int(p['y'])
        if (i in zd) and (j in zc):
            sims_cd.append(_cos(zd[i], zc[j]))
            y_cd.append(y)
    
    sims_cd = np.array(sims_cd, np.float32)
    y_cd = np.array(y_cd, np.int32)
    
    inter = sorted(set(zd.keys()) & set(zc.keys()))
    def _cos_np(i):
        return float((zd[i].numpy() * zc[i].numpy()).sum())
    cos_pres = float(np.mean([_cos_np(i) for i in inter])) if inter else 0.0
    
    if len(y_cd) == 0 or (y_cd.sum() == 0) or (y_cd.sum() == len(y_cd)):
        return {'auc': 0.5, 'eer': 1.0, 'tar@1e-2': 0.0, 'tar@1e-3': 0.0, 'cos_preserve_mean': cos_pres}
    
    fpr, tpr, _ = roc_curve(y_cd, sims_cd)
    
    return {
        'auc': float(auc(fpr, tpr)),
        'eer': _eer(fpr, tpr),
        'tar@1e-2': _tar_at_far(fpr, tpr, 1e-2),
        'tar@1e-3': _tar_at_far(fpr, tpr, 1e-3),
        'cos_preserve_mean': cos_pres
    }
# RECONSTRUCTION VISUALIZATION 
@torch.no_grad()
def save_reconstruction_samples(aiaf, val_loader, device, save_path, epoch, n_samples=8):
    """
    Save reconstruction samples showing: Original | Reconstructed | Difference
    """
    aiaf.eval()
    
    x_batch, _ = next(iter(val_loader))
    x_batch = x_batch[:n_samples].to(device, memory_format=torch.channels_last)
    recon_batch, *_ = aiaf(x_batch)
    recon_batch = recon_batch.clamp(0, 1)
    diff_batch = torch.abs(x_batch - recon_batch) * 3.0
    diff_batch = diff_batch.clamp(0, 1)
    images = []
    for i in range(n_samples):
        images.extend([x_batch[i], recon_batch[i], diff_batch[i]])
    grid = make_grid(images, nrow=3, padding=2, normalize=False, pad_value=1.0)
    save_image(grid, save_path)
    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        img_orig = x_batch[i].cpu().permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img_orig)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=12, fontweight='bold')
        img_recon = recon_batch[i].cpu().permute(1, 2, 0).numpy()
        axes[i, 1].imshow(img_recon)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('Reconstructed', fontsize=12, fontweight='bold')
        img_diff = diff_batch[i].cpu().permute(1, 2, 0).numpy()
        axes[i, 2].imshow(img_diff)
        axes[i, 2].axis('off')
        if i == 0:
            axes[i, 2].set_title('Difference (×3)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_labeled.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[VIS] Saved reconstruction samples to {save_path}")
    
    aiaf.train()


#  MAIN 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--arcface_weights", type=str, default="D:/research project/implentation/arcface/ms1mv3_arcface_r100_fp16.pth")
    ap.add_argument("--mask", type=str, default="mask_template.png")
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--save_dir", type=str, default="./model_out_final")
    ap.add_argument("--ckpt_dir", type=str, default="./checkpoints_final")
    ap.add_argument("--val_id_frac", type=float, default=0.1)
    args = ap.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    vis_dir = Path(args.save_dir) / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Visualizations will be saved to: {vis_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    free_mixer = FastAttackMixer(m=4)
    tr, va, ncls, val_ids = split_dataset_by_id(args.data_root, val_id_frac=args.val_id_frac, seed=42)
    print(f"[INFO] Open-set split: {len(val_ids)} / {ncls} IDs held out")

    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=12)
    vl = DataLoader(va, batch_size=args.batch // 2, shuffle=False, num_workers=max(2, args.workers // 2), pin_memory=True, drop_last=True, persistent_workers=False, prefetch_factor=2)

    print(f"[INFO] Train: {len(tr)}, Val: {len(va)}, Batches: {len(tl)}")
    arc = iresnet100()
    arc.load_state_dict(torch.load(args.arcface_weights, map_location="cpu"), strict=False)
    arc.eval().to(device).to(memory_format=torch.channels_last)
    for p in arc.parameters():
        p.requires_grad_(False)
    aiaf = AIAF_Tiny(lambd=1.0).to(device).to(memory_format=torch.channels_last)
    val_ids_list = sorted(list(val_ids))
    dev_ids = set(val_ids_list)
    pairs_per_id = max(20, 3000 // max(1, len(dev_ids)))
    FROZEN_PAIRS_DEV = _make_frozen_pairs(va, dev_ids, pairs_per_id=pairs_per_id, pos_frac=0.5, seed=1337)
    print(f"[INFO] Frozen pairs: {FROZEN_PAIRS_DEV['meta']['num_ids']} IDs, {FROZEN_PAIRS_DEV['meta']['total_pairs']} pairs")
    dec_params = list(aiaf.decoder.parameters())
    dec_ids = {id(p) for p in dec_params}
    other_params = [p for p in aiaf.parameters() if p.requires_grad and id(p) not in dec_ids]
    
    param_groups = [
        {"params": other_params, "lr": args.lr, "weight_decay": 1e-5},
        {"params": dec_params, "lr": args.lr * 5.0, "weight_decay": 1e-5},
    ]

    opt = torch.optim.AdamW(param_groups, fused=True)
    scaler = torch.cuda.amp.GradScaler()
    TRAINING_RANGES = {
        'pgd': {'eps': (4/255, 16/255), 'iters': (10, 40)},
        'cw': {'iters': (50, 150), 'conf': (5, 30), 'c': (0.001, 0.05)},
        'patch': {'frac': (0.10, 0.35), 'iters': (40, 80)},
        'mask': {'alpha': (0.20, 0.50)}
    }

    MIN_WARMUP_EPOCHS = 3
    MAX_WARMUP_EPOCHS = 15
    COSINE_TARGET = 0.5
    warmup_complete = False
    best_cos_preserve = -1.0
    best_clean_openset_auc = -1.0
    best_adv_openset_auc = -1.0
    start_epoch = 0

    hist = {"epoch": [], "clean_def_cos_seen": []}

    # RESUME FROM CHECKPOINT
    if args.resume and os.path.isfile(args.resume):
        print(f"[INFO] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        aiaf.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        warmup_complete = checkpoint.get('warmup_complete', False)
        
        # Load best metrics if available
        best_cos_preserve = checkpoint.get('best_cos_preserve', -1.0)
        best_clean_openset_auc = checkpoint.get('best_clean_openset_auc', -1.0)
        best_adv_openset_auc = checkpoint.get('best_adv_openset_auc', -1.0)
        
        print(f"[INFO] Resumed from epoch {start_epoch}, warmup_complete={warmup_complete}")
    csv_path = os.path.join(args.save_dir, "open_set_verif_dev.csv")
    csv_mode = "a" if (args.resume and os.path.isfile(csv_path)) else "w"
    csv_file = open(csv_path, csv_mode, newline="")
    csv_writer = csv.writer(csv_file)
    if csv_mode == "w":
        csv_writer.writerow([
            "epoch", "total_loss", "rec_loss", "percep_loss", "id_loss", "adv_loss", 
            "clean_auc", "clean_eer", "clean_tar1e-2", "clean_tar1e-3", "clean_cos_preserve",
            "adv_attack", "adv_auc", "adv_eer", "adv_tar1e-2", "adv_tar1e-3", "adv_cos_mean", "adv_cos_preserve" 
        ])
        csv_file.flush()
        print("[INFO] Created new CSV file with headers")
    else:
        print(f"[INFO] Appending to existing CSV file: {csv_path}")

    with Progress(TextColumn("{task.description}"), MofNCompleteColumn(), BarColumn(), TimeElapsedColumn(), TimeRemainingColumn()) as bar:
        e_task = bar.add_task("Epoch", total=args.epoch)
        step = 0

        for epoch in range(start_epoch, args.epoch):
            t0 = time.time()
            bar.update(e_task, description=f"Epoch {epoch + 1}/{args.epoch}")

            aiaf.train()
            aiaf.lambd = 0.0 if not warmup_complete else 0.5

            weights, p_clean, alpha_id, beta_adv = attack_schedule(epoch, args.epoch, warmup_complete)
            beta_adv = max(beta_adv, 1e-6)

            param_schedule = create_stratified_random_schedule(TRAINING_RANGES, n_batches=len(tl), n_bins=5)
            attack_plan = _make_attack_plan(len(tl), p_clean, weights, seed=epoch)

            print(f"[Epoch {epoch + 1}] λ={aiaf.lambd:.2f} p_clean={p_clean:.2f} α={alpha_id:.2f} β={beta_adv:.4f}")

            b_task = bar.add_task(f"Batch {epoch + 1}", total=len(tl))
            running_loss = {'total': 0, 'rec': 0, 'percep': 0, 'id': 0, 'adv': 0} 
            n_all = 0

            median_mask_alpha = (TRAINING_RANGES['mask']['alpha'][0] + TRAINING_RANGES['mask']['alpha'][1]) / 2
            mask_attack_fn = make_mask_overlay(args.mask, alpha=median_mask_alpha)

            for b_idx, (xb, yb) in enumerate(tl):
                xb = xb.to(device, memory_format=torch.channels_last, non_blocking=True)
                current_hps = param_schedule[b_idx]
                atk_name = attack_plan[b_idx]

                for replay_idx in range(free_mixer.m):
                    if atk_name == "pgd":
                        pgd_fn = PGDAttack(arc, eps=current_hps['pgd']['eps'], alpha=current_hps['pgd']['alpha'], iters=current_hps['pgd']['iters'])
                        xb_adv = pgd_fn(xb)
                    elif atk_name == "cw":
                        cw_fn = CWAWrapper(arc, eps=8/255, iters=current_hps['cw']['iters'], c=current_hps['cw']['c'])
                        xb_adv = cw_fn(xb)
                    elif atk_name == "patch":
                        patch_fn = AdvPatchWrapper(frac=current_hps['patch']['frac'])
                        xb_adv = patch_fn(xb)
                    elif atk_name == "mask":
                        mask_fn = MaskOverlay(args.mask, alpha=current_hps['mask']['alpha'])
                        xb_adv = mask_fn(xb)
                    else:
                        xb_adv = xb

                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        recon, pred_adv, zid, _ = aiaf(xb_adv)

                    recon = safe_img(recon.clone())
                    if recon.shape[-2:] != (112, 112):
                        recon = F.interpolate(recon, (112, 112), mode="bilinear", align_corners=False)

                    with torch.cuda.amp.autocast(enabled=False):
                        xb_fp32 = xb.float()
                        recon_fp32 = recon.float()
                        pred_adv_fp32 = pred_adv.float()

                        with torch.no_grad():
                            emb_clean = l2n(arc(to_arcface_range(xb_fp32)))
                        emb_rec = l2n(arc(to_arcface_range(recon_fp32)))

                        if not (is_finite_tensor(pred_adv_fp32) and is_finite_tensor(emb_clean) and is_finite_tensor(emb_rec)):
                            continue

                        is_adversarial = (atk_name != 'clean')
                        y_adv = torch.full((xb.size(0),), 1 if is_adversarial else 0, device=xb.device, dtype=torch.float32)

                        alpha_use = 12.0 if not warmup_complete else 20                        
                        beta_use = 0.0 if not warmup_complete else beta_adv                              
                        gamma_use = 3.0 if not warmup_complete else 2.0     
                        delta_use = 12.0 if not warmup_complete else 5.0
                        loss_total, parts = aiaf.loss(                                            
                                    xb_fp32, recon_fp32, pred_adv_fp32, y_adv,                            
                                    emb_clean, emb_rec,                                                   
                                    alpha=alpha_use,                                                     
                                    beta=beta_use,                                                        
                                    gamma=gamma_use,                   
                                    delta=delta_use                                                       
                                ) 
                    scaler.scale(loss_total / args.accum).backward()

                    global_step = b_idx * free_mixer.m + replay_idx + 1
                    do_step = (global_step % args.accum == 0) or (b_idx == len(tl) - 1 and replay_idx == free_mixer.m - 1)

                    if do_step:
                        scaler.unscale_(opt)  # Must unscale before clipping!
                        torch.nn.utils.clip_grad_norm_(aiaf.parameters(), max_norm=1.0)
                        scaler.step(opt)
                        scaler.update()
                        opt.zero_grad(set_to_none=True)

                    if replay_idx == 0:
                        n_all += xb.size(0)
                        running_loss['total'] += float(loss_total.item() * xb.size(0))
                        running_loss['rec'] += float(parts.get('rec', 0.0) * xb.size(0))
                        running_loss['percep'] += float(parts.get('percep', 0.0) * xb.size(0))
                        running_loss['id'] += float(parts.get('id', 0.0) * xb.size(0))
                        running_loss['adv'] += float(parts.get('adv', 0.0) * xb.size(0))

                bar.advance(b_task)
                step += 1

            dt = time.time() - t0
            print(f" Epoch {epoch + 1} in {dt / 3600:.2f}h | loss={running_loss['total'] / max(1, n_all):.4f}")

            tl_eval = DataLoader(tr, batch_size=args.batch, shuffle=False, num_workers=max(2, args.workers // 2), pin_memory=True, drop_last=False)
            cos_def_seen = evaluate_embedding_consistency(aiaf, arc, tl_eval, device, max_batches=12)

            if not warmup_complete:
                print(f"\n[Warmup Check - Epoch {epoch + 1}]")
                print(f"  Cosine: {cos_def_seen:.3f} / {COSINE_TARGET}")
                if epoch + 1 >= MAX_WARMUP_EPOCHS or (epoch + 1 >= MIN_WARMUP_EPOCHS and cos_def_seen >= COSINE_TARGET):
                    warmup_complete = True
                    actual_warmup_epochs = epoch + 1
                    print(f" WARMUP COMPLETE!")

            print(f" Cosine similarity: {cos_def_seen:.4f}")
            
            cd_f = eval_frozen_pairs(aiaf, arc, vl, device, FROZEN_PAIRS_DEV)
            print(f"[Open-set Clean] AUC={cd_f['auc']:.4f} EER={cd_f['eer']:.3f} TAR@1e-2={cd_f['tar@1e-2']:.3f} cos={cd_f['cos_preserve_mean']:.3f}")
            if epoch == start_epoch or (epoch + 1) % 1 == 0:
                vis_path = vis_dir / f"reconstruction_epoch_{epoch + 1:03d}.png"
                save_reconstruction_samples(aiaf, vl, device, str(vis_path), epoch + 1, n_samples=8)
            avg_total_loss = running_loss['total'] / max(1, n_all)
            avg_rec_loss = running_loss['rec'] / max(1, n_all)
            avg_id_loss = running_loss['id'] / max(1, n_all)
            avg_adv_loss = running_loss['adv'] / max(1, n_all)
            avg_percep_loss= running_loss['percep'] / max(1, n_all)
            csv_writer.writerow([
    epoch + 1,
    f"{avg_total_loss:.6f}",
    f"{avg_rec_loss:.6f}",
    f"{avg_percep_loss:.6f}",
    f"{avg_id_loss:.6f}",
    f"{avg_adv_loss:.6f}",
    f"{cd_f['auc']:.4f}",
    f"{cd_f['eer']:.4f}",
    f"{cd_f['tar@1e-2']:.4f}",
    f"{cd_f['tar@1e-3']:.4f}",
    f"{cd_f['cos_preserve_mean']:.4f}",
    "", "", "", "", "", "", "" 
])
            csv_file.flush() 
            print(f"[CSV] Wrote epoch {epoch + 1}: loss={avg_total_loss:.4f}, clean_auc={cd_f['auc']:.4f}")

            adv_metrics = None
            if warmup_complete:
                adv_metrics = eval_adversarial_openset(aiaf, arc, vl, device, dev_ids=set(FROZEN_PAIRS_DEV['ids']), mask_path=args.mask, max_batches=200)
                for name, m in adv_metrics.items():
                    print(f"  [Adv:{name}] AUC={m['auc']:.4f} EER={m['eer']:.3f} cos_preserve={m['cos_preserve']:.3f}") 
                    csv_writer.writerow([
                        epoch + 1,
                        "", "", "", "", "", 
                        "", "", "", "", "",  
                        name,
                        f"{m['auc']:.4f}",
                        f"{m['eer']:.4f}",
                        f"{m['tar@1e-2']:.4f}",
                        f"{m['tar@1e-3']:.4f}",
                        f"{m['cos_mean']:.4f}",
                        f"{m['cos_preserve']:.4f}"
                    ])
                    csv_file.flush()
                    print(f"[CSV]  Wrote adversarial metrics for {name}, epoch {epoch + 1}")

            hist["epoch"].append(epoch + 1)
            hist["clean_def_cos_seen"].append(cos_def_seen)

            if cd_f['cos_preserve_mean'] > best_cos_preserve:
                best_cos_preserve = cd_f['cos_preserve_mean']
                torch.save(aiaf.state_dict(), os.path.join(args.save_dir, "best_identity.pth"))
                print(f"[SAVE]  New best identity: {best_cos_preserve:.4f}")

            if cd_f['auc'] > best_clean_openset_auc:
                best_clean_openset_auc = cd_f['auc']
                torch.save(aiaf.state_dict(), os.path.join(args.save_dir, "best_clean_openset.pth"))
                print(f"[SAVE]  New best clean AUC: {best_clean_openset_auc:.4f}")

            if adv_metrics:
                avg_adv_auc = np.mean([m['auc'] for m in adv_metrics.values()])
                if avg_adv_auc > best_adv_openset_auc:
                    best_adv_openset_auc = avg_adv_auc
                    torch.save(aiaf.state_dict(), os.path.join(args.save_dir, "best_adv_openset.pth"))
                    print(f"[SAVE]  New best adversarial AUC: {best_adv_openset_auc:.4f}")
            checkpoint_data = {
                "model_state_dict": aiaf.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch + 1,
                "warmup_complete": warmup_complete,
                "best_cos_preserve": best_cos_preserve,
                "best_clean_openset_auc": best_clean_openset_auc,
                "best_adv_openset_auc": best_adv_openset_auc
            }
            torch.save(checkpoint_data, os.path.join(args.ckpt_dir, "last.pth"))
            print(f"[SAVE] Checkpoint saved to {args.ckpt_dir}/last.pth")
            epoch_ckpt_path = os.path.join(args.ckpt_dir, f"epoch_{epoch + 1:03d}.pth")
            torch.save(checkpoint_data, epoch_ckpt_path)
            print(f"[SAVE]  Epoch checkpoint saved to {epoch_ckpt_path}\n")

        csv_file.close()
        print(f"[CSV]  CSV file closed: {csv_path}")
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE (Option B: Fully Identity-Agnostic)")
        print(f"Best identity preservation: {best_cos_preserve:.3f}")
        print(f"Best clean AUC: {best_clean_openset_auc:.3f}")
        print(f"Best adversarial AUC: {best_adv_openset_auc:.3f}")
        print(f"\n Outputs saved to:")
        print(f"   Models: {args.save_dir}")
        print(f"   Checkpoints: {args.ckpt_dir}")
        print(f"   Visualizations: {vis_dir}")
        print(f"   Metrics CSV: {csv_path}")
        print("=" * 80)
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)

    main()
