# attack_wrappers.py
import torch, torch.nn.functional as F
from torch import nn
import io
import torchattacks, autoattack
from torchattacks import CW
import contextlib
import torchvision.io as tvio
import numpy as np
def _to_torch(arr, dtype=torch.float32):
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
    if arr.dtype == torch.uint8:
        arr = arr.float().div_(255)
    return arr.to(device=arr.device if arr.is_cuda else 'cpu', dtype=dtype)

# ---------- PGD --------------------------------------------------------------
class PGDAttack(nn.Module):
    def __init__(self, model, eps=8/255, alpha=2/255, iters=20, p_mix=0.5):
        super().__init__()
        self.atk = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=iters)
        self.p_mix = p_mix              

    def forward(self, x, y):
        mask = torch.rand(x.size(0), device=x.device) < self.p_mix
        if mask.any():
            x_adv = self.atk(x[mask], y[mask])
            x = x.clone()
            x[mask] = x_adv
        return x

# ---------- C&W ----------------------------------------------------------------
class CWAWrapper(nn.Module):
    def __init__(self, model, conf=2, iters=500, c=5e-2, p_mix=0.5):
        super().__init__()
        self.atk = CW(model, c=c, kappa=conf, steps=iters)
        self.p_mix = p_mix

    def forward(self, x, y):
        mask = torch.rand_like(y.float()) < self.p_mix
        if mask.any():
            x_adv = self.atk(x[mask], y[mask])
            x = x.clone()
            x[mask] = x_adv
        return x

# ---------- AutoAttack-FR ------------------------------------------------------
class AutoAttackWrapper(nn.Module):
    def __init__(self, model, eps=8/255, version="rand", p_mix=0.5):
        super().__init__()
        self.atk = autoattack.AutoAttack(model, norm="Linf",
                                         eps=eps, version=version,
                                         is_tf_model=False)
        self.p_mix = p_mix

    def forward(self, x, y):
        if self.training and self.p_mix < 1.0:
            keep = torch.rand(len(y), device=x.device) >= self.p_mix
            x_mix, y_mix = x[~keep], y[~keep]
            if len(y_mix) >= 2:
                x_adv = self.atk.run_standard_evaluation(x_mix, y_mix, bs=len(y_mix))
                x = x.clone()
                x[~keep] = _to_torch(x_adv, dtype=x.dtype).to(x.device)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                x_adv = self.atk.run_standard_evaluation(x, y, bs=len(y))
            x = _to_torch(x_adv, dtype=x.dtype).to(x.device)
        return x

# ---------- Adversarial Patch --------------------------------------------------
class AdvPatchWrapper(nn.Module):
    
    def __init__(self, model, p_mix=0.5, patch_frac=0.3, iters=60, step=0.75):
        super().__init__()
        self.model = model
        self.p_mix = float(p_mix)
        self.patch_frac = float(patch_frac)
        self.iters = int(iters)
        self.step = float(step)

    def _make_patch(self, H, W, device, dtype):
        side = max(4, int(self.patch_frac * min(H, W)))
        patch = torch.rand(1, 3, side, side, device=device, dtype=dtype, requires_grad=True)
        return patch

    def _paste(self, x, patch, top, left):
        B, C, H, W = x.shape
        _, _, ph, pw = patch.shape if patch.dim() == 4 else (1, 3, patch.shape[-2], patch.shape[-1])

        if patch.dim() == 3:
            patch = patch.unsqueeze(0)
        if patch.size(0) == 1 and B > 1:
            patch = patch.expand(B, -1, -1, -1)

        out = x.clone()
        for i in range(B):
            t = int(top[i])
            l = int(left[i])
            out[i, :, t:t+ph, l:l+pw] = patch[i].to(out.dtype)
        return out


    def _attack_subset(self, x, y):
        B, C, H, W = x.shape
        dev, dt = x.device, x.dtype

        ph = max(4, int(self.patch_frac * min(H, W)))
        if ph >= min(H, W):
            return x

        top  = torch.randint(0, H - ph + 1, (B,), device=dev)
        left = torch.randint(0, W - ph + 1, (B,), device=dev)

        patch = self._make_patch(H, W, dev, dt) 
        if patch.dim() == 3:
            patch = patch.unsqueeze(0)
        patch = patch.to(dev).requires_grad_(True)

        opt = torch.optim.SGD([patch], lr=self.step)
        for _ in range(self.iters):
            opt.zero_grad(set_to_none=True)
            x_adv  = self._paste(x, patch.clamp(0, 1), top, left)
            logits = self.model(x_adv)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            loss = -F.cross_entropy(logits, y, reduction="mean")
            loss.backward()
            opt.step()
            patch.data.clamp_(0, 1)

        return self._paste(x, patch, top, left).clamp(0, 1)

    def forward(self, x, y):
        if self.p_mix < 1.0 and self.training:
            keep = torch.rand(x.size(0), device=x.device) >= self.p_mix
            if (~keep).any():
                x_sub = self._attack_subset(x[~keep], y[~keep])
                x = x.clone()
                x[~keep] = x_sub
            return x
        return self._attack_subset(x, y)
# ---------- Sibling Attack --------------------------------------------------
class SiblingAttack(nn.Module):
    def __init__(self, model, surrogate=None, eps=8/255, alpha=2/255, iters=30, p_mix=0.5):
        super().__init__()
        self.surrogate = surrogate if surrogate is not None else model
        self.target = model  
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        self.p_mix = p_mix
        
    def forward(self, x, y):
        """Generate adversarial examples on surrogate model"""
        mask = torch.rand(x.size(0), device=x.device) < self.p_mix
        if not mask.any():
            return x
        
        x_work = x[mask].clone().detach()
        delta = torch.zeros_like(x_work).uniform_(-self.eps, self.eps)
        delta.requires_grad = True
        for _ in range(self.iters):
            x_adv = (x_work + delta).clamp(0, 1)
            logits = self.surrogate(x_adv)
            
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            loss = F.cross_entropy(logits, y[mask])
            grad = torch.autograd.grad(loss, delta, create_graph=False)[0]
            delta = delta.detach() + self.alpha * grad.sign()
            delta = torch.clamp(delta, -self.eps, self.eps)
            delta.requires_grad = True
        x_adv = (x_work + delta.detach()).clamp(0, 1)
        x_out = x.clone()
        x_out[mask] = x_adv
        return x_out
# ---------- Hybrid Fast Attack Mixer ---------------------------------------
class FastAttackMixer:
    """
    Manages Free-AT style replays for multiple attack types.
    Reduces training time by 3-4× while maintaining robustness.
    """
    def __init__(self, m=4):
        self.m = m 
        self.deltas = {} 
        self.current_replay = 0
        
    def get_delta(self, name, x, eps):
        if name not in self.deltas or self.deltas[name].size(0) != x.size(0):
            self.deltas[name] = torch.zeros_like(x).uniform_(-eps, eps)
        return self.deltas[name]
    
    def update_delta(self, name, grad, alpha, eps):
        if name in self.deltas and grad is not None:
            self.deltas[name] = (
                self.deltas[name] + alpha * grad.sign()
            ).clamp(-eps, eps).detach()


# ---------- Print mask overlay -------------------------------------
class MaskOverlay(nn.Module):
    def __init__(self, png_path, alpha=0.5, p_mix=0.5):
        super().__init__()
        self.alpha, self.p_mix = alpha, p_mix
        ov = tvio.read_image(png_path)         # (C,H,W), 0-1
        self.register_buffer("overlay", ov, persistent=True)

    @torch.no_grad()
    def forward(self, x, y=None):
        B, C, H, W = x.shape
        dev = x.device
        ov = F.interpolate(self.overlay.unsqueeze(0), size=(H,W), mode="bilinear", align_corners=False)[0].to(dev, dtype=x.dtype)
        # light random brightness/contrast on GPU
        g = 1 + 0.2*(torch.rand(B,1,1,1, device=dev)-0.5)
        b = 0.1*(torch.rand(B,1,1,1, device=dev)-0.5)
        ovB = (ov.unsqueeze(0)*g +b).clamp(0,1)
        mix = torch.rand(B, device=dev) < self.p_mix
        if mix.any():
            x = x.clone()
            x[mix] = (self.alpha*ovB[mix]  +(1-self.alpha)*x[mix]).clamp(0,1)
        return x
