import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import lpips
# proposed AIAF 
class StableEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        resnet.maxpool = nn.Identity()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        # Simple forward pass
        features = self.backbone(x) 
        return features.flatten(1)
class StableDecoder(nn.Module):
    """Decoder that maintains information flow"""
    def __init__(self, z_dim=512):
        super().__init__()
        self.proj = nn.Linear(z_dim, 256 * 4 * 4) 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), 
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 7, 2, 3, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 3, 3, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        x = self.proj(z)
        x = x.view(-1, 256, 4, 4)
        x = self.decoder(x)
        x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        return x
    
#Tiny AIAF for computational constraint (Only ~2.5M parameters)

class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.net(x).flatten(1)


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(512, 128 * 7 * 7)
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), 
            nn.BatchNorm2d(16), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        x = self.proj(z).view(-1, 128, 7, 7)
        return self.net(x)


class AIAF_Tiny(nn.Module):

    def __init__(self, lambd=0.3, num_domains=1):
        super().__init__()
        
        self.encoder = TinyEncoder()
        self.enc = self.encoder 
        
        # Identity branch
        self.fc_id = nn.Sequential(
            nn.Linear(12544, 512),
            nn.BatchNorm1d(512)
        )
        
        # Attack branch
        self.fc_adv = nn.Sequential(
            nn.Linear(12544, 256), 
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.cls = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_domains)
        )
        
        self.decoder = TinyDecoder()
        self.dec = self.decoder
        self.lambd = lambd
        self.current_epoch = 0
        
        self._init_weights()
        self.lpips_fn = lpips.LPIPS(net='alex', verbose=False)
        self.lpips_fn.eval()
        for param in self.lpips_fn.parameters():
            param.requires_grad = False
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.weight.data *= 0.5
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        h = self.encoder(x) 
        z_id = self.fc_id(h)
        z_id = F.leaky_relu(z_id, 0.1)
        z_adv = self.fc_adv(h)
        if self.training and self.lambd > 0:
            z_adv_rev = grad_reverse(z_adv, self.lambd)
            pred_adv = self.cls(z_adv_rev)
        else:
            pred_adv = self.cls(z_adv)
        recon = self.decoder(z_id)
        zid = z_id.view(-1, 512, 1, 1).expand(-1, -1, 7, 7)[:, :256, :, :]
        zadv = z_adv.view(-1, 256, 1, 1).expand(-1, -1, 7, 7).mean(dim=(2,3))
        
        return recon, pred_adv, zid, zadv

    def loss(self, x, recon, pred_adv, y_adv, emb_clean, emb_recon,
             alpha=1.5, beta=0.3, gamma=2.0, delta=5.0):
        l_rec = gamma * F.mse_loss(recon, x)
        recon_norm = recon * 2 - 1 
        x_norm = x * 2 - 1
        l_percep = self.lpips_fn(recon_norm, x_norm).mean()
        cos_sim = F.cosine_similarity(emb_clean, emb_recon, dim=1)
        l_id = (1 - cos_sim).mean()
        l_adv = F.binary_cross_entropy_with_logits(
            pred_adv.squeeze(-1), y_adv.float()
        )
        total_loss = l_rec + delta * l_percep + alpha * l_id + beta * l_adv
        
        return total_loss, {
            'rec': (l_rec / gamma).item(),
            'percep': l_percep.item(),
            'id': l_id.item(),
            'adv': l_adv.item()
        }
def grad_reverse(x, lambd=1.0):
    """Gradient reversal for adversarial training"""
    return GradReverseFunction.apply(x, lambd)


class GradReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None
class AIAF_Stable(nn.Module):
    def __init__(self, lambd=0.1, num_domains=1):
        super().__init__()
        self.encoder = StableEncoder()
        self.enc = self.encoder
        self.fc_id = nn.Linear(512, 512)
        self.id_norm = nn.BatchNorm1d(512)
        self.fc_adv = nn.Linear(512, 256)
        self.cls = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_domains)
        )
        self.decoder = StableDecoder(z_dim=512)
        self.dec = self.decoder
        self.lambd = lambd
        self.current_epoch = 0
        self._init_weights()
        import lpips
        self.lpips_fn = lpips.LPIPS(net='alex', verbose=False)
        self.lpips_fn.eval()
        for param in self.lpips_fn.parameters():
            param.requires_grad = False
    def _init_weights(self):
        nn.init.eye_(self.fc_id.weight)
        nn.init.zeros_(self.fc_id.bias)
        nn.init.xavier_uniform_(self.fc_adv.weight, gain=0.5)
        nn.init.zeros_(self.fc_adv.bias)
        for m in self.decoder.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Encode
        h = self.encoder(x)
        z_id = self.fc_id(h) 
        z_id = self.id_norm(z_id) 
        z_id = F.leaky_relu(z_id, 0.1)
        z_adv = self.fc_adv(h)
        if self.training and self.lambd > 0:
            z_adv_rev = grad_reverse(z_adv, self.lambd)
            pred_adv = self.cls(z_adv_rev)
        else:
            pred_adv = self.cls(z_adv)
        recon = self.decoder(z_id)
        zid = z_id.view(x.size(0), 512, 1, 1).expand(-1, -1, 7, 7)[:, :256, :, :]
        zadv = z_adv.view(x.size(0), 256, 1, 1).expand(-1, -1, 7, 7).mean(dim=(2,3))
        
        return recon, pred_adv, zid, zadv
    
    def loss(self, x, recon, pred_adv, y_adv, emb_clean, emb_recon,
         alpha=15.0, beta=0.3, gamma=2.0, delta=5.0):
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # 1. Reconstruction loss
        l_rec = F.mse_loss(recon, x)
        
        # 2. Perceptual loss (LPIPS)
        if not hasattr(self, 'lpips_fn'):
            import lpips
            self.lpips_fn = lpips.LPIPS(net='alex', verbose=False)
            self.lpips_fn.eval()
            for param in self.lpips_fn.parameters():
                param.requires_grad = False
        recon_norm = recon * 2 - 1
        x_norm = x * 2 - 1
        l_percep = self.lpips_fn(recon_norm, x_norm).mean()
        
        # 3. Identity preservation 
        cos_sim = F.cosine_similarity(emb_clean, emb_recon, dim=1)
        l_id = (1 - cos_sim).mean()
        
        # 4. Adversarial loss
        l_adv = F.binary_cross_entropy_with_logits(
            pred_adv.squeeze(-1), y_adv.float()
        )
        
        # Total loss
        total_loss = gamma * l_rec + delta * l_percep + alpha * l_id + beta * l_adv
        
        return total_loss, {
            'rec': (l_rec).item(),
            'percep': l_percep.item(),
            'id': l_id.item(),
            'adv': l_adv.item()
        }


def grad_reverse(x, lambd=1.0):
    """Gradient reversal for adversarial training"""
    return GradReverseFunction.apply(x, lambd)


class GradReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None