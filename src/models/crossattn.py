import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, lora_alpha=16, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scale = lora_alpha / r
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight) 

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scale

class CrossAttentionFusion(nn.Module):
    def __init__(self, img_dim=256, morph_dim=128, num_heads=4,
                 r=4, lora_alpha=16, dropout=0.1):
        super().__init__()
        assert img_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = img_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = LoRALinear(img_dim,   img_dim, r, lora_alpha, bias=False)
        self.k_proj = LoRALinear(morph_dim, img_dim, r, lora_alpha, bias=False)
        self.v_proj = LoRALinear(morph_dim, img_dim, r, lora_alpha, bias=False)
        self.out_proj = nn.Linear(img_dim, img_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(img_dim)

    def forward(self, z_img, z_morph):
        B = z_img.size(0)
        Q = self.q_proj(z_img).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(z_morph).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(z_morph).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.attn_drop(attn)
        out  = (attn @ V).transpose(1, 2).contiguous().view(B, -1)
        out  = self.proj_drop(self.out_proj(out))

        return self.norm(z_img + out)  

class MorphMLP(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.1), 
            nn.Linear(256, out_dim), nn.BatchNorm1d(out_dim), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)

class FusionModelCrossAttn(nn.Module):
    def __init__(self, n_morph, latent_dim=256, morph_dim=128,
                 num_heads=4, lora_r=4, lora_alpha=16):
        super().__init__()

        backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        backbone.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.img_encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.img_fc = LoRALinear(2048, latent_dim, lora_r, lora_alpha)

        self.morph_mlp = MorphMLP(n_morph, out_dim=morph_dim)

        self.cross_attn = CrossAttentionFusion(
            img_dim=latent_dim, morph_dim=morph_dim,
            num_heads=num_heads, r=lora_r, lora_alpha=lora_alpha)

        self.head = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, img, morph):
        z_img = self.img_fc(self.img_encoder(img).flatten(1))  
        z_morph = self.morph_mlp(morph)                         
        z_fused = self.cross_attn(z_img, z_morph)               
        return self.head(z_fused).squeeze(-1)

class RedshiftLoss(nn.Module):
    def __init__(self, alpha=0.8, z_bins=None, bin_weights=None):
        super().__init__()
        self.alpha = alpha
        self.smooth = nn.SmoothL1Loss(beta=0.1, reduction='none')
        self.z_bins = z_bins          
        self.bin_weights = bin_weights 

    def forward(self, pred, target):
        base = self.smooth(pred, target) 
        if self.z_bins is not None:
            idx = torch.bucketize(target, self.z_bins)
            w = self.bin_weights[idx.clamp(0, len(self.bin_weights)-1)]
            base = (base * w).mean()
        else:
            base = base.mean()
        dz = (pred - target).abs() / (1.0 + target.abs())
        mask = dz > 0.12
        outlier = dz[mask].mean() if mask.any() else torch.tensor(0.0, device=pred.device)
        return self.alpha * base + (1.0 - self.alpha) * outlier
    
def set_phase(model):
    for name, p in model.img_encoder.named_parameters():
        p.requires_grad = "6" in name 

    model.img_fc.linear.weight.requires_grad = False
    if model.img_fc.linear.bias is not None:
        model.img_fc.linear.bias.requires_grad = False
    for p in model.img_fc.lora_A.parameters(): p.requires_grad = True
    for p in model.img_fc.lora_B.parameters(): p.requires_grad = True

    for proj in [model.cross_attn.q_proj,
                 model.cross_attn.k_proj,
                 model.cross_attn.v_proj]:
        proj.linear.weight.requires_grad = False
        for p in proj.lora_A.parameters(): p.requires_grad = True
        for p in proj.lora_B.parameters(): p.requires_grad = True
    for p in model.cross_attn.out_proj.parameters(): p.requires_grad = True
    for p in model.cross_attn.norm.parameters(): p.requires_grad = True

    for p in model.morph_mlp.parameters(): p.requires_grad = True
    for p in model.head.parameters(): p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

def make_optimizer(model):
    backbone_params = [p for n, p in model.img_encoder.named_parameters() if p.requires_grad]
    return optim.AdamW([
        {"params": backbone_params, "lr": 1e-5},
        {"params": list(model.img_fc.lora_A.parameters()) +
                   list(model.img_fc.lora_B.parameters()), "lr": 1e-4},
        {"params": list(model.cross_attn.q_proj.lora_A.parameters()) +
                   list(model.cross_attn.q_proj.lora_B.parameters()) +
                   list(model.cross_attn.k_proj.lora_A.parameters()) +
                   list(model.cross_attn.k_proj.lora_B.parameters()) +
                   list(model.cross_attn.v_proj.lora_A.parameters()) +
                   list(model.cross_attn.v_proj.lora_B.parameters()), "lr": 2e-4},
        {"params": model.cross_attn.out_proj.parameters(), "lr": 2e-4},
        {"params": model.cross_attn.norm.parameters(), "lr": 2e-4},
        {"params": model.morph_mlp.parameters(),"lr": 2e-4},
        {"params": model.head.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4, betas=(0.9, 0.999))

def make_scheduler(optimizer, total_epochs):
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=3)
    cosine = CosineAnnealingLR(optimizer, T_max=max(total_epochs - 3, 1), eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[3])