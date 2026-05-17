import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


@dataclass
class Config:
    base_data: str = os.path.abspath("../data/")
    models_dir: str = os.path.abspath("../models/")
    reports_dir: str = os.path.abspath("../reports/")

    img_ch: int = 5
    img_size: int = 128

    morph_keys: List[str] = field(default_factory=lambda: [
        "r_sersic_index", "r_ellipticity", "r_half_light_radius"
    ])
    n_morph: int = 3

    latent_ch: int = 4
    latent_size: int = 16
    vae_base_ch: int = 48
    kl_weight: float = 5e-3
    vae_lr: float = 2e-4
    vae_epochs: int = 60
    vae_bs: int = 16
    vae_patience: int = 12

    unet_base_ch: int = 96
    unet_ch_mult: List[int] = field(default_factory=lambda: [1, 2])
    n_res_blocks: int = 2
    time_dim: int = 256

    redshift_sigma: float = 0.02
    z_embed_dim: int = 64
    morph_embed_dim: int = 64
    cond_dim: int = 128

    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    physics_predictor_epochs: int = 8
    lambda_z: float = 0.1
    lambda_morph: float = 0.10
    physics_loss_freq: int = 8

    ldm_bs: int = 16
    grad_accum: int = 2
    ldm_epochs: int = 160
    ldm_patience: int = 30
    ldm_lr: float = 5e-5
    ema_decay: float = 0.9999
    latent_scale: float = 0.4762
    infer_scale: float = 1.0

    p_uncond: float = 0.15
    guidance_scale: float = 2.0

    num_workers: int = 0
    seed: int = 42

    band_mean: List[float] = field(default_factory=lambda: [
        0.07781, 0.15729, 0.23103, 0.30393, 0.36941])
    band_std: List[float] = field(default_factory=lambda: [
        0.82795, 1.44230, 1.78690, 2.59873, 3.14001])

    # Populated by build_paths()
    train_path: str = ""
    val_path: str = ""
    test_path: str = ""
    vae_best: str = ""
    vae_latest: str = ""
    ldm_best: str = ""
    ldm_latest: str = ""
    phys_ckpt: str = ""

    def build_paths(self) -> "Config":
        self.train_path = os.path.join(self.base_data, "5x127x127_training_with_morphology.hdf5")
        self.val_path = os.path.join(self.base_data, "5x127x127_validation_with_morphology.hdf5")
        self.test_path = os.path.join(self.base_data, "5x127x127_testing_with_morphology.hdf5")
        self.vae_best = os.path.join(self.models_dir, "galaxy_vae_best.pt")
        self.vae_latest = os.path.join(self.models_dir, "galaxy_vae_latest.pt")
        self.ldm_best = os.path.join(self.models_dir, "galaxy_ldm_best.pt")
        self.ldm_latest = os.path.join(self.models_dir, "galaxy_ldm_latest.pt")
        self.phys_ckpt = os.path.join(self.models_dir, "galaxy_physics_predictor.pt")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        return self


class EMA:

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


def get_norm(ch: int, groups: int = 32) -> nn.GroupNorm:
    g = min(groups, ch)
    while g > 1 and ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: Optional[int] = None):
        super().__init__()
        self.norm1 = get_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = get_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()
        self.time_proj = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch * 2))
            if time_dim else None
        )
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        if t_emb is not None and self.time_proj is not None:
            s, b = self.time_proj(t_emb).chunk(2, dim=-1)
            h = h * (1 + s[:, :, None, None]) + b[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttn2d(nn.Module):
    """Multi-head self-attention over spatial dimensions."""

    def __init__(self, ch: int, n_heads: int = 4):
        super().__init__()
        self.norm = get_norm(ch)
        self.n_h = n_heads
        self.d_h = ch // n_heads
        self.to_q = nn.Conv1d(ch, ch, 1, bias=False)
        self.to_k = nn.Conv1d(ch, ch, 1, bias=False)
        self.to_v = nn.Conv1d(ch, ch, 1, bias=False)
        self.proj = nn.Conv1d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W)
        q = self.to_q(h).unflatten(1, (self.n_h, self.d_h))
        k = self.to_k(h).unflatten(1, (self.n_h, self.d_h))
        v = self.to_v(h).unflatten(1, (self.n_h, self.d_h))
        q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(2, 3).flatten(1, 2)
        return x + self.proj(out).reshape(B, C, H, W)


class CrossAttn2d(nn.Module):

    def __init__(self, ch: int, cond_dim: int, n_heads: int = 4):
        super().__init__()
        self.norm = get_norm(ch)
        self.n_h = n_heads
        self.d_h = ch // n_heads
        self.to_q = nn.Linear(ch, ch, bias=False)
        self.to_k = nn.Linear(cond_dim, ch, bias=False)
        self.to_v = nn.Linear(cond_dim, ch, bias=False)
        self.proj = nn.Linear(ch, ch)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).transpose(1, 2)
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)
        q = self.to_q(h).unflatten(-1, (self.n_h, self.d_h)).transpose(1, 2)
        k = self.to_k(cond).unflatten(-1, (self.n_h, self.d_h)).transpose(1, 2)
        v = self.to_v(cond).unflatten(-1, (self.n_h, self.d_h)).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).flatten(2)
        out = self.proj(out).transpose(1, 2).reshape(B, C, H, W)
        return x + out


class VAEEncoder(nn.Module):

    def __init__(self, in_ch: int = 5, base_ch: int = 64, latent_ch: int = 4):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4]

        self.input_conv = nn.Conv2d(in_ch, ch[0], 3, padding=1)

        self.down0 = nn.Sequential(
            ResBlock(ch[0], ch[0]), ResBlock(ch[0], ch[0]), nn.Conv2d(ch[0], ch[0], 3, stride=2, padding=1))
        self.down1 = nn.Sequential(
            ResBlock(ch[0], ch[1]), ResBlock(ch[1], ch[1]), nn.Conv2d(ch[1], ch[1], 3, stride=2, padding=1))
        self.down2 = nn.Sequential(
            ResBlock(ch[1], ch[2]), ResBlock(ch[2], ch[2]), nn.Conv2d(ch[2], ch[2], 3, stride=2, padding=1))

        self.mid = nn.Sequential(
            ResBlock(ch[2], ch[2]), SelfAttn2d(ch[2]), ResBlock(ch[2], ch[2]))

        self.out_norm = get_norm(ch[2])
        self.out_conv = nn.Conv2d(ch[2], latent_ch * 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_conv(x)
        h = self.down0(h)
        h = self.down1(h)
        h = self.down2(h)
        h = self.mid(h)
        h = F.silu(self.out_norm(h))
        h = self.out_conv(h)
        mu, log_var = h.chunk(2, dim=1)
        return mu, log_var


class VAEDecoder(nn.Module):

    def __init__(self, out_ch: int = 5, base_ch: int = 64, latent_ch: int = 4):
        super().__init__()
        ch = [base_ch * 4, base_ch * 2, base_ch]

        self.input_conv = nn.Conv2d(latent_ch, ch[0], 3, padding=1)
        self.mid = nn.Sequential(
            ResBlock(ch[0], ch[0]), SelfAttn2d(ch[0]), ResBlock(ch[0], ch[0]))

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch[0], ch[0], 3, padding=1), ResBlock(ch[0], ch[1]), ResBlock(ch[1], ch[1]))
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch[1], ch[1], 3, padding=1), ResBlock(ch[1], ch[2]), ResBlock(ch[2], ch[2]))
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch[2], ch[2], 3, padding=1), ResBlock(ch[2], ch[2]))

        self.out_norm = get_norm(ch[2])
        self.out_conv = nn.Conv2d(ch[2], out_ch, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.input_conv(z)
        h = self.mid(h)
        h = self.up2(h)
        h = self.up1(h)
        h = self.up0(h)
        return self.out_conv(F.silu(self.out_norm(h)))


class VAE(nn.Module):
    def __init__(self, img_ch: int = 5, base_ch: int = 64, latent_ch: int = 4):
        super().__init__()
        self.encoder = VAEEncoder(img_ch, base_ch, latent_ch)
        self.decoder = VAEDecoder(img_ch, base_ch, latent_ch)

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * log_var).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, lv = self.encoder(x)
        return self.reparameterise(mu, lv), mu, lv

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, lv = self.encode(x)
        return self.decode(z), mu, lv


def vae_loss(
    x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, kl_weight: float = 5e-3,
) -> Tuple[torch.Tensor, float, float]:
    """ELBO = reconstruction (L1) + β·KL. Returns (total, recon.item, kl.item)."""
    recon = F.l1_loss(x_hat, x)
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()
    return recon + kl_weight * kl, recon.item(), kl.item()


def _vram_str(device: torch.device) -> str:
    if device.type == "cuda":
        return f"{torch.cuda.memory_allocated() / 1e9:.1f}GB"
    if device.type == "mps":
        return "MPS"
    return ""


def train_vae_epoch(
    model: VAE, loader, opt: torch.optim.Optimizer, device: torch.device, ep: int, total_ep: int, scaler=None, use_amp: bool = False, kl_weight: float = 5e-3,
) -> Tuple[float, float, float]:
    """One training epoch. Returns (avg_loss, avg_recon, avg_kl)."""
    model.train()
    total_l = total_r = total_k = 0.0
    opt.zero_grad(set_to_none=True)
    n_batches = len(loader)

    _amp_device = "cuda" if use_amp else "cpu"
    pbar = tqdm(
        loader, desc=f"VAE train ep {ep+1:03d}/{total_ep}", leave=False, dynamic_ncols=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}", )

    for i, (imgs, _, _) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast(_amp_device, enabled=use_amp):
            x_hat, mu, lv = model(imgs)
            loss, r, k = vae_loss(imgs, x_hat, mu, lv, kl_weight)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        opt.zero_grad(set_to_none=True)

        total_l += loss.item(); total_r += r; total_k += k
        pbar.set_postfix(ordered_dict={
            "ELBO": f"{total_l/(i+1):.4f}", "recon": f"{total_r/(i+1):.4f}", "KL": f"{total_k/(i+1):.5f}", "VRAM": _vram_str(device), })

    pbar.close()
    return total_l / n_batches, total_r / n_batches, total_k / n_batches


@torch.no_grad()
def eval_vae(
    model: VAE, loader, device: torch.device, ep: int, total_ep: int, use_amp: bool = False, kl_weight: float = 5e-3,
) -> Tuple[float, float, float]:
    """Evaluation pass. Returns (avg_loss, avg_recon, avg_kl)."""
    model.eval()
    total_l = total_r = total_k = 0.0
    n_batches = len(loader)

    _amp_device = "cuda" if use_amp else "cpu"
    pbar = tqdm(
        loader, desc=f"VAE val   ep {ep+1:03d}/{total_ep}", leave=False, dynamic_ncols=True, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}", )

    for i, (imgs, _, _) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast(_amp_device, enabled=use_amp):
            x_hat, mu, lv = model(imgs)
            loss, r, k = vae_loss(imgs, x_hat, mu, lv, kl_weight)
        total_l += loss.item(); total_r += r; total_k += k
        pbar.set_postfix(ordered_dict={
            "ELBO": f"{total_l/(i+1):.4f}", "recon": f"{total_r/(i+1):.4f}", })

    pbar.close()
    return total_l / n_batches, total_r / n_batches, total_k / n_batches

class SinusoidalEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class LatentUNet(nn.Module):

    def __init__(
        self, latent_ch: int = 4, base_ch: int = 128, ch_mult: Tuple[int, ...] = (1, 2), n_res: int = 2, time_dim: int = 256, cond_dim: int = 128, ):
        super().__init__()
        ch = [base_ch * m for m in ch_mult]

        self.time_embed = nn.Sequential(
            SinusoidalEmbed(base_ch), nn.Linear(base_ch, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim), )

        self.input_conv = nn.Conv2d(latent_ch, ch[0], 3, padding=1)

        self.d0_res = nn.ModuleList([ResBlock(ch[0], ch[0], time_dim) for _ in range(n_res)])
        self.d0_cross = nn.ModuleList([CrossAttn2d(ch[0], cond_dim) for _ in range(n_res)])
        self.d0_down = nn.Conv2d(ch[0], ch[0], 3, stride=2, padding=1)

        self.d1_res = nn.ModuleList(
            [ResBlock(ch[0] if i == 0 else ch[1], ch[1], time_dim) for i in range(n_res)])
        self.d1_cross = nn.ModuleList([CrossAttn2d(ch[1], cond_dim) for _ in range(n_res)])
        self.d1_down = nn.Conv2d(ch[1], ch[1], 3, stride=2, padding=1)

        self.mid_res1 = ResBlock(ch[1], ch[1], time_dim)
        self.mid_self = SelfAttn2d(ch[1])
        self.mid_cross = CrossAttn2d(ch[1], cond_dim)
        self.mid_res2 = ResBlock(ch[1], ch[1], time_dim)

        self.u1_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch[1], ch[1], 3, padding=1))
        self.u1_res = nn.ModuleList(
            [ResBlock(ch[1] * 2 if i == 0 else ch[1], ch[1], time_dim) for i in range(n_res)])
        self.u1_cross = nn.ModuleList([CrossAttn2d(ch[1], cond_dim) for _ in range(n_res)])

        self.u0_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch[1], ch[0], 3, padding=1))
        self.u0_res = nn.ModuleList(
            [ResBlock(ch[0] * 2 if i == 0 else ch[0], ch[0], time_dim) for i in range(n_res)])
        self.u0_cross = nn.ModuleList([CrossAttn2d(ch[0], cond_dim) for _ in range(n_res)])

        self.out_norm = get_norm(ch[0])
        self.out_conv = nn.Conv2d(ch[0], latent_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, ) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.input_conv(x)

        for res, cross in zip(self.d0_res, self.d0_cross):
            h = res(h, t_emb); h = cross(h, cond)
        skip0 = h
        h = self.d0_down(h)

        for res, cross in zip(self.d1_res, self.d1_cross):
            h = res(h, t_emb); h = cross(h, cond)
        skip1 = h
        h = self.d1_down(h)

        h = self.mid_res1(h, t_emb)
        h = self.mid_self(h)
        h = self.mid_cross(h, cond)
        h = self.mid_res2(h, t_emb)

        h = self.u1_up(h)
        h = torch.cat([h, skip1], dim=1)
        for res, cross in zip(self.u1_res, self.u1_cross):
            h = res(h, t_emb); h = cross(h, cond)

        h = self.u0_up(h)
        h = torch.cat([h, skip0], dim=1)
        for res, cross in zip(self.u0_res, self.u0_cross):
            h = res(h, t_emb); h = cross(h, cond)

        return self.out_conv(F.silu(self.out_norm(h)))


class DDPMScheduler:

    def __init__(self, T: int = 1000, b0: float = 1e-4, b1: float = 0.02):
        self.T = T
        betas = torch.linspace(b0, b1, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_p = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.alpha_bar_p = alpha_bar_p
        self.sqrt_ab = alpha_bar.sqrt()
        self.sqrt_1mab = (1 - alpha_bar).sqrt()
        self.post_var = betas * (1 - alpha_bar_p) / (1 - alpha_bar)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None, ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self.sqrt_ab.to(t.device)[t].view(-1, 1, 1, 1)
        s1a = self.sqrt_1mab.to(t.device)[t].view(-1, 1, 1, 1)
        return sa * x0 + s1a * noise, noise

    def predict_x0(
        self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor, ) -> torch.Tensor:
        sa = self.sqrt_ab.to(t.device)[t].view(-1, 1, 1, 1)
        s1a = self.sqrt_1mab.to(t.device)[t].view(-1, 1, 1, 1)
        return (x_t - s1a * eps_pred) / sa.clamp(min=1e-8)

    @torch.no_grad()
    def ddpm_sample(
        self, model: nn.Module, n: int, cond: torch.Tensor, device: torch.device, latent_ch: int = 4, latent_size: int = 16, use_amp: bool = False, ) -> torch.Tensor:
        """Full DDPM ancestral sampling. Slow — use ddim_sample for inference."""
        shape = (n, latent_ch, latent_size, latent_size)
        x = torch.randn(shape, device=device)
        model.eval()
        _amp_device = "cuda" if device.type == "cuda" else "cpu"
        for t_idx in tqdm(reversed(range(self.T)), total=self.T, desc="DDPM sample", leave=False):
            t_b = torch.full((n,), t_idx, device=device, dtype=torch.long)
            with torch.amp.autocast(_amp_device, enabled=use_amp):
                eps = model(x, t_b, cond)
            beta_t = self.betas[t_idx].to(device)
            alpha_t = self.alphas[t_idx].to(device)
            ab_t = self.alpha_bar[t_idx].to(device)
            mu = (1 / alpha_t.sqrt()) * (x - (beta_t / (1 - ab_t).sqrt()) * eps)
            if t_idx > 0:
                sigma = self.post_var[t_idx].to(device).clamp(min=1e-20).sqrt()
                x = mu + sigma * torch.randn_like(x)
            else:
                x = mu
        return x

    @torch.no_grad()
    def ddim_sample(
        self, model: nn.Module, n: int, cond: torch.Tensor, device: torch.device, steps: int = 50, eta: float = 0.0, latent_ch: int = 4, latent_size: int = 16, use_amp: bool = False, ) -> torch.Tensor:
        shape = (n, latent_ch, latent_size, latent_size)
        x = torch.randn(shape, device=device)
        model.eval()
        _amp_device = "cuda" if device.type == "cuda" else "cpu"
        ts = torch.linspace(self.T - 1, 0, steps + 1).long()
        for i in range(steps):
            t_cur = ts[i].item()
            t_prev = ts[i + 1].item()
            t_b = torch.full((n,), t_cur, device=device, dtype=torch.long)
            with torch.amp.autocast(_amp_device, enabled=use_amp):
                eps = model(x, t_b, cond)
            ab_cur = self.alpha_bar[t_cur].to(device)
            ab_prev = (self.alpha_bar[t_prev].to(device)
                       if t_prev >= 0 else torch.ones(1, device=device))
            x0_pred = ((x - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()).clamp(-4, 4)
            sigma = eta * ((1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)).sqrt()
            x = (ab_prev.sqrt() * x0_pred
                 + (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
                 + sigma * torch.randn_like(x))
        return x

class FourierEmbed(nn.Module):
    def __init__(self, dim: int = 64, max_period: float = 10.0):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(
            torch.arange(half, dtype=torch.float32) * -math.log(max_period) / (half - 1)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        freqs = self.freqs.to(x.device)
        args = x * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ConditioningModule(nn.Module):

    def __init__(
        self, n_morph: int = 3, z_dim: int = 64, morph_dim: int = 64, out_dim: int = 128, sigma: float = 0.02, ):
        super().__init__()
        self.sigma = sigma

        self.z_embed = nn.Sequential(
            FourierEmbed(z_dim), nn.Linear(z_dim, z_dim), nn.SiLU(), nn.Linear(z_dim, z_dim), )
        self.morph_embed = nn.Sequential(
            nn.Linear(n_morph, morph_dim), nn.SiLU(), nn.Linear(morph_dim, morph_dim), nn.LayerNorm(morph_dim), )
        self.fusion = nn.Sequential(
            nn.Linear(z_dim + morph_dim, out_dim * 2), nn.SiLU(), nn.Linear(out_dim * 2, out_dim), nn.LayerNorm(out_dim), )

    def forward(
        self, z: torch.Tensor, morph: torch.Tensor, training: bool = True, ) -> torch.Tensor:
        if training and self.sigma > 0:
            z = z + torch.randn_like(z) * self.sigma
        z_feat = self.z_embed(z)
        morph_feat = self.morph_embed(morph)
        return self.fusion(torch.cat([z_feat, morph_feat], dim=-1))

class PhysicsPredictor(nn.Module):

    def __init__(self, img_ch: int = 5, n_morph: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(img_ch, 32, 3, stride=2, padding=1), nn.GELU(), nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(), nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(), nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GELU(), nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), )
        self.z_head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
        self.morph_head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, n_morph))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        z_pred = self.z_head(feat).squeeze(-1)
        morph_pred = self.morph_head(feat)
        return z_pred, morph_pred


class CFGWrapper(nn.Module):
    def __init__(
        self, model: nn.Module, null_cond: torch.Tensor, guidance_scale: float = 3.5, device: Optional[torch.device] = None, use_amp: bool = False, ):
        super().__init__()
        self.model = model
        self.null_cond = null_cond
        self.guidance_scale = guidance_scale
        self._device = device
        self.use_amp = use_amp

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        null = self.null_cond.expand(B, -1)
        dev = self._device or x.device
        _amp_device = dev.type if dev.type != "mps" else "cpu"
        with torch.amp.autocast(_amp_device, enabled=self.use_amp):
            eps_cond = self.model(x, t, cond)
            eps_uncond = self.model(x, t, null)
        return eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)


@torch.no_grad()
def generate(
    z_vals: torch.Tensor,
    morph_vals: torch.Tensor,
    cond_module: "ConditioningModule",
    unet: "LatentUNet",
    vae: "VAE",
    noise_sched: "DDPMScheduler",
    cfg: "Config",
    device: torch.device,
    method: str = "ddim",
    ddim_steps: int = 50,
    eta: float = 0.0,
    guidance_scale: Optional[float] = None,
    use_amp: bool = False,
) -> torch.Tensor:
    if guidance_scale is None:
        guidance_scale = cfg.guidance_scale

    z_vals = z_vals.to(device)
    morph_vals = morph_vals.to(device)
    cond = cond_module(z_vals, morph_vals, training=False)

    if guidance_scale > 1.0:
        null_cond = torch.zeros(1, cfg.cond_dim, device=device)
        active_unet = CFGWrapper(unet, null_cond, guidance_scale, device=device, use_amp=use_amp)
    else:
        active_unet = unet

    sample_kwargs = dict(
        latent_ch=cfg.latent_ch, latent_size=cfg.latent_size, use_amp=use_amp,
    )

    if method == "ddim":
        latent = noise_sched.ddim_sample(
            active_unet, len(z_vals), cond, device, steps=ddim_steps, eta=eta, **sample_kwargs)
    else:
        latent = noise_sched.ddpm_sample(
            active_unet, len(z_vals), cond, device, **sample_kwargs)

    imgs = vae.decode(latent * cfg.infer_scale)
    return imgs.cpu()
