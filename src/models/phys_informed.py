import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


BAND_WAVELENGTHS = [4800., 6200., 7700., 8900., 9800.]

class BandEmbedding(nn.Module):
    def __init__(self, embed_dim: int, wavelengths: list = BAND_WAVELENGTHS):
        super().__init__()
        self.embed = nn.Embedding(len(wavelengths), embed_dim)
        wl = torch.tensor(wavelengths, dtype=torch.float32)
        wl_n = (wl - wl.min()) / (wl.max() - wl.min())
        with torch.no_grad():
            pos = wl_n.unsqueeze(1)
            dim = torch.arange(embed_dim, dtype=torch.float32)
            freq = 1.0 / (10000 ** (2 * (dim // 2) / embed_dim))
            enc = pos * freq.unsqueeze(0)
            enc[:, 0::2] = enc[:, 0::2].sin()
            enc[:, 1::2] = enc[:, 1::2].cos()
            self.embed.weight.copy_(enc)

    def forward(self, band_idx):
        return self.embed(band_idx)

class BandAwareImageBranch(nn.Module):
    def __init__(self, embed_dim, band_embed, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.band_embed = band_embed
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(5, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        backbone.features[0][0] = new_conv
        self.global_features = backbone.features
        self.global_pool = backbone.avgpool
        backbone_out = backbone.classifier[1].in_features
        if freeze_backbone:
            for p in self.global_features.parameters():
                p.requires_grad = False
        self.global_proj = nn.Sequential(
            nn.Linear(backbone_out, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.band_cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.band_proj = nn.Sequential(
            nn.Linear(128, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())

    def forward(self, x):
        g_feat = self.global_pool(self.global_features(x)).flatten(1)
        g_token = self.global_proj(g_feat).unsqueeze(1)            
        wl_emb = self.band_embed(torch.arange(5, device=x.device))
        band_tokens = []
        for b in range(5):
            tok = self.band_proj(self.band_cnn(x[:, b:b+1])) + wl_emb[b]
            band_tokens.append(tok)
        return torch.cat([g_token, torch.stack(band_tokens, 1)], 1)  

class BandAwareMorphologyBranch(nn.Module):
    def __init__(self, num_features, feature_band_idx, band_embed,
                 token_dim=128, num_heads=4, num_layers=3,
                 dropout=0.1, embed_dim=256):
        super().__init__()
        self.num_features = num_features
        self.register_buffer('feature_band_idx', feature_band_idx)
        self.band_embed = band_embed
        self.value_proj = nn.Linear(1, token_dim)
        self.pos_embed = nn.Embedding(num_features, token_dim)
        self.band_adapter = nn.Linear(embed_dim, token_dim, bias=False)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=num_heads,
            dim_feedforward=token_dim*2, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Sequential(
            nn.Linear(token_dim, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())

    def forward(self, x):
        dev = x.device
        pos_idx = torch.arange(self.num_features, device=dev)
        wl_emb = self.band_adapter(self.band_embed(self.feature_band_idx.to(dev)))
        tokens = (self.value_proj(x.unsqueeze(-1))
                   + self.pos_embed(pos_idx).unsqueeze(0)
                   + wl_emb.unsqueeze(0))
        tokens = self.encoder(tokens)
        wl_out = self.band_embed(torch.arange(5, device=dev))
        return torch.stack([
            self.out_proj(tokens[:, self.feature_band_idx==b, :].mean(1)) + wl_out[b]
            for b in range(5)
        ], dim=1)                                                   

class GaussianHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
        )
        self.mu_head  = nn.Linear(hidden_dim // 2, 1)
        self.log_sig_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        h = self.trunk(x)
        mu = self.mu_head(h).squeeze(-1)
        sigma = nn.functional.softplus(self.log_sig_head(h)).squeeze(-1) + 1e-4
        return mu, sigma

class BandAlignedFusion(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2,
                 dropout=0.1, hidden_dim=256):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim*2, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.fusion_transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = GaussianHead(embed_dim, hidden_dim, dropout)

    def forward(self, img_tokens, morph_tokens):
        tokens = torch.cat([img_tokens, morph_tokens], dim=1)   
        tokens = self.fusion_transformer(tokens)
        return self.head(tokens.mean(dim=1))                    

class PhysicallyInformedLateFusionNet(nn.Module):
    def __init__(self, num_tabular, feature_band_idx, embed_dim=256,
                 morph_token_dim=128, morph_heads=4, morph_layers=3,
                 fusion_heads=4, fusion_layers=2, dropout=0.1,
                 img_pretrained=True, img_freeze=False):
        super().__init__()
        self.band_embed = BandEmbedding(embed_dim)
        self.image_branch = BandAwareImageBranch(
            embed_dim, self.band_embed, img_pretrained, img_freeze)
        self.morph_branch = BandAwareMorphologyBranch(
            num_tabular, feature_band_idx, self.band_embed,
            morph_token_dim, morph_heads, morph_layers, dropout, embed_dim)
        self.fusion = BandAlignedFusion(
            embed_dim, fusion_heads, fusion_layers, dropout, embed_dim)
        self.aux_img_head = nn.Linear(embed_dim, 1)
        self.aux_morph_head = nn.Linear(embed_dim, 1)

    def forward(self, image, tabular):
        img_tok = self.image_branch(image)     
        morph_tok = self.morph_branch(tabular)   
        mu, sigma = self.fusion(img_tok, morph_tok)
        z_img_aux = self.aux_img_head(img_tok[:, 0]).squeeze(-1)
        z_morph_aux = self.aux_morph_head(morph_tok.mean(1)).squeeze(-1)
        return mu, sigma, z_img_aux, z_morph_aux

    @torch.no_grad()
    def predict(self, image, tabular):
        self.eval()
        mu, sigma, _, _ = self.forward(image, tabular)
        return mu, sigma


class ProbabilisticRedshiftLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.5, huber_delta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.nll = nn.GaussianNLLLoss(eps=1e-6, reduction='mean')
        self.huber = nn.SmoothL1Loss(beta=huber_delta)

    def forward(self, mu, sigma, z_img_aux, z_morph_aux, z_true, beta_scale=1.0):
        l_nll = self.nll(mu, z_true, sigma ** 2)
        l_huber = self.huber(mu, z_true)
        l_img = self.huber(z_img_aux,   z_true)
        l_morph = self.huber(z_morph_aux, z_true)
        total = (l_nll
                   + self.beta * beta_scale * l_huber
                   + self.alpha * (l_img + l_morph))
        return total, {
            'loss_nll'  : l_nll.item(),
            'loss_huber': l_huber.item(),
            'loss_img'  : l_img.item(),
            'loss_morph': l_morph.item(),
            'loss_total': total.item(),
        }
