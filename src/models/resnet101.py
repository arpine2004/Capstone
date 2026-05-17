import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GalaxyAutoencoder(nn.Module):
    def __init__(self, in_channels=5, latent_dim=256):
        super().__init__()
        backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(backbone.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.encoder_cnn = nn.Sequential(*list(backbone.children())[:-1])
        self.encoder_fc = nn.Linear(2048, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.SiLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),  nn.SiLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),  nn.SiLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),  nn.SiLU(inplace=True),
            nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1),
        )

    def encode(self, x):
        return self.encoder_fc(self.encoder_cnn(x).flatten(1))

    def decode(self, z):
        x = self.decoder_input(z).view(-1, 256, 4, 4)
        return F.interpolate(self.decoder(x), size=(127, 127), mode='bilinear', align_corners=False)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

class ResNet101Regressor(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.xavier_uniform_(self.head[0].weight)
        nn.init.xavier_uniform_(self.head[4].weight)
        nn.init.xavier_uniform_(self.head[8].weight)

    def forward(self, z):
        return self.head(z).squeeze(-1)

class RedshiftLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha  = alpha
        self.smooth = nn.SmoothL1Loss(beta=0.1)

    def forward(self, pred, target):
        base = self.smooth(pred, target)
        dz = (pred - target).abs() / (1.0 + target.abs())
        outlier_mask = dz > 0.15
        outlier_loss = dz[outlier_mask].mean() if outlier_mask.any() else torch.tensor(0.0, device=pred.device)
        return self.alpha * base + (1.0 - self.alpha) * outlier_loss

class FullModel(nn.Module):
    def __init__(self, autoencoder, regressor):
        super().__init__()
        self.encoder_cnn = autoencoder.encoder_cnn
        self.encoder_fc = autoencoder.encoder_fc
        self.regressor = regressor

    def forward(self, x):
        z = self.encoder_cnn(x).flatten(1)
        z = self.encoder_fc(z)
        return self.regressor(z)