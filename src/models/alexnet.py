import torch
import torch.nn as nn
from torchvision import models


class AlexNetRegressor(nn.Module):
    def __init__(self, in_channels=5, latent_dim=256, freeze_backbone=False):
        super().__init__()

        backbone = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        new_conv = nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2)
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        with torch.no_grad():
            new_conv.weight[:, :3] = backbone.features[0].weight
            new_conv.weight[:, 3] = backbone.features[0].weight.mean(dim=1)
            new_conv.weight[:, 4] = backbone.features[0].weight.mean(dim=1)
        backbone.features[0] = new_conv

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.head = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 1),
        )

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.head(x).squeeze(1)
