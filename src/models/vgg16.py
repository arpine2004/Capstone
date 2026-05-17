import torch
import torch.nn as nn
from torchvision import models

class VGG16Regressor(nn.Module):
    def __init__(self, in_channels=5, freeze_backbone=False):
        super().__init__()
        backbone = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        new_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        backbone.features[0] = new_conv

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(512, 512), nn.SiLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.SiLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.SiLU(inplace=True),
            nn.Linear(128, 1),
        )

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1)).squeeze(1)
