import torch
import torch.nn as nn

class GalaxiesMLCNN(nn.Module):
    def __init__(self, in_channels: int = 5, width: int = 32, dropout: float = 0.15):
        super().__init__()

        def conv_bn_act(cin, cout, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
                nn.BatchNorm2d(cout),
                nn.SiLU(inplace=True),
            )

        def down_block(cin, cout):
            return nn.Sequential(
                conv_bn_act(cin, cout),
                conv_bn_act(cout, cout),
                nn.MaxPool2d(2),
            )

        c1, c2, c3, c4 = width, width*2, width*4, width*8

        self.backbone = nn.Sequential(
            down_block(in_channels, c1), 
            down_block(c1, c2),          
            down_block(c2, c3),          
            down_block(c3, c4),          
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4, c4 // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(c4 // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.backbone(x))).squeeze(1)

_model_tmp = GalaxiesMLCNN(in_channels=5, width=32, dropout=0.15)
n_params = sum(p.numel() for p in _model_tmp.parameters() if p.requires_grad)
print(f"GalaxiesMLCNN parameters: {n_params:,}")