import torch.nn as nn
from torchvision.models import VisionTransformer

def get_vit_regressor(in_channels=5):
    model = VisionTransformer(
        image_size = 224,
        patch_size = 16,
        num_layers = 12,
        num_heads  = 12,
        hidden_dim = 768,
        mlp_dim = 3072,
        num_classes = 1,
        dropout = 0.1,
        attention_dropout = 0.1,
    )
    model.conv_proj = nn.Conv2d(in_channels, 768, kernel_size=16, stride=16, bias=False)
    nn.init.xavier_uniform_(model.conv_proj.weight)
    return model