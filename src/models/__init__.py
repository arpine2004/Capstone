from .alexnet import AlexNetRegressor
from .vgg16 import VGG16Regressor
from .vit import get_vit_regressor
from .resnet101 import GalaxyAutoencoder, ResNet101Regressor, FullModel, RedshiftLoss
from .crossattn import LoRALinear, CrossAttentionFusion, FusionModelCrossAttn
from .crossattn import RedshiftLoss as CrossAttnRedshiftLoss
from .phys_informed import (
    BandEmbedding,
    BandAwareImageBranch,
    BandAwareMorphologyBranch,
    GaussianHead,
    BandAlignedFusion,
    PhysicallyInformedLateFusionNet,
    ProbabilisticRedshiftLoss,
)

__all__ = [
    "AlexNetRegressor",
    "VGG16Regressor",
    "get_vit_regressor",
    "GalaxyAutoencoder",
    "ResNet101Regressor",
    "FullModel",
    "RedshiftLoss",
    "FusionRedshiftLoss",
    "MorphMLP",
    "FusionModel",
    "LoRALinear",
    "CrossAttentionFusion",
    "FusionModelCrossAttn",
    "CrossAttnRedshiftLoss",
    "BandEmbedding",
    "BandAwareImageBranch",
    "BandAwareMorphologyBranch",
    "GaussianHead",
    "BandAlignedFusion",
    "PhysicallyInformedLateFusionNet",
    "ProbabilisticRedshiftLoss",
]

# Note: The above imports are ordered to reflect the chronological development of the models, from simpler baselines to more complex architectures.