import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


def get_pure_efficientnet(num_classes=10, pretrained=True):
    """不包含SE模块的纯EfficientNetV2-S"""
    if pretrained:
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = efficientnet_v2_s(weights=weights)
    else:
        model = efficientnet_v2_s(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(1280, num_classes)
    )
    return model