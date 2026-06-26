import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力模块"""

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ImprovedEfficientNet(nn.Module):
    """改进的EfficientNet：添加SE注意力模块"""

    def __init__(self, num_classes=10, pretrained=True):
        super(ImprovedEfficientNet, self).__init__()
        if pretrained:
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_v2_s(weights=weights)
        else:
            self.backbone = efficientnet_v2_s(weights=None)

        self.feature_dim = 1280
        self.se_block = SEBlock(self.feature_dim, reduction=16)

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(self.feature_dim, num_classes)
        )

    def forward(self, x):
        # 1. 提取特征，输出维度为 [batch_size, 1280, 7, 7]
        features = self.backbone.features(x)

        # 2. 关键修复：在 GAP 之前插入 SE 模块！
        # 此时空间维度是 7x7，SE 模块的“空间压缩”逻辑在此处完美成立
        features = self.se_block(features)

        # 3. 全局平均池化层 (GAP)，压缩为 1x1
        features = self.backbone.avgpool(features)

        # 4. 展平并传入分类头
        features = torch.flatten(features, 1)
        output = self.backbone.classifier(features)
        return output


def get_model(num_classes=10, pretrained=True):
    return ImprovedEfficientNet(num_classes=num_classes, pretrained=pretrained)