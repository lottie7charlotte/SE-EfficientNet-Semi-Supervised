import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torchvision.models import resnet50
from tqdm import tqdm

from data_loader import get_dataloaders
from model import get_model  # 你的主模型
from model_no_se import get_pure_efficientnet  # 无SE模型

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = r"./data/RS_Classification"
NUM_CLASSES = 10
# 测试阶段无梯度，显存占用极小，Batch Size 开到 16 毫无压力
BATCH_SIZE = 16


# ================= 重新定义 ResNet50 以加载权重 =================
class SEResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(SEResNet50, self).__init__()
        self.backbone = resnet50()
        self.feature_dim = 2048
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, self.feature_dim // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim // 16, self.feature_dim, bias=False),
            nn.Sigmoid()
        )
        self.backbone.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        b, c, h, w = features.shape
        se_weight = self.se(features).view(b, c, 1, 1)
        features = features * se_weight
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        output = self.backbone.fc(features)
        return output


# ================================================================

def test_model(model, model_name, checkpoint_path, test_loader):
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ 找不到权重文件: {checkpoint_path}，跳过测试。")
        return

    # 加载权重
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()  # 开启测试模式

    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = test_correct / test_total
    print(f"✅ {model_name} - 独立测试集最终准确率: {test_acc * 100:.2f}%\n")


def main():
    print("=" * 60)
    print("🚀 开始进行最终独立测试集 (Test Dataset) 性能评估")
    print("=" * 60)

    # 提取 test_loader (20% 纯净测试集)
    _, _, test_loader, _ = get_dataloaders(DATA_ROOT, BATCH_SIZE, num_workers=0)
    print(f"独立测试集图片总数: {len(test_loader.dataset)} 张\n")

    # 1. MobileNetV3 (纯监督)
    mbnet = mobilenet_v3_small()
    mbnet.classifier[3] = nn.Linear(mbnet.classifier[3].in_features, NUM_CLASSES)
    test_model(mbnet, "MobileNetV3 (纯监督)", "./checkpoints/mobilenet_best.pth", test_loader)

    # 2. ResNet50 + SE (纯监督)
    resnet = SEResNet50(num_classes=NUM_CLASSES)
    test_model(resnet, "ResNet50 + SE (纯监督)", "./checkpoints/resnet50_best.pth", test_loader)

    # 3. EffNet-NoSE (纯监督)
    effnet_no_se_base = get_pure_efficientnet(num_classes=NUM_CLASSES, pretrained=False)
    test_model(effnet_no_se_base, "EffNet-无SE (纯监督)", "./checkpoints/effnet_no_se_base.pth", test_loader)

    # 4. EffNet-NoSE (半监督)
    effnet_no_se_final = get_pure_efficientnet(num_classes=NUM_CLASSES, pretrained=False)
    test_model(effnet_no_se_final, "EffNet-无SE (半监督)", "./checkpoints/effnet_no_se_final.pth", test_loader)

    # 5. EffNet-SE (纯监督主模型)
    effnet_se_base = get_model(num_classes=NUM_CLASSES, pretrained=False)
    test_model(effnet_se_base, "SE-EffNet (纯监督)", "./checkpoints/base_best.pth", test_loader)

    # 6. EffNet-SE (半监督终极形态)
    effnet_se_final = get_model(num_classes=NUM_CLASSES, pretrained=False)
    test_model(effnet_se_final, "SE-EffNet (半监督)", "./checkpoints/final_best.pth", test_loader)

    print("🎉 所有模型的最终 Test Accuracy 已生成！请将上述数据填入论文表格。")


if __name__ == "__main__":
    main()
