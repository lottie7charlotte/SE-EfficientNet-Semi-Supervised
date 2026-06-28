import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from data_loader import get_dataloaders
from model import get_model

# 配置
NUM_CLASSES = 10
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = r"./data/RS_Classification"
CHECKPOINT = "./checkpoints/final_best.pth"  # 使用你最强模型的权重
CLASS_NAMES = ['airplane', 'airport', 'beach', 'bridge', 'forest',
               'freeway', 'harbor', 'industrial', 'parking', 'stadium']


def main():
    print("正在加载测试集与终极模型...")
    _, _, test_loader, _ = get_dataloaders(DATA_ROOT, BATCH_SIZE, num_workers=0)
    model = get_model(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE))
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n" + "=" * 50)
    print("请将以下数据填入 Table 5 和 Table 6：")
    print("=" * 50)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

    # 绘制高颜值混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={"size": 12})
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title('Confusion Matrix - Final SE-EfficientNetV2-S', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('new_confusion_matrix.png')
    print("\n✅ 全新的高清混淆矩阵已保存为 'New_confusion_matrix.png'，请替换论文里的 Figure 8！")


if __name__ == "__main__":
    main()
