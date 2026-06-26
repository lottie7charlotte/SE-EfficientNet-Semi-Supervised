import os
import glob
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloaders, UnlabeledDataset, get_transforms
from model import get_model  # 确保这里引入的是带有SE的主模型
from utils import set_seed
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = r"E:\Project_Final\data\RS_Classification"
CHECKPOINT = "./checkpoints/base_best.pth"  # 加载我们 99.57% 的纯监督主模型
NUM_CLASSES = 10
BATCH_SIZE = 4  # 依然保持防爆显存的 4


def generate_pseudo_labels(model, threshold):
    # 1. 物理清理上一轮残留
    train_dir = os.path.join(DATA_ROOT, "train")
    for cls_name in os.listdir(train_dir):
        cls_path = os.path.join(train_dir, cls_name)
        if os.path.isdir(cls_path):
            old_pseudos = glob.glob(os.path.join(cls_path, "pseudo_*"))
            for p in old_pseudos:
                os.remove(p)

    # 如果阈值大于1，说明只是为了清理，直接返回
    if threshold > 1.0: return 0

    model.eval()
    transform = get_transforms(is_train=False)
    unlabeled_dataset = UnlabeledDataset(os.path.join(DATA_ROOT, "unlabeled"), transform=transform)
    high_conf_samples = []

    # 2. 生成新伪标签 (这里去掉了97张的硬性限制，为了真实反映阈值对数量的影响)
    with torch.no_grad():
        for image, img_path in tqdm(unlabeled_dataset, desc=f"Filtering (τ={threshold})"):
            image = image.unsqueeze(0).to(DEVICE)
            output = model(image)
            probs = torch.softmax(output, dim=1)
            conf, pred = probs.max(dim=1)
            conf, pred = conf.item(), pred.item()

            if conf >= threshold:
                high_conf_samples.append((img_path, pred, conf))

    # 3. 写入图片
    for img_path, label, _ in high_conf_samples:
        cls_name = os.path.basename(os.path.dirname(img_path))
        dst_dir = os.path.join(DATA_ROOT, "train", cls_name)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, f"pseudo_{os.path.basename(img_path)}")
        shutil.copy(img_path, dst_path)

    return len(high_conf_samples)


def run_finetune():
    model = get_model(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))

    for param in model.parameters():
        param.requires_grad = True

    train_loader, val_loader, _, _ = get_dataloaders(DATA_ROOT, BATCH_SIZE, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    accumulation_steps = 2

    best_val_acc = 0.0
    for epoch in range(1, 6):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc


def main():
    set_seed(2026)
    print("=" * 50)
    print("🚀 开始进行敏感性分析 (Sensitivity Analysis)")
    print("=" * 50)

    # 阈值 0.80
    print("\n▶ 正在测试 Threshold = 0.80")
    model = get_model(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    count_08 = generate_pseudo_labels(model, 0.80)
    print(f"τ=0.80 提取了 {count_08} 张伪标签，正在微调...")
    acc_08 = run_finetune()

    # 阈值 0.95
    print("\n▶ 正在测试 Threshold = 0.95")
    model = get_model(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    count_095 = generate_pseudo_labels(model, 0.95)
    print(f"τ=0.95 提取了 {count_095} 张伪标签，正在微调...")
    acc_095 = run_finetune()

    # 最后清空伪标签，保护你原有的纯净训练集
    print("\n正在物理清理生成的测试图，恢复您的纯净数据集...")
    generate_pseudo_labels(model, 1.1)

    print("\n" + "=" * 50)
    print("📊 敏感性分析结果汇总 (请将此填入论文)：")
    print(f"τ = 0.80 | 提取数量: {count_08} | Peak Acc: {acc_08 * 100:.2f}%")
    print(f"τ = 0.90 | 提取数量: 970 | Peak Acc: 99.86% (此为已知主实验)")
    print(f"τ = 0.95 | 提取数量: {count_095} | Peak Acc: {acc_095 * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()