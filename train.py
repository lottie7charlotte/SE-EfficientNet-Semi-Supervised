import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from data_loader import get_dataloaders
from model import get_model
# 导入固定随机种子的函数
from utils import set_seed

# ==========================================
# 1. 设置随机种子 (先用 1024 跑，跑完改成 2026 再跑一次)
# ==========================================
set_seed(2026)

# ==========================================
# 2. 基础配置
# ==========================================
NUM_CLASSES = 10
BATCH_SIZE = 4  # 降为 4，拯救 4GB 显存
EPOCHS = 15
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = r"E:\Project_Final\data\RS_Classification"
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)


def train_one_epoch(model, loader, criterion, optimizer):
    """带梯度累加的单轮训练函数"""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="Training")

    # 开启梯度累加 (4 * 2 = 等效 Batch Size 8)
    accumulation_steps = 2
    optimizer.zero_grad()

    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        # 损失值除以累加步数，保证梯度幅值一致
        loss = loss / accumulation_steps
        loss.backward()

        # 凑够 2 次才更新一次参数，大幅降低显存压力
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        # 还原真实的 loss 记录用于显示
        running_loss += loss.item() * accumulation_steps
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

    return running_loss / len(loader), accuracy_score(all_labels, all_preds)


def main():
    print("=" * 50)
    print(f"正在进行纯监督基线训练 (Batch Size={BATCH_SIZE}, Accumulation=2)")
    print("=" * 50)

    # 加载数据
    train_loader, val_loader, _, _ = get_dataloaders(DATA_ROOT, BATCH_SIZE, num_workers=0)

    # 加载你的 SE-EfficientNetV2-S 模型 (不加载预训练权重，因为是从头微调)
    model = get_model(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)

    # 确保全量解冻
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable:,} / {total:,}")

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # 训练阶段
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)

        # 验证阶段
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")

        # 保存当前随机种子下的最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 存为一个临时的名字，避免覆盖你之前最好的 base_best.pth
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "seed_temp_best.pth"))
            print(f"  -> 保存最佳模型 (Val Acc: {val_acc:.4f})")

    print(f"\n训练完成！当前随机种子下最佳验证准确率: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()