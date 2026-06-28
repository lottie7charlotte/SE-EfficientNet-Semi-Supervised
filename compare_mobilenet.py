import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from tqdm import tqdm
from data_loader import get_dataloaders
from utils import set_seed  # 核心：必须固定种子

set_seed(2026)

NUM_CLASSES = 10
BATCH_SIZE = 4  # 核心：强制降为 4 防 OOM
EPOCHS = 15
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = r"./data/RS_Classification"
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)


def train_mobilenet(model, train_loader, val_loader, epochs, save_name):
    model = model.to(DEVICE)
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable:,} (全量解冻)")

    optimizer = optim.Adam(model.parameters(), lr=LR,weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    accumulation_steps = 2  # 配合 Batch Size 4

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        optimizer.zero_grad()
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps
            preds = outputs.argmax(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

        train_acc = train_correct / train_total

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

        print(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, save_name))
            print(f"  -> 保存最佳模型 (Val Acc: {val_acc:.4f})")

    return best_val_acc

def main():
    print("=" * 50)
    print("对比实验：MobileNetV3-Small (轻量级基线)")
    print("=" * 50)
    train_loader, val_loader, _, _ = get_dataloaders(DATA_ROOT, BATCH_SIZE, num_workers=0)
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    best_acc = train_mobilenet(model, train_loader, val_loader, EPOCHS, "mobilenet_best.pth")
    print(f"\n✅ MobileNetV3 最佳验证准确率: {best_acc:.4f}")

if __name__ == "__main__":
    main()
