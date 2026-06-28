import os
import copy  # 必须导入 copy，解决 GPT 的浅拷贝 Bug
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from data_loader import get_dataloaders
from model import get_model
from model_no_se import get_pure_efficientnet
from utils import set_seed

# ================== 配置区 ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = r"E:\Project_Final\data\RS_Classification"
NUM_CLASSES = 10
BATCH_SIZE = 4
ACCUMULATION_STEPS = 2
EPOCHS = 15
SEEDS = [1024, 2026, 3038]


def train_and_eval_supervised(model, train_loader, val_loader, test_loader):
    model = model.to(DEVICE)
    for param in model.parameters(): param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()  # 确保梯度清零

        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)

            # 【保留好孩子的梯度累加，保护您的 4GB 显存】
            loss = criterion(outputs, labels) / ACCUMULATION_STEPS
            loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        # Validation (只看验证集，绝对不碰测试集)
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        # 早停逻辑：验证集创新高时，保存模型权重
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 【修复 GPT 的致命 Bug：必须深拷贝，否则保存的权重会跟着变！】
            best_state = copy.deepcopy(model.state_dict())

        scheduler.step()

    # ==================================================
    # 终极测试：所有 Epoch 跑完后，加载最好的权重，只测一次！
    # ==================================================
    model.load_state_dict(best_state)
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Final Testing", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            test_correct += (outputs.argmax(1) == labels).sum().item()
            test_total += labels.size(0)

    return (test_correct / test_total) * 100


def main():
    se_accs, nose_accs = [], []
    for seed in SEEDS:
        print(f"\n--- 正在进行种子 {seed} 的纯监督方差分析 ---")
        set_seed(seed)
        train_loader, val_loader, test_loader, _ = get_dataloaders(DATA_ROOT, BATCH_SIZE, num_workers=0)

        # 跑 No-SE 基线
        model_nose = get_pure_efficientnet(NUM_CLASSES, pretrained=True)
        acc_nose = train_and_eval_supervised(model_nose, train_loader, val_loader, test_loader)
        nose_accs.append(acc_nose)
        print(f"Seed {seed} | No-SE 最终测试准确率: {acc_nose:.2f}%")

        # 跑 SE 提纯版
        set_seed(seed)
        model_se = get_model(NUM_CLASSES, pretrained=True)
        acc_se = train_and_eval_supervised(model_se, train_loader, val_loader, test_loader)
        se_accs.append(acc_se)
        print(f"Seed {seed} | SE-EffNet 最终测试准确率: {acc_se:.2f}%")

    print("\n" + "=" * 60)
    print("【纯监督】多种子方差分析结果 (Mean ± Std):")
    print(f"No-SE Baseline: {np.mean(nose_accs):.2f}% ± {np.std(nose_accs):.2f}%")
    print(f"SE-EfficientNet: {np.mean(se_accs):.2f}% ± {np.std(se_accs):.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()