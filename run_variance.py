import os
import glob
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

# ================== 导入你的模块 ==================
from data_loader import get_dataloaders, UnlabeledDataset, get_transforms
from model import get_model  # SE 模型
from model_no_se import get_pure_efficientnet  # No-SE 模型
from utils import set_seed

# ================== 核心配置区 ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = r"E:\Project_Final\data\RS_Classification"
CHECKPOINT_DIR = "./checkpoints_variance"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

NUM_CLASSES = 10
BATCH_SIZE = 4
ACCUMULATION_STEPS = 2
EPOCHS_BASE = 15
EPOCHS_FINETUNE = 5
LR_BASE = 1e-4
LR_FINETUNE = 1e-5
CONFIDENCE_THRESHOLD = 0.90
MAX_PER_CLASS = 97

CLASSES = ['airplane', 'airport', 'beach', 'bridge', 'forest',
           'freeway', 'harbor', 'industrial_area', 'parking_lot', 'stadium']

SEEDS = [1024, 2026, 3038]  # 使用3个绝对不同的随机种子


# ================== 物理清理函数 ==================
def clean_pseudo_labels():
    """极其严厉的物理清理：强制删除所有 pseudo_ 开头的图片，保证数据绝对纯净"""
    train_dir = os.path.join(DATA_ROOT, "train")
    count = 0
    for cls_name in os.listdir(train_dir):
        cls_path = os.path.join(train_dir, cls_name)
        if os.path.isdir(cls_path):
            old_pseudos = glob.glob(os.path.join(cls_path, "pseudo_*"))
            for p in old_pseudos:
                os.remove(p)
                count += 1
    print(f"🧹 已彻底清理 {count} 张残留的伪标签，数据集现已绝对纯净！")


# ================== 核心训练函数 ==================
def train_loop(model, loader, val_loader, epochs, lr, save_path):
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        optimizer.zero_grad()

        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels) / ACCUMULATION_STEPS
            loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()

        # 验证
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    # 清理显存
    del optimizer, scheduler, criterion
    torch.cuda.empty_cache()


# ================== 提取伪标签 ==================
def generate_pseudo_labels(model):
    model.eval()
    transform = get_transforms(is_train=False)
    unlabeled_dataset = UnlabeledDataset(os.path.join(DATA_ROOT, "unlabeled"), transform=transform)

    high_conf_samples = []
    class_counts = {i: 0 for i in range(NUM_CLASSES)}

    with torch.no_grad():
        for image, img_path in unlabeled_dataset:
            image = image.unsqueeze(0).to(DEVICE)
            output = model(image)
            probs = torch.softmax(output, dim=1)
            conf, pred = probs.max(dim=1)
            conf, pred = conf.item(), pred.item()

            if conf >= CONFIDENCE_THRESHOLD and class_counts[pred] < MAX_PER_CLASS:
                high_conf_samples.append((img_path, pred, conf))
                class_counts[pred] += 1

    for img_path, label, _ in high_conf_samples:
        cls_name = os.path.basename(os.path.dirname(img_path))
        dst_dir = os.path.join(DATA_ROOT, "train", cls_name)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, f"pseudo_{os.path.basename(img_path)}")
        shutil.copy(img_path, dst_path)

    print(f"📥 成功提取 {len(high_conf_samples)} 张高置信度伪标签。")


# ================== 测试并获取单类F1 ==================
def evaluate_model_f1(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算单类 F1，乘以 100 换算为百分比
    f1_scores = f1_score(all_labels, all_preds, average=None) * 100
    return f1_scores


# ================== 完整单次实验流水线 ==================
def run_pipeline(use_se, seed):
    set_seed(seed)
    model_name = "SE-EfficientNet" if use_se else "No-SE EfficientNet"
    print(f"\n" + "★" * 50)
    print(f"★ 正在执行: {model_name} | Random Seed: {seed}")
    print("★" * 50)

    # 0. 获取 Dataloader & 清理脏数据
    train_loader, val_loader, test_loader, _ = get_dataloaders(DATA_ROOT, BATCH_SIZE, num_workers=0)
    clean_pseudo_labels()

    base_weight_path = os.path.join(CHECKPOINT_DIR, f"base_{'se' if use_se else 'nose'}_{seed}.pth")
    final_weight_path = os.path.join(CHECKPOINT_DIR, f"final_{'se' if use_se else 'nose'}_{seed}.pth")

    # 1. 实例化模型
    model = get_model(NUM_CLASSES, pretrained=True).to(DEVICE) if use_se else get_pure_efficientnet(NUM_CLASSES,
                                                                                                    pretrained=True).to(
        DEVICE)

    # 2. 纯监督基线训练
    print(f"\n>>> Phase 1: 纯监督基线训练 ({EPOCHS_BASE} Epochs)...")
    train_loop(model, train_loader, val_loader, EPOCHS_BASE, LR_BASE, base_weight_path)

    # 3. 生成伪标签
    print(f"\n>>> Phase 2: 生成平衡伪标签...")
    model.load_state_dict(torch.load(base_weight_path, map_location=DEVICE))
    generate_pseudo_labels(model)

    # 重新加载含有伪标签的数据集 (必须重载 Dataloader)
    train_loader, val_loader, test_loader, _ = get_dataloaders(DATA_ROOT, BATCH_SIZE, num_workers=0)

    # 4. 半监督微调
    print(f"\n>>> Phase 3: 半监督微调 ({EPOCHS_FINETUNE} Epochs)...")
    train_loop(model, train_loader, val_loader, EPOCHS_FINETUNE, LR_FINETUNE, final_weight_path)

    # 5. 独立测试集评估
    print(f"\n>>> Phase 4: 在 1400 张独立测试集上计算单类 F1...")
    model.load_state_dict(torch.load(final_weight_path, map_location=DEVICE))
    f1_scores = evaluate_model_f1(model, test_loader)

    # 清理现场，释放显存
    del model
    torch.cuda.empty_cache()
    clean_pseudo_labels()

    return f1_scores


# ================== 主干执行逻辑 ==================
def main():
    results_se = {cls: [] for cls in CLASSES}
    results_nose = {cls: [] for cls in CLASSES}

    # 循环遍历种子和模型配置
    for seed in SEEDS:
        # 跑 No-SE
        f1_nose = run_pipeline(use_se=False, seed=seed)
        for i, cls in enumerate(CLASSES):
            results_nose[cls].append(f1_nose[i])

        # 跑 SE
        f1_se = run_pipeline(use_se=True, seed=seed)
        for i, cls in enumerate(CLASSES):
            results_se[cls].append(f1_se[i])

    # ================== 打印终极表格 ==================
    print("\n\n" + "=" * 80)
    print("🎯 挂机任务圆满完成！这是你要复制进论文 Table 9 的终极数据 (Mean ± Std)")
    print("=" * 80)
    print(f"{'Category':<15} | {'No-SE Baseline (%)':<25} | {'SE-EfficientNet (%)':<25}")
    print("-" * 80)

    for cls in CLASSES:
        nose_mean, nose_std = np.mean(results_nose[cls]), np.std(results_nose[cls])
        se_mean, se_std = np.mean(results_se[cls]), np.std(results_se[cls])

        nose_str = f"{nose_mean:.2f} ± {nose_std:.2f}"
        se_str = f"{se_mean:.2f} ± {se_std:.2f}"

        print(f"{cls:<15} | {nose_str:<25} | {se_str:<25}")
    print("=" * 80)


if __name__ == "__main__":
    main()