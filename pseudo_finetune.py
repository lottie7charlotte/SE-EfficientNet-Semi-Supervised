import os
import glob
import torch
import shutil
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score

from model import get_model
from data_loader import get_dataloaders, UnlabeledDataset, get_transforms
from utils import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = r"E:\Project_Final\data\RS_Classification"
CHECKPOINT = "./checkpoints/base_best.pth"  # 指向 99.29% 的权重
NUM_CLASSES = 10

# ================== 核心超参数区 ==================
BATCH_SIZE = 4  # 降回 4，防 OOM
CONFIDENCE_THRESHOLD = 0.90  # 恢复到你论文中最终承诺的 0.90
MAX_PER_CLASS = 97  # 严格履行完美均衡


# ==================================================

def generate_pseudo_labels(model):
    print("正在清理上一轮实验残留的伪标签...")
    train_dir = os.path.join(DATA_ROOT, "train")
    for cls_name in os.listdir(train_dir):
        cls_path = os.path.join(train_dir, cls_name)
        if os.path.isdir(cls_path):
            old_pseudos = glob.glob(os.path.join(cls_path, "pseudo_*"))
            for p in old_pseudos:
                os.remove(p)
    print("清理完毕！")

    model.eval()
    transform = get_transforms(is_train=False)
    unlabeled_dataset = UnlabeledDataset(
        os.path.join(DATA_ROOT, "unlabeled"), transform=transform
    )

    high_conf_samples = []
    class_counts = {i: 0 for i in range(NUM_CLASSES)}

    print("正在生成伪标签...")
    with torch.no_grad():
        for image, img_path in tqdm(unlabeled_dataset):
            image = image.unsqueeze(0).to(DEVICE)
            output = model(image)
            probs = torch.softmax(output, dim=1)
            conf, pred = probs.max(dim=1)
            conf, pred = conf.item(), pred.item()

            if conf >= CONFIDENCE_THRESHOLD and class_counts[pred] < MAX_PER_CLASS:
                high_conf_samples.append((img_path, pred, conf))
                class_counts[pred] += 1

    print(f"筛选出 {len(high_conf_samples)} 个高置信度样本")
    print(f"各类别分布: {class_counts}")

    for img_path, label, _ in high_conf_samples:
        cls_name = os.path.basename(os.path.dirname(img_path))
        dst_dir = os.path.join(DATA_ROOT, "train", cls_name)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, f"pseudo_{os.path.basename(img_path)}")
        shutil.copy(img_path, dst_path)

    return len(high_conf_samples)


def finetune():
    print(f"使用设备: {DEVICE}")
    model = get_model(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))

    pseudo_count = generate_pseudo_labels(model)
    print(f"已将 {pseudo_count} 张伪标签图像加入训练集")

    train_loader, val_loader, _, _ = get_dataloaders(DATA_ROOT, BATCH_SIZE, num_workers=0)

    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    accumulation_steps = 2

    best_val_acc = 0.0
    for epoch in range(1, 6):
        model.train()
        all_preds, all_labels = [], []
        pbar = tqdm(train_loader, desc=f"Finetune Epoch {epoch}")

        optimizer.zero_grad()

        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

        train_acc = accuracy_score(all_labels, all_preds)

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

        print(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "./checkpoints/final_best.pth")

    print(f"微调完成！最佳验证准确率: {best_val_acc:.4f}")


if __name__ == "__main__":
    set_seed(2026)
    finetune()