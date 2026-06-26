import os
import glob
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from data_loader import get_dataloaders, UnlabeledDataset, get_transforms
from model_no_se import get_pure_efficientnet
from utils import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = r"E:\Project_Final\data\RS_Classification"
CHECKPOINT = "./checkpoints/effnet_no_se_base.pth"
NUM_CLASSES = 10
BATCH_SIZE = 4
CONFIDENCE_THRESHOLD = 0.90
MAX_PER_CLASS = 97


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

    for img_path, label, _ in high_conf_samples:
        cls_name = os.path.basename(os.path.dirname(img_path))
        dst_dir = os.path.join(DATA_ROOT, "train", cls_name)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, f"pseudo_{os.path.basename(img_path)}")
        shutil.copy(img_path, dst_path)
    return len(high_conf_samples)


def finetune():
    model = get_pure_efficientnet(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    pseudo_count = generate_pseudo_labels(model)

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
        # +++ 加上这三行打印代码，否则你跑完不知道成绩！ +++
        print(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "./checkpoints/effnet_no_se_final.pth")
            print(f"  -> 保存最佳模型 (Val Acc: {val_acc:.4f})")

        # +++ 在 for 循环结束后，加上最后这句总结 +++
    print(f"\n✅ 无SE模块半监督微调完成！最佳验证准确率: {best_val_acc:.4f}")



if __name__ == "__main__":
    set_seed(2026)
    finetune()