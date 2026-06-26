import os
import shutil
import random
from tqdm import tqdm
from utils import set_seed

# 1. 强制对齐随机种子，保证每次切出来的测试集都完全一样
set_seed(2026)

# ================= 配置区 =================
# 你刚刚找到的源数据集路径 (包含 45 个类别，每个 700 张)
SOURCE_DIR = r"E:\Project_Final\data\NWPU-RESISC45"

# 新生成的数据集存放位置 (我们重新建一个干净的 RS_Classification)
TARGET_DIR = r"E:\Project_Final\data\RS_Classification"

# 你的 10 个目标类别
CLASSES = [
    "airplane", "airport", "beach", "bridge", "forest",
    "freeway", "harbor", "industrial_area", "parking_lot", "stadium"
]

# 每个类别的分配数量：总共 700
NUM_TRAIN = 392
NUM_VAL = 70
NUM_TEST = 140
NUM_UNLABELED = 98


# ==========================================

def create_dirs():
    """创建空的目录结构"""
    for split in ["train", "val", "test", "unlabeled"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)


def split_dataset():
    print(f"🔪 开始重新物理划分极其纯净的数据集...")
    create_dirs()

    for cls in tqdm(CLASSES, desc="Processing Classes"):
        cls_source_dir = os.path.join(SOURCE_DIR, cls)

        # 获取该类别下的所有图片路径并排序（保证平台无关性，且不受系统索引影响）
        if not os.path.exists(cls_source_dir):
            print(f"❌ 找不到源文件夹: {cls_source_dir}")
            return

        images = sorted(os.listdir(cls_source_dir))

        if len(images) != 700:
            print(f"⚠️ 警告: 类别 {cls} 的图片数量不是 700，而是 {len(images)}！")
            continue

        # 强制打乱图片顺序 (有了 set_seed(2026)，每次打乱的顺序都一样)
        random.shuffle(images)

        # 严格按照数量切片
        train_imgs = images[:NUM_TRAIN]
        val_imgs = images[NUM_TRAIN: NUM_TRAIN + NUM_VAL]
        test_imgs = images[NUM_TRAIN + NUM_VAL: NUM_TRAIN + NUM_VAL + NUM_TEST]
        unlabeled_imgs = images[NUM_TRAIN + NUM_VAL + NUM_TEST: NUM_TRAIN + NUM_VAL + NUM_TEST + NUM_UNLABELED]

        # 复制文件到新目录
        def copy_files(img_list, split_name):
            for img in img_list:
                src_path = os.path.join(cls_source_dir, img)
                dst_path = os.path.join(TARGET_DIR, split_name, cls, img)
                shutil.copy(src_path, dst_path)

        copy_files(train_imgs, "train")
        copy_files(val_imgs, "val")
        copy_files(test_imgs, "test")
        copy_files(unlabeled_imgs, "unlabeled")

    print("\n✅ 数据集重铸完成！")
    print(f"请检查 {TARGET_DIR}\\train 文件夹，现在每个类别绝对只有 392 张图！")


if __name__ == "__main__":
    split_dataset()