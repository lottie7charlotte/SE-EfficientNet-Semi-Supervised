import os
import glob

print("正在清理残留的伪标签...")
train_dir = r"E:\Project_Final\data\RS_Classification\train"
count = 0
for cls_name in os.listdir(train_dir):
    cls_path = os.path.join(train_dir, cls_name)
    if os.path.isdir(cls_path):
        old_pseudos = glob.glob(os.path.join(cls_path, "pseudo_*"))
        for p in old_pseudos:
            os.remove(p)
            count += 1
print(f"清理完毕！共删除了 {count} 张伪标签图片。现在的 train 文件夹绝对纯净了！")