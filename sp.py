
import os
from data.dataset2 import Merge_Two_modality

PATH_VIDEO = "/home/a2/Datasets/多模态数据集/Video"

files = Merge_Two_modality(PATH_VIDEO, threshold=0)

train_ids = {f"{i:02d}" for i in range(1, 9)}
val_ids = {"09", "10"}

train_count = 0
val_count = 0

for rgb_path, rfid_path, label in files:
    subject_id = os.path.basename(os.path.dirname(rfid_path))

    if subject_id in train_ids:
        train_count += 1
    elif subject_id in val_ids:
        val_count += 1

print("========== 数据集统计 ==========")
print(f"训练集样本：{train_count}")
print(f"验证集样本：{val_count}")
print(f"实际使用样本总数：{train_count + val_count}")
