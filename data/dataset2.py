import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import re


# =========================================================================

# =========================================================================

def Get_Folder_File(path, filetype):
    txt_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(filetype):
                txt_files.append(os.path.join(root, file))
    return txt_files


def get_mapping(path):
    label = path[path.rfind('/') + 1:path.find('(')]
    level1_mapping = {"Video/Origin": "RFID/Origin", "Video": "RFID"}
    original_path = path
    for old, new in level1_mapping.items():
        original_path = original_path.replace(old, new)
    path_parts = original_path.split(os.sep)
    if len(path_parts) >= 3:
        path_parts[-3] = re.sub(r'\D', '', path_parts[-3])
    if len(path_parts) >= 2:
        path_parts.pop(-2)
    if len(path_parts) >= 1:
        path_parts[-1] = "rfid_" + path_parts[-1]
    new_path = os.sep.join(path_parts)
    return new_path, label


def Merge_Two_modality(path_Video, threshold=0):
    FL_Video = Get_Folder_File(path_Video, ".txt")
    if threshold > 0:
        FL_Video = FL_Video[:threshold]
    elif threshold < 0:
        FL_Video = FL_Video[threshold:]
    FL = []
    for i in FL_Video:
        rfid_path, label = get_mapping(i)
        if os.path.exists(rfid_path):
            FL.append([i, rfid_path, label])
    return FL


def Get_RGB_datas(i):
    datas = []
    try:
        with open(i, 'r') as file:
            for line in file: datas.append(eval(line.strip()))
    except:
        return np.zeros((165, 17, 2))
    if len(datas) < 165:
        for i in range(165 - len(datas)): datas.append([[0, 0] for _ in range(17)])
    return np.array(datas[:165])


def Get_RFID_datas(i):
    rows = 50
    data_matrix = np.zeros((rows, 16, 3))
    try:
        with open(i, "r") as f:
            position_isdata = [0] * 16
            for index, line in enumerate(f):
                item = line.strip().split(" ")
                node_index = int(item[1][2:])
                if node_index >= 16 or position_isdata[node_index] >= rows: continue
                data_matrix[position_isdata[node_index], node_index, :] = [float(item[3]), float(item[4]),
                                                                           float(item[5])]
                position_isdata[node_index] += 1
    except:
        pass
    return data_matrix


# =========================================================================
# 【Part 2】 你的双向干扰逻辑 (GCDF 专用)
# =========================================================================

def apply_rgb_occlusion(data_numpy, ratio):
    """
    视觉干扰：将选中的关键点替换为高斯噪声
    """
    # 1. 如果遮挡率 <= 0，直接返回原数据
    if ratio <= 0: return data_numpy

    T, V, C = data_numpy.shape
    occluded = data_numpy.copy()
    # ====================================================

    # 2. 计算要干扰多少个点
    num_mask = int(V * ratio)

    if num_mask > 0:
        # 随机选择受害点
        mask_indices = np.random.choice(V, num_mask, replace=False)

        # 【改为高斯噪声】模拟传感器故障、恶劣天气
        # 这样 Baseline 就无法通过 ReLU 偷懒了，它会崩盘
        noise = np.random.normal(loc=0, scale=1.0, size=(T, num_mask, C))

        # 现在 occluded 已经定义了，这里就不会报错了
        occluded[:, mask_indices, :] = noise

    return occluded

def apply_rfid_noise(data_numpy, ratio):
    """
    RFID干扰策略：高斯噪声 (Gaussian Noise)
    物理含义：模拟多径效应、同频干扰、环境反射。
    射频传感器特性：受干扰时信号不会消失，而是变得杂乱无章。
    """
    if ratio <= 0: return data_numpy

    T, V, C = data_numpy.shape
    noisy = data_numpy.copy()

    # 计算要干扰多少个标签
    num_mask = int(V * ratio)

    if num_mask > 0:
        mask_indices = np.random.choice(V, num_mask, replace=False)

        # 【保持不变】RFID 依然使用噪声填入
        # scale=1.0 模拟强烈的环境噪声
        noise = np.random.normal(loc=0, scale=1, size=(T, num_mask, C))

        # 填入噪声
        noisy[:, mask_indices, :] = noise

    return noisy


# =========================================================================
# 【Part 3】 超级兼容版 Dataset (解决所有报错)
# =========================================================================

class GCDF_Dataset(Dataset):
    def __init__(self, file_list, mode='train', rgb_ratio=0.0, rfid_ratio=0.0, occlusion_ratio=None):
        """
        全能初始化函数：
        - 如果你只传 occlusion_ratio (旧代码)，它会自动转给 rgb_ratio。
        - 如果你传 rgb_ratio / rfid_ratio (新代码)，它就用新的。
        """
        self.file_list = file_list
        self.mode = mode

        # --- 兼容性处理 ---
        if occlusion_ratio is not None:
            # 兼容旧的基线测试代码
            self.rgb_ratio = occlusion_ratio
            self.rfid_ratio = 0.0
        else:
            # 使用新的双向参数
            self.rgb_ratio = rgb_ratio
            self.rfid_ratio = rfid_ratio

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        rgb_path, rfid_path, label = self.file_list[idx]

        # 1. 师兄的读取
        rgb_np = Get_RGB_datas(rgb_path)
        rfid_np = Get_RFID_datas(rfid_path)

        tcp_rgb = 1.0
        tcp_rfid = 1.0

        # 2. 你的干扰逻辑
        # 应用 RGB 干扰
        if self.rgb_ratio > 0:
            prob = 0.5 if self.mode == 'train' else 1.0
            if np.random.random() < prob:
                rgb_np = apply_rgb_occlusion(rgb_np, self.rgb_ratio)
                tcp_rgb = 1.0 - self.rgb_ratio

        # 应用 RFID 干扰
        if self.rfid_ratio > 0:
            prob = 0.5 if self.mode == 'train' else 1.0
            if np.random.random() < prob:
                rfid_np = apply_rfid_noise(rfid_np, self.rfid_ratio)
                tcp_rfid = 1.0 - self.rfid_ratio

        # 3. 转 Tensor (统一格式)
        rgb_tensor = torch.tensor(rgb_np, dtype=torch.float32)
        rfid_tensor = torch.tensor(rfid_np, dtype=torch.float32)
        label_tensor = torch.tensor(int(label), dtype=torch.long)

        # 4. 返回 5 个值
        # 即使是基线模型，多接收几个返回值也不会报错(只要你在循环里用 _ 接住就行)
        return rgb_tensor, rfid_tensor, label_tensor, torch.tensor(tcp_rgb, dtype=torch.float32), torch.tensor(tcp_rfid,

                                                                                                      dtype=torch.float32)


def split_by_volunteer_id(all_files, test_ids):
    """
    根据文件夹名称（志愿者ID）进行严格划分
    all_files: [[rgb_path, rfid_path, label], ...]
    test_ids: 测试集的ID列表 (字符串格式), 例如 ['09', '10']
    """
    train_files = []
    test_files = []

    # 统计一下每个ID有多少数据，方便核对
    id_counts = {}

    print(f"🔄 正在进行跨对象划分...")
    print(f" -> 测试集目标 ID: {test_ids}")

    for item in all_files:
        rgb_path = item[0]  # 例如 .../RFID/01/rfid_0(10_40).txt

        # === 核心逻辑：提取父文件夹名称 ===
        # os.path.dirname 获取文件的目录 (.../RFID/01)
        # os.path.basename 获取目录的最后一部分 (01)
        # ⚠️ 注意：这里假设 RGB 和 RFID 的文件夹结构一致，都是 .../ID/文件
        # 如果你的 RGB 路径不一样，请根据实际情况调整，这里以 item[1] (RFID路径) 为准更稳妥

        # 既然截图展示的是 RFID 文件夹结构，我们用 RFID 路径来判断 ID
        rfid_path = item[1]
        subject_id = os.path.basename(os.path.dirname(rfid_path))

        # 记录统计信息
        if subject_id not in id_counts:
            id_counts[subject_id] = 0
        id_counts[subject_id] += 1

        # 划分
        if subject_id in test_ids:
            test_files.append(item)
        else:
            train_files.append(item)

    print("\n📊 数据分布统计:")
    for uid, count in sorted(id_counts.items()):
        status = "【测试集】" if uid in test_ids else "训练集"
        print(f"   ID: {uid} -> {count} 条数据 {status}")

    print(f"\n✅ 划分结果: 训练集 {len(train_files)} | 测试集 {len(test_files)}")

    if len(train_files) == 0 or len(test_files) == 0:
        raise ValueError("❌ 划分失败！训练集或测试集为空。请检查代码里的 test_ids 是否和文件夹名 '01', '02' 等匹配！")

    return train_files, test_files