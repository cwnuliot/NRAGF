# -*- coding: utf-8 -*-
"""
QMF-RGB/RFID 公平对比训练脚本（PyCharm 直接运行）

建议保存位置：NRAGF/train_qmf_fair.py
模型文件：NRAGF/models/qmf_rgb_rfid.py

原则：
1) 01-08 训练，09-10 验证；
2) 公共训练超参数与当前 NRAGF 对齐；
3) 训练时局部随机节点退化、Gaussian noise、modality dropout 与 NRAGF 对齐；
4) 保留 QMF 核心：能量置信度 + 动态模态级融合 + ranking loss；
5) 全新训练，不继承 NRAGF 或其他 baseline 权重。
"""

# =============================================================================
# 1. 统一配置
# =============================================================================
PATH_VIDEO = "/home/a2/Datasets/多模态数据集/Video"
TRAIN_SUBJECT_IDS = ("01", "02", "03", "04", "05", "06", "07", "08")
VAL_SUBJECT_IDS = ("09", "10")

SEED = 2026
BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 50
LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 0.00005
WEIGHT_DECAY = 5e-4
GRAD_CLIP_NORM = 5.0
USE_AMP = True
LABEL_SMOOTHING = 0.10

GRAPH_STRATEGY = "spatial"
BACKBONE_DROPOUT = 0.50

# QMF 官方 RGB-D 训练代码默认 lamb=0.1；confidence scale 也为 0.1。
QMF_RANK_LOSS_WEIGHT = 0.10
QMF_CONFIDENCE_SCALE = 0.10

# 与 NRAGF 当前训练脚本一致。
CORRUPTION_MODE = "random_nodes"
RGB_CORRUPTION_RATIO = 0.30
RFID_CORRUPTION_RATIO = 0.30
CORRUPTION_PROBABILITY = 0.50
RGB_NOISE_STD = 1.0
RFID_NOISE_STD = 1.0
MODALITY_DROPOUT_PROBABILITY = 0.15
PART_K_VALUES = (1, 2, 3)

OUTPUT_DIR = "outputs/qmf_fair_seed2026"
BEST_CHECKPOINT_NAME = "best_qmf_fair.pth"
LAST_CHECKPOINT_NAME = "last_qmf_fair.pth"
PRINT_INTERVAL = 20

# =============================================================================
# 2. 导入
# =============================================================================
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from data.dataset2 import Merge_Two_modality
from data.dataset_nragf import NRAGFDataset
from models.qmf_rgb_rfid import QMFRGBRFID


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_subject_id(item: Sequence[str]) -> str:
    return os.path.basename(os.path.dirname(item[1]))


def split_by_subject(all_files, train_subject_ids, val_subject_ids) -> Tuple[List, List]:
    train_ids = set(train_subject_ids)
    val_ids = set(val_subject_ids)
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f"训练集与验证集对象重复：{sorted(overlap)}")

    train_files, val_files = [], []
    for item in all_files:
        subject_id = get_subject_id(item)
        if subject_id in train_ids:
            train_files.append(item)
        elif subject_id in val_ids:
            val_files.append(item)

    print("\n========== 数据划分 ==========")
    print(f"训练对象：{sorted(train_ids)}")
    print(f"验证对象：{sorted(val_ids)}")
    print(f"训练样本数：{len(train_files)}")
    print(f"验证样本数：{len(val_files)}")
    if not train_files or not val_files:
        raise RuntimeError("训练集或验证集为空，请检查数据路径与对象ID。")
    return train_files, val_files


class IndexedDataset(Dataset):
    """给训练样本附加固定 index，供 QMF ranking loss 使用。"""
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        rgb, rfid, label, rgb_mask, rfid_mask = self.dataset[index]
        return rgb, rfid, label, rgb_mask, rfid_mask, torch.tensor(index, dtype=torch.long)


def build_loaders():
    print("正在读取并配对 RGB/RFID 数据……")
    all_files = Merge_Two_modality(PATH_VIDEO, threshold=0)
    if not all_files:
        raise RuntimeError("没有找到配对数据，请检查 PATH_VIDEO。")
    all_files = sorted(all_files, key=lambda item: (str(item[1]), str(item[0])))

    train_files, val_files = split_by_subject(
        all_files, TRAIN_SUBJECT_IDS, VAL_SUBJECT_IDS
    )

    # 公平控制：训练数据退化分布与 NRAGF 一致。
    train_base = NRAGFDataset(
        file_list=train_files,
        mode="train",
        corruption_mode=CORRUPTION_MODE,
        rgb_ratio=RGB_CORRUPTION_RATIO,
        rfid_ratio=RFID_CORRUPTION_RATIO,
        corruption_probability=CORRUPTION_PROBABILITY,
        rgb_noise_std=RGB_NOISE_STD,
        rfid_noise_std=RFID_NOISE_STD,
        modality_dropout_probability=MODALITY_DROPOUT_PROBABILITY,
        part_k_values=PART_K_VALUES,
    )
    train_dataset = IndexedDataset(train_base)

    # checkpoint 选择时保持与 NRAGF 一样的 clean validation。
    val_dataset = NRAGFDataset(
        file_list=val_files,
        mode="val",
        corruption_mode=CORRUPTION_MODE,
        rgb_ratio=0.0,
        rfid_ratio=0.0,
        corruption_probability=0.0,
        modality_dropout_probability=0.0,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)
    common = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        persistent_workers=NUM_WORKERS > 0,
    )

    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True,
        generator=generator, **common
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False, **common
    )
    return train_loader, val_loader


class History:
    """
    QMF 官方 crl_utils.History 的 RGB-RFID 适配。

    注意：官方变量名叫 correctness，但 QMF RGB-D 训练脚本实际传入
    每样本单模态 CE loss，并累计它作为排序依据。
    """
    def __init__(self, n_data: int, device: torch.device):
        self.cumulative_loss = np.zeros(n_data, dtype=np.float64)
        self.confidence = np.zeros(n_data, dtype=np.float64)
        self.device = device

    def update(self, indices, sample_loss, confidence):
        idx = indices.detach().cpu().numpy()
        self.cumulative_loss[idx] += sample_loss.detach().cpu().numpy()
        self.confidence[idx] = confidence.detach().cpu().numpy()

    def _normalize(self, values):
        data_min = float(self.cumulative_loss.min())
        data_max = float(self.cumulative_loss.max())
        denominator = data_max - data_min
        if denominator < 1e-12:
            return np.zeros_like(values, dtype=np.float64)
        return (values - data_min) / denominator

    def get_target_margin(self, indices1, indices2):
        idx1 = indices1.detach().cpu().numpy()
        idx2 = indices2.detach().cpu().numpy()
        v1 = self._normalize(self.cumulative_loss[idx1])
        v2 = self._normalize(self.cumulative_loss[idx2])

        greater = (v1 > v2).astype(np.float32)
        less = (v1 < v2).astype(np.float32) * -1.0
        target = greater + less
        margin = np.abs(v1 - v2).astype(np.float32)

        return (
            torch.from_numpy(target).to(self.device),
            torch.from_numpy(margin).to(self.device),
        )


def rank_loss(confidence, indices, history: History):
    confidence = confidence.squeeze(1)
    input1 = confidence
    input2 = torch.roll(confidence, shifts=-1, dims=0)
    indices2 = torch.roll(indices, shifts=-1, dims=0)

    target, margin = history.get_target_margin(indices, indices2)
    target_nonzero = target.clone()
    target_nonzero[target_nonzero == 0] = 1.0
    input2 = input2 + margin / target_nonzero

    return nn.MarginRankingLoss(margin=0.0)(input1, input2, -target)


def build_model():
    return QMFRGBRFID(
        graph_args={"strategy": GRAPH_STRATEGY},
        num_class=21,
        dropout=BACKBONE_DROPOUT,
        confidence_scale=QMF_CONFIDENCE_SCALE,
    )


def train_one_epoch(
    model, loader, optimizer, scaler,
    criterion_mean, criterion_none,
    rgb_history, rfid_history,
    device, epoch,
):
    model.train()
    total_loss = total_fused = total_rgb = total_rfid = total_rank = 0.0
    correct = sample_count = 0
    start_time = time.time()

    for batch_index, batch in enumerate(loader, start=1):
        rgb, rfid, labels, _, _, indices = batch
        rgb = rgb.to(device, non_blocking=True)
        rfid = rfid.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=USE_AMP and device.type == "cuda"):
            fused_logits, rgb_logits, rfid_logits, rgb_conf, rfid_conf = model(rgb, rfid)

            # QMF 原始思想：融合分类 + 两个单模态分类。
            loss_fused = criterion_mean(fused_logits, labels)
            loss_rgb = criterion_mean(rgb_logits, labels)
            loss_rfid = criterion_mean(rfid_logits, labels)

            rgb_sample_loss = criterion_none(rgb_logits, labels).detach()
            rfid_sample_loss = criterion_none(rfid_logits, labels).detach()

            loss_rank_rgb = rank_loss(rgb_conf, indices, rgb_history)
            loss_rank_rfid = rank_loss(rfid_conf, indices, rfid_history)
            loss_rank = loss_rank_rgb + loss_rank_rfid

            loss = (
                loss_fused + loss_rgb + loss_rfid
                + QMF_RANK_LOSS_WEIGHT * loss_rank
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"非有限 loss：epoch={epoch}, batch={batch_index}, loss={loss.item()}"
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        # 与官方 QMF 一致：本 batch ranking 计算后再更新历史。
        rgb_history.update(indices, rgb_sample_loss, rgb_conf.squeeze(1))
        rfid_history.update(indices, rfid_sample_loss, rfid_conf.squeeze(1))

        sample_count += labels.size(0)
        correct += (fused_logits.argmax(1) == labels).sum().item()
        total_loss += loss.item()
        total_fused += loss_fused.item()
        total_rgb += loss_rgb.item()
        total_rfid += loss_rfid.item()
        total_rank += loss_rank.item()

        if batch_index % PRINT_INTERVAL == 0 or batch_index == len(loader):
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Batch {batch_index:04d}/{len(loader):04d} | "
                f"Loss {loss.item():.4f}"
            )

    return {
        "loss": total_loss / len(loader),
        "loss_fused": total_fused / len(loader),
        "loss_rgb": total_rgb / len(loader),
        "loss_rfid": total_rfid / len(loader),
        "loss_rank": total_rank / len(loader),
        "acc": 100.0 * correct / max(sample_count, 1),
        "time": time.time() - start_time,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = sample_count = 0

    for rgb, rfid, labels, _, _ in loader:
        rgb = rgb.to(device, non_blocking=True)
        rfid = rfid.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        fused_logits, _, _, _, _ = model(rgb, rfid)
        loss = criterion(fused_logits, labels)
        total_loss += loss.item()
        correct += (fused_logits.argmax(1) == labels).sum().item()
        sample_count += labels.size(0)

    return {
        "loss": total_loss / len(loader),
        "acc": 100.0 * correct / max(sample_count, 1),
    }


def append_log(path: Path, row: dict):
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    set_seed(SEED)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "PATH_VIDEO": PATH_VIDEO,
        "TRAIN_SUBJECT_IDS": TRAIN_SUBJECT_IDS,
        "VAL_SUBJECT_IDS": VAL_SUBJECT_IDS,
        "SEED": SEED,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "MIN_LEARNING_RATE": MIN_LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "LABEL_SMOOTHING": LABEL_SMOOTHING,
        "QMF_RANK_LOSS_WEIGHT": QMF_RANK_LOSS_WEIGHT,
        "QMF_CONFIDENCE_SCALE": QMF_CONFIDENCE_SCALE,
        "CORRUPTION_MODE": CORRUPTION_MODE,
        "RGB_CORRUPTION_RATIO": RGB_CORRUPTION_RATIO,
        "RFID_CORRUPTION_RATIO": RFID_CORRUPTION_RATIO,
        "CORRUPTION_PROBABILITY": CORRUPTION_PROBABILITY,
        "RGB_NOISE_STD": RGB_NOISE_STD,
        "RFID_NOISE_STD": RFID_NOISE_STD,
        "MODALITY_DROPOUT_PROBABILITY": MODALITY_DROPOUT_PROBABILITY,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(0)}")

    train_loader, val_loader = build_loaders()
    model = build_model().to(device)
    print(f"QMF 参数量：{sum(p.numel() for p in model.parameters()):,}")

    # 为统一控制实验，公共 CE 设置与 NRAGF 对齐。
    criterion_mean = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    criterion_none = nn.CrossEntropyLoss(
        reduction="none", label_smoothing=LABEL_SMOOTHING
    )

    optimizer = Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=MIN_LEARNING_RATE
    )
    scaler = GradScaler(enabled=USE_AMP and device.type == "cuda")

    rgb_history = History(len(train_loader.dataset), device)
    rfid_history = History(len(train_loader.dataset), device)

    best_val_acc = -1.0
    best_epoch = -1
    best_path = output_dir / BEST_CHECKPOINT_NAME
    last_path = output_dir / LAST_CHECKPOINT_NAME
    log_path = output_dir / "train_log.csv"

    print("\n========== 开始 QMF 公平对比训练 ==========")
    for epoch in range(1, EPOCHS + 1):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scaler,
            criterion_mean, criterion_none,
            rgb_history, rfid_history, device, epoch,
        )
        val_stats = validate(model, val_loader, criterion_mean, device)
        scheduler.step()

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS} | "
            f"Train Acc={train_stats['acc']:.2f}% | "
            f"Val Acc={val_stats['acc']:.2f}% | "
            f"Train Loss={train_stats['loss']:.4f} | "
            f"Val Loss={val_stats['loss']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_acc": val_stats["acc"],
            "config": config,
        }
        torch.save(checkpoint, last_path)

        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            best_epoch = epoch
            torch.save(checkpoint, best_path)
            print(f"✓ 保存最佳 QMF：epoch={epoch}, val={best_val_acc:.2f}%")

        append_log(log_path, {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_fused_loss": train_stats["loss_fused"],
            "train_rgb_loss": train_stats["loss_rgb"],
            "train_rfid_loss": train_stats["loss_rfid"],
            "train_rank_loss": train_stats["loss_rank"],
            "train_acc": train_stats["acc"],
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "lr": optimizer.param_groups[0]["lr"],
        })
        print("-" * 70)

    print("\n========== 训练完成 ==========")
    print(f"Best epoch：{best_epoch}")
    print(f"Best clean validation accuracy：{best_val_acc:.2f}%")
    print(f"Best checkpoint：{best_path}")
    print(
        "\n下一步请不要重新实现 K=1~4 污染。"
        "\n应直接复用你当前 MF-Mean/LF-CMSA/NRAGF 的同一份 K 测试脚本，"
        "只增加 QMF 模型加载，以保证四种方法收到完全相同的受损样本。"
    )


if __name__ == "__main__":
    main()
