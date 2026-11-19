# -*- coding: utf-8 -*-
"""
Base：最基础双模态均值融合基线（仅干净样本训练）

目的：
- 与 MF-Mean 使用相同的模型结构；
- 与其他方法使用相同的数据划分、训练轮数、优化器和主要超参数；
- 唯一关键区别：训练阶段不加入节点退化、不加入高斯污染、不加入模态 dropout；
- 用于观察“完全没有鲁棒训练/可靠性机制”的普通融合模型，在 K=1~4 局部污染下的自然性能下降。

PyCharm 直接运行。
"""

# =============================================================================
# 1. 参数
# =============================================================================

PATH_VIDEO = "/home/a2/Datasets/多模态数据集/Video"

TRAIN_SUBJECT_IDS = (
    "01", "02", "03", "04",
    "05", "06", "07", "08",
)
VAL_SUBJECT_IDS = ("09", "10")

SEED = 2026

BATCH_SIZE = 64
NUM_WORKERS = 4
EPOCHS = 50

LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 0.00005
WEIGHT_DECAY = 5e-4
GRAD_CLIP_NORM = 5.0

LABEL_SMOOTHING = 0.10
USE_AMP = True

GRAPH_STRATEGY = "spatial"
DROPOUT = 0.50
NUM_CLASS = 21

# 关键：Base 完全关闭鲁棒训练。
RGB_CORRUPTION_RATIO = 0.0
RFID_CORRUPTION_RATIO = 0.0
CORRUPTION_PROBABILITY = 0.0
MODALITY_DROPOUT_PROBABILITY = 0.0

OUTPUT_DIR = "outputs/comparison/base_clean"
BEST_CHECKPOINT_NAME = "best.pth"
LAST_CHECKPOINT_NAME = "last.pth"
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
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.dataset2 import Merge_Two_modality
from data.dataset_nragf import NRAGFDataset
from models.comparison_models import MFMeanNet


# =============================================================================
# 3. 随机性
# =============================================================================

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


# =============================================================================
# 4. 数据
# =============================================================================

def subject_id_from_item(
    item: Sequence[str],
) -> str:
    return os.path.basename(
        os.path.dirname(item[1])
    )


def split_by_subject(
    all_files: Sequence[Sequence[str]],
) -> Tuple[List, List]:

    train_ids = set(TRAIN_SUBJECT_IDS)
    val_ids = set(VAL_SUBJECT_IDS)

    if train_ids & val_ids:
        raise ValueError(
            "训练对象与验证对象存在重复。"
        )

    train_files = []
    val_files = []

    for item in all_files:
        sid = subject_id_from_item(item)

        if sid in train_ids:
            train_files.append(item)
        elif sid in val_ids:
            val_files.append(item)

    print("\n========== 数据划分 ==========")
    print(f"训练对象：{sorted(train_ids)}")
    print(f"验证对象：{sorted(val_ids)}")
    print(f"训练样本数：{len(train_files)}")
    print(f"验证样本数：{len(val_files)}")

    if not train_files or not val_files:
        raise RuntimeError(
            "训练集或验证集为空，请检查数据路径。"
        )

    return train_files, val_files


def build_loaders():

    print("正在读取并配对 RGB / RFID 数据……")

    all_files = Merge_Two_modality(
        PATH_VIDEO,
        threshold=0,
    )

    if not all_files:
        raise RuntimeError(
            "未找到 RGB/RFID 配对数据。"
        )

    all_files = sorted(
        all_files,
        key=lambda item: (
            str(item[1]),
            str(item[0]),
        ),
    )

    train_files, val_files = split_by_subject(
        all_files
    )

    # ---------------------------------------------------------
    # Base 的关键：
    # mode 仍然使用 train，但所有退化概率/比例全部为 0。
    # 因此训练时读取的是完整 RGB + 完整 RFID。
    # ---------------------------------------------------------
    train_dataset = NRAGFDataset(
        file_list=train_files,
        mode="train",
        corruption_mode="random_nodes",
        rgb_ratio=RGB_CORRUPTION_RATIO,
        rfid_ratio=RFID_CORRUPTION_RATIO,
        corruption_probability=CORRUPTION_PROBABILITY,
        rgb_noise_std=1.0,
        rfid_noise_std=1.0,
        modality_dropout_probability=(
            MODALITY_DROPOUT_PROBABILITY
        ),
    )

    val_dataset = NRAGFDataset(
        file_list=val_files,
        mode="val",
        corruption_mode="random_nodes",
        rgb_ratio=0.0,
        rfid_ratio=0.0,
        corruption_probability=0.0,
        modality_dropout_probability=0.0,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    common_args = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "persistent_workers": NUM_WORKERS > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common_args,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common_args,
    )

    return train_loader, val_loader


# =============================================================================
# 5. 模型
# =============================================================================

def build_model() -> nn.Module:
    """
    直接复用当前 MF-Mean 的模型结构。
    因此 Base 与 MF-Mean 的差异主要来自：
        Base：clean-only training
        MF-Mean：robust corruption training
    """
    return MFMeanNet(
        graph_args={
            "strategy": GRAPH_STRATEGY,
        },
        num_class=NUM_CLASS,
        dropout=DROPOUT,
    )


def get_logits(
    model_output,
) -> torch.Tensor:
    """
    与统一评估脚本保持一致：
    comparison_models 中的模型第一项为融合分类输出。
    同时兼容直接返回 Tensor 的情况。
    """
    if isinstance(
        model_output,
        (tuple, list),
    ):
        return model_output[0]

    return model_output


# =============================================================================
# 6. 训练 / 验证
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):
        rgb, rfid, labels, _, _ = batch

        rgb = rgb.to(
            device,
            non_blocking=True,
        )
        rfid = rfid.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(
            enabled=(
                USE_AMP
                and device.type == "cuda"
            )
        ):
            model_output = model(
                rgb,
                rfid,
            )

            logits = get_logits(
                model_output
            )

            loss = criterion(
                logits,
                labels,
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Loss 非有限："
                f"epoch={epoch}, "
                f"batch={batch_index}"
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP_NORM,
        )

        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)

        total += batch_size
        correct += (
            logits.argmax(dim=1)
            == labels
        ).sum().item()

        total_loss += float(
            loss.item()
        )

        if (
            batch_index % PRINT_INTERVAL == 0
            or batch_index == len(loader)
        ):
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Batch {batch_index:04d}/{len(loader):04d} | "
                f"Loss={loss.item():.4f}"
            )

    return {
        "loss": (
            total_loss
            / max(len(loader), 1)
        ),
        "acc": (
            100.0
            * correct
            / max(total, 1)
        ),
        "time": (
            time.time()
            - start_time
        ),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        rgb, rfid, labels, _, _ = batch

        rgb = rgb.to(
            device,
            non_blocking=True,
        )
        rfid = rfid.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        model_output = model(
            rgb,
            rfid,
        )

        logits = get_logits(
            model_output
        )

        loss = criterion(
            logits,
            labels,
        )

        total_loss += float(
            loss.item()
        )

        total += labels.size(0)

        correct += (
            logits.argmax(dim=1)
            == labels
        ).sum().item()

    return {
        "loss": (
            total_loss
            / max(len(loader), 1)
        ),
        "acc": (
            100.0
            * correct
            / max(total, 1)
        ),
    }


def append_log(
    path: Path,
    row: Dict,
) -> None:

    exists = path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys()),
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


# =============================================================================
# 7. 主函数
# =============================================================================

def main() -> None:

    set_seed(SEED)

    output_dir = Path(
        OUTPUT_DIR
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "method": "Base",
        "training_type": "clean_only",
        "PATH_VIDEO": PATH_VIDEO,
        "TRAIN_SUBJECT_IDS": TRAIN_SUBJECT_IDS,
        "VAL_SUBJECT_IDS": VAL_SUBJECT_IDS,
        "SEED": SEED,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "MIN_LEARNING_RATE": MIN_LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "DROPOUT": DROPOUT,
        "LABEL_SMOOTHING": LABEL_SMOOTHING,
        "RGB_CORRUPTION_RATIO": (
            RGB_CORRUPTION_RATIO
        ),
        "RFID_CORRUPTION_RATIO": (
            RFID_CORRUPTION_RATIO
        ),
        "CORRUPTION_PROBABILITY": (
            CORRUPTION_PROBABILITY
        ),
        "MODALITY_DROPOUT_PROBABILITY": (
            MODALITY_DROPOUT_PROBABILITY
        ),
    }

    with (
        output_dir / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"运行设备：{device}")

    if device.type == "cuda":
        print(
            f"GPU："
            f"{torch.cuda.get_device_name(0)}"
        )

    train_loader, val_loader = (
        build_loaders()
    )

    model = build_model().to(
        device
    )

    print(
        f"Base 参数量："
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING,
    )

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=MIN_LEARNING_RATE,
    )

    scaler = GradScaler(
        enabled=(
            USE_AMP
            and device.type == "cuda"
        )
    )

    best_val_acc = -1.0
    best_epoch = -1

    best_path = (
        output_dir
        / BEST_CHECKPOINT_NAME
    )
    last_path = (
        output_dir
        / LAST_CHECKPOINT_NAME
    )
    log_path = (
        output_dir
        / "train_log.csv"
    )

    print(
        "\n========== "
        "开始 Base（clean-only）训练 "
        "=========="
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        val_stats = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

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

        torch.save(
            checkpoint,
            last_path,
        )

        if (
            val_stats["acc"]
            > best_val_acc
        ):
            best_val_acc = (
                val_stats["acc"]
            )
            best_epoch = epoch

            torch.save(
                checkpoint,
                best_path,
            )

            print(
                f"✓ 保存最佳 Base："
                f"epoch={epoch}, "
                f"val={best_val_acc:.2f}%"
            )

        append_log(
            log_path,
            {
                "epoch": epoch,
                "train_loss": (
                    train_stats["loss"]
                ),
                "train_acc": (
                    train_stats["acc"]
                ),
                "val_loss": (
                    val_stats["loss"]
                ),
                "val_acc": (
                    val_stats["acc"]
                ),
                "lr": (
                    optimizer
                    .param_groups[0]["lr"]
                ),
            },
        )

        print("-" * 70)

    print(
        "\n========== "
        "Base 训练完成 "
        "=========="
    )

    print(
        f"Best epoch：{best_epoch}"
    )
    print(
        f"Best clean validation accuracy："
        f"{best_val_acc:.2f}%"
    )
    print(
        f"Best checkpoint：{best_path}"
    )


if __name__ == "__main__":
    main()
