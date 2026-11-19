# -*- coding: utf-8 -*-
"""
TMC RGB-skeleton + RFID fair-comparison training script.

PyCharm: run this file directly.

Fair-comparison principle:
- subjects 01-08 train, 09-10 validation;
- same AG-STGCN modality backbones as NRAGF;
- same 50-epoch training budget, optimizer family, LR schedule, dropout,
  random-node corruption, Gaussian noise and modality dropout as NRAGF-highK-v2;
- TMC-specific evidence / Dirichlet / Dempster-Shafer mechanism and evidential
  loss are preserved.

Important:
This is a task-adapted TMC, not the authors' original handwritten-data network.
"""

# =============================================================================
# 1. Parameters
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

# Unified optimizer settings used by the current NRAGF comparison protocol.
LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 0.00005
WEIGHT_DECAY = 5e-4
GRAD_CLIP_NORM = 5.0

USE_AMP = True

GRAPH_STRATEGY = "spatial"
DROPOUT = 0.50
NUM_CLASS = 21

# TMC-specific parameter.
# The official ICLR code uses lambda_epochs=50 by default.
TMC_ANNEALING_EPOCHS = 50
TMC_DS_EPS = 1e-8

# Same robust-training distribution as NRAGF-highK-v2.
CORRUPTION_MODE = "random_nodes"
RGB_CORRUPTION_RATIO = 0.30
RFID_CORRUPTION_RATIO = 0.30
CORRUPTION_PROBABILITY = 0.50
RGB_NOISE_STD = 1.0
RFID_NOISE_STD = 1.0
MODALITY_DROPOUT_PROBABILITY = 0.15
PART_K_VALUES = (1, 2, 3)

OUTPUT_DIR = "outputs/tmc_fair_seed2026"
BEST_CHECKPOINT_NAME = "best_tmc_fair.pth"
LAST_CHECKPOINT_NAME = "last_tmc_fair.pth"
PRINT_INTERVAL = 20

# NOTE:
# Original TMC ICLR demo uses roughly:
#   epochs=500, lr=3e-4, weight_decay=1e-5, lambda_epochs=50.
# We intentionally DO NOT use that optimizer/training budget in the primary
# comparison because the current paper compares methods under a unified budget.


# =============================================================================
# 2. Imports
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
from models.tmc_rgb_rfid import (
    TMCRGBRFID,
    tmc_total_loss,
)


# =============================================================================
# 3. Reproducibility
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
# 4. Data split
# =============================================================================

def get_subject_id(item: Sequence[str]) -> str:
    return os.path.basename(
        os.path.dirname(item[1])
    )


def split_by_subject(
    all_files: Sequence[Sequence[str]],
    train_subject_ids: Sequence[str],
    val_subject_ids: Sequence[str],
) -> Tuple[List, List]:
    train_ids = set(train_subject_ids)
    val_ids = set(val_subject_ids)

    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(
            f"训练与验证对象重复：{sorted(overlap)}"
        )

    train_files = []
    val_files = []

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
        raise RuntimeError(
            "训练集或验证集为空，请检查数据路径和对象ID。"
        )

    return train_files, val_files


def build_loaders():
    print("正在读取并配对 RGB/RFID 数据……")

    all_files = Merge_Two_modality(
        PATH_VIDEO,
        threshold=0,
    )

    if not all_files:
        raise RuntimeError(
            "没有找到配对数据，请检查 PATH_VIDEO。"
        )

    all_files = sorted(
        all_files,
        key=lambda item: (
            str(item[1]),
            str(item[0]),
        ),
    )

    train_files, val_files = split_by_subject(
        all_files,
        TRAIN_SUBJECT_IDS,
        VAL_SUBJECT_IDS,
    )

    train_dataset = NRAGFDataset(
        file_list=train_files,
        mode="train",
        corruption_mode=CORRUPTION_MODE,
        rgb_ratio=RGB_CORRUPTION_RATIO,
        rfid_ratio=RFID_CORRUPTION_RATIO,
        corruption_probability=CORRUPTION_PROBABILITY,
        rgb_noise_std=RGB_NOISE_STD,
        rfid_noise_std=RFID_NOISE_STD,
        modality_dropout_probability=(
            MODALITY_DROPOUT_PROBABILITY
        ),
        part_k_values=PART_K_VALUES,
    )

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
# 5. Model
# =============================================================================

def build_model() -> TMCRGBRFID:
    return TMCRGBRFID(
        graph_args={
            "strategy": GRAPH_STRATEGY,
        },
        num_class=NUM_CLASS,
        dropout=DROPOUT,
        ds_eps=TMC_DS_EPS,
    )


# =============================================================================
# 6. Train / validation
# =============================================================================

def train_one_epoch(
    model: TMCRGBRFID,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_rgb_loss = 0.0
    total_rfid_loss = 0.0
    total_fused_loss = 0.0

    correct = 0
    sample_count = 0
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
            enabled=USE_AMP
            and device.type == "cuda"
        ):
            outputs = model(rgb, rfid)

            (
                evidence_fused,
                evidence_rgb,
                evidence_rfid,
                alpha_fused,
                alpha_rgb,
                alpha_rfid,
                _,
                _,
                _,
            ) = outputs

        # digamma/lgamma are more stable in FP32.
        loss, loss_details = tmc_total_loss(
            labels=labels,
            alpha_rgb=alpha_rgb.float(),
            alpha_rfid=alpha_rfid.float(),
            alpha_fused=alpha_fused.float(),
            num_class=NUM_CLASS,
            global_step=epoch,
            annealing_step=TMC_ANNEALING_EPOCHS,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"TMC loss 非有限："
                f"epoch={epoch}, "
                f"batch={batch_index}, "
                f"loss={loss.item()}"
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP_NORM,
        )

        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        sample_count += batch_size

        correct += (
            evidence_fused.argmax(dim=1)
            == labels
        ).sum().item()

        total_loss += float(loss.item())
        total_rgb_loss += loss_details["loss_rgb"]
        total_rfid_loss += loss_details["loss_rfid"]
        total_fused_loss += loss_details["loss_fused"]

        if (
            batch_index % PRINT_INTERVAL == 0
            or batch_index == len(loader)
        ):
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Batch {batch_index:04d}/{len(loader):04d} | "
                f"Loss {loss.item():.4f}"
            )

    elapsed = time.time() - start_time

    return {
        "loss": total_loss / len(loader),
        "loss_rgb": total_rgb_loss / len(loader),
        "loss_rfid": total_rfid_loss / len(loader),
        "loss_fused": total_fused_loss / len(loader),
        "acc": (
            100.0 * correct
            / max(sample_count, 1)
        ),
        "time": elapsed,
    }


@torch.no_grad()
def validate(
    model: TMCRGBRFID,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    sample_count = 0

    total_u_rgb = 0.0
    total_u_rfid = 0.0
    total_u_fused = 0.0

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

        outputs = model(rgb, rfid)

        (
            evidence_fused,
            _,
            _,
            alpha_fused,
            alpha_rgb,
            alpha_rfid,
            uncertainty_fused,
            uncertainty_rgb,
            uncertainty_rfid,
        ) = outputs

        loss, _ = tmc_total_loss(
            labels=labels,
            alpha_rgb=alpha_rgb.float(),
            alpha_rfid=alpha_rfid.float(),
            alpha_fused=alpha_fused.float(),
            num_class=NUM_CLASS,
            global_step=epoch,
            annealing_step=TMC_ANNEALING_EPOCHS,
        )

        batch_size = labels.size(0)
        sample_count += batch_size

        correct += (
            evidence_fused.argmax(dim=1)
            == labels
        ).sum().item()

        total_loss += float(loss.item())

        total_u_rgb += float(
            uncertainty_rgb.sum().item()
        )
        total_u_rfid += float(
            uncertainty_rfid.sum().item()
        )
        total_u_fused += float(
            uncertainty_fused.sum().item()
        )

    return {
        "loss": total_loss / len(loader),
        "acc": (
            100.0 * correct
            / max(sample_count, 1)
        ),
        "u_rgb": (
            total_u_rgb
            / max(sample_count, 1)
        ),
        "u_rfid": (
            total_u_rfid
            / max(sample_count, 1)
        ),
        "u_fused": (
            total_u_fused
            / max(sample_count, 1)
        ),
    }


# =============================================================================
# 7. Save / log
# =============================================================================

def append_log(
    path: Path,
    row: Dict,
) -> None:
    file_exists = path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys()),
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# =============================================================================
# 8. Main
# =============================================================================

def main() -> None:
    set_seed(SEED)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "PATH_VIDEO": PATH_VIDEO,
        "TRAIN_SUBJECT_IDS": TRAIN_SUBJECT_IDS,
        "VAL_SUBJECT_IDS": VAL_SUBJECT_IDS,
        "SEED": SEED,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "EPOCHS": EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "MIN_LEARNING_RATE": MIN_LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "DROPOUT": DROPOUT,
        "TMC_ANNEALING_EPOCHS": TMC_ANNEALING_EPOCHS,
        "CORRUPTION_MODE": CORRUPTION_MODE,
        "RGB_CORRUPTION_RATIO": RGB_CORRUPTION_RATIO,
        "RFID_CORRUPTION_RATIO": RFID_CORRUPTION_RATIO,
        "CORRUPTION_PROBABILITY": CORRUPTION_PROBABILITY,
        "RGB_NOISE_STD": RGB_NOISE_STD,
        "RFID_NOISE_STD": RFID_NOISE_STD,
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

    train_loader, val_loader = build_loaders()

    model = build_model().to(device)

    print(
        "TMC 参数量："
        f"{sum(p.numel() for p in model.parameters()):,}"
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
        enabled=USE_AMP
        and device.type == "cuda"
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
        "\n========== 开始 TMC 公平对比训练 =========="
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
            device=device,
            epoch=epoch,
        )

        val_stats = validate(
            model=model,
            loader=val_loader,
            device=device,
            epoch=epoch,
        )

        scheduler.step()

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS} | "
            f"Train Acc={train_stats['acc']:.2f}% | "
            f"Val Acc={val_stats['acc']:.2f}% | "
            f"Train Loss={train_stats['loss']:.4f} | "
            f"Val Loss={val_stats['loss']:.4f}"
        )
        print(
            f"Val uncertainty | "
            f"RGB={val_stats['u_rgb']:.4f} | "
            f"RFID={val_stats['u_rfid']:.4f} | "
            f"Fused={val_stats['u_fused']:.4f}"
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
                f"✓ 保存最佳 TMC："
                f"epoch={epoch}, "
                f"val={best_val_acc:.2f}%"
            )

        append_log(
            log_path,
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_rgb_loss": train_stats["loss_rgb"],
                "train_rfid_loss": train_stats["loss_rfid"],
                "train_fused_loss": train_stats["loss_fused"],
                "train_acc": train_stats["acc"],
                "val_loss": val_stats["loss"],
                "val_acc": val_stats["acc"],
                "val_u_rgb": val_stats["u_rgb"],
                "val_u_rfid": val_stats["u_rfid"],
                "val_u_fused": val_stats["u_fused"],
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

        print("-" * 70)

    print("\n========== TMC 训练完成 ==========")
    print(f"Best epoch：{best_epoch}")
    print(
        f"Best clean validation accuracy："
        f"{best_val_acc:.2f}%"
    )
    print(
        f"Best checkpoint：{best_path}"
    )


if __name__ == "__main__":
    main()
