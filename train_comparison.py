# -*- coding: utf-8 -*-
"""
统一对比方法训练脚本
保存位置：NRAGF/train_comparison.py

PyCharm 使用：
1. 修改 METHOD_NAME；
2. 右键本文件；
3. 点击“运行 train_comparison”。

分别运行：
    METHOD_NAME = "MF_MEAN"
    METHOD_NAME = "LF_CMSA"
"""

# =============================================================================
# 1. 所有参数集中在这里
# =============================================================================

METHOD_NAME = "LF_CMSA"  # 可选："MF_MEAN" 或 "LF_CMSA"

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

USE_AMP = True
LABEL_SMOOTHING = 0.10
AUX_LOSS_WEIGHT = 0.50

GRAPH_STRATEGY = "spatial"
DROPOUT = 0.50

# 必须与 NRAGF 训练时保持一致
CORRUPTION_MODE = "random_nodes"
RGB_CORRUPTION_RATIO = 0.30
RFID_CORRUPTION_RATIO = 0.30
CORRUPTION_PROBABILITY = 0.50
RGB_NOISE_STD = 1.0
RFID_NOISE_STD = 1.0
MODALITY_DROPOUT_PROBABILITY = 0.15
PART_K_VALUES = (1, 2, 3)

OUTPUT_ROOT = "outputs/comparison"
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
from models.comparison_models import MFMeanNet, LFCMSANet


def normalize_method_name(name: str) -> str:
    value = name.strip().upper()
    aliases = {
        "MF-MEAN": "MF_MEAN",
        "MFMEAN": "MF_MEAN",
        "LF-CMSA": "LF_CMSA",
        "LFCMSA": "LF_CMSA",
    }
    value = aliases.get(value, value)

    if value not in ("MF_MEAN", "LF_CMSA"):
        raise ValueError(
            "METHOD_NAME 只能是 'MF_MEAN' 或 'LF_CMSA'。"
        )
    return value


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


def subject_id_from_item(item: Sequence[str]) -> str:
    return os.path.basename(os.path.dirname(item[1]))


def split_files(
    all_files: Sequence[Sequence[str]],
) -> Tuple[List, List]:
    train_ids = set(TRAIN_SUBJECT_IDS)
    val_ids = set(VAL_SUBJECT_IDS)

    if train_ids & val_ids:
        raise ValueError("训练和验证对象存在重复。")

    train_files = []
    val_files = []

    for item in all_files:
        subject_id = subject_id_from_item(item)

        if subject_id in train_ids:
            train_files.append(item)
        elif subject_id in val_ids:
            val_files.append(item)

    print("\n========== 数据划分 ==========")
    print(f"训练对象：{sorted(train_ids)}")
    print(f"验证对象：{sorted(val_ids)}")
    print(f"训练样本：{len(train_files)}")
    print(f"验证样本：{len(val_files)}")

    if not train_files or not val_files:
        raise RuntimeError(
            "训练集或验证集为空，请检查 PATH_VIDEO 和对象编号。"
        )

    return train_files, val_files


def build_model(method_name: str) -> nn.Module:
    graph_args = {"strategy": GRAPH_STRATEGY}

    common_kwargs = {
        "graph_args": graph_args,
        "num_class": 21,
        "dropout": DROPOUT,
    }

    if method_name == "MF_MEAN":
        return MFMeanNet(**common_kwargs)

    if method_name == "LF_CMSA":
        return LFCMSANet(**common_kwargs)

    raise ValueError(method_name)


def build_loaders():
    all_files = Merge_Two_modality(
        PATH_VIDEO,
        threshold=0,
    )

    if not all_files:
        raise RuntimeError(
            "没有找到 RGB/RFID 配对样本，请检查 PATH_VIDEO。"
        )

    all_files = sorted(
        all_files,
        key=lambda item: (str(item[1]), str(item[0])),
    )

    train_files, val_files = split_files(all_files)

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
    total_main = 0.0
    total_aux = 0.0
    correct = 0
    sample_count = 0
    start_time = time.time()

    for batch_index, batch in enumerate(loader, start=1):
        rgb, rfid, labels, _, _ = batch

        rgb = rgb.to(device, non_blocking=True)
        rfid = rfid.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(
            enabled=USE_AMP and device.type == "cuda"
        ):
            output, logits_rgb, logits_rfid = model(
                rgb,
                rfid,
            )

            loss_main = criterion(output, labels)
            loss_aux = (
                criterion(logits_rgb, labels)
                + criterion(logits_rfid, labels)
            )
            loss = (
                loss_main
                + AUX_LOSS_WEIGHT * loss_aux
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Loss 非有限：epoch={epoch}, "
                f"batch={batch_index}, loss={loss.item()}"
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

        correct += int(
            (output.argmax(dim=1) == labels).sum().item()
        )

        total_loss += float(loss.item()) * batch_size
        total_main += float(loss_main.item()) * batch_size
        total_aux += float(loss_aux.item()) * batch_size

        if (
            batch_index % PRINT_INTERVAL == 0
            or batch_index == len(loader)
        ):
            print(
                f"Epoch {epoch:03d}/{EPOCHS:03d} "
                f"Batch {batch_index:04d}/{len(loader):04d} "
                f"Loss {total_loss / sample_count:.4f} "
                f"Acc {100.0 * correct / sample_count:.2f}%"
            )

    denominator = max(sample_count, 1)

    return {
        "loss": total_loss / denominator,
        "main_loss": total_main / denominator,
        "aux_loss": total_aux / denominator,
        "accuracy": 100.0 * correct / denominator,
        "seconds": time.time() - start_time,
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
    sample_count = 0

    for rgb, rfid, labels, _, _ in loader:
        rgb = rgb.to(device, non_blocking=True)
        rfid = rfid.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        output, _, _ = model(rgb, rfid)

        batch_size = labels.size(0)
        sample_count += batch_size

        total_loss += (
            float(criterion(output, labels).item())
            * batch_size
        )
        correct += int(
            (output.argmax(dim=1) == labels).sum().item()
        )

    denominator = max(sample_count, 1)

    return {
        "loss": total_loss / denominator,
        "accuracy": 100.0 * correct / denominator,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    best_val_accuracy: float,
    method_name: str,
) -> None:
    torch.save(
        {
            "method": method_name,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_accuracy": best_val_accuracy,
        },
        path,
    )


def append_log(path: Path, row: Dict) -> None:
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


def main() -> None:
    method_name = normalize_method_name(METHOD_NAME)
    set_seed(SEED)

    output_dir = (
        Path(OUTPUT_ROOT) / method_name.lower()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "METHOD_NAME": method_name,
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
        "AUX_LOSS_WEIGHT": AUX_LOSS_WEIGHT,
        "CORRUPTION_MODE": CORRUPTION_MODE,
        "RGB_CORRUPTION_RATIO": RGB_CORRUPTION_RATIO,
        "RFID_CORRUPTION_RATIO": RFID_CORRUPTION_RATIO,
        "CORRUPTION_PROBABILITY": CORRUPTION_PROBABILITY,
        "MODALITY_DROPOUT_PROBABILITY": (
            MODALITY_DROPOUT_PROBABILITY
        ),
    }

    with (
        output_dir / "config.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"当前方法：{method_name}")
    print(f"运行设备：{device}")

    train_loader, val_loader = build_loaders()
    model = build_model(method_name).to(device)

    print(
        f"参数量："
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
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
        enabled=USE_AMP and device.type == "cuda"
    )

    best_val_accuracy = -1.0
    log_path = output_dir / "training_log.csv"

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        print(
            "\n"
            f"========== {method_name} Epoch {epoch:03d} ==========\n"
            f"Train loss: {train_metrics['loss']:.4f}\n"
            f"Train acc:  {train_metrics['accuracy']:.2f}%\n"
            f"Val loss:   {val_metrics['loss']:.4f}\n"
            f"Val acc:    {val_metrics['accuracy']:.2f}%\n"
            f"LR:         {current_lr:.8f}\n"
        )

        append_log(
            log_path,
            {
                "method": method_name,
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_main_loss": train_metrics["main_loss"],
                "train_aux_loss": train_metrics["aux_loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "seconds": train_metrics["seconds"],
            },
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]

            save_checkpoint(
                path=output_dir / "best.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val_accuracy=best_val_accuracy,
                method_name=method_name,
            )

            print(
                f"保存最佳 {method_name}："
                f"{best_val_accuracy:.2f}%"
            )

        save_checkpoint(
            path=output_dir / "last.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val_accuracy=best_val_accuracy,
            method_name=method_name,
        )

    print("\n训练完成。")
    print(f"方法：{method_name}")
    print(f"最佳验证准确率：{best_val_accuracy:.2f}%")
    print(f"最佳权重：{output_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
