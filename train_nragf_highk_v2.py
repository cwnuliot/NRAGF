# -*- coding: utf-8 -*-
"""
NRAGF-v2 高 K 鲁棒训练脚本

保存位置：
    NRAGF/train_nragf_optimized.py

PyCharm 直接运行本文件即可。

重要原则：
1. 原 baseline.py 不修改；
2. ID 01-08 为训练集；
3. ID 09-10 为验证集；
4. 每轮验证并保存 best checkpoint；
5. 不使用 Generator；
6. 所有主要参数集中在文件最前面。
"""

# =============================================================================
# 1. 统一参数配置：平时主要修改这里
# =============================================================================

# ------------------------- 数据路径 -------------------------
PATH_VIDEO = "/home/a2/Datasets/多模态数据集/Video"

# ------------------------- 对象划分 -------------------------
TRAIN_SUBJECT_IDS = (
    "01", "02", "03", "04",
    "05", "06", "07", "08",
)
VAL_SUBJECT_IDS = ("09", "10")

# ------------------------- 训练参数 -------------------------
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

# ------------------------- 模型参数 -------------------------
GRAPH_STRATEGY = "spatial"
DROPOUT = 0.50

TCP_HIDDEN_DIM = 128
TCP_DROPOUT = 0.20

# 可靠性归一化节点聚合参数
POOL_WEIGHT_FLOOR = 0.05
POOL_WEIGHT_POWER = 2.00

# 模态可靠性先验
QUALITY_PRIOR_STRENGTH = 1.00

# 单模态质量失衡校正强度
ASYMMETRIC_FUSION_STRENGTH = 1.00

# 双模态共同退化校正强度；提高 K=3/4 时主要调这个参数
JOINT_DEGRADATION_STRENGTH = 1.00

RELIABILITY_WARMUP_EPOCHS = 5

# ------------------------- 损失权重 -------------------------
AUX_LOSS_WEIGHT = 0.50
TCP_LOSS_WEIGHT = 2.00
CORRUPTED_NODE_WEIGHT = 2.00

# ------------------------- 动态干扰 -------------------------
# random_nodes：与原训练最接近
# five_parts：按五个人体部位干扰
CORRUPTION_MODE = "random_nodes"

RGB_CORRUPTION_RATIO = 0.30
RFID_CORRUPTION_RATIO = 0.30
CORRUPTION_PROBABILITY = 0.50

RGB_NOISE_STD = 1.0
RFID_NOISE_STD = 1.0

MODALITY_DROPOUT_PROBABILITY = 0.15
PART_K_VALUES = (1, 2, 3)

# ------------------------- 保存与恢复 -------------------------
OUTPUT_DIR = "outputs/nragf_highk_v2_seed2026"
BEST_CHECKPOINT_NAME = "best_nragf_highk_v2.pth"
LAST_CHECKPOINT_NAME = "last_nragf_highk_v2.pth"

# 全新训练保持 None
RESUME_CHECKPOINT = None

PRINT_INTERVAL = 20


# =============================================================================
# 2. 导入库
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
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.dataset2 import Merge_Two_modality
from data.dataset_nragf import NRAGFDataset
from models.nragf_highk_v2 import NRAGFHighKNet


# =============================================================================
# 3. 随机种子
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
# 4. 数据划分
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
            f"训练集和验证集对象重复：{sorted(overlap)}"
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
    print(f"训练样本：{len(train_files)}")
    print(f"验证样本：{len(val_files)}")

    if not train_files:
        raise RuntimeError(
            "训练集为空，请检查 PATH_VIDEO 和对象目录名称。"
        )

    if not val_files:
        raise RuntimeError(
            "验证集为空，请检查 VAL_SUBJECT_IDS。"
        )

    return train_files, val_files


# =============================================================================
# 5. TCP 监督目标
# =============================================================================

def sample_true_class_probability(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    样本级 TCP，并保留原训练中的“预测错误减半”策略。
    """
    probabilities = F.softmax(
        logits,
        dim=1,
    )

    target = probabilities.gather(
        dim=1,
        index=labels.view(-1, 1),
    )

    predictions = logits.argmax(dim=1)
    incorrect = predictions != labels

    target = target.clone()
    target[incorrect] *= 0.5

    return target.detach()


def node_tcp_target(
    sample_tcp: torch.Tensor,
    clean_mask: torch.Tensor,
) -> torch.Tensor:
    """
    样本级 TCP × 节点 clean mask，形成节点级监督。
    """
    return (
        sample_tcp.unsqueeze(2)
        * clean_mask.unsqueeze(1)
    )


def weighted_tcp_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    clean_mask: torch.Tensor,
) -> torch.Tensor:
    """
    已知受损节点获得更高损失权重。
    """
    corrupted_mask = (
        1.0 - clean_mask.unsqueeze(1)
    )

    weight = (
        torch.ones_like(prediction)
        + CORRUPTED_NODE_WEIGHT * corrupted_mask
    )

    squared_error = (
        prediction - target
    ).pow(2)

    return (
        weight * squared_error
    ).sum() / weight.sum().clamp_min(1.0)


# =============================================================================
# 6. 模型与 DataLoader
# =============================================================================

def build_model() -> NRAGFHighKNet:
    graph_args = {
        "strategy": GRAPH_STRATEGY,
    }

    return NRAGFHighKNet(
        graph_args=graph_args,
        num_class=21,
        dropout=DROPOUT,
        tcp_hidden_dim=TCP_HIDDEN_DIM,
        tcp_dropout=TCP_DROPOUT,
        pool_weight_floor=POOL_WEIGHT_FLOOR,
        pool_weight_power=POOL_WEIGHT_POWER,
        quality_prior_strength=QUALITY_PRIOR_STRENGTH,
        asymmetric_fusion_strength=(
            ASYMMETRIC_FUSION_STRENGTH
        ),
        joint_degradation_strength=(
            JOINT_DEGRADATION_STRENGTH
        ),
    )


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

    common_arguments = {
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
        **common_arguments,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common_arguments,
    )

    return train_loader, val_loader


# =============================================================================
# 7. 训练与验证
# =============================================================================

def reliability_scale_for_epoch(epoch: int) -> float:
    if RELIABILITY_WARMUP_EPOCHS <= 1:
        return 1.0

    progress = (
        epoch - 1
    ) / float(
        RELIABILITY_WARMUP_EPOCHS - 1
    )

    return float(
        max(0.0, min(1.0, progress))
    )


def train_one_epoch(
    model: NRAGFHighKNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    criterion_cls: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    model.train()

    reliability_scale = reliability_scale_for_epoch(
        epoch
    )

    total_loss = 0.0
    total_main = 0.0
    total_aux = 0.0
    total_tcp = 0.0

    correct = 0
    sample_count = 0
    start_time = time.time()

    for batch_index, batch in enumerate(
        loader,
        start=1,
    ):
        (
            rgb,
            rfid,
            labels,
            rgb_clean_mask,
            rfid_clean_mask,
        ) = batch

        rgb = rgb.to(device, non_blocking=True)
        rfid = rfid.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        rgb_clean_mask = rgb_clean_mask.to(
            device,
            non_blocking=True,
        )
        rfid_clean_mask = rfid_clean_mask.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with autocast(
            enabled=USE_AMP and device.type == "cuda"
        ):
            outputs = model(
                rgb,
                rfid,
                reliability_scale=reliability_scale,
                use_tcp=True,
                use_quality_fusion=True,
                return_details=True,
            )

            (
                logits,
                conf_rgb,
                conf_rfid,
                logits_rgb,
                logits_rfid,
                details,
            ) = outputs

            loss_main = criterion_cls(
                logits,
                labels,
            )

            loss_aux = (
                criterion_cls(logits_rgb, labels)
                + criterion_cls(logits_rfid, labels)
            )

            with torch.no_grad():
                sample_tcp_rgb = (
                    sample_true_class_probability(
                        logits_rgb,
                        labels,
                    )
                )
                sample_tcp_rfid = (
                    sample_true_class_probability(
                        logits_rfid,
                        labels,
                    )
                )

                target_rgb = node_tcp_target(
                    sample_tcp_rgb,
                    rgb_clean_mask,
                )
                target_rfid = node_tcp_target(
                    sample_tcp_rfid,
                    rfid_clean_mask,
                )

            loss_tcp = (
                weighted_tcp_mse(
                    conf_rgb,
                    target_rgb,
                    rgb_clean_mask,
                )
                + weighted_tcp_mse(
                    conf_rfid,
                    target_rfid,
                    rfid_clean_mask,
                )
            )

            loss = (
                loss_main
                + AUX_LOSS_WEIGHT * loss_aux
                + TCP_LOSS_WEIGHT * loss_tcp
            )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Loss 非有限：epoch={epoch}, "
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

        correct += int(
            (
                logits.argmax(dim=1)
                == labels
            ).sum().item()
        )

        total_loss += float(loss.item()) * batch_size
        total_main += float(loss_main.item()) * batch_size
        total_aux += float(loss_aux.item()) * batch_size
        total_tcp += float(loss_tcp.item()) * batch_size

        if (
            batch_index % PRINT_INTERVAL == 0
            or batch_index == len(loader)
        ):
            print(
                f"Epoch {epoch:03d}/{EPOCHS:03d} "
                f"Batch {batch_index:04d}/{len(loader):04d} "
                f"Loss {total_loss / sample_count:.4f} "
                f"Acc {100.0 * correct / sample_count:.2f}% "
                f"Reliability {reliability_scale:.2f}"
            )

    denominator = max(sample_count, 1)

    return {
        "loss": total_loss / denominator,
        "main_loss": total_main / denominator,
        "aux_loss": total_aux / denominator,
        "tcp_loss": total_tcp / denominator,
        "accuracy": 100.0 * correct / denominator,
        "seconds": time.time() - start_time,
        "reliability_scale": reliability_scale,
    }


@torch.no_grad()
def validate(
    model: NRAGFHighKNet,
    loader: DataLoader,
    criterion_cls: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    correct = 0
    sample_count = 0

    rgb_confidence_sum = 0.0
    rfid_confidence_sum = 0.0

    rgb_fusion_weight_sum = 0.0
    rfid_fusion_weight_sum = 0.0
    asymmetric_trigger_sum = 0.0
    joint_trigger_sum = 0.0

    for batch in loader:
        rgb, rfid, labels, _, _ = batch

        rgb = rgb.to(device, non_blocking=True)
        rfid = rfid.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(
            rgb,
            rfid,
            reliability_scale=1.0,
            use_tcp=True,
            use_quality_fusion=True,
            return_details=True,
        )

        (
            logits,
            conf_rgb,
            conf_rfid,
            _,
            _,
            details,
        ) = outputs

        batch_size = labels.size(0)
        sample_count += batch_size

        loss = criterion_cls(
            logits,
            labels,
        )

        total_loss += float(
            loss.item()
        ) * batch_size

        correct += int(
            (
                logits.argmax(dim=1)
                == labels
            ).sum().item()
        )

        rgb_confidence_sum += float(
            conf_rgb.mean().item()
        ) * batch_size

        rfid_confidence_sum += float(
            conf_rfid.mean().item()
        ) * batch_size

        fusion_weights = details[
            "fusion_weights"
        ]

        rgb_fusion_weight_sum += float(
            fusion_weights[:, 0].mean().item()
        ) * batch_size

        rfid_fusion_weight_sum += float(
            fusion_weights[:, 1].mean().item()
        ) * batch_size

        asymmetric_trigger_sum += float(
            details["asymmetric_trigger"].mean().item()
        ) * batch_size

        joint_trigger_sum += float(
            details["joint_trigger"].mean().item()
        ) * batch_size

    denominator = max(sample_count, 1)

    return {
        "loss": total_loss / denominator,
        "accuracy": 100.0 * correct / denominator,
        "rgb_confidence": (
            rgb_confidence_sum / denominator
        ),
        "rfid_confidence": (
            rfid_confidence_sum / denominator
        ),
        "rgb_fusion_weight": (
            rgb_fusion_weight_sum / denominator
        ),
        "rfid_fusion_weight": (
            rfid_fusion_weight_sum / denominator
        ),
        "asymmetric_trigger": (
            asymmetric_trigger_sum / denominator
        ),
        "joint_trigger": (
            joint_trigger_sum / denominator
        ),
    }


# =============================================================================
# 8. 保存、恢复与日志
# =============================================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    best_val_accuracy: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_accuracy": best_val_accuracy,
        },
        path,
    )


def resume_if_needed(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[int, float]:
    if RESUME_CHECKPOINT is None:
        return 1, -1.0

    checkpoint_path = Path(
        RESUME_CHECKPOINT
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"找不到断点：{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model"]
    )
    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )
    scheduler.load_state_dict(
        checkpoint["scheduler"]
    )

    if "scaler" in checkpoint:
        scaler.load_state_dict(
            checkpoint["scaler"]
        )

    start_epoch = int(
        checkpoint.get("epoch", 0)
    ) + 1

    best_val_accuracy = float(
        checkpoint.get(
            "best_val_accuracy",
            -1.0,
        )
    )

    print(
        f"从第 {start_epoch} 轮继续训练，"
        f"历史最佳验证准确率："
        f"{best_val_accuracy:.2f}%"
    )

    return start_epoch, best_val_accuracy


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
# 9. 主函数：PyCharm 直接运行
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
        "LABEL_SMOOTHING": LABEL_SMOOTHING,
        "POOL_WEIGHT_FLOOR": POOL_WEIGHT_FLOOR,
        "POOL_WEIGHT_POWER": POOL_WEIGHT_POWER,
        "QUALITY_PRIOR_STRENGTH": QUALITY_PRIOR_STRENGTH,
        "ASYMMETRIC_FUSION_STRENGTH": (
            ASYMMETRIC_FUSION_STRENGTH
        ),
        "JOINT_DEGRADATION_STRENGTH": (
            JOINT_DEGRADATION_STRENGTH
        ),
        "AUX_LOSS_WEIGHT": AUX_LOSS_WEIGHT,
        "TCP_LOSS_WEIGHT": TCP_LOSS_WEIGHT,
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

    model = build_model().to(device)

    print(
        f"模型参数量："
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    criterion_cls = nn.CrossEntropyLoss(
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
        enabled=USE_AMP
        and device.type == "cuda"
    )

    start_epoch, best_val_accuracy = (
        resume_if_needed(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
    )

    log_path = (
        output_dir / "training_log.csv"
    )

    for epoch in range(
        start_epoch,
        EPOCHS + 1,
    ):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion_cls=criterion_cls,
            device=device,
            epoch=epoch,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion_cls=criterion_cls,
            device=device,
        )

        current_lr = optimizer.param_groups[0][
            "lr"
        ]
        scheduler.step()

        print(
            "\n"
            f"========== Epoch {epoch:03d} ==========\n"
            f"Train loss: {train_metrics['loss']:.4f}\n"
            f"Train acc:  {train_metrics['accuracy']:.2f}%\n"
            f"TCP loss:   {train_metrics['tcp_loss']:.4f}\n"
            f"Val loss:   {val_metrics['loss']:.4f}\n"
            f"Val acc:    {val_metrics['accuracy']:.2f}%\n"
            f"RGB conf:   {val_metrics['rgb_confidence']:.4f}\n"
            f"RFID conf:  {val_metrics['rfid_confidence']:.4f}\n"
            f"RGB weight: {val_metrics['rgb_fusion_weight']:.4f}\n"
            f"RFID weight:{val_metrics['rfid_fusion_weight']:.4f}\n"
            f"Asym trig:  {val_metrics['asymmetric_trigger']:.4f}\n"
            f"Joint trig: {val_metrics['joint_trigger']:.4f}\n"
            f"LR:         {current_lr:.8f}\n"
        )

        append_log(
            log_path,
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_main_loss": train_metrics[
                    "main_loss"
                ],
                "train_aux_loss": train_metrics[
                    "aux_loss"
                ],
                "train_tcp_loss": train_metrics[
                    "tcp_loss"
                ],
                "train_accuracy": train_metrics[
                    "accuracy"
                ],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics[
                    "accuracy"
                ],
                "val_rgb_confidence": val_metrics[
                    "rgb_confidence"
                ],
                "val_rfid_confidence": val_metrics[
                    "rfid_confidence"
                ],
                "val_rgb_fusion_weight": val_metrics[
                    "rgb_fusion_weight"
                ],
                "val_rfid_fusion_weight": val_metrics[
                    "rfid_fusion_weight"
                ],
                "val_asymmetric_trigger": val_metrics[
                    "asymmetric_trigger"
                ],
                "val_joint_trigger": val_metrics[
                    "joint_trigger"
                ],
                "reliability_scale": train_metrics[
                    "reliability_scale"
                ],
                "seconds": train_metrics["seconds"],
            },
        )

        if (
            val_metrics["accuracy"]
            > best_val_accuracy
        ):
            best_val_accuracy = (
                val_metrics["accuracy"]
            )

            save_checkpoint(
                path=(
                    output_dir
                    / BEST_CHECKPOINT_NAME
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val_accuracy=best_val_accuracy,
            )

            print(
                f"保存最佳模型："
                f"Val Acc={best_val_accuracy:.2f}%"
            )

        save_checkpoint(
            path=(
                output_dir
                / LAST_CHECKPOINT_NAME
            ),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val_accuracy=best_val_accuracy,
        )

    print("\n训练完成。")
    print(
        f"最佳验证准确率："
        f"{best_val_accuracy:.2f}%"
    )
    print(
        f"最佳权重："
        f"{output_dir / BEST_CHECKPOINT_NAME}"
    )


if __name__ == "__main__":
    main()
