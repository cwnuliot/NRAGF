# -*- coding: utf-8 -*-
"""Train the corrected original dual-stream baseline on identical corruptions."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.dataset2 import Merge_Two_modality
from data.node_corruption import SCENARIOS
from models.baseline import DualStreamSpatialTemporalGraph
from train_nragf_part import (
    TrainConfig,
    build_evaluation_loader,
    build_train_loader,
    set_seed,
    split_files,
    write_csv,
)
from utils.reliability_metrics import macro_f1


def parse_args() -> Tuple[TrainConfig, int, float]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", default=TrainConfig.video_root)
    parser.add_argument("--output-dir", default="outputs/corrected_baseline")
    parser.add_argument("--protocol", choices=("tune", "final"), default="tune")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rgb-noise-std", type=float, default=1.0)
    parser.add_argument("--rfid-noise-std", type=float, default=1.0)
    parser.add_argument("--modality-dropout", type=float, default=0.05)
    parser.add_argument("--validation-interval", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    config = TrainConfig(
        video_root=args.video_root,
        output_dir=args.output_dir,
        protocol=args.protocol,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        rgb_noise_std=args.rgb_noise_std,
        rfid_noise_std=args.rfid_noise_std,
        modality_dropout_probability=args.modality_dropout,
        validation_interval=max(1, args.validation_interval),
        use_amp=not args.no_amp,
    )
    return config, args.epochs, args.learning_rate


def build_baseline(config: TrainConfig) -> DualStreamSpatialTemporalGraph:
    graph_args = {
        "layout": config.graph_layout,
        "strategy": config.graph_strategy,
        "max_hop": config.max_hop,
        "dilation": config.dilation,
    }
    return DualStreamSpatialTemporalGraph(
        graph_args,
        config.num_class,
        None,
        None,
        dropout=config.dropout,
    )


@torch.no_grad()
def evaluate_loader(model, loader, device) -> Dict[str, float]:
    model.eval()
    confusion = np.zeros((21, 21), dtype=np.int64)
    for rgb, rfid, labels, _, _, _, _ in loader:
        rgb = rgb.to(device, non_blocking=True)
        rfid = rfid.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, _, _ = model(rgb, rfid)
        prediction = logits.argmax(dim=1)
        np.add.at(
            confusion,
            (labels.cpu().numpy(), prediction.cpu().numpy()),
            1,
        )
    total = float(confusion.sum())
    return {
        "accuracy": float(np.trace(confusion)) / max(total, 1.0),
        "macro_f1": macro_f1(confusion),
    }


@torch.no_grad()
def evaluate_matrix(
    model,
    files: Sequence[Sequence[str]],
    config: TrainConfig,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    conditions = [("clean", 0)] + [
        (scenario, k)
        for scenario in SCENARIOS
        if scenario != "clean"
        for k in (1, 2, 3, 4)
    ]
    rows: List[Dict[str, object]] = []
    for scenario, k in conditions:
        metrics = evaluate_loader(
            model,
            build_evaluation_loader(files, config, scenario, k),
            device,
        )
        row: Dict[str, object] = {"scenario": scenario, "k": k}
        row.update(metrics)
        rows.append(row)
    robust = [float(row["accuracy"]) for row in rows[1:]]
    k4 = [
        float(row["accuracy"])
        for row in rows
        if row["scenario"] != "clean" and row["k"] == 4
    ]
    summary = {
        "clean_accuracy": float(rows[0]["accuracy"]),
        "clean_macro_f1": float(rows[0]["macro_f1"]),
        "mean_robust_accuracy": float(np.mean(robust)),
        "mean_robust_macro_f1": float(
            np.mean([float(row["macro_f1"]) for row in rows[1:]])
        ),
        "worst_k4_accuracy": float(np.min(k4)),
    }
    summary["selection_score"] = (
        0.7 * summary["mean_robust_accuracy"]
        + 0.3 * summary["worst_k4_accuracy"]
    )
    return rows, summary


def save_checkpoint(path, model, epoch, config, summary) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "model_type": "corrected_baseline",
            "epoch": epoch,
            "config": asdict(config),
            "summary": summary,
            "baseline_residual_fixed": True,
        },
        str(path),
    )


def main() -> None:
    config, epochs, learning_rate = parse_args()
    set_seed(config.seed)
    output_dir = Path(config.output_dir) / (
        "%s_seed%d" % (config.protocol, config.seed)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(asdict(config), stream, ensure_ascii=False, indent=2)

    all_files = sorted(
        Merge_Two_modality(config.video_root),
        key=lambda item: (str(item[1]), str(item[0])),
    )
    train_files, evaluation_files = split_files(
        all_files, config.train_subjects, config.evaluation_subjects
    )
    dataset, loader = build_train_loader(train_files, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_baseline(config).to(device)
    optimizer = Adam(
        model.parameters(), lr=learning_rate, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=config.min_learning_rate
    )
    scaler = GradScaler(enabled=config.use_amp and device.type == "cuda")
    best_score = -float("inf")
    best_path = output_dir / "best.pt"
    history: List[Dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        dataset.set_epoch(epoch)
        started = time.time()
        loss_sum = 0.0
        correct = 0
        samples = 0
        for rgb, rfid, labels, _, _, _, _ in loader:
            rgb = rgb.to(device, non_blocking=True)
            rfid = rfid.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=config.use_amp and device.type == "cuda"):
                logits, logits_rgb, logits_rfid = model(rgb, rfid)
                main_loss = F.cross_entropy(logits, labels)
                auxiliary = 0.5 * (
                    F.cross_entropy(logits_rgb, labels)
                    + F.cross_entropy(logits_rfid, labels)
                )
                loss = main_loss + config.lambda_aux * auxiliary
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(labels.size(0))
            samples += batch_size
            loss_sum += float(loss.detach().item()) * batch_size
            correct += int((logits.argmax(1) == labels).sum().item())
        scheduler.step()
        row: Dict[str, object] = {
            "epoch": epoch,
            "loss": loss_sum / max(samples, 1),
            "train_accuracy": correct / max(samples, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        if (
            config.protocol == "tune"
            and (epoch % config.validation_interval == 0 or epoch == epochs)
        ):
            _, summary = evaluate_matrix(
                model, evaluation_files, config, device
            )
            row.update(summary)
            if summary["selection_score"] > best_score:
                best_score = summary["selection_score"]
                save_checkpoint(best_path, model, epoch, config, summary)
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    write_csv(output_dir / "history.csv", history)
    if config.protocol == "tune" and best_path.exists():
        model.load_state_dict(
            torch.load(str(best_path), map_location=device)["model"]
        )
    rows, summary = evaluate_matrix(
        model, evaluation_files, config, device
    )
    write_csv(output_dir / "evaluation_matrix.csv", rows)
    with (output_dir / "evaluation_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    save_checkpoint(output_dir / "final_model.pt", model, epochs, config, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
