# -*- coding: utf-8 -*-
"""
统一鲁棒性对比评估脚本
保存位置：NRAGF/evaluate_comparison.py

PyCharm 直接运行本文件。

评估：
- Clean
- RGB K=1..4
- RFID K=1..4
- RGB+RFID 同部位 K=1..4
- RGB+RFID 不同部位 K=1..4
"""

# =============================================================================
# 1. 所有参数集中在这里
# =============================================================================

PATH_VIDEO = "/home/a2/Datasets/多模态数据集/Video"
VAL_SUBJECT_IDS = ("09", "10")

BATCH_SIZE = 64
NUM_WORKERS = 4
EVAL_SEED = 2026

GRAPH_STRATEGY = "spatial"
DROPOUT = 0.50

RGB_NOISE_STD = 1.0
RFID_NOISE_STD = 1.0

MF_MEAN_CHECKPOINT = (
    "outputs/comparison/mf_mean/best.pth"
)
LF_CMSA_CHECKPOINT = (
    "outputs/comparison/lf_cmsa/best.pth"
)
NRAGF_CHECKPOINT = (
    "outputs/nragf_optimized_seed2026/"
    "best_nragf_optimized.pth"
)

OUTPUT_DIR = "outputs/comparison_results"

RUN_MF_MEAN = True
RUN_LF_CMSA = True
RUN_NRAGF = True


# =============================================================================
# 2. 导入
# =============================================================================

import csv
import itertools
import json
import os
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.dataset2 import (
    Get_RGB_datas,
    Get_RFID_datas,
    Merge_Two_modality,
)
from data.dataset_nragf import (
    RGB_PARTS,
    RFID_PARTS,
)
from models.comparison_models import (
    MFMeanNet,
    LFCMSANet,
)
from models.nragf_optimized import NRAGFNet


SCENARIO_OFFSETS = {
    "CLEAN": 0,
    "RGB": 100_000,
    "RFID": 200_000,
    "SAME": 300_000,
    "DIFFERENT": 400_000,
}


class FixedPartCorruptionDataset(Dataset):
    """
    每个样本的部位选择和噪声由固定种子决定，
    因此所有方法看到完全相同的输入。
    """

    def __init__(
        self,
        file_list,
        scenario: str,
        k: int = 0,
    ) -> None:
        self.file_list = list(file_list)
        self.scenario = scenario.upper()
        self.k = int(k)

        if self.scenario not in SCENARIO_OFFSETS:
            raise ValueError(self.scenario)

        if self.scenario != "CLEAN":
            if self.k not in (1, 2, 3, 4):
                raise ValueError(
                    "K must be 1, 2, 3, or 4."
                )

        self.combinations = {
            value: list(
                itertools.combinations(range(5), value)
            )
            for value in (1, 2, 3, 4)
        }

    def __len__(self) -> int:
        return len(self.file_list)

    @staticmethod
    def replace_parts(
        data: np.ndarray,
        parts,
        selected_part_ids,
        noise_std: float,
        rng: np.random.Generator,
    ) -> None:
        node_indices = []

        for part_id in selected_part_ids:
            node_indices.extend(parts[int(part_id)])

        node_indices = np.asarray(
            sorted(set(node_indices)),
            dtype=np.int64,
        )

        noise = rng.normal(
            loc=0.0,
            scale=float(noise_std),
            size=(
                data.shape[0],
                node_indices.size,
                data.shape[2],
            ),
        ).astype(np.float32)

        data[:, node_indices, :] = noise

    def select_part_sets(
        self,
        rng: np.random.Generator,
    ):
        combinations = self.combinations[self.k]

        rgb_index = int(
            rng.integers(0, len(combinations))
        )
        rgb_parts = combinations[rgb_index]

        if self.scenario == "SAME":
            return rgb_parts, rgb_parts

        if self.scenario == "DIFFERENT":
            # 选择另一个组合，确保两个模态的受损部位集合不同。
            shift = int(
                rng.integers(1, len(combinations))
            )
            rfid_index = (
                rgb_index + shift
            ) % len(combinations)

            return rgb_parts, combinations[rfid_index]

        return rgb_parts, rgb_parts

    def __getitem__(self, index: int):
        rgb_path, rfid_path, label = self.file_list[index]

        rgb = np.asarray(
            Get_RGB_datas(rgb_path),
            dtype=np.float32,
        ).copy()

        rfid = np.asarray(
            Get_RFID_datas(rfid_path),
            dtype=np.float32,
        ).copy()

        if self.scenario != "CLEAN":
            seed = (
                EVAL_SEED
                + SCENARIO_OFFSETS[self.scenario]
                + self.k * 10_000
                + index
            )
            rng = np.random.default_rng(seed)

            selected_rgb, selected_rfid = (
                self.select_part_sets(rng)
            )

            if self.scenario == "RGB":
                self.replace_parts(
                    rgb,
                    RGB_PARTS,
                    selected_rgb,
                    RGB_NOISE_STD,
                    rng,
                )

            elif self.scenario == "RFID":
                self.replace_parts(
                    rfid,
                    RFID_PARTS,
                    selected_rfid,
                    RFID_NOISE_STD,
                    rng,
                )

            elif self.scenario in (
                "SAME",
                "DIFFERENT",
            ):
                self.replace_parts(
                    rgb,
                    RGB_PARTS,
                    selected_rgb,
                    RGB_NOISE_STD,
                    rng,
                )
                self.replace_parts(
                    rfid,
                    RFID_PARTS,
                    selected_rfid,
                    RFID_NOISE_STD,
                    rng,
                )

        return (
            torch.from_numpy(rgb),
            torch.from_numpy(rfid),
            torch.tensor(int(label), dtype=torch.long),
        )


def graph_args() -> Dict:
    return {"strategy": GRAPH_STRATEGY}


def build_models() -> Dict[str, Tuple[nn.Module, str]]:
    models = {}

    if RUN_MF_MEAN:
        models["MF-Mean"] = (
            MFMeanNet(
                graph_args=graph_args(),
                num_class=21,
                dropout=DROPOUT,
            ),
            MF_MEAN_CHECKPOINT,
        )

    if RUN_LF_CMSA:
        models["LF-CMSA"] = (
            LFCMSANet(
                graph_args=graph_args(),
                num_class=21,
                dropout=DROPOUT,
            ),
            LF_CMSA_CHECKPOINT,
        )

    if RUN_NRAGF:
        models["NRAGF"] = (
            NRAGFNet(
                graph_args=graph_args(),
                num_class=21,
                dropout=DROPOUT,
                tcp_hidden_dim=128,
                tcp_dropout=0.20,
                suppress_strength=0.70,
                suppress_power=2.0,
                quality_prior_strength=1.0,
                fusion_residual_strength=1.0,
            ),
            NRAGF_CHECKPOINT,
        )

    return models


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> nn.Module:
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(
            f"找不到权重：{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
    ):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )
    model.to(device)
    model.eval()

    return model


def subject_id_from_item(item: Sequence[str]) -> str:
    return os.path.basename(
        os.path.dirname(item[1])
    )


def get_validation_files():
    all_files = Merge_Two_modality(
        PATH_VIDEO,
        threshold=0,
    )

    val_ids = set(VAL_SUBJECT_IDS)

    val_files = [
        item
        for item in all_files
        if subject_id_from_item(item) in val_ids
    ]

    val_files = sorted(
        val_files,
        key=lambda item: (
            str(item[1]),
            str(item[0]),
        ),
    )

    if not val_files:
        raise RuntimeError(
            "验证集为空，请检查路径和 VAL_SUBJECT_IDS。"
        )

    print(f"验证样本数量：{len(val_files)}")
    return val_files


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    correct = 0
    total = 0

    for rgb, rfid, labels in loader:
        rgb = rgb.to(device, non_blocking=True)
        rfid = rfid.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        model_output = model(rgb, rfid)
        logits = model_output[0]

        correct += int(
            (
                logits.argmax(dim=1)
                == labels
            ).sum().item()
        )
        total += labels.size(0)

    return 100.0 * correct / max(total, 1)


def write_long_csv(
    path: Path,
    rows,
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "scenario",
                "k",
                "accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_table_csv(
    path: Path,
    rows,
) -> None:
    lookup = {
        (
            row["method"],
            row["scenario"],
            row["k"],
        ): row["accuracy"]
        for row in rows
        if row["scenario"] != "CLEAN"
    }

    methods = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(row["method"])

    scenarios = [
        "RGB",
        "RFID",
        "SAME",
        "DIFFERENT",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "scenario",
                "method",
                "K=1",
                "K=2",
                "K=3",
                "K=4",
            ]
        )

        for scenario in scenarios:
            for method in methods:
                writer.writerow(
                    [
                        scenario,
                        method,
                        *[
                            f"{lookup[(method, scenario, k)]:.2f}"
                            for k in (1, 2, 3, 4)
                        ],
                    ]
                )


def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"评估设备：{device}")

    val_files = get_validation_files()
    model_specs = build_models()

    models = {}
    for method_name, (
        model,
        checkpoint_path,
    ) in model_specs.items():
        print(
            f"加载 {method_name}："
            f"{checkpoint_path}"
        )

        models[method_name] = (
            load_model_checkpoint(
                model=model,
                checkpoint_path=checkpoint_path,
                device=device,
            )
        )

    rows = []

    clean_dataset = FixedPartCorruptionDataset(
        val_files,
        scenario="CLEAN",
        k=0,
    )
    clean_loader = DataLoader(
        clean_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print("\n========== Clean ==========")
    for method_name, model in models.items():
        accuracy = evaluate_accuracy(
            model,
            clean_loader,
            device,
        )
        print(f"{method_name}: {accuracy:.2f}%")

        rows.append(
            {
                "method": method_name,
                "scenario": "CLEAN",
                "k": 0,
                "accuracy": accuracy,
            }
        )

    for scenario in (
        "RGB",
        "RFID",
        "SAME",
        "DIFFERENT",
    ):
        for k in (1, 2, 3, 4):
            dataset = FixedPartCorruptionDataset(
                val_files,
                scenario=scenario,
                k=k,
            )

            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
            )

            print(
                f"\n========== {scenario}, K={k} =========="
            )

            for method_name, model in models.items():
                accuracy = evaluate_accuracy(
                    model,
                    loader,
                    device,
                )

                print(
                    f"{method_name}: {accuracy:.2f}%"
                )

                rows.append(
                    {
                        "method": method_name,
                        "scenario": scenario,
                        "k": k,
                        "accuracy": accuracy,
                    }
                )

    long_csv = output_dir / "comparison_long.csv"
    table_csv = output_dir / "comparison_table.csv"

    write_long_csv(long_csv, rows)
    write_table_csv(table_csv, rows)

    with (
        output_dir / "evaluation_config.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            {
                "PATH_VIDEO": PATH_VIDEO,
                "VAL_SUBJECT_IDS": VAL_SUBJECT_IDS,
                "EVAL_SEED": EVAL_SEED,
                "RGB_NOISE_STD": RGB_NOISE_STD,
                "RFID_NOISE_STD": RFID_NOISE_STD,
                "MF_MEAN_CHECKPOINT": MF_MEAN_CHECKPOINT,
                "LF_CMSA_CHECKPOINT": LF_CMSA_CHECKPOINT,
                "NRAGF_CHECKPOINT": NRAGF_CHECKPOINT,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n评估完成。")
    print(f"长表：{long_csv}")
    print(f"论文表：{table_csv}")


if __name__ == "__main__":
    main()
