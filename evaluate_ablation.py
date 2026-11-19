# -*- coding: utf-8 -*-
"""
NRAGF 正式消融统一评估脚本
保存位置：NRAGF/evaluate_ablation.py
PyCharm 直接运行本文件。
输出：outputs/ablation_results/ablation_summary.csv 等
"""

# ========================= 1. 参数区 =========================
PATH_VIDEO = "/home/a2/Datasets/多模态数据集/Video"
VAL_SUBJECT_IDS = ("09", "10")
BATCH_SIZE = 32
NUM_WORKERS = 0
EVAL_SEED = 2026
GRAPH_STRATEGY = "spatial"
DROPOUT = 0.50
RGB_NOISE_STD = 1.0
RFID_NOISE_STD = 1.0
BASELINE_CHECKPOINT = "outputs/ablation/baseline/best.pth"
MNRL_CHECKPOINT = "outputs/ablation/mnrl/best.pth"
MNRL_RNA_CHECKPOINT = "outputs/ablation/mnrl_rna/best.pth"
FULL_CHECKPOINT = "outputs/nragf_highk_v2_seed2026/best_nragf_highk_v2.pth"
OUTPUT_DIR = "outputs/ablation_results"
RESUME_EVALUATION = True

# ========================= 2. 导入 =========================
import csv, gc, itertools, json, os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from data.dataset2 import Get_RGB_datas, Get_RFID_datas, Merge_Two_modality
from data.dataset_nragf import RGB_PARTS, RFID_PARTS
from models.nragf_ablation import NRAGFAblationNet

SCENARIO_OFFSETS = {"CLEAN": 0, "RGB": 100000, "RFID": 200000, "SAME": 300000, "DIFFERENT": 400000}

class FixedPartCorruptionDataset(Dataset):
    def __init__(self, file_list, scenario: str, k: int = 0) -> None:
        self.file_list = list(file_list)
        self.scenario = scenario.upper()
        self.k = int(k)
        if self.scenario not in SCENARIO_OFFSETS:
            raise ValueError(self.scenario)
        if self.scenario != "CLEAN" and self.k not in (1, 2, 3, 4):
            raise ValueError("K must be 1, 2, 3, or 4.")
        self.combinations = {v: list(itertools.combinations(range(5), v)) for v in (1, 2, 3, 4)}

    def __len__(self) -> int:
        return len(self.file_list)

    @staticmethod
    def replace_parts(data: np.ndarray, parts, selected_part_ids, noise_std: float, rng: np.random.Generator) -> None:
        node_indices = []
        for part_id in selected_part_ids:
            node_indices.extend(parts[int(part_id)])
        node_indices = np.asarray(sorted(set(node_indices)), dtype=np.int64)
        noise = rng.normal(0.0, float(noise_std), size=(data.shape[0], node_indices.size, data.shape[2])).astype(np.float32)
        data[:, node_indices, :] = noise

    def select_part_sets(self, rng: np.random.Generator):
        combinations = self.combinations[self.k]
        rgb_index = int(rng.integers(0, len(combinations)))
        rgb_parts = combinations[rgb_index]
        if self.scenario == "SAME":
            return rgb_parts, rgb_parts
        if self.scenario == "DIFFERENT":
            shift = int(rng.integers(1, len(combinations)))
            return rgb_parts, combinations[(rgb_index + shift) % len(combinations)]
        return rgb_parts, rgb_parts

    def __getitem__(self, index: int):
        rgb_path, rfid_path, label = self.file_list[index]
        rgb = np.asarray(Get_RGB_datas(rgb_path), dtype=np.float32).copy()
        rfid = np.asarray(Get_RFID_datas(rfid_path), dtype=np.float32).copy()
        if self.scenario != "CLEAN":
            seed = EVAL_SEED + SCENARIO_OFFSETS[self.scenario] + self.k * 10000 + index
            rng = np.random.default_rng(seed)
            selected_rgb, selected_rfid = self.select_part_sets(rng)
            if self.scenario == "RGB":
                self.replace_parts(rgb, RGB_PARTS, selected_rgb, RGB_NOISE_STD, rng)
            elif self.scenario == "RFID":
                self.replace_parts(rfid, RFID_PARTS, selected_rfid, RFID_NOISE_STD, rng)
            elif self.scenario in ("SAME", "DIFFERENT"):
                self.replace_parts(rgb, RGB_PARTS, selected_rgb, RGB_NOISE_STD, rng)
                self.replace_parts(rfid, RFID_PARTS, selected_rfid, RFID_NOISE_STD, rng)
        return torch.from_numpy(rgb), torch.from_numpy(rfid), torch.tensor(int(label), dtype=torch.long)

def graph_args() -> Dict:
    return {"strategy": GRAPH_STRATEGY}

def build_ablation_model(mode: str) -> NRAGFAblationNet:
    return NRAGFAblationNet(
        graph_args=graph_args(), ablation_mode=mode, num_class=21, dropout=DROPOUT,
        tcp_hidden_dim=128, tcp_dropout=0.20, pool_weight_floor=0.05, pool_weight_power=2.0,
        quality_prior_strength=1.0, asymmetric_fusion_strength=1.0, joint_degradation_strength=1.0,
    )

def model_specs() -> List[Tuple[str, str, str]]:
    return [
        ("Baseline", "BASELINE", BASELINE_CHECKPOINT),
        ("+MNRL", "MNRL", MNRL_CHECKPOINT),
        ("+MNRL+RNA", "MNRL_RNA", MNRL_RNA_CHECKPOINT),
        ("Full NRAGF", "FULL", FULL_CHECKPOINT),
    ]

def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> nn.Module:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到权重：{path}")
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("权重加载提示：")
        if missing: print("  Missing keys:", missing)
        if unexpected: print("  Unexpected keys:", unexpected)
    model.to(device).eval()
    return model

def subject_id_from_item(item: Sequence[str]) -> str:
    return os.path.basename(os.path.dirname(item[1]))

def get_validation_files():
    all_files = Merge_Two_modality(PATH_VIDEO, threshold=0)
    val_ids = set(VAL_SUBJECT_IDS)
    val_files = [item for item in all_files if subject_id_from_item(item) in val_ids]
    val_files = sorted(val_files, key=lambda item: (str(item[1]), str(item[0])))
    if not val_files:
        raise RuntimeError("验证集为空，请检查 PATH_VIDEO 和 VAL_SUBJECT_IDS。")
    print(f"验证样本数量：{len(val_files)}")
    return val_files

@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    correct = 0
    total = 0
    for rgb, rfid, labels in loader:
        rgb = rgb.to(device, non_blocking=True)
        rfid = rfid.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        output = model(rgb, rfid, reliability_scale=1.0)
        logits = output[0]
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
    return 100.0 * correct / max(total, 1)

def read_existing_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append({"method": row["method"], "scenario": row["scenario"], "k": int(row["k"]), "accuracy": float(row["accuracy"])})
    return rows

def write_long_csv(path: Path, rows: List[Dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["method", "scenario", "k", "accuracy"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["method"], r["scenario"], int(r["k"]))))

def write_full_table_csv(path: Path, rows: List[Dict]) -> None:
    lookup = {(r["method"], r["scenario"], int(r["k"])): r["accuracy"] for r in rows}
    methods = [spec[0] for spec in model_specs()]
    scenarios = ["RGB", "RFID", "SAME", "DIFFERENT"]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["scenario", "method", "K=1", "K=2", "K=3", "K=4"])
        for scenario in scenarios:
            for method in methods:
                writer.writerow([scenario, method, *[(f"{lookup[(method, scenario, k)]:.2f}" if (method, scenario, k) in lookup else "") for k in (1,2,3,4)]])

def mean_of(values: List[float]) -> float:
    return float(sum(values) / max(len(values), 1))

def write_summary_csv(path: Path, rows: List[Dict]) -> None:
    lookup = {(r["method"], r["scenario"], int(r["k"])): r["accuracy"] for r in rows}
    methods = [spec[0] for spec in model_specs()]
    scenarios = ["RGB", "RFID", "SAME", "DIFFERENT"]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "Clean", "RGB Avg.", "RFID Avg.", "Same Avg.", "Different Avg.", "All Corrupt Avg.", "High-K Avg."])
        for method in methods:
            clean = lookup.get((method, "CLEAN", 0), None)
            scenario_avgs = {}
            for scenario in scenarios:
                vals = [lookup[(method, scenario, k)] for k in (1,2,3,4) if (method, scenario, k) in lookup]
                scenario_avgs[scenario] = mean_of(vals)
            all_vals = [lookup[(method, s, k)] for s in scenarios for k in (1,2,3,4) if (method, s, k) in lookup]
            high_vals = [lookup[(method, s, k)] for s in scenarios for k in (3,4) if (method, s, k) in lookup]
            writer.writerow([method, f"{clean:.2f}" if clean is not None else "", f"{scenario_avgs['RGB']:.2f}", f"{scenario_avgs['RFID']:.2f}", f"{scenario_avgs['SAME']:.2f}", f"{scenario_avgs['DIFFERENT']:.2f}", f"{mean_of(all_vals):.2f}", f"{mean_of(high_vals):.2f}"])

def save_config(output_dir: Path) -> None:
    with (output_dir / "ablation_config.json").open("w", encoding="utf-8") as file:
        json.dump({
            "PATH_VIDEO": PATH_VIDEO, "VAL_SUBJECT_IDS": VAL_SUBJECT_IDS, "BATCH_SIZE": BATCH_SIZE,
            "NUM_WORKERS": NUM_WORKERS, "EVAL_SEED": EVAL_SEED, "RGB_NOISE_STD": RGB_NOISE_STD,
            "RFID_NOISE_STD": RFID_NOISE_STD, "BASELINE_CHECKPOINT": BASELINE_CHECKPOINT,
            "MNRL_CHECKPOINT": MNRL_CHECKPOINT, "MNRL_RNA_CHECKPOINT": MNRL_RNA_CHECKPOINT,
            "FULL_CHECKPOINT": FULL_CHECKPOINT,
        }, file, ensure_ascii=False, indent=2)

def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    long_csv = output_dir / "ablation_long.csv"
    full_table_csv = output_dir / "ablation_table_full.csv"
    summary_csv = output_dir / "ablation_summary.csv"
    save_config(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"评估设备：{device}")
    print(f"DataLoader workers: {NUM_WORKERS}")
    val_files = get_validation_files()
    rows = read_existing_rows(long_csv) if RESUME_EVALUATION else []
    completed = {(r["method"], r["scenario"], int(r["k"])) for r in rows}
    tasks = [("CLEAN", 0)] + [(s, k) for s in ("RGB", "RFID", "SAME", "DIFFERENT") for k in (1, 2, 3, 4)]

    for method_name, mode, checkpoint_path in model_specs():
        print(f"\n========== 加载 {method_name} ==========")
        print(f"模式：{mode}")
        print(f"权重：{checkpoint_path}")
        model = load_checkpoint(build_ablation_model(mode), checkpoint_path, device)
        for scenario, k in tasks:
            key = (method_name, scenario, k)
            if key in completed:
                print(f"跳过已完成：{method_name}, {scenario}, K={k}")
                continue
            dataset = FixedPartCorruptionDataset(val_files, scenario=scenario, k=k)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
            accuracy = evaluate_accuracy(model, loader, device)
            print(f"{method_name:12s} | {scenario:10s} | K={k} | Acc={accuracy:.2f}%")
            rows.append({"method": method_name, "scenario": scenario, "k": k, "accuracy": accuracy})
            write_long_csv(long_csv, rows)
            write_full_table_csv(full_table_csv, rows)
            write_summary_csv(summary_csv, rows)
            del loader, dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_long_csv(long_csv, rows)
    write_full_table_csv(full_table_csv, rows)
    write_summary_csv(summary_csv, rows)
    print("\n消融评估完成。")
    print(f"长表：{long_csv}")
    print(f"完整 K 表：{full_table_csv}")
    print(f"摘要表：{summary_csv}")

if __name__ == "__main__":
    main()
