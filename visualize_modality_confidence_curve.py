# -*- coding: utf-8 -*-
"""
不同退化强度下的模态可信度响应曲线

保存位置：
NRAGF/visualize_modality_confidence_curve.py

输出：
outputs/modality_confidence_visualization/modality_confidence_single.png/pdf/svg
outputs/modality_confidence_visualization/modality_confidence_dual.png/pdf/svg
outputs/modality_confidence_visualization/modality_confidence_summary.csv

图 1：单模态退化场景
(a) RGB
(b) RFID

图 2：双模态退化场景
(a) Same
(b) Different
"""

import os
import random
import importlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.dataset2 import GCDF_Dataset, Merge_Two_modality, split_by_volunteer_id


# =========================================================
# 1. 参数区
# =========================================================

PATH_VIDEO = "/home/a2/Datasets/多模态数据集/Video"
VAL_SUBJECT_IDS = ("09", "10")

BATCH_SIZE = 32
NUM_WORKERS = 0
EVAL_SEED = 2026

GRAPH_STRATEGY = "spatial"
DROPOUT = 0.50
NUM_CLASS = 21

RGB_NOISE_STD = 1.0
RFID_NOISE_STD = 1.0

FULL_CHECKPOINT = "outputs/nragf_highk_v2_seed2026/best_nragf_highk_v2.pth"

MODEL_MODULE = "models.nragf_highk_v2"
MODEL_CLASS_CANDIDATES = [
    "NRAGF_Net",
    "GRDF_Net",
    "NRAGFHighKV2",
    "NRAGF",
]

K_LIST = [1, 2, 3, 4]
SCENARIOS = ["RGB", "RFID", "SAME", "DIFFERENT"]

OUTPUT_DIR = "outputs/modality_confidence_visualization"

# 正式画图用 None；调试时可改成 2
MAX_BATCHES = None


# =========================================================
# 2. 五个身体部位划分
# =========================================================

RGB_PARTS = [
    [0, 1, 2, 3, 4],
    [5, 7, 9],
    [6, 8, 10],
    [11, 13, 15],
    [12, 14, 16],
]

RFID_PARTS = [
    [0, 1, 2, 3, 8],
    [4, 6],
    [5, 7],
    [9, 10, 12, 14],
    [11, 13, 15],
]


# =========================================================
# 3. 基础工具函数
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loader():
    all_files = Merge_Two_modality(PATH_VIDEO, threshold=0)
    _, val_files = split_by_volunteer_id(all_files, VAL_SUBJECT_IDS)

    dataset = GCDF_Dataset(
        val_files,
        mode="test",
        rgb_ratio=0.0,
        rfid_ratio=0.0
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False
    )
    return loader


def find_model_class():
    module = importlib.import_module(MODEL_MODULE)

    for cls_name in MODEL_CLASS_CANDIDATES:
        if hasattr(module, cls_name):
            print(f"找到模型类：{MODEL_MODULE}.{cls_name}")
            return getattr(module, cls_name)

    raise RuntimeError(
        f"在 {MODEL_MODULE} 中没有找到模型类。"
        f"已尝试：{MODEL_CLASS_CANDIDATES}"
    )


def build_model(device):
    model_cls = find_model_class()
    graph_args = {"strategy": GRAPH_STRATEGY}

    try:
        model = model_cls(
            graph_args=graph_args,
            num_class=NUM_CLASS,
            dropout=DROPOUT
        )
    except TypeError:
        try:
            model = model_cls(
                graph_args=graph_args,
                num_class=NUM_CLASS
            )
        except TypeError:
            model = model_cls(num_class=NUM_CLASS)

    checkpoint = torch.load(FULL_CHECKPOINT, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        clean_state[k] = v

    msg = model.load_state_dict(clean_state, strict=False)

    print("权重加载完成")
    print("missing keys:", len(msg.missing_keys))
    print("unexpected keys:", len(msg.unexpected_keys))

    model.to(device)
    model.eval()
    return model


def get_batch(batch, device):
    rgb = batch[0].float().to(device)
    rfid = batch[1].float().to(device)
    label = batch[2].long().to(device)
    return rgb, rfid, label


def nodes_from_parts(parts, selected_ids):
    nodes = []
    for pid in selected_ids:
        nodes.extend(parts[pid])
    return sorted(list(set(nodes)))


# =========================================================
# 4. 退化构造
# =========================================================

def choose_parts(k, scenario, seed):
    rng = random.Random(seed)
    all_parts = list(range(5))

    if scenario == "RGB":
        rgb_part_ids = rng.sample(all_parts, k)
        rfid_part_ids = []

    elif scenario == "RFID":
        rgb_part_ids = []
        rfid_part_ids = rng.sample(all_parts, k)

    elif scenario == "SAME":
        selected = rng.sample(all_parts, k)
        rgb_part_ids = selected
        rfid_part_ids = selected

    elif scenario == "DIFFERENT":
        rgb_part_ids = rng.sample(all_parts, k)
        rfid_part_ids = rng.sample(all_parts, k)

        # K=4 时无法做到完全不重叠，这里只避免两边完全相同。
        retry = 0
        while set(rgb_part_ids) == set(rfid_part_ids) and retry < 20:
            rfid_part_ids = rng.sample(all_parts, k)
            retry += 1

    else:
        raise ValueError("scenario 只能是 RGB / RFID / SAME / DIFFERENT")

    return rgb_part_ids, rfid_part_ids


def apply_degradation(rgb, rfid, k, scenario, seed):
    """
    对选中部位施加高斯噪声扰动。
    """
    rgb_deg = rgb.clone()
    rfid_deg = rfid.clone()

    rgb_part_ids, rfid_part_ids = choose_parts(k, scenario, seed)

    rgb_nodes = nodes_from_parts(RGB_PARTS, rgb_part_ids)
    rfid_nodes = nodes_from_parts(RFID_PARTS, rfid_part_ids)

    if len(rgb_nodes) > 0:
        noise = torch.randn_like(rgb_deg[:, :, rgb_nodes, :]) * RGB_NOISE_STD
        rgb_deg[:, :, rgb_nodes, :] = noise

    if len(rfid_nodes) > 0:
        noise = torch.randn_like(rfid_deg[:, :, rfid_nodes, :]) * RFID_NOISE_STD
        rfid_deg[:, :, rfid_nodes, :] = noise

    return rgb_deg, rfid_deg


# =========================================================
# 5. 模型 forward 与可信度提取
# =========================================================

def forward_model(model, rgb, rfid):
    try:
        return model(rgb, rfid, return_details=True)
    except TypeError:
        return model(rgb, rfid)


def extract_confidence(model_output):
    """
    highk_v2 / ablation 模型通常返回：
    output, conf_rgb, conf_rfid, ...

    因此：
    conf_rgb = model_output[1]
    conf_rfid = model_output[2]
    """
    if not isinstance(model_output, (tuple, list)):
        raise RuntimeError("模型输出不是 tuple/list，无法提取 conf_rgb/conf_rfid。")

    if len(model_output) < 3:
        raise RuntimeError("模型输出长度不足 3，无法提取 conf_rgb/conf_rfid。")

    conf_rgb = model_output[1]
    conf_rfid = model_output[2]

    if not torch.is_tensor(conf_rgb) or not torch.is_tensor(conf_rfid):
        raise RuntimeError("model_output[1] 或 model_output[2] 不是 Tensor。")

    if conf_rgb.dim() != 3 or conf_rfid.dim() != 3:
        raise RuntimeError(
            f"conf 维度异常：conf_rgb={conf_rgb.shape}, conf_rfid={conf_rfid.shape}"
        )

    return conf_rgb, conf_rfid


def modality_confidence(conf):
    """
    conf: [B, 1, V]
    返回当前 batch 的平均模态可信度。
    """
    return conf.mean().item()


# =========================================================
# 6. 统计四种场景下 K=1~4 的 q_rgb / q_rfid
# =========================================================

def evaluate_modality_confidence(model, loader, device):
    results = {}

    for scenario in SCENARIOS:
        print(f"\n========== 场景：{scenario} ==========")

        results[scenario] = {
            "K": [],
            "q_rgb": [],
            "q_rfid": [],
        }

        for k in K_LIST:
            q_rgb_vals = []
            q_rfid_vals = []

            for batch_idx, batch in enumerate(loader):
                if MAX_BATCHES is not None and batch_idx >= MAX_BATCHES:
                    break

                rgb, rfid, label = get_batch(batch, device)

                rgb_deg, rfid_deg = apply_degradation(
                    rgb=rgb,
                    rfid=rfid,
                    k=k,
                    scenario=scenario,
                    seed=EVAL_SEED + k * 1000 + batch_idx
                )

                with torch.no_grad():
                    output = forward_model(model, rgb_deg, rfid_deg)
                    conf_rgb, conf_rfid = extract_confidence(output)

                q_rgb_vals.append(modality_confidence(conf_rgb))
                q_rfid_vals.append(modality_confidence(conf_rfid))

            q_rgb_mean = float(np.mean(q_rgb_vals))
            q_rfid_mean = float(np.mean(q_rfid_vals))

            results[scenario]["K"].append(k)
            results[scenario]["q_rgb"].append(q_rgb_mean)
            results[scenario]["q_rfid"].append(q_rfid_mean)

            print(
                f"K={k} | "
                f"q_rgb={q_rgb_mean:.4f}, "
                f"q_rfid={q_rfid_mean:.4f}"
            )

    return results


# =========================================================
# 7. 保存 CSV
# =========================================================

def save_csv(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "modality_confidence_summary.csv")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("scenario,K,q_rgb,q_rfid\n")

        for scenario in SCENARIOS:
            for i, k in enumerate(results[scenario]["K"]):
                f.write(
                    f"{scenario},"
                    f"{k},"
                    f"{results[scenario]['q_rgb'][i]:.6f},"
                    f"{results[scenario]['q_rfid'][i]:.6f}\n"
                )

    print(f"\nCSV 已保存：{csv_path}")


# =========================================================
# 8. 绘图
# =========================================================

def setup_matplotlib():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"


def plot_two_subplots(results, scenarios, titles, save_prefix):
    """
    一张图包含两个子图。
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_matplotlib()

    cm = 1 / 2.54

    # 双子图，适合中文学报单栏宽度附近。
    # 若期刊严格要求不超过 8 cm，可将 8.6 改为 8.0。
    fig_w = 8.6 * cm
    fig_h = 5.2 * cm

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=300)

    for ax, scenario, title in zip(axes, scenarios, titles):
        k = results[scenario]["K"]

        ax.plot(
            k,
            results[scenario]["q_rgb"],
            linestyle="-",
            marker="o",
            linewidth=1.1,
            markersize=3.5,
            color="black",
            label="RGB"
        )

        ax.plot(
            k,
            results[scenario]["q_rfid"],
            linestyle="--",
            marker="s",
            linewidth=1.1,
            markersize=3.5,
            color="black",
            label="RFID"
        )

        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlabel("K", fontsize=9)
        ax.set_xticks(k)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(True, linestyle=":", linewidth=0.5)

    axes[0].set_ylabel("Average confidence", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02)
    )

    fig.subplots_adjust(
        left=0.14,
        right=0.98,
        top=0.82,
        bottom=0.25,
        wspace=0.45
    )

    png_path = os.path.join(OUTPUT_DIR, f"{save_prefix}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{save_prefix}.pdf")
    svg_path = os.path.join(OUTPUT_DIR, f"{save_prefix}.svg")

    plt.savefig(png_path, bbox_inches="tight", dpi=600)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()

    print(f"PNG 已保存：{png_path}")
    print(f"PDF 已保存：{pdf_path}")
    print(f"SVG 已保存：{svg_path}")


def plot_all(results):
    # 图 1：单模态退化
    plot_two_subplots(
        results=results,
        scenarios=["RGB", "RFID"],
        titles=["(a) RGB", "(b) RFID"],
        save_prefix="modality_confidence_single"
    )

    # 图 2：双模态退化
    plot_two_subplots(
        results=results,
        scenarios=["SAME", "DIFFERENT"],
        titles=["(a) Same", "(b) Different"],
        save_prefix="modality_confidence_dual"
    )


# =========================================================
# 9. 主函数
# =========================================================

def main():
    set_seed(EVAL_SEED)
    device = get_device()
    print("使用设备：", device)

    print("正在加载评估数据...")
    loader = build_loader()

    print("正在加载 Full NRAGF 模型...")
    model = build_model(device)

    print("开始统计不同退化强度下的模态可信度...")
    results = evaluate_modality_confidence(model, loader, device)

    save_csv(results)
    plot_all(results)

    print("\n全部完成。")


if __name__ == "__main__":
    main()