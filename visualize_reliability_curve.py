# -*- coding: utf-8 -*-
"""
节点可靠性响应曲线可视化脚本
保存位置：NRAGF/visualize_reliability_curve.py

功能：
1. 加载 Full NRAGF 权重；
2. 在 ID 09、10 跨对象评估数据上构造 K=1~4 退化；
3. 统计正常节点与受损节点的平均可靠性；
4. 保存 csv、png、pdf、svg 曲线图。
"""

import os
import random
import importlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.dataset2 import GCDF_Dataset, Merge_Two_modality, split_by_volunteer_id


# ========================== 1. 参数区 ==========================

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

SCENARIO = "DIFFERENT"  # 建议用 DIFFERENT，可视化两种模态都有受损节点
K_LIST = [1, 2, 3, 4]

OUTPUT_DIR = "outputs/reliability_visualization"
MAX_BATCHES = None      # 正式画图用 None；调试可改成 2


# ========================== 2. 部位划分 ==========================
# 5 个宏观部位，与前面实验设置保持一致。

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


# ========================== 3. 工具函数 ==========================

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

    val_dataset = GCDF_Dataset(
        val_files,
        mode="test",
        rgb_ratio=0.0,
        rfid_ratio=0.0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False
    )
    return val_loader


def find_model_class():
    module = importlib.import_module(MODEL_MODULE)
    for cls_name in MODEL_CLASS_CANDIDATES:
        if hasattr(module, cls_name):
            print(f"✅ 找到模型类：{MODEL_MODULE}.{cls_name}")
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
    print("✅ 权重加载完成")
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

        # K=4 时完全不重叠不可能，这里只避免两边完全相同。
        retry = 0
        while set(rgb_part_ids) == set(rfid_part_ids) and retry < 20:
            rfid_part_ids = rng.sample(all_parts, k)
            retry += 1

    else:
        raise ValueError("SCENARIO 只能是 RGB / RFID / SAME / DIFFERENT")

    return rgb_part_ids, rfid_part_ids


def apply_degradation(rgb, rfid, k, scenario, seed):
    """
    mask:
    1 表示正常节点；
    0 表示受损节点。
    """
    rgb_deg = rgb.clone()
    rfid_deg = rfid.clone()

    b_rgb, _, v_rgb, _ = rgb_deg.shape
    b_rfid, _, v_rfid, _ = rfid_deg.shape

    rgb_mask = torch.ones((b_rgb, v_rgb), device=rgb.device)
    rfid_mask = torch.ones((b_rfid, v_rfid), device=rfid.device)

    rgb_part_ids, rfid_part_ids = choose_parts(k, scenario, seed)

    rgb_nodes = nodes_from_parts(RGB_PARTS, rgb_part_ids)
    rfid_nodes = nodes_from_parts(RFID_PARTS, rfid_part_ids)

    if len(rgb_nodes) > 0:
        noise = torch.randn_like(rgb_deg[:, :, rgb_nodes, :]) * RGB_NOISE_STD
        rgb_deg[:, :, rgb_nodes, :] = noise
        rgb_mask[:, rgb_nodes] = 0.0

    if len(rfid_nodes) > 0:
        noise = torch.randn_like(rfid_deg[:, :, rfid_nodes, :]) * RFID_NOISE_STD
        rfid_deg[:, :, rfid_nodes, :] = noise
        rfid_mask[:, rfid_nodes] = 0.0

    return rgb_deg, rfid_deg, rgb_mask, rfid_mask


def forward_model(model, rgb, rfid):
    try:
        return model(rgb, rfid, return_details=True)
    except TypeError:
        return model(rgb, rfid)


def extract_confidence(model_output):
    """
    你的模型返回格式通常是：
    output, conf_rgb, conf_rfid, feat_rgb, clean_rgb, feat_rfid, clean_rfid, logits_rgb, logits_rfid, ...
    所以 conf_rgb = output[1], conf_rfid = output[2]
    """
    if not isinstance(model_output, (tuple, list)):
        raise RuntimeError("模型输出不是 tuple/list，无法直接提取 conf_rgb/conf_rfid。")

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


def mean_by_mask(conf, mask, value):
    """
    conf: [B, 1, V]
    mask: [B, V]
    value: 1.0 正常节点；0.0 受损节点
    """
    conf = conf.squeeze(1)
    selected = (mask == value)

    if selected.sum().item() == 0:
        return None

    return conf[selected].mean().item()


# ========================== 4. 统计可靠性 ==========================

def evaluate_curve(model, loader, device):
    results = {
        "K": [],
        "RGB_intact": [],
        "RGB_corrupted": [],
        "RFID_intact": [],
        "RFID_corrupted": [],
    }

    for k in K_LIST:
        rgb_intact_vals = []
        rgb_corrupted_vals = []
        rfid_intact_vals = []
        rfid_corrupted_vals = []

        for batch_idx, batch in enumerate(loader):
            if MAX_BATCHES is not None and batch_idx >= MAX_BATCHES:
                break

            rgb, rfid, label = get_batch(batch, device)

            rgb_deg, rfid_deg, rgb_mask, rfid_mask = apply_degradation(
                rgb=rgb,
                rfid=rfid,
                k=k,
                scenario=SCENARIO,
                seed=EVAL_SEED + k * 1000 + batch_idx
            )

            with torch.no_grad():
                output = forward_model(model, rgb_deg, rfid_deg)
                conf_rgb, conf_rfid = extract_confidence(output)

            v = mean_by_mask(conf_rgb, rgb_mask, 1.0)
            if v is not None:
                rgb_intact_vals.append(v)

            v = mean_by_mask(conf_rgb, rgb_mask, 0.0)
            if v is not None:
                rgb_corrupted_vals.append(v)

            v = mean_by_mask(conf_rfid, rfid_mask, 1.0)
            if v is not None:
                rfid_intact_vals.append(v)

            v = mean_by_mask(conf_rfid, rfid_mask, 0.0)
            if v is not None:
                rfid_corrupted_vals.append(v)

        def safe_mean(x):
            return float(np.mean(x)) if len(x) > 0 else np.nan

        results["K"].append(k)
        results["RGB_intact"].append(safe_mean(rgb_intact_vals))
        results["RGB_corrupted"].append(safe_mean(rgb_corrupted_vals))
        results["RFID_intact"].append(safe_mean(rfid_intact_vals))
        results["RFID_corrupted"].append(safe_mean(rfid_corrupted_vals))

        print(
            f"K={k} | "
            f"RGB intact={results['RGB_intact'][-1]:.4f}, "
            f"RGB corrupted={results['RGB_corrupted'][-1]:.4f}, "
            f"RFID intact={results['RFID_intact'][-1]:.4f}, "
            f"RFID corrupted={results['RFID_corrupted'][-1]:.4f}"
        )

    return results


# ========================== 5. 保存结果和画图 ==========================

def save_csv(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"reliability_curve_{SCENARIO}.csv")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("K,RGB_intact,RGB_corrupted,RFID_intact,RFID_corrupted\n")
        for i, k in enumerate(results["K"]):
            f.write(
                f"{k},"
                f"{results['RGB_intact'][i]:.6f},"
                f"{results['RGB_corrupted'][i]:.6f},"
                f"{results['RFID_intact'][i]:.6f},"
                f"{results['RFID_corrupted'][i]:.6f}\n"
            )

    print(f"✅ CSV 已保存：{csv_path}")


def plot_curve(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 单栏图宽控制在 8 cm 以内，适合中文学报排版
    cm = 1 / 2.54
    fig_w = 8.0 * cm
    fig_h = 4.8 * cm

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    k = results["K"]

    # 可靠性差值：正常节点平均可靠性 - 受损节点平均可靠性
    rgb_gap = [
        intact - corrupted
        for intact, corrupted in zip(results["RGB_intact"], results["RGB_corrupted"])
    ]

    rfid_gap = [
        intact - corrupted
        for intact, corrupted in zip(results["RFID_intact"], results["RFID_corrupted"])
    ]

    plt.figure(figsize=(fig_w, fig_h), dpi=300)

    plt.plot(
        k,
        rgb_gap,
        linestyle="-",
        marker="o",
        linewidth=1.1,
        markersize=3.5,
        color="black",
        label="RGB"
    )

    plt.plot(
        k,
        rfid_gap,
        linestyle="--",
        marker="s",
        linewidth=1.1,
        markersize=3.5,
        color="black",
        label="RFID"
    )

    plt.xlabel("K", fontsize=9)
    plt.ylabel("Reliability gap", fontsize=9)
    plt.xticks(k)
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.legend(frameon=False, fontsize=8)

    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, f"reliability_gap_{SCENARIO}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"reliability_gap_{SCENARIO}.pdf")
    svg_path = os.path.join(OUTPUT_DIR, f"reliability_gap_{SCENARIO}.svg")

    plt.savefig(png_path, bbox_inches="tight", dpi=600)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()

    print(f"✅ PNG 已保存：{png_path}")
    print(f"✅ PDF 已保存：{pdf_path}")
    print(f"✅ SVG 已保存：{svg_path}")

    print("\n可靠性差值统计：")
    for i, kk in enumerate(k):
        print(
            f"K={kk} | "
            f"RGB gap={rgb_gap[i]:.4f}, "
            f"RFID gap={rfid_gap[i]:.4f}"
        )


# ========================== 6. 主函数 ==========================

def main():
    set_seed(EVAL_SEED)
    device = get_device()
    print("使用设备：", device)

    print("正在加载评估数据...")
    loader = build_loader()

    print("正在加载 Full NRAGF 模型...")
    model = build_model(device)

    print(f"开始统计节点可靠性曲线，场景：{SCENARIO}")
    results = evaluate_curve(model, loader, device)

    save_csv(results)
    plot_curve(results)

    print("全部完成。")


if __name__ == "__main__":
    main()