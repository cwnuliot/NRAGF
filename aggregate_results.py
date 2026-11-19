# -*- coding: utf-8 -*-
"""Aggregate three-seed final checkpoints and apply acceptance criteria."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch


CORE_METRICS = (
    "clean_accuracy",
    "clean_macro_f1",
    "mean_robust_accuracy",
    "mean_robust_macro_f1",
    "worst_k4_accuracy",
    "quality_auroc",
    "tcp_mae",
    "tcp_ece",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs=3, required=True)
    parser.add_argument("--full", nargs=3, required=True)
    parser.add_argument("--output-dir", default="outputs/three_seed_summary")
    return parser.parse_args()


def load_summaries(paths: Sequence[str]) -> List[Dict[str, float]]:
    summaries = []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu")
        summary = checkpoint.get("summary", {})
        if not summary:
            raise ValueError("Checkpoint has no final summary: %s" % path)
        summaries.append({key: float(value) for key, value in summary.items()})
    return summaries


def aggregate(name: str, summaries: Sequence[Dict[str, float]]) -> Dict[str, float]:
    row: Dict[str, float] = {"model": name}  # type: ignore
    for metric in CORE_METRICS:
        values = np.asarray(
            [summary[metric] for summary in summaries if metric in summary],
            dtype=np.float64,
        )
        if values.size:
            row[metric + "_mean"] = float(np.nanmean(values))
            row[metric + "_std"] = float(np.nanstd(values, ddof=1))
    return row


def main() -> None:
    args = parse_args()
    baseline = aggregate("corrected_baseline", load_summaries(args.baseline))
    full = aggregate("node_tcp_full", load_summaries(args.full))
    acceptance = {
        "mean_robust_improved": (
            full["mean_robust_accuracy_mean"]
            > baseline["mean_robust_accuracy_mean"]
        ),
        "worst_k4_improved": (
            full["worst_k4_accuracy_mean"]
            > baseline["worst_k4_accuracy_mean"]
        ),
        "clean_drop_within_one_point": (
            baseline["clean_accuracy_mean"] - full["clean_accuracy_mean"]
            <= 0.01
        ),
    }
    acceptance["passed"] = bool(all(acceptance.values()))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [baseline, full]
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with (output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    result = {"models": rows, "acceptance": acceptance}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
