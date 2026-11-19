# -*- coding: utf-8 -*-
"""Evaluate deterministic corruption matrix and node-reliability ablations."""

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from data.dataset2 import Merge_Two_modality
from train_baseline import build_baseline, evaluate_matrix as evaluate_baseline
from train_nragf_part import (
    TrainConfig,
    build_model,
    evaluate_matrix as evaluate_node_model,
    subject_id,
    write_csv,
)


NODE_PROFILES = {
    "no_reliability": ("ones", False),
    "quality": ("quality_only", False),
    "quality_tcp": ("predicted", False),
    "full": ("predicted", True),
    "oracle": ("oracle", True),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(NODE_PROFILES.keys()),
        default=list(NODE_PROFILES.keys()),
    )
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--check-determinism", action="store_true")
    return parser.parse_args()


def config_from_checkpoint(values: Dict) -> TrainConfig:
    allowed = {field.name for field in fields(TrainConfig)}
    selected = {key: value for key, value in values.items() if key in allowed}
    return TrainConfig(**selected)


def assert_same_summary(first: Dict[str, float], second: Dict[str, float]) -> None:
    for key in first:
        a = float(first[key])
        b = float(second[key])
        if np.isnan(a) and np.isnan(b):
            continue
        if a != b:
            raise AssertionError(
                "Determinism check failed for %s: %.12f != %.12f"
                % (key, a, b)
            )


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    config = config_from_checkpoint(checkpoint.get("config", {}))
    if args.video_root is not None:
        config.video_root = args.video_root
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    subjects = tuple(args.subjects) if args.subjects else config.evaluation_subjects
    all_files = sorted(
        Merge_Two_modality(config.video_root),
        key=lambda item: (str(item[1]), str(item[0])),
    )
    evaluation_files = [item for item in all_files if subject_id(item) in subjects]
    if not evaluation_files:
        raise RuntimeError("No evaluation files for subjects %s" % (subjects,))

    output_dir = Path(args.output_dir) if args.output_dir else (
        checkpoint_path.parent / "robustness_eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = checkpoint.get("model_type", "node_tcp_nragf")
    summaries: List[Dict[str, object]] = []

    if model_type == "corrected_baseline":
        model = build_baseline(config).to(device)
        model.load_state_dict(checkpoint["model"])
        rows, summary = evaluate_baseline(
            model, evaluation_files, config, device
        )
        write_csv(output_dir / "baseline_matrix.csv", rows)
        summary_row: Dict[str, object] = {"profile": "corrected_baseline"}
        summary_row.update(summary)
        summaries.append(summary_row)
        if args.check_determinism:
            _, repeated = evaluate_baseline(
                model, evaluation_files, config, device
            )
            assert_same_summary(summary, repeated)
    else:
        model = build_model(config).to(device)
        model.load_state_dict(checkpoint["model"])
        for profile in args.profiles:
            mode, gate = NODE_PROFILES[profile]
            rows, summary = evaluate_node_model(
                model,
                evaluation_files,
                config,
                device,
                reliability_mode=mode,
                gate_attention=gate,
            )
            write_csv(output_dir / (profile + "_matrix.csv"), rows)
            summary_row = {"profile": profile}
            summary_row.update(summary)
            summaries.append(summary_row)
            if args.check_determinism:
                _, repeated = evaluate_node_model(
                    model,
                    evaluation_files,
                    config,
                    device,
                    reliability_mode=mode,
                    gate_attention=gate,
                )
                assert_same_summary(summary, repeated)

    write_csv(output_dir / "summary.csv", summaries)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summaries, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
