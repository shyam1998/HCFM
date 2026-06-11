#!/usr/bin/env python3
"""Minimal univariate HCFM benchmark entry point for reviewers.

This script runs the scalar-potential HCFM data-space benchmark on all UCR
datasets found under ``data/UCR`` and writes the same CSV artifacts used in the
paper experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hcfm_dataspace_v1 import make_hcfm_v1_cfg, run_single_dataset_seed, select_key_scores  # noqa: E402


DEFAULT_METHODS = ["Vanilla Data FM", "Data HCFM"]


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def discover_ucr_datasets(data_root: Path) -> list[str]:
    if not data_root.exists():
        raise FileNotFoundError(f"UCR data root not found: {data_root}")
    dataset_ids = []
    for path in sorted(data_root.rglob("UCR_*.txt")):
        stem = path.stem
        parts = stem.split("_")
        if len(parts) >= 2 and parts[0] == "UCR" and parts[1].isdigit():
            dataset_ids.append(f"UCR_{int(parts[1])}")
    for path in sorted(data_root.rglob("UCR_*_train.npy")):
        stem = path.name.replace("_train.npy", "")
        parts = stem.split("_")
        if len(parts) >= 2 and parts[0] == "UCR" and parts[1].isdigit():
            test_path = path.with_name(f"{stem}_test.npy")
            label_path = path.with_name(f"{stem}_test_label.npy")
            if test_path.exists() and label_path.exists():
                dataset_ids.append(f"UCR_{int(parts[1])}")
    dataset_ids = sorted(set(dataset_ids), key=lambda name: int(name.split("_")[1]))
    if not dataset_ids:
        raise FileNotFoundError(f"No UCR datasets found under: {data_root} or its subdirectories")
    return dataset_ids


def run_one(dataset_id: str, seed: int, args: argparse.Namespace, output_root: Path) -> Path:
    run_dir = output_root / f"{dataset_id}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = make_hcfm_v1_cfg(
        dataset_id=dataset_id,
        seed=seed,
        output_dir=run_dir,
        data_root=Path(args.data_root),
        train_steps=args.train_steps,
        methods=parse_csv(args.methods),
        score_profile=args.score_profile,
        plot_profile="none",
        compute_exact_likelihood_scores=True,
        compute_vus_metrics=True,
        run_full_field_hutchinson_scoring=not args.no_full_divergence,
        strict_v1_config=False,
        run_label="code_submission_univariate",
        hcfm_variant="code_submission",
        hcfm_use_physics_residual_loss=False,
        hcfm_lambda_physics_residual=0.0,
    )
    cfg.window = int(args.window)
    cfg.stride = int(args.stride)
    cfg.ode_method = str(args.ode_method)
    cfg.ode_steps = int(args.ode_steps)
    cfg.eval_n_probe = int(args.eval_n_probe)
    cfg.train_n_probe = int(args.train_n_probe)
    cfg.score_batch_size = int(args.score_batch_size)
    cfg.hcfm_component_score_batch_size = int(args.hcfm_component_score_batch_size)
    cfg.print_every = int(args.print_every)
    cfg.verbose_tables = False

    run_single_dataset_seed(cfg)
    return run_dir


def consolidate(run_dirs: list[Path], output_root: Path) -> None:
    frames = []
    key_frames = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        frames.append(metrics)
        try:
            key_frames.append(select_key_scores(metrics))
        except Exception:
            pass

    if frames:
        all_metrics = pd.concat(frames, ignore_index=True)
        all_metrics.to_csv(output_root / "all_metrics.csv", index=False)
        if key_frames:
            pd.concat(key_frames, ignore_index=True).to_csv(output_root / "key_scores.csv", index=False)

        preferred_cols = [
            "dataset_id",
            "seed",
            "method",
            "Model",
            "score",
            "Score",
            "window_AUPRC",
            "point_AUPRC_mean",
            "point_AUROC_mean",
            "vus_pr_point_mean",
            "FP@95R Normal",
            "point_best_F1_mean",
            "hundman_window_f1_mean_plus_2sd",
            "hundman_window_f1_mean_plus_3sd",
        ]
        existing = [c for c in preferred_cols if c in all_metrics.columns]
        summary = all_metrics[existing].copy()
        summary.to_csv(output_root / "metrics_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal univariate HCFM benchmark.")
    parser.add_argument("--seeds", default="42", help="Comma-separated integer seeds.")
    parser.add_argument("--data_root", default=str(ROOT / "data" / "UCR"))
    parser.add_argument("--output_root", default=str(ROOT / "outputs" / "univariate"))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--train_steps", type=int, default=15000)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--ode_method", default="rk4")
    parser.add_argument("--ode_steps", type=int, default=8)
    parser.add_argument("--train_n_probe", type=int, default=1)
    parser.add_argument("--eval_n_probe", type=int, default=4)
    parser.add_argument("--score_batch_size", type=int, default=512)
    parser.add_argument("--hcfm_component_score_batch_size", type=int, default=64)
    parser.add_argument("--score_profile", default="core", choices=["core", "extended", "debug"])
    parser.add_argument("--no_full_divergence", action="store_true", help="Disable full-field Hutchinson divergence scoring for monolithic/FDM.")
    parser.add_argument("--print_every", type=int, default=500)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) / f"run_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "command_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    run_dirs = []
    dataset_ids = discover_ucr_datasets(Path(args.data_root))
    pd.DataFrame({"dataset_id": dataset_ids}).to_csv(output_root / "discovered_datasets.csv", index=False)
    for dataset_id in dataset_ids:
        for seed_text in parse_csv(args.seeds):
            run_dirs.append(run_one(dataset_id, int(seed_text), args, output_root))

    consolidate(run_dirs, output_root)
    print(f"Done. Outputs written to: {output_root}")


if __name__ == "__main__":
    main()
