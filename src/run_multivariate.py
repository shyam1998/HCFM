#!/usr/bin/env python3
"""Minimal multivariate HCFM reviewer helper.

The multivariate experiment in this prototype is notebook-first because it
contains dataset-specific parsing and diagnostics. This script provides the
reproducible command-line pieces that are useful for code review:

1. audit available multivariate datasets and label prevalence;
2. consolidate existing multivariate HCFM output folders into compact tables;
3. fail clearly if no runnable data/artifacts are present.

For full retraining, run ``hcfm_multivariate.ipynb`` after setting the desired
dataset id in the first configuration cell.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


TRAIN_NAMES = ["train.npy", "train.csv"]
TEST_NAMES = ["test.npy", "test.csv"]
LABEL_NAMES = ["test_label.npy", "test_labels.npy", "labels.npy", "label.npy", "test_label.csv", "test_labels.csv", "labels.csv"]


def find_first(directory: Path, names: list[str], kind: str) -> Path | None:
    for name in names:
        direct = directory / name
        if direct.exists():
            return direct
    lower_map = {p.name.lower(): p for p in directory.iterdir() if p.is_file()}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    files = [p for p in directory.iterdir() if p.is_file()]
    lowered = [(p.name.lower(), p) for p in files]
    if kind == "train":
        candidates = [p for name, p in lowered if "train" in name and p.suffix.lower() in {".npy", ".csv"}]
    elif kind == "test":
        candidates = [
            p
            for name, p in lowered
            if "test" in name and "label" not in name and p.suffix.lower() in {".npy", ".csv"}
        ]
        if not candidates:
            candidates = [p for name, p in lowered if name in {"swat2.csv"}]
    elif kind == "label":
        candidates = [
            p
            for name, p in lowered
            if ("label" in name or "labels" in name) and p.suffix.lower() in {".npy", ".csv"}
        ]
    else:
        candidates = []
    if candidates:
        return sorted(candidates, key=lambda p: p.name.lower())[0]
    return None


def load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=True)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return df.to_numpy()
    raise ValueError(f"Unsupported file type: {path}")


def load_label_vector(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        label_like = [c for c in df.columns if "label" in str(c).lower() or "anomaly" in str(c).lower() or "attack" in str(c).lower()]
        if label_like:
            values = df[label_like[-1]].to_numpy()
        else:
            numeric = df.apply(pd.to_numeric, errors="coerce")
            useful = [c for c in numeric.columns if numeric[c].notna().any()]
            values = numeric[useful[-1]].to_numpy() if useful else df.iloc[:, -1].to_numpy()
        return as_label_vector(values)
    return as_label_vector(load_array(path))


def as_label_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim > 1:
        numeric = pd.DataFrame(arr).apply(pd.to_numeric, errors="coerce").to_numpy()
        if numeric.shape[1] == 1:
            arr = numeric[:, 0]
        else:
            arr = np.nanmax(numeric, axis=1)
    arr = pd.Series(np.asarray(arr).reshape(-1)).apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    return (arr > 0).astype(int)


def audit_datasets(data_root: Path, output_root: Path) -> pd.DataFrame:
    rows = []
    for dataset_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        train_path = find_first(dataset_dir, TRAIN_NAMES, "train")
        test_path = find_first(dataset_dir, TEST_NAMES, "test")
        label_path = find_first(dataset_dir, LABEL_NAMES, "label")
        row = {
            "dataset": dataset_dir.name,
            "train_file": train_path.name if train_path else None,
            "test_file": test_path.name if test_path else None,
            "label_file": label_path.name if label_path else None,
            "status": "ok" if train_path and test_path and label_path else "missing_files",
        }
        try:
            if train_path:
                train = load_array(train_path)
                row["train_shape"] = str(tuple(train.shape))
            if test_path:
                test = load_array(test_path)
                row["test_shape"] = str(tuple(test.shape))
            if label_path:
                labels = load_label_vector(label_path)
                row["label_len"] = int(len(labels))
                row["label_anomaly_points"] = int(labels.sum())
                row["label_anomaly_fraction"] = float(labels.mean()) if len(labels) else np.nan
        except Exception as exc:
            row["status"] = "parse_error"
            row["error"] = repr(exc)
        rows.append(row)
    audit = pd.DataFrame(rows)
    audit.to_csv(output_root / "multivariate_dataset_audit.csv", index=False)
    return audit


def consolidate_existing_outputs(outputs_root: Path, output_root: Path) -> None:
    metric_frames = []
    winner_frames = []
    for run_dir in sorted(outputs_root.glob("*_dataspace_hcfm_multivariate_seed*")):
        metrics_path = run_dir / "dataspace_cnn_metrics.csv"
        winners_path = run_dir / "metric_winners.csv"
        if metrics_path.exists():
            metric_frames.append(pd.read_csv(metrics_path))
        if winners_path.exists():
            winner_frames.append(pd.read_csv(winners_path))

    if metric_frames:
        metrics = pd.concat(metric_frames, ignore_index=True)
        metrics.to_csv(output_root / "multivariate_all_metrics.csv", index=False)
        cols = [
            "dataset_id",
            "seed",
            "Model",
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
        cols = [c for c in cols if c in metrics.columns]
        metrics[cols].to_csv(output_root / "multivariate_metrics_summary.csv", index=False)

    if winner_frames:
        winners = pd.concat(winner_frames, ignore_index=True)
        winners.to_csv(output_root / "multivariate_metric_winners.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit/summarize multivariate HCFM benchmark artifacts.")
    parser.add_argument("--data_root", default=str(ROOT / "data" / "multivariate"))
    parser.add_argument("--existing_outputs", default=str(ROOT / "outputs" / "hcfm_multivariate"))
    parser.add_argument("--output_root", default=str(ROOT / "outputs" / "multivariate"))
    parser.add_argument("--skip_dataset_audit", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "command_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    if not args.skip_dataset_audit:
        data_root = Path(args.data_root)
        if data_root.exists():
            audit = audit_datasets(data_root, output_root)
            print(audit[["dataset", "status", "test_shape", "label_anomaly_fraction"]].to_string(index=False))
        else:
            print(f"Dataset root not found: {data_root}")

    existing_outputs = Path(args.existing_outputs)
    if existing_outputs.exists():
        consolidate_existing_outputs(existing_outputs, output_root)
    else:
        print(f"Existing multivariate output root not found: {existing_outputs}")

    print(f"Done. Outputs written to: {output_root}")


if __name__ == "__main__":
    main()
