from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def load_ucr_dataset(root: Path, dataset_id: str):
    train = np.asarray(np.load(root / f"{dataset_id}_train.npy"), dtype=np.float32)
    test = np.asarray(np.load(root / f"{dataset_id}_test.npy"), dtype=np.float32)
    label = np.asarray(np.load(root / f"{dataset_id}_test_label.npy")).astype(np.int64).reshape(-1)
    if train.ndim == 1:
        train = train[:, None]
    if test.ndim == 1:
        test = test[:, None]
    return train, test, label

def robust_standardize(train: np.ndarray, test: np.ndarray):
    med = np.median(train, axis=0, keepdims=True)
    mad = np.median(np.abs(train - med), axis=0, keepdims=True)
    scale = 1.4826 * mad + 1e-6
    return (train - med) / scale, (test - med) / scale

def make_windows(x: np.ndarray, labels: np.ndarray | None, window: int, stride: int):
    xs, ys, starts, frac = [], [], [], []
    for s in range(0, len(x) - window + 1, stride):
        e = s + window
        xs.append(x[s:e])
        starts.append(s)
        if labels is None:
            ys.append(0)
            frac.append(0.0)
        else:
            window_labels = labels[s:e]
            ys.append(int(window_labels.max() > 0))
            frac.append(float(window_labels.mean()))
    return (
        np.stack(xs).astype(np.float32),
        np.asarray(ys, dtype=np.int64),
        np.asarray(starts, dtype=np.int64),
        np.asarray(frac, dtype=np.float32),
    )

def prepare_data(cfg):
    train_raw, test_raw, point_labels = load_ucr_dataset(cfg.data_root, cfg.dataset_id)
    train_std, test_std = robust_standardize(train_raw, test_raw)
    train_x, train_y, train_starts, train_frac = make_windows(train_std, None, cfg.window, cfg.stride_train)
    test_x, test_y, test_starts, test_frac = make_windows(test_std, point_labels, cfg.window, cfg.stride_test)
    return train_x, train_y, train_starts, test_x, test_y, test_starts, test_frac, point_labels

def score_to_numpy(score):
    if torch.is_tensor(score):
        return score.detach().cpu().numpy().astype(np.float64)
    return np.asarray(score, dtype=np.float64)

def standardize_from_calib(calib_score, test_score):
    calib_np = score_to_numpy(calib_score)
    test_np = score_to_numpy(test_score)
    mu = float(np.mean(calib_np))
    sd = max(float(np.std(calib_np)), 1e-6)
    stats = {
        "calib_mean": mu,
        "calib_std": sd,
        "calib_min": float(np.min(calib_np)),
        "calib_max": float(np.max(calib_np)),
    }
    return (calib_np - mu) / sd, (test_np - mu) / sd, stats

def best_f1(y, score) -> float:
    precision, recall, _ = precision_recall_curve(y, score)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(np.nanmax(f1))

def threshold_at_95_recall(y, score) -> float:
    if not np.any(y == 1):
        return np.nan
    return float(np.quantile(score[y == 1], 0.05))

def fp_at_95_recall_mask(y, score, mask) -> float:
    threshold = threshold_at_95_recall(y, score)
    if np.isnan(threshold) or not np.any(mask):
        return np.nan
    return float(np.mean(score[mask] >= threshold))

def window_scores_to_points(scores, starts, n_points: int, window: int, mode: str = "mean"):
    if mode == "max":
        point_scores = np.full(n_points, -np.inf, dtype=np.float64)
        for s, score in zip(starts, scores):
            point_scores[s : s + window] = np.maximum(point_scores[s : s + window], score)
        finite = np.isfinite(point_scores)
        if not finite.all():
            floor = np.nanmin(point_scores[finite]) if finite.any() else 0.0
            point_scores[~finite] = floor
        return point_scores
    point_scores = np.zeros(n_points, dtype=np.float64)
    counts = np.zeros(n_points, dtype=np.float64)
    for s, score in zip(starts, scores):
        point_scores[s : s + window] += score
        counts[s : s + window] += 1
    return point_scores / np.maximum(counts, 1)

def metric_row(model, score_name, y, score, test_starts, point_labels, window, rare_normal_mask=None):
    rare_normal_mask = np.zeros_like(y, dtype=bool) if rare_normal_mask is None else rare_normal_mask
    point_y = point_labels.astype(int)
    point_score_mean = window_scores_to_points(score, test_starts, len(point_labels), window, mode="mean")
    point_score_max = window_scores_to_points(score, test_starts, len(point_labels), window, mode="max")
    row = {
        "Model": model,
        "Score": score_name,
        "window_AUROC": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "window_AUPRC": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "window_best_F1": best_f1(y, score),
        "FP@95R Normal": fp_at_95_recall_mask(y, score, y == 0),
        "FP@95R Rare Normal": fp_at_95_recall_mask(y, score, rare_normal_mask),
        "point_AUROC_mean": roc_auc_score(point_y, point_score_mean) if len(np.unique(point_y)) == 2 else np.nan,
        "point_AUPRC_mean": average_precision_score(point_y, point_score_mean) if len(np.unique(point_y)) == 2 else np.nan,
        "point_best_F1_mean": best_f1(point_y, point_score_mean),
        "point_AUROC_max": roc_auc_score(point_y, point_score_max) if len(np.unique(point_y)) == 2 else np.nan,
        "point_AUPRC_max": average_precision_score(point_y, point_score_max) if len(np.unique(point_y)) == 2 else np.nan,
        "point_best_F1_max": best_f1(point_y, point_score_max),
        "mean score normal": float(np.mean(score[y == 0])) if np.any(y == 0) else np.nan,
        "mean score anomaly": float(np.mean(score[y == 1])) if np.any(y == 1) else np.nan,
    }
    # Legacy aliases preserve existing table code; mean-overlap remains the default point view.
    row["point_AUROC"] = row["point_AUROC_mean"]
    row["point_AUPRC"] = row["point_AUPRC_mean"]
    row["point_best_F1"] = row["point_best_F1_mean"]
    return row

def add_score(
    rows,
    scores_z,
    calibration_rows,
    model,
    score_name,
    calib_score,
    test_score,
    y,
    test_starts,
    point_labels,
    window,
    rare_normal_mask,
):
    _, score, stats = standardize_from_calib(calib_score, test_score)
    key = f"{model} / {score_name}"
    scores_z[key] = score
    calibration_rows.append({"score_key": key, **stats})
    rows.append(metric_row(model, score_name, y, score, test_starts, point_labels, window, rare_normal_mask))
    return score


def build_ucr_catalog(root: Path) -> pd.DataFrame:
    rows = []
    for train_path in sorted(root.glob("*_train.npy")):
        dataset_id = train_path.name.replace("_train.npy", "")
        test_path = root / f"{dataset_id}_test.npy"
        label_path = root / f"{dataset_id}_test_label.npy"
        if not test_path.exists() or not label_path.exists():
            continue
        train = np.load(train_path, mmap_mode="r")
        test = np.load(test_path, mmap_mode="r")
        labels = np.load(label_path, mmap_mode="r")
        rows.append({
            "dataset_id": dataset_id,
            "train_len": int(train.shape[0]),
            "test_len": int(test.shape[0]),
            "channels": int(train.shape[1]) if len(train.shape) > 1 else 1,
            "point_anomalies": int(np.asarray(labels).sum()),
            "point_anomaly_frac": float(np.asarray(labels).mean()),
        })
    return pd.DataFrame(rows)


def contiguous_true_ranges(mask):
    mask = np.asarray(mask).astype(bool)
    if len(mask) == 0:
        return []
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts, ends))
