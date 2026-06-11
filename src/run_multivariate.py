#!/usr/bin/env python3
"""Self-contained multivariate HCFM benchmark runner.

This file is generated from the active code path in ``hcfm_multivariate.ipynb``
and vendored for reviewer execution inside the code submission folder.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

KNOWN_DATASETS = ["MSL", "SMAP", "SMD", "PSM", "NIPS_TS_GECCO", "NIPS_TS_Swan", "NIPS_TS_Creditcard"]


def _parse_seeds(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_cli():
    parser = argparse.ArgumentParser(description="Run the multivariate scalar-potential HCFM benchmark.")
    parser.add_argument("--dataset_id", default="all", help="Dataset id, or 'all' to run all known multivariate datasets.")
    parser.add_argument("--data_root", default=str(SUBMISSION_ROOT / "data" / "multivariate"))
    parser.add_argument("--output_root", default=str(SUBMISSION_ROOT / "outputs" / "multivariate"))
    parser.add_argument("--seeds", default="123", help="Comma-separated integer seeds.")
    parser.add_argument("--train_steps", type=int, default=15000)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--eval_n_probe", type=int, default=2)
    parser.add_argument("--score_batch_size", type=int, default=512)
    parser.add_argument("--hcfm_component_score_batch_size", type=int, default=64)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--print_every", type=int, default=2000)
    parser.add_argument("--metric_n_jobs", type=int, default=4)
    return parser.parse_args()


_CLI_ARGS = _parse_cli()
_CLI_SEEDS = _parse_seeds(_CLI_ARGS.seeds)
if str(_CLI_ARGS.dataset_id).lower() == "all" or len(_CLI_SEEDS) > 1:
    datasets = KNOWN_DATASETS if str(_CLI_ARGS.dataset_id).lower() == "all" else [str(_CLI_ARGS.dataset_id)]
    for dataset in datasets:
        for seed in _CLI_SEEDS:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--dataset_id", dataset,
                "--data_root", _CLI_ARGS.data_root,
                "--output_root", _CLI_ARGS.output_root,
                "--seeds", str(seed),
                "--train_steps", str(_CLI_ARGS.train_steps),
                "--window", str(_CLI_ARGS.window),
                "--stride", str(_CLI_ARGS.stride),
                "--eval_n_probe", str(_CLI_ARGS.eval_n_probe),
                "--score_batch_size", str(_CLI_ARGS.score_batch_size),
                "--hcfm_component_score_batch_size", str(_CLI_ARGS.hcfm_component_score_batch_size),
                "--print_every", str(_CLI_ARGS.print_every),
                "--metric_n_jobs", str(_CLI_ARGS.metric_n_jobs),
            ]
            if _CLI_ARGS.fast_dev_run:
                cmd.append("--fast_dev_run")
            if _CLI_ARGS.use_compile:
                cmd.append("--use_compile")
            print("Running", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
    raise SystemExit(0)
if len(_CLI_SEEDS) != 1:
    raise ValueError("Internal seed fanout expected exactly one seed for this process")
_CLI_SEED = _CLI_SEEDS[0]


# %% notebook cell 2

import gc
import json
import math
import os
import pickle
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn as nn
import torch.nn.functional as F
def display(obj):
    print(obj)
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

try:
    import torchdiffeq
except ImportError:
    torchdiffeq = None

# Allow imports from extracted helper modules whether the kernel starts in
# the notebook directory or one level above it.
if not Path("hcfm_data_metrics_utils.py").exists() and Path("hcfm-prototype/hcfm_data_metrics_utils.py").exists():
    sys.path.insert(0, str(Path("hcfm-prototype").resolve()))

from hcfm_data_metrics_utils import (
    contiguous_true_ranges,
    metric_row,
    score_to_numpy,
    window_scores_to_points,
)
from hcfm_rng_utils import (
    add_config_metadata,
    make_score_generator,
    make_torch_generator,
    sample_probe_like,
    seed_everything,
    seed_worker,
)


# %% notebook cell 4
ACTIVE_EXPERIMENT = "dataspace_cnn_multivariate"

dataspace_cfg = SimpleNamespace(
    # identity / data
    experiment_name="gecco_dataspace_hcfm_multivariate_seed123",
    dataset_id="gecco",
    seed=123,
    deterministic=False,
    fast_dev_run=True,
    score_profile="core",
    diagnostic_profile="none",
    plot_profile="core",
    use_compile=False,
    print_every=2000,
    verbose_train_logs=False,
    print_loss_components=True,
    keep_train_tensors_on_device=True,
    zero_grad_set_to_none=True,
    use_dedicated_fm_generator=True,
    fm_sampling_seed_offset=12345,
    num_workers=0,
    use_same_seed_per_method=True,
    repro_check=False,

    data_root=Path("datasets/multivariate"),
    output_dir=Path("outputs/hcfm_multivariate/gecco_dataspace_hcfm_multivariate_seed123"),

    # data shape, inferred after loading
    window=64,
    stride=1,
    channels=None,
    length=64,
    flat_dim=None,
    path_eps=0.0,

    # multivariate preprocessing / score calibration
    multivar_scaling_mode="standard",
    multivar_use_clipping=True,
    multivar_clip_value=10.0,
    multivar_scale_floor=1e-3,
    multivar_drop_flat_features=True,
    multivar_flat_feature_tol=0.0,
    multivar_audit_binary_features=True,
    multivar_drop_binary_features=False,
    multivar_timestamp_feature_drop_map={"NIPS_TS_GECCO": [0], "gecco": [0]},
    multivar_audit_timestamp_like_features=True,
    multivar_drop_detected_timestamp_like_features=False,
    multivar_timestamp_spearman_threshold=0.95,
    multivar_timestamp_max_unique_frac=0.02,
    score_calibration_mode="robust",
    score_calibration_floor=1e-3,

    # model family switches
    run_raw_mahalanobis=False,
    run_vanilla_fm=False,
    run_fdm_lite=False,
    run_scalar_hcfm=True,
    run_scalar_potential_hcfm=False,
    use_compact_latent=False,
    use_sequence_latent=False,
    run_cnf_likelihood=False,
    run_full_field_hutchinson_scoring=False,

    # shared model settings
    time_emb_type="sinusoidal",
    time_emb_dim=32,
    time_emb_max_period=10000,
    hcfm_time_emb_scale=1000.0,
    hcfm_time_emb_use_2pi=False,
    hcfm_use_film_time_conditioning=True,
    hidden=128,
    base_depth=3,
    hcfm_residual_depth=0,
    hcfm_residual_arch="shallow_cnn",
    hcfm_residual_hidden=64,
    hcfm_residual_kernel_size=3,

    # optimization
    lr=1e-4,
    weight_decay=1e-4,
    grad_clip=5.0,
    vanilla_iters=20000,
    vanilla_batch_size=128,
    fdm_iters=20000,
    fdm_batch_size=128,
    hcfm_iters=15000,
    hcfm_batch_size=256,

    # divergence / Hutchinson
    divergence_estimator="hutchinson",
    hutchinson_n_probe=1,
    train_n_probe=1,
    eval_n_probe=2,
    eval_n_probe_core=2,
    eval_n_probe_full=2,
    probe_type="rademacher",
    hutchinson_probe_type="rademacher",
    add_hp_cycle_scores=False,
    hp_lambda=1600.0,
    stopgrad_div_target=True,

    # metric / scoring flags
    compute_full_likelihood_proxy=False,
    compute_residual_signed_div=False,
    compute_head_divergence_diagnostics=False,
    compute_mechanism_fusions=False,
    compute_fm_consistency=False,
    compute_all_generic_smoothing=False,
    compute_hundman_metrics=True,
    hundman_threshold_quantile=0.99,
    hundman_sd_thresholds=(2.0, 3.0),
    compute_vus_metrics=True,
    metric_n_jobs=max(1, min(8, (os.cpu_count() or 2) - 1)),
    vus_sliding_window=None,
    vus_use_point_scores=True,
    vus_point_projection_modes=("mean", "max"),
    compute_exact_likelihood_scores=True,
    exact_likelihood_include_residual=True,
    compute_trapz_cnf_integral_scores=True,
    trapz_cnf_include_residual=False,
    run_hcfm_residual_signed_div_score=False,
    diagnostic_max_diag_windows_calib=1000,
    diagnostic_max_diag_windows_test=2000,
    diagnostic_exact_jacobian_diag_windows=64,

    # FDM-lite, inactive by default
    fdm_lambda_div=1,

    # HCFM decomposition
    hcfm_component_type="skew_transport_potential_compression_residual",
    hcfm_compression_type="scalar_potential",
    hcfm_transport_type="low_rank_skew",
    hcfm_transport_rank=None,
    hcfm_gamma_compression=1.0,
    hcfm_gamma_residual=0.25,
    hcfm_lambda_compression_energy=0.0,
    hcfm_lambda_residual_energy=5e-4,
    hcfm_lambda_ortho=10.0,
    hcfm_lambda_compression_div=0.0,
    hcfm_residual_warmup_iters=0,
    hcfm_residual_ramp_iters=0,
    hcfm_freeze_residual_during_warmup=False,
    hcfm_combined_alpha=0.25,

    # residual controls
    hcfm_use_residual_mismatch_gate=False,
    hcfm_residual_gate_min=0.1,
    hcfm_residual_gate_threshold=0.5,
    hcfm_residual_gate_power=1.0,
    hcfm_lambda_residual_div=0.0,
    hcfm_use_residual_div_bound=False,
    hcfm_lambda_residual_div_bound=0.0,
    hcfm_div_bound_kappa=0.75,
    hcfm_use_physics_residual_loss=False,
    hcfm_lambda_physics_residual=1e-5,
    hcfm_physics_residual_mode="ratio",
    hcfm_physics_residual_kappa=0.50,
    hcfm_physics_residual_eps=1e-6,
    hcfm_physics_residual_detach_compression=True,
    hcfm_inference_residual_scale_mode="full",

    # scalar-potential compression backbone
    hcfm_potential_backbone="cnn",
    hcfm_potential_norm_type="none",
    hcfm_potential_residual_scale=1.0,
    hcfm_potential_scale=1.0,
    hcfm_potential_out_init="default",
    hcfm_potential_out_init_std=1e-3,
    hcfm_log_head_scale_diagnostics=True,
    hcfm_mixer_depth=3,
    hcfm_mixer_token_mlp_dim=128,
    hcfm_mixer_channel_mlp_dim=128,
    hcfm_mixer_norm_type="none",
    hcfm_mixer_residual_scale=0.10,

    # auxiliary conditioning
    hcfm_use_fft_features=False,
    hcfm_fft_feature_mode="global_bands",
    hcfm_fft_num_bands=8,
    hcfm_fft_include_centroid=True,
    hcfm_fft_include_entropy=True,
    hcfm_fft_include_dominant=True,
    hcfm_fft_log_magnitude=True,
    hcfm_fft_eps=1e-8,
    hcfm_fft_detach_features=True,
    hcfm_normalize_fft_features=True,
    hcfm_scale_aux_channels_by_x_stats=True,
    hcfm_context_focus_scale=10.0,
    hcfm_use_position_embedding=True,
    hcfm_position_emb_dim=32,
    hcfm_position_proj_dim=1,
    hcfm_position_max_period=10000.0,
    hcfm_position_use_integer_positions=True,

    # scoring / calibration
    ode_steps=8,
    ode_method="rk4",
    ode_atol=1e-4,
    ode_rtol=1e-4,
    score_batch_size=512,
    hcfm_component_score_batch_size=64,
    # AE is only a reconstruction baseline. It can be expensive and weak on high-C multivariate windows.
    run_ae_baseline=False,
    ae_hidden=64,
    ae_train_steps=300,
    ae_batch_size=64,
    ae_score_batch_size=512,
    fm_consistency_K=4,
    calibration_fraction=0.10,
    calibration_min=1000,
    calibration_max=5000,
)

def finalize_dataspace_cfg(cfg):
    """
    Validate and derive simple paths/settings for the active data-space run.
    This keeps the top-level config readable and prevents scattered derived
    variables throughout the notebook.
    """
    cfg.data_root = Path(cfg.data_root)
    cfg.output_dir = Path(cfg.output_dir)
    if bool(getattr(cfg, "hcfm_use_fft_features", False)) and not str(cfg.output_dir).endswith("_fft"):
        cfg.experiment_name = f"{cfg.experiment_name}_fft" if not str(cfg.experiment_name).endswith("_fft") else cfg.experiment_name
        cfg.output_dir = cfg.output_dir.with_name(f"{cfg.output_dir.name}_fft")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.score_profile = getattr(cfg, "score_profile", "core")
    cfg.plot_profile = getattr(cfg, "plot_profile", cfg.score_profile)
    cfg.diagnostic_profile = getattr(cfg, "diagnostic_profile", "none")
    assert cfg.score_profile in {"core", "extended", "debug"}
    assert cfg.plot_profile in {"core", "extended", "debug"}
    assert cfg.diagnostic_profile in {"none", "scalar_potential_audit"}
    cfg.run_scalar_potential_audit = bool(cfg.diagnostic_profile == "scalar_potential_audit" or cfg.score_profile in {"extended", "debug"})
    cfg.compute_full_likelihood_proxy = bool(getattr(cfg, "compute_full_likelihood_proxy", False))
    cfg.compute_residual_signed_div = bool(getattr(cfg, "compute_residual_signed_div", False))
    cfg.compute_head_divergence_diagnostics = bool(getattr(cfg, "compute_head_divergence_diagnostics", False))
    cfg.compute_mechanism_fusions = bool(getattr(cfg, "compute_mechanism_fusions", False))
    cfg.compute_fm_consistency = bool(getattr(cfg, "compute_fm_consistency", False))
    cfg.compute_all_generic_smoothing = bool(getattr(cfg, "compute_all_generic_smoothing", False))
    cfg.multivar_scaling_mode = str(getattr(cfg, "multivar_scaling_mode", "standard"))
    assert cfg.multivar_scaling_mode in {"standard", "robust"}
    cfg.multivar_use_clipping = bool(getattr(cfg, "multivar_use_clipping", True))
    cfg.multivar_clip_value = float(getattr(cfg, "multivar_clip_value", 10.0))
    cfg.multivar_scale_floor = float(getattr(cfg, "multivar_scale_floor", 1e-3))
    cfg.multivar_drop_flat_features = bool(getattr(cfg, "multivar_drop_flat_features", True))
    cfg.multivar_flat_feature_tol = float(getattr(cfg, "multivar_flat_feature_tol", 0.0))
    cfg.multivar_audit_binary_features = bool(getattr(cfg, "multivar_audit_binary_features", True))
    cfg.multivar_drop_binary_features = bool(getattr(cfg, "multivar_drop_binary_features", False))
    cfg.multivar_timestamp_feature_drop_map = dict(getattr(cfg, "multivar_timestamp_feature_drop_map", {"NIPS_TS_GECCO": [0], "gecco": [0]}))
    cfg.multivar_audit_timestamp_like_features = bool(getattr(cfg, "multivar_audit_timestamp_like_features", True))
    cfg.multivar_drop_detected_timestamp_like_features = bool(getattr(cfg, "multivar_drop_detected_timestamp_like_features", False))
    cfg.multivar_timestamp_spearman_threshold = float(getattr(cfg, "multivar_timestamp_spearman_threshold", 0.95))
    cfg.multivar_timestamp_max_unique_frac = float(getattr(cfg, "multivar_timestamp_max_unique_frac", 0.02))
    cfg.score_calibration_mode = str(getattr(cfg, "score_calibration_mode", "robust"))
    assert cfg.score_calibration_mode in {"zscore", "robust"}
    cfg.score_calibration_floor = float(getattr(cfg, "score_calibration_floor", 1e-3))
    assert cfg.multivar_clip_value > 0
    assert cfg.multivar_scale_floor > 0
    assert cfg.multivar_flat_feature_tol >= 0.0
    assert 0.0 < cfg.multivar_timestamp_spearman_threshold <= 1.0
    assert 0.0 < cfg.multivar_timestamp_max_unique_frac <= 1.0
    assert cfg.score_calibration_floor > 0
    cfg.compute_hundman_metrics = bool(getattr(cfg, "compute_hundman_metrics", True))
    cfg.hundman_threshold_quantile = float(getattr(cfg, "hundman_threshold_quantile", 0.99))
    cfg.hundman_sd_thresholds = tuple(float(x) for x in getattr(cfg, "hundman_sd_thresholds", (2.0, 3.0)))
    cfg.compute_vus_metrics = bool(getattr(cfg, "compute_vus_metrics", True))
    cfg.metric_n_jobs = int(getattr(cfg, "metric_n_jobs", max(1, min(8, (os.cpu_count() or 2) - 1))))
    cfg.vus_sliding_window = getattr(cfg, "vus_sliding_window", None)
    if cfg.vus_sliding_window is not None:
        cfg.vus_sliding_window = int(cfg.vus_sliding_window)
        assert cfg.vus_sliding_window >= 1
    cfg.vus_use_point_scores = bool(getattr(cfg, "vus_use_point_scores", True))
    cfg.vus_point_projection_modes = tuple(getattr(cfg, "vus_point_projection_modes", ("mean", "max")))
    assert all(mode in {"mean", "max"} for mode in cfg.vus_point_projection_modes)
    cfg.compute_exact_likelihood_scores = bool(getattr(cfg, "compute_exact_likelihood_scores", True))
    cfg.exact_likelihood_include_residual = bool(getattr(cfg, "exact_likelihood_include_residual", True))
    cfg.compute_trapz_cnf_integral_scores = bool(getattr(cfg, "compute_trapz_cnf_integral_scores", True))
    cfg.trapz_cnf_include_residual = bool(getattr(cfg, "trapz_cnf_include_residual", False))
    if cfg.score_profile == "extended":
        cfg.compute_full_likelihood_proxy = True
        cfg.compute_residual_signed_div = True
    elif cfg.score_profile == "debug":
        cfg.compute_full_likelihood_proxy = True
        cfg.compute_residual_signed_div = True
        cfg.compute_head_divergence_diagnostics = True
        cfg.compute_mechanism_fusions = True
        cfg.compute_fm_consistency = True
        cfg.compute_all_generic_smoothing = True
    if cfg.compute_exact_likelihood_scores:
        print("Mean-divergence likelihood-style rows are derived algebraically from existing scores; no extra scoring pass.")
    if getattr(cfg, "run_scalar_potential_audit", False):
        cfg.compute_full_likelihood_proxy = True
        cfg.compute_residual_signed_div = True
        cfg.compute_head_divergence_diagnostics = True
        cfg.compute_fm_consistency = True
        cfg.compute_mechanism_fusions = False
        cfg.compute_all_generic_smoothing = False
        cfg.add_hp_cycle_scores = False
    cfg.eval_n_probe_core = int(getattr(cfg, "eval_n_probe_core", 2))
    cfg.eval_n_probe_full = int(getattr(cfg, "eval_n_probe_full", 4))
    cfg.eval_n_probe = cfg.eval_n_probe_core if cfg.score_profile == "core" else cfg.eval_n_probe_full
    if cfg.channels is not None and cfg.flat_dim is not None:
        cfg.channels = int(cfg.channels)
        cfg.length = int(cfg.length)
        cfg.flat_dim = int(cfg.flat_dim)
        cfg.data_shape = (cfg.channels, cfg.length)
    else:
        cfg.data_shape = None
    cfg.fast_dev_run = getattr(cfg, "fast_dev_run", False)
    if cfg.fast_dev_run:
        cfg.deterministic = False
        cfg.eval_n_probe = min(getattr(cfg, "eval_n_probe", 4), 2)
        print("WARNING: fast_dev_run=True reduces eval_n_probe and is not final-report quality.")
    cfg.use_compile = getattr(cfg, "use_compile", False)
    cfg.print_every = getattr(cfg, "print_every", 500)
    cfg.plot_every = getattr(cfg, "plot_every", 0)
    cfg.verbose_train_logs = getattr(cfg, "verbose_train_logs", False)
    cfg.print_loss_components = getattr(cfg, "print_loss_components", False)
    cfg.keep_train_tensors_on_device = getattr(cfg, "keep_train_tensors_on_device", True)
    cfg.zero_grad_set_to_none = getattr(cfg, "zero_grad_set_to_none", True)
    cfg.use_dedicated_fm_generator = getattr(cfg, "use_dedicated_fm_generator", True)
    cfg.fm_sampling_seed_offset = getattr(cfg, "fm_sampling_seed_offset", 12345)
    cfg.num_workers = getattr(cfg, "num_workers", 0)
    cfg.train_n_probe = getattr(cfg, "train_n_probe", getattr(cfg, "hutchinson_n_probe", 1))
    cfg.eval_n_probe = int(getattr(cfg, "eval_n_probe", cfg.eval_n_probe_core if cfg.score_profile == "core" else cfg.eval_n_probe_full))
    if getattr(cfg, "run_scalar_potential_audit", False):
        cfg.eval_n_probe = max(cfg.eval_n_probe, 4)
    cfg.probe_type = getattr(cfg, "probe_type", getattr(cfg, "hutchinson_probe_type", "rademacher"))
    cfg.run_hcfm_head_divergence_diagnostics = bool(cfg.compute_head_divergence_diagnostics)
    cfg.run_hcfm_residual_signed_div_score = bool(cfg.compute_residual_signed_div or cfg.compute_full_likelihood_proxy or cfg.compute_head_divergence_diagnostics)
    cfg.hutchinson_probe_type = getattr(cfg, "hutchinson_probe_type", cfg.probe_type)
    cfg.add_hp_cycle_scores = getattr(cfg, "add_hp_cycle_scores", False)
    cfg.hp_lambda = getattr(cfg, "hp_lambda", 1600.0)
    cfg.ode_atol = float(getattr(cfg, "ode_atol", 1e-4))
    cfg.ode_rtol = float(getattr(cfg, "ode_rtol", 1e-4))
    cfg.hcfm_time_emb_scale = float(getattr(cfg, "hcfm_time_emb_scale", 1000.0))
    cfg.hcfm_time_emb_use_2pi = bool(getattr(cfg, "hcfm_time_emb_use_2pi", False))
    cfg.hcfm_use_film_time_conditioning = bool(getattr(cfg, "hcfm_use_film_time_conditioning", True))
    cfg.hcfm_residual_depth = getattr(cfg, "hcfm_residual_depth", cfg.base_depth)
    cfg.hcfm_residual_arch = getattr(cfg, "hcfm_residual_arch", "standard")
    cfg.hcfm_residual_hidden = int(getattr(cfg, "hcfm_residual_hidden", 16))
    cfg.hcfm_residual_kernel_size = int(getattr(cfg, "hcfm_residual_kernel_size", 3))
    cfg.hcfm_residual_warmup_iters = int(getattr(cfg, "hcfm_residual_warmup_iters", 0))
    cfg.hcfm_residual_ramp_iters = int(getattr(cfg, "hcfm_residual_ramp_iters", 0))
    cfg.hcfm_freeze_residual_during_warmup = bool(getattr(cfg, "hcfm_freeze_residual_during_warmup", False))
    cfg.hcfm_compression_type = getattr(cfg, "hcfm_compression_type", "scalar_potential")
    cfg.hcfm_use_residual_mismatch_gate = bool(getattr(cfg, "hcfm_use_residual_mismatch_gate", True))
    cfg.hcfm_residual_gate_min = float(getattr(cfg, "hcfm_residual_gate_min", 0.10))
    cfg.hcfm_residual_gate_threshold = float(getattr(cfg, "hcfm_residual_gate_threshold", 0.0))
    cfg.hcfm_residual_gate_power = float(getattr(cfg, "hcfm_residual_gate_power", 1.0))
    cfg.hcfm_lambda_residual_div = float(getattr(cfg, "hcfm_lambda_residual_div", 0.0))
    cfg.hcfm_use_residual_div_bound = bool(getattr(cfg, "hcfm_use_residual_div_bound", False))
    cfg.hcfm_lambda_residual_div_bound = float(getattr(cfg, "hcfm_lambda_residual_div_bound", 0.0))
    cfg.hcfm_div_bound_kappa = float(getattr(cfg, "hcfm_div_bound_kappa", 0.75))
    cfg.hcfm_use_physics_residual_loss = bool(getattr(cfg, "hcfm_use_physics_residual_loss", True))
    cfg.hcfm_lambda_physics_residual = float(getattr(cfg, "hcfm_lambda_physics_residual", 1e-5))
    cfg.hcfm_physics_residual_mode = getattr(cfg, "hcfm_physics_residual_mode", "ratio")
    cfg.hcfm_physics_residual_kappa = float(getattr(cfg, "hcfm_physics_residual_kappa", 0.50))
    cfg.hcfm_physics_residual_eps = float(getattr(cfg, "hcfm_physics_residual_eps", 1e-6))
    cfg.hcfm_physics_residual_detach_compression = bool(getattr(cfg, "hcfm_physics_residual_detach_compression", True))
    cfg.hcfm_inference_residual_scale_mode = getattr(cfg, "hcfm_inference_residual_scale_mode", "floor")
    cfg.hcfm_potential_backbone = getattr(cfg, "hcfm_potential_backbone", "cnn")
    cfg.hcfm_use_fft_features = bool(getattr(cfg, "hcfm_use_fft_features", False))
    cfg.hcfm_fft_feature_mode = getattr(cfg, "hcfm_fft_feature_mode", "global_bands")
    cfg.hcfm_fft_num_bands = int(getattr(cfg, "hcfm_fft_num_bands", 4))
    cfg.hcfm_fft_include_centroid = bool(getattr(cfg, "hcfm_fft_include_centroid", True))
    cfg.hcfm_fft_include_entropy = bool(getattr(cfg, "hcfm_fft_include_entropy", True))
    cfg.hcfm_fft_include_dominant = bool(getattr(cfg, "hcfm_fft_include_dominant", True))
    cfg.hcfm_fft_log_magnitude = bool(getattr(cfg, "hcfm_fft_log_magnitude", True))
    cfg.hcfm_fft_eps = float(getattr(cfg, "hcfm_fft_eps", 1e-8))
    cfg.hcfm_fft_detach_features = bool(getattr(cfg, "hcfm_fft_detach_features", False))
    cfg.hcfm_normalize_fft_features = bool(getattr(cfg, "hcfm_normalize_fft_features", True))
    cfg.hcfm_scale_aux_channels_by_x_stats = bool(getattr(cfg, "hcfm_scale_aux_channels_by_x_stats", True))
    cfg.hcfm_context_focus_scale = float(getattr(cfg, "hcfm_context_focus_scale", 10.0))
    cfg.hcfm_mixer_depth = int(getattr(cfg, "hcfm_mixer_depth", 3))
    cfg.hcfm_mixer_token_mlp_dim = int(getattr(cfg, "hcfm_mixer_token_mlp_dim", 128))
    cfg.hcfm_mixer_channel_mlp_dim = int(getattr(cfg, "hcfm_mixer_channel_mlp_dim", 128))
    cfg.hcfm_mixer_norm_type = getattr(cfg, "hcfm_mixer_norm_type", "none")
    cfg.hcfm_mixer_residual_scale = float(getattr(cfg, "hcfm_mixer_residual_scale", 0.10))
    cfg.hcfm_potential_norm_type = getattr(cfg, "hcfm_potential_norm_type", "none")
    cfg.hcfm_potential_residual_scale = float(getattr(cfg, "hcfm_potential_residual_scale", 0.10))
    cfg.hcfm_potential_scale = float(getattr(cfg, "hcfm_potential_scale", 0.10))
    cfg.hcfm_potential_out_init = getattr(cfg, "hcfm_potential_out_init", "small")
    cfg.hcfm_potential_out_init_std = float(getattr(cfg, "hcfm_potential_out_init_std", 1e-3))
    cfg.hcfm_log_head_scale_diagnostics = bool(getattr(cfg, "hcfm_log_head_scale_diagnostics", True))
    cfg.hcfm_use_position_embedding = bool(getattr(cfg, "hcfm_use_position_embedding", True))
    cfg.hcfm_position_emb_dim = int(getattr(cfg, "hcfm_position_emb_dim", 16))
    cfg.hcfm_position_proj_dim = int(getattr(cfg, "hcfm_position_proj_dim", 4))
    cfg.hcfm_position_max_period = float(getattr(cfg, "hcfm_position_max_period", 10000.0))
    cfg.hcfm_position_use_integer_positions = bool(getattr(cfg, "hcfm_position_use_integer_positions", True))
    cfg.max_diag_windows_calib = int(getattr(cfg, "max_diag_windows_calib", getattr(cfg, "diagnostic_max_diag_windows_calib", 1000)))
    cfg.max_diag_windows_test = int(getattr(cfg, "max_diag_windows_test", getattr(cfg, "diagnostic_max_diag_windows_test", 2000)))
    cfg.exact_jacobian_diag_windows = int(getattr(cfg, "exact_jacobian_diag_windows", getattr(cfg, "diagnostic_exact_jacobian_diag_windows", 64)))
    cfg.seed = cfg.seed

    assert ACTIVE_EXPERIMENT == "dataspace_cnn_multivariate"
    assert cfg.window == cfg.length
    if cfg.channels is not None and cfg.flat_dim is not None:
        assert cfg.flat_dim == cfg.channels * cfg.length
        assert cfg.data_shape == (cfg.channels, cfg.length)
    assert cfg.time_emb_type == "sinusoidal"
    assert cfg.hcfm_time_emb_scale > 0
    assert cfg.divergence_estimator == "hutchinson"
    assert cfg.hutchinson_n_probe >= 1
    assert cfg.train_n_probe >= 1
    assert cfg.eval_n_probe >= 1
    assert cfg.ode_atol > 0
    assert cfg.ode_rtol > 0
    assert 0.0 < cfg.hundman_threshold_quantile < 1.0
    assert all(k > 0 for k in cfg.hundman_sd_thresholds)
    assert cfg.metric_n_jobs >= 1
    assert cfg.max_diag_windows_calib >= 1
    assert cfg.max_diag_windows_test >= 1
    assert 1 <= cfg.exact_jacobian_diag_windows <= 64
    assert cfg.probe_type in {"rademacher", "gaussian"}
    assert cfg.hutchinson_probe_type in {"rademacher", "gaussian"}
    assert not cfg.use_compact_latent
    assert not cfg.use_sequence_latent
    assert not cfg.run_cnf_likelihood
    assert not cfg.run_scalar_potential_hcfm
    assert cfg.hcfm_component_type in {"skew_transport_compression_residual", "skew_transport_potential_compression_residual"}
    assert cfg.hcfm_compression_type in {"scalar_potential", "cnn_vector"}
    assert cfg.hcfm_transport_type == "low_rank_skew"
    assert cfg.hcfm_residual_depth >= 0
    assert cfg.hcfm_residual_arch in {"standard", "shallow_cnn"}
    assert cfg.hcfm_residual_hidden >= 1
    assert cfg.hcfm_residual_kernel_size in {1, 3, 5, 7}
    assert cfg.hcfm_residual_warmup_iters >= 0
    assert cfg.hcfm_residual_ramp_iters >= 0
    assert 0.0 <= cfg.hcfm_residual_gate_min <= 1.0
    assert 0.0 <= cfg.hcfm_residual_gate_threshold < 1.0
    assert cfg.hcfm_residual_gate_power > 0.0
    assert cfg.hcfm_inference_residual_scale_mode in {"floor", "full"}
    assert cfg.hcfm_potential_backbone in {"cnn", "mixer"}
    assert cfg.hcfm_fft_feature_mode in {"global_bands"}
    assert cfg.hcfm_fft_num_bands >= 1
    assert cfg.hcfm_fft_eps > 0
    assert cfg.hcfm_context_focus_scale > 0
    if cfg.hcfm_fft_detach_features:
        print("WARNING: hcfm_fft_detach_features=True breaks spectral gradients and is for ablation only.")
    assert cfg.hcfm_mixer_depth >= 1
    assert cfg.hcfm_mixer_token_mlp_dim >= 1
    assert cfg.hcfm_mixer_channel_mlp_dim >= 1
    assert cfg.hcfm_mixer_norm_type in {"none", "layernorm"}
    assert cfg.hcfm_mixer_residual_scale > 0
    assert cfg.hcfm_potential_norm_type in {"groupnorm", "none"}
    assert cfg.hcfm_potential_residual_scale > 0
    assert cfg.hcfm_potential_scale > 0
    assert cfg.hcfm_potential_out_init in {"default", "small", "zero"}
    assert cfg.hcfm_potential_out_init_std > 0
    assert cfg.hcfm_position_emb_dim > 0
    assert cfg.hcfm_position_proj_dim > 0
    assert cfg.hcfm_position_max_period > 0
    assert cfg.hcfm_div_bound_kappa >= 0.0
    assert cfg.hcfm_physics_residual_mode in {"ratio", "hinge"}
    assert cfg.hcfm_lambda_physics_residual >= 0.0
    assert cfg.hcfm_physics_residual_kappa >= 0.0
    assert cfg.hcfm_physics_residual_eps > 0.0

    return cfg


def print_dataspace_config_summary(cfg):
    print("=== Active experiment ===")
    print(f"name: {cfg.experiment_name}")
    print(f"dataset: {cfg.dataset_id}")
    if hasattr(cfg, "train_raw_shape") and hasattr(cfg, "test_raw_shape"):
        print(f"raw shapes: train={cfg.train_raw_shape}, test={cfg.test_raw_shape}")
    print(f"multivar scaling: mode={cfg.multivar_scaling_mode}, clipping={cfg.multivar_use_clipping}, clip_value={cfg.multivar_clip_value}, scale_floor={cfg.multivar_scale_floor}")
    print(f"score calibration: mode={cfg.score_calibration_mode}, floor={cfg.score_calibration_floor}")
    print(f"flat/binary feature handling: drop_flat={cfg.multivar_drop_flat_features}, flat_tol={cfg.multivar_flat_feature_tol}, audit_binary={cfg.multivar_audit_binary_features}, drop_binary={cfg.multivar_drop_binary_features}")
    print(f"timestamp-like feature handling: explicit_drop_map={cfg.multivar_timestamp_feature_drop_map}, audit={cfg.multivar_audit_timestamp_like_features}, drop_detected={cfg.multivar_drop_detected_timestamp_like_features}, spearman_threshold={cfg.multivar_timestamp_spearman_threshold}, max_unique_frac={cfg.multivar_timestamp_max_unique_frac}")
    print(f"output_dir: {cfg.output_dir}")
    print(f"channels={cfg.channels}, window={cfg.length}, flat_dim={cfg.flat_dim}, stride={cfg.stride}")
    print(f"models: vanilla={cfg.run_vanilla_fm}, fdm={cfg.run_fdm_lite}, data_hcfm={cfg.run_scalar_hcfm}")
    print(f"time embedding: {cfg.time_emb_type}, dim={cfg.time_emb_dim}")
    print(f"divergence: {cfg.divergence_estimator}, train_n_probe={cfg.train_n_probe}, eval_n_probe={cfg.eval_n_probe}, probe={cfg.probe_type}")
    print(f"HCFM component score batch size: {cfg.hcfm_component_score_batch_size}")
    print(f"AE baseline: run={cfg.run_ae_baseline}, hidden={cfg.ae_hidden}, steps={cfg.ae_train_steps}, batch={cfg.ae_batch_size}, score_batch={cfg.ae_score_batch_size}")
    print(f"time conditioning: scale={cfg.hcfm_time_emb_scale}, use_2pi={cfg.hcfm_time_emb_use_2pi}, film={cfg.hcfm_use_film_time_conditioning}")
    print(f"score_profile={cfg.score_profile}, plot_profile={cfg.plot_profile}, diagnostic_profile={cfg.diagnostic_profile}")
    print(f"run_scalar_potential_audit={getattr(cfg, 'run_scalar_potential_audit', False)}, diag_windows calib/test/exact={cfg.max_diag_windows_calib}/{cfg.max_diag_windows_test}/{cfg.exact_jacobian_diag_windows}")
    print(f"score flags: full_likelihood_proxy={cfg.compute_full_likelihood_proxy}, residual_signed_div={cfg.compute_residual_signed_div}, head_div_debug={cfg.compute_head_divergence_diagnostics}, fm_consistency={cfg.compute_fm_consistency}, mechanism_fusions={cfg.compute_mechanism_fusions}, all_smoothing={cfg.compute_all_generic_smoothing}")
    print(f"hundman metrics: compute={cfg.compute_hundman_metrics}, q={cfg.hundman_threshold_quantile}, sd_thresholds={cfg.hundman_sd_thresholds}")
    print(f"VUS metrics: compute={cfg.compute_vus_metrics}, sliding_window={cfg.vus_sliding_window}, use_point_scores={cfg.vus_use_point_scores}, point_modes={cfg.vus_point_projection_modes}")
    print(f"metric parallelism: metric_n_jobs={cfg.metric_n_jobs}")
    print(f"exact likelihood scores: compute={cfg.compute_exact_likelihood_scores}, include_residual={cfg.exact_likelihood_include_residual}")
    print(f"trapz CNF integral scores: compute={cfg.compute_trapz_cnf_integral_scores}, include_residual={cfg.trapz_cnf_include_residual}")
    print(f"HCFM head divergence diagnostics: {cfg.run_hcfm_head_divergence_diagnostics}")
    print(f"HCFM residual signed divergence score: {cfg.run_hcfm_residual_signed_div_score}")
    print(f"iters: vanilla={cfg.vanilla_iters}, fdm={cfg.fdm_iters}, hcfm={cfg.hcfm_iters}")
    print(f"batches: vanilla={cfg.vanilla_batch_size}, fdm={cfg.fdm_batch_size}, hcfm={cfg.hcfm_batch_size}")
    print(f"hcfm: {cfg.hcfm_component_type}, compression={cfg.hcfm_compression_type}, transport={cfg.hcfm_transport_type}, rank={cfg.hcfm_transport_rank}, residual_depth={cfg.hcfm_residual_depth}")
    print(f"hcfm residual: arch={cfg.hcfm_residual_arch}, hidden={cfg.hcfm_residual_hidden}, kernel={cfg.hcfm_residual_kernel_size}, warmup={cfg.hcfm_residual_warmup_iters}, ramp={cfg.hcfm_residual_ramp_iters}, freeze_warmup={cfg.hcfm_freeze_residual_during_warmup}, gamma={cfg.hcfm_gamma_residual}")
    print(f"hcfm gate: use={cfg.hcfm_use_residual_mismatch_gate}, gate_min={cfg.hcfm_residual_gate_min}, gate_threshold={cfg.hcfm_residual_gate_threshold}, gate_power={cfg.hcfm_residual_gate_power}, inference_scale_mode={cfg.hcfm_inference_residual_scale_mode}")
    print(f"hcfm potential scale: backbone={cfg.hcfm_potential_backbone}, norm={cfg.hcfm_potential_norm_type}, residual_scale={cfg.hcfm_potential_residual_scale}, potential_scale={cfg.hcfm_potential_scale}, out_init={cfg.hcfm_potential_out_init}, out_init_std={cfg.hcfm_potential_out_init_std}")
    print(f"hcfm mixer: depth={cfg.hcfm_mixer_depth}, token_mlp_dim={cfg.hcfm_mixer_token_mlp_dim}, channel_mlp_dim={cfg.hcfm_mixer_channel_mlp_dim}, norm={cfg.hcfm_mixer_norm_type}, residual_scale={cfg.hcfm_mixer_residual_scale}")
    print(f"hcfm FFT features: use={cfg.hcfm_use_fft_features}, mode={cfg.hcfm_fft_feature_mode}, bands={cfg.hcfm_fft_num_bands}, centroid={cfg.hcfm_fft_include_centroid}, entropy={cfg.hcfm_fft_include_entropy}, dominant={cfg.hcfm_fft_include_dominant}, log_mag={cfg.hcfm_fft_log_magnitude}, detach={cfg.hcfm_fft_detach_features}, normalize={cfg.hcfm_normalize_fft_features}, aux_scale_by_x={cfg.hcfm_scale_aux_channels_by_x_stats}, focus_scale={cfg.hcfm_context_focus_scale}")
    print(f"hcfm position embedding: use={cfg.hcfm_use_position_embedding}, emb_dim={cfg.hcfm_position_emb_dim}, proj_dim={cfg.hcfm_position_proj_dim}, max_period={cfg.hcfm_position_max_period}, integer_positions={cfg.hcfm_position_use_integer_positions}")
    print(f"hcfm residual div: lambda={cfg.hcfm_lambda_residual_div}, bound={cfg.hcfm_use_residual_div_bound}, bound_lambda={cfg.hcfm_lambda_residual_div_bound}, kappa={cfg.hcfm_div_bound_kappa}")
    print(f"hcfm physics residual: use={cfg.hcfm_use_physics_residual_loss}, lambda={cfg.hcfm_lambda_physics_residual}, mode={cfg.hcfm_physics_residual_mode}, kappa={cfg.hcfm_physics_residual_kappa}, eps={cfg.hcfm_physics_residual_eps}, detach_compression={cfg.hcfm_physics_residual_detach_compression}")
    print(f"scoring: ode_steps={cfg.ode_steps}, ode_method={cfg.ode_method}, ode_atol={cfg.ode_atol}, ode_rtol={cfg.ode_rtol}, fm_consistency_K={cfg.fm_consistency_K}")
    print(f"speed: use_compile={cfg.use_compile}, fast_dev_run={cfg.fast_dev_run}, print_every={cfg.print_every}, plot_every={cfg.plot_every}")



# Device and seed are available before data loading; final config validation happens
# after multivariate channels/flat_dim are inferred from the loaded dataset.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(dataspace_cfg.seed, deterministic=dataspace_cfg.deterministic)
print("Prepared multivariate config stub; dimensions will be inferred after loading data.")
print("device:", device)

# -----------------------------------------------------------------------------
# CLI overrides for script execution.
# -----------------------------------------------------------------------------
dataspace_cfg.dataset_id = str(_CLI_ARGS.dataset_id)
dataspace_cfg.seed = int(_CLI_SEED)
dataspace_cfg.data_root = Path(_CLI_ARGS.data_root)
dataspace_cfg.output_dir = Path(_CLI_ARGS.output_root) / f"{dataspace_cfg.dataset_id.lower()}_dataspace_hcfm_multivariate_seed{dataspace_cfg.seed}"
dataspace_cfg.experiment_name = dataspace_cfg.output_dir.name
dataspace_cfg.window = int(_CLI_ARGS.window)
dataspace_cfg.length = int(_CLI_ARGS.window)
dataspace_cfg.stride = int(_CLI_ARGS.stride)
dataspace_cfg.hcfm_iters = int(_CLI_ARGS.train_steps)
dataspace_cfg.eval_n_probe = int(_CLI_ARGS.eval_n_probe)
dataspace_cfg.eval_n_probe_core = int(_CLI_ARGS.eval_n_probe)
dataspace_cfg.eval_n_probe_full = int(_CLI_ARGS.eval_n_probe)
dataspace_cfg.score_batch_size = int(_CLI_ARGS.score_batch_size)
dataspace_cfg.hcfm_component_score_batch_size = int(_CLI_ARGS.hcfm_component_score_batch_size)
dataspace_cfg.fast_dev_run = bool(_CLI_ARGS.fast_dev_run)
dataspace_cfg.use_compile = bool(_CLI_ARGS.use_compile)
dataspace_cfg.print_every = int(_CLI_ARGS.print_every)
dataspace_cfg.metric_n_jobs = int(_CLI_ARGS.metric_n_jobs)
dataspace_cfg.plot_profile = "core"
dataspace_cfg.diagnostic_profile = "none"
dataspace_cfg.verbose_train_logs = False
dataspace_cfg.output_dir.mkdir(parents=True, exist_ok=True)


# %% notebook cell 6
# Data loading and metric helpers live in hcfm_data_metrics_utils.py.
# RNG/reproducibility helpers live in hcfm_rng_utils.py.


# %% notebook cell 8
# Multivariate datasets are loaded by `load_multivariate_dataset` in the next cell.
dataset_dir = dataspace_cfg.data_root / dataspace_cfg.dataset_id
print("expected dataset directory:", dataset_dir)
if dataset_dir.exists():
    display(pd.DataFrame({"file": [p.name for p in sorted(dataset_dir.iterdir())]}).head(5))
else:
    print("Directory does not exist yet; the loader cell will raise a clear FileNotFoundError.")


# %% notebook cell 10
def _as_2d_time_channel(arr, name="array"):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D after squeeze; got shape {arr.shape}")
    # Common mistake/layout: [C, T]. If the first dimension is much smaller,
    # treat it as channels and transpose to [T, C].
    if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 512:
        arr = arr.T
    return arr.astype(np.float32, copy=False)


def _as_1d_labels(arr, expected_len=None, name="labels"):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        if 1 in arr.shape:
            arr = arr.reshape(-1)
        else:
            axis = 0 if arr.shape[0] == expected_len else 1 if arr.shape[1] == expected_len else None
            if axis is not None:
                candidates = [arr[:, j] for j in range(arr.shape[1])] if axis == 0 else [arr[j, :] for j in range(arr.shape[0])]
                binary_like = []
                for candidate in candidates:
                    vals = pd.Series(candidate).dropna().astype(float).unique()
                    vals = np.asarray(vals)
                    if len(vals) <= 4 and np.all(np.isin(vals, [-1.0, 0.0, 1.0])):
                        binary_like.append(candidate)
                if binary_like:
                    arr = binary_like[-1]
                else:
                    arr = arr.max(axis=1) if axis == 0 else arr.max(axis=0)
            else:
                arr = arr.max(axis=1) if arr.shape[0] >= arr.shape[1] else arr.max(axis=0)
    if arr.ndim != 1:
        raise ValueError(f"{name} must reduce to 1D labels; got shape {arr.shape}")
    labels = (arr.astype(float) > 0).astype(np.int64)
    if expected_len is not None and len(labels) != expected_len:
        raise ValueError(f"{name} length {len(labels)} does not match expected test length {expected_len}")
    return labels


def _label_columns_from_dataframe(df: pd.DataFrame):
    label_tokens = ["label", "labels", "test_label", "test_labels", "anomaly", "is_anomaly", "attack", "normal/attack"]
    lower_to_col = {str(col).strip().lower(): col for col in df.columns}
    exact = [lower_to_col[token] for token in label_tokens if token in lower_to_col]
    if exact:
        return exact
    fuzzy = []
    for col in df.columns:
        name = str(col).strip().lower()
        if "label" in name or "attack" in name or "anomaly" in name:
            fuzzy.append(col)
    return fuzzy


def _load_csv_label_column(path: Path):
    df = pd.read_csv(path)
    label_cols = _label_columns_from_dataframe(df)
    if not label_cols:
        raise ValueError(f"No embedded label column found in {path}")
    return df[label_cols[-1]].to_numpy()


def _load_array_file(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        for key in ["arr_0", "data", "train", "test", "x", "values", "labels", "label"]:
            if key in data:
                return data[key]
        return data[list(data.files)[0]]
    if suffix == ".csv":
        df = pd.read_csv(path)
        label_cols = _label_columns_from_dataframe(df)
        if ("label" in path.stem.lower() or "labels" in path.stem.lower()) and label_cols:
            return df[label_cols[-1]].to_numpy()
        drop_cols = [
            col for col in df.columns
            if any(token in str(col).strip().lower() for token in ["timestamp", "date", "datetime"])
        ]
        drop_cols += [col for col in label_cols if col not in drop_cols]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        return df.values
    if suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            for key in ["data", "values", "x", "train", "test", "labels", "label"]:
                if key in obj:
                    return obj[key]
        return obj
    raise ValueError(f"Unsupported file type: {path}")


def _find_first_existing(base: Path, names):
    for name in names:
        path = base / name
        if path.exists():
            return path
    return None


def drop_flat_and_optionally_binary_features(train_raw, test_raw, cfg, dataset_id=""):
    tol = float(getattr(cfg, "multivar_flat_feature_tol", 0.0))
    drop_flat = bool(getattr(cfg, "multivar_drop_flat_features", True))
    audit_binary = bool(getattr(cfg, "multivar_audit_binary_features", True))
    drop_binary = bool(getattr(cfg, "multivar_drop_binary_features", False))
    diagnostics = []
    drop_idx = []
    for j in range(train_raw.shape[1]):
        tr = np.asarray(train_raw[:, j], dtype=float)
        te = np.asarray(test_raw[:, j], dtype=float)
        both = np.concatenate([tr[np.isfinite(tr)], te[np.isfinite(te)]])
        if both.size == 0:
            is_flat_both = True
            is_binary_like = False
            unique_count = 0
        else:
            unique_vals = np.unique(both)
            unique_count = int(len(unique_vals))
            is_flat_both = bool(np.nanmax(both) - np.nanmin(both) <= tol)
            is_binary_like = bool(unique_count <= 2 and np.all(np.isin(unique_vals.astype(float), [0.0, 1.0, 2.0])))
        reason = None
        if is_flat_both and drop_flat:
            reason = "flat_both"
            drop_idx.append(j)
        elif is_binary_like and drop_binary:
            reason = "binary_like"
            drop_idx.append(j)
        if is_flat_both or (audit_binary and is_binary_like):
            diagnostics.append({
                "channel": j,
                "unique_count_train_test": unique_count,
                "flat_both": is_flat_both,
                "binary_like": is_binary_like,
                "drop_reason": reason or "audit_only",
            })
    if diagnostics:
        print(f"Flat/binary feature audit for {dataset_id}: {diagnostics}")
    if drop_idx:
        drop_idx = sorted(set(drop_idx))
        keep = [j for j in range(train_raw.shape[1]) if j not in set(drop_idx)]
        print(f"Dropping flat/binary feature columns for {dataset_id}: {drop_idx}")
        return train_raw[:, keep], test_raw[:, keep], diagnostics
    return train_raw, test_raw, diagnostics


def _rank_correlation_with_index(x):
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() < 3 or np.nanstd(x[finite]) <= 0:
        return np.nan
    idx = np.arange(len(x), dtype=float)[finite]
    vals = x[finite]
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(vals), dtype=float)
    if np.std(ranks) <= 0 or np.std(idx) <= 0:
        return np.nan
    return float(np.corrcoef(idx, ranks)[0, 1])


def drop_timestamp_like_features(train_raw, test_raw, cfg, dataset_id=""):
    drop_map = dict(getattr(cfg, "multivar_timestamp_feature_drop_map", {}))
    dataset_keys = {str(dataset_id), str(dataset_id).lower(), str(getattr(cfg, "dataset_id", "")), str(getattr(cfg, "dataset_id", "")).lower()}
    explicit_drop = []
    for key, cols in drop_map.items():
        if str(key) in dataset_keys or str(key).lower() in dataset_keys:
            explicit_drop.extend([int(c) for c in cols])
    explicit_drop = sorted(set(c for c in explicit_drop if 0 <= c < train_raw.shape[1]))

    threshold = float(getattr(cfg, "multivar_timestamp_spearman_threshold", 0.95))
    max_unique_frac = float(getattr(cfg, "multivar_timestamp_max_unique_frac", 0.02))
    audit_enabled = bool(getattr(cfg, "multivar_audit_timestamp_like_features", True))
    drop_detected = bool(getattr(cfg, "multivar_drop_detected_timestamp_like_features", False))
    diagnostics = []
    detected_drop = []
    if audit_enabled or drop_detected:
        for j in range(train_raw.shape[1]):
            tr = np.asarray(train_raw[:, j], dtype=float)
            te = np.asarray(test_raw[:, j], dtype=float)
            tr_s = _rank_correlation_with_index(tr)
            te_s = _rank_correlation_with_index(te)
            tr_unique = len(np.unique(tr[np.isfinite(tr)])) / max(int(np.isfinite(tr).sum()), 1)
            te_unique = len(np.unique(te[np.isfinite(te)])) / max(int(np.isfinite(te).sum()), 1)
            continuity_scale = max(float(np.nanstd(tr)), float(np.nanstd(te)), 1e-12)
            continuity_gap = abs(float(tr[-1]) - float(te[0])) / continuity_scale if len(tr) and len(te) else np.inf
            timestamp_like = (
                abs(tr_s) >= threshold
                and abs(te_s) >= threshold
                and tr_unique <= max_unique_frac
                and te_unique <= max_unique_frac
                and continuity_gap <= 1e-3
            )
            if timestamp_like:
                diagnostics.append({
                    "channel": j,
                    "train_spearman_index": tr_s,
                    "test_spearman_index": te_s,
                    "train_unique_frac": tr_unique,
                    "test_unique_frac": te_unique,
                    "continuity_gap_std_units": continuity_gap,
                    "drop_reason": "explicit" if j in explicit_drop else "detected_only",
                })
                if drop_detected:
                    detected_drop.append(j)

    drop_idx = sorted(set(explicit_drop + detected_drop))
    if diagnostics:
        print(f"Timestamp-like feature audit for {dataset_id}: {diagnostics}")
    if drop_idx:
        keep = [j for j in range(train_raw.shape[1]) if j not in set(drop_idx)]
        print(f"Dropping timestamp-like feature columns for {dataset_id}: {drop_idx}")
        return train_raw[:, keep], test_raw[:, keep], diagnostics
    return train_raw, test_raw, diagnostics


def load_multivariate_dataset(data_root: Path, dataset_id: str):
    """
    Return:
        train_raw: np.ndarray [T_train, C]
        test_raw:  np.ndarray [T_test, C]
        point_labels: np.ndarray [T_test] binary
    """
    data_root = Path(data_root)
    dataset_dir = data_root / dataset_id
    if not dataset_dir.exists():
        if not data_root.exists():
            raise FileNotFoundError(f"Data root not found: {data_root}")
        dataset_key = dataset_id.lower()
        dirs = [p for p in data_root.iterdir() if p.is_dir()]
        exact_matches = [p for p in dirs if p.name.lower() == dataset_key]
        contains_matches = [p for p in dirs if dataset_key in p.name.lower()]
        if exact_matches:
            dataset_dir = exact_matches[0]
        elif len(contains_matches) == 1:
            dataset_dir = contains_matches[0]
        elif contains_matches:
            names = [p.name for p in contains_matches]
            raise FileNotFoundError(f"Dataset id {dataset_id!r} matched multiple directories under {data_root}: {names}")
        else:
            available_dirs = sorted([p.name for p in dirs])
            raise FileNotFoundError(f"Dataset directory not found: {data_root / dataset_id}. Available dirs: {available_dirs}")

    stems = [dataset_id, dataset_dir.name, dataset_id.lower(), dataset_id.upper()]
    stems = list(dict.fromkeys([s for s in stems if s]))
    exts = ["npy", "npz", "csv", "pkl", "pickle"]

    train_candidates = [f"train.{ext}" for ext in exts]
    test_candidates = [f"test.{ext}" for ext in exts]
    label_candidates = [
        f"{name}.{ext}"
        for name in ["test_label", "test_labels", "labels", "label"]
        for ext in exts
    ]
    for stem in stems:
        train_candidates += [f"{stem}_train.{ext}" for ext in exts]
        test_candidates += [f"{stem}_test.{ext}" for ext in exts]
        label_candidates += [
            f"{stem}_{name}.{ext}"
            for name in ["test_label", "test_labels", "labels", "label"]
            for ext in exts
        ]

    train_path = _find_first_existing(dataset_dir, train_candidates)
    test_path = _find_first_existing(dataset_dir, test_candidates)
    label_path = _find_first_existing(dataset_dir, label_candidates)

    def _fallback_match(kind: str):
        files = sorted([p for p in dataset_dir.iterdir() if p.is_file()])
        valid_exts = {f".{ext}" for ext in exts}
        files = [p for p in files if p.suffix.lower() in valid_exts]
        if kind == "train":
            matches = [p for p in files if "train" in p.stem.lower() and "label" not in p.stem.lower()]
        elif kind == "test":
            matches = [p for p in files if "test" in p.stem.lower() and "label" not in p.stem.lower()]
            if not matches:
                matches = [
                    p for p in files
                    if "train" not in p.stem.lower()
                    and "label" not in p.stem.lower()
                    and "labels" not in p.stem.lower()
                ]
        elif kind == "label":
            matches = [p for p in files if "label" in p.stem.lower() or "labels" in p.stem.lower()]
        else:
            raise ValueError(kind)
        return matches[0] if len(matches) == 1 else None

    train_path = train_path or _fallback_match("train")
    test_path = test_path or _fallback_match("test")
    label_path = label_path or _fallback_match("label")
    embedded_label_path = None
    if label_path is None and test_path is not None and test_path.suffix.lower() == ".csv":
        test_df_head = pd.read_csv(test_path, nrows=5)
        if _label_columns_from_dataframe(test_df_head):
            embedded_label_path = test_path
    if train_path is None or test_path is None or (label_path is None and embedded_label_path is None):
        available = sorted([p.name for p in dataset_dir.iterdir() if p.is_file()])
        raise FileNotFoundError(
            f"Missing train/test/label files under {dataset_dir}. "
            f"Expected bare names, dataset-prefixed names, unique files containing train/test/label, or labels embedded in test CSV. "
            f"Resolved train={train_path}, test={test_path}, label={label_path}, embedded_label={embedded_label_path}. "
            f"Available files: {available[:20]}"
        )
    label_source_name = label_path.name if label_path is not None else f"{embedded_label_path.name}::embedded_label"
    print(f"Loaded dataset files: train={train_path.name}, test={test_path.name}, labels={label_source_name}")

    train_raw = _as_2d_time_channel(_load_array_file(train_path), "train_raw")
    test_raw = _as_2d_time_channel(_load_array_file(test_path), "test_raw")
    if train_raw.shape[1] != test_raw.shape[1]:
        raise ValueError(f"Train/test channel mismatch: train {train_raw.shape}, test {test_raw.shape}")
    train_raw, test_raw, flat_binary_feature_diagnostics = drop_flat_and_optionally_binary_features(train_raw, test_raw, dataspace_cfg, dataset_id=dataset_dir.name)
    if flat_binary_feature_diagnostics:
        pd.DataFrame(flat_binary_feature_diagnostics).to_csv(dataspace_cfg.output_dir / "flat_binary_feature_diagnostics.csv", index=False)
    train_raw, test_raw, timestamp_feature_diagnostics = drop_timestamp_like_features(train_raw, test_raw, dataspace_cfg, dataset_id=dataset_dir.name)
    if timestamp_feature_diagnostics:
        pd.DataFrame(timestamp_feature_diagnostics).to_csv(dataspace_cfg.output_dir / "timestamp_like_feature_diagnostics.csv", index=False)
    label_raw = _load_array_file(label_path) if label_path is not None else _load_csv_label_column(embedded_label_path)
    point_labels = _as_1d_labels(label_raw, expected_len=test_raw.shape[0], name="point_labels")
    return train_raw, test_raw, point_labels


def standardize_multivariate_train_test(
    train_raw,
    test_raw,
    mode="standard",
    use_clipping=True,
    clip_value=10.0,
    scale_floor=1e-3,
):
    """
    Standardize each channel independently using train statistics only.

    mode="standard": center=nanmean(train), scale=nanstd(train)
    mode="robust": center=nanmedian(train), scale=q75(train)-q25(train)
    """
    train_raw = _as_2d_time_channel(train_raw, "train_raw").astype(np.float32, copy=False)
    test_raw = _as_2d_time_channel(test_raw, "test_raw").astype(np.float32, copy=False)
    train_nan_frac = float(np.isnan(train_raw).mean())
    test_nan_frac = float(np.isnan(test_raw).mean())
    mode = str(mode)
    if mode == "standard":
        center = np.nanmean(train_raw, axis=0, keepdims=True)
        scale = np.nanstd(train_raw, axis=0, keepdims=True)
    elif mode == "robust":
        center = np.nanmedian(train_raw, axis=0, keepdims=True)
        q25 = np.nanquantile(train_raw, 0.25, axis=0, keepdims=True)
        q75 = np.nanquantile(train_raw, 0.75, axis=0, keepdims=True)
        scale = q75 - q25
    else:
        raise ValueError(f"Unknown multivariate scaling mode: {mode}")
    center = np.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.nan_to_num(scale, nan=scale_floor, posinf=scale_floor, neginf=scale_floor)
    scale = np.where(scale > scale_floor, scale, scale_floor)
    train_z = (np.nan_to_num(train_raw, nan=0.0, posinf=0.0, neginf=0.0) - center) / scale
    test_z = (np.nan_to_num(test_raw, nan=0.0, posinf=0.0, neginf=0.0) - center) / scale
    if use_clipping:
        train_z = np.clip(train_z, -float(clip_value), float(clip_value))
        test_z = np.clip(test_z, -float(clip_value), float(clip_value))
    abs_train = np.abs(train_z)
    abs_test = np.abs(test_z)
    diagnostics = {
        "scaling_mode": np.asarray([mode]),
        "train_nan_frac": np.asarray([train_nan_frac], dtype=np.float32),
        "test_nan_frac": np.asarray([test_nan_frac], dtype=np.float32),
        "scale_min": np.asarray([float(np.min(scale))], dtype=np.float32),
        "scale_median": np.asarray([float(np.median(scale))], dtype=np.float32),
        "scale_max": np.asarray([float(np.max(scale))], dtype=np.float32),
        "train_abs_max": np.asarray([float(abs_train.max())], dtype=np.float32),
        "train_abs_q999": np.asarray([float(np.quantile(abs_train, 0.999))], dtype=np.float32),
        "test_abs_max": np.asarray([float(abs_test.max())], dtype=np.float32),
        "test_abs_q999": np.asarray([float(np.quantile(abs_test, 0.999))], dtype=np.float32),
        "scale_floor": np.asarray([float(scale_floor)], dtype=np.float32),
        "use_clipping": np.asarray([bool(use_clipping)]),
        "clip_value": np.asarray([float(clip_value)], dtype=np.float32),
        "center": center.squeeze(0).astype(np.float32),
        "scale": scale.squeeze(0).astype(np.float32),
    }
    return train_z.astype(np.float32), test_z.astype(np.float32), diagnostics


# Backward-compatible alias; default notebook behavior is controlled by dataspace_cfg.
def robust_standardize_multivariate(train_raw, test_raw, eps=1e-3):
    return standardize_multivariate_train_test(train_raw, test_raw, mode="robust", scale_floor=eps)


def make_centered_windows_multivariate(x, point_labels=None, window=64, stride=1):
    """
    x: [T, C]
    point_labels: optional [T]
    Return:
        windows: [N, W, C]
        labels: [N]
        starts: [N]
        centers: [N]
        overlap_frac: [N]
    """
    x = _as_2d_time_channel(x, "x")
    T, C = x.shape
    half = window // 2
    if window % 2 != 0:
        raise ValueError("This centered extractor expects an even window length.")
    if T < window:
        raise ValueError(f"Series length {T} is shorter than window {window}")
    labels_arr = None if point_labels is None else _as_1d_labels(point_labels, expected_len=T, name="point_labels")
    windows, labels, starts, centers, overlap_frac = [], [], [], [], []
    for center in range(half, T - half + 1, stride):
        start = center - half
        end = center + half
        window_x = x[start:end]
        if window_x.shape[0] != window:
            continue
        windows.append(window_x)
        starts.append(start)
        centers.append(center)
        if labels_arr is None:
            labels.append(0)
            overlap_frac.append(0.0)
        else:
            frac = float(labels_arr[start:end].mean())
            labels.append(int(frac > 0.0))
            overlap_frac.append(frac)
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(starts, dtype=np.int64),
        np.asarray(centers, dtype=np.int64),
        np.asarray(overlap_frac, dtype=np.float32),
    )


train_raw, test_raw, point_labels = load_multivariate_dataset(dataspace_cfg.data_root, dataspace_cfg.dataset_id)
train_std, test_std, standardization_stats = standardize_multivariate_train_test(
    train_raw,
    test_raw,
    mode=dataspace_cfg.multivar_scaling_mode,
    use_clipping=dataspace_cfg.multivar_use_clipping,
    clip_value=dataspace_cfg.multivar_clip_value,
    scale_floor=dataspace_cfg.multivar_scale_floor,
)

train_x, train_y, train_starts, train_centers, train_frac = make_centered_windows_multivariate(
    train_std,
    point_labels=None,
    window=dataspace_cfg.window,
    stride=dataspace_cfg.stride,
)

test_x, test_y, test_starts, test_centers, test_frac = make_centered_windows_multivariate(
    test_std,
    point_labels=point_labels,
    window=dataspace_cfg.window,
    stride=dataspace_cfg.stride,
)

print("multivar scaling mode:", dataspace_cfg.multivar_scaling_mode)
print("clipping:", dataspace_cfg.multivar_use_clipping, "clip_value:", dataspace_cfg.multivar_clip_value)
print("train scaled abs max:", float(np.abs(train_std).max()))
print("test scaled abs max:", float(np.abs(test_std).max()))
print("train scaled q999 abs:", float(np.quantile(np.abs(train_std), 0.999)))
print("test scaled q999 abs:", float(np.quantile(np.abs(test_std), 0.999)))

dataspace_cfg.channels = int(train_x.shape[-1])
dataspace_cfg.length = int(dataspace_cfg.window)
dataspace_cfg.flat_dim = int(dataspace_cfg.channels * dataspace_cfg.length)
dataspace_cfg.data_shape = (dataspace_cfg.channels, dataspace_cfg.length)
dataspace_cfg.train_raw_shape = tuple(train_raw.shape)
dataspace_cfg.test_raw_shape = tuple(test_raw.shape)
if dataspace_cfg.hcfm_transport_rank is None:
    dataspace_cfg.hcfm_transport_rank = max(1, min(32, dataspace_cfg.flat_dim // 4))

dataspace_cfg = finalize_dataspace_cfg(dataspace_cfg)
run_cfg = SimpleNamespace(
    dataset_id=dataspace_cfg.dataset_id,
    data_root=dataspace_cfg.data_root,
    output_dir=dataspace_cfg.output_dir,
    seed=dataspace_cfg.seed,
    window=dataspace_cfg.window,
    stride_train=dataspace_cfg.stride,
    stride_test=dataspace_cfg.stride,
    make_plots=True,
)
print_dataspace_config_summary(dataspace_cfg)

with open(dataspace_cfg.output_dir / "config.json", "w", encoding="utf-8") as f:
    json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(dataspace_cfg).items()}, f, indent=2)
np.savez(dataspace_cfg.output_dir / "standardization_stats.npz", **standardization_stats)

summary = pd.DataFrame({
    "split": ["train", "test"],
    "raw_shape": [tuple(train_raw.shape), tuple(test_raw.shape)],
    "windows": [len(train_x), len(test_x)],
    "anomaly_windows": [int(train_y.sum()), int(test_y.sum())],
    "point_count": [np.nan, len(point_labels)],
    "point_anomalies": [np.nan, int(point_labels.sum())],
    "channels": [dataspace_cfg.channels, dataspace_cfg.channels],
    "flat_dim": [dataspace_cfg.flat_dim, dataspace_cfg.flat_dim],
    "window": [dataspace_cfg.window, dataspace_cfg.window],
    "stride": [dataspace_cfg.stride, dataspace_cfg.stride],
})
add_config_metadata(summary, dataspace_cfg).to_csv(dataspace_cfg.output_dir / "data_summary.csv", index=False)
display(summary)
print("train windows:", train_x.shape, "test windows:", test_x.shape)
print("test anomaly windows:", int(test_y.sum()), "of", len(test_y))
print("first train window start/center:", int(train_starts[0]), int(train_centers[0]))
print("first test window start/center:", int(test_starts[0]), int(test_centers[0]))


# %% notebook cell 13
t0 = time.time()

# This path intentionally removes AE latent learning to test flows directly in data space.
# Existing window tensors are [B, W, C]; CNN models consume [B, C, W].
tensor_device = device if dataspace_cfg.keep_train_tensors_on_device else torch.device("cpu")
train_x_seq = torch.tensor(np.transpose(train_x, (0, 2, 1)), dtype=torch.float32, device=tensor_device)
test_x_seq = torch.tensor(np.transpose(test_x, (0, 2, 1)), dtype=torch.float32, device=device)
if dataspace_cfg.keep_train_tensors_on_device:
    train_x_seq = train_x_seq.to(device, non_blocking=True)

assert train_x_seq.ndim == 3
assert test_x_seq.ndim == 3
assert tuple(train_x_seq.shape[1:]) == dataspace_cfg.data_shape
assert tuple(test_x_seq.shape[1:]) == dataspace_cfg.data_shape

print("train_x_seq shape:", tuple(train_x_seq.shape))
print("test_x_seq shape:", tuple(test_x_seq.shape))
print("train mean/std:", float(train_x_seq.mean().detach().cpu()), float(train_x_seq.std().detach().cpu()))
print("test mean/std:", float(test_x_seq.mean().detach().cpu()), float(test_x_seq.std().detach().cpu()))
print("device:", device)

calib_size = min(len(train_x_seq), max(dataspace_cfg.calibration_min, min(dataspace_cfg.calibration_max, int(round(dataspace_cfg.calibration_fraction * len(train_x_seq))))))
calib_rng = np.random.default_rng(dataspace_cfg.seed)
calib_idx = np.sort(calib_rng.choice(len(train_x_seq), size=calib_size, replace=False))
calib_x_seq = train_x_seq[torch.as_tensor(calib_idx, device=train_x_seq.device)].to(device, non_blocking=True)
calibration_fraction_actual = float(calib_size / len(train_x_seq))

np.save(dataspace_cfg.output_dir / "calibration_indices.npy", calib_idx)
with open(dataspace_cfg.output_dir / "calibration_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "train_windows": int(len(train_x_seq)),
        "calibration_windows": int(calib_size),
        "dataspace_cfg.calibration_fraction_requested": float(dataspace_cfg.calibration_fraction),
        "calibration_fraction_actual": calibration_fraction_actual,
        "dataspace_cfg.calibration_min": int(dataspace_cfg.calibration_min),
        "dataspace_cfg.calibration_max": int(dataspace_cfg.calibration_max),
        "dataspace_cfg.seed": int(dataspace_cfg.seed),
    }, f, indent=2)
print("calibration windows:", int(calib_size))


# %% notebook cell 15
# Archived latent-space/CNF flow matching code was moved to hcfm_legacy_latent_flows.py.
# The active notebook definitions below are only the data-space CNN models and ODE helpers.

class SinusoidalTimeEmbedding(nn.Module):
    """
    Fixed sinusoidal time embedding for continuous flow time t.

    time_scale controls the argument scale before the sinusoidal frequencies.
    Set time_scale=1 and use_2pi=False to recover the old behavior.
    """

    def __init__(self, dim: int, max_period: float = 10000.0, use_projection: bool = True, time_scale: float = 1000.0, use_2pi: bool = False):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.time_scale = float(time_scale)
        self.use_2pi = bool(use_2pi)
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / max(half, 1))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)) if use_projection else nn.Identity()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(1)
        if t.ndim == 2:
            t = t[:, 0]
        t = t.float()
        scale = self.time_scale * (2.0 * math.pi if self.use_2pi else 1.0)
        args = (t[:, None] * scale) * self.freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb[:, : self.dim])


class ResidualConvBlock1D(nn.Module):
    """Residual Conv1D block with optional FiLM time conditioning."""

    def __init__(self, hidden: int, t_dim: int, groups: int = 8, use_film_time_conditioning: bool = True):
        super().__init__()
        num_groups = max(1, min(groups, hidden))
        self.use_film_time_conditioning = bool(use_film_time_conditioning)
        if self.use_film_time_conditioning:
            self.time1 = nn.Linear(t_dim, 2 * hidden)
            self.time2 = nn.Linear(t_dim, 2 * hidden)
        else:
            self.t_proj = nn.Linear(t_dim, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=hidden)

    @staticmethod
    def apply_film(h: torch.Tensor, film: torch.Tensor) -> torch.Tensor:
        scale, shift = film.chunk(2, dim=1)
        return h * (1.0 + scale[:, :, None]) + shift[:, :, None]

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.use_film_time_conditioning:
            h = self.norm1(x)
            h = self.apply_film(h, self.time1(t_emb))
            h = F.silu(h)
            h = self.conv1(h)
            h = self.norm2(h)
            h = self.apply_film(h, self.time2(t_emb))
            h = F.silu(h)
            h = self.conv2(h)
            return x + h
        h = x + self.t_proj(t_emb)[:, :, None]
        h = self.conv1(h)
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return x + h


class Conv1DVectorField(nn.Module):
    """
    CNN vector field v_theta(x_t, t) for data-space flow matching.

    Input/output shape is [B, C, W]. The model preserves temporal layout
    and uses residual Conv1D blocks instead of flattening the window into
    an MLP. This is used for Vanilla Data FM, Data FDM-lite, and the
    compression/residual components inside Data HCFM.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden: int, depth: int, t_dim: int, max_period: float = 10000.0, time_scale: float = 1000.0, time_use_2pi: bool = False, use_film_time_conditioning: bool = True):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(t_dim, max_period=max_period, use_projection=True, time_scale=time_scale, use_2pi=time_use_2pi)
        self.in_proj = nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualConvBlock1D(hidden, t_dim, use_film_time_conditioning=use_film_time_conditioning) for _ in range(depth)])
        self.out_proj = nn.Conv1d(hidden, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, x_ref=None, context=None, focus=None) -> torch.Tensor:
        t_emb = self.time(t)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out_proj(F.silu(h))



class ShallowResidualVectorField1D(nn.Module):
    """Intentionally weak residual correction head for Data HCFM."""

    def __init__(
        self,
        channels: int,
        hidden: int = 16,
        t_dim: int = 64,
        kernel_size: int = 3,
        max_period: float = 10000.0,
        time_scale: float = 1000.0,
        time_use_2pi: bool = False,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.time = SinusoidalTimeEmbedding(
            t_dim,
            max_period=max_period,
            use_projection=True,
            time_scale=time_scale,
            use_2pi=time_use_2pi,
        )
        self.in_proj = nn.Conv1d(channels, hidden, kernel_size=kernel_size, padding=padding)
        self.t_proj = nn.Linear(t_dim, hidden)
        self.out_proj = nn.Conv1d(hidden, channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, t: torch.Tensor, x_ref=None, context=None, focus=None) -> torch.Tensor:
        h = self.in_proj(x)
        h = h + self.t_proj(self.time(t))[:, :, None]
        h = F.silu(h)
        return self.out_proj(h)




class SinusoidalPositionEmbedding1D(nn.Module):
    """
    Fixed window-position embedding with a small learned projection.

    The projected embedding is concatenated into the scalar-potential CNN as
    conditioning. It is not part of the vector-field state, so compression uses
    grad_x phi([x, e_pos], t) rather than gradients with respect to position.
    """

    def __init__(self, emb_dim: int, proj_dim: int, length: int, max_period: float = 10000.0, use_integer_positions: bool = True):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.proj_dim = int(proj_dim)
        self.length = int(length)
        self.use_integer_positions = bool(use_integer_positions)
        half = self.emb_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / max(half, 1)
        )
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(self.emb_dim, self.proj_dim),
            nn.SiLU(),
            nn.Linear(self.proj_dim, self.proj_dim),
        )

    def forward(self, batch_size: int, device, dtype) -> torch.Tensor:
        if self.use_integer_positions:
            pos = torch.arange(self.length, device=device, dtype=dtype)
        else:
            pos = torch.linspace(0.0, 1.0, self.length, device=device, dtype=dtype)
        args = pos[:, None] * self.freqs[None, :].to(device=device, dtype=dtype)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < self.emb_dim:
            emb = F.pad(emb, (0, self.emb_dim - emb.shape[-1]))
        emb = self.proj(emb[:, : self.emb_dim])
        emb = emb.transpose(0, 1).unsqueeze(0)
        return emb.expand(batch_size, -1, -1)


class PotentialResidualConvBlock1D(nn.Module):
    """Potential-head-only residual block with optional FiLM time conditioning."""

    def __init__(self, hidden: int, t_dim: int, norm_type: str = "none", residual_scale: float = 0.10, groups: int = 8, use_film_time_conditioning: bool = True):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.norm_type = norm_type
        self.use_film_time_conditioning = bool(use_film_time_conditioning)
        if self.use_film_time_conditioning:
            self.time1 = nn.Linear(t_dim, 2 * hidden)
            self.time2 = nn.Linear(t_dim, 2 * hidden)
        else:
            self.t_proj = nn.Linear(t_dim, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        if norm_type == "groupnorm":
            num_groups = max(1, min(groups, hidden))
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=hidden)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=hidden)
        elif norm_type == "none":
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            raise ValueError(f"unsupported potential norm_type: {norm_type}")

    @staticmethod
    def apply_film(h: torch.Tensor, film: torch.Tensor) -> torch.Tensor:
        scale, shift = film.chunk(2, dim=1)
        return h * (1.0 + scale[:, :, None]) + shift[:, :, None]

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.use_film_time_conditioning:
            h = self.norm1(x)
            h = self.apply_film(h, self.time1(t_emb))
            h = F.silu(h)
            h = self.conv1(h)
            h = self.norm2(h)
            h = self.apply_film(h, self.time2(t_emb))
            h = F.silu(h)
            h = self.conv2(h)
            return x + self.residual_scale * h
        h = x + self.t_proj(t_emb)[:, :, None]
        h = self.conv1(h)
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return x + self.residual_scale * h


class MixerPotentialBlock1D(nn.Module):
    """MLP-Mixer style block over [B, W, hidden] potential tokens."""

    def __init__(self, length: int, hidden: int, token_mlp_dim: int, channel_mlp_dim: int, norm_type: str = "none", residual_scale: float = 0.10):
        super().__init__()
        self.residual_scale = float(residual_scale)
        if norm_type == "layernorm":
            self.norm_token = nn.LayerNorm(hidden)
            self.norm_channel = nn.LayerNorm(hidden)
        elif norm_type == "none":
            self.norm_token = nn.Identity()
            self.norm_channel = nn.Identity()
        else:
            raise ValueError(f"unsupported mixer norm_type: {norm_type}")
        self.token_mlp = nn.Sequential(
            nn.Linear(length, token_mlp_dim),
            nn.SiLU(),
            nn.Linear(token_mlp_dim, length),
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden, channel_mlp_dim),
            nn.SiLU(),
            nn.Linear(channel_mlp_dim, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, W, H]
        h = self.norm_token(x).transpose(1, 2)
        x = x + self.residual_scale * self.token_mlp(h).transpose(1, 2)
        h = self.norm_channel(x)
        x = x + self.residual_scale * self.channel_mlp(h)
        return x


class MixerScalarPotentialBackbone1D(nn.Module):
    """Scalar potential backbone using token/channel MLP mixing."""

    def __init__(self, in_channels: int, hidden: int, length: int, t_dim: int, depth: int, token_mlp_dim: int, channel_mlp_dim: int, norm_type: str = "none", residual_scale: float = 0.10, out_init: str = "small", out_init_std: float = 1e-3):
        super().__init__()
        self.length = int(length)
        self.in_proj = nn.Linear(in_channels, hidden)
        self.t_proj = nn.Linear(t_dim, hidden)
        self.blocks = nn.ModuleList([
            MixerPotentialBlock1D(
                length=length,
                hidden=hidden,
                token_mlp_dim=token_mlp_dim,
                channel_mlp_dim=channel_mlp_dim,
                norm_type=norm_type,
                residual_scale=residual_scale,
            )
            for _ in range(depth)
        ])
        self.out_proj = nn.Linear(hidden, 1)
        if out_init == "zero":
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        elif out_init == "small":
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=float(out_init_std))
            nn.init.zeros_(self.out_proj.bias)
        elif out_init == "default":
            pass
        else:
            raise ValueError(f"unsupported potential out_init: {out_init}")

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: [B, C, W] -> tokens [B, W, C]
        h = x.transpose(1, 2)
        h = self.in_proj(h) + self.t_proj(t_emb)[:, None, :]
        for block in self.blocks:
            h = block(h)
        phi_density = self.out_proj(F.silu(h)).transpose(1, 2)
        return phi_density



class DifferentiableFFTFeatures1D(nn.Module):
    """Differentiable global spectral conditioning features for [B, C, W]."""

    def __init__(
        self,
        num_bands: int = 4,
        feature_mode: str = "global_bands",
        include_centroid: bool = True,
        include_entropy: bool = True,
        include_dominant: bool = True,
        log_magnitude: bool = True,
        eps: float = 1e-8,
        dominant_temperature: float = 10.0,
    ):
        super().__init__()
        if feature_mode != "global_bands":
            raise ValueError(f"unsupported FFT feature_mode: {feature_mode}")
        self.num_bands = int(num_bands)
        self.feature_mode = feature_mode
        self.include_centroid = bool(include_centroid)
        self.include_entropy = bool(include_entropy)
        self.include_dominant = bool(include_dominant)
        self.log_magnitude = bool(log_magnitude)
        self.eps = float(eps)
        self.dominant_temperature = float(dominant_temperature)
        self.feature_dim = self.num_bands + int(self.include_centroid) + int(self.include_entropy) + int(self.include_dominant)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.rfft(x, dim=-1)
        # Avoid torch.abs(complex) on CUDA because some Windows CUDA builds
        # require nvrtc-builtins DLLs at runtime for that generated kernel.
        mag = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2) + self.eps).to(dtype=x.dtype)
        mag_feat = torch.log1p(mag) if self.log_magnitude else mag
        if mag_feat.shape[-1] > 1:
            band_source = mag_feat[..., 1:]
            mag_no_dc = mag[..., 1:]
        else:
            band_source = mag_feat
            mag_no_dc = mag
        num_bins = band_source.shape[-1]
        band_values = []
        for band in torch.tensor_split(band_source, self.num_bands, dim=-1):
            if band.shape[-1] == 0:
                band_values.append(torch.zeros(x.shape[0], device=x.device, dtype=x.dtype))
            else:
                band_values.append(band.pow(2).mean(dim=(-1, -2)))
        features = [torch.stack(band_values, dim=1)]
        if self.include_centroid or self.include_entropy or self.include_dominant:
            weights = mag_no_dc + self.eps
            freqs = torch.linspace(0.0, 1.0, weights.shape[-1], device=x.device, dtype=x.dtype)
        if self.include_centroid:
            centroid = (weights * freqs.view(1, 1, -1)).sum(dim=-1) / weights.sum(dim=-1).clamp_min(self.eps)
            features.append(centroid.mean(dim=1, keepdim=True))
        if self.include_entropy:
            p = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            entropy = -(p * torch.log(p + self.eps)).sum(dim=-1)
            if num_bins > 1:
                entropy = entropy / math.log(float(num_bins))
            features.append(entropy.mean(dim=1, keepdim=True))
        if self.include_dominant:
            soft = torch.softmax(self.dominant_temperature * weights, dim=-1)
            dominant = (soft * freqs.view(1, 1, -1)).sum(dim=-1)
            features.append(dominant.mean(dim=1, keepdim=True))
        return torch.cat(features, dim=1)


class Conv1DScalarPotentialCompression(nn.Module):
    """
    Scalar-potential compression head for active DataHCFM.

    Defines a scalar potential phi_theta(x,t) and returns
        v_C(x,t) = grad_x phi_theta(x,t).

    This makes the compression head an irrotational/potential component. Its
    divergence is the Laplacian of the learned potential and is the intended
    density-change diagnostic.
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        depth: int,
        t_dim: int,
        length: int,
        max_period: float = 10000.0,
        time_scale: float = 1000.0,
        time_use_2pi: bool = False,
        use_film_time_conditioning: bool = True,
        norm_type: str = "none",
        residual_scale: float = 0.10,
        potential_backbone: str = "cnn",
        mixer_depth: int = 3,
        mixer_token_mlp_dim: int = 128,
        mixer_channel_mlp_dim: int = 128,
        mixer_norm_type: str = "none",
        mixer_residual_scale: float = 0.10,
        potential_scale: float = 0.10,
        out_init: str = "small",
        out_init_std: float = 1e-3,
        use_position_embedding: bool = True,
        position_emb_dim: int = 16,
        position_proj_dim: int = 4,
        position_max_period: float = 10000.0,
        position_use_integer_positions: bool = True,
        use_fft_features: bool = False,
        fft_feature_mode: str = "global_bands",
        fft_num_bands: int = 4,
        fft_include_centroid: bool = True,
        fft_include_entropy: bool = True,
        fft_include_dominant: bool = True,
        fft_log_magnitude: bool = True,
        fft_eps: float = 1e-8,
        fft_detach_features: bool = False,
        normalize_fft_features: bool = True,
        scale_aux_channels_by_x_stats: bool = True,
        context_focus_scale: float = 10.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.length = length
        self.flat_dim = in_channels * length
        self.potential_scale = float(potential_scale)
        self.potential_backbone = potential_backbone
        self.norm_type = norm_type
        self.residual_scale = float(residual_scale)
        self.mixer_depth = int(mixer_depth)
        self.mixer_token_mlp_dim = int(mixer_token_mlp_dim)
        self.mixer_channel_mlp_dim = int(mixer_channel_mlp_dim)
        self.mixer_norm_type = mixer_norm_type
        self.mixer_residual_scale = float(mixer_residual_scale)
        self.out_init = out_init
        self.out_init_std = float(out_init_std)
        self.use_position_embedding = bool(use_position_embedding)
        self.position_emb_dim = int(position_emb_dim)
        self.position_proj_dim = int(position_proj_dim)
        self.position_max_period = float(position_max_period)
        self.position_use_integer_positions = bool(position_use_integer_positions)
        self.use_fft_features = bool(use_fft_features)
        self.fft_feature_mode = fft_feature_mode
        self.fft_num_bands = int(fft_num_bands)
        self.fft_include_centroid = bool(fft_include_centroid)
        self.fft_include_entropy = bool(fft_include_entropy)
        self.fft_include_dominant = bool(fft_include_dominant)
        self.fft_log_magnitude = bool(fft_log_magnitude)
        self.fft_eps = float(fft_eps)
        self.fft_detach_features = bool(fft_detach_features)
        self.normalize_fft_features_flag = bool(normalize_fft_features)
        self.scale_aux_channels_by_x_stats = bool(scale_aux_channels_by_x_stats)
        self.context_focus_scale = float(context_focus_scale)
        self.use_film_time_conditioning = bool(use_film_time_conditioning)
        self.time = SinusoidalTimeEmbedding(t_dim, max_period=max_period, use_projection=True, time_scale=time_scale, use_2pi=time_use_2pi)
        if self.use_position_embedding:
            self.position_embedding = SinusoidalPositionEmbedding1D(
                emb_dim=self.position_emb_dim,
                proj_dim=self.position_proj_dim,
                length=length,
                max_period=self.position_max_period,
                use_integer_positions=self.position_use_integer_positions,
            )
            potential_in_channels = in_channels + self.position_proj_dim
        else:
            self.position_embedding = None
            potential_in_channels = in_channels
        if self.use_fft_features:
            self.fft_features = DifferentiableFFTFeatures1D(
                num_bands=self.fft_num_bands,
                feature_mode=self.fft_feature_mode,
                include_centroid=self.fft_include_centroid,
                include_entropy=self.fft_include_entropy,
                include_dominant=self.fft_include_dominant,
                log_magnitude=self.fft_log_magnitude,
                eps=self.fft_eps,
            )
            self.fft_feature_dim = self.fft_features.feature_dim
            potential_in_channels += self.fft_feature_dim
        else:
            self.fft_features = None
            self.fft_feature_dim = 0
        if potential_backbone == "cnn":
            self.in_proj = nn.Conv1d(potential_in_channels, hidden, kernel_size=3, padding=1)
            self.blocks = nn.ModuleList([
                PotentialResidualConvBlock1D(hidden, t_dim, norm_type=norm_type, residual_scale=residual_scale, use_film_time_conditioning=use_film_time_conditioning)
                for _ in range(depth)
            ])
            self.out_proj = nn.Conv1d(hidden, 1, kernel_size=3, padding=1)
            if out_init == "zero":
                nn.init.zeros_(self.out_proj.weight)
                nn.init.zeros_(self.out_proj.bias)
            elif out_init == "small":
                nn.init.normal_(self.out_proj.weight, mean=0.0, std=self.out_init_std)
                nn.init.zeros_(self.out_proj.bias)
            elif out_init == "default":
                pass
            else:
                raise ValueError(f"unsupported potential out_init: {out_init}")
            self.mixer = None
        elif potential_backbone == "mixer":
            self.in_proj = None
            self.blocks = nn.ModuleList()
            self.out_proj = None
            self.mixer = MixerScalarPotentialBackbone1D(
                in_channels=potential_in_channels,
                hidden=hidden,
                length=length,
                t_dim=t_dim,
                depth=self.mixer_depth,
                token_mlp_dim=self.mixer_token_mlp_dim,
                channel_mlp_dim=self.mixer_channel_mlp_dim,
                norm_type=self.mixer_norm_type,
                residual_scale=self.mixer_residual_scale,
                out_init=out_init,
                out_init_std=self.out_init_std,
            )
        else:
            raise ValueError(f"unsupported potential_backbone: {potential_backbone}")

    def normalize_fft_features(self, fft_feat: torch.Tensor) -> torch.Tensor:
        if not self.normalize_fft_features_flag:
            return fft_feat
        fft_feat = torch.nan_to_num(fft_feat, nan=0.0, posinf=5.0, neginf=-5.0)
        fft_feat = torch.sign(fft_feat) * torch.log1p(fft_feat.abs())
        return torch.clamp(fft_feat, -5.0, 5.0)

    def get_x_ref_stats(self, x: torch.Tensor, x_ref: torch.Tensor | None = None):
        ref = x if x_ref is None else x_ref
        ref_mean = ref.mean(dim=(-1, -2), keepdim=True)
        ref_std = ref.std(dim=(-1, -2), keepdim=True).clamp_min(1e-4)
        return ref_mean, ref_std

    def scale_aux_channel(self, aux: torch.Tensor, x_std: torch.Tensor) -> torch.Tensor:
        if not self.scale_aux_channels_by_x_stats:
            return aux
        aux = torch.nan_to_num(aux, nan=0.0, posinf=5.0, neginf=-5.0)
        aux_mean = aux.mean(dim=(1, 2), keepdim=True)
        aux_std = aux.std(dim=(1, 2), keepdim=True).clamp_min(1e-4)
        aux = (aux - aux_mean) / aux_std
        aux = aux * x_std
        return torch.nan_to_num(aux, nan=0.0, posinf=5.0, neginf=-5.0)

    def potential_input(self, x: torch.Tensor, x_ref=None, context=None, focus=None) -> torch.Tensor:
        _, x_std = self.get_x_ref_stats(x, x_ref=x_ref)
        parts = [x]
        if self.use_position_embedding:
            pos = self.position_embedding(batch_size=x.shape[0], device=x.device, dtype=x.dtype)
            pos = self.scale_aux_channel(pos, x_std)
            parts.append(pos)
        if self.use_fft_features:
            fft_source = x if x_ref is None else x_ref
            fft_feat = self.fft_features(fft_source)
            if self.fft_detach_features:
                fft_feat = fft_feat.detach()
            fft_feat = self.normalize_fft_features(fft_feat)
            fft_channels = fft_feat[:, :, None].expand(-1, -1, x.shape[-1])
            fft_channels = self.scale_aux_channel(fft_channels, x_std)
            parts.append(fft_channels)
        if context is not None and torch.is_tensor(context) and context.ndim == 3:
            parts.append(self.scale_aux_channel(context, x_std))
        if focus is not None:
            focus = focus.clamp_min(0.0)
            if focus.ndim == 1:
                focus = focus[:, None]
            focus_scaled = (focus * self.context_focus_scale).clamp(0.0, 5.0)
            focus_channels = focus_scaled[:, :, None].expand(-1, 1, x.shape[-1]) * x_std
            parts.append(focus_channels)
        return torch.cat(parts, dim=1)

    def potential(self, x: torch.Tensor, t: torch.Tensor, x_ref=None, context=None, focus=None) -> torch.Tensor:
        t_emb = self.time(t)
        h_in = self.potential_input(x, x_ref=x_ref, context=context, focus=focus)
        if self.potential_backbone == "cnn":
            h = self.in_proj(h_in)
            for block in self.blocks:
                h = block(h, t_emb)
            phi_density = self.out_proj(F.silu(h))
        elif self.potential_backbone == "mixer":
            phi_density = self.mixer(h_in, t_emb)
        else:
            raise ValueError(f"unsupported potential_backbone: {self.potential_backbone}")
        phi = phi_density.flatten(1).sum(dim=1) / math.sqrt(float(self.flat_dim))
        return self.potential_scale * phi

    def forward(self, x: torch.Tensor, t: torch.Tensor, x_ref=None, context=None, focus=None) -> torch.Tensor:
        # Compression is a gradient field. Even during evaluation/scoring, we
        # need gradients with respect to x to compute grad_x phi.
        if not x.requires_grad:
            x = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            phi = self.potential(x, t, x_ref=x_ref, context=context, focus=focus)
            v = torch.autograd.grad(
                phi.sum(),
                x,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
        return v



class LowRankSkewTransport(nn.Module):
    """
    Divergence-free transport field for data-space HCFM.

    The transport field is parameterized as v_transport(x,t) = A(t) x,
    where A(t) = U(t)V(t)^T - V(t)U(t)^T is skew-symmetric. Since
    trace(A(t)) = 0, this component is divergence-free with respect to x
    when A depends only on t.
    """

    def __init__(self, channels: int, length: int, rank: int = 16, time_emb_dim: int = 64, hidden: int = 128, max_period: float = 10000.0, time_scale: float = 1000.0, time_use_2pi: bool = False):
        super().__init__()
        self.channels = channels
        self.length = length
        self.flat_dim = channels * length
        self.rank = rank
        self.time = SinusoidalTimeEmbedding(time_emb_dim, max_period=max_period, use_projection=True, time_scale=time_scale, use_2pi=time_use_2pi)
        self.head = nn.Sequential(
            nn.Linear(time_emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * self.flat_dim * rank),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, x_ref=None, context=None, focus=None) -> torch.Tensor:
        x_flat = x.flatten(1)
        params = self.head(self.time(t)).view(x.shape[0], 2, self.flat_dim, self.rank)
        u = params[:, 0]
        v = params[:, 1]
        x_col = x_flat.unsqueeze(-1)
        vtx = torch.bmm(v.transpose(1, 2), x_col)
        utx = torch.bmm(u.transpose(1, 2), x_col)
        ax = torch.bmm(u, vtx) - torch.bmm(v, utx)
        return ax.squeeze(-1).view_as(x)


class DataHCFM(nn.Module):
    """
    Structured data-space HCFM with optional scalar-potential compression.

    v_total = v_transport + gamma_c * v_compression + gamma_r * scale * v_residual

    v_transport is a divergence-free low-rank skew-symmetric transport field.
    v_compression is either a scalar-potential gradient field or a CNN vector
    field ablation. v_residual is a lightweight correction field for remaining mismatch.
    """

    def __init__(
        self,
        channels: int,
        length: int,
        hidden: int,
        depth: int,
        t_dim: int,
        transport_rank: int = 16,
        residual_depth: int | None = None,
        residual_arch: str = "standard",
        residual_hidden: int = 16,
        residual_kernel_size: int = 3,
        gamma_compression: float = 1.0,
        gamma_residual: float = 1.0,
        max_period: float = 10000.0,
        time_scale: float = 1000.0,
        time_use_2pi: bool = False,
        use_film_time_conditioning: bool = True,
        compression_type: str = "scalar_potential",
        use_residual_mismatch_gate: bool = False,
        residual_gate_min: float = 0.10,
        inference_residual_scale_mode: str = "full",
        potential_norm_type: str = "none",
        potential_residual_scale: float = 0.10,
        potential_backbone: str = "cnn",
        mixer_depth: int = 3,
        mixer_token_mlp_dim: int = 128,
        mixer_channel_mlp_dim: int = 128,
        mixer_norm_type: str = "none",
        mixer_residual_scale: float = 0.10,
        potential_scale: float = 0.10,
        potential_out_init: str = "small",
        potential_out_init_std: float = 1e-3,
        use_position_embedding: bool = True,
        position_emb_dim: int = 16,
        position_proj_dim: int = 4,
        position_max_period: float = 10000.0,
        position_use_integer_positions: bool = True,
        use_fft_features: bool = False,
        fft_feature_mode: str = "global_bands",
        fft_num_bands: int = 4,
        fft_include_centroid: bool = True,
        fft_include_entropy: bool = True,
        fft_include_dominant: bool = True,
        fft_log_magnitude: bool = True,
        fft_eps: float = 1e-8,
        fft_detach_features: bool = False,
        normalize_fft_features: bool = True,
        scale_aux_channels_by_x_stats: bool = True,
        context_focus_scale: float = 10.0,
    ):
        super().__init__()
        self.gamma_compression = gamma_compression
        self.gamma_residual = gamma_residual
        self.compression_type = compression_type
        self.use_residual_mismatch_gate = bool(use_residual_mismatch_gate)
        self.residual_gate_min = float(residual_gate_min)
        self.inference_residual_scale_mode = inference_residual_scale_mode
        if self.use_residual_mismatch_gate and inference_residual_scale_mode == "floor":
            self.inference_residual_scale = self.residual_gate_min
        else:
            self.inference_residual_scale = 1.0
        residual_depth = depth if residual_depth is None else residual_depth
        self.transport = LowRankSkewTransport(
            channels=channels,
            length=length,
            rank=transport_rank,
            time_emb_dim=t_dim,
            hidden=max(128, hidden),
            max_period=max_period,
            time_scale=time_scale,
            time_use_2pi=time_use_2pi,
        )
        if compression_type == "scalar_potential":
            self.compression = Conv1DScalarPotentialCompression(
                in_channels=channels,
                hidden=hidden,
                depth=depth,
                t_dim=t_dim,
                length=length,
                max_period=max_period,
                time_scale=time_scale,
                time_use_2pi=time_use_2pi,
                use_film_time_conditioning=use_film_time_conditioning,
                norm_type=potential_norm_type,
                residual_scale=potential_residual_scale,
                potential_backbone=potential_backbone,
                mixer_depth=mixer_depth,
                mixer_token_mlp_dim=mixer_token_mlp_dim,
                mixer_channel_mlp_dim=mixer_channel_mlp_dim,
                mixer_norm_type=mixer_norm_type,
                mixer_residual_scale=mixer_residual_scale,
                potential_scale=potential_scale,
                out_init=potential_out_init,
                out_init_std=potential_out_init_std,
                use_position_embedding=use_position_embedding,
                position_emb_dim=position_emb_dim,
                position_proj_dim=position_proj_dim,
                position_max_period=position_max_period,
                position_use_integer_positions=position_use_integer_positions,
                use_fft_features=use_fft_features,
                fft_feature_mode=fft_feature_mode,
                fft_num_bands=fft_num_bands,
                fft_include_centroid=fft_include_centroid,
                fft_include_entropy=fft_include_entropy,
                fft_include_dominant=fft_include_dominant,
                fft_log_magnitude=fft_log_magnitude,
                fft_eps=fft_eps,
                fft_detach_features=fft_detach_features,
                normalize_fft_features=normalize_fft_features,
                scale_aux_channels_by_x_stats=scale_aux_channels_by_x_stats,
                context_focus_scale=context_focus_scale,
            )
        elif compression_type == "cnn_vector":
            self.compression = Conv1DVectorField(channels, channels, hidden, depth, t_dim, max_period=max_period, time_scale=time_scale, time_use_2pi=time_use_2pi, use_film_time_conditioning=use_film_time_conditioning)
        else:
            raise ValueError(f"Unsupported compression_type: {compression_type}")
        if residual_arch == "standard":
            self.residual = Conv1DVectorField(channels, channels, hidden, residual_depth, t_dim, max_period=max_period, time_scale=time_scale, time_use_2pi=time_use_2pi, use_film_time_conditioning=use_film_time_conditioning)
        elif residual_arch == "shallow_cnn":
            self.residual = ShallowResidualVectorField1D(
                channels=channels,
                hidden=residual_hidden,
                t_dim=t_dim,
                kernel_size=residual_kernel_size,
                max_period=max_period,
                time_scale=time_scale,
                time_use_2pi=time_use_2pi,
            )
        else:
            raise ValueError(f"Unsupported residual_arch: {residual_arch}")

    def components(self, x: torch.Tensor, t: torch.Tensor, x_ref=None, context=None, focus=None):
        vt = self.transport(x, t)
        vc = self.compression(x, t, x_ref=x_ref, context=context, focus=focus)
        vr = self.residual(x, t)
        return vt, vc, vr

    def compose(self, vt: torch.Tensor, vc: torch.Tensor, vr: torch.Tensor, residual_gate: torch.Tensor | None = None, residual_multiplier: float = 1.0) -> torch.Tensor:
        v_tc = vt + self.gamma_compression * vc
        residual_scale = self.inference_residual_scale if residual_gate is None else residual_gate
        return v_tc + residual_multiplier * self.gamma_residual * residual_scale * vr

    def forward(self, x: torch.Tensor, t: torch.Tensor, x_ref=None, context=None, focus=None) -> torch.Tensor:
        vt, vc, vr = self.components(x, t, x_ref=x_ref, context=context, focus=focus)
        return self.compose(vt, vc, vr, residual_gate=None)


# Archived / inactive: legacy scalar-potential HCFM definitions remain disabled.
# The active DataHCFM above now implements scalar-potential compression directly.




class DataSpaceReconAE(nn.Module):
    """Small CNN reconstruction baseline used only for AE / reconstruction_mse_x."""

    def __init__(self, channels: int = 1, hidden: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden, channels, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def sample_data_fm_batch(x_train: torch.Tensor, batch_size: int, path_eps: float, generator=None):
    """
    Sample a Flow Matching batch.

    If a generator is provided, batch indices, base noise x0, and path time t
    use that generator. Otherwise PyTorch's global RNG is used.
    """
    idx = torch.randint(
        low=0,
        high=len(x_train),
        size=(batch_size,),
        device=x_train.device,
        generator=generator,
    )
    x1 = x_train[idx]
    if generator is None:
        x0 = torch.randn_like(x1)
    else:
        x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    t = torch.rand(batch_size, device=x1.device, dtype=x1.dtype, generator=generator)
    t = path_eps + (1.0 - 2.0 * path_eps) * t
    view_shape = (batch_size,) + (1,) * (x1.ndim - 1)
    xt = (1.0 - t).view(view_shape) * x0 + t.view(view_shape) * x1
    target = x1 - x0
    return x0, x1, t, xt, target



def data_energy(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x.flatten(1).pow(2).sum(dim=1)




def sample_hutchinson_probe_like(x, probe_type: str = "rademacher", generator=None):
    """Sample a Hutchinson probe, optionally using a deterministic generator."""
    return sample_probe_like(x, generator=generator, probe_type=probe_type)


def hutchinson_divergence_vector_field(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    n_probe: int = 1,
    probe_type: str = "rademacher",
    create_graph: bool = True,
    generator=None,
):
    """
    Estimate div v(x,t) = trace(dv/dx) using Hutchinson's estimator.

    For epsilon with E[epsilon epsilon^T] = I:
        trace(J_v) = E_epsilon [epsilon^T J_v epsilon]

    This avoids looping over all dimensions and requires only one VJP per
    probe instead of d VJPs.
    """
    x_req = x.requires_grad_(True) if create_graph else x.detach().clone().requires_grad_(True)
    div_estimates = []
    for _ in range(n_probe):
        eps = sample_hutchinson_probe_like(x_req, probe_type, generator=generator)
        y = model(x_req, t)
        inner = (y * eps).sum()
        grad = torch.autograd.grad(
            inner,
            x_req,
            create_graph=create_graph,
            retain_graph=True,
            only_inputs=True,
        )[0]
        div_estimates.append((grad * eps).flatten(1).sum(dim=1))
    return torch.stack(div_estimates, dim=0).mean(dim=0)


def rk4_step(field_fn, x: torch.Tensor, t: float, dt: float):
    t0 = torch.full((x.shape[0],), float(t), device=x.device)
    t1 = torch.full((x.shape[0],), float(t + 0.5 * dt), device=x.device)
    t2 = torch.full((x.shape[0],), float(t + dt), device=x.device)
    k1 = field_fn(x, t0)
    k2 = field_fn(x + 0.5 * dt * k1, t1)
    k3 = field_fn(x + 0.5 * dt * k2, t1)
    k4 = field_fn(x + dt * k3, t2)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def call_model_with_optional_x_ref(model, state: torch.Tensor, t: torch.Tensor, x_ref=None):
    ref = state if x_ref is None else x_ref
    try:
        return model(state, t, x_ref=ref)
    except TypeError:
        return model(state, t)


def _ode_time_grid(t_start: float, t_end: float, steps: int, device, dtype):
    return torch.linspace(float(t_start), float(t_end), int(steps) + 1, device=device, dtype=dtype)


def _torchdiffeq_integrate_dataspace(model, x0: torch.Tensor, times: torch.Tensor, method: str, atol: float, rtol: float, x_ref=None):
    if torchdiffeq is None:
        raise ImportError("ode_method requires torchdiffeq, but torchdiffeq is not installed/importable")

    class DataSpaceODEFunc(nn.Module):
        def __init__(self, model, x_ref=None):
            super().__init__()
            self.model = model
            self.x_ref = x_ref

        def forward(self, t, state):
            tt = torch.full((state.shape[0],), float(t.item()), device=state.device, dtype=state.dtype)
            ref = state if self.x_ref is None else self.x_ref
            with torch.no_grad():
                return call_model_with_optional_x_ref(self.model, state, tt, x_ref=ref)

    with torch.no_grad():
        return torchdiffeq.odeint(
            DataSpaceODEFunc(model, x_ref=x_ref),
            x0,
            times,
            method=method,
            atol=float(atol),
            rtol=float(rtol),
        )


def integrate_reverse_dataspace(model, x1: torch.Tensor, steps: int = 4, method: str = "rk4", return_path: bool = False, path_eps: float = 0.0, atol: float = 1e-4, rtol: float = 1e-4, x_ref=None):
    """
    Reverse integration maps a test window x_1 back to base space x_0.
    Base energy 0.5 ||x_0||^2 is a Jacobian-free anomaly score.

    Training samples t in [path_eps, 1-path_eps], so scoring should avoid
    evaluating the vector field at raw t=0 or t=1 endpoints.
    """
    x = x1
    t_start = 1.0 - float(path_eps)
    t_end = float(path_eps)
    path = [(t_start, x.detach())]

    def field_fn(state, tt):
        with torch.no_grad():
            ref = state if x_ref is None else x_ref
            return call_model_with_optional_x_ref(model, state, tt, x_ref=ref)

    if method in {"rk4", "euler"}:
        dt = (t_end - t_start) / steps
        for step in range(steps):
            t = t_start + step * dt
            if method == "rk4":
                x = rk4_step(field_fn, x, t, dt).detach()
            else:
                tt = torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
                x = (x + dt * field_fn(x, tt)).detach()
            path.append((t_start + (step + 1) * dt, x.detach()))
        return (x, path) if return_path else x

    times = _ode_time_grid(t_start, t_end, steps, x.device, x.dtype)
    sol = _torchdiffeq_integrate_dataspace(model, x, times, method=method, atol=atol, rtol=rtol, x_ref=x_ref)
    x = sol[-1].detach()
    path = [(float(times[j].detach().cpu().item()), sol[j].detach()) for j in range(len(times))]
    return (x, path) if return_path else x



def integrate_forward_dataspace(model, x0: torch.Tensor, steps: int = 4, method: str = "rk4", return_path: bool = False, path_eps: float = 0.0, atol: float = 1e-4, rtol: float = 1e-4, x_ref=None):
    """Forward fixed/adaptive ODE integration from base space to data space."""
    x = x0
    t_start = float(path_eps)
    t_end = 1.0 - float(path_eps)
    path = [(t_start, x.detach())]

    def field_fn(state, tt):
        with torch.no_grad():
            ref = state if x_ref is None else x_ref
            return call_model_with_optional_x_ref(model, state, tt, x_ref=ref)

    if method in {"rk4", "euler"}:
        dt = (t_end - t_start) / steps
        for step in range(steps):
            t = t_start + step * dt
            if method == "rk4":
                x = rk4_step(field_fn, x, t, dt).detach()
            else:
                tt = torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
                x = (x + dt * field_fn(x, tt)).detach()
            path.append((t_start + (step + 1) * dt, x.detach()))
        return (x, path) if return_path else x

    times = _ode_time_grid(t_start, t_end, steps, x.device, x.dtype)
    sol = _torchdiffeq_integrate_dataspace(model, x, times, method=method, atol=atol, rtol=rtol, x_ref=x_ref)
    x = sol[-1].detach()
    path = [(float(times[j].detach().cpu().item()), sol[j].detach()) for j in range(len(times))]
    return (x, path) if return_path else x



def exact_divergence_vector_field(model, x: torch.Tensor, t: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
    """Exact trace(dv/dx) by looping over flattened output dimensions. Audit only."""
    x_req = x if x.requires_grad else x.detach().clone().requires_grad_(True)
    y = model(x_req, t)
    y_flat = y.flatten(1)
    div = torch.zeros(x_req.shape[0], device=x_req.device, dtype=x_req.dtype)
    for j in range(y_flat.shape[1]):
        grad_j = torch.autograd.grad(
            y_flat[:, j].sum(),
            x_req,
            create_graph=create_graph,
            retain_graph=True,
            only_inputs=True,
        )[0]
        div = div + grad_j.flatten(1)[:, j]
    return div


# %% notebook cell 17
def scalarize_for_log(x):
    """Convert a scalar tensor/value for sparse logging only."""
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().item()
    return float(x)


def format_loss_components(components):
    """Format optional same-line loss components for compact training logs."""
    return " ".join(
        f"{k}={scalarize_for_log(v):.4g}"
        for k, v in components.items()
        if v is not None
    )


def log_train_step(method_name: str, step: int, total_steps: int, loss, cfg, components=None, debug_payload=None):
    """Print one compact training line, with optional components/debug details."""
    msg = f"[{method_name}] step {step:05d}/{total_steps} loss={scalarize_for_log(loss):.6f}"
    if getattr(cfg, "print_loss_components", False) and components:
        component_str = format_loss_components(components)
        if component_str:
            msg = f"{msg} {component_str}"
    print(msg, flush=True)
    if getattr(cfg, "verbose_train_logs", False) and debug_payload is not None:
        print(debug_payload, flush=True)


def append_train_history(hist, step: int, loss, cfg, components=None):
    """Store sparse checkpoint history; components are opt-in to avoid syncs."""
    row = {"step": step, "loss": scalarize_for_log(loss)}
    if getattr(cfg, "print_loss_components", False) and components:
        for key, value in components.items():
            if value is not None:
                row[key] = scalarize_for_log(value)
    hist.append(row)


def maybe_compile_model(model, cfg):
    """Optionally compile a model with torch.compile for long runs."""
    if getattr(cfg, "use_compile", False):
        try:
            model = torch.compile(model)
            print("Compiled model with torch.compile")
        except Exception as e:
            print(f"torch.compile failed; continuing without compile: {e}")
    return model


def make_fm_train_generator(x_train: torch.Tensor, cfg):
    """Create the optional dedicated FM sampling generator for training."""
    if not getattr(cfg, "use_dedicated_fm_generator", True):
        return None
    gen = torch.Generator(device=x_train.device)
    gen.manual_seed(cfg.seed + cfg.fm_sampling_seed_offset)
    return gen




def compute_hcfm_mismatch_gate(
    target,
    v_tc,
    gate_min=0.10,
    gate_threshold=0.0,
    gate_power=1.0,
    eps=1e-8,
    return_diagnostics=False,
):
    with torch.no_grad():
        tc_error = target - v_tc.detach()
        tc_error_energy = tc_error.flatten(1).pow(2).sum(dim=1, keepdim=True)
        target_energy = target.flatten(1).pow(2).sum(dim=1, keepdim=True).clamp_min(eps)
        raw_gate = (tc_error_energy / target_energy).clamp(0.0, 1.0)
        active_gate = ((raw_gate - gate_threshold) / (1.0 - gate_threshold + eps)).clamp(0.0, 1.0)
        shaped_gate = active_gate.pow(gate_power)
        gate_unviewed = gate_min + (1.0 - gate_min) * shaped_gate
        gate = gate_unviewed.view((target.shape[0],) + (1,) * (target.ndim - 1))
        if return_diagnostics:
            diagnostics = {
                "raw_gate_mean": raw_gate.mean(),
                "active_gate_mean": active_gate.mean(),
                "shaped_gate_mean": shaped_gate.mean(),
                "gate_mean": gate_unviewed.mean(),
                "gate_min": gate_unviewed.min(),
                "gate_max": gate_unviewed.max(),
            }
            return gate, diagnostics
    return gate


def train_dataspace_recon_ae(train_x_seq: torch.Tensor, cfg):
    model = DataSpaceReconAE(cfg.channels, hidden=cfg.ae_hidden).to(train_x_seq.device)
    model = maybe_compile_model(model, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    hist = []
    start_time = time.time()
    train_steps = int(cfg.ae_train_steps)
    for step in range(train_steps):
        idx = torch.randint(0, len(train_x_seq), (cfg.ae_batch_size,), device=train_x_seq.device)
        xb = train_x_seq[idx]
        recon = model(xb)
        loss = F.mse_loss(recon, xb)
        if not torch.isfinite(loss):
            print(f"Non-finite loss at step {step}")
            break
        opt.zero_grad(set_to_none=getattr(cfg, "zero_grad_set_to_none", True))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        should_log = (step % cfg.print_every == 0) or (step == train_steps - 1)
        if should_log:
            append_train_history(hist, step, loss, cfg)
            log_train_step("Data-space AE", step, train_steps, loss, cfg)
    elapsed = time.time() - start_time
    steps_per_sec = train_steps / max(elapsed, 1e-12)
    print(f"Data-space AE training time: {elapsed:.2f} seconds ({steps_per_sec:.2f} steps/sec)")
    hist_df = pd.DataFrame(hist) if hist else pd.DataFrame([{}])
    hist_df["train_time_sec"] = elapsed
    hist_df["steps_per_sec"] = steps_per_sec
    model.eval()
    return model, hist_df


@torch.no_grad()
def score_reconstruction_mse_x(model, x_all: torch.Tensor, batch_size: int):
    outs = []
    for i in range(0, len(x_all), batch_size):
        xb = x_all[i:i + batch_size]
        recon = model(xb)
        outs.append((recon - xb).flatten(1).pow(2).mean(dim=1).detach().cpu())
    return torch.cat(outs)


# Vanilla/FDM training helpers removed from the active HCFM-only multivariate notebook.



def residual_ramp_factor(step: int, cfg) -> float:
    """Return residual multiplier for warmup/ramp scheduling."""
    warmup = int(getattr(cfg, "hcfm_residual_warmup_iters", 0))
    ramp = int(getattr(cfg, "hcfm_residual_ramp_iters", 0))
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    return float(min(1.0, max(0.0, (step - warmup + 1) / ramp)))


def train_data_hcfm(train_x_seq: torch.Tensor, cfg):
    """Train Data HCFM with transport/compression/residual decomposition."""
    model = DataHCFM(
        channels=cfg.channels,
        length=cfg.length,
        hidden=cfg.hidden,
        depth=cfg.base_depth,
        t_dim=cfg.time_emb_dim,
        transport_rank=cfg.hcfm_transport_rank,
        residual_depth=cfg.hcfm_residual_depth,
        residual_arch=cfg.hcfm_residual_arch,
        residual_hidden=cfg.hcfm_residual_hidden,
        residual_kernel_size=cfg.hcfm_residual_kernel_size,
        gamma_compression=cfg.hcfm_gamma_compression,
        gamma_residual=cfg.hcfm_gamma_residual,
        max_period=cfg.time_emb_max_period,
        time_scale=cfg.hcfm_time_emb_scale,
        time_use_2pi=cfg.hcfm_time_emb_use_2pi,
        use_film_time_conditioning=cfg.hcfm_use_film_time_conditioning,
        compression_type=cfg.hcfm_compression_type,
        use_residual_mismatch_gate=cfg.hcfm_use_residual_mismatch_gate,
        residual_gate_min=cfg.hcfm_residual_gate_min,
        inference_residual_scale_mode=cfg.hcfm_inference_residual_scale_mode,
        potential_norm_type=cfg.hcfm_potential_norm_type,
        potential_residual_scale=cfg.hcfm_potential_residual_scale,
        potential_backbone=cfg.hcfm_potential_backbone,
        mixer_depth=cfg.hcfm_mixer_depth,
        mixer_token_mlp_dim=cfg.hcfm_mixer_token_mlp_dim,
        mixer_channel_mlp_dim=cfg.hcfm_mixer_channel_mlp_dim,
        mixer_norm_type=cfg.hcfm_mixer_norm_type,
        mixer_residual_scale=cfg.hcfm_mixer_residual_scale,
        potential_scale=cfg.hcfm_potential_scale,
        potential_out_init=cfg.hcfm_potential_out_init,
        potential_out_init_std=cfg.hcfm_potential_out_init_std,
        use_position_embedding=cfg.hcfm_use_position_embedding,
        position_emb_dim=cfg.hcfm_position_emb_dim,
        position_proj_dim=cfg.hcfm_position_proj_dim,
        position_max_period=cfg.hcfm_position_max_period,
        position_use_integer_positions=cfg.hcfm_position_use_integer_positions,
        use_fft_features=cfg.hcfm_use_fft_features,
        fft_feature_mode=cfg.hcfm_fft_feature_mode,
        fft_num_bands=cfg.hcfm_fft_num_bands,
        fft_include_centroid=cfg.hcfm_fft_include_centroid,
        fft_include_entropy=cfg.hcfm_fft_include_entropy,
        fft_include_dominant=cfg.hcfm_fft_include_dominant,
        fft_log_magnitude=cfg.hcfm_fft_log_magnitude,
        fft_eps=cfg.hcfm_fft_eps,
        fft_detach_features=cfg.hcfm_fft_detach_features,
        normalize_fft_features=cfg.hcfm_normalize_fft_features,
        scale_aux_channels_by_x_stats=cfg.hcfm_scale_aux_channels_by_x_stats,
        context_focus_scale=cfg.hcfm_context_focus_scale,
    ).to(train_x_seq.device)
    model = maybe_compile_model(model, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.hcfm_iters), eta_min=cfg.lr * 0.1)
    residual_module = getattr(model, "_orig_mod", model).residual
    freeze_residual = bool(cfg.hcfm_freeze_residual_during_warmup and cfg.hcfm_residual_warmup_iters > 0)
    if freeze_residual:
        for p in residual_module.parameters():
            p.requires_grad_(False)
    residual_requires_grad = [p.requires_grad for p in residual_module.parameters()]
    print(
        "Data HCFM residual schedule: "
        f"arch={cfg.hcfm_residual_arch}, hidden={cfg.hcfm_residual_hidden}, kernel={cfg.hcfm_residual_kernel_size}, "
        f"warmup={cfg.hcfm_residual_warmup_iters}, ramp={cfg.hcfm_residual_ramp_iters}, "
        f"freeze_warmup={cfg.hcfm_freeze_residual_during_warmup}, gamma_residual={cfg.hcfm_gamma_residual}"
    )
    if residual_requires_grad:
        print(f"Initial residual requires_grad: any={any(residual_requires_grad)}, all={all(residual_requires_grad)}")
    hist = []
    start_time = time.time()
    fm_gen = make_fm_train_generator(train_x_seq, cfg)
    for step in range(cfg.hcfm_iters):
        if freeze_residual and step == cfg.hcfm_residual_warmup_iters:
            for p in residual_module.parameters():
                p.requires_grad_(True)
            print(f"Residual head enabled/unfrozen at step {step}")
        residual_ramp = residual_ramp_factor(step, cfg)
        gamma_residual_effective = cfg.hcfm_gamma_residual * residual_ramp
        x0, x1, t, xt, target = sample_data_fm_batch(train_x_seq, cfg.hcfm_batch_size, cfg.path_eps, generator=fm_gen)
        xt_req = xt.detach().requires_grad_(True)
        vt, vc, vr = model.components(xt_req, t, x_ref=x1)
        v_tc = vt + model.gamma_compression * vc
        gate_diag = None
        if cfg.hcfm_use_residual_mismatch_gate:
            gate, gate_diag = compute_hcfm_mismatch_gate(
                target=target,
                v_tc=v_tc,
                gate_min=cfg.hcfm_residual_gate_min,
                gate_threshold=cfg.hcfm_residual_gate_threshold,
                gate_power=cfg.hcfm_residual_gate_power,
                return_diagnostics=True,
            )
        else:
            gate = None
        v_total = model.compose(vt, vc, vr, residual_gate=gate, residual_multiplier=residual_ramp)
        fm_loss = F.mse_loss(v_total, target)
        residual_energy = vr.flatten(1).pow(2).mean()
        compression_energy = vc.flatten(1).pow(2).mean()
        flat_t = vt.flatten(1)
        flat_c = vc.flatten(1)
        flat_r = vr.flatten(1)
        cos2_t_c_loss = F.cosine_similarity(flat_t, flat_c, dim=1, eps=1e-8).pow(2).mean()
        cos2_t_r_loss = F.cosine_similarity(flat_t, flat_r, dim=1, eps=1e-8).pow(2).mean()
        cos2_c_r_loss = F.cosine_similarity(flat_c, flat_r, dim=1, eps=1e-8).pow(2).mean()
        if residual_ramp <= 0.0:
            ortho_loss = cos2_t_c_loss
        else:
            ortho_loss = cos2_t_c_loss + residual_ramp * cos2_t_r_loss + residual_ramp * cos2_c_r_loss
        should_log = (step % cfg.print_every == 0) or (step == cfg.hcfm_iters - 1)
        # Scalar-potential compression makes divergence a Laplacian, so computing
        # it is second-order and expensive. Only compute it when it contributes to
        # an active loss term or residual-div bound.
        need_compression_div_grad = cfg.hcfm_lambda_compression_div > 0
        need_residual_div_bound = (
            cfg.hcfm_use_residual_div_bound
            or cfg.hcfm_lambda_residual_div_bound > 0
        )
        need_physics_residual = (
            cfg.hcfm_use_physics_residual_loss
            and cfg.hcfm_lambda_physics_residual > 0.0
        )
        need_compression_div = need_compression_div_grad or need_residual_div_bound or need_physics_residual
        div_compression = None
        div_loss = zero = fm_loss.new_tensor(0.0)
        physics_residual_loss = zero
        physics_residual_ratio_rms = zero
        if need_compression_div:
            div_compression = hutchinson_divergence_vector_field(
                lambda x_in, t_in: model.compression(x_in, t_in, x_ref=x1),
                xt_req,
                t,
                n_probe=cfg.train_n_probe,
                probe_type=cfg.probe_type,
                create_graph=(
                    need_compression_div_grad
                    or need_residual_div_bound
                    or (need_physics_residual and not cfg.hcfm_physics_residual_detach_compression)
                ),
            )
            div_loss = div_compression.pow(2).mean()
        need_residual_div = (
            cfg.hcfm_lambda_residual_div > 0
            or cfg.hcfm_use_residual_div_bound
            or cfg.hcfm_lambda_residual_div_bound > 0
            or need_physics_residual
        )
        div_residual = None
        residual_div_loss = None
        residual_div_bound_loss = None
        if need_residual_div:
            div_residual = hutchinson_divergence_vector_field(
                lambda x_in, t_in: model.residual(x_in, t_in),
                xt_req,
                t,
                n_probe=cfg.train_n_probe,
                probe_type=cfg.probe_type,
                create_graph=True,
            )
            gate_vec = torch.ones_like(div_residual) if gate is None else gate.detach().view(-1)
            residual_div_loss = (gate_vec.pow(2) * div_residual.pow(2)).mean()
            if need_residual_div_bound:
                residual_div_eff = gate_vec * div_residual.abs()
                if div_compression is None:
                    div_compression = hutchinson_divergence_vector_field(
                        lambda x_in, t_in: model.compression(x_in, t_in, x_ref=x1),
                        xt_req,
                        t,
                        n_probe=cfg.train_n_probe,
                        probe_type=cfg.probe_type,
                        create_graph=True,
                    )
                compression_div_ref = div_compression.detach().abs()
                residual_div_bound_loss = F.relu(
                    residual_div_eff - cfg.hcfm_div_bound_kappa * compression_div_ref
                ).pow(2).mean()
        if need_physics_residual:
            if div_residual is None:
                div_residual = hutchinson_divergence_vector_field(
                    lambda x_in, t_in: model.residual(x_in, t_in),
                    xt_req,
                    t,
                    n_probe=cfg.train_n_probe,
                    probe_type=cfg.probe_type,
                    create_graph=True,
                )
            if div_compression is None:
                div_compression = hutchinson_divergence_vector_field(
                    lambda x_in, t_in: model.compression(x_in, t_in, x_ref=x1),
                    xt_req,
                    t,
                    n_probe=cfg.train_n_probe,
                    probe_type=cfg.probe_type,
                    create_graph=not cfg.hcfm_physics_residual_detach_compression,
                )
            compression_ref = div_compression.abs()
            if cfg.hcfm_physics_residual_detach_compression:
                compression_ref = compression_ref.detach()
            residual_abs = div_residual.abs()
            phys_eps = cfg.hcfm_physics_residual_eps
            if cfg.hcfm_physics_residual_mode == "ratio":
                physics_residual_loss = (
                    residual_abs.pow(2).mean()
                    / compression_ref.pow(2).mean().clamp_min(phys_eps)
                )
            elif cfg.hcfm_physics_residual_mode == "hinge":
                physics_residual_loss = F.relu(
                    residual_abs - cfg.hcfm_physics_residual_kappa * compression_ref
                ).pow(2).mean()
            else:
                raise ValueError(f"unknown hcfm_physics_residual_mode: {cfg.hcfm_physics_residual_mode}")
            physics_residual_ratio_rms = (
                div_residual.pow(2).mean().sqrt()
                / div_compression.detach().pow(2).mean().sqrt().clamp_min(cfg.hcfm_physics_residual_eps)
            )
        zero = fm_loss.new_tensor(0.0)
        residual_energy_weight = cfg.hcfm_lambda_residual_energy * residual_ramp
        total_loss = (
            fm_loss
            + residual_energy_weight * residual_energy
            + cfg.hcfm_lambda_ortho * ortho_loss
            + cfg.hcfm_lambda_compression_div * div_loss
            + cfg.hcfm_lambda_residual_div * (residual_div_loss if residual_div_loss is not None else zero)
            + cfg.hcfm_lambda_residual_div_bound * (residual_div_bound_loss if residual_div_bound_loss is not None else zero)
            + cfg.hcfm_lambda_physics_residual * physics_residual_loss
        )
        if not torch.isfinite(total_loss):
            print(f"Non-finite loss at step {step}")
            break
        opt.zero_grad(set_to_none=getattr(cfg, "zero_grad_set_to_none", True))
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        scheduler.step()
        if should_log:
            eps = 1e-8
            with torch.no_grad():
                fm_transport = F.mse_loss(vt, target)
                fm_transport_compression = F.mse_loss(v_tc, target)
                target_mse = target.flatten(1).pow(2).mean()
                relative_fm_loss = fm_loss / (target_mse + eps)
                norm_T = flat_t.norm(dim=1).mean()
                norm_C = flat_c.norm(dim=1).mean()
                norm_scaled_C = (model.gamma_compression * vc).flatten(1).norm(dim=1).mean()
                norm_R = flat_r.norm(dim=1).mean()
                norm_scaled_R = (gamma_residual_effective * vr).flatten(1).norm(dim=1).mean()
                norm_target = target.flatten(1).norm(dim=1).mean()
                norm_total = v_total.flatten(1).norm(dim=1).mean()
                cos2_T_C = F.cosine_similarity(flat_t, flat_c, dim=1, eps=eps).pow(2).mean()
                cos2_T_R = F.cosine_similarity(flat_t, flat_r, dim=1, eps=eps).pow(2).mean()
                cos2_C_R = F.cosine_similarity(flat_c, flat_r, dim=1, eps=eps).pow(2).mean()
            audit_mode = bool(getattr(cfg, "run_scalar_potential_audit", False))
            components = {
                "fm_T": fm_transport,
                "fm_TC": fm_transport_compression,
                "fm": fm_loss,
                "residual_ramp": residual_ramp,
                "gamma_residual_effective": gamma_residual_effective,
            }
            if need_physics_residual:
                components.update({
                    "physics_residual_loss_raw": physics_residual_loss,
                    "physics_residual_loss_weighted": cfg.hcfm_lambda_physics_residual * physics_residual_loss,
                    "physics_residual_ratio_rms": physics_residual_ratio_rms,
                    "residual_div_rms": div_residual.pow(2).mean().sqrt(),
                    "compression_div_rms": div_compression.detach().pow(2).mean().sqrt(),
                })
            if audit_mode:
                components.update({
                    "target_mse": target_mse,
                    "relative_fm_loss": relative_fm_loss,
                    "residual_energy_raw": residual_energy,
                    "residual_energy_weighted": residual_energy_weight * residual_energy,
                    "compression_energy_raw": compression_energy,
                    "compression_energy_weighted": cfg.hcfm_lambda_compression_energy * compression_energy,
                    "ortho_loss_raw": ortho_loss,
                    "ortho_loss_weighted": cfg.hcfm_lambda_ortho * ortho_loss,
                    "weighted_residual_energy_over_fm": (residual_energy_weight * residual_energy) / (fm_loss + eps),
                    "weighted_ortho_over_fm": (cfg.hcfm_lambda_ortho * ortho_loss) / (fm_loss + eps),
                    "weighted_compression_energy_over_fm": (cfg.hcfm_lambda_compression_energy * compression_energy) / (fm_loss + eps),
                    "fm_TC_over_fm_T": fm_transport_compression / (fm_transport + eps),
                    "fm_over_fm_TC": fm_loss / (fm_transport_compression + eps),
                    "norm_T": norm_T,
                    "norm_C": norm_C,
                    "norm_scaled_C": norm_scaled_C,
                    "norm_R": norm_R,
                    "norm_scaled_R": norm_scaled_R,
                    "norm_target": norm_target,
                    "norm_total": norm_total,
                    "scaled_C_over_T": norm_scaled_C / (norm_T + eps),
                    "scaled_R_over_scaled_C": norm_scaled_R / (norm_scaled_C + eps),
                    "scaled_R_over_total": norm_scaled_R / (norm_total + eps),
                    "scaled_C_over_target": norm_scaled_C / (norm_target + eps),
                    "scaled_R_over_target": norm_scaled_R / (norm_target + eps),
                    "cos2_T_C": cos2_T_C,
                    "cos2_T_R": cos2_T_R,
                    "cos2_C_R": cos2_C_R,
                    "total_loss": total_loss,
                })
                if need_compression_div:
                    components["compression_div_loss_raw"] = div_loss
                    components["compression_div_loss_weighted"] = cfg.hcfm_lambda_compression_div * div_loss
                if residual_div_loss is not None:
                    components["residual_div_loss_raw"] = residual_div_loss
                    components["residual_div_loss_weighted"] = cfg.hcfm_lambda_residual_div * residual_div_loss
                if residual_div_bound_loss is not None:
                    components["residual_div_bound_loss_raw"] = residual_div_bound_loss
                    components["residual_div_bound_loss_weighted"] = cfg.hcfm_lambda_residual_div_bound * residual_div_bound_loss
            else:
                if residual_energy_weight > 0:
                    components["res"] = residual_energy
                    components["res_w"] = residual_energy_weight * residual_energy
                if cfg.hcfm_lambda_ortho > 0:
                    components["ortho"] = ortho_loss
                    components["cos2_TC"] = cos2_t_c_loss
                    if residual_ramp > 0:
                        components["cos2_TR"] = cos2_t_r_loss
                        components["cos2_CR"] = cos2_c_r_loss
                if cfg.hcfm_lambda_compression_div > 0:
                    components["div"] = div_loss
                if cfg.hcfm_lambda_residual_div > 0:
                    components["residual_div"] = residual_div_loss
                if cfg.hcfm_lambda_residual_div_bound > 0:
                    components["residual_div_bound"] = residual_div_bound_loss
            append_train_history(hist, step, total_loss, cfg, components=components)
            log_train_step("Data HCFM", step, cfg.hcfm_iters, total_loss, cfg, components=components)
    elapsed = time.time() - start_time
    steps_per_sec = cfg.hcfm_iters / max(elapsed, 1e-12)
    print(f"Data HCFM training time: {elapsed:.2f} seconds ({steps_per_sec:.2f} steps/sec)")
    hist_df = pd.DataFrame(hist) if hist else pd.DataFrame([{}])
    hist_df["train_time_sec"] = elapsed
    hist_df["steps_per_sec"] = steps_per_sec
    return model, hist_df


# %% notebook cell 19
@torch.no_grad()
def score_dataspace_base_energy(model, x_all: torch.Tensor, batch_size: int, cfg, desc: str):
    model.eval()
    outs = []
    iterator = tqdm(range(0, len(x_all), batch_size), total=int(np.ceil(len(x_all) / batch_size)), desc=desc, leave=False)
    for i in iterator:
        x0_hat = integrate_reverse_dataspace(model, x_all[i:i + batch_size], steps=cfg.ode_steps, method=cfg.ode_method, path_eps=cfg.path_eps, atol=cfg.ode_atol, rtol=cfg.ode_rtol, x_ref=x_all[i:i + batch_size])
        outs.append(data_energy(x0_hat).detach().cpu())
    return torch.cat(outs)


def score_fm_consistency_data(
    model,
    x_all: torch.Tensor,
    batch_size: int,
    K: int = 4,
    path_eps: float = 1e-3,
    device=None,
    model_kind: str = "vector_field",
    desc: str = "FM consistency",
):
    """
    Jacobian-free local flow-matching consistency score.

    For each test window x1, sample K noise endpoints x0 and times t,
    construct xt = (1-t)x0 + t*x1, and measure how well the learned
    vector field predicts the conditional FM target x1 - x0.

    This score does not use ODE integration or divergence/Jacobian traces.
    It is a local normality score: anomalous windows should be harder for
    a model trained on normal windows to explain under the FM target.
    """
    model.eval()
    eval_gen = make_score_generator(
        dataspace_cfg.seed,
        method_name="FM consistency",
        score_name=desc,
        device=x_all.device,
    )
    outs = []
    iterator = tqdm(range(0, len(x_all), batch_size), total=int(np.ceil(len(x_all) / batch_size)), desc=desc, leave=False)
    for i in iterator:
        x1 = x_all[i:i + batch_size]
        if device is not None:
            x1 = x1.to(device)
        scores = []
        for _ in range(K):
            x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=eval_gen)
            t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype, generator=eval_gen) * (1 - 2 * path_eps) + path_eps
            view_shape = (t.shape[0],) + (1,) * (x1.ndim - 1)
            xt = (1 - t).view(view_shape) * x0 + t.view(view_shape) * x1
            target = x1 - x0
            if model_kind != "vector_field":
                raise ValueError(f"unsupported model_kind: {model_kind}")
            with torch.no_grad():
                pred = model(xt, t)
                score_k = (pred - target).flatten(1).pow(2).mean(dim=1).detach().cpu()
            scores.append(score_k)
        outs.append(torch.stack(scores, dim=0).mean(dim=0))
    return torch.cat(outs).numpy()


# Backward-compatible alias for earlier notebook cells.
def score_dataspace_fm_consistency(model, x_all: torch.Tensor, batch_size: int, cfg, desc: str):
    return score_fm_consistency_data(
        model,
        x_all,
        batch_size=batch_size,
        K=cfg.fm_consistency_K,
        path_eps=cfg.path_eps,
        model_kind="vector_field",
        desc=desc,
    )


def score_hcfm_fm_consistency_data(model: DataHCFM, x_all: torch.Tensor, batch_size: int, cfg, desc: str):
    """HCFM FM-consistency score that can use the target-dependent mismatch gate."""
    model.eval()
    eval_gen = make_score_generator(
        cfg.seed,
        method_name="Data HCFM",
        score_name=desc,
        device=x_all.device,
    )
    outs = []
    iterator = tqdm(range(0, len(x_all), batch_size), total=int(np.ceil(len(x_all) / batch_size)), desc=desc, leave=False)
    for i in iterator:
        x1 = x_all[i:i + batch_size]
        scores = []
        for _ in range(cfg.fm_consistency_K):
            x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=eval_gen)
            t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype, generator=eval_gen) * (1 - 2 * cfg.path_eps) + cfg.path_eps
            view_shape = (t.shape[0],) + (1,) * (x1.ndim - 1)
            xt = (1 - t).view(view_shape) * x0 + t.view(view_shape) * x1
            target = x1 - x0
            with torch.no_grad():
                vt, vc, vr = model.components(xt, t, x_ref=x1)
                v_tc = vt + model.gamma_compression * vc
                if cfg.hcfm_use_residual_mismatch_gate:
                    gate = compute_hcfm_mismatch_gate(
                        target,
                        v_tc,
                        gate_min=cfg.hcfm_residual_gate_min,
                        gate_threshold=cfg.hcfm_residual_gate_threshold,
                        gate_power=cfg.hcfm_residual_gate_power,
                    )
                else:
                    gate = None
                pred = model.compose(vt, vc, vr, residual_gate=gate)
                score_k = (pred - target).flatten(1).pow(2).mean(dim=1).detach().cpu()
            scores.append(score_k)
        outs.append(torch.stack(scores, dim=0).mean(dim=0))
    return torch.cat(outs).numpy()


def trapz_integral_from_steps(values_steps, time_steps):
    """Trapezoidal integral over pathwise scalar values sorted by ascending time."""
    if not values_steps:
        raise ValueError("values_steps must be non-empty")
    order = np.argsort(np.asarray(time_steps, dtype=np.float64))
    values_sorted = torch.stack([values_steps[j] for j in order], dim=0)
    times_sorted = torch.tensor(
        np.asarray(time_steps, dtype=np.float32)[order],
        device=values_sorted.device,
        dtype=values_sorted.dtype,
    )
    if values_sorted.shape[0] < 2:
        return values_sorted.mean(dim=0)
    dt = (times_sorted[1:] - times_sorted[:-1]).view(-1, 1)
    return (0.5 * dt * (values_sorted[:-1] + values_sorted[1:])).sum(dim=0)


def score_data_hcfm(
    model: DataHCFM,
    x_all: torch.Tensor,
    batch_size: int,
    cfg,
    desc: str,
    compute_residual_signed_div: bool = False,
    compute_head_divergence_diagnostics: bool = False,
    compute_extra_compression_variants: bool = False,
):
    """
    Score Data HCFM without full-field likelihood. Base energy uses reverse
    integration. Component scores are averaged along the reverse path.

    Compression divergence is returned with signed, positive, negative,
    and absolute variants. The signed/one-sided variants are important because
    magnitude-only divergence can be anti-correlated with anomaly evidence.
    Transport and residual divergence are diagnostics only and run when
    cfg.run_hcfm_head_divergence_diagnostics=True. effective_residual_div_abs
    reflects the residual scale actually used by ODE inference.
    """
    model.eval()
    comp_div_gen = make_score_generator(cfg.seed, method_name="Data HCFM", score_name="compression_div_abs_x", device=x_all.device)
    run_head_div = bool(compute_head_divergence_diagnostics)
    run_residual_signed_div = bool(compute_residual_signed_div) or run_head_div
    run_extra_compression = bool(compute_extra_compression_variants)
    run_trapz_cnf = bool(getattr(cfg, "compute_trapz_cnf_integral_scores", True))
    run_trapz_residual = bool(run_trapz_cnf and getattr(cfg, "trapz_cnf_include_residual", False) and run_residual_signed_div)
    transport_div_gen = make_score_generator(cfg.seed, method_name="Data HCFM", score_name="transport_div_abs_x", device=x_all.device) if run_head_div else None
    residual_div_gen = make_score_generator(cfg.seed, method_name="Data HCFM", score_name="residual_div_abs_x", device=x_all.device) if run_head_div else None
    residual_signed_div_gen = make_score_generator(cfg.seed, method_name="Data HCFM", score_name="residual_div_signed_x", device=x_all.device) if run_residual_signed_div and not run_head_div else residual_div_gen

    base_out, transport_energy_out, compression_energy_out, residual_energy_out = [], [], [], []
    compression_div_abs_out, compression_div_signed_out = [], []
    compression_div_pos_out, compression_div_neg_out = [], []
    compression_div_sq_out, compression_div_neg_sq_out = [], []
    transport_div_abs_out, residual_div_abs_out, effective_residual_div_abs_out = [], [], []
    residual_div_signed_out = []
    compression_div_trapz_out, residual_div_trapz_out = [], []
    cnf_base_plus_compression_trapz_out, cnf_base_minus_compression_trapz_out = [], []
    cnf_base_plus_full_trapz_out, cnf_base_minus_full_trapz_out = [], []
    iterator = tqdm(range(0, len(x_all), batch_size), total=int(np.ceil(len(x_all) / batch_size)), desc=desc, leave=False)
    for i in iterator:
        xb = x_all[i:i + batch_size]
        x0_hat, path_states = integrate_reverse_dataspace(model, xb, steps=cfg.ode_steps, method=cfg.ode_method, return_path=True, path_eps=cfg.path_eps, atol=cfg.ode_atol, rtol=cfg.ode_rtol, x_ref=xb)
        base_energy_cpu = data_energy(x0_hat).detach().cpu()
        base_out.append(base_energy_cpu)
        transport_steps, compression_steps, residual_steps = [], [], []
        div_abs_steps, div_signed_steps, div_pos_steps, div_neg_steps = [], [], [], []
        div_sq_steps, div_neg_sq_steps = [], []
        transport_div_steps, residual_div_steps, effective_residual_div_steps = [], [], []
        residual_div_signed_steps = []
        path_time_steps = []
        for t_value, x_t in path_states:
            t_safe = min(max(float(t_value), cfg.path_eps), 1.0 - cfg.path_eps)
            path_time_steps.append(t_safe)
            t = torch.full((x_t.shape[0],), t_safe, device=x_t.device, dtype=x_t.dtype)
            with torch.no_grad():
                vt = model.transport(x_t, t)
                transport_steps.append(vt.flatten(1).pow(2).sum(dim=1).detach().cpu())
                vr = model.residual(x_t, t)
                residual_steps.append(vr.flatten(1).pow(2).sum(dim=1).detach().cpu())
                del vt, vr
            with torch.enable_grad():
                x_comp = x_t.detach().clone().requires_grad_(True)
                vc = model.compression(x_comp, t, x_ref=xb)
                compression_steps.append(vc.flatten(1).pow(2).sum(dim=1).detach().cpu())
                del x_comp, vc
                div_c = hutchinson_divergence_vector_field(
                    lambda x_in, t_in: model.compression(x_in, t_in, x_ref=xb),
                    x_t,
                    t,
                    n_probe=cfg.eval_n_probe,
                    probe_type=cfg.probe_type,
                    create_graph=False,
                    generator=comp_div_gen,
                )
                div_r = None
                if run_head_div:
                    div_t = hutchinson_divergence_vector_field(
                        lambda x_in, t_in: model.transport(x_in, t_in),
                        x_t,
                        t,
                        n_probe=cfg.eval_n_probe,
                        probe_type=cfg.probe_type,
                        create_graph=False,
                        generator=transport_div_gen,
                    )
                if run_residual_signed_div:
                    div_r = hutchinson_divergence_vector_field(
                        lambda x_in, t_in: model.residual(x_in, t_in),
                        x_t,
                        t,
                        n_probe=cfg.eval_n_probe,
                        probe_type=cfg.probe_type,
                        create_graph=False,
                        generator=residual_signed_div_gen,
                    )
            div_c_cpu = div_c.detach().cpu()
            div_signed_steps.append(div_c_cpu)
            div_neg_sq_steps.append(-div_c_cpu.pow(2))
            if run_extra_compression:
                div_abs_steps.append(div_c_cpu.abs())
                div_pos_steps.append(div_c_cpu.clamp_min(0.0))
                div_neg_steps.append((-div_c_cpu).clamp_min(0.0))
                div_sq_steps.append(div_c_cpu.pow(2))
            if run_head_div:
                transport_div_steps.append(div_t.abs().detach().cpu())
            if run_residual_signed_div:
                residual_div_signed_steps.append(div_r.detach().cpu())
                if run_head_div:
                    residual_div_steps.append(div_r.abs().detach().cpu())
                    effective_residual_div_steps.append((float(model.inference_residual_scale) * div_r.abs()).detach().cpu())
            if run_head_div:
                del div_t
            if run_residual_signed_div:
                del div_r
            del div_c
        transport_energy_out.append(torch.stack(transport_steps, dim=0).mean(dim=0))
        compression_energy_out.append(torch.stack(compression_steps, dim=0).mean(dim=0))
        residual_energy_out.append(torch.stack(residual_steps, dim=0).mean(dim=0))
        compression_div_signed_out.append(torch.stack(div_signed_steps, dim=0).mean(dim=0))
        compression_div_neg_sq_out.append(torch.stack(div_neg_sq_steps, dim=0).mean(dim=0))
        if run_extra_compression:
            compression_div_abs_out.append(torch.stack(div_abs_steps, dim=0).mean(dim=0))
            compression_div_pos_out.append(torch.stack(div_pos_steps, dim=0).mean(dim=0))
            compression_div_neg_out.append(torch.stack(div_neg_steps, dim=0).mean(dim=0))
            compression_div_sq_out.append(torch.stack(div_sq_steps, dim=0).mean(dim=0))
        if run_head_div:
            transport_div_abs_out.append(torch.stack(transport_div_steps, dim=0).mean(dim=0))
            residual_div_abs_out.append(torch.stack(residual_div_steps, dim=0).mean(dim=0))
            effective_residual_div_abs_out.append(torch.stack(effective_residual_div_steps, dim=0).mean(dim=0))
        if run_residual_signed_div:
            residual_div_signed_out.append(torch.stack(residual_div_signed_steps, dim=0).mean(dim=0))
        if run_trapz_cnf:
            comp_trapz = trapz_integral_from_steps(div_signed_steps, path_time_steps).detach().cpu()
            comp_trapz_scaled = float(cfg.hcfm_gamma_compression) * comp_trapz
            compression_div_trapz_out.append(comp_trapz)
            cnf_base_plus_compression_trapz_out.append(base_energy_cpu + comp_trapz_scaled)
            cnf_base_minus_compression_trapz_out.append(base_energy_cpu - comp_trapz_scaled)
            if run_trapz_residual:
                residual_trapz = trapz_integral_from_steps(residual_div_signed_steps, path_time_steps).detach().cpu()
                full_trapz_scaled = comp_trapz_scaled + float(cfg.hcfm_gamma_residual) * residual_trapz
                residual_div_trapz_out.append(residual_trapz)
                cnf_base_plus_full_trapz_out.append(base_energy_cpu + full_trapz_scaled)
                cnf_base_minus_full_trapz_out.append(base_energy_cpu - full_trapz_scaled)
    scores = {
        "base_energy": torch.cat(base_out),
        "transport_energy": torch.cat(transport_energy_out),
        "compression_energy": torch.cat(compression_energy_out),
        "residual_energy": torch.cat(residual_energy_out),
        "compression_div_signed": torch.cat(compression_div_signed_out),
        "compression_div_neg_sq": torch.cat(compression_div_neg_sq_out),
    }
    if run_extra_compression:
        scores["compression_div_abs"] = torch.cat(compression_div_abs_out)
        scores["compression_div_pos"] = torch.cat(compression_div_pos_out)
        scores["compression_div_neg"] = torch.cat(compression_div_neg_out)
        scores["compression_div_sq"] = torch.cat(compression_div_sq_out)
    if run_head_div:
        scores["transport_div_abs"] = torch.cat(transport_div_abs_out)
        scores["residual_div_abs"] = torch.cat(residual_div_abs_out)
        scores["effective_residual_div_abs"] = torch.cat(effective_residual_div_abs_out)
    if run_residual_signed_div:
        scores["residual_div_signed"] = torch.cat(residual_div_signed_out)
    if run_trapz_cnf:
        scores["compression_div_trapz"] = torch.cat(compression_div_trapz_out)
        scores["cnf_base_plus_compression_trapz"] = torch.cat(cnf_base_plus_compression_trapz_out)
        scores["cnf_base_minus_compression_trapz"] = torch.cat(cnf_base_minus_compression_trapz_out)
        if run_trapz_residual:
            scores["residual_div_trapz"] = torch.cat(residual_div_trapz_out)
            scores["cnf_base_plus_full_trapz"] = torch.cat(cnf_base_plus_full_trapz_out)
            scores["cnf_base_minus_full_trapz"] = torch.cat(cnf_base_minus_full_trapz_out)
    return scores



# Old separate exact-like likelihood scoring helper removed from active notebook.
# CNF-style trapz scores are computed inside score_data_hcfm without a second reverse-ODE pass.


def _subset_indices(n: int, k: int, seed: int):
    k = min(int(k), int(n))
    if k <= 0:
        return np.array([], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(int(n), size=k, replace=False))


def _component_divergence(model, head_name: str, x: torch.Tensor, t: torch.Tensor, cfg, generator=None, exact: bool = False):
    if head_name == "transport":
        fn = lambda x_in, t_in: model.transport(x_in, t_in)
    elif head_name == "compression":
        fn = lambda x_in, t_in: model.compression(x_in, t_in)
    elif head_name == "residual":
        fn = lambda x_in, t_in: model.residual(x_in, t_in)
    else:
        raise ValueError(head_name)
    if exact:
        return exact_divergence_vector_field(fn, x, t, create_graph=False)
    return hutchinson_divergence_vector_field(
        fn,
        x,
        t,
        n_probe=cfg.eval_n_probe,
        probe_type=cfg.probe_type,
        create_graph=False,
        generator=generator,
    )


def score_hcfm_trapz_cycle_diagnostics(model: DataHCFM, x_all: torch.Tensor, batch_size: int, cfg, desc: str = "HCFM trapz/cycle"):
    """Diagnostic-only CNF-style trapezoid divergence integral and cycle scores."""
    model.eval()
    outs = {
        "cnf_nll_compression_trapz": [],
        "cnf_nll_full_trapz": [],
        "cnf_nll_full_trapz_minus": [],
        "cycle_reconstruction_mse": [],
        "reverse_base_energy": [],
        "forward_cycle_error": [],
    }
    gen_c = make_score_generator(cfg.seed, method_name="Data HCFM audit", score_name=f"{desc}_div_c", device=x_all.device)
    gen_r = make_score_generator(cfg.seed, method_name="Data HCFM audit", score_name=f"{desc}_div_r", device=x_all.device)
    iterator = tqdm(range(0, len(x_all), batch_size), total=int(np.ceil(len(x_all) / batch_size)), desc=desc, leave=False)
    for i in iterator:
        xb = x_all[i:i + batch_size]
        x0_hat, path_states = integrate_reverse_dataspace(model, xb, steps=cfg.ode_steps, method=cfg.ode_method, return_path=True, path_eps=cfg.path_eps, atol=cfg.ode_atol, rtol=cfg.ode_rtol, x_ref=xb)
        x1_recon = integrate_forward_dataspace(model, x0_hat, steps=cfg.ode_steps, method=cfg.ode_method, path_eps=cfg.path_eps, atol=cfg.ode_atol, rtol=cfg.ode_rtol, x_ref=xb)
        base = data_energy(x0_hat).detach().cpu()
        div_c_steps, div_r_steps, times = [], [], []
        for t_value, x_t in path_states:
            t_safe = min(max(float(t_value), cfg.path_eps), 1.0 - cfg.path_eps)
            t = torch.full((x_t.shape[0],), t_safe, device=x_t.device, dtype=x_t.dtype)
            with torch.enable_grad():
                div_c = hutchinson_divergence_vector_field(
                    lambda x_in, t_in: model.compression(x_in, t_in, x_ref=xb),
                    x_t,
                    t,
                    n_probe=cfg.eval_n_probe,
                    probe_type=cfg.probe_type,
                    create_graph=False,
                    generator=gen_c,
                )
                div_r = _component_divergence(model, "residual", x_t, t, cfg, generator=gen_r, exact=False)
            div_c_steps.append(div_c.detach().cpu())
            div_r_steps.append(div_r.detach().cpu())
            times.append(t_safe)
            del div_c, div_r
        order = np.argsort(np.asarray(times, dtype=np.float64))
        t_sorted = torch.tensor(np.asarray(times, dtype=np.float32)[order])
        div_c_sorted = torch.stack([div_c_steps[j] for j in order], dim=0)
        div_r_sorted = torch.stack([div_r_steps[j] for j in order], dim=0)
        dt = (t_sorted[1:] - t_sorted[:-1]).view(-1, 1)
        div_int_c = (0.5 * dt * (div_c_sorted[:-1] + div_c_sorted[1:])).sum(dim=0)
        div_int_r = (0.5 * dt * (div_r_sorted[:-1] + div_r_sorted[1:])).sum(dim=0)
        full_div = float(cfg.hcfm_gamma_compression) * div_int_c + float(cfg.hcfm_gamma_residual) * div_int_r
        comp_div = float(cfg.hcfm_gamma_compression) * div_int_c
        outs["cnf_nll_compression_trapz"].append(base + comp_div)
        outs["cnf_nll_full_trapz"].append(base + full_div)
        outs["cnf_nll_full_trapz_minus"].append(base - full_div)
        cycle = (x1_recon - xb).flatten(1).pow(2).mean(dim=1).detach().cpu()
        outs["cycle_reconstruction_mse"].append(cycle)
        outs["reverse_base_energy"].append(base)
        outs["forward_cycle_error"].append(cycle)
        del x0_hat, x1_recon
    return {k: torch.cat(v) for k, v in outs.items()}


def audit_hcfm_component_scales(model: DataHCFM, calib_x_seq: torch.Tensor, test_x_seq: torch.Tensor, point_labels, cfg, test_labels=None, rare_normal_mask=None):
    """Subset component audit saved as raw rows and grouped summary CSVs."""
    model.eval()
    calib_idx = _subset_indices(len(calib_x_seq), cfg.max_diag_windows_calib, cfg.seed + 2001)
    test_idx = _subset_indices(len(test_x_seq), cfg.max_diag_windows_test, cfg.seed + 3001)
    raw_parts = []
    for split, x_src, idx_np, labels in [
        ("calibration", calib_x_seq, calib_idx, None),
        ("test", test_x_seq, test_idx, test_labels),
    ]:
        if len(idx_np) == 0:
            continue
        scores = score_data_hcfm(
            model,
            x_src[torch.as_tensor(idx_np, device=x_src.device)],
            cfg.hcfm_component_score_batch_size,
            cfg,
            desc=f"audit component scales {split}",
            compute_residual_signed_div=True,
            compute_head_divergence_diagnostics=True,
            compute_extra_compression_variants=True,
        )
        df = pd.DataFrame({"split": split, "window_index": idx_np})
        if labels is None:
            df["group"] = "calibration_normal"
        else:
            label_np = np.asarray(labels)[idx_np]
            df["group"] = np.where(label_np == 1, "test_anomaly", "test_normal")
            if rare_normal_mask is not None:
                rare_np = np.asarray(rare_normal_mask)[idx_np]
                df.loc[(label_np == 0) & rare_np, "group"] = "rare_normal"
        for key in [
            "base_energy", "transport_energy", "compression_energy", "residual_energy",
            "compression_div_signed", "compression_div_abs", "compression_div_pos", "compression_div_neg",
            "residual_div_signed", "residual_div_abs", "transport_div_abs",
        ]:
            if key in scores:
                df[key] = score_to_numpy(scores[key])
        if "compression_div_signed" in df:
            div = df["compression_div_signed"].to_numpy()
            df["compression_div_pos_frac"] = (div > 0).astype(float)
            df["compression_div_neg_frac"] = (div < 0).astype(float)
        if "compression_div_abs" in df:
            df["scaled_compression_div_abs"] = abs(float(cfg.hcfm_gamma_compression)) * df["compression_div_abs"]
        if "residual_div_abs" in df:
            df["scaled_residual_div_abs"] = abs(float(cfg.hcfm_gamma_residual)) * df["residual_div_abs"]
        if "scaled_compression_div_abs" in df and "scaled_residual_div_abs" in df:
            df["scaled_residual_to_compression_div_ratio"] = df["scaled_residual_div_abs"] / np.maximum(df["scaled_compression_div_abs"], 1e-12)
        raw_parts.append(df)
    raw_df = pd.concat(raw_parts, ignore_index=True) if raw_parts else pd.DataFrame()
    if raw_df.empty:
        return raw_df, pd.DataFrame()
    raw_df = add_config_metadata(raw_df, cfg)
    raw_df.to_csv(cfg.output_dir / "hcfm_component_audit_raw_samples.csv", index=False)
    value_cols = [
        c for c in raw_df.columns
        if c not in {"split", "window_index", "group", "dataset_id", "seed", "experiment_name", "score_profile", "plot_profile", "diagnostic_profile"}
        and np.issubdtype(raw_df[c].dtype, np.number)
    ]
    rows = []
    for (split, group), g in raw_df.groupby(["split", "group"]):
        for col in value_cols:
            vals = g[col].dropna().to_numpy(dtype=np.float64)
            if vals.size == 0:
                continue
            rows.append({
                "split": split,
                "group": group,
                "diagnostic": col,
                "n": int(vals.size),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
                "q05": float(np.quantile(vals, 0.05)),
                "q25": float(np.quantile(vals, 0.25)),
                "q75": float(np.quantile(vals, 0.75)),
                "q95": float(np.quantile(vals, 0.95)),
                "max_abs": float(np.max(np.abs(vals))),
            })
    summary_df = add_config_metadata(pd.DataFrame(rows), cfg)
    summary_df.to_csv(cfg.output_dir / "hcfm_component_audit_summary.csv", index=False)
    return raw_df, summary_df


def audit_exact_vs_hutchinson_divergence(model: DataHCFM, x_all: torch.Tensor, cfg, max_windows: int | None = None):
    """Compare exact divergence against Hutchinson estimates on a tiny fixed subset."""
    model.eval()
    max_windows = cfg.exact_jacobian_diag_windows if max_windows is None else max_windows
    idx_np = _subset_indices(len(x_all), min(max_windows, 64), cfg.seed + 4001)
    if len(idx_np) == 0:
        return pd.DataFrame(), pd.DataFrame()
    xb_all = x_all[torch.as_tensor(idx_np, device=x_all.device)]
    t = torch.full((xb_all.shape[0],), 0.5, device=xb_all.device, dtype=xb_all.dtype)
    metric_rows, raw_rows = [], []
    for head in ["compression", "residual", "transport"]:
        with torch.enable_grad():
            exact = _component_divergence(model, head, xb_all, t, cfg, exact=True).detach().cpu().numpy()
        for n_probe in [1, 4, 8, 16]:
            gen = make_score_generator(cfg.seed, method_name="Data HCFM exact-hutch", score_name=f"{head}_{n_probe}", device=x_all.device)
            cfg_tmp = SimpleNamespace(**vars(cfg))
            cfg_tmp.eval_n_probe = n_probe
            with torch.enable_grad():
                hutch = _component_divergence(model, head, xb_all, t, cfg_tmp, generator=gen, exact=False).detach().cpu().numpy()
            pearson = float(np.corrcoef(exact, hutch)[0, 1]) if np.std(exact) > 0 and np.std(hutch) > 0 else np.nan
            spearman = float(pd.Series(exact).corr(pd.Series(hutch), method="spearman"))
            mae = float(np.mean(np.abs(hutch - exact)))
            rel = float(np.mean(np.abs(hutch - exact) / np.maximum(np.abs(exact), 1e-6)))
            sign = float(np.mean(np.sign(exact) == np.sign(hutch)))
            metric_rows.append({
                "head": head,
                "n_probe": n_probe,
                "pearson": pearson,
                "spearman": spearman,
                "mean_abs_error": mae,
                "relative_abs_error": rel,
                "sign_agreement_rate": sign,
            })
            for idx, e, h in zip(idx_np, exact, hutch):
                raw_rows.append({"head": head, "n_probe": n_probe, "window_index": int(idx), "exact_div": float(e), "hutchinson_div": float(h)})
    metrics_df = add_config_metadata(pd.DataFrame(metric_rows), cfg)
    raw_df = add_config_metadata(pd.DataFrame(raw_rows), cfg)
    metrics_df.to_csv(cfg.output_dir / "hcfm_exact_vs_hutchinson_divergence.csv", index=False)
    raw_df.to_csv(cfg.output_dir / "hcfm_exact_vs_hutchinson_divergence_raw.csv", index=False)
    return metrics_df, raw_df


def fit_raw_mahalanobis(x_train: torch.Tensor, ridge: float = 1e-3):
    x = x_train.detach().flatten(1).cpu().numpy().astype(np.float64)
    mu = x.mean(axis=0, keepdims=True)
    cov = np.cov(x, rowvar=False) + ridge * np.eye(x.shape[1], dtype=np.float64)
    inv_cov = np.linalg.pinv(cov)
    return {"mean": mu, "inv_cov": inv_cov, "ridge": ridge}


def score_raw_mahalanobis(model, x_all: torch.Tensor) -> torch.Tensor:
    x = x_all.detach().flatten(1).cpu().numpy().astype(np.float64)
    diff = x - model["mean"]
    score = np.einsum("bi,ij,bj->b", diff, model["inv_cov"], diff)
    return torch.tensor(score, dtype=torch.float32)


def standardize_dataspace_score(calib_score, test_score, mode=None, floor=None):
    calib_np = score_to_numpy(calib_score)
    test_np = score_to_numpy(test_score)
    mode = mode or getattr(dataspace_cfg, "score_calibration_mode", "zscore")
    floor = float(floor if floor is not None else getattr(dataspace_cfg, "score_calibration_floor", 1e-3))

    if mode == "zscore":
        mu = float(np.mean(calib_np))
        sd_raw = float(np.std(calib_np))
        sd = max(sd_raw, floor)
        calib_z = (calib_np - mu) / sd
        test_z = (test_np - mu) / sd
        q25, q75 = np.quantile(calib_np, [0.25, 0.75])
        stats = {
            "calibration_mode": "zscore",
            "calib_center": mu,
            "calib_scale": sd,
            "calib_mean": float(np.mean(calib_np)),
            "calib_std": sd_raw,
            "calib_median": float(np.median(calib_np)),
            "calib_iqr": float(q75 - q25),
            "calib_min": float(np.min(calib_np)),
            "calib_max": float(np.max(calib_np)),
        }
    elif mode == "robust":
        med = float(np.median(calib_np))
        q25, q75 = np.quantile(calib_np, [0.25, 0.75])
        iqr_raw = float(q75 - q25)
        iqr = max(iqr_raw, floor)
        calib_z = (calib_np - med) / iqr
        test_z = (test_np - med) / iqr
        stats = {
            "calibration_mode": "robust",
            "calib_center": med,
            "calib_scale": iqr,
            "calib_mean": float(np.mean(calib_np)),
            "calib_std": float(np.std(calib_np)),
            "calib_median": med,
            "calib_iqr": iqr_raw,
            "calib_min": float(np.min(calib_np)),
            "calib_max": float(np.max(calib_np)),
        }
    else:
        raise ValueError(f"Unknown score calibration mode: {mode}")
    return calib_z, test_z, stats


_VUS_GET_METRICS = None
_VUS_IMPORT_ERROR = None
_VUS_IMPORT_WARNED = False


def try_import_vus_metrics():
    """Return VUS get_metrics if available; otherwise warn once and return None."""
    global _VUS_GET_METRICS, _VUS_IMPORT_ERROR, _VUS_IMPORT_WARNED
    if _VUS_GET_METRICS is not None:
        return _VUS_GET_METRICS
    import_attempts = [
        ("vus.metrics", "get_metrics"),
        ("metrics.metrics", "get_metrics"),
        ("TSB_UAD.vus.metrics", "get_metrics"),
    ]
    last_error = None
    for module_name, attr_name in import_attempts:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            _VUS_GET_METRICS = getattr(module, attr_name)
            _VUS_IMPORT_ERROR = None
            return _VUS_GET_METRICS
        except Exception as exc:
            last_error = exc
    _VUS_IMPORT_ERROR = last_error
    if not _VUS_IMPORT_WARNED:
        print(
            "VUS metrics package not found. Install the official VUS/TSB-AD metrics package "
            "or place it on PYTHONPATH. Skipping VUS-PR."
        )
        if last_error is not None:
            print("Last VUS import error:", repr(last_error))
        _VUS_IMPORT_WARNED = True
    return None


def infer_vus_sliding_window(point_labels, default_window):
    labels = np.asarray(point_labels).astype(bool)
    lengths = [int(e - s) for s, e in contiguous_true_ranges(labels)]
    if lengths:
        return max(1, int(np.median(lengths)))
    return max(1, int(default_window))


def get_vus_sliding_window(cfg, point_labels):
    if getattr(cfg, "vus_sliding_window", None) is not None:
        return int(cfg.vus_sliding_window)
    return infer_vus_sliding_window(point_labels, cfg.window)


def _extract_vus_metric(metrics_obj, keys):
    if metrics_obj is None:
        return np.nan
    if isinstance(metrics_obj, dict):
        for key in keys:
            if key in metrics_obj:
                return float(metrics_obj[key])
        lowered = {str(k).lower().replace("-", "_"): v for k, v in metrics_obj.items()}
        for key in keys:
            norm = key.lower().replace("-", "_")
            if norm in lowered:
                return float(lowered[norm])
    if hasattr(metrics_obj, "to_dict"):
        return _extract_vus_metric(metrics_obj.to_dict(), keys)
    return np.nan


def compute_vus_for_point_score(point_labels, point_score, sliding_window):
    labels = np.asarray(point_labels).astype(int).reshape(-1)
    score = np.asarray(point_score, dtype=float).reshape(-1)
    n = min(len(labels), len(score))
    labels = labels[:n]
    score = score[:n]
    finite = np.isfinite(score)
    fill = float(np.nanmedian(score[finite])) if finite.any() else 0.0
    score = np.nan_to_num(score, nan=fill, posinf=fill, neginf=fill)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"vus_pr": np.nan, "vus_roc": np.nan}
    get_metrics = try_import_vus_metrics()
    if get_metrics is None:
        return {"vus_pr": np.nan, "vus_roc": np.nan}
    call_attempts = [
        lambda: get_metrics(score, labels, slidingWindow=sliding_window),
        lambda: get_metrics(score, labels, sliding_window=sliding_window),
        lambda: get_metrics(labels, score, slidingWindow=sliding_window),
        lambda: get_metrics(labels, score, sliding_window=sliding_window),
    ]
    last_error = None
    metrics_obj = None
    for call in call_attempts:
        try:
            metrics_obj = call()
            break
        except Exception as exc:
            last_error = exc
    if metrics_obj is None:
        global _VUS_IMPORT_ERROR
        _VUS_IMPORT_ERROR = last_error
        return {"vus_pr": np.nan, "vus_roc": np.nan}
    return {
        "vus_pr": _extract_vus_metric(metrics_obj, ["VUS_PR", "VUS-PR", "vus_pr", "VUS_PR_score"]),
        "vus_roc": _extract_vus_metric(metrics_obj, ["VUS_ROC", "VUS-ROC", "vus_roc", "VUS_ROC_score"]),
    }


def add_vus_fields_to_row(row, test_score_z, test_starts, point_labels, window, cfg):
    if not getattr(cfg, "compute_vus_metrics", False):
        return row
    sliding_window = get_vus_sliding_window(cfg, point_labels)
    row["vus_sliding_window"] = int(sliding_window)
    if getattr(cfg, "vus_use_point_scores", True):
        for mode in getattr(cfg, "vus_point_projection_modes", ("mean", "max")):
            point_score = window_scores_to_points(
                test_score_z,
                test_starts,
                len(point_labels),
                window,
                mode=mode,
            )
            metrics = compute_vus_for_point_score(point_labels, point_score, sliding_window)
            row[f"vus_pr_point_{mode}"] = metrics["vus_pr"]
            row[f"vus_roc_point_{mode}"] = metrics["vus_roc"]
    return row


def compute_dataspace_metric_row(job):
    row = metric_row(
        job["model"],
        job["score_name"],
        job["y"],
        job["test_z"],
        job["test_starts"],
        job["point_labels"],
        job["window"],
        job["rare_normal_mask"],
    )
    add_hundman_fields_to_row(
        row,
        job["y"],
        job["test_z"],
        job["calib_z"],
        test_starts=job["test_starts"],
        point_labels=job["point_labels"],
        window=job["window"],
    )
    add_vus_fields_to_row(row, job["test_z"], job["test_starts"], job["point_labels"], job["window"], dataspace_cfg)
    return row


def enqueue_dataspace_metric(metrics_rows, model, score_name, calib_z, test_z, y, test_starts, point_labels, window, rare_normal_mask):
    job = {
        "model": model,
        "score_name": score_name,
        "calib_z": np.asarray(calib_z),
        "test_z": np.asarray(test_z),
        "y": np.asarray(y),
        "test_starts": np.asarray(test_starts),
        "point_labels": np.asarray(point_labels),
        "window": int(window),
        "rare_normal_mask": np.asarray(rare_normal_mask),
    }
    if "metric_jobs" in globals() and isinstance(metric_jobs, list):
        metric_jobs.append(job)
    else:
        metrics_rows.append(compute_dataspace_metric_row(job))


def finalize_parallel_metric_rows(metrics_rows, metric_jobs, cfg):
    if not metric_jobs:
        return metrics_rows
    if getattr(cfg, "compute_vus_metrics", False):
        try_import_vus_metrics()
    n_jobs = max(1, int(getattr(cfg, "metric_n_jobs", 1)))
    print(f"Computing metric rows: {len(metric_jobs)} scores with metric_n_jobs={n_jobs}")
    t0 = time.time()
    if n_jobs == 1:
        new_rows = [compute_dataspace_metric_row(job) for job in tqdm(metric_jobs, desc="metrics", leave=False)]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            new_rows = list(tqdm(executor.map(compute_dataspace_metric_row, metric_jobs), total=len(metric_jobs), desc="metrics", leave=False))
    elapsed = time.time() - t0
    print(f"Metric rows computed in {elapsed:.1f}s")
    metrics_rows.extend(new_rows)
    metric_jobs.clear()
    return metrics_rows


def add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, model, score_name, calib_score, test_score, y, test_starts, point_labels, window, rare_normal_mask):
    calib_z, test_z, stats = standardize_dataspace_score(calib_score, test_score)
    key = f"{model} / {score_name}"
    scores_z[key] = test_z
    scores_raw[key] = {"calib_raw": score_to_numpy(calib_score), "test_raw": score_to_numpy(test_score), "calib_z": calib_z, "test_z": test_z}
    score_stats.append({"score_key": key, **stats})
    enqueue_dataspace_metric(metrics_rows, model, score_name, calib_z, test_z, y, test_starts, point_labels, window, rare_normal_mask)
    return calib_z, test_z


# %% notebook cell 21
# AE / reconstruction_mse_x is a cheap data-space reconstruction baseline only.
# It is not used to create latents and no latent flow is trained.
dataspace_ae = None
dataspace_ae_hist = pd.DataFrame()
ae_rec_calib = None
ae_rec_test = None

if dataspace_cfg.run_ae_baseline:
    if dataspace_cfg.use_same_seed_per_method:
        seed_everything(dataspace_cfg.seed, deterministic=dataspace_cfg.deterministic)
    dataspace_ae, dataspace_ae_hist = train_dataspace_recon_ae(train_x_seq, dataspace_cfg)
    add_config_metadata(dataspace_ae_hist, dataspace_cfg).to_csv(dataspace_cfg.output_dir / "dataspace_ae_training_history.csv", index=False)

    ae_rec_calib = score_reconstruction_mse_x(dataspace_ae, calib_x_seq, dataspace_cfg.ae_score_batch_size)
    ae_rec_test = score_reconstruction_mse_x(dataspace_ae, test_x_seq, dataspace_cfg.ae_score_batch_size)
else:
    print("Skipping AE reconstruction baseline (dataspace_cfg.run_ae_baseline=False).")

raw_gaussian = None
mahal_calib = mahal_test = None
if dataspace_cfg.run_raw_mahalanobis:
    raw_gaussian = fit_raw_mahalanobis(train_x_seq, ridge=1e-3)
    mahal_calib = score_raw_mahalanobis(raw_gaussian, calib_x_seq)
    mahal_test = score_raw_mahalanobis(raw_gaussian, test_x_seq)
else:
    print("Skipping Raw Gaussian / Mahalanobis baseline (dataspace_cfg.run_raw_mahalanobis=False).")


# %% notebook cell 23
# HCFM-only multivariate notebook: Vanilla/FDM training is omitted from the active path.
vanilla_model = None
fdm_model = None
hcfm_model = None
vanilla_history = pd.DataFrame()
fdm_history = pd.DataFrame()
hcfm_history = None


# %% notebook cell 25
# Data FDM-lite training omitted from the active HCFM-only multivariate path.


# %% notebook cell 29
if dataspace_cfg.run_scalar_hcfm and dataspace_cfg.hcfm_compression_type == "scalar_potential":
    sanity_model = DataHCFM(
        channels=dataspace_cfg.channels,
        length=dataspace_cfg.length,
        hidden=dataspace_cfg.hidden,
        depth=dataspace_cfg.base_depth,
        t_dim=dataspace_cfg.time_emb_dim,
        transport_rank=dataspace_cfg.hcfm_transport_rank,
        residual_depth=dataspace_cfg.hcfm_residual_depth,
        residual_arch=dataspace_cfg.hcfm_residual_arch,
        residual_hidden=dataspace_cfg.hcfm_residual_hidden,
        residual_kernel_size=dataspace_cfg.hcfm_residual_kernel_size,
        gamma_compression=dataspace_cfg.hcfm_gamma_compression,
        gamma_residual=dataspace_cfg.hcfm_gamma_residual,
        max_period=dataspace_cfg.time_emb_max_period,
        time_scale=dataspace_cfg.hcfm_time_emb_scale,
        time_use_2pi=dataspace_cfg.hcfm_time_emb_use_2pi,
        use_film_time_conditioning=dataspace_cfg.hcfm_use_film_time_conditioning,
        compression_type=dataspace_cfg.hcfm_compression_type,
        use_residual_mismatch_gate=False,
        residual_gate_min=dataspace_cfg.hcfm_residual_gate_min,
        inference_residual_scale_mode="full",
        potential_norm_type=dataspace_cfg.hcfm_potential_norm_type,
        potential_residual_scale=dataspace_cfg.hcfm_potential_residual_scale,
        potential_backbone=dataspace_cfg.hcfm_potential_backbone,
        mixer_depth=dataspace_cfg.hcfm_mixer_depth,
        mixer_token_mlp_dim=dataspace_cfg.hcfm_mixer_token_mlp_dim,
        mixer_channel_mlp_dim=dataspace_cfg.hcfm_mixer_channel_mlp_dim,
        mixer_norm_type=dataspace_cfg.hcfm_mixer_norm_type,
        mixer_residual_scale=dataspace_cfg.hcfm_mixer_residual_scale,
        potential_scale=dataspace_cfg.hcfm_potential_scale,
        potential_out_init=dataspace_cfg.hcfm_potential_out_init,
        potential_out_init_std=dataspace_cfg.hcfm_potential_out_init_std,
        use_position_embedding=dataspace_cfg.hcfm_use_position_embedding,
        position_emb_dim=dataspace_cfg.hcfm_position_emb_dim,
        position_proj_dim=dataspace_cfg.hcfm_position_proj_dim,
        position_max_period=dataspace_cfg.hcfm_position_max_period,
        position_use_integer_positions=dataspace_cfg.hcfm_position_use_integer_positions,
        use_fft_features=dataspace_cfg.hcfm_use_fft_features,
        fft_feature_mode=dataspace_cfg.hcfm_fft_feature_mode,
        fft_num_bands=dataspace_cfg.hcfm_fft_num_bands,
        fft_include_centroid=dataspace_cfg.hcfm_fft_include_centroid,
        fft_include_entropy=dataspace_cfg.hcfm_fft_include_entropy,
        fft_include_dominant=dataspace_cfg.hcfm_fft_include_dominant,
        fft_log_magnitude=dataspace_cfg.hcfm_fft_log_magnitude,
        fft_eps=dataspace_cfg.hcfm_fft_eps,
        fft_detach_features=dataspace_cfg.hcfm_fft_detach_features,
        normalize_fft_features=dataspace_cfg.hcfm_normalize_fft_features,
        scale_aux_channels_by_x_stats=dataspace_cfg.hcfm_scale_aux_channels_by_x_stats,
        context_focus_scale=dataspace_cfg.hcfm_context_focus_scale,
    ).to(device)

    xb = train_x_seq[:8].detach().clone().requires_grad_(True)
    tb = torch.rand(8, device=device)
    x0_sanity = torch.randn_like(xb)
    t_view = tb.view((tb.shape[0],) + (1,) * (xb.ndim - 1))
    xt_sanity = (1.0 - t_view) * x0_sanity + t_view * xb
    xt_sanity = xt_sanity.detach().clone().requires_grad_(True)
    vt, vc, vr = sanity_model.components(xt_sanity, tb, x_ref=xb)
    print("vt shape:", tuple(vt.shape))
    print("vc shape:", tuple(vc.shape))
    print("vr shape:", tuple(vr.shape))
    print("vc requires_grad:", vc.requires_grad)
    print("potential_norm_type:", dataspace_cfg.hcfm_potential_norm_type)
    print("potential_residual_scale:", dataspace_cfg.hcfm_potential_residual_scale)
    print("potential_scale:", dataspace_cfg.hcfm_potential_scale)
    print("potential_out_init:", dataspace_cfg.hcfm_potential_out_init)
    print("position_embedding:", dataspace_cfg.hcfm_use_position_embedding, "emb_dim=", dataspace_cfg.hcfm_position_emb_dim, "proj_dim=", dataspace_cfg.hcfm_position_proj_dim)
    print("fft_features:", dataspace_cfg.hcfm_use_fft_features, "mode=", dataspace_cfg.hcfm_fft_feature_mode, "bands=", dataspace_cfg.hcfm_fft_num_bands, "feature_dim=", getattr(sanity_model.compression, "fft_feature_dim", 0), "detach=", dataspace_cfg.hcfm_fft_detach_features, "normalize=", dataspace_cfg.hcfm_normalize_fft_features)
    if dataspace_cfg.hcfm_use_fft_features:
        raw_fft_feat = sanity_model.compression.fft_features(xb)
        norm_fft_feat = sanity_model.compression.normalize_fft_features(raw_fft_feat)
        _, x_ref_std = sanity_model.compression.get_x_ref_stats(xt_sanity, x_ref=xb)
        fft_channels_scaled = sanity_model.compression.scale_aux_channel(norm_fft_feat[:, :, None].expand(-1, -1, xt_sanity.shape[-1]), x_ref_std)
        print("x_ref mean/std/min/max=", float(xb.mean().item()), float(xb.std().item()), float(xb.min().item()), float(xb.max().item()))
        print("x_t mean/std/min/max=", float(xt_sanity.mean().item()), float(xt_sanity.std().item()), float(xt_sanity.min().item()), float(xt_sanity.max().item()))
        if dataspace_cfg.hcfm_use_position_embedding and sanity_model.compression.position_embedding is not None:
            pos_channels = sanity_model.compression.position_embedding(batch_size=xb.shape[0], device=xb.device, dtype=xb.dtype)
            pos_scaled = sanity_model.compression.scale_aux_channel(pos_channels, x_ref_std)
            print("position_raw mean/std/min/max=", float(pos_channels.mean().item()), float(pos_channels.std().item()), float(pos_channels.min().item()), float(pos_channels.max().item()))
            print("position_scaled mean/std/min/max=", float(pos_scaled.mean().item()), float(pos_scaled.std().item()), float(pos_scaled.min().item()), float(pos_scaled.max().item()))
        print("raw_fft_feat shape:", tuple(raw_fft_feat.shape), "mean/std/min/max=", float(raw_fft_feat.mean().item()), float(raw_fft_feat.std().item()), float(raw_fft_feat.min().item()), float(raw_fft_feat.max().item()), "requires_grad=", raw_fft_feat.requires_grad)
        print("norm_fft_feat shape:", tuple(norm_fft_feat.shape), "mean/std/min/max=", float(norm_fft_feat.mean().item()), float(norm_fft_feat.std().item()), float(norm_fft_feat.min().item()), float(norm_fft_feat.max().item()), "requires_grad=", norm_fft_feat.requires_grad)
        print("fft_scaled_channels mean/std/min/max=", float(fft_channels_scaled.mean().item()), float(fft_channels_scaled.std().item()), float(fft_channels_scaled.min().item()), float(fft_channels_scaled.max().item()))
        assert torch.isfinite(norm_fft_feat).all()
        if not dataspace_cfg.hcfm_fft_detach_features:
            assert raw_fft_feat.requires_grad
            assert norm_fft_feat.requires_grad
        potential_input_sanity = sanity_model.compression.potential_input(xt_sanity, x_ref=xb)
        print("potential_input mean/std/min/max=", float(potential_input_sanity.mean().item()), float(potential_input_sanity.std().item()), float(potential_input_sanity.min().item()), float(potential_input_sanity.max().item()), "finite=", bool(torch.isfinite(potential_input_sanity).all().item()))
        phi = sanity_model.compression.potential(xt_sanity, tb, x_ref=xb)
        grad_x = torch.autograd.grad(phi.sum(), xt_sanity, retain_graph=True, create_graph=True)[0]
        print("fft potential grad_x norm:", float(grad_x.flatten(1).norm(dim=1).mean().item()), "finite=", bool(torch.isfinite(grad_x).all().item()))
        xb_perturbed = xb + 1e-3 * torch.randn_like(xb)
        fft_1 = sanity_model.compression.fft_features(xb)
        fft_2 = sanity_model.compression.fft_features(xb_perturbed)
        fft_delta = (fft_1 - fft_2).norm(dim=1).mean()
        print("fft_delta under 1e-3 perturbation:", float(fft_delta.item()))
    vc_det = vc.detach()
    scaled_vc_det = (dataspace_cfg.hcfm_gamma_compression * vc).detach()
    print(f"vc mean/std/max: {vc_det.mean().item():.6g} / {vc_det.std().item():.6g} / {vc_det.abs().max().item():.6g}")
    print(f"gamma_C*vc mean/std/max: {scaled_vc_det.mean().item():.6g} / {scaled_vc_det.std().item():.6g} / {scaled_vc_det.abs().max().item():.6g}")

    loss = (vt + dataspace_cfg.hcfm_gamma_compression * vc + dataspace_cfg.hcfm_gamma_residual * vr).pow(2).mean()
    loss.backward()
    compression_grad_norm = 0.0
    for p in sanity_model.compression.parameters():
        if p.grad is not None:
            compression_grad_norm += float(p.grad.detach().norm().cpu())
    print("compression_grad_norm:", compression_grad_norm)
    assert vc.shape == xb.shape
    assert vc.requires_grad
    assert compression_grad_norm > 0.0

    sanity_model.zero_grad(set_to_none=True)
    x_ref_div = train_x_seq[:8].detach().clone().requires_grad_(True)
    x0_div = torch.randn_like(x_ref_div)
    tb = torch.rand(8, device=device)
    t_view = tb.view((tb.shape[0],) + (1,) * (x_ref_div.ndim - 1))
    xt_div = ((1.0 - t_view) * x0_div + t_view * x_ref_div).detach().clone().requires_grad_(True)
    div_c = hutchinson_divergence_vector_field(
        lambda x_in, t_in: sanity_model.compression(x_in, t_in, x_ref=x_ref_div),
        xt_div,
        tb,
        n_probe=1,
        probe_type=dataspace_cfg.probe_type,
        create_graph=True,
    )
    print("div_c finite:", torch.isfinite(div_c).all().item())
    div_c_det = div_c.detach()
    print(f"div_c mean/std/max: {div_c_det.mean().item():.6g} / {div_c_det.std().item():.6g} / {div_c_det.abs().max().item():.6g}")
    assert torch.isfinite(div_c).all()
    del sanity_model, xb, tb, vt, vc, vr, div_c, x_ref_div, x0_div, xt_div
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
else:
    print("Skipping scalar-potential compression sanity check for this config.")


# %% notebook cell 30
if dataspace_cfg.run_scalar_hcfm:   
    if dataspace_cfg.use_same_seed_per_method:  
        seed_everything(dataspace_cfg.seed, deterministic=dataspace_cfg.deterministic)
    hcfm_model, hcfm_history = train_data_hcfm(train_x_seq, dataspace_cfg)
    add_config_metadata(hcfm_history, dataspace_cfg).to_csv(dataspace_cfg.output_dir / "data_hcfm_training_history.csv", index=False)


# %% notebook cell 31
# HCFM time/position conditioning sanity diagnostics.
if globals().get("hcfm_model") is not None:
    hcfm_model.eval()
    diag_n = min(32, len(calib_x_seq))
    xb = calib_x_seq[:diag_n]
    times = [0.1, 0.5, 0.9]
    with torch.enable_grad():   
        comps = {}
        for tv in times:
            tb = torch.full((xb.shape[0],), tv, device=xb.device, dtype=xb.dtype)
            xt_time = xb.detach().clone().requires_grad_(True)
            vt, vc, vr = hcfm_model.components(xt_time, tb, x_ref=xb)
            total = hcfm_model.compose(vt, vc, vr)
            comps[tv] = {"transport": vt.detach(), "compression": vc.detach(), "residual": vr.detach(), "total": total.detach()}
    time_rows = []
    for a, b in [(0.1, 0.5), (0.5, 0.9)]:
        for name in ["transport", "compression", "residual", "total"]:
            diff = (comps[a][name] - comps[b][name]).flatten(1).norm(dim=1).mean().item()
            time_rows.append({"component": name, "time_pair": f"{a}-{b}", "mean_l2_delta": diff})
    hcfm_time_sensitivity_df = add_config_metadata(pd.DataFrame(time_rows), dataspace_cfg)
    hcfm_time_sensitivity_df.to_csv(dataspace_cfg.output_dir / "hcfm_time_sensitivity_diagnostics.csv", index=False)
    print("HCFM time sensitivity diagnostics")
    display(hcfm_time_sensitivity_df)
    del xb, comps

    if dataspace_cfg.hcfm_use_position_embedding and hasattr(hcfm_model.compression, "position_embedding") and hcfm_model.compression.position_embedding is not None:
        pos = hcfm_model.compression.position_embedding(batch_size=1, device=calib_x_seq.device, dtype=calib_x_seq.dtype)[0].transpose(0, 1).detach()
        adj = (pos[1:] - pos[:-1]).norm(dim=1)
        pos_rows = [{
            "mean_adjacent_distance": float(adj.mean().item()),
            "min_adjacent_distance": float(adj.min().item()),
            "max_adjacent_distance": float(adj.max().item()),
            "first_last_distance": float((pos[0] - pos[-1]).norm().item()),
            "use_integer_positions": bool(dataspace_cfg.hcfm_position_use_integer_positions),
            "position_emb_dim": int(dataspace_cfg.hcfm_position_emb_dim),
            "position_proj_dim": int(dataspace_cfg.hcfm_position_proj_dim),
        }]
        hcfm_position_embedding_df = add_config_metadata(pd.DataFrame(pos_rows), dataspace_cfg)
        hcfm_position_embedding_df.to_csv(dataspace_cfg.output_dir / "hcfm_position_embedding_diagnostics.csv", index=False)
        print("HCFM position embedding diagnostics")
        display(hcfm_position_embedding_df)
else:
    print("Skipping HCFM time/position conditioning diagnostics because hcfm_model is None.")


# %% notebook cell 32
def count_trainable_params(model):
    """Count trainable parameters in a torch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


param_rows = []
if globals().get("vanilla_model") is not None:
    param_rows.append({"model": "Vanilla Data FM", "trainable_params": count_trainable_params(vanilla_model)})
if globals().get("fdm_model") is not None:
    param_rows.append({"model": "Data FDM-lite", "trainable_params": count_trainable_params(fdm_model)})
if globals().get("hcfm_model") is not None:
    param_rows.append({"model": "Data HCFM", "trainable_params": count_trainable_params(hcfm_model)})
param_df = add_config_metadata(pd.DataFrame(param_rows), dataspace_cfg)
print("Trainable parameter counts")
if param_df.empty:
    print("No trainable models were created by the active config.")
else:
    display(param_df[["model", "trainable_params"]])
param_df.to_csv(dataspace_cfg.output_dir / "parameter_counts.csv", index=False)

training_timing_by_method = {}
for method_name, hist in [
    ("AE", globals().get("dataspace_ae_hist")),
    ("Vanilla Data FM", globals().get("vanilla_history")),
    ("Data FDM-lite", globals().get("fdm_history")),
    ("Data HCFM", globals().get("hcfm_history")),
]:
    if hist is not None and "train_time_sec" in hist.columns and len(hist) > 0:
        training_timing_by_method[method_name] = {
            "train_time_sec": float(hist["train_time_sec"].iloc[-1]),
            "steps_per_sec": float(hist["steps_per_sec"].iloc[-1]),
        }


history_specs = []
for title, name, cols in [
    ("Data-space AE", "dataspace_ae_hist", ["loss"]),
    ("Vanilla Data FM", "vanilla_history", ["loss", "target_mse", "rel"]),
    ("Data FDM-lite", "fdm_history", ["loss", "fm", "div", "rel"]),
    ("Data HCFM", "hcfm_history", ["loss", "fm", "div", "ortho", "res"]),
]:
    hist = globals().get(name)
    if isinstance(hist, pd.DataFrame) and "step" in hist.columns:
        history_specs.append((title, hist, cols))
if history_specs:
    fig, axes = plt.subplots(len(history_specs), 1, figsize=(11, 2.8 * len(history_specs)), sharex=False)
    if len(history_specs) == 1:
        axes = [axes]
    for ax, (title, hist, cols) in zip(axes, history_specs):
        for col in cols:
            if col in hist.columns:
                ax.plot(hist["step"], hist[col], marker="o", ms=2.5, label=col)
        ax.set_title(title)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(dataspace_cfg.output_dir / "fig_dataspace_training_losses.png", dpi=160)
    plt.close(fig)
else:
    print("No checkpointed training metrics available for training-loss plot.")


# %% notebook cell 35
def require_trained_model(model, name: str):
    """Return model or raise a clear out-of-order/disabled-run error."""
    if model is None:
        raise RuntimeError(
            f"{name} is None. Run its training cell first, or enable the corresponding "
            "dataspace_cfg.run_* flag before running the scoring cells."
        )
    return model


for model in [vanilla_model, fdm_model, hcfm_model, dataspace_ae]:
    if model is not None:
        model.eval()
clear_cuda_cache() if "clear_cuda_cache" in globals() else None
print("data shape:", dataspace_cfg.data_shape)
print("ode:", {"steps": dataspace_cfg.ode_steps, "method": dataspace_cfg.ode_method})
print("HCFM component score batch:", dataspace_cfg.hcfm_component_score_batch_size)
print("cuda allocated GB:", round(torch.cuda.memory_allocated() / 1e9, 3) if torch.cuda.is_available() else "cpu")


# %% notebook cell 37
if "clear_cuda_cache" not in globals():
    def clear_cuda_cache() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

clear_cuda_cache()


# %% notebook cell 38
# Vanilla/FDM scoring omitted from the active HCFM-only multivariate path.
van_base_calib = van_base_test = None
van_consistency_calib = van_consistency_test = None
fdm_base_calib = fdm_base_test = None
fdm_consistency_calib = fdm_consistency_test = None


# %% notebook cell 39
score_t0 = time.time()
hcfm_calib_scores = hcfm_test_scores = None
hcfm_consistency_calib = hcfm_consistency_test = None
hcfm_diag_calib_scores = hcfm_diag_test_scores = None
hcfm_exact_calib_scores = hcfm_exact_test_scores = None

if hcfm_model is not None:
    hcfm_model_scoring = require_trained_model(hcfm_model, "Data HCFM")
    need_residual_signed_for_scores = bool(
        dataspace_cfg.compute_residual_signed_div
        or dataspace_cfg.compute_full_likelihood_proxy
        or getattr(dataspace_cfg, "run_scalar_potential_audit", False)
        or (dataspace_cfg.compute_trapz_cnf_integral_scores and dataspace_cfg.trapz_cnf_include_residual)
    )
    hcfm_score_kwargs = dict(
        compute_residual_signed_div=need_residual_signed_for_scores,
        compute_head_divergence_diagnostics=dataspace_cfg.compute_head_divergence_diagnostics,
        compute_extra_compression_variants=dataspace_cfg.score_profile in {"extended", "debug"} or getattr(dataspace_cfg, "run_scalar_potential_audit", False),
    )
    hcfm_calib_scores = score_data_hcfm(hcfm_model_scoring, calib_x_seq, dataspace_cfg.hcfm_component_score_batch_size, dataspace_cfg, "Data HCFM calib", **hcfm_score_kwargs)
    hcfm_test_scores = score_data_hcfm(hcfm_model_scoring, test_x_seq, dataspace_cfg.hcfm_component_score_batch_size, dataspace_cfg, "Data HCFM test", **hcfm_score_kwargs)

    if dataspace_cfg.compute_exact_likelihood_scores:
        print("Mean-divergence likelihood-style rows will be derived from existing base_energy and path-mean component-divergence scores; no extra scoring pass.")
    else:
        print("Skipping exact HCFM likelihood-style scoring.")

    if dataspace_cfg.compute_fm_consistency:
        hcfm_consistency_calib = score_hcfm_fm_consistency_data(
            hcfm_model_scoring,
            calib_x_seq,
            batch_size=dataspace_cfg.hcfm_component_score_batch_size,
            cfg=dataspace_cfg,
            desc="Data HCFM consistency calib",
        )
        hcfm_consistency_test = score_hcfm_fm_consistency_data(
            hcfm_model_scoring,
            test_x_seq,
            batch_size=dataspace_cfg.hcfm_component_score_batch_size,
            cfg=dataspace_cfg,
            desc="Data HCFM consistency test",
        )
    else:
        print("Skipping HCFM FM-consistency scoring in core profile.")
    if getattr(dataspace_cfg, "run_scalar_potential_audit", False):
        hcfm_diag_calib_scores = score_hcfm_trapz_cycle_diagnostics(
            hcfm_model_scoring,
            calib_x_seq,
            dataspace_cfg.hcfm_component_score_batch_size,
            dataspace_cfg,
            desc="Data HCFM trapz/cycle calib",
        )
        hcfm_diag_test_scores = score_hcfm_trapz_cycle_diagnostics(
            hcfm_model_scoring,
            test_x_seq,
            dataspace_cfg.hcfm_component_score_batch_size,
            dataspace_cfg,
            desc="Data HCFM trapz/cycle test",
        )
    print(f"Data HCFM component scoring elapsed: {time.time() - score_t0:.1f}s")

    if dataspace_cfg.repro_check:
        repro_x = calib_x_seq[: min(64, len(calib_x_seq))]
        s1 = score_data_hcfm(hcfm_model_scoring, repro_x, dataspace_cfg.hcfm_component_score_batch_size, dataspace_cfg, "Data HCFM repro check 1", **hcfm_score_kwargs)
        s2 = score_data_hcfm(hcfm_model_scoring, repro_x, dataspace_cfg.hcfm_component_score_batch_size, dataspace_cfg, "Data HCFM repro check 2", **hcfm_score_kwargs)
        repro_key = "compression_div_neg_sq"
        max_diff = np.max(np.abs(score_to_numpy(s1[repro_key]) - score_to_numpy(s2[repro_key])))
        print(f"Repro check max diff {repro_key}: {max_diff:.6e}")
else:
    print("Skipping Data HCFM scoring because dataspace_cfg.run_scalar_hcfm=False or model is not trained.")


# %% notebook cell 41

# ============================================================
# Hundman-style window-overlap event metrics
# ============================================================

def threshold_scores_for_hundman(scores, calib_scores=None, q=0.99):
    scores_np = np.asarray(scores, dtype=np.float64)
    ref = scores_np if calib_scores is None else np.asarray(calib_scores, dtype=np.float64)
    threshold = float(np.nanquantile(ref, q))
    pred = scores_np >= threshold
    return pred.astype(bool), threshold


def threshold_scores_for_hundman_sd(scores, calib_scores=None, k=2.0):
    scores_np = np.asarray(scores, dtype=np.float64)
    ref = scores_np if calib_scores is None else np.asarray(calib_scores, dtype=np.float64)
    mu = float(np.nanmean(ref))
    sd = max(float(np.nanstd(ref)), 1e-6)
    threshold = mu + float(k) * sd
    pred = scores_np >= threshold
    return pred.astype(bool), float(threshold), mu, sd


def contiguous_window_ranges(binary_labels):
    labels = np.asarray(binary_labels).astype(bool)
    ranges = []
    start = None
    for idx, value in enumerate(labels):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            ranges.append((start, idx - 1))
            start = None
    if start is not None:
        ranges.append((start, len(labels) - 1))
    return ranges


def intervals_overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def hundman_window_overlap_metrics(y_true_window, y_pred_window):
    """
    Contextual overlap-segment metric matching contextual_confusion_matrix(..., weighted=False).

    TP is counted per ground-truth event that has at least one overlapping
    predicted event. FP is the number of predicted events that overlap no true
    event. FN is the number of true events missed by all predicted events.
    Multiple predicted events overlapping the same true event are not counted as
    multiple true positives and are removed from the FP pool, matching the
    unweighted overlap implementation in contextual.py.
    """
    true_events = contiguous_window_ranges(y_true_window)
    pred_events = contiguous_window_ranges(y_pred_window)
    unmatched_pred = list(pred_events)
    tp = 0
    fn = 0
    matched_pred_count = 0
    for true in true_events:
        found = False
        for pred in pred_events:
            if intervals_overlap(true, pred):
                if not found:
                    tp += 1
                    found = True
                if pred in unmatched_pred:
                    unmatched_pred.remove(pred)
                    matched_pred_count += 1
        if not found:
            fn += 1
    fp = len(unmatched_pred)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "hundman_tp": int(tp),
        "hundman_fp": int(fp),
        "hundman_fn": int(fn),
        "hundman_matched_pred_events": int(matched_pred_count),
        "hundman_precision": float(precision),
        "hundman_recall": float(recall),
        "hundman_f1": float(f1),
        "hundman_num_true_events": int(len(true_events)),
        "hundman_num_pred_events": int(len(pred_events)),
    }


def _add_contextual_overlap_prefixed(row, prefix, y_true_sequence, test_score_sequence, calib_score_sequence=None):
    y_true_sequence = np.asarray(y_true_sequence).astype(bool)
    test_score_sequence = np.asarray(test_score_sequence, dtype=np.float64)
    if len(y_true_sequence) != len(test_score_sequence):
        raise ValueError(f"{prefix} contextual metric length mismatch: y={len(y_true_sequence)} score={len(test_score_sequence)}")
    pred, threshold = threshold_scores_for_hundman(
        test_score_sequence,
        calib_scores=calib_score_sequence,
        q=dataspace_cfg.hundman_threshold_quantile,
    )
    hund = hundman_window_overlap_metrics(y_true_sequence, pred)
    for key, value in hund.items():
        row[f"{prefix}_{key.replace('hundman_', '')}"] = value
    row[f"{prefix}_threshold"] = float(threshold)
    row[f"{prefix}_threshold_quantile"] = float(dataspace_cfg.hundman_threshold_quantile)
    for k in getattr(dataspace_cfg, "hundman_sd_thresholds", (2.0, 3.0)):
        pred_sd, threshold_sd, mu_sd, sd_sd = threshold_scores_for_hundman_sd(
            test_score_sequence,
            calib_scores=calib_score_sequence,
            k=k,
        )
        hund_sd = hundman_window_overlap_metrics(y_true_sequence, pred_sd)
        suffix = f"mean_plus_{int(k) if float(k).is_integer() else str(k).replace('.', 'p')}sd"
        for key, value in hund_sd.items():
            row[f"{prefix}_{key.replace('hundman_', '')}_{suffix}"] = value
        row[f"{prefix}_threshold_{suffix}"] = float(threshold_sd)
        row[f"{prefix}_threshold_mean_{suffix}"] = float(mu_sd)
        row[f"{prefix}_threshold_sd_{suffix}"] = float(sd_sd)
        row[f"{prefix}_threshold_k_{suffix}"] = float(k)
    return row


def add_hundman_fields_to_row(
    row,
    y_true_window,
    test_score_z,
    calib_score_z=None,
    test_starts=None,
    point_labels=None,
    window=None,
):
    """
    Add contextual overlap metrics for three ordered sequences:

    hundman_window_*: raw window score sequence before point aggregation.
    hundman_point_mean_*: point score sequence from window_scores_to_points(..., mode="mean").
    hundman_point_max_*: point score sequence from window_scores_to_points(..., mode="max").
    """
    if not bool(getattr(dataspace_cfg, "compute_hundman_metrics", False)):
        return row
    _add_contextual_overlap_prefixed(row, "hundman_window", y_true_window, test_score_z, calib_score_z)
    row["hundman_window_sequence"] = "raw_window_scores"
    if test_starts is not None and point_labels is not None and window is not None:
        point_labels_bool = np.asarray(point_labels).astype(bool)
        point_mean = window_scores_to_points(test_score_z, test_starts, len(point_labels_bool), window, mode="mean")
        point_max = window_scores_to_points(test_score_z, test_starts, len(point_labels_bool), window, mode="max")
        _add_contextual_overlap_prefixed(row, "hundman_point_mean", point_labels_bool, point_mean, calib_score_sequence=None)
        _add_contextual_overlap_prefixed(row, "hundman_point_max", point_labels_bool, point_max, calib_score_sequence=None)
        row["hundman_point_mean_sequence"] = "point_scores_mean_aggregation"
        row["hundman_point_max_sequence"] = "point_scores_max_aggregation"
    return row


# ============================================================
# Generic derived scoring: smoothing and temporal post-processing
# ============================================================

def robust_zscore(scores, calib_scores=None, eps=1e-8, clip=10.0):
    """
    Robustly normalize anomaly scores using median and MAD.

    If calib_scores is provided, compute median/MAD from calib_scores.
    Otherwise compute them from scores. Larger returned values mean more
    anomalous. No anomaly labels are used.
    """
    scores = np.asarray(scores, dtype=np.float64)
    ref = scores if calib_scores is None else np.asarray(calib_scores, dtype=np.float64)
    median = np.nanmedian(ref)
    mad = np.nanmedian(np.abs(ref - median))
    z = (scores - median) / (1.4826 * mad + eps)
    return np.clip(np.nan_to_num(z, nan=0.0, posinf=clip, neginf=-clip), -clip, clip)


def positive_part(x):
    """Return max(x, 0) elementwise."""
    return np.maximum(np.asarray(x, dtype=np.float64), 0.0)


def stable_sigmoid(x):
    """Numerically stable sigmoid with clipping."""
    x = np.clip(np.asarray(x, dtype=np.float64), -10, 10)
    return 1.0 / (1.0 + np.exp(-x))


def hp_filter_score(scores, lamb=1600.0, component="trend", mode="raw", negate=False):
    """
    Apply Hodrick-Prescott filtering to a 1D anomaly score sequence.

    statsmodels.hpfilter returns cycle, trend = hpfilter(x, lamb=lamb).
    component="trend" uses the low-frequency smoothed score, matching the
    TadFlow-style HP trend usage. component="cycle" is a high-frequency
    diagnostic. mode controls whether the selected component is returned raw,
    positive-part clipped, or absolute-valued.
    """
    arr = np.asarray(scores, dtype=np.float64)
    fill = np.nanmedian(arr) if np.isfinite(arr).any() else 0.0
    x = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    if negate:
        x = -x
    try:
        from statsmodels.tsa.filters.hp_filter import hpfilter

        cycle, trend = hpfilter(x, lamb=lamb)
        cycle = np.asarray(cycle, dtype=np.float64)
        trend = np.asarray(trend, dtype=np.float64)
    except Exception:
        trend = pd.Series(x).rolling(window=31, center=True, min_periods=1).median().to_numpy()
        cycle = x - trend

    if component == "trend":
        selected = trend
    elif component == "cycle":
        selected = cycle
    else:
        raise ValueError(f"unsupported HP component: {component}")

    if mode == "raw":
        return np.asarray(selected, dtype=np.float64)
    if mode == "positive":
        return positive_part(selected)
    if mode == "abs":
        return np.abs(selected)
    raise ValueError(f"unsupported HP mode: {mode}")


def rolling_mean_score(scores, window=5):
    """Smooth score sequence using centered rolling mean."""
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0)
    return pd.Series(scores).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def rolling_max_score(scores, window=5):
    """Smooth score sequence using centered rolling max."""
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0)
    return pd.Series(scores).rolling(window=window, center=True, min_periods=1).max().to_numpy()


def _is_score_array(value):
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    return arr.ndim == 1 and arr.size > 0 and np.issubdtype(arr.dtype, np.number)


def build_generic_derived_scores(score_dict, hp_lambda=1600.0, rolling_window=5, add_hp_cycle_scores=False):
    """
    Given a dictionary of raw window-level scores for any method, create
    generic derived scores. This applies fairly to AE, Raw Gaussian, Vanilla
    FM, FDM-lite, and HCFM.

    For every score key, create hp_trend_<score_name>, rollmean_<score_name>,
    and rollmax_<score_name>. Optional hp_cycle_<score_name> diagnostics are
    created only when add_hp_cycle_scores=True. The input dictionary is not
    modified.
    """
    derived = {}
    for name, values in score_dict.items():
        if not _is_score_array(values):
            continue
        arr = np.asarray(values, dtype=np.float64)
        derived[f"hp_trend_{name}"] = hp_filter_score(arr, lamb=hp_lambda, component="trend", mode="raw", negate=False)
        if add_hp_cycle_scores:
            derived[f"hp_cycle_{name}"] = hp_filter_score(arr, lamb=hp_lambda, component="cycle", mode="abs", negate=False)
        derived[f"rollmean_{name}"] = rolling_mean_score(arr, window=rolling_window)
        derived[f"rollmax_{name}"] = rolling_max_score(arr, window=rolling_window)
    return derived



SELECTED_DERIVED_SCORES = {
    "Data HCFM": {
        "transport_energy_x": ["hp_trend", "rollmean", "rollmax"],
        "compression_div_neg_sq_x": ["hp_trend", "rollmean", "rollmax"],
        "compression_div_signed_x": ["hp_trend", "rollmean", "rollmax"],
        "compression_div_centered_abs_x": ["hp_trend", "rollmean", "rollmax"],
        "compression_div_centered_sq_x": ["hp_trend", "rollmean", "rollmax"],
        "compression_div_centered_neg_abs_x": ["hp_trend", "rollmean", "rollmax"],
        "compression_div_centered_neg_sq_x": ["hp_trend", "rollmean", "rollmax"],
        "compression_div_neg_abs_x": ["hp_trend", "rollmean", "rollmax"],
        "base_plus_compression_div_mean_x": ["hp_trend", "rollmean", "rollmax"],
        "base_minus_compression_div_mean_x": ["hp_trend", "rollmean", "rollmax"],
        "base_plus_full_div_mean_x": ["hp_trend", "rollmean", "rollmax"],
        "base_minus_full_div_mean_x": ["hp_trend", "rollmean", "rollmax"],
        "cnf_base_plus_compression_trapz_x": ["hp_trend", "rollmean", "rollmax"],
        "cnf_base_minus_compression_trapz_x": ["hp_trend", "rollmean", "rollmax"],
        "cnf_base_plus_full_trapz_x": ["hp_trend", "rollmean", "rollmax"],
        "cnf_base_minus_full_trapz_x": ["hp_trend", "rollmean", "rollmax"],
    },
    "AE": {"reconstruction_mse_x": ["rollmax"]},
    "Raw Gaussian": {"mahalanobis_x": ["rollmax"]},
}


def build_selected_derived_scores(method, score_dict, selected=None, hp_lambda=1600.0, rolling_window=5):
    selected = SELECTED_DERIVED_SCORES if selected is None else selected
    derived = {}
    for name, transforms in selected.get(method, {}).items():
        if name not in score_dict or not _is_score_array(score_dict[name]):
            continue
        arr = np.asarray(score_dict[name], dtype=np.float64)
        for transform in transforms:
            if transform == "hp_trend":
                derived[f"hp_trend_{name}"] = hp_filter_score(arr, lamb=hp_lambda, component="trend", mode="raw", negate=False)
            elif transform == "rollmean":
                derived[f"rollmean_{name}"] = rolling_mean_score(arr, window=rolling_window)
            elif transform == "rollmax":
                derived[f"rollmax_{name}"] = rolling_max_score(arr, window=rolling_window)
            else:
                raise ValueError(f"unknown selected derived transform: {transform}")
    return derived

# ============================================================
# HCFM mechanism-aware derived scoring
# ============================================================

def build_hcfm_mechanism_scores(score_dict, calib_score_dict=None, hp_lambda=1600.0, rolling_window=5):
    """
    Build HCFM-only mechanism-aware fusion scores from HCFM component scores.

    Expected raw HCFM score keys when available: base_energy_x,
    compression_div_abs_x, compression_div_signed_x, compression_div_pos_x,
    compression_div_neg_x, transport_energy_x, compression_energy_x, residual_energy_x, and
    fm_consistency_x. These scores are HCFM-only because Vanilla FM/FDM do not
    expose separate transport/compression/residual mechanisms. No anomaly
    labels are used.
    """
    # TODO: Prefer calibration-window statistics here to avoid any test-set
    # distribution dependence when calib_score_dict is unavailable.
    derived = {}

    def raw(name):
        return np.asarray(score_dict[name], dtype=np.float64) if name in score_dict and _is_score_array(score_dict[name]) else None

    def ref(name):
        if calib_score_dict is not None and name in calib_score_dict and _is_score_array(calib_score_dict[name]):
            return np.asarray(calib_score_dict[name], dtype=np.float64)
        return None

    z = {}
    for short, name in {
        "base": "base_energy_x",
        "div": "compression_div_abs_x",
        "signed_div": "compression_div_signed_x",
        "pos_div": "compression_div_pos_x",
        "neg_div": "compression_div_neg_x",
        "comp_energy": "compression_energy_x",
        "residual": "residual_energy_x",
        "consistency": "fm_consistency_x",
    }.items():
        values = raw(name)
        if values is not None:
            z[short] = robust_zscore(values, ref(name))

    hp_aliases = {
        "base_energy_x": "hcfm_hp_trend_base_energy_x",
        "compression_div_abs_x": "hcfm_hp_trend_compression_div_abs_x",
        "compression_div_signed_x": "hcfm_hp_trend_compression_div_signed_x",
        "compression_div_pos_x": "hcfm_hp_trend_compression_div_pos_x",
        "compression_div_neg_x": "hcfm_hp_trend_compression_div_neg_x",
        "compression_energy_x": "hcfm_hp_trend_compression_energy_x",
        "residual_energy_x": "hcfm_hp_trend_residual_energy_x",
        "fm_consistency_x": "hcfm_hp_trend_fm_consistency_x",
    }
    for raw_name, alias in hp_aliases.items():
        values = raw(raw_name)
        if values is not None:
            derived[alias] = hp_filter_score(values, lamb=hp_lambda, component="trend", mode="raw", negate=False)

    mechanism_parts = [z[k] for k in ["base", "div", "comp_energy", "residual"] if k in z]
    if mechanism_parts:
        derived["hcfm_max_mechanism_z"] = np.maximum.reduce(mechanism_parts)

    if "base" in z and "div" in z:
        derived["hcfm_add_base_div_abs_z"] = z["base"] + z["div"]
        derived["hcfm_mul_base_div_abs_z"] = np.sqrt(positive_part(z["base"]) * positive_part(z["div"]))
        derived["hcfm_gated_base_plus_div_abs_z"] = z["base"] + stable_sigmoid(z["base"]) * positive_part(z["div"])
        derived["hcfm_gated_div_abs_plus_base_z"] = z["div"] + stable_sigmoid(z["div"]) * positive_part(z["base"])
    if "base" in z and "signed_div" in z:
        derived["hcfm_add_base_signed_div_z"] = z["base"] + z["signed_div"]
        derived["hcfm_gated_base_plus_signed_div_z"] = z["base"] + stable_sigmoid(z["base"]) * z["signed_div"]
    if "base" in z and "pos_div" in z:
        derived["hcfm_add_base_pos_div_z"] = z["base"] + z["pos_div"]
        derived["hcfm_gated_base_plus_pos_div_z"] = z["base"] + stable_sigmoid(z["base"]) * positive_part(z["pos_div"])
    if "base" in z and "neg_div" in z:
        derived["hcfm_add_base_neg_div_z"] = z["base"] + z["neg_div"]
        derived["hcfm_gated_base_plus_neg_div_z"] = z["base"] + stable_sigmoid(z["base"]) * positive_part(z["neg_div"])
    if "base" in z and "div" in z and "comp_energy" in z:
        derived["hcfm_add_base_div_abs_comp_z"] = z["base"] + z["div"] + z["comp_energy"]
    if "base" in z and "signed_div" in z and "comp_energy" in z:
        derived["hcfm_add_base_signed_div_comp_z"] = z["base"] + z["signed_div"] + z["comp_energy"]
    if all(k in z for k in ["base", "div", "comp_energy", "residual"]):
        derived["hcfm_add_all_absdiv_mechanisms_z"] = z["base"] + z["div"] + z["comp_energy"] + z["residual"]
    if all(k in z for k in ["base", "signed_div", "comp_energy", "residual"]):
        derived["hcfm_add_all_signeddiv_mechanisms_z"] = z["base"] + z["signed_div"] + z["comp_energy"] + z["residual"]
    if "comp_energy" in z and "div" in z:
        derived["hcfm_mul_compenergy_div_z"] = np.sqrt(positive_part(z["comp_energy"]) * positive_part(z["div"]))
    if "base" in z and "residual" in z:
        derived["hcfm_mul_base_residual_z"] = np.sqrt(positive_part(z["base"]) * positive_part(z["residual"]))

    rolling_aliases = {
        "compression_div_abs_x": "hcfm_rollmean_compression_div_abs_x",
        "compression_div_signed_x": "hcfm_rollmean_compression_div_signed_x",
        "compression_div_pos_x": "hcfm_rollmean_compression_div_pos_x",
        "compression_div_neg_x": "hcfm_rollmean_compression_div_neg_x",
        "base_energy_x": "hcfm_rollmean_base_energy_x",
        "compression_energy_x": "hcfm_rollmean_compression_energy_x",
        "residual_energy_x": "hcfm_rollmean_residual_energy_x",
    }
    for raw_name, alias in rolling_aliases.items():
        values = raw(raw_name)
        if values is not None:
            derived[alias] = rolling_mean_score(values, window=rolling_window)
    if "hcfm_max_mechanism_z" in derived:
        derived["hcfm_rollmean_max_mechanism_z"] = rolling_mean_score(derived["hcfm_max_mechanism_z"], window=rolling_window)

    return derived


# %% notebook cell 42
y = test_y.astype(int)
print("score calibration mode:", dataspace_cfg.score_calibration_mode)
metrics_rows, scores_z, scores_raw, score_stats = [], {}, {}, []
metric_jobs = []

# Rare-normal subset is only for diagnostics. Prefer Vanilla FM base, then
# Mahalanobis if enabled; otherwise disable the rare-normal diagnostic mask.
rare_source_calib = None
rare_source_test = None
if globals().get("van_base_calib") is not None and globals().get("van_base_test") is not None:
    rare_source_calib = van_base_calib
    rare_source_test = van_base_test
elif globals().get("mahal_calib") is not None and globals().get("mahal_test") is not None:
    rare_source_calib = mahal_calib
    rare_source_test = mahal_test

if rare_source_calib is not None:
    _, rare_score_for_normal, _ = standardize_dataspace_score(rare_source_calib, rare_source_test)
    rare_threshold = np.quantile(rare_score_for_normal[y == 0], 0.90) if np.any(y == 0) else np.nan
    rare_normal_mask = (y == 0) & (rare_score_for_normal >= rare_threshold)
else:
    rare_normal_mask = np.zeros_like(y, dtype=bool)
print("rare normal windows:", int(rare_normal_mask.sum()), "of", int((y == 0).sum()))

if globals().get("ae_rec_calib") is not None and globals().get("ae_rec_test") is not None:
    add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "AE", "reconstruction_mse_x", ae_rec_calib, ae_rec_test, y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
if globals().get("mahal_calib") is not None and globals().get("mahal_test") is not None:
    add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Raw Gaussian", "mahalanobis_x", mahal_calib, mahal_test, y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)

if globals().get("van_base_calib") is not None:
    add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Vanilla Data FM", "base_energy_x", van_base_calib, van_base_test, y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
if globals().get("van_consistency_calib") is not None:
    add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Vanilla Data FM", "fm_consistency_x", van_consistency_calib, van_consistency_test, y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)

if globals().get("fdm_base_calib") is not None:
    add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data FDM-lite", "base_energy_x", fdm_base_calib, fdm_base_test, y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
if globals().get("fdm_consistency_calib") is not None:
    add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data FDM-lite", "fm_consistency_x", fdm_consistency_calib, fdm_consistency_test, y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)

hcfm_base_calib_z = hcfm_base_z = None
hcfm_consistency_calib_z = hcfm_consistency_z = None
hcfm_compression_div_abs_calib_z = hcfm_compression_div_abs_z = None
hcfm_compression_div_signed_calib_z = hcfm_compression_div_signed_z = None
hcfm_compression_div_pos_calib_z = hcfm_compression_div_pos_z = None
hcfm_compression_div_neg_calib_z = hcfm_compression_div_neg_z = None
hcfm_compression_div_neg_abs_calib_z = hcfm_compression_div_neg_abs_z = None
hcfm_compression_div_sq_calib_z = hcfm_compression_div_sq_z = None
hcfm_compression_div_neg_sq_calib_z = hcfm_compression_div_neg_sq_z = None
hcfm_compression_div_centered_abs_calib_z = hcfm_compression_div_centered_abs_z = None
hcfm_compression_div_centered_sq_calib_z = hcfm_compression_div_centered_sq_z = None
hcfm_compression_div_centered_neg_abs_calib_z = hcfm_compression_div_centered_neg_abs_z = None
hcfm_compression_div_centered_neg_sq_calib_z = hcfm_compression_div_centered_neg_sq_z = None
hcfm_residual_div_signed_calib_z = hcfm_residual_div_signed_z = None

if globals().get("hcfm_calib_scores") is not None:
    hcfm_base_calib_z, hcfm_base_z = add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "base_energy_x", hcfm_calib_scores["base_energy"], hcfm_test_scores["base_energy"], y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
    if globals().get("hcfm_consistency_calib") is not None:
        hcfm_consistency_calib_z, hcfm_consistency_z = add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "fm_consistency_x", hcfm_consistency_calib, hcfm_consistency_test, y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
    if "transport_energy" in hcfm_calib_scores:
        add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "transport_energy_x", hcfm_calib_scores["transport_energy"], hcfm_test_scores["transport_energy"], y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
    if "compression_energy" in hcfm_calib_scores:
        add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "compression_energy_x", hcfm_calib_scores["compression_energy"], hcfm_test_scores["compression_energy"], y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
    if "residual_energy" in hcfm_calib_scores:
        add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "residual_energy_x", hcfm_calib_scores["residual_energy"], hcfm_test_scores["residual_energy"], y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
    if "compression_div_abs" in hcfm_calib_scores:
        hcfm_compression_div_abs_calib_z, hcfm_compression_div_abs_z = add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "compression_div_abs_x", hcfm_calib_scores["compression_div_abs"], hcfm_test_scores["compression_div_abs"], y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
    if "compression_div_signed" in hcfm_calib_scores:
        hcfm_compression_div_signed_calib_z, hcfm_compression_div_signed_z = add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "compression_div_signed_x", hcfm_calib_scores["compression_div_signed"], hcfm_test_scores["compression_div_signed"], y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
    if "compression_div_pos" in hcfm_calib_scores:
        hcfm_compression_div_pos_calib_z, hcfm_compression_div_pos_z = add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "compression_div_pos_x", hcfm_calib_scores["compression_div_pos"], hcfm_test_scores["compression_div_pos"], y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
    if "compression_div_neg" in hcfm_calib_scores:
        hcfm_compression_div_neg_calib_z, hcfm_compression_div_neg_z = add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "compression_div_neg_x", hcfm_calib_scores["compression_div_neg"], hcfm_test_scores["compression_div_neg"], y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
    if hcfm_compression_div_abs_calib_z is not None:
        hcfm_compression_div_neg_abs_calib_z, hcfm_compression_div_neg_abs_z = add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "compression_div_neg_abs_x", -hcfm_calib_scores["compression_div_abs"], -hcfm_test_scores["compression_div_abs"], y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
    if "compression_div_sq" in hcfm_calib_scores:
        hcfm_compression_div_sq_calib_z, hcfm_compression_div_sq_z = add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "compression_div_sq_x", hcfm_calib_scores["compression_div_sq"], hcfm_test_scores["compression_div_sq"], y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
    if "compression_div_neg_sq" in hcfm_calib_scores:
        hcfm_compression_div_neg_sq_calib_z, hcfm_compression_div_neg_sq_z = add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "compression_div_neg_sq_x", hcfm_calib_scores["compression_div_neg_sq"], hcfm_test_scores["compression_div_neg_sq"], y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
    if "compression_div_signed" in hcfm_calib_scores:
        calib_div_signed_raw = score_to_numpy(hcfm_calib_scores["compression_div_signed"])
        test_div_signed_raw = score_to_numpy(hcfm_test_scores["compression_div_signed"])

        calib_div_median = float(np.median(calib_div_signed_raw))
        calib_div_mad = float(np.median(np.abs(calib_div_signed_raw - calib_div_median)))
        calib_div_scale = 1.4826 * calib_div_mad + 1e-8

        calib_centered = (calib_div_signed_raw - calib_div_median) / calib_div_scale
        test_centered = (test_div_signed_raw - calib_div_median) / calib_div_scale

        print("Compression-divergence calibration centering:")
        print(f"median={calib_div_median:.6g}, MAD={calib_div_mad:.6g}, robust_scale={calib_div_scale:.6g}")

        hcfm_compression_div_centered_abs_calib_z, hcfm_compression_div_centered_abs_z = add_dataspace_score(
            metrics_rows, scores_z, scores_raw, score_stats,
            "Data HCFM", "compression_div_centered_abs_x",
            np.abs(calib_centered), np.abs(test_centered),
            y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask,
        )
        hcfm_compression_div_centered_sq_calib_z, hcfm_compression_div_centered_sq_z = add_dataspace_score(
            metrics_rows, scores_z, scores_raw, score_stats,
            "Data HCFM", "compression_div_centered_sq_x",
            np.square(calib_centered), np.square(test_centered),
            y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask,
        )
        hcfm_compression_div_centered_neg_abs_calib_z, hcfm_compression_div_centered_neg_abs_z = add_dataspace_score(
            metrics_rows, scores_z, scores_raw, score_stats,
            "Data HCFM", "compression_div_centered_neg_abs_x",
            -np.abs(calib_centered), -np.abs(test_centered),
            y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask,
        )
        hcfm_compression_div_centered_neg_sq_calib_z, hcfm_compression_div_centered_neg_sq_z = add_dataspace_score(
            metrics_rows, scores_z, scores_raw, score_stats,
            "Data HCFM", "compression_div_centered_neg_sq_x",
            -np.square(calib_centered), -np.square(test_centered),
            y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask,
        )
    if "transport_div_abs" in hcfm_calib_scores:
        add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "transport_div_abs_x", hcfm_calib_scores["transport_div_abs"], hcfm_test_scores["transport_div_abs"], y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
    if "residual_div_abs" in hcfm_calib_scores:
        add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "residual_div_abs_x", hcfm_calib_scores["residual_div_abs"], hcfm_test_scores["residual_div_abs"], y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
    if "residual_div_signed" in hcfm_calib_scores:
        hcfm_residual_div_signed_calib_z, hcfm_residual_div_signed_z = add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "residual_div_signed_x", hcfm_calib_scores["residual_div_signed"], hcfm_test_scores["residual_div_signed"], y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
    if "effective_residual_div_abs" in hcfm_calib_scores:
        add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "effective_residual_div_abs_x", hcfm_calib_scores["effective_residual_div_abs"], hcfm_test_scores["effective_residual_div_abs"], y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
    if "transport_energy" in hcfm_calib_scores:
        for split, vals in [("calib", hcfm_calib_scores["transport_energy"]), ("test", hcfm_test_scores["transport_energy"] )]:
            arr = score_to_numpy(vals)
            print(
                f"transport_energy {split}: "
                f"mean={arr.mean():.6g}, std={arr.std():.6g}, "
                f"min={arr.min():.6g}, max={arr.max():.6g}"
            )
    if dataspace_cfg.compute_full_likelihood_proxy and "base_energy" in hcfm_calib_scores and "compression_div_signed" in hcfm_calib_scores:
        hcfm_base_plus_scaled_compression_div_calib = (
            hcfm_calib_scores["base_energy"]
            + dataspace_cfg.hcfm_gamma_compression * hcfm_calib_scores["compression_div_signed"]
        )
        hcfm_base_plus_scaled_compression_div_test = (
            hcfm_test_scores["base_energy"]
            + dataspace_cfg.hcfm_gamma_compression * hcfm_test_scores["compression_div_signed"]
        )
        add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "base_plus_scaled_compression_div_x", hcfm_base_plus_scaled_compression_div_calib, hcfm_base_plus_scaled_compression_div_test, y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
        add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "cnf_proxy_base_plus_compression_div_x", hcfm_base_plus_scaled_compression_div_calib, hcfm_base_plus_scaled_compression_div_test, y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
        hcfm_base_minus_scaled_compression_div_calib = (
            hcfm_calib_scores["base_energy"]
            - dataspace_cfg.hcfm_gamma_compression * hcfm_calib_scores["compression_div_signed"]
        )
        hcfm_base_minus_scaled_compression_div_test = (
            hcfm_test_scores["base_energy"]
            - dataspace_cfg.hcfm_gamma_compression * hcfm_test_scores["compression_div_signed"]
        )
        add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "base_minus_scaled_compression_div_x", hcfm_base_minus_scaled_compression_div_calib, hcfm_base_minus_scaled_compression_div_test, y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
        if "residual_div_signed" in hcfm_calib_scores:
            hcfm_base_plus_scaled_full_div_calib = (
                hcfm_calib_scores["base_energy"]
                + dataspace_cfg.hcfm_gamma_compression * hcfm_calib_scores["compression_div_signed"]
                + dataspace_cfg.hcfm_gamma_residual * hcfm_calib_scores["residual_div_signed"]
            )
            hcfm_base_plus_scaled_full_div_test = (
                hcfm_test_scores["base_energy"]
                + dataspace_cfg.hcfm_gamma_compression * hcfm_test_scores["compression_div_signed"]
                + dataspace_cfg.hcfm_gamma_residual * hcfm_test_scores["residual_div_signed"]
            )
            add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "base_plus_scaled_full_div_x", hcfm_base_plus_scaled_full_div_calib, hcfm_base_plus_scaled_full_div_test, y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
            add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "cnf_proxy_base_plus_full_div_x", hcfm_base_plus_scaled_full_div_calib, hcfm_base_plus_scaled_full_div_test, y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
            hcfm_base_minus_scaled_full_div_calib = (
                hcfm_calib_scores["base_energy"]
                - dataspace_cfg.hcfm_gamma_compression * hcfm_calib_scores["compression_div_signed"]
                - dataspace_cfg.hcfm_gamma_residual * hcfm_calib_scores["residual_div_signed"]
            )
            hcfm_base_minus_scaled_full_div_test = (
                hcfm_test_scores["base_energy"]
                - dataspace_cfg.hcfm_gamma_compression * hcfm_test_scores["compression_div_signed"]
                - dataspace_cfg.hcfm_gamma_residual * hcfm_test_scores["residual_div_signed"]
            )
            add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "base_minus_scaled_full_div_x", hcfm_base_minus_scaled_full_div_calib, hcfm_base_minus_scaled_full_div_test, y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
    def add_hcfm_z_combined_score(score_name, div_calib_z, div_test_z):
        if hcfm_consistency_calib_z is None or div_calib_z is None:
            return None, None
        combined_calib_z = hcfm_consistency_calib_z + dataspace_cfg.hcfm_combined_alpha * div_calib_z
        combined_z = hcfm_consistency_z + dataspace_cfg.hcfm_combined_alpha * div_test_z
        key = f"Data HCFM / {score_name}"
        scores_z[key] = combined_z
        scores_raw[key] = {
            "calib_raw": combined_calib_z,
            "test_raw": combined_z,
            "calib_z": combined_calib_z,
            "test_z": combined_z,
        }
        combined_q25, combined_q75 = np.quantile(combined_calib_z, [0.25, 0.75])
        score_stats.append({
            "score_key": key,
            "calibration_mode": "prestandardized_fusion",
            "calib_center": 0.0,
            "calib_scale": 1.0,
            "calib_mean": float(np.mean(combined_calib_z)),
            "calib_std": float(np.std(combined_calib_z)),
            "calib_median": float(np.median(combined_calib_z)),
            "calib_iqr": float(combined_q75 - combined_q25),
            "calib_min": float(np.min(combined_calib_z)),
            "calib_max": float(np.max(combined_calib_z)),
        })
        enqueue_dataspace_metric(
            metrics_rows,
            "Data HCFM",
            score_name,
            combined_calib_z,
            combined_z,
            y,
            test_starts,
            point_labels,
            dataspace_cfg.window,
            rare_normal_mask,
        )
        return combined_calib_z, combined_z

    if dataspace_cfg.compute_exact_likelihood_scores:
        if "base_energy" in hcfm_calib_scores and "compression_div_signed" in hcfm_calib_scores:
            comp_div_mean_calib = dataspace_cfg.hcfm_gamma_compression * hcfm_calib_scores["compression_div_signed"]
            comp_div_mean_test = dataspace_cfg.hcfm_gamma_compression * hcfm_test_scores["compression_div_signed"]
            full_div_mean_calib = comp_div_mean_calib
            full_div_mean_test = comp_div_mean_test
            if dataspace_cfg.exact_likelihood_include_residual and "residual_div_signed" in hcfm_calib_scores:
                full_div_mean_calib = full_div_mean_calib + dataspace_cfg.hcfm_gamma_residual * hcfm_calib_scores["residual_div_signed"]
                full_div_mean_test = full_div_mean_test + dataspace_cfg.hcfm_gamma_residual * hcfm_test_scores["residual_div_signed"]
            elif dataspace_cfg.exact_likelihood_include_residual:
                print("WARNING: mean full-div rows requested residual contribution, but residual_div_signed was unavailable; using compression-only full_div fallback.")

            mean_div_derived = [
                ("compression_div_mean_scaled_x", comp_div_mean_calib, comp_div_mean_test),
                ("full_div_mean_scaled_x", full_div_mean_calib, full_div_mean_test),
                ("base_plus_compression_div_mean_x", hcfm_calib_scores["base_energy"] + comp_div_mean_calib, hcfm_test_scores["base_energy"] + comp_div_mean_test),
                ("base_minus_compression_div_mean_x", hcfm_calib_scores["base_energy"] - comp_div_mean_calib, hcfm_test_scores["base_energy"] - comp_div_mean_test),
                ("base_plus_full_div_mean_x", hcfm_calib_scores["base_energy"] + full_div_mean_calib, hcfm_test_scores["base_energy"] + full_div_mean_test),
                ("base_minus_full_div_mean_x", hcfm_calib_scores["base_energy"] - full_div_mean_calib, hcfm_test_scores["base_energy"] - full_div_mean_test),
            ]
            for score_name, calib_score, test_score in mean_div_derived:
                add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", score_name, calib_score, test_score, y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
        else:
            print("WARNING: compute_exact_likelihood_scores=True, but base_energy/compression_div_signed were unavailable; no mean-div likelihood rows added.")

    for direct_key, score_name in [
        ("compression_div_trapz", "compression_div_trapz_x"),
        ("cnf_base_plus_compression_trapz", "cnf_base_plus_compression_trapz_x"),
        ("cnf_base_minus_compression_trapz", "cnf_base_minus_compression_trapz_x"),
        ("residual_div_trapz", "residual_div_trapz_x"),
        ("cnf_base_plus_full_trapz", "cnf_base_plus_full_trapz_x"),
        ("cnf_base_minus_full_trapz", "cnf_base_minus_full_trapz_x"),
    ]:
        if direct_key in hcfm_calib_scores and direct_key in hcfm_test_scores:
            add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", score_name, hcfm_calib_scores[direct_key], hcfm_test_scores[direct_key], y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)

    if getattr(dataspace_cfg, "run_scalar_potential_audit", False) and globals().get("hcfm_diag_calib_scores") is not None:
        for diag_key, score_name in [
            ("cnf_nll_compression_trapz", "cnf_nll_compression_trapz_x"),
            ("cnf_nll_full_trapz", "cnf_nll_full_trapz_x"),
            ("cnf_nll_full_trapz_minus", "cnf_nll_full_trapz_minus_x"),
            ("cycle_reconstruction_mse", "cycle_reconstruction_mse_x"),
            ("reverse_base_energy", "reverse_base_energy_x"),
            ("forward_cycle_error", "forward_cycle_error_x"),
        ]:
            if diag_key in hcfm_diag_calib_scores and diag_key in hcfm_diag_test_scores:
                add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", score_name, hcfm_diag_calib_scores[diag_key], hcfm_diag_test_scores[diag_key], y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask)
    if dataspace_cfg.compute_mechanism_fusions:
        # Consistency fusion rows are exploratory/debug-only. Core profile keeps
        # the compression-curvature scores clean and avoids the extra consistency pass.
        if hcfm_compression_div_signed_calib_z is not None:
            add_hcfm_z_combined_score("consistency_plus_compression_div_x", hcfm_compression_div_signed_calib_z, hcfm_compression_div_signed_z)
            add_hcfm_z_combined_score("consistency_plus_compression_signed_div_x", hcfm_compression_div_signed_calib_z, hcfm_compression_div_signed_z)
        elif hcfm_compression_div_abs_calib_z is not None:
            add_hcfm_z_combined_score("consistency_plus_compression_div_x", hcfm_compression_div_abs_calib_z, hcfm_compression_div_abs_z)
        if hcfm_compression_div_abs_calib_z is not None:
            add_hcfm_z_combined_score("consistency_plus_compression_div_abs_x", hcfm_compression_div_abs_calib_z, hcfm_compression_div_abs_z)
            add_hcfm_z_combined_score("consistency_minus_compression_div_abs_x", -hcfm_compression_div_abs_calib_z, -hcfm_compression_div_abs_z)
        if hcfm_compression_div_neg_sq_calib_z is not None:
            add_hcfm_z_combined_score("consistency_plus_compression_neg_sq_x", hcfm_compression_div_neg_sq_calib_z, hcfm_compression_div_neg_sq_z)
        if hcfm_compression_div_centered_neg_sq_calib_z is not None:
            add_hcfm_z_combined_score("consistency_plus_compression_centered_neg_sq_x", hcfm_compression_div_centered_neg_sq_calib_z, hcfm_compression_div_centered_neg_sq_z)
        if hcfm_compression_div_pos_calib_z is not None:
            add_hcfm_z_combined_score("consistency_plus_compression_pos_div_x", hcfm_compression_div_pos_calib_z, hcfm_compression_div_pos_z)
        if hcfm_compression_div_neg_calib_z is not None:
            add_hcfm_z_combined_score("consistency_plus_compression_neg_div_x", hcfm_compression_div_neg_calib_z, hcfm_compression_div_neg_z)
else:
    print("No Data HCFM scores available; skipping HCFM metric rows.")


# Add generic temporal post-processing for every method and HCFM-only
# mechanism-aware fusion scores. These are scoring-only and use no labels.
hp_lambda = getattr(dataspace_cfg, "hp_lambda", 1600.0)
rolling_window = 5
add_hp_cycle_scores = getattr(dataspace_cfg, "add_hp_cycle_scores", False)
raw_items = list(scores_raw.items())
method_score_names = {}
for key, values in raw_items:
    method, score_name = key.split(" / ", 1)
    method_score_names.setdefault(method, []).append(score_name)

for method, score_names in method_score_names.items():
    calib_dict = {name: scores_raw[f"{method} / {name}"]["calib_raw"] for name in score_names}
    test_dict = {name: scores_raw[f"{method} / {name}"]["test_raw"] for name in score_names}

    if dataspace_cfg.compute_all_generic_smoothing:
        generic_calib = build_generic_derived_scores(calib_dict, hp_lambda=hp_lambda, rolling_window=rolling_window, add_hp_cycle_scores=add_hp_cycle_scores)
        generic_test = build_generic_derived_scores(test_dict, hp_lambda=hp_lambda, rolling_window=rolling_window, add_hp_cycle_scores=add_hp_cycle_scores)
    else:
        generic_calib = build_selected_derived_scores(method, calib_dict, hp_lambda=hp_lambda, rolling_window=rolling_window)
        generic_test = build_selected_derived_scores(method, test_dict, hp_lambda=hp_lambda, rolling_window=rolling_window)
    for score_name in generic_test:
        if score_name in generic_calib:
            add_dataspace_score(
                metrics_rows, scores_z, scores_raw, score_stats,
                method, score_name, generic_calib[score_name], generic_test[score_name],
                y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask,
            )

    if method == "Data HCFM" and dataspace_cfg.compute_mechanism_fusions:
        hcfm_mech_calib = build_hcfm_mechanism_scores(calib_dict, calib_score_dict=calib_dict, hp_lambda=hp_lambda, rolling_window=rolling_window)
        hcfm_mech_test = build_hcfm_mechanism_scores(test_dict, calib_score_dict=calib_dict, hp_lambda=hp_lambda, rolling_window=rolling_window)
        for score_name in hcfm_mech_test:
            if score_name in hcfm_mech_calib:
                add_dataspace_score(
                    metrics_rows, scores_z, scores_raw, score_stats,
                    method, score_name, hcfm_mech_calib[score_name], hcfm_mech_test[score_name],
                    y, test_starts, point_labels, dataspace_cfg.window, rare_normal_mask,
                )

score_calibration_stats = pd.DataFrame(score_stats)
add_config_metadata(score_calibration_stats, dataspace_cfg).to_csv(dataspace_cfg.output_dir / "score_calibration_stats.csv", index=False)

metrics_rows = finalize_parallel_metric_rows(metrics_rows, metric_jobs, dataspace_cfg)
metrics = pd.DataFrame(metrics_rows).sort_values(["Model", "Score"])
metrics = add_config_metadata(metrics, dataspace_cfg)
if "training_timing_by_method" in globals():
    metrics["train_time_sec"] = metrics["Model"].map(lambda m: training_timing_by_method.get(m, {}).get("train_time_sec", np.nan))
    metrics["steps_per_sec"] = metrics["Model"].map(lambda m: training_timing_by_method.get(m, {}).get("steps_per_sec", np.nan))
metrics.insert(0, "calibration_fraction_actual", calibration_fraction_actual)
metrics.insert(0, "calibration_windows", int(calib_size))
if "seed" not in metrics.columns:
    metrics.insert(0, "seed", dataspace_cfg.seed)
if "dataset_id" not in metrics.columns:
    metrics.insert(0, "dataset_id", dataspace_cfg.dataset_id)
def unique_existing_columns(cols, df):
    return [col for col in dict.fromkeys(cols) if col in df.columns]

metrics.to_csv(dataspace_cfg.output_dir / "dataspace_cnn_metrics.csv", index=False)
print(f"Saved full metrics with {len(metrics)} rows to {dataspace_cfg.output_dir / 'dataspace_cnn_metrics.csv'}")

vus_enabled = bool(getattr(dataspace_cfg, "compute_vus_metrics", False))
vus_package_available = (globals().get("_VUS_GET_METRICS", None) is not None or try_import_vus_metrics() is not None) if vus_enabled else False
vus_window_print = get_vus_sliding_window(dataspace_cfg, point_labels) if vus_enabled else None
print("VUS enabled:", vus_enabled)
print("VUS sliding window:", vus_window_print)
print("VUS package available:", bool(vus_package_available))
if vus_enabled and not vus_package_available:
    print("VUS import/call error:", repr(globals().get("_VUS_IMPORT_ERROR", None)))

vus_summary_cols = [
    "Model", "Score", "vus_sliding_window",
    "vus_pr_point_mean", "vus_roc_point_mean",
    "vus_pr_point_max", "vus_roc_point_max",
]
vus_available_cols = unique_existing_columns(vus_summary_cols, metrics)
if getattr(dataspace_cfg, "compute_vus_metrics", False) and vus_available_cols:
    vus_summary = metrics[vus_available_cols].copy().rename(columns={"Model": "model", "Score": "score_name"})
    vus_sort_col = "vus_pr_point_mean" if "vus_pr_point_mean" in vus_summary.columns else ("vus_pr_point_max" if "vus_pr_point_max" in vus_summary.columns else None)
    if vus_sort_col is not None:
        vus_summary = vus_summary.sort_values(vus_sort_col, ascending=False)
    vus_summary.to_csv(dataspace_cfg.output_dir / "vus_metrics_summary.csv", index=False)
    print(f"Saved VUS metrics summary to {dataspace_cfg.output_dir / 'vus_metrics_summary.csv'}")
    print("Top 20 VUS rows")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        display(vus_summary.head(5))
    if "vus_pr_point_mean" in vus_summary.columns:
        print("Top VUS-PR point-mean rows")
        with pd.option_context("display.max_columns", None, "display.width", 180):
            display(vus_summary.sort_values("vus_pr_point_mean", ascending=False).head(5))
    if "vus_pr_point_max" in vus_summary.columns:
        print("Top VUS-PR point-max rows")
        with pd.option_context("display.max_columns", None, "display.width", 180):
            display(vus_summary.sort_values("vus_pr_point_max", ascending=False).head(5))
likelihood_style_score = metrics["Score"].astype(str)
likelihood_prefixes = (
    "base_plus_compression_div_mean_",
    "base_minus_compression_div_mean_",
    "base_plus_full_div_mean_",
    "base_minus_full_div_mean_",
    "compression_div_mean_scaled_",
    "full_div_mean_scaled_",
    "compression_div_trapz_",
    "residual_div_trapz_",
    "cnf_base_plus_compression_trapz_",
    "cnf_base_minus_compression_trapz_",
    "cnf_base_plus_full_trapz_",
    "cnf_base_minus_full_trapz_",
)
core_exact_mask = likelihood_style_score.apply(lambda s: str(s).startswith(likelihood_prefixes))
smoothed_exact_masks = {
    "hp_trend": likelihood_style_score.apply(lambda s: str(s).startswith(tuple(f"hp_trend_{p}" for p in likelihood_prefixes))),
    "rollmean": likelihood_style_score.apply(lambda s: str(s).startswith(tuple(f"rollmean_{p}" for p in likelihood_prefixes))),
    "rollmax": likelihood_style_score.apply(lambda s: str(s).startswith(tuple(f"rollmax_{p}" for p in likelihood_prefixes))),
}
new_metric_rows = metrics[core_exact_mask | np.logical_or.reduce(list(smoothed_exact_masks.values()))].copy()
hundman_cols_present = [col for col in ["hundman_window_precision", "hundman_window_recall", "hundman_window_f1"] if col in metrics.columns]
print("Hundman columns present:", hundman_cols_present)
print(
    "Likelihood-style mean/trapz rows:",
    f"core={int(core_exact_mask.sum())}",
    ", ".join(f"{name}={int(mask.sum())}" for name, mask in smoothed_exact_masks.items()),
    f"total={int(len(new_metric_rows))}",
)
new_metric_view_cols = [
    "Model", "Score", "window_AUROC", "window_AUPRC", "FP@95R_Normal",
    "point_AUROC", "point_AUPRC", "point_best_F1",
    "vus_pr_point_mean", "vus_roc_point_mean", "vus_pr_point_max", "vus_roc_point_max",
    "hundman_window_precision", "hundman_window_recall", "hundman_window_f1", "hundman_window_f1_mean_plus_2sd", "hundman_window_f1_mean_plus_3sd", "hundman_point_mean_f1", "hundman_point_max_f1",
]
new_metric_view_cols = unique_existing_columns(new_metric_view_cols, metrics)
new_metrics_view = new_metric_rows[new_metric_view_cols].copy() if len(new_metric_rows) else pd.DataFrame(columns=new_metric_view_cols)
if len(new_metrics_view):
    exact_sort_candidates = [
        "hundman_window_f1_mean_plus_2sd",
        "point_AUPRC",
        "window_AUPRC",
        "point_best_F1",
    ]
    exact_sort_col = next((col for col in exact_sort_candidates if col in new_metrics_view.columns), None)
    if exact_sort_col is not None:
        new_metrics_view = new_metrics_view.sort_values(exact_sort_col, ascending=False)
    print("Top 5 likelihood-style mean/trapz metric rows")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        display(new_metrics_view.head(5))
if dataspace_cfg.compute_hundman_metrics:
    hundman_cols = [
        "Model", "Score",
        "hundman_window_precision", "hundman_window_recall", "hundman_window_f1", "hundman_window_f1_mean_plus_2sd", "hundman_window_f1_mean_plus_3sd", "hundman_point_mean_f1", "hundman_point_max_f1",
        "hundman_window_tp", "hundman_window_fp", "hundman_window_fn", "hundman_window_matched_pred_events",
        "hundman_window_num_true_events", "hundman_window_num_pred_events",
        "hundman_window_threshold", "hundman_window_threshold_quantile",
        "hundman_window_precision_mean_plus_2sd", "hundman_window_recall_mean_plus_2sd", "hundman_window_f1_mean_plus_2sd",
        "hundman_point_mean_precision_mean_plus_2sd", "hundman_point_mean_recall_mean_plus_2sd", "hundman_point_mean_f1_mean_plus_2sd",
        "hundman_point_max_precision_mean_plus_2sd", "hundman_point_max_recall_mean_plus_2sd", "hundman_point_max_f1_mean_plus_2sd",
        "hundman_window_tp_mean_plus_2sd", "hundman_window_fp_mean_plus_2sd", "hundman_window_fn_mean_plus_2sd", "hundman_window_matched_pred_events_mean_plus_2sd",
        "hundman_window_threshold_mean_plus_2sd", "hundman_window_threshold_k_mean_plus_2sd",
        "hundman_window_precision_mean_plus_3sd", "hundman_window_recall_mean_plus_3sd", "hundman_window_f1_mean_plus_3sd",
        "hundman_point_mean_precision_mean_plus_3sd", "hundman_point_mean_recall_mean_plus_3sd", "hundman_point_mean_f1_mean_plus_3sd",
        "hundman_point_max_precision_mean_plus_3sd", "hundman_point_max_recall_mean_plus_3sd", "hundman_point_max_f1_mean_plus_3sd",
        "hundman_window_tp_mean_plus_3sd", "hundman_window_fp_mean_plus_3sd", "hundman_window_fn_mean_plus_3sd", "hundman_window_matched_pred_events_mean_plus_3sd",
        "hundman_window_threshold_mean_plus_3sd", "hundman_window_threshold_k_mean_plus_3sd",
        "hundman_window_sequence", "hundman_point_mean_sequence", "hundman_point_max_sequence",
    ]
    hundman_available = unique_existing_columns(hundman_cols, metrics)
    hundman_summary = metrics[hundman_available].copy()
    hundman_summary = hundman_summary.rename(columns={"Model": "model", "Score": "score_name"})
    hundman_summary.to_csv(dataspace_cfg.output_dir / "hundman_metrics_summary.csv", index=False)
    print(f"Saved Hundman/contextual metrics to {dataspace_cfg.output_dir / 'hundman_metrics_summary.csv'}")
    hundman_f1_display_cols = [
        "model", "score_name",
        "hundman_window_f1",
        "hundman_window_f1_mean_plus_2sd",
        "hundman_window_f1_mean_plus_3sd",
        "hundman_point_mean_f1",
        "hundman_point_mean_f1_mean_plus_2sd",
        "hundman_point_mean_f1_mean_plus_3sd",
        "hundman_point_max_f1",
        "hundman_point_max_f1_mean_plus_2sd",
        "hundman_point_max_f1_mean_plus_3sd",
    ]
    missing_hundman_f1_cols = [col for col in hundman_f1_display_cols if col not in hundman_summary.columns]
    if missing_hundman_f1_cols:
        print("Missing Hundman F1 display columns:", missing_hundman_f1_cols)
    hundman_f1_display_cols = unique_existing_columns(hundman_f1_display_cols, hundman_summary)
    hundman_f1_display = hundman_summary[hundman_f1_display_cols].copy()

    def display_top_hundman_f1(title, primary_col, extra_cols):
        if primary_col not in hundman_f1_display.columns:
            print(f"Skipping {title}: missing {primary_col}")
            return
        cols = unique_existing_columns(["model", "score_name", primary_col] + extra_cols, hundman_f1_display)
        top = hundman_f1_display[cols].sort_values(primary_col, ascending=False).head(5)
        print(title)
        with pd.option_context("display.max_columns", None, "display.width", 200):
            display(top)

    display_top_hundman_f1(
        "Top 5 Hundman/window F1 scores",
        "hundman_window_f1",
        ["hundman_window_f1_mean_plus_2sd", "hundman_window_f1_mean_plus_3sd"],
    )
    display_top_hundman_f1(
        "Top 5 Hundman/point-mean F1 scores",
        "hundman_point_mean_f1",
        ["hundman_point_mean_f1_mean_plus_2sd", "hundman_point_mean_f1_mean_plus_3sd"],
    )
    display_top_hundman_f1(
        "Top 5 Hundman/point-max F1 scores",
        "hundman_point_max_f1",
        ["hundman_point_max_f1_mean_plus_2sd", "hundman_point_max_f1_mean_plus_3sd"],
    )

    hundman_display_cols = [
        "model", "score_name",
        "hundman_window_precision", "hundman_window_recall", "hundman_window_f1",
        "hundman_window_precision_mean_plus_2sd", "hundman_window_recall_mean_plus_2sd", "hundman_window_f1_mean_plus_2sd",
        "hundman_window_precision_mean_plus_3sd", "hundman_window_recall_mean_plus_3sd", "hundman_window_f1_mean_plus_3sd",
        "hundman_window_tp", "hundman_window_fp", "hundman_window_fn",
    ]
    hundman_display_cols = unique_existing_columns(hundman_display_cols, hundman_summary)
    hundman_display = hundman_summary[hundman_display_cols].copy()
    hundman_event_sort_col = "hundman_window_f1_mean_plus_2sd" if "hundman_window_f1_mean_plus_2sd" in hundman_display.columns else "hundman_window_f1"
    if hundman_event_sort_col in hundman_display.columns:
        hundman_display = hundman_display.sort_values(hundman_event_sort_col, ascending=False).head(5)
    else:
        hundman_display = hundman_display.head(5)
    print("Top 5 Hundman/window event details")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        display(hundman_display)
if globals().get("VERBOSE_TABLES", False):
    display(metrics.head(5))


# %% notebook cell 44
# ============================================================
# Compact result summaries
# ============================================================

VERBOSE_TABLES = False


def is_raw_score(score_name):
    """Return True if score is an original/raw score."""
    s = str(score_name)
    return not (
        s.startswith("hp_trend_")
        or s.startswith("hp_cycle_")
        or s.startswith("rollmean_")
        or s.startswith("rollmax_")
        or s.startswith("hcfm_")
    )


def is_generic_score(score_name):
    """Return True if score is a generic post-processed score."""
    s = str(score_name)
    return (
        s.startswith("hp_trend_")
        or s.startswith("hp_cycle_")
        or s.startswith("rollmean_")
        or s.startswith("rollmax_")
    )


def is_hcfm_mechanism_score(score_name):
    """Return True if score is an HCFM-specific mechanism/fusion score."""
    return str(score_name).startswith("hcfm_")


def get_metric_columns(results_df):
    """Choose available metric columns safely."""
    point_metric = (
        "point_AUPRC_mean"
        if "point_AUPRC_mean" in results_df.columns
        else "point_AUPRC"
        if "point_AUPRC" in results_df.columns
        else "point_AUPRC_max"
        if "point_AUPRC_max" in results_df.columns
        else None
    )
    point_max_metric = "point_AUPRC_max" if "point_AUPRC_max" in results_df.columns else None
    auroc_metric = (
        "point_AUROC_mean"
        if "point_AUROC_mean" in results_df.columns
        else "point_AUROC"
        if "point_AUROC" in results_df.columns
        else "point_AUROC_max"
        if "point_AUROC_max" in results_df.columns
        else None
    )
    fp_metric = "FP@95R Normal" if "FP@95R Normal" in results_df.columns else None
    return point_metric, point_max_metric, auroc_metric, fp_metric


def select_display_columns(df):
    """Select compact display columns that exist in df."""
    candidate_cols = [
        "method",
        "score",
        "window_AUPRC",
        "window_AUROC",
        "point_AUPRC_mean",
        "point_AUPRC_max",
        "point_AUPRC",
        "point_AUROC_mean",
        "point_AUROC_max",
        "point_AUROC",
        "point_best_F1_mean",
        "point_best_F1_max",
        "point_best_F1",
        "FP@95R Normal",
    ]
    return [c for c in candidate_cols if c in df.columns]


def best_rows_by_method(df, metric):
    """Return the best row per method according to the given metric."""
    if df.empty or metric is None or metric not in df.columns:
        return df
    valid = df.dropna(subset=[metric])
    if valid.empty:
        return valid
    idx = valid.groupby("method")[metric].idxmax()
    return valid.loc[idx].sort_values(metric, ascending=False)


results_df = metrics.rename(columns={"Model": "method", "Score": "score"}).copy()
results_df = add_config_metadata(results_df, dataspace_cfg)
results_df.to_csv(dataspace_cfg.output_dir / "dataspace_cnn_results_with_derived.csv", index=False)

point_metric, point_max_metric, auroc_metric, fp_metric = get_metric_columns(results_df)
if point_metric is None:
    raise ValueError("No point AUPRC metric column found in results_df.")

# ------------------------------------------------------------
# Table 1: Best raw score per method
# ------------------------------------------------------------
raw_df = results_df[results_df["score"].apply(is_raw_score)].copy()
raw_best = best_rows_by_method(raw_df, point_metric)

print("Table 1: Best raw score per method")
display(raw_best.head(5)[select_display_columns(raw_best)])

# ------------------------------------------------------------
# Table 2: Best fair generic score per method
# Raw + generic HP/rolling scores. Excludes HCFM-only hcfm_*.
# ------------------------------------------------------------
generic_fair_df = results_df[
    results_df["score"].apply(lambda s: is_raw_score(s) or is_generic_score(s))
].copy()
generic_best = best_rows_by_method(generic_fair_df, point_metric)

print("Table 2: Best fair generic score per method")
display(generic_best.head(5)[select_display_columns(generic_best)])

# ------------------------------------------------------------
# Table 3: Top HCFM raw/component/mechanism scores
# Includes raw HCFM component scores and hcfm_* scores.
# Excludes generic hp_trend_/hp_cycle_/rollmean_/rollmax_ rows unless hcfm_*.
# ------------------------------------------------------------
hcfm_df = results_df[results_df["method"].str.contains("HCFM", case=False, na=False)].copy()
hcfm_mechanism_df = hcfm_df[
    hcfm_df["score"].apply(lambda s: is_raw_score(s) or is_hcfm_mechanism_score(s))
].copy()

print("Table 3: Top HCFM raw/component/mechanism scores")
display(
    hcfm_mechanism_df
    .sort_values(point_metric, ascending=False)
    .head(5)[select_display_columns(hcfm_mechanism_df)]
)


# ------------------------------------------------------------
# Fixed HCFM component diagnostics
# ------------------------------------------------------------
fixed_hcfm_scores = [
    "base_energy_x",
    "transport_energy_x",
    "hp_trend_transport_energy_x",
    "rollmean_transport_energy_x",
    "rollmax_transport_energy_x",
    "residual_energy_x",
    "compression_energy_x",
    "compression_div_abs_x",
    "compression_div_signed_x",
    "compression_div_pos_x",
    "compression_div_neg_x",
    "compression_div_neg_abs_x",
    "compression_div_centered_abs_x",
    "compression_div_centered_sq_x",
    "compression_div_centered_neg_abs_x",
    "compression_div_centered_neg_sq_x",
    "rollmax_compression_div_centered_sq_x",
    "hp_trend_compression_div_centered_sq_x",
    "rollmean_compression_div_abs_x",
    "rollmean_compression_div_signed_x",
    "rollmax_compression_div_abs_x",
    "hp_trend_compression_div_abs_x",
    "hp_trend_compression_div_signed_x",
    "hcfm_mul_base_div_abs_z",
    "hcfm_gated_base_plus_div_abs_z",
    "hcfm_gated_div_abs_plus_base_z",
    "hcfm_add_base_signed_div_z",
    "hcfm_gated_base_plus_signed_div_z",
]
hcfm_fixed_diag = results_df[
    results_df["method"].str.contains("HCFM", case=False, na=False)
    & results_df["score"].isin(fixed_hcfm_scores)
].copy()
print("Fixed HCFM component diagnostics")
if hcfm_fixed_diag.empty:
    print("No fixed HCFM diagnostic rows found.")
else:
    display(hcfm_fixed_diag.head(5)[select_display_columns(hcfm_fixed_diag)])


# ------------------------------------------------------------
# Fixed smoothing sanity diagnostics
# ------------------------------------------------------------
fixed_sanity_pairs = [
    ("Data HCFM", "compression_div_abs_x"),
    ("Data HCFM", "compression_div_signed_x"),
    ("Data HCFM", "compression_div_pos_x"),
    ("Data HCFM", "compression_div_neg_x"),
    ("Data HCFM", "compression_div_neg_abs_x"),
    ("Data HCFM", "compression_div_centered_abs_x"),
    ("Data HCFM", "compression_div_centered_sq_x"),
    ("Data HCFM", "compression_div_centered_neg_abs_x"),
    ("Data HCFM", "compression_div_centered_neg_sq_x"),
    ("Data HCFM", "rollmax_compression_div_centered_sq_x"),
    ("Data HCFM", "hp_trend_compression_div_centered_sq_x"),
    ("Data HCFM", "rollmean_compression_div_abs_x"),
    ("Data HCFM", "rollmax_compression_div_abs_x"),
    ("Data HCFM", "hp_trend_compression_div_abs_x"),
    ("Data HCFM", "residual_energy_x"),
    ("Data FDM-lite", "hp_trend_base_energy_x"),
    ("Vanilla Data FM", "hp_trend_base_energy_x"),
]
fixed_sanity_df = results_df[
    results_df.apply(lambda row: (row["method"], row["score"]) in fixed_sanity_pairs, axis=1)
].copy()
print("Fixed smoothing sanity diagnostics")
if fixed_sanity_df.empty:
    print("No fixed smoothing diagnostic rows found.")
else:
    display(fixed_sanity_df.head(5)[select_display_columns(fixed_sanity_df)])

# ------------------------------------------------------------
# Table 4: Overall top scores by primary point metric
# ------------------------------------------------------------
print(f"Table 4: Overall top scores by {point_metric}")
display(
    results_df
    .sort_values(point_metric, ascending=False)
    .head(5)[select_display_columns(results_df)]
)

# ------------------------------------------------------------
# Table 5: Overall top scores by window_AUPRC
# ------------------------------------------------------------
if "window_AUPRC" in results_df.columns:
    print("Table 5: Overall top scores by window_AUPRC")
    display(
        results_df
        .sort_values("window_AUPRC", ascending=False)
        .head(5)[select_display_columns(results_df)]
    )

# ------------------------------------------------------------
# Table 6: Metric winners
# ------------------------------------------------------------
winner_metrics = [
    c
    for c in [
        "window_AUPRC",
        "point_AUPRC_mean",
        "point_AUPRC_max",
        "point_AUPRC",
        "point_AUROC_mean",
        "point_AUROC_max",
        "point_AUROC",
    ]
    if c in results_df.columns
]

winner_rows = []
for metric in winner_metrics:
    if results_df[metric].dropna().empty:
        continue
    best_idx = results_df[metric].idxmax()
    row = results_df.loc[best_idx]
    winner_rows.append({
        "metric": metric,
        "method": row["method"],
        "score": row["score"],
        "value": row[metric],
    })

winner_df = pd.DataFrame(winner_rows)

print("Table 6: Metric winners")
display(winner_df.head(5))

# ------------------------------------------------------------
# Save compact summaries
# ------------------------------------------------------------
add_config_metadata(raw_best, dataspace_cfg).to_csv(dataspace_cfg.output_dir / "raw_best_by_method.csv", index=False)
add_config_metadata(generic_best, dataspace_cfg).to_csv(dataspace_cfg.output_dir / "generic_best_by_method.csv", index=False)
add_config_metadata(hcfm_mechanism_df, dataspace_cfg).to_csv(dataspace_cfg.output_dir / "hcfm_mechanism_scores.csv", index=False)
winner_df_to_save = add_config_metadata(winner_df.copy(), dataspace_cfg)
winner_df_to_save.to_csv(dataspace_cfg.output_dir / "metric_winners.csv", index=False)

# ------------------------------------------------------------
# Optional verbose outputs
# ------------------------------------------------------------
if VERBOSE_TABLES:
    print("Verbose full results_df")
    display(results_df.head(5))

    print("Verbose all raw scores")
    display(raw_df.head(5)[select_display_columns(raw_df)])

    print("Verbose all generic fair scores")
    display(generic_fair_df.head(5)[select_display_columns(generic_fair_df)])

    print("Verbose all HCFM rows")
    display(hcfm_df.head(5)[select_display_columns(hcfm_df)])


# %% notebook cell 45
core_main_scores = [
    ("AE", "reconstruction_mse_x"),
    ("Raw Gaussian", "mahalanobis_x"),
    ("Data HCFM", "base_energy_x"),
    ("Data HCFM", "transport_energy_x"),
    ("Data HCFM", "hp_trend_transport_energy_x"),
    ("Data HCFM", "rollmean_transport_energy_x"),
    ("Data HCFM", "rollmax_transport_energy_x"),
    ("Data HCFM", "compression_energy_x"),
    ("Data HCFM", "residual_energy_x"),
    ("Data HCFM", "compression_div_signed_x"),
    ("Data HCFM", "compression_div_neg_sq_x"),
    ("Data HCFM", "compression_div_centered_abs_x"),
    ("Data HCFM", "compression_div_centered_sq_x"),
    ("Data HCFM", "hp_trend_compression_div_neg_sq_x"),
    ("Data HCFM", "rollmax_compression_div_neg_sq_x"),
    ("Data HCFM", "base_plus_compression_div_mean_x"),
    ("Data HCFM", "base_minus_compression_div_mean_x"),
    ("Data HCFM", "base_plus_full_div_mean_x"),
    ("Data HCFM", "base_minus_full_div_mean_x"),
    ("Data HCFM", "cnf_base_plus_compression_trapz_x"),
    ("Data HCFM", "cnf_base_minus_compression_trapz_x"),
    ("Data HCFM", "rollmax_compression_div_centered_sq_x"),
    ("Data HCFM", "hp_trend_compression_div_centered_sq_x"),
]
likelihood_main_scores = [("Data HCFM", "cnf_proxy_base_plus_compression_div_x"), ("Data HCFM", "cnf_proxy_base_plus_full_div_x")]
if dataspace_cfg.trapz_cnf_include_residual:
    likelihood_main_scores.extend([
        ("Data HCFM", "cnf_base_plus_full_trapz_x"),
        ("Data HCFM", "cnf_base_minus_full_trapz_x"),
    ])
debug_main_scores = [("Data HCFM", "residual_div_signed_x"), ("Data HCFM", "transport_div_abs_x"), ("Data HCFM", "residual_div_abs_x"), ("Data HCFM", "effective_residual_div_abs_x")]
main_scores = list(core_main_scores)
if dataspace_cfg.compute_full_likelihood_proxy:
    main_scores.extend(likelihood_main_scores)
if dataspace_cfg.score_profile == "debug":
    main_scores.extend(debug_main_scores)
main_spec = pd.DataFrame(main_scores, columns=["Model", "Score"])
main_table = main_spec.merge(metrics, on=["Model", "Score"], how="left")
main_cols = ["Model", "Score", "window_AUPRC", "point_AUPRC", "point_AUPRC_mean", "point_AUPRC_max", "FP@95R Normal", "window_AUROC", "point_AUROC", "point_AUROC_mean", "point_AUROC_max"]
main_table = main_table[[col for col in main_cols if col in main_table.columns]]
add_config_metadata(main_table, dataspace_cfg).to_csv(dataspace_cfg.output_dir / "dataspace_cnn_main_table.csv", index=False)
core_keys = set(core_main_scores)
core_scores_df = metrics[metrics.apply(lambda row: (row["Model"], row["Score"]) in core_keys, axis=1)].copy()
ablation_scores_df = metrics[~metrics.apply(lambda row: (row["Model"], row["Score"]) in core_keys, axis=1)].copy()
add_config_metadata(core_scores_df, dataspace_cfg).to_csv(dataspace_cfg.output_dir / "dataspace_cnn_core_scores.csv", index=False)
add_config_metadata(ablation_scores_df, dataspace_cfg).to_csv(dataspace_cfg.output_dir / "dataspace_cnn_ablation_scores.csv", index=False)
if globals().get("VERBOSE_TABLES", False):
    display(main_table)


# %% notebook cell 54
diagnostic_cols = {
    "label": np.where(y == 1, "anomaly", np.where(rare_normal_mask, "rare normal", "normal")),
}
optional_diagnostic_scores = {
    "hcfm_base_energy_x": "Data HCFM / base_energy_x",
    "hcfm_base_plus_scaled_compression_div_x": "Data HCFM / base_plus_scaled_compression_div_x",
    "hcfm_base_minus_scaled_compression_div_x": "Data HCFM / base_minus_scaled_compression_div_x",
    "hcfm_base_plus_scaled_full_div_x": "Data HCFM / base_plus_scaled_full_div_x",
    "hcfm_base_minus_scaled_full_div_x": "Data HCFM / base_minus_scaled_full_div_x",
    "hcfm_fm_consistency_x": "Data HCFM / fm_consistency_x",
    "hcfm_compression_div_abs_x": "Data HCFM / compression_div_abs_x",
    "hcfm_compression_div_signed_x": "Data HCFM / compression_div_signed_x",
    "hcfm_compression_div_pos_x": "Data HCFM / compression_div_pos_x",
    "hcfm_compression_div_neg_x": "Data HCFM / compression_div_neg_x",
    "hcfm_compression_div_neg_abs_x": "Data HCFM / compression_div_neg_abs_x",
    "hcfm_compression_div_sq_x": "Data HCFM / compression_div_sq_x",
    "hcfm_compression_div_neg_sq_x": "Data HCFM / compression_div_neg_sq_x",
    "hcfm_compression_div_centered_abs_x": "Data HCFM / compression_div_centered_abs_x",
    "hcfm_compression_div_centered_sq_x": "Data HCFM / compression_div_centered_sq_x",
    "hcfm_compression_div_centered_neg_abs_x": "Data HCFM / compression_div_centered_neg_abs_x",
    "hcfm_compression_div_centered_neg_sq_x": "Data HCFM / compression_div_centered_neg_sq_x",
    "hcfm_transport_div_abs_x": "Data HCFM / transport_div_abs_x",
    "hcfm_residual_div_abs_x": "Data HCFM / residual_div_abs_x",
    "hcfm_residual_div_signed_x": "Data HCFM / residual_div_signed_x",
    "hcfm_effective_residual_div_abs_x": "Data HCFM / effective_residual_div_abs_x",
    "hcfm_consistency_plus_compression_div_x": "Data HCFM / consistency_plus_compression_div_x",
    "hcfm_consistency_plus_compression_signed_div_x": "Data HCFM / consistency_plus_compression_signed_div_x",
    "hcfm_consistency_plus_compression_pos_div_x": "Data HCFM / consistency_plus_compression_pos_div_x",
    "hcfm_consistency_plus_compression_neg_div_x": "Data HCFM / consistency_plus_compression_neg_div_x",
    "hcfm_consistency_plus_compression_div_abs_x": "Data HCFM / consistency_plus_compression_div_abs_x",
    "hcfm_consistency_minus_compression_div_abs_x": "Data HCFM / consistency_minus_compression_div_abs_x",
    "hcfm_consistency_plus_compression_neg_sq_x": "Data HCFM / consistency_plus_compression_neg_sq_x",
    "hcfm_consistency_plus_compression_centered_neg_sq_x": "Data HCFM / consistency_plus_compression_centered_neg_sq_x",
}
for col, key in optional_diagnostic_scores.items():
    if key in scores_z:
        diagnostic_cols[col] = scores_z[key]

diagnostic_df = pd.DataFrame(diagnostic_cols)
add_config_metadata(diagnostic_df, dataspace_cfg).to_csv(dataspace_cfg.output_dir / "data_hcfm_diagnostics.csv", index=False)

print("HCFM diagnostics columns:", [c for c in diagnostic_df.columns if c != "label"])
print("HCFM diagnostic label counts:")
print(diagnostic_df["label"].value_counts().to_string())

numeric_cols = [c for c in diagnostic_df.columns if c != "label" and pd.api.types.is_numeric_dtype(diagnostic_df[c])]
if numeric_cols:
    diag_summary = diagnostic_df.groupby("label")[numeric_cols].agg(["mean", "std", "median"])
    print("HCFM diagnostics by label written to data_hcfm_diagnostics.csv")
    if globals().get("VERBOSE_TABLES", False):
        display(diag_summary)

    div_cols = [c for c in numeric_cols if "div" in c]
    if div_cols:
        if globals().get("VERBOSE_TABLES", False):
            print("HCFM divergence diagnostics by label:")
            display(diagnostic_df.groupby("label")[div_cols].agg(["mean", "std", "median", "max"]))

    div_mean_cols = [
        ("transport", "hcfm_transport_div_abs_x"),
        ("compression", "hcfm_compression_div_abs_x"),
        ("residual", "hcfm_residual_div_abs_x"),
        ("effective_residual", "hcfm_effective_residual_div_abs_x"),
    ]
    div_means = {
        name: float(diagnostic_df[col].abs().mean())
        for name, col in div_mean_cols
        if col in diagnostic_df.columns
    }
    if div_means:
        print("HCFM mean absolute head divergences:")
        for name, value in div_means.items():
            print(f"mean |{name} div| = {value:.6g}")
    if {"transport", "compression"}.issubset(div_means):
        print(f"compression/transport divergence ratio = {div_means['compression'] / max(div_means['transport'], 1e-12):.6g}")
    if {"transport", "residual"}.issubset(div_means):
        print(f"residual/transport divergence ratio = {div_means['residual'] / max(div_means['transport'], 1e-12):.6g}")
    if {"compression", "residual"}.issubset(div_means):
        print(f"raw_residual/compression divergence ratio = {div_means['residual'] / max(div_means['compression'], 1e-12):.6g}")
    if {"compression", "effective_residual"}.issubset(div_means):
        print(f"effective_residual/compression divergence ratio = {div_means['effective_residual'] / max(div_means['compression'], 1e-12):.6g}")

if globals().get("VERBOSE_TABLES", False):
    display(diagnostic_df.groupby("label").agg(["count", "mean", "std"]))


# %% notebook cell 56
save_payload = {
    "labels": y,
    "starts": test_starts,
    "anomaly_fraction": test_frac,
    "point_labels": point_labels.astype(int),
    "rare_normal_mask": rare_normal_mask,
    "calib_idx": calib_idx,
}
for key, values in scores_raw.items():
    safe = key.replace(" / ", "__").replace(" ", "_").replace("-", "_")
    save_payload[f"{safe}__calib_raw"] = values["calib_raw"]
    save_payload[f"{safe}__test_raw"] = values["test_raw"]
    save_payload[f"{safe}__calib_z"] = values["calib_z"]
    save_payload[f"{safe}__test_z"] = values["test_z"]

score_aliases = {
    "vanilla_fm_consistency_x": "Vanilla Data FM / fm_consistency_x",
    "fdm_fm_consistency_x": "Data FDM-lite / fm_consistency_x",
    "hcfm_fm_consistency_x": "Data HCFM / fm_consistency_x",
    "hcfm_base_plus_scaled_compression_div_x": "Data HCFM / base_plus_scaled_compression_div_x",
    "hcfm_cnf_proxy_base_plus_compression_div_x": "Data HCFM / cnf_proxy_base_plus_compression_div_x",
    "hcfm_base_minus_scaled_compression_div_x": "Data HCFM / base_minus_scaled_compression_div_x",
    "hcfm_base_plus_scaled_full_div_x": "Data HCFM / base_plus_scaled_full_div_x",
    "hcfm_cnf_proxy_base_plus_full_div_x": "Data HCFM / cnf_proxy_base_plus_full_div_x",
    "hcfm_base_minus_scaled_full_div_x": "Data HCFM / base_minus_scaled_full_div_x",
    "hcfm_transport_energy_x": "Data HCFM / transport_energy_x",
    "hcfm_hp_trend_transport_energy_x": "Data HCFM / hp_trend_transport_energy_x",
    "hcfm_rollmean_transport_energy_x": "Data HCFM / rollmean_transport_energy_x",
    "hcfm_rollmax_transport_energy_x": "Data HCFM / rollmax_transport_energy_x",
    "hcfm_compression_energy_x": "Data HCFM / compression_energy_x",
    "hcfm_residual_energy_x": "Data HCFM / residual_energy_x",
    "hcfm_compression_div_abs_x": "Data HCFM / compression_div_abs_x",
    "hcfm_compression_div_signed_x": "Data HCFM / compression_div_signed_x",
    "hcfm_compression_div_pos_x": "Data HCFM / compression_div_pos_x",
    "hcfm_compression_div_neg_x": "Data HCFM / compression_div_neg_x",
    "hcfm_compression_div_neg_abs_x": "Data HCFM / compression_div_neg_abs_x",
    "hcfm_compression_div_sq_x": "Data HCFM / compression_div_sq_x",
    "hcfm_compression_div_neg_sq_x": "Data HCFM / compression_div_neg_sq_x",
    "hcfm_compression_div_centered_abs_x": "Data HCFM / compression_div_centered_abs_x",
    "hcfm_compression_div_centered_sq_x": "Data HCFM / compression_div_centered_sq_x",
    "hcfm_compression_div_centered_neg_abs_x": "Data HCFM / compression_div_centered_neg_abs_x",
    "hcfm_compression_div_centered_neg_sq_x": "Data HCFM / compression_div_centered_neg_sq_x",
    "hcfm_transport_div_abs_x": "Data HCFM / transport_div_abs_x",
    "hcfm_residual_div_abs_x": "Data HCFM / residual_div_abs_x",
    "hcfm_residual_div_signed_x": "Data HCFM / residual_div_signed_x",
    "hcfm_effective_residual_div_abs_x": "Data HCFM / effective_residual_div_abs_x",
    "hcfm_consistency_plus_compression_div_x": "Data HCFM / consistency_plus_compression_div_x",
    "hcfm_consistency_plus_compression_signed_div_x": "Data HCFM / consistency_plus_compression_signed_div_x",
    "hcfm_consistency_plus_compression_pos_div_x": "Data HCFM / consistency_plus_compression_pos_div_x",
    "hcfm_consistency_plus_compression_neg_div_x": "Data HCFM / consistency_plus_compression_neg_div_x",
    "hcfm_consistency_plus_compression_div_abs_x": "Data HCFM / consistency_plus_compression_div_abs_x",
    "hcfm_consistency_minus_compression_div_abs_x": "Data HCFM / consistency_minus_compression_div_abs_x",
    "hcfm_consistency_plus_compression_neg_sq_x": "Data HCFM / consistency_plus_compression_neg_sq_x",
    "hcfm_consistency_plus_compression_centered_neg_sq_x": "Data HCFM / consistency_plus_compression_centered_neg_sq_x",
}
for alias, key in score_aliases.items():
    if key in scores_raw:
        values = scores_raw[key]
        save_payload[f"{alias}__calib_raw"] = values["calib_raw"]
        save_payload[f"{alias}__test_raw"] = values["test_raw"]
        save_payload[f"{alias}__calib_z"] = values["calib_z"]
        save_payload[f"{alias}__test_z"] = values["test_z"]

np.savez_compressed(dataspace_cfg.output_dir / "scores_raw_and_standardized.npz", **save_payload)
print("saved data-space outputs to", dataspace_cfg.output_dir)
print("elapsed_sec", round(time.time() - t0, 2))
