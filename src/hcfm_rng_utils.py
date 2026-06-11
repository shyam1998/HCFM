from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int, deterministic: bool = True):
    """
    Seed Python, NumPy, PyTorch CPU, and PyTorch CUDA RNGs.

    deterministic=True reduces run-to-run variance. It may slow training
    and may warn if some operations do not have deterministic CUDA kernels.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)


def seed_worker(worker_id):
    """Seed NumPy and Python random inside each DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int, device: str = "cpu"):
    """Create a seeded torch.Generator. CPU generator is safest for DataLoader."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def make_eval_generator(seed: int, device):
    """Create a seeded generator for deterministic evaluation-time randomness."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def stable_int_hash(text: str, modulo: int = 1_000_000):
    """Stable deterministic hash independent of Python's randomized hash()."""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % modulo


def make_score_generator(seed: int, method_name: str, score_name: str, device):
    """Create a deterministic torch.Generator for a specific method/score pair."""
    offset = stable_int_hash(f"{method_name}:{score_name}")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 9999 + offset)
    return gen


def sample_probe_like(x, generator=None, probe_type: str = "rademacher"):
    """
    Sample a Hutchinson probe with the same shape/device/dtype as x.

    probe_type:
        "rademacher" -> values in {-1, +1}
        "gaussian"   -> standard normal
    """
    if generator is None:
        if probe_type == "rademacher":
            return torch.empty_like(x).bernoulli_(0.5).mul_(2).sub_(1)
        if probe_type == "gaussian":
            return torch.randn_like(x)
        raise ValueError(f"unsupported probe_type: {probe_type}")

    if probe_type == "rademacher":
        probe = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        probe.bernoulli_(0.5, generator=generator)
        probe.mul_(2).sub_(1)
        return probe
    if probe_type == "gaussian":
        return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    raise ValueError(f"unsupported probe_type: {probe_type}")


def add_config_metadata(df: pd.DataFrame, cfg, dataset_name=None) -> pd.DataFrame:
    """Add run/config metadata to a results dataframe."""
    df = df.copy()

    def safe_get(name, default=None):
        return getattr(cfg, name, default)

    if dataset_name is not None:
        df["dataset"] = dataset_name
    elif hasattr(cfg, "dataset_id"):
        df["dataset"] = cfg.dataset_id
    elif hasattr(cfg, "dataset_name"):
        df["dataset"] = cfg.dataset_name

    metadata = {
        "seed": safe_get("seed"),
        "train_steps": safe_get("train_steps", safe_get("hcfm_iters")),
        "base_depth": safe_get("base_depth"),
        "path_eps": safe_get("path_eps"),
        "eval_n_probe": safe_get("eval_n_probe"),
        "train_n_probe": safe_get("train_n_probe"),
        "probe_type": safe_get("probe_type", safe_get("hutchinson_probe_type")),
        "use_dedicated_fm_generator": safe_get("use_dedicated_fm_generator"),
        "fm_sampling_seed_offset": safe_get("fm_sampling_seed_offset"),
        "hcfm_transport_rank": safe_get("hcfm_transport_rank"),
        "hcfm_residual_depth": safe_get("hcfm_residual_depth"),
        "hcfm_lambda_compression_div": safe_get("hcfm_lambda_compression_div"),
        "hcfm_lambda_ortho": safe_get("hcfm_lambda_ortho"),
        "hcfm_gamma_residual": safe_get("hcfm_gamma_residual"),
        "hcfm_lambda_residual_energy": safe_get("hcfm_lambda_residual_energy"),
        "use_compile": safe_get("use_compile"),
        "deterministic": safe_get("deterministic", True),
        "fast_dev_run": safe_get("fast_dev_run", False),
    }

    for key, value in metadata.items():
        if value is not None:
            df[key] = value
    return df


def add_repro_metadata_to_results(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Backward-compatible alias for older notebook cells."""
    return add_config_metadata(df, cfg, dataset_name=getattr(cfg, "dataset_id", None))
