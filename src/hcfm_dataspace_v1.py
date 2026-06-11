from __future__ import annotations

import gc
import json
import math
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from hcfm_data_metrics_utils import (
    contiguous_true_ranges,
    load_ucr_dataset,
    metric_row,
    prepare_data,
    robust_standardize,
    score_to_numpy,
    window_scores_to_points,
)
from hcfm_rng_utils import add_config_metadata, make_score_generator, sample_probe_like, seed_everything

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is cosmetic.
    tqdm = None


METHODS = ["AE", "Raw Gaussian", "Vanilla Data FM", "Data FDM-lite", "Data HCFM"]


def _progress(iterable, *, total=None, desc=None, disable=False):
    if tqdm is None or disable:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


def make_hcfm_v1_cfg(
    dataset_id: str,
    seed: int,
    output_dir: Path,
    data_root: Path = Path("datasets/UCR"),
    train_steps: int | None = None,
    device: str | None = None,
    verbose_tables: bool = False,
    fdm_lambda_div: float = 1.0,
    hcfm_residual_depth: int | None = None,
    hcfm_lambda_compression_div: float | None = None,
    hcfm_lambda_compression_energy: float | None = None,
    hcfm_lambda_residual_energy: float | None = None,
    hcfm_lambda_ortho: float | None = None,
    hcfm_gamma_residual: float | None = None,
    hcfm_residual_div_diagnostics: bool = False,
    hcfm_component_diagnostics: bool = False,
    hcfm_use_transport: bool = True,
    hcfm_use_compression: bool = True,
    hcfm_use_residual: bool = True,
    methods: list[str] | None = None,
    score_profile: str = "core",
    plot_profile: str | None = None,
    config_profile: str = "hcfm_v1",
    compute_exact_likelihood_scores: bool = True,
    compute_vus_metrics: bool = True,
    run_full_field_hutchinson_scoring: bool = False,
    strict_v1_config: bool = True,
    run_label: str = "hcfm_v1",
    hcfm_variant: str | None = None,
    hcfm_residual_warmup_iters: int | None = None,
    hcfm_residual_ramp_iters: int | None = None,
    hcfm_freeze_residual_during_warmup: bool | None = None,
    hcfm_use_physics_residual_loss: bool | None = None,
    hcfm_lambda_physics_residual: float | None = None,
    hcfm_physics_residual_mode: str | None = None,
    hcfm_physics_residual_kappa: float | None = None,
    hcfm_physics_residual_eps: float | None = None,
    hcfm_physics_residual_detach_compression: bool | None = None,
    save_score_npz: bool = False,
):
    """Create the frozen HCFM-v1 benchmark config."""
    steps = 20000 if train_steps is None else int(train_steps)
    plot_profile = score_profile if plot_profile is None else plot_profile
    is_potential = config_profile == "potential_compression"

    def override(value, default):
        return default if value is None else value

    if is_potential:
        active_methods = methods or ["AE", "Raw Gaussian", "Data HCFM"]
        hcfm_iters = 15000 if train_steps is None else int(train_steps)
        warmup = 5000
        ramp = 5000
    else:
        active_methods = methods or list(METHODS)
        hcfm_iters = steps
        warmup = 0
        ramp = 0

    cfg = SimpleNamespace(
        experiment_name=f"{dataset_id.lower()}_{run_label}_seed{seed}",
        run_label=run_label,
        config_profile=config_profile,
        hcfm_variant=hcfm_variant or ("warmup_residual" if is_potential else "hcfm_v1"),
        strict_v1_config=False if is_potential else strict_v1_config,
        dataset_id=dataset_id,
        seed=int(seed),
        deterministic=False if is_potential else True,
        fast_dev_run=False,
        use_compile=False,
        print_every=2000,
        verbose_train_logs=False,
        print_loss_components=True,
        active_methods=active_methods,
        keep_train_tensors_on_device=True,
        zero_grad_set_to_none=True,
        use_dedicated_fm_generator=True,
        fm_sampling_seed_offset=12345,
        num_workers=0,
        use_same_seed_per_method=True,
        repro_check=False,
        verbose_tables=verbose_tables,
        save_score_npz=bool(save_score_npz),
        data_root=Path(data_root),
        output_dir=Path(output_dir),
        device=device,
        window=64,
        stride=1,
        stride_train=1,
        stride_test=1,
        channels=1,
        length=64,
        flat_dim=64,
        path_eps=1e-3,
        run_vanilla_fm=not is_potential,
        run_fdm_lite=not is_potential,
        run_scalar_hcfm=True,
        run_scalar_potential_hcfm=False,
        use_compact_latent=False,
        use_sequence_latent=False,
        run_cnf_likelihood=False,
        run_full_field_hutchinson_scoring=bool(run_full_field_hutchinson_scoring),
        time_emb_type="sinusoidal",
        time_emb_dim=32,
        time_emb_max_period=10000,
        hidden=64,
        base_depth=3,
        hcfm_residual_depth=int(override(hcfm_residual_depth, 0 if is_potential else 1)),
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=5.0,
        train_steps=steps,
        ae_iters=5000,
        vanilla_iters=steps,
        vanilla_batch_size=128,
        fdm_iters=steps,
        fdm_batch_size=128,
        hcfm_iters=hcfm_iters,
        hcfm_batch_size=256 if is_potential else 128,
        divergence_estimator="hutchinson",
        hutchinson_n_probe=1,
        train_n_probe=1,
        eval_n_probe=4,
        probe_type="rademacher",
        hutchinson_probe_type="rademacher",
        add_hp_cycle_scores=False,
        hp_lambda=1600.0,
        stopgrad_div_target=True,
        fdm_lambda_div=float(fdm_lambda_div),
        score_profile=score_profile,
        diagnostic_profile="none",
        plot_profile=plot_profile,
        hcfm_component_type="skew_transport_potential_compression_residual" if is_potential else "skew_transport_compression_residual",
        hcfm_compression_type="scalar_potential" if is_potential else "cnn_vector",
        hcfm_transport_type="low_rank_skew",
        hcfm_transport_rank=8 if is_potential else 16,
        hcfm_gamma_compression=1.0,
        hcfm_gamma_residual=float(override(hcfm_gamma_residual, 0.25 if is_potential else 1.0)),
        hcfm_lambda_compression_energy=float(override(hcfm_lambda_compression_energy, 0.0)),
        hcfm_lambda_residual_energy=float(override(hcfm_lambda_residual_energy, 5e-4 if is_potential else 1e-4)),
        hcfm_lambda_ortho=float(override(hcfm_lambda_ortho, 10.0)),
        hcfm_lambda_compression_div=float(override(hcfm_lambda_compression_div, 0.0 if is_potential else 0.0)),
        hcfm_lambda_residual_div=0.0,
        hcfm_use_physics_residual_loss=bool(override(hcfm_use_physics_residual_loss, False)),
        hcfm_lambda_physics_residual=float(override(hcfm_lambda_physics_residual, 0.0)),
        hcfm_physics_residual_mode=override(hcfm_physics_residual_mode, "ratio"),
        hcfm_physics_residual_kappa=float(override(hcfm_physics_residual_kappa, 0.50)),
        hcfm_physics_residual_eps=float(override(hcfm_physics_residual_eps, 1e-6)),
        hcfm_physics_residual_detach_compression=bool(override(hcfm_physics_residual_detach_compression, True)),
        hcfm_use_transport=bool(hcfm_use_transport),
        hcfm_use_compression=bool(hcfm_use_compression),
        hcfm_use_residual=bool(hcfm_use_residual),
        hcfm_residual_arch="shallow_cnn" if is_potential else "standard",
        hcfm_residual_hidden=64 if is_potential else 64,
        hcfm_residual_kernel_size=3,
        hcfm_residual_warmup_iters=int(override(hcfm_residual_warmup_iters, warmup)),
        hcfm_residual_ramp_iters=int(override(hcfm_residual_ramp_iters, ramp)),
        hcfm_freeze_residual_during_warmup=bool(override(hcfm_freeze_residual_during_warmup, True if is_potential else False)),
        hcfm_use_residual_mismatch_gate=False,
        hcfm_residual_gate_min=0.1,
        hcfm_residual_gate_threshold=0.5,
        hcfm_residual_gate_power=1.0,
        hcfm_use_residual_div_bound=False,
        hcfm_lambda_residual_div_bound=0.0,
        hcfm_div_bound_kappa=0.75,
        hcfm_inference_residual_scale_mode="full",
        hcfm_potential_backbone="cnn",
        hcfm_potential_norm_type="none",
        hcfm_potential_residual_scale=1.0,
        hcfm_potential_scale=1.0,
        hcfm_potential_out_init="default",
        hcfm_potential_out_init_std=1e-3,
        hcfm_use_fft_features=False,
        hcfm_fft_detach_features=True,
        hcfm_normalize_fft_features=True,
        hcfm_scale_aux_channels_by_x_stats=True,
        hcfm_use_position_embedding=True,
        hcfm_position_emb_dim=32,
        hcfm_position_proj_dim=1,
        hcfm_position_max_period=10000.0,
        hcfm_position_use_integer_positions=True,
        hcfm_time_emb_scale=1000.0,
        hcfm_use_film_time_conditioning=True,
        hcfm_combined_alpha=0.25,
        ode_steps=8 if is_potential else 4,
        ode_method="rk4",
        ode_atol=1e-4,
        ode_rtol=1e-4,
        score_batch_size=512 if is_potential else 256,
        hcfm_component_score_batch_size=64 if is_potential else 32,
        fm_consistency_K=4,
        eval_n_probe_core=4,
        eval_n_probe_full=4,
        compute_head_divergence_diagnostics=bool(hcfm_component_diagnostics),
        compute_full_likelihood_proxy=False,
        compute_residual_signed_div=False,
        compute_mechanism_fusions=False,
        compute_fm_consistency=False,
        compute_all_generic_smoothing=False,
        run_hcfm_residual_signed_div_score=False,
        compute_hundman_metrics=True,
        hundman_threshold_quantile=0.99,
        hundman_sd_thresholds=(2.0, 3.0),
        compute_exact_likelihood_scores=bool(compute_exact_likelihood_scores),
        exact_likelihood_include_residual=True,
        compute_vus_metrics=bool(compute_vus_metrics),
        vus_sliding_window=None,
        vus_use_point_scores=True,
        vus_point_projection_modes=("mean", "max"),
        calibration_fraction=0.10,
        calibration_min=1000,
        calibration_max=5000,
    )
    return finalize_dataspace_cfg(cfg)


def finalize_dataspace_cfg(cfg):
    """Validate and derive simple settings for one frozen data-space run."""
    cfg.data_root = Path(cfg.data_root)
    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.config_profile = getattr(cfg, "config_profile", "hcfm_v1")
    cfg.data_shape = (cfg.channels, cfg.length)
    cfg.train_n_probe = getattr(cfg, "train_n_probe", getattr(cfg, "hutchinson_n_probe", 1))
    cfg.eval_n_probe = getattr(cfg, "eval_n_probe", 4)
    cfg.eval_n_probe_core = getattr(cfg, "eval_n_probe_core", cfg.eval_n_probe)
    cfg.eval_n_probe_full = getattr(cfg, "eval_n_probe_full", cfg.eval_n_probe)
    cfg.probe_type = getattr(cfg, "probe_type", getattr(cfg, "hutchinson_probe_type", "rademacher"))
    cfg.hutchinson_probe_type = getattr(cfg, "hutchinson_probe_type", cfg.probe_type)
    cfg.score_profile = getattr(cfg, "score_profile", "core")
    cfg.plot_profile = getattr(cfg, "plot_profile", cfg.score_profile)
    cfg.diagnostic_profile = getattr(cfg, "diagnostic_profile", "none")
    cfg.compute_hundman_metrics = bool(getattr(cfg, "compute_hundman_metrics", True))
    cfg.hundman_threshold_quantile = float(getattr(cfg, "hundman_threshold_quantile", 0.99))
    cfg.hundman_sd_thresholds = tuple(float(x) for x in getattr(cfg, "hundman_sd_thresholds", (2.0, 3.0)))
    cfg.compute_vus_metrics = bool(getattr(cfg, "compute_vus_metrics", True))
    cfg.hcfm_variant = getattr(cfg, "hcfm_variant", "hcfm_v1")
    cfg.hcfm_use_physics_residual_loss = bool(getattr(cfg, "hcfm_use_physics_residual_loss", False))
    cfg.hcfm_lambda_physics_residual = float(getattr(cfg, "hcfm_lambda_physics_residual", 0.0))
    cfg.hcfm_physics_residual_mode = getattr(cfg, "hcfm_physics_residual_mode", "ratio")
    cfg.hcfm_physics_residual_kappa = float(getattr(cfg, "hcfm_physics_residual_kappa", 0.50))
    cfg.hcfm_physics_residual_eps = float(getattr(cfg, "hcfm_physics_residual_eps", 1e-6))
    cfg.hcfm_physics_residual_detach_compression = bool(getattr(cfg, "hcfm_physics_residual_detach_compression", True))
    cfg.hcfm_use_transport = bool(getattr(cfg, "hcfm_use_transport", True))
    cfg.hcfm_use_compression = bool(getattr(cfg, "hcfm_use_compression", True))
    cfg.hcfm_use_residual = bool(getattr(cfg, "hcfm_use_residual", True))
    cfg.vus_sliding_window = getattr(cfg, "vus_sliding_window", None)
    if cfg.vus_sliding_window is not None:
        cfg.vus_sliding_window = int(cfg.vus_sliding_window)
        assert cfg.vus_sliding_window >= 1
    cfg.vus_point_projection_modes = tuple(getattr(cfg, "vus_point_projection_modes", ("mean", "max")))
    cfg.vus_use_point_scores = bool(getattr(cfg, "vus_use_point_scores", True))
    assert cfg.window == cfg.length
    assert cfg.flat_dim == cfg.channels * cfg.length
    assert cfg.time_emb_type == "sinusoidal"
    assert cfg.divergence_estimator == "hutchinson"
    assert cfg.train_n_probe == 1
    assert cfg.eval_n_probe >= 1
    assert cfg.probe_type == "rademacher"
    if getattr(cfg, "strict_v1_config", True):
        assert cfg.fdm_lambda_div == 1.0
        assert cfg.hcfm_residual_depth == 1
    assert cfg.hcfm_residual_depth >= 0
    assert cfg.hcfm_gamma_residual >= 0.0
    assert cfg.hcfm_lambda_compression_div >= 0.0
    assert cfg.hcfm_lambda_physics_residual >= 0.0
    assert cfg.hcfm_physics_residual_mode in {"ratio", "hinge"}
    assert cfg.hcfm_physics_residual_kappa >= 0.0
    assert cfg.hcfm_physics_residual_eps > 0.0
    assert 0.0 < cfg.hundman_threshold_quantile < 1.0
    assert all(k > 0 for k in cfg.hundman_sd_thresholds)
    assert cfg.hcfm_component_type in {"skew_transport_compression_residual", "skew_transport_potential_compression_residual"}
    assert cfg.hcfm_compression_type in {"cnn_vector", "scalar_potential"}
    assert cfg.hcfm_transport_type == "low_rank_skew"
    assert all(mode in {"mean", "max"} for mode in cfg.vus_point_projection_modes)
    assert not cfg.fast_dev_run
    assert not cfg.use_compile
    return cfg


def cfg_to_json_dict(cfg):
    out = {}
    for key, value in vars(cfg).items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (np.integer, np.floating)):
            out[key] = value.item()
        elif isinstance(value, torch.device):
            out[key] = str(value)
        else:
            out[key] = value
    return out


class SinusoidalTimeEmbedding(nn.Module):
    """Fixed sinusoidal time embedding for continuous flow time t."""

    def __init__(self, dim: int, max_period: float = 10000.0, use_projection: bool = True):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / max(half, 1))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)) if use_projection else nn.Identity()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(1)
        if t.ndim == 2:
            t = t[:, 0]
        args = t.float()[:, None] * self.freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb[:, : self.dim])


class SinusoidalPositionEmbedding1D(nn.Module):
    """Fixed positional encoding with a small learned projection."""

    def __init__(self, emb_dim: int, proj_dim: int, length: int, max_period: float = 10000.0, use_integer_positions: bool = True):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.length = length
        self.use_integer_positions = use_integer_positions
        half = emb_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / max(half, 1))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(nn.Linear(emb_dim, proj_dim), nn.SiLU(), nn.Linear(proj_dim, proj_dim))

    def forward(self, batch_size: int, device, dtype):
        if self.use_integer_positions:
            pos = torch.arange(self.length, device=device, dtype=dtype)
        else:
            pos = torch.linspace(0.0, 1.0, self.length, device=device, dtype=dtype)
        args = pos[:, None] * self.freqs[None, :].to(device=device, dtype=dtype)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < self.emb_dim:
            emb = F.pad(emb, (0, self.emb_dim - emb.shape[-1]))
        emb = self.proj(emb[:, : self.emb_dim]).transpose(0, 1).unsqueeze(0)
        return emb.expand(batch_size, -1, -1)


class ResidualConvBlock1D(nn.Module):
    """Residual Conv1D block with sinusoidal time conditioning."""

    def __init__(self, hidden: int, t_dim: int, groups: int = 8):
        super().__init__()
        num_groups = max(1, min(groups, hidden))
        self.t_proj = nn.Linear(t_dim, hidden)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=hidden)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = x + self.t_proj(t_emb)[:, :, None]
        h = F.silu(self.norm1(self.conv1(h)))
        h = self.norm2(self.conv2(h))
        return x + h


class ShallowResidualVectorField1D(nn.Module):
    """Small residual correction head; intended as a low-capacity cleanup term."""

    def __init__(self, channels: int, hidden: int, t_dim: int, kernel_size: int = 3, max_period: float = 10000.0):
        super().__init__()
        padding = kernel_size // 2
        self.time = SinusoidalTimeEmbedding(t_dim, max_period=max_period, use_projection=True)
        self.t_proj = nn.Linear(t_dim, hidden)
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=kernel_size, padding=padding),
            nn.SiLU(),
            nn.Conv1d(hidden, channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.net[0](x) + self.t_proj(self.time(t))[:, :, None]
        h = self.net[1](h)
        return self.net[2](h)


class Conv1DVectorField(nn.Module):
    """CNN vector field v_theta(x_t, t) for data-space flow matching."""

    def __init__(self, in_channels: int, out_channels: int, hidden: int, depth: int, t_dim: int, max_period: float = 10000.0):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(t_dim, max_period=max_period, use_projection=True)
        self.in_proj = nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualConvBlock1D(hidden, t_dim) for _ in range(depth)])
        self.out_proj = nn.Conv1d(hidden, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time(t)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out_proj(F.silu(h))


class Conv1DScalarPotentialCompression(nn.Module):
    """Scalar potential phi([x, pos], t) with compression field grad_x phi."""

    def __init__(
        self,
        channels: int,
        length: int,
        hidden: int,
        depth: int,
        t_dim: int,
        max_period: float = 10000.0,
        use_position_embedding: bool = True,
        position_emb_dim: int = 32,
        position_proj_dim: int = 1,
        position_max_period: float = 10000.0,
        position_use_integer_positions: bool = True,
        potential_scale: float = 1.0,
    ):
        super().__init__()
        self.potential_scale = float(potential_scale)
        self.use_position_embedding = bool(use_position_embedding)
        self.time = SinusoidalTimeEmbedding(t_dim, max_period=max_period, use_projection=True)
        self.position_embedding = (
            SinusoidalPositionEmbedding1D(position_emb_dim, position_proj_dim, length, position_max_period, position_use_integer_positions)
            if self.use_position_embedding else None
        )
        in_channels = channels + (position_proj_dim if self.use_position_embedding else 0)
        self.in_proj = nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1)
        self.t_proj = nn.Linear(t_dim, hidden)
        self.blocks = nn.ModuleList([ResidualConvBlock1D(hidden, t_dim) for _ in range(depth)])
        self.out_proj = nn.Conv1d(hidden, 1, kernel_size=3, padding=1)

    def potential(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        parts = [x]
        if self.position_embedding is not None:
            parts.append(self.position_embedding(x.shape[0], x.device, x.dtype))
        h = torch.cat(parts, dim=1)
        t_emb = self.time(t)
        h = self.in_proj(h) + self.t_proj(t_emb)[:, :, None]
        for block in self.blocks:
            h = block(h, t_emb)
        phi_density = self.out_proj(F.silu(h)).flatten(1)
        return self.potential_scale * phi_density.mean(dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_req = x if x.requires_grad else x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            phi = self.potential(x_req, t)
            grad = torch.autograd.grad(phi.sum(), x_req, create_graph=True, retain_graph=True, only_inputs=True)[0]
        return grad


class LowRankSkewTransport(nn.Module):
    """Divergence-free transport field v(x,t)=A(t)x with A=UV^T-VU^T."""

    def __init__(self, channels: int, length: int, rank: int = 16, time_emb_dim: int = 64, hidden: int = 128, max_period: float = 10000.0):
        super().__init__()
        self.flat_dim = channels * length
        self.rank = rank
        self.time = SinusoidalTimeEmbedding(time_emb_dim, max_period=max_period, use_projection=True)
        self.head = nn.Sequential(
            nn.Linear(time_emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * self.flat_dim * rank),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_flat = x.flatten(1)
        params = self.head(self.time(t)).view(x.shape[0], 2, self.flat_dim, self.rank)
        u, v = params[:, 0], params[:, 1]
        x_col = x_flat.unsqueeze(-1)
        ax = torch.bmm(u, torch.bmm(v.transpose(1, 2), x_col)) - torch.bmm(v, torch.bmm(u.transpose(1, 2), x_col))
        return ax.squeeze(-1).view_as(x)


class DataHCFM(nn.Module):
    """Original structured data-space HCFM: transport + compression + residual."""

    def __init__(
        self,
        channels: int,
        length: int,
        hidden: int,
        depth: int,
        t_dim: int,
        transport_rank: int = 16,
        residual_depth: int | None = None,
        gamma_compression: float = 1.0,
        gamma_residual: float = 1.0,
        max_period: float = 10000.0,
        cfg=None,
    ):
        super().__init__()
        self.gamma_compression = gamma_compression
        self.gamma_residual = gamma_residual
        self.use_transport = bool(getattr(cfg, "hcfm_use_transport", True))
        self.use_compression = bool(getattr(cfg, "hcfm_use_compression", True))
        self.use_residual = bool(getattr(cfg, "hcfm_use_residual", True))
        residual_depth = depth if residual_depth is None else residual_depth
        self.transport = LowRankSkewTransport(channels, length, transport_rank, t_dim, max(128, hidden), max_period)
        if getattr(cfg, "hcfm_compression_type", "cnn_vector") == "scalar_potential":
            self.compression = Conv1DScalarPotentialCompression(
                channels,
                length,
                hidden,
                depth,
                t_dim,
                max_period=max_period,
                use_position_embedding=getattr(cfg, "hcfm_use_position_embedding", True),
                position_emb_dim=getattr(cfg, "hcfm_position_emb_dim", 32),
                position_proj_dim=getattr(cfg, "hcfm_position_proj_dim", 1),
                position_max_period=getattr(cfg, "hcfm_position_max_period", 10000.0),
                position_use_integer_positions=getattr(cfg, "hcfm_position_use_integer_positions", True),
                potential_scale=getattr(cfg, "hcfm_potential_scale", 1.0),
            )
        else:
            self.compression = Conv1DVectorField(channels, channels, hidden, depth, t_dim, max_period=max_period)
        if not self.use_residual:
            self.residual = nn.Identity()
        elif getattr(cfg, "hcfm_residual_arch", "standard") == "shallow_cnn":
            self.residual = ShallowResidualVectorField1D(
                channels,
                getattr(cfg, "hcfm_residual_hidden", hidden),
                t_dim,
                kernel_size=getattr(cfg, "hcfm_residual_kernel_size", 3),
                max_period=max_period,
            )
        else:
            self.residual = Conv1DVectorField(channels, channels, hidden, residual_depth, t_dim, max_period=max_period)

    def components(self, x: torch.Tensor, t: torch.Tensor):
        vt = torch.zeros_like(x) if not self.use_transport else self.transport(x, t)
        vc = torch.zeros_like(x) if not self.use_compression else self.compression(x, t)
        vr = torch.zeros_like(x) if isinstance(self.residual, nn.Identity) else self.residual(x, t)
        return vt, vc, vr

    def compose(self, vt: torch.Tensor, vc: torch.Tensor, vr: torch.Tensor, residual_multiplier: float | torch.Tensor = 1.0):
        return vt + self.gamma_compression * vc + self.gamma_residual * residual_multiplier * vr

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        vt, vc, vr = self.components(x, t)
        return self.compose(vt, vc, vr)


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
    """Sample x0, x1, t, xt, target for data-space Flow Matching."""
    idx = torch.randint(0, len(x_train), (batch_size,), device=x_train.device, generator=generator)
    x1 = x_train[idx]
    x0 = torch.randn_like(x1) if generator is None else torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    t = torch.rand(batch_size, device=x1.device, dtype=x1.dtype, generator=generator)
    t = path_eps + (1.0 - 2.0 * path_eps) * t
    view_shape = (batch_size,) + (1,) * (x1.ndim - 1)
    xt = (1.0 - t).view(view_shape) * x0 + t.view(view_shape) * x1
    return x0, x1, t, xt, x1 - x0


def data_energy(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x.flatten(1).pow(2).sum(dim=1)


def hutchinson_divergence_vector_field(model, x, t, n_probe=1, probe_type="rademacher", create_graph=True, generator=None):
    """Estimate div v(x,t) with Hutchinson probes."""
    x_req = x.requires_grad_(True) if create_graph else x.detach().clone().requires_grad_(True)
    estimates = []
    for _ in range(n_probe):
        eps = sample_probe_like(x_req, generator=generator, probe_type=probe_type)
        y = model(x_req, t)
        grad = torch.autograd.grad((y * eps).sum(), x_req, create_graph=create_graph, retain_graph=True, only_inputs=True)[0]
        estimates.append((grad * eps).flatten(1).sum(dim=1))
    return torch.stack(estimates, dim=0).mean(dim=0)


def rk4_step(field_fn, x: torch.Tensor, t: float, dt: float):
    dtype = x.dtype
    t0 = torch.full((x.shape[0],), float(t), device=x.device, dtype=dtype)
    t1 = torch.full((x.shape[0],), float(t + 0.5 * dt), device=x.device, dtype=dtype)
    t2 = torch.full((x.shape[0],), float(t + dt), device=x.device, dtype=dtype)
    k1 = field_fn(x, t0)
    k2 = field_fn(x + 0.5 * dt * k1, t1)
    k3 = field_fn(x + 0.5 * dt * k2, t1)
    k4 = field_fn(x + dt * k3, t2)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_reverse_dataspace(model, x1: torch.Tensor, steps: int = 4, method: str = "rk4", return_path: bool = False, path_eps: float = 0.0):
    """Reverse integrate x_1 to x_0 for Jacobian-free base energy scoring."""
    x = x1
    t_start = 1.0 - float(path_eps)
    t_end = float(path_eps)
    dt = (t_end - t_start) / steps
    path = [(t_start, x.detach())]

    def field_fn(state, tt):
        with torch.no_grad():
            return model(state, tt)

    for step in range(steps):
        t = t_start + step * dt
        if method == "rk4":
            x = rk4_step(field_fn, x, t, dt).detach()
        elif method == "euler":
            tt = torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
            x = (x + dt * field_fn(x, tt)).detach()
        else:
            raise ValueError(f"unsupported fixed-step method: {method}")
        path.append((t_start + (step + 1) * dt, x.detach()))
    return (x, path) if return_path else x


def scalarize_for_log(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().item()
    return float(x)


def log_train_step(method_name: str, step: int, total_steps: int, loss, cfg, components=None):
    msg = f"[{method_name}] step {step:05d}/{total_steps} loss={scalarize_for_log(loss):.6f}"
    if getattr(cfg, "print_loss_components", False) and components:
        comp = " ".join(f"{k}={scalarize_for_log(v):.4g}" for k, v in components.items() if v is not None)
        if comp:
            msg = f"{msg} {comp}"
    print(msg, flush=True)


def append_train_history(hist, step: int, loss, cfg, components=None, force_components: bool = False):
    row = {"step": step, "loss": scalarize_for_log(loss)}
    if (force_components or getattr(cfg, "print_loss_components", False)) and components:
        for key, value in components.items():
            if value is not None:
                row[key] = scalarize_for_log(value)
    hist.append(row)


def maybe_compile_model(model, cfg):
    if getattr(cfg, "use_compile", False):
        try:
            return torch.compile(model)
        except Exception as exc:
            print(f"torch.compile failed; continuing without compile: {exc}")
    return model


def make_fm_train_generator(x_train: torch.Tensor, cfg):
    if not getattr(cfg, "use_dedicated_fm_generator", True):
        return None
    gen = torch.Generator(device=x_train.device)
    gen.manual_seed(cfg.seed + cfg.fm_sampling_seed_offset)
    return gen


def hcfm_residual_multiplier_for_step(step: int, cfg) -> float:
    warmup = int(getattr(cfg, "hcfm_residual_warmup_iters", 0))
    ramp = int(getattr(cfg, "hcfm_residual_ramp_iters", 0))
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    return float(min(1.0, max(0.0, (step - warmup) / max(ramp, 1))))


def _history_df(hist, elapsed, steps):
    df = pd.DataFrame(hist) if hist else pd.DataFrame([{}])
    df["train_time_sec"] = elapsed
    df["steps_per_sec"] = steps / max(elapsed, 1e-12)
    return df


def train_dataspace_recon_ae(train_x_seq: torch.Tensor, cfg):
    model = maybe_compile_model(DataSpaceReconAE(cfg.channels, hidden=32).to(train_x_seq.device), cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    hist, start = [], time.time()
    for step in range(cfg.ae_iters):
        idx = torch.randint(0, len(train_x_seq), (cfg.vanilla_batch_size,), device=train_x_seq.device)
        xb = train_x_seq[idx]
        loss = F.mse_loss(model(xb), xb)
        if not torch.isfinite(loss):
            print(f"Non-finite AE loss at step {step}")
            break
        opt.zero_grad(set_to_none=cfg.zero_grad_set_to_none)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if step % cfg.print_every == 0 or step == cfg.ae_iters - 1:
            append_train_history(hist, step, loss, cfg)
            log_train_step("Data-space AE", step, cfg.ae_iters, loss, cfg)
    elapsed = time.time() - start
    print(f"Data-space AE training time: {elapsed:.2f} seconds ({cfg.ae_iters / max(elapsed, 1e-12):.2f} steps/sec)")
    model.eval()
    return model, _history_df(hist, elapsed, cfg.ae_iters)


@torch.no_grad()
def score_reconstruction_mse_x(model, x_all: torch.Tensor, batch_size: int):
    return torch.cat([(model(x_all[i:i + batch_size]) - x_all[i:i + batch_size]).flatten(1).pow(2).mean(dim=1).detach().cpu() for i in range(0, len(x_all), batch_size)])


def train_vanilla_data_fm(train_x_seq: torch.Tensor, cfg):
    model = maybe_compile_model(Conv1DVectorField(cfg.channels, cfg.channels, cfg.hidden, cfg.base_depth, cfg.time_emb_dim, cfg.time_emb_max_period).to(train_x_seq.device), cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.vanilla_iters), eta_min=cfg.lr * 0.1)
    hist, start, fm_gen = [], time.time(), make_fm_train_generator(train_x_seq, cfg)
    for step in range(cfg.vanilla_iters):
        _, _, t, xt, target = sample_data_fm_batch(train_x_seq, cfg.vanilla_batch_size, cfg.path_eps, generator=fm_gen)
        fm_loss = F.mse_loss(model(xt, t), target)
        if not torch.isfinite(fm_loss):
            print(f"Non-finite Vanilla FM loss at step {step}")
            break
        opt.zero_grad(set_to_none=cfg.zero_grad_set_to_none)
        fm_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        scheduler.step()
        if step % cfg.print_every == 0 or step == cfg.vanilla_iters - 1:
            append_train_history(hist, step, fm_loss, cfg)
            log_train_step("Vanilla Data FM", step, cfg.vanilla_iters, fm_loss, cfg)
    elapsed = time.time() - start
    print(f"Vanilla Data FM training time: {elapsed:.2f} seconds ({cfg.vanilla_iters / max(elapsed, 1e-12):.2f} steps/sec)")
    return model, _history_df(hist, elapsed, cfg.vanilla_iters)


def train_fdm_data(train_x_seq: torch.Tensor, cfg):
    model = maybe_compile_model(Conv1DVectorField(cfg.channels, cfg.channels, cfg.hidden, cfg.base_depth, cfg.time_emb_dim, cfg.time_emb_max_period).to(train_x_seq.device), cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.fdm_iters), eta_min=cfg.lr * 0.1)
    hist, start, fm_gen = [], time.time(), make_fm_train_generator(train_x_seq, cfg)
    for step in range(cfg.fdm_iters):
        _, _, t, xt, target = sample_data_fm_batch(train_x_seq, cfg.fdm_batch_size, cfg.path_eps, generator=fm_gen)
        xt_req = xt.detach().requires_grad_(True)
        pred = model(xt_req, t)
        fm_loss = F.mse_loss(pred, target)
        div_v = hutchinson_divergence_vector_field(model, xt_req, t, n_probe=cfg.train_n_probe, probe_type=cfg.probe_type, create_graph=True)
        div_target = -cfg.flat_dim / (1 - t.clamp(max=1 - cfg.path_eps))
        if cfg.stopgrad_div_target:
            div_target = div_target.detach()
        div_loss = (((div_v - div_target) / (div_target.abs() + 1e-6)).pow(2)).mean()
        total_loss = fm_loss + cfg.fdm_lambda_div * div_loss
        if not torch.isfinite(total_loss):
            print(f"Non-finite FDM loss at step {step}")
            break
        opt.zero_grad(set_to_none=cfg.zero_grad_set_to_none)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        scheduler.step()
        if step % cfg.print_every == 0 or step == cfg.fdm_iters - 1:
            append_train_history(hist, step, total_loss, cfg)
            log_train_step("Data FDM-lite", step, cfg.fdm_iters, total_loss, cfg)
    elapsed = time.time() - start
    print(f"Data FDM-lite training time: {elapsed:.2f} seconds ({cfg.fdm_iters / max(elapsed, 1e-12):.2f} steps/sec)")
    return model, _history_df(hist, elapsed, cfg.fdm_iters)


def train_data_hcfm(train_x_seq: torch.Tensor, cfg):
    model = maybe_compile_model(DataHCFM(
        cfg.channels, cfg.length, cfg.hidden, cfg.base_depth, cfg.time_emb_dim,
        cfg.hcfm_transport_rank, cfg.hcfm_residual_depth, cfg.hcfm_gamma_compression,
        cfg.hcfm_gamma_residual, cfg.time_emb_max_period, cfg=cfg,
    ).to(train_x_seq.device), cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.hcfm_iters), eta_min=cfg.lr * 0.1)
    hist, start, fm_gen = [], time.time(), make_fm_train_generator(train_x_seq, cfg)
    for step in range(cfg.hcfm_iters):
        _, _, t, xt, target = sample_data_fm_batch(train_x_seq, cfg.hcfm_batch_size, cfg.path_eps, generator=fm_gen)
        xt_req = xt.detach().requires_grad_(True)
        vt, vc, vr = model.components(xt_req, t)
        residual_multiplier = hcfm_residual_multiplier_for_step(step, cfg)
        if residual_multiplier == 0.0 and getattr(cfg, "hcfm_freeze_residual_during_warmup", False):
            vr = vr.detach() * 0.0
        v_total = model.compose(vt, vc, vr, residual_multiplier=residual_multiplier)
        fm_loss = F.mse_loss(v_total, target)
        residual_energy = vr.flatten(1).pow(2).mean()
        ortho_loss = F.cosine_similarity(vt.flatten(1), vc.flatten(1), dim=1, eps=1e-8).pow(2).mean()
        if residual_multiplier > 0:
            ortho_loss = (
                ortho_loss
                + F.cosine_similarity(vt.flatten(1), vr.flatten(1), dim=1, eps=1e-8).pow(2).mean()
                + F.cosine_similarity(vc.flatten(1), vr.flatten(1), dim=1, eps=1e-8).pow(2).mean()
            )
        need_physics_residual = bool(getattr(cfg, "hcfm_use_physics_residual_loss", False)) and cfg.hcfm_lambda_physics_residual > 0
        need_compression_div_grad = cfg.hcfm_lambda_compression_div > 0
        div_compression = None
        div_residual = None
        div_loss = fm_loss.new_tensor(0.0)
        physics_residual_loss = fm_loss.new_tensor(0.0)
        physics_residual_ratio_rms = fm_loss.new_tensor(0.0)
        residual_div_rms = fm_loss.new_tensor(0.0)
        compression_div_rms = fm_loss.new_tensor(0.0)
        compression_create_graph = need_compression_div_grad or (need_physics_residual and not cfg.hcfm_physics_residual_detach_compression)
        if need_compression_div_grad or need_physics_residual:
            div_compression = hutchinson_divergence_vector_field(
                lambda x_in, t_in: model.compression(x_in, t_in),
                xt_req,
                t,
                n_probe=cfg.train_n_probe,
                probe_type=cfg.probe_type,
                create_graph=compression_create_graph,
            )
        if need_compression_div_grad:
            div_loss = div_compression.pow(2).mean()
        if need_physics_residual:
            div_residual = hutchinson_divergence_vector_field(
                lambda x_in, t_in: model.residual(x_in, t_in),
                xt_req,
                t,
                n_probe=cfg.train_n_probe,
                probe_type=cfg.probe_type,
                create_graph=True,
            )
            compression_for_bound = div_compression.abs()
            if cfg.hcfm_physics_residual_detach_compression:
                compression_for_bound = compression_for_bound.detach()
            residual_abs = div_residual.abs()
            residual_div_rms = div_residual.pow(2).mean().sqrt()
            compression_div_rms = div_compression.detach().pow(2).mean().sqrt()
            compression_sq_mean = compression_for_bound.pow(2).mean().clamp_min(cfg.hcfm_physics_residual_eps)
            if cfg.hcfm_physics_residual_mode == "ratio":
                physics_residual_loss = residual_abs.pow(2).mean() / compression_sq_mean
            elif cfg.hcfm_physics_residual_mode == "hinge":
                physics_residual_loss = F.relu(
                    residual_abs - cfg.hcfm_physics_residual_kappa * compression_for_bound
                ).pow(2).mean()
            else:
                raise ValueError(f"unknown hcfm_physics_residual_mode: {cfg.hcfm_physics_residual_mode}")
            physics_residual_ratio_rms = residual_div_rms / compression_div_rms.clamp_min(cfg.hcfm_physics_residual_eps)
        total_loss = (
            fm_loss
            + cfg.hcfm_lambda_residual_energy * residual_energy
            + cfg.hcfm_lambda_ortho * ortho_loss
            + cfg.hcfm_lambda_compression_div * div_loss
            + cfg.hcfm_lambda_physics_residual * physics_residual_loss
        )
        if not torch.isfinite(total_loss):
            print(f"Non-finite HCFM loss at step {step}")
            break
        opt.zero_grad(set_to_none=cfg.zero_grad_set_to_none)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        scheduler.step()
        if step % cfg.print_every == 0 or step == cfg.hcfm_iters - 1:
            with torch.no_grad():
                fm_transport = F.mse_loss(vt, target)
                fm_transport_compression = F.mse_loss(vt + cfg.hcfm_gamma_compression * vc, target)
            components = {
                "fm_T": fm_transport,
                "fm_TC": fm_transport_compression,
                "fm": fm_loss,
                "res_mult": residual_multiplier,
                "residual_ramp": residual_multiplier,
                "gamma_residual_effective": cfg.hcfm_gamma_residual * residual_multiplier,
            }
            if cfg.hcfm_lambda_residual_energy > 0:
                components["res"] = residual_energy
            if cfg.hcfm_lambda_ortho > 0:
                components["ortho"] = ortho_loss
            if cfg.hcfm_lambda_compression_div > 0:
                components["div"] = div_loss
            if need_physics_residual:
                components["physics_residual_loss_raw"] = physics_residual_loss
                components["physics_residual_loss_weighted"] = cfg.hcfm_lambda_physics_residual * physics_residual_loss
                components["physics_residual_ratio_rms"] = physics_residual_ratio_rms
                components["residual_div_rms"] = residual_div_rms
                components["compression_div_rms"] = compression_div_rms
            append_train_history(hist, step, total_loss, cfg, components=components, force_components=True)
            log_train_step("Data HCFM", step, cfg.hcfm_iters, total_loss, cfg, components=components)
    elapsed = time.time() - start
    print(f"Data HCFM training time: {elapsed:.2f} seconds ({cfg.hcfm_iters / max(elapsed, 1e-12):.2f} steps/sec)")
    return model, _history_df(hist, elapsed, cfg.hcfm_iters)


@torch.no_grad()
def score_dataspace_base_energy(model, x_all: torch.Tensor, batch_size: int, cfg, desc: str = ""):
    model.eval()
    outs = []
    iterator = _progress(range(0, len(x_all), batch_size), total=int(np.ceil(len(x_all) / batch_size)), desc=desc, disable=getattr(cfg, "quiet_scoring", False))
    for i in iterator:
        x0_hat = integrate_reverse_dataspace(model, x_all[i:i + batch_size], steps=cfg.ode_steps, method=cfg.ode_method, path_eps=cfg.path_eps)
        outs.append(data_energy(x0_hat).detach().cpu())
    return torch.cat(outs)


def score_dataspace_full_divergence_like(model, x_all: torch.Tensor, batch_size: int, cfg, method_name: str, desc: str = ""):
    """Score a monolithic vector field with base energy plus path-integrated divergence."""
    model.eval()
    eval_gen = make_score_generator(cfg.seed, method_name=method_name, score_name="full_div_integral_x", device=x_all.device)
    base_out, div_integral_out = [], []
    iterator = _progress(range(0, len(x_all), batch_size), total=int(np.ceil(len(x_all) / batch_size)), desc=desc, disable=getattr(cfg, "quiet_scoring", False))
    for i in iterator:
        xb = x_all[i:i + batch_size]
        x0_hat, path_states = integrate_reverse_dataspace(model, xb, steps=cfg.ode_steps, method=cfg.ode_method, return_path=True, path_eps=cfg.path_eps)
        base_out.append(data_energy(x0_hat).detach().cpu())
        div_steps, times = [], []
        for t_value, x_t in path_states:
            t_safe = min(max(float(t_value), cfg.path_eps), 1.0 - cfg.path_eps)
            t = torch.full((x_t.shape[0],), t_safe, device=x_t.device, dtype=x_t.dtype)
            with torch.enable_grad():
                div_v = hutchinson_divergence_vector_field(
                    model,
                    x_t,
                    t,
                    n_probe=cfg.eval_n_probe,
                    probe_type=cfg.probe_type,
                    create_graph=False,
                    generator=eval_gen,
                )
            div_steps.append(div_v.detach().cpu())
            times.append(t_safe)
        if len(times) >= 2:
            order = np.argsort(np.asarray(times, dtype=np.float64))
            t_sorted = torch.tensor(np.asarray(times, dtype=np.float32)[order], device=x_all.device)
            div_sorted = torch.stack([div_steps[j] for j in order], dim=0).to(x_all.device)
            dt = (t_sorted[1:] - t_sorted[:-1]).view(-1, 1)
            div_int = (0.5 * dt * (div_sorted[:-1] + div_sorted[1:])).sum(dim=0)
        else:
            div_int = torch.zeros((xb.shape[0],), dtype=x_all.dtype, device=x_all.device)
        div_integral_out.append(div_int.detach().cpu())
    base = torch.cat(base_out)
    div_integral = torch.cat(div_integral_out)
    return {
        "base_energy": base,
        "full_div_integral": div_integral,
        "exact_like_base_plus_full_div": base + div_integral,
        "exact_like_base_minus_full_div": base - div_integral,
    }


def score_fm_consistency_data(model, x_all: torch.Tensor, batch_size: int, cfg, method_name: str, score_name: str, desc: str = ""):
    model.eval()
    eval_gen = make_score_generator(cfg.seed, method_name=method_name, score_name=score_name, device=x_all.device)
    outs = []
    iterator = _progress(range(0, len(x_all), batch_size), total=int(np.ceil(len(x_all) / batch_size)), desc=desc, disable=getattr(cfg, "quiet_scoring", False))
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
                scores.append((model(xt, t) - target).flatten(1).pow(2).mean(dim=1).detach().cpu())
        outs.append(torch.stack(scores, dim=0).mean(dim=0))
    return torch.cat(outs).numpy()


def score_data_hcfm(model: DataHCFM, x_all: torch.Tensor, batch_size: int, cfg, desc: str = ""):
    model.eval()
    eval_gen = make_score_generator(cfg.seed, method_name="Data HCFM", score_name="compression_div_abs_x", device=x_all.device)
    compute_exact_like = bool(getattr(cfg, "compute_exact_likelihood_scores", False))
    base_out, transport_energy_out, compression_energy_out, residual_energy_out = [], [], [], []
    compression_div_signed_out, compression_div_abs_out = [], []
    compression_div_integral_out, full_div_integral_out = [], []
    iterator = _progress(range(0, len(x_all), batch_size), total=int(np.ceil(len(x_all) / batch_size)), desc=desc, disable=getattr(cfg, "quiet_scoring", False))
    for i in iterator:
        xb = x_all[i:i + batch_size]
        x0_hat, path_states = integrate_reverse_dataspace(model, xb, steps=cfg.ode_steps, method=cfg.ode_method, return_path=True, path_eps=cfg.path_eps)
        base_out.append(data_energy(x0_hat).detach().cpu())
        transport_steps, compression_steps, residual_steps = [], [], []
        div_signed_steps, div_abs_steps = [], []
        div_c_steps, div_r_steps, times = [], [], []
        for t_value, x_t in path_states:
            t_safe = min(max(float(t_value), cfg.path_eps), 1.0 - cfg.path_eps)
            t = torch.full((x_t.shape[0],), t_safe, device=x_t.device, dtype=x_t.dtype)
            with torch.no_grad():
                vt, vc, vr = model.components(x_t, t)
                transport_steps.append(vt.flatten(1).pow(2).sum(dim=1).detach().cpu())
                compression_steps.append(vc.flatten(1).pow(2).sum(dim=1).detach().cpu())
                residual_steps.append(vr.flatten(1).pow(2).sum(dim=1).detach().cpu())
            with torch.enable_grad():
                if getattr(model, "use_compression", True):
                    div_c = hutchinson_divergence_vector_field(lambda x_in, t_in: model.compression(x_in, t_in), x_t, t, n_probe=cfg.eval_n_probe, probe_type=cfg.probe_type, create_graph=False, generator=eval_gen)
                else:
                    div_c = torch.zeros((x_t.shape[0],), device=x_t.device, dtype=x_t.dtype)
                if compute_exact_like:
                    if isinstance(model.residual, nn.Identity):
                        div_r = torch.zeros_like(div_c)
                    else:
                        div_r = hutchinson_divergence_vector_field(lambda x_in, t_in: model.residual(x_in, t_in), x_t, t, n_probe=cfg.eval_n_probe, probe_type=cfg.probe_type, create_graph=False, generator=eval_gen)
                else:
                    div_r = torch.zeros_like(div_c)
            div_c_steps.append(div_c.detach().cpu())
            div_r_steps.append(div_r.detach().cpu())
            times.append(t_safe)
            div_signed_steps.append(div_c.detach().cpu())
            div_abs_steps.append(div_c.abs().detach().cpu())
        transport_energy_out.append(torch.stack(transport_steps, dim=0).mean(dim=0))
        compression_energy_out.append(torch.stack(compression_steps, dim=0).mean(dim=0))
        residual_energy_out.append(torch.stack(residual_steps, dim=0).mean(dim=0))
        compression_div_signed_out.append(torch.stack(div_signed_steps, dim=0).mean(dim=0))
        compression_div_abs_out.append(torch.stack(div_abs_steps, dim=0).mean(dim=0))
        if len(times) >= 2:
            order = np.argsort(np.asarray(times, dtype=np.float64))
            t_sorted = torch.tensor(np.asarray(times, dtype=np.float32)[order], device=x_all.device)
            div_c_sorted = torch.stack([div_c_steps[j] for j in order], dim=0).to(x_all.device)
            div_r_sorted = torch.stack([div_r_steps[j] for j in order], dim=0).to(x_all.device)
            dt = (t_sorted[1:] - t_sorted[:-1]).view(-1, 1)
            comp_div_int = (0.5 * dt * (div_c_sorted[:-1] + div_c_sorted[1:])).sum(dim=0)
            full_div_int = cfg.hcfm_gamma_compression * comp_div_int + cfg.hcfm_gamma_residual * (0.5 * dt * (div_r_sorted[:-1] + div_r_sorted[1:])).sum(dim=0)
        else:
            comp_div_int = torch.zeros((xb.shape[0],), dtype=x_all.dtype, device=x_all.device)
            full_div_int = torch.zeros((xb.shape[0],), dtype=x_all.dtype, device=x_all.device)
        compression_div_integral_out.append(comp_div_int.detach().cpu())
        full_div_integral_out.append(full_div_int.detach().cpu())
    base = torch.cat(base_out)
    compression_div_signed = torch.cat(compression_div_signed_out)
    compression_div_neg_sq = -compression_div_signed.pow(2)
    compression_integral = torch.cat(compression_div_integral_out)
    full_integral = torch.cat(full_div_integral_out)
    return {
        "base_energy": base,
        "transport_energy": torch.cat(transport_energy_out),
        "compression_energy": torch.cat(compression_energy_out),
        "residual_energy": torch.cat(residual_energy_out),
        "compression_div_signed": compression_div_signed,
        "compression_div_abs": torch.cat(compression_div_abs_out),
        "compression_div_neg_sq": compression_div_neg_sq,
        "exact_like_base_plus_compression_div": base + cfg.hcfm_gamma_compression * compression_integral,
        "exact_like_base_minus_compression_div": base - cfg.hcfm_gamma_compression * compression_integral,
        "exact_like_base_plus_full_div": base + full_integral,
        "exact_like_base_minus_full_div": base - full_integral,
        "exact_like_compression_div_integral": compression_integral,
        "exact_like_full_div_integral": full_integral,
    }


def fit_raw_mahalanobis(x_train: torch.Tensor, ridge: float = 1e-3):
    x = x_train.detach().flatten(1).cpu().numpy().astype(np.float64)
    mu = x.mean(axis=0, keepdims=True)
    cov = np.cov(x, rowvar=False) + ridge * np.eye(x.shape[1], dtype=np.float64)
    return {"mean": mu, "inv_cov": np.linalg.pinv(cov), "ridge": ridge}


def score_raw_mahalanobis(model, x_all: torch.Tensor) -> torch.Tensor:
    x = x_all.detach().flatten(1).cpu().numpy().astype(np.float64)
    diff = x - model["mean"]
    return torch.tensor(np.einsum("bi,ij,bj->b", diff, model["inv_cov"], diff), dtype=torch.float32)


def standardize_dataspace_score(calib_score, test_score):
    calib_np, test_np = score_to_numpy(calib_score), score_to_numpy(test_score)
    mu, sd = float(np.mean(calib_np)), max(float(np.std(calib_np)), 1e-6)
    stats = {"calib_mean": mu, "calib_std": sd, "calib_min": float(np.min(calib_np)), "calib_max": float(np.max(calib_np))}
    return (calib_np - mu) / sd, (test_np - mu) / sd, stats


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
    multiple true positives and are removed from the FP pool.
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


def _add_contextual_overlap_prefixed(row, prefix, y_true_sequence, test_score_sequence, calib_score_sequence=None, cfg=None):
    cfg = cfg or SimpleNamespace(hundman_threshold_quantile=0.99, hundman_sd_thresholds=(2.0, 3.0))
    y_true_sequence = np.asarray(y_true_sequence).astype(bool)
    test_score_sequence = np.asarray(test_score_sequence, dtype=np.float64)
    if len(y_true_sequence) != len(test_score_sequence):
        raise ValueError(f"{prefix} contextual metric length mismatch: y={len(y_true_sequence)} score={len(test_score_sequence)}")
    pred, threshold = threshold_scores_for_hundman(
        test_score_sequence,
        calib_scores=calib_score_sequence,
        q=cfg.hundman_threshold_quantile,
    )
    hund = hundman_window_overlap_metrics(y_true_sequence, pred)
    for key, value in hund.items():
        row[f"{prefix}_{key.replace('hundman_', '')}"] = value
    row[f"{prefix}_threshold"] = float(threshold)
    row[f"{prefix}_threshold_quantile"] = float(cfg.hundman_threshold_quantile)
    for k in getattr(cfg, "hundman_sd_thresholds", (2.0, 3.0)):
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


def add_hundman_fields_to_row(row, y, test_score_z, calib_z, test_starts, point_labels, window, cfg):
    if not getattr(cfg, "compute_hundman_metrics", False):
        return row
    _add_contextual_overlap_prefixed(row, "hundman_window", y, test_score_z, calib_z, cfg)
    row["hundman_window_sequence"] = "raw_window_scores"
    if test_starts is not None and point_labels is not None and window is not None:
        point_labels_bool = np.asarray(point_labels).astype(bool)
        point_mean = window_scores_to_points(test_score_z, test_starts, len(point_labels_bool), window, mode="mean")
        point_max = window_scores_to_points(test_score_z, test_starts, len(point_labels_bool), window, mode="max")
        _add_contextual_overlap_prefixed(row, "hundman_point_mean", point_labels_bool, point_mean, calib_score_sequence=None, cfg=cfg)
        _add_contextual_overlap_prefixed(row, "hundman_point_max", point_labels_bool, point_max, calib_score_sequence=None, cfg=cfg)
        row["hundman_point_mean_sequence"] = "point_scores_mean_aggregation"
        row["hundman_point_max_sequence"] = "point_scores_max_aggregation"
    return row


_VUS_GET_METRICS = None
_VUS_IMPORT_ERROR = None
_VUS_IMPORT_WARNED = False


def try_import_vus_metrics():
    global _VUS_GET_METRICS, _VUS_IMPORT_ERROR, _VUS_IMPORT_WARNED
    if _VUS_GET_METRICS is not None:
        return _VUS_GET_METRICS
    last_error = None
    for module_name in ["vus.metrics", "metrics.metrics", "TSB_UAD.vus.metrics"]:
        try:
            module = __import__(module_name, fromlist=["get_metrics"])
            _VUS_GET_METRICS = getattr(module, "get_metrics")
            _VUS_IMPORT_ERROR = None
            return _VUS_GET_METRICS
        except Exception as exc:
            last_error = exc
    _VUS_IMPORT_ERROR = last_error
    if not _VUS_IMPORT_WARNED:
        print("VUS metrics package not found. Install VUS/TSB-AD metrics or place it on PYTHONPATH. Skipping VUS-PR.")
        if last_error is not None:
            print("Last VUS import error:", repr(last_error))
        _VUS_IMPORT_WARNED = True
    return None


def get_vus_sliding_window(cfg, point_labels):
    if getattr(cfg, "vus_sliding_window", None) is not None:
        return int(cfg.vus_sliding_window)
    lengths = [int(e - s) for s, e in contiguous_true_ranges(np.asarray(point_labels).astype(bool))]
    return max(1, int(np.median(lengths)) if lengths else int(cfg.window))


def _extract_vus_metric(obj, keys):
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                return float(obj[key])
        lowered = {str(k).lower().replace("-", "_"): v for k, v in obj.items()}
        for key in keys:
            norm = key.lower().replace("-", "_")
            if norm in lowered:
                return float(lowered[norm])
    if hasattr(obj, "to_dict"):
        return _extract_vus_metric(obj.to_dict(), keys)
    return np.nan


def compute_vus_for_point_score(point_labels, point_score, sliding_window):
    labels = np.asarray(point_labels).astype(int).reshape(-1)
    score = np.asarray(point_score, dtype=float).reshape(-1)
    n = min(len(labels), len(score))
    labels, score = labels[:n], score[:n]
    finite = np.isfinite(score)
    fill = float(np.nanmedian(score[finite])) if finite.any() else 0.0
    score = np.nan_to_num(score, nan=fill, posinf=fill, neginf=fill)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"vus_pr": np.nan, "vus_roc": np.nan}
    get_metrics = try_import_vus_metrics()
    if get_metrics is None:
        return {"vus_pr": np.nan, "vus_roc": np.nan}
    metrics_obj = None
    last_error = None
    for call in [
        lambda: get_metrics(score, labels, slidingWindow=sliding_window),
        lambda: get_metrics(score, labels, sliding_window=sliding_window),
        lambda: get_metrics(labels, score, slidingWindow=sliding_window),
        lambda: get_metrics(labels, score, sliding_window=sliding_window),
    ]:
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
            point_score = window_scores_to_points(test_score_z, test_starts, len(point_labels), window, mode=mode)
            vals = compute_vus_for_point_score(point_labels, point_score, sliding_window)
            row[f"vus_pr_point_{mode}"] = vals["vus_pr"]
            row[f"vus_roc_point_{mode}"] = vals["vus_roc"]
    return row


def add_dataspace_score(rows, scores_z, scores_raw, stats_rows, model, score_name, calib_score, test_score, y, test_starts, point_labels, window, rare_normal_mask, cfg=None):
    calib_z, test_z, stats = standardize_dataspace_score(calib_score, test_score)
    key = f"{model} / {score_name}"
    scores_z[key] = test_z
    scores_raw[key] = {"calib_raw": score_to_numpy(calib_score), "test_raw": score_to_numpy(test_score), "calib_z": calib_z, "test_z": test_z}
    stats_rows.append({"score_key": key, **stats})
    row = metric_row(model, score_name, y, test_z, test_starts, point_labels, window, rare_normal_mask)
    cfg = cfg or SimpleNamespace(compute_hundman_metrics=False, compute_vus_metrics=False)
    add_hundman_fields_to_row(row, y, test_z, calib_z, test_starts, point_labels, window, cfg)
    add_vus_fields_to_row(row, test_z, test_starts, point_labels, window, cfg)
    rows.append(row)
    return calib_z, test_z


def robust_zscore(scores, calib_scores=None, eps=1e-8, clip=10.0):
    scores = np.asarray(scores, dtype=np.float64)
    ref = scores if calib_scores is None else np.asarray(calib_scores, dtype=np.float64)
    median = np.nanmedian(ref)
    mad = np.nanmedian(np.abs(ref - median))
    z = (scores - median) / (1.4826 * mad + eps)
    return np.clip(np.nan_to_num(z, nan=0.0, posinf=clip, neginf=-clip), -clip, clip)


def positive_part(x):
    return np.maximum(np.asarray(x, dtype=np.float64), 0.0)


def stable_sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -10, 10)
    return 1.0 / (1.0 + np.exp(-x))


def hp_filter_score(scores, lamb=1600.0, component="trend", mode="raw", negate=False):
    arr = np.asarray(scores, dtype=np.float64)
    fill = np.nanmedian(arr) if np.isfinite(arr).any() else 0.0
    x = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    if negate:
        x = -x
    try:
        from statsmodels.tsa.filters.hp_filter import hpfilter
        cycle, trend = hpfilter(x, lamb=lamb)
        cycle, trend = np.asarray(cycle, dtype=np.float64), np.asarray(trend, dtype=np.float64)
    except Exception:
        trend = pd.Series(x).rolling(window=31, center=True, min_periods=1).median().to_numpy()
        cycle = x - trend
    selected = trend if component == "trend" else cycle
    if mode == "raw":
        return np.asarray(selected, dtype=np.float64)
    if mode == "positive":
        return positive_part(selected)
    if mode == "abs":
        return np.abs(selected)
    raise ValueError(f"unsupported HP mode: {mode}")


def rolling_mean_score(scores, window=5):
    return pd.Series(np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0)).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def rolling_max_score(scores, window=5):
    return pd.Series(np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0)).rolling(window=window, center=True, min_periods=1).max().to_numpy()


def _is_score_array(value):
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    return arr.ndim == 1 and arr.size > 0 and np.issubdtype(arr.dtype, np.number)


def build_generic_derived_scores(score_dict, hp_lambda=1600.0, rolling_window=5, add_hp_cycle_scores=False):
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


def build_hcfm_mechanism_scores(score_dict, calib_score_dict=None, hp_lambda=1600.0, rolling_window=5):
    derived = {}

    def raw(name):
        return np.asarray(score_dict[name], dtype=np.float64) if name in score_dict and _is_score_array(score_dict[name]) else None

    def ref(name):
        if calib_score_dict is not None and name in calib_score_dict and _is_score_array(calib_score_dict[name]):
            return np.asarray(calib_score_dict[name], dtype=np.float64)
        return None

    z = {}
    for short, name in {"base": "base_energy_x", "div": "compression_div_abs_x", "comp_energy": "compression_energy_x", "residual": "residual_energy_x", "consistency": "fm_consistency_x"}.items():
        values = raw(name)
        if values is not None:
            z[short] = robust_zscore(values, ref(name))

    for raw_name, alias in {
        "base_energy_x": "hcfm_hp_trend_base_energy_x",
        "compression_div_abs_x": "hcfm_hp_trend_compression_div_abs_x",
        "compression_energy_x": "hcfm_hp_trend_compression_energy_x",
        "residual_energy_x": "hcfm_hp_trend_residual_energy_x",
        "fm_consistency_x": "hcfm_hp_trend_fm_consistency_x",
    }.items():
        values = raw(raw_name)
        if values is not None:
            derived[alias] = hp_filter_score(values, lamb=hp_lambda, component="trend", mode="raw", negate=False)

    parts = [z[k] for k in ["base", "div", "comp_energy", "residual"] if k in z]
    if parts:
        derived["hcfm_max_mechanism_z"] = np.maximum.reduce(parts)
    if "base" in z and "div" in z:
        derived["hcfm_add_base_div_z"] = z["base"] + z["div"]
        derived["hcfm_mul_base_div_z"] = np.sqrt(positive_part(z["base"]) * positive_part(z["div"]))
        derived["hcfm_gated_base_plus_div_z"] = z["base"] + stable_sigmoid(z["base"]) * positive_part(z["div"])
        derived["hcfm_gated_div_plus_base_z"] = z["div"] + stable_sigmoid(z["div"]) * positive_part(z["base"])
    if "base" in z and "div" in z and "comp_energy" in z:
        derived["hcfm_add_base_div_comp_z"] = z["base"] + z["div"] + z["comp_energy"]
    if all(k in z for k in ["base", "div", "comp_energy", "residual"]):
        derived["hcfm_add_all_mechanisms_z"] = z["base"] + z["div"] + z["comp_energy"] + z["residual"]
    if "comp_energy" in z and "div" in z:
        derived["hcfm_mul_compenergy_div_z"] = np.sqrt(positive_part(z["comp_energy"]) * positive_part(z["div"]))
    if "base" in z and "residual" in z:
        derived["hcfm_mul_base_residual_z"] = np.sqrt(positive_part(z["base"]) * positive_part(z["residual"]))
    for raw_name, alias in {
        "compression_div_abs_x": "hcfm_rollmean_compression_div_abs_x",
        "base_energy_x": "hcfm_rollmean_base_energy_x",
        "compression_energy_x": "hcfm_rollmean_compression_energy_x",
        "residual_energy_x": "hcfm_rollmean_residual_energy_x",
    }.items():
        values = raw(raw_name)
        if values is not None:
            derived[alias] = rolling_mean_score(values, window=rolling_window)
    if "hcfm_max_mechanism_z" in derived:
        derived["hcfm_rollmean_max_mechanism_z"] = rolling_mean_score(derived["hcfm_max_mechanism_z"], window=rolling_window)
    return derived


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _method_error(run_dir: Path, method: str, exc: BaseException):
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    safe = method.lower().replace(" ", "_").replace("/", "_")
    (run_dir / f"error_{safe}.log").write_text(text, encoding="utf-8")
    print(f"[{method}] failed: {exc}")
    return {"method": method, "error": repr(exc), "traceback": text}


def _record_history(run_dir: Path, filename: str, hist, cfg):
    if isinstance(hist, pd.DataFrame):
        add_config_metadata(hist, cfg).to_csv(run_dir / filename, index=False)


def _scores_for_method(scores_raw, method):
    prefix = f"{method} / "
    return {k[len(prefix):]: v for k, v in scores_raw.items() if k.startswith(prefix)}


def _scoring_measurement_row(label: str, elapsed_sec: float, peak_memory_bytes: float, cfg):
    return {
        "scoring_label": label,
        "scoring_time_sec": float(elapsed_sec),
        "scoring_peak_memory_bytes": float(peak_memory_bytes) if peak_memory_bytes is not None else np.nan,
        "cuda_available": bool(torch.cuda.is_available()),
        "dataset_id": cfg.dataset_id,
        "seed": int(cfg.seed),
        "hcfm_variant": getattr(cfg, "hcfm_variant", None),
    }


def _measure_scoring(label: str, cfg, fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak = float(torch.cuda.max_memory_allocated())
    else:
        peak = np.nan
    elapsed = time.perf_counter() - start
    return result, _scoring_measurement_row(label, elapsed, peak, cfg)


def run_single_dataset_seed(cfg):
    """Run all frozen HCFM-v1 methods for one dataset/seed."""
    seed_everything(cfg.seed, deterministic=bool(getattr(cfg, "deterministic", True)))
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg.device = str(device)
    run_dir = cfg.output_dir
    _save_json(run_dir / "config.json", cfg_to_json_dict(cfg))

    train_x, train_y, train_starts, test_x, test_y, test_starts, test_frac, point_labels = prepare_data(cfg)
    summary = pd.DataFrame({
        "split": ["train", "test"],
        "windows": [len(train_x), len(test_x)],
        "anomaly_windows": [int(train_y.sum()), int(test_y.sum())],
        "point_count": [np.nan, len(point_labels)],
        "point_anomalies": [np.nan, int(point_labels.sum())],
    })
    add_config_metadata(summary, cfg).to_csv(run_dir / "data_summary.csv", index=False)

    train_x_seq = torch.tensor(np.transpose(train_x, (0, 2, 1)), dtype=torch.float32, device=device)
    test_x_seq = torch.tensor(np.transpose(test_x, (0, 2, 1)), dtype=torch.float32, device=device)
    calib_size = min(len(train_x_seq), max(cfg.calibration_min, min(cfg.calibration_max, int(round(cfg.calibration_fraction * len(train_x_seq))))))
    calib_rng = np.random.default_rng(cfg.seed)
    calib_idx = np.sort(calib_rng.choice(len(train_x_seq), size=calib_size, replace=False))
    calib_x_seq = train_x_seq[torch.as_tensor(calib_idx, device=train_x_seq.device)]
    np.save(run_dir / "calibration_indices.npy", calib_idx)
    _save_json(run_dir / "calibration_config.json", {
        "train_windows": int(len(train_x_seq)),
        "calibration_windows": int(calib_size),
        "calibration_fraction_actual": float(calib_size / len(train_x_seq)),
        "seed": int(cfg.seed),
    })

    method_errors, metrics_rows, scores_z, scores_raw, score_stats, scoring_perf_rows = [], [], {}, {}, [], []
    models, histories = {}, {}

    active_methods = set(getattr(cfg, "active_methods", METHODS))

    # AE and Raw Gaussian baselines.
    if "AE" in active_methods:
        try:
            seed_everything(cfg.seed, deterministic=bool(getattr(cfg, "deterministic", True)))
            models["AE"], histories["AE"] = train_dataspace_recon_ae(train_x_seq, cfg)
            _record_history(run_dir, "dataspace_ae_training_history.csv", histories["AE"], cfg)
            ae_rec_calib = score_reconstruction_mse_x(models["AE"], calib_x_seq, cfg.score_batch_size)
            ae_rec_test = score_reconstruction_mse_x(models["AE"], test_x_seq, cfg.score_batch_size)
        except Exception as exc:
            method_errors.append(_method_error(run_dir, "AE", exc))
            ae_rec_calib = ae_rec_test = None
    else:
        ae_rec_calib = ae_rec_test = None

    if "Raw Gaussian" in active_methods:
        try:
            raw_gaussian = fit_raw_mahalanobis(train_x_seq, ridge=1e-3)
            mahal_calib = score_raw_mahalanobis(raw_gaussian, calib_x_seq)
            mahal_test = score_raw_mahalanobis(raw_gaussian, test_x_seq)
        except Exception as exc:
            method_errors.append(_method_error(run_dir, "Raw Gaussian", exc))
            mahal_calib = mahal_test = None
    else:
        mahal_calib = mahal_test = None

    # Train stochastic flow methods independently with the same seed.
    for method, train_fn, hist_name in [
        ("Vanilla Data FM", train_vanilla_data_fm, "vanilla_data_fm_training_history.csv"),
        ("Data FDM-lite", train_fdm_data, "fdm_data_training_history.csv"),
        ("Data HCFM", train_data_hcfm, "data_hcfm_training_history.csv"),
    ]:
        if method not in active_methods:
            models[method] = None
            continue
        try:
            seed_everything(cfg.seed, deterministic=bool(getattr(cfg, "deterministic", True)))
            models[method], histories[method] = train_fn(train_x_seq, cfg)
            _record_history(run_dir, hist_name, histories[method], cfg)
        except Exception as exc:
            method_errors.append(_method_error(run_dir, method, exc))
            models[method] = None

    param_rows = [{"model": m, "trainable_params": count_trainable_params(model)} for m, model in models.items() if model is not None]
    param_df = add_config_metadata(pd.DataFrame(param_rows), cfg)
    param_df.to_csv(run_dir / "parameter_counts.csv", index=False)
    param_map = dict(zip(param_df.get("model", []), param_df.get("trainable_params", [])))
    for model_name, model in models.items():
        if model is not None:
            safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
            try:
                torch.save(model.state_dict(), run_dir / ("%s_state_dict.pt" % safe_name))
            except Exception as exc:
                method_errors.append(_method_error(run_dir, "%s checkpoint save" % model_name, exc))

    timing_by_method = {}
    for method, hist in histories.items():
        if isinstance(hist, pd.DataFrame) and "train_time_sec" in hist.columns and len(hist) > 0:
            timing_by_method[method] = {
                "train_time_sec": float(hist["train_time_sec"].iloc[-1]),
                "steps_per_sec": float(hist["steps_per_sec"].iloc[-1]),
            }

    y = test_y.astype(int)
    rare_normal_mask = np.zeros_like(y, dtype=bool)

    # Score vanilla first when available to define the rare-normal diagnostic.
    scored = {}
    try:
        if models.get("Vanilla Data FM") is not None:
            van_base_calib, perf = _measure_scoring(
                "Vanilla Data FM: base energy calib",
                cfg,
                lambda: score_dataspace_base_energy(models["Vanilla Data FM"], calib_x_seq, cfg.score_batch_size, cfg, "Vanilla base calib"),
            )
            scoring_perf_rows.append(perf)
            van_base_test, perf = _measure_scoring(
                "Vanilla Data FM: base energy test",
                cfg,
                lambda: score_dataspace_base_energy(models["Vanilla Data FM"], test_x_seq, cfg.score_batch_size, cfg, "Vanilla base test"),
            )
            scoring_perf_rows.append(perf)
            _, vanilla_base_for_rare, _ = standardize_dataspace_score(van_base_calib, van_base_test)
            rare_threshold = np.quantile(vanilla_base_for_rare[y == 0], 0.90) if np.any(y == 0) else np.nan
            rare_normal_mask = (y == 0) & (vanilla_base_for_rare >= rare_threshold)
            scored["Vanilla Data FM"] = {"base_energy_x": (van_base_calib, van_base_test)}
            if getattr(cfg, "run_full_field_hutchinson_scoring", False):
                van_full_calib, perf = _measure_scoring(
                    "Vanilla Data FM: full divergence calib",
                    cfg,
                    lambda: score_dataspace_full_divergence_like(models["Vanilla Data FM"], calib_x_seq, cfg.score_batch_size, cfg, "Vanilla Data FM", "Vanilla full-div calib"),
                )
                scoring_perf_rows.append(perf)
                van_full_test, perf = _measure_scoring(
                    "Vanilla Data FM: full divergence test",
                    cfg,
                    lambda: score_dataspace_full_divergence_like(models["Vanilla Data FM"], test_x_seq, cfg.score_batch_size, cfg, "Vanilla Data FM", "Vanilla full-div test"),
                )
                scoring_perf_rows.append(perf)
                scored["Vanilla Data FM"].update({
                    "full_div_integral_x": (van_full_calib["full_div_integral"], van_full_test["full_div_integral"]),
                    "exact_like_base_plus_full_div_x": (van_full_calib["exact_like_base_plus_full_div"], van_full_test["exact_like_base_plus_full_div"]),
                    "exact_like_base_minus_full_div_x": (van_full_calib["exact_like_base_minus_full_div"], van_full_test["exact_like_base_minus_full_div"]),
                })
            if getattr(cfg, "compute_fm_consistency", False):
                scored["Vanilla Data FM"]["fm_consistency_x"] = (
                    score_fm_consistency_data(models["Vanilla Data FM"], calib_x_seq, cfg.score_batch_size, cfg, "Vanilla Data FM", "fm_consistency_x", "Vanilla consistency calib"),
                    score_fm_consistency_data(models["Vanilla Data FM"], test_x_seq, cfg.score_batch_size, cfg, "Vanilla Data FM", "fm_consistency_x", "Vanilla consistency test"),
                )
    except Exception as exc:
        method_errors.append(_method_error(run_dir, "Vanilla Data FM scoring", exc))

    if ae_rec_calib is not None:
        scored["AE"] = {"reconstruction_mse_x": (ae_rec_calib, ae_rec_test)}
    if mahal_calib is not None:
        scored["Raw Gaussian"] = {"mahalanobis_x": (mahal_calib, mahal_test)}

    try:
        if models.get("Data FDM-lite") is not None:
            fdm_base_calib, perf = _measure_scoring(
                "Data FDM-lite: base energy calib",
                cfg,
                lambda: score_dataspace_base_energy(models["Data FDM-lite"], calib_x_seq, cfg.score_batch_size, cfg, "FDM base calib"),
            )
            scoring_perf_rows.append(perf)
            fdm_base_test, perf = _measure_scoring(
                "Data FDM-lite: base energy test",
                cfg,
                lambda: score_dataspace_base_energy(models["Data FDM-lite"], test_x_seq, cfg.score_batch_size, cfg, "FDM base test"),
            )
            scoring_perf_rows.append(perf)
            scored["Data FDM-lite"] = {"base_energy_x": (fdm_base_calib, fdm_base_test)}
            if getattr(cfg, "run_full_field_hutchinson_scoring", False):
                fdm_full_calib, perf = _measure_scoring(
                    "Data FDM-lite: full divergence calib",
                    cfg,
                    lambda: score_dataspace_full_divergence_like(models["Data FDM-lite"], calib_x_seq, cfg.score_batch_size, cfg, "Data FDM-lite", "FDM full-div calib"),
                )
                scoring_perf_rows.append(perf)
                fdm_full_test, perf = _measure_scoring(
                    "Data FDM-lite: full divergence test",
                    cfg,
                    lambda: score_dataspace_full_divergence_like(models["Data FDM-lite"], test_x_seq, cfg.score_batch_size, cfg, "Data FDM-lite", "FDM full-div test"),
                )
                scoring_perf_rows.append(perf)
                scored["Data FDM-lite"].update({
                    "full_div_integral_x": (fdm_full_calib["full_div_integral"], fdm_full_test["full_div_integral"]),
                    "exact_like_base_plus_full_div_x": (fdm_full_calib["exact_like_base_plus_full_div"], fdm_full_test["exact_like_base_plus_full_div"]),
                    "exact_like_base_minus_full_div_x": (fdm_full_calib["exact_like_base_minus_full_div"], fdm_full_test["exact_like_base_minus_full_div"]),
                })
            if getattr(cfg, "compute_fm_consistency", False):
                scored["Data FDM-lite"]["fm_consistency_x"] = (
                    score_fm_consistency_data(models["Data FDM-lite"], calib_x_seq, cfg.score_batch_size, cfg, "Data FDM-lite", "fm_consistency_x", "FDM consistency calib"),
                    score_fm_consistency_data(models["Data FDM-lite"], test_x_seq, cfg.score_batch_size, cfg, "Data FDM-lite", "fm_consistency_x", "FDM consistency test"),
                )
    except Exception as exc:
        method_errors.append(_method_error(run_dir, "Data FDM-lite scoring", exc))

    try:
        if models.get("Data HCFM") is not None:
            hcfm_calib, perf = _measure_scoring(
                "Data HCFM: component scoring calib",
                cfg,
                lambda: score_data_hcfm(models["Data HCFM"], calib_x_seq, cfg.hcfm_component_score_batch_size, cfg, "HCFM calib"),
            )
            scoring_perf_rows.append(perf)
            hcfm_test, perf = _measure_scoring(
                "Data HCFM: component scoring test",
                cfg,
                lambda: score_data_hcfm(models["Data HCFM"], test_x_seq, cfg.hcfm_component_score_batch_size, cfg, "HCFM test"),
            )
            scoring_perf_rows.append(perf)
            scored["Data HCFM"] = {
                "base_energy_x": (hcfm_calib["base_energy"], hcfm_test["base_energy"]),
                "transport_energy_x": (hcfm_calib["transport_energy"], hcfm_test["transport_energy"]),
                "compression_energy_x": (hcfm_calib["compression_energy"], hcfm_test["compression_energy"]),
                "residual_energy_x": (hcfm_calib["residual_energy"], hcfm_test["residual_energy"]),
                "compression_div_signed_x": (hcfm_calib["compression_div_signed"], hcfm_test["compression_div_signed"]),
                "compression_div_abs_x": (hcfm_calib["compression_div_abs"], hcfm_test["compression_div_abs"]),
                "compression_div_neg_sq_x": (hcfm_calib["compression_div_neg_sq"], hcfm_test["compression_div_neg_sq"]),
            }
            calib_div = score_to_numpy(hcfm_calib["compression_div_signed"])
            test_div = score_to_numpy(hcfm_test["compression_div_signed"])
            div_median = float(np.median(calib_div))
            div_mad = float(np.median(np.abs(calib_div - div_median)))
            div_scale = 1.4826 * div_mad + 1e-8
            calib_centered = (calib_div - div_median) / div_scale
            test_centered = (test_div - div_median) / div_scale
            scored["Data HCFM"].update({
                "compression_div_centered_abs_x": (np.abs(calib_centered), np.abs(test_centered)),
                "compression_div_centered_sq_x": (np.square(calib_centered), np.square(test_centered)),
                "compression_div_centered_neg_abs_x": (-np.abs(calib_centered), -np.abs(test_centered)),
                "compression_div_centered_neg_sq_x": (-np.square(calib_centered), -np.square(test_centered)),
            })
            if getattr(cfg, "compute_exact_likelihood_scores", False):
                for key, score_name in [
                    ("exact_like_base_plus_compression_div", "exact_like_base_plus_compression_div_x"),
                    ("exact_like_base_minus_compression_div", "exact_like_base_minus_compression_div_x"),
                    ("exact_like_base_plus_full_div", "exact_like_base_plus_full_div_x"),
                    ("exact_like_base_minus_full_div", "exact_like_base_minus_full_div_x"),
                    ("exact_like_compression_div_integral", "exact_like_compression_div_integral_x"),
                    ("exact_like_full_div_integral", "exact_like_full_div_integral_x"),
                ]:
                    scored["Data HCFM"][score_name] = (hcfm_calib[key], hcfm_test[key])
            if getattr(cfg, "compute_fm_consistency", False):
                hcfm_cons_calib = score_fm_consistency_data(models["Data HCFM"], calib_x_seq, cfg.hcfm_component_score_batch_size, cfg, "Data HCFM", "fm_consistency_x", "HCFM consistency calib")
                hcfm_cons_test = score_fm_consistency_data(models["Data HCFM"], test_x_seq, cfg.hcfm_component_score_batch_size, cfg, "Data HCFM", "fm_consistency_x", "HCFM consistency test")
                scored["Data HCFM"]["fm_consistency_x"] = (hcfm_cons_calib, hcfm_cons_test)
    except Exception as exc:
        method_errors.append(_method_error(run_dir, "Data HCFM scoring", exc))

    hcfm_calib_z = {}
    for method, method_scores in scored.items():
        for score_name, (calib_score, test_score) in method_scores.items():
            calib_z, test_z = add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, method, score_name, calib_score, test_score, y, test_starts, point_labels, cfg.window, rare_normal_mask, cfg=cfg)
            if method == "Data HCFM":
                hcfm_calib_z[score_name] = calib_z

    if getattr(cfg, "compute_mechanism_fusions", False) and "Data HCFM" in scored and {"fm_consistency_x", "compression_div_abs_x"}.issubset(hcfm_calib_z):
        hcfm_consistency_calib_z = scores_raw["Data HCFM / fm_consistency_x"]["calib_z"]
        hcfm_consistency_z = scores_raw["Data HCFM / fm_consistency_x"]["test_z"]
        hcfm_div_calib_z = scores_raw["Data HCFM / compression_div_abs_x"]["calib_z"]
        hcfm_div_z = scores_raw["Data HCFM / compression_div_abs_x"]["test_z"]
        combined_calib = hcfm_consistency_calib_z + cfg.hcfm_combined_alpha * hcfm_div_calib_z
        combined_test = hcfm_consistency_z + cfg.hcfm_combined_alpha * hcfm_div_z
        add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, "Data HCFM", "consistency_plus_compression_div_x", combined_calib, combined_test, y, test_starts, point_labels, cfg.window, rare_normal_mask, cfg=cfg)

    raw_items = list(scores_raw.items())
    method_score_names = {}
    for key, values in raw_items:
        method, score_name = key.split(" / ", 1)
        method_score_names.setdefault(method, []).append(score_name)
    for method, score_names in method_score_names.items():
        calib_dict = {name: scores_raw[f"{method} / {name}"]["calib_raw"] for name in score_names}
        test_dict = {name: scores_raw[f"{method} / {name}"]["test_raw"] for name in score_names}
        generic_calib = build_generic_derived_scores(calib_dict, hp_lambda=cfg.hp_lambda, rolling_window=5, add_hp_cycle_scores=cfg.add_hp_cycle_scores)
        generic_test = build_generic_derived_scores(test_dict, hp_lambda=cfg.hp_lambda, rolling_window=5, add_hp_cycle_scores=cfg.add_hp_cycle_scores)
        if not getattr(cfg, "compute_all_generic_smoothing", False):
            selected_roots = {
                "transport_energy_x",
                "compression_div_neg_sq_x",
                "compression_div_signed_x",
                "compression_div_centered_abs_x",
                "compression_div_centered_sq_x",
                "compression_div_centered_neg_abs_x",
                "compression_div_centered_neg_sq_x",
                "exact_like_base_plus_compression_div_x",
                "exact_like_base_minus_compression_div_x",
                "exact_like_base_plus_full_div_x",
                "exact_like_base_minus_full_div_x",
            }
            generic_calib = {k: v for k, v in generic_calib.items() if any(k.endswith(root) for root in selected_roots)}
            generic_test = {k: v for k, v in generic_test.items() if any(k.endswith(root) for root in selected_roots)}
        for score_name in generic_test:
            if score_name in generic_calib:
                add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, method, score_name, generic_calib[score_name], generic_test[score_name], y, test_starts, point_labels, cfg.window, rare_normal_mask, cfg=cfg)
        if method == "Data HCFM" and getattr(cfg, "compute_mechanism_fusions", False):
            hcfm_mech_calib = build_hcfm_mechanism_scores(calib_dict, calib_score_dict=calib_dict, hp_lambda=cfg.hp_lambda, rolling_window=5)
            hcfm_mech_test = build_hcfm_mechanism_scores(test_dict, calib_score_dict=calib_dict, hp_lambda=cfg.hp_lambda, rolling_window=5)
            for score_name in hcfm_mech_test:
                if score_name in hcfm_mech_calib:
                    add_dataspace_score(metrics_rows, scores_z, scores_raw, score_stats, method, score_name, hcfm_mech_calib[score_name], hcfm_mech_test[score_name], y, test_starts, point_labels, cfg.window, rare_normal_mask, cfg=cfg)

    score_calibration_stats = add_config_metadata(pd.DataFrame(score_stats), cfg)
    score_calibration_stats.to_csv(run_dir / "score_calibration_stats.csv", index=False)
    scoring_runtime_memory = add_config_metadata(pd.DataFrame(scoring_perf_rows), cfg)
    if not scoring_runtime_memory.empty:
        scoring_runtime_memory.to_csv(run_dir / "scoring_runtime_memory.csv", index=False)
    metrics = pd.DataFrame(metrics_rows).sort_values(["Model", "Score"]) if metrics_rows else pd.DataFrame()
    metrics = add_config_metadata(metrics, cfg)
    metrics.insert(0, "calibration_fraction_actual", float(calib_size / len(train_x_seq)))
    metrics.insert(0, "calibration_windows", int(calib_size))
    metrics.insert(0, "seed", cfg.seed) if "seed" not in metrics.columns else None
    metrics.insert(0, "dataset_id", cfg.dataset_id) if "dataset_id" not in metrics.columns else None
    if not metrics.empty:
        metrics["train_time_sec"] = metrics["Model"].map(lambda m: timing_by_method.get(m, {}).get("train_time_sec", np.nan))
        metrics["steps_per_sec"] = metrics["Model"].map(lambda m: timing_by_method.get(m, {}).get("steps_per_sec", np.nan))
        metrics["trainable_params"] = metrics["Model"].map(lambda m: param_map.get(m, np.nan))
    metrics.to_csv(run_dir / "results.csv", index=False)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    metrics.to_csv(run_dir / "dataspace_cnn_metrics.csv", index=False)
    score_calibration_stats.to_csv(run_dir / "score_calibration_stats.csv", index=False)
    score_calibration_stats.to_csv(run_dir / "score_stats.csv", index=False)

    if not metrics.empty:
        hundman_cols = [c for c in [
            "Model",
            "Score",
            "hundman_window_precision",
            "hundman_window_recall",
            "hundman_window_f1",
            "hundman_window_f1_mean_plus_2sd",
            "hundman_window_f1_mean_plus_3sd",
            "hundman_point_mean_precision",
            "hundman_point_mean_recall",
            "hundman_point_mean_f1",
            "hundman_point_mean_f1_mean_plus_2sd",
            "hundman_point_mean_f1_mean_plus_3sd",
            "hundman_point_max_precision",
            "hundman_point_max_recall",
            "hundman_point_max_f1",
            "hundman_point_max_f1_mean_plus_2sd",
            "hundman_point_max_f1_mean_plus_3sd",
        ] if c in metrics.columns]
        if hundman_cols:
            hundman_summary = metrics[hundman_cols].rename(columns={"Model": "model", "Score": "score_name"}).copy()
            add_config_metadata(hundman_summary, cfg).to_csv(run_dir / "hundman_metrics_summary.csv", index=False)
        vus_cols = [c for c in ["Model", "Score", "vus_sliding_window", "vus_pr_point_mean", "vus_roc_point_mean", "vus_pr_point_max", "vus_roc_point_max"] if c in metrics.columns]
        if vus_cols:
            vus_summary = metrics[vus_cols].rename(columns={"Model": "model", "Score": "score_name"}).copy()
            sort_col = "vus_pr_point_mean" if "vus_pr_point_mean" in vus_summary.columns else ("vus_pr_point_max" if "vus_pr_point_max" in vus_summary.columns else None)
            if sort_col:
                vus_summary = vus_summary.sort_values(sort_col, ascending=False)
            add_config_metadata(vus_summary, cfg).to_csv(run_dir / "vus_metrics_summary.csv", index=False)

    key_scores = select_key_scores(metrics.rename(columns={"Model": "method", "Score": "score"}))
    key_scores.to_csv(run_dir / "key_scores.csv", index=False)

    train_logs = []
    for method, hist in histories.items():
        if isinstance(hist, pd.DataFrame):
            h = add_config_metadata(hist, cfg)
            h["method"] = method
            train_logs.append(h)
    train_log_df = pd.concat(train_logs, ignore_index=True) if train_logs else pd.DataFrame()
    if not train_log_df.empty:
        train_log_df.to_csv(run_dir / "train_log.csv", index=False)

    if bool(getattr(cfg, "save_score_npz", False)):
        score_npz = {
            "labels": np.asarray(test_y, dtype=np.int64),
            "starts": np.asarray(test_starts, dtype=np.int64),
            "point_labels": np.asarray(point_labels, dtype=np.int64),
            "rare_normal_mask": np.asarray(rare_normal_mask, dtype=bool),
            "calib_idx": np.asarray(calib_idx, dtype=np.int64),
        }
        for key, payload in scores_raw.items():
            safe_key = key.replace(" / ", "__").replace(" ", "_")
            for suffix, values in payload.items():
                score_npz[f"{safe_key}__{suffix}"] = np.asarray(values)
        np.savez_compressed(run_dir / "scores_raw_and_standardized.npz", **score_npz)

    if method_errors:
        pd.DataFrame(method_errors).to_csv(run_dir / "method_failures.csv", index=False)
    _save_json(run_dir / "final_run_summary.json", {
        "dataset_id": cfg.dataset_id,
        "seed": int(cfg.seed),
        "rows": int(len(metrics)),
        "method_failures": int(len(method_errors)),
        "compute_hundman_metrics": bool(getattr(cfg, "compute_hundman_metrics", False)),
        "compute_vus_metrics": bool(getattr(cfg, "compute_vus_metrics", False)),
        "compute_exact_likelihood_scores": bool(getattr(cfg, "compute_exact_likelihood_scores", False)),
    })
    clear_cuda_cache()
    return {
        "results": metrics,
        "key_scores": key_scores,
        "parameter_counts": param_df,
        "train_log": train_log_df,
        "method_failures": pd.DataFrame(method_errors),
    }


def is_raw_score(score_name):
    s = str(score_name)
    return not (s.startswith("hp_trend_") or s.startswith("hp_cycle_") or s.startswith("rollmean_") or s.startswith("rollmax_") or s.startswith("hcfm_"))


def is_generic_score(score_name):
    s = str(score_name)
    return s.startswith("hp_trend_") or s.startswith("hp_cycle_") or s.startswith("rollmean_") or s.startswith("rollmax_")


def select_key_scores(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df
    df = results_df.rename(columns={"Model": "method", "Score": "score"}).copy()
    key_names = {
        "compression_div_signed_x",
        "compression_div_neg_sq_x",
        "hp_trend_compression_div_neg_sq_x",
        "rollmax_compression_div_neg_sq_x",
        "compression_div_centered_neg_sq_x",
        "hp_trend_compression_div_centered_neg_sq_x",
        "transport_energy_x",
        "exact_like_base_plus_compression_div_x",
        "exact_like_base_plus_full_div_x",
        "hcfm_gated_div_plus_base_z",
        "hcfm_mul_base_div_z",
    }
    parts = [df[(df["method"] == "Data HCFM") & (df["score"].isin(key_names))]]
    metric = "point_AUPRC_mean" if "point_AUPRC_mean" in df.columns else "point_AUPRC"
    if metric in df.columns:
        raw = df[df["score"].apply(is_raw_score)]
        generic = df[df["score"].apply(lambda s: is_raw_score(s) or is_generic_score(s))]
        if not raw.empty:
            parts.append(raw.loc[raw.groupby("method")[metric].idxmax()])
        if not generic.empty:
            parts.append(generic.loc[generic.groupby("method")[metric].idxmax()])
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=[c for c in ["dataset_id", "seed", "method", "score"] if c in df.columns])
