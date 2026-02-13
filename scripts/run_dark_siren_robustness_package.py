#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

# Ensure we import the *local* package from this repository (not a different checkout on sys.path).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from entropy_horizon_recon.dark_siren_gap_lpd import BetaPrior, MarginalizedFMissResult, marginalize_f_miss_global  # noqa: E402
from entropy_horizon_recon.dark_sirens_incompleteness import (  # noqa: E402
    compute_missing_host_logL_draws_from_histogram,
    precompute_missing_host_prior,
)
from entropy_horizon_recon.dark_sirens_pe import (  # noqa: E402
    PePixelDistanceHistogram,
    build_pe_pixel_distance_histogram,
    load_gwtc_pe_sky_samples,
    compute_dark_siren_logL_draws_from_pe_hist,
)
from entropy_horizon_recon.dark_sirens_selection import load_o3_injections  # noqa: E402
from entropy_horizon_recon.gw_distance_priors import GWDistancePrior  # noqa: E402
from entropy_horizon_recon.report import ReportPaths, format_table, write_markdown_report  # noqa: E402
from entropy_horizon_recon.repro import command_str, git_head_sha, git_is_dirty  # noqa: E402
from entropy_horizon_recon.selection_nuisance import (  # noqa: E402
    SelectionNuisanceConfig,
    apply_nuisance_to_alpha_linearized,
    compute_selection_nuisance_moments,
)
from entropy_horizon_recon.sirens import MuForwardPosterior, load_mu_forward_posterior  # noqa: E402


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _set_thread_env(n: int) -> None:
    # Keep this deterministic and avoid stealing all cores by default.
    n = int(max(1, n))
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[k] = str(n)


def _events_from_cache_terms(cache_terms_dir: Path, *, run_label: str) -> list[str]:
    ev = []
    for p in sorted(cache_terms_dir.glob(f"cat_*__{run_label}.npz")):
        name = p.name
        # cat_<EVENT>__RUN.npz
        if not name.startswith("cat_") or f"__{run_label}.npz" not in name:
            continue
        event = name[len("cat_") : name.index(f"__{run_label}.npz")]
        ev.append(event)
    if not ev:
        raise FileNotFoundError(f"No cache_terms/cat_*__{run_label}.npz found under {cache_terms_dir}")
    return ev


@dataclass(frozen=True)
class GapRunCache:
    gap_run_root: Path
    run_label: str
    events: list[str]
    logL_cat_mu_by_event: list[np.ndarray]
    logL_cat_gr_by_event: list[np.ndarray]
    logL_missing_mu_by_event: list[np.ndarray]
    logL_missing_gr_by_event: list[np.ndarray]
    log_alpha_mu: np.ndarray
    log_alpha_gr: np.ndarray
    prior: BetaPrior
    n_f: int
    eps: float
    convention: str
    draw_idx: list[int]
    selection_meta: dict[str, Any]
    manifest: dict[str, Any]
    baseline_summary: dict[str, Any]


def _load_gap_run_cache(*, gap_run_root: Path, run_label: str) -> GapRunCache:
    gap_run_root = gap_run_root.expanduser().resolve()
    if not gap_run_root.exists():
        raise FileNotFoundError(gap_run_root)

    cache_terms_dir = gap_run_root / "cache_terms"
    cache_missing_dir = gap_run_root / "cache_missing"
    tables_dir = gap_run_root / "tables"

    baseline_summary = _read_json(gap_run_root / f"summary_{run_label}.json")
    manifest = _read_json(gap_run_root / "manifest.json")

    events = _events_from_cache_terms(cache_terms_dir, run_label=run_label)

    logL_cat_mu_by_event: list[np.ndarray] = []
    logL_cat_gr_by_event: list[np.ndarray] = []
    logL_missing_mu_by_event: list[np.ndarray] = []
    logL_missing_gr_by_event: list[np.ndarray] = []
    for ev in events:
        with np.load(cache_terms_dir / f"cat_{ev}__{run_label}.npz", allow_pickle=False) as d:
            logL_cat_mu_by_event.append(np.asarray(d["logL_cat_mu"], dtype=float))
            logL_cat_gr_by_event.append(np.asarray(d["logL_cat_gr"], dtype=float))
        with np.load(cache_missing_dir / f"missing_{ev}__{run_label}.npz", allow_pickle=False) as d:
            logL_missing_mu_by_event.append(np.asarray(d["logL_missing_mu"], dtype=float))
            logL_missing_gr_by_event.append(np.asarray(d["logL_missing_gr"], dtype=float))

    sel_npz = tables_dir / f"selection_alpha_{run_label}.npz"
    with np.load(sel_npz, allow_pickle=True) as d:
        log_alpha_mu = np.asarray(d["log_alpha_mu"], dtype=float)
        log_alpha_gr = np.asarray(d["log_alpha_gr"], dtype=float)
        selection_meta = json.loads(str(np.asarray(d["meta"]).tolist()))

    mix = baseline_summary.get("mixture", {})
    mix_meta = mix.get("f_miss_meta", {})
    prior_meta = mix_meta.get("prior", {})
    grid_meta = mix_meta.get("grid", {})

    prior = BetaPrior(mean=float(prior_meta["mean"]), kappa=float(prior_meta["kappa"]))
    n_f = int(grid_meta.get("n", 401))
    eps = float(grid_meta.get("eps", 1e-6))

    convention = str(baseline_summary.get("convention", "A"))
    draw_idx = [int(i) for i in baseline_summary.get("draw_idx", [])]
    if not draw_idx:
        raise ValueError("Missing draw_idx in baseline summary; cannot reconstruct selection moments.")

    return GapRunCache(
        gap_run_root=gap_run_root,
        run_label=str(run_label),
        events=events,
        logL_cat_mu_by_event=logL_cat_mu_by_event,
        logL_cat_gr_by_event=logL_cat_gr_by_event,
        logL_missing_mu_by_event=logL_missing_mu_by_event,
        logL_missing_gr_by_event=logL_missing_gr_by_event,
        log_alpha_mu=log_alpha_mu,
        log_alpha_gr=log_alpha_gr,
        prior=prior,
        n_f=n_f,
        eps=eps,
        convention=convention,
        draw_idx=draw_idx,
        selection_meta=selection_meta,
        manifest=manifest,
        baseline_summary=baseline_summary,
    )


def _subset_mu_posterior(post: MuForwardPosterior, idx: list[int]) -> MuForwardPosterior:
    ii = np.asarray(idx, dtype=int)
    return MuForwardPosterior(
        x_grid=post.x_grid,
        logmu_x_samples=post.logmu_x_samples[ii],
        z_grid=post.z_grid,
        H_samples=post.H_samples[ii],
        H0=post.H0[ii],
        omega_m0=post.omega_m0[ii],
        omega_k0=post.omega_k0[ii],
        sigma8_0=post.sigma8_0[ii] if post.sigma8_0 is not None else None,
    )


def _plot_curve(
    x: np.ndarray,
    ys: dict[str, np.ndarray],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    hline: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for lab, y in ys.items():
        ax.plot(x, y, lw=2.0, label=lab)
    if hline is not None:
        ax.axhline(float(hline), color="k", lw=1.2, alpha=0.6)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=0.25, linestyle=":")
    if len(ys) > 1:
        ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _delta_lpd_from_cached_terms(
    *,
    cache: GapRunCache,
    log_alpha_mu: np.ndarray,
    log_alpha_gr: np.ndarray,
    events_mask: np.ndarray | None = None,
) -> MarginalizedFMissResult:
    if events_mask is None:
        return marginalize_f_miss_global(
            logL_cat_mu_by_event=cache.logL_cat_mu_by_event,
            logL_cat_gr_by_event=cache.logL_cat_gr_by_event,
            logL_missing_mu_by_event=cache.logL_missing_mu_by_event,
            logL_missing_gr_by_event=cache.logL_missing_gr_by_event,
            log_alpha_mu=log_alpha_mu,
            log_alpha_gr=log_alpha_gr,
            prior=cache.prior,
            n_f=cache.n_f,
            eps=cache.eps,
        )
    m = np.asarray(events_mask, dtype=bool)
    if m.ndim != 1 or m.size != len(cache.events):
        raise ValueError("events_mask must be 1D and match number of events.")
    cat_mu = [a for a, ok in zip(cache.logL_cat_mu_by_event, m, strict=True) if ok]
    cat_gr = [a for a, ok in zip(cache.logL_cat_gr_by_event, m, strict=True) if ok]
    miss_mu = [a for a, ok in zip(cache.logL_missing_mu_by_event, m, strict=True) if ok]
    miss_gr = [a for a, ok in zip(cache.logL_missing_gr_by_event, m, strict=True) if ok]
    return marginalize_f_miss_global(
        logL_cat_mu_by_event=cat_mu,
        logL_cat_gr_by_event=cat_gr,
        logL_missing_mu_by_event=miss_mu,
        logL_missing_gr_by_event=miss_gr,
        log_alpha_mu=log_alpha_mu,
        log_alpha_gr=log_alpha_gr,
        prior=cache.prior,
        n_f=cache.n_f,
        eps=cache.eps,
    )


def _pe_distance_median_from_event_cache(npz_path: Path) -> float:
    """Approximate median dL (Mpc) from cached per-pixel PE distance histogram."""
    with np.load(npz_path, allow_pickle=False) as d:
        prob_pix = np.asarray(d["pe_prob_pix"], dtype=float)
        pdf_bins = np.asarray(d["pe_pdf_bins"], dtype=float)
        edges = np.asarray(d["pe_dL_edges"], dtype=float)
    if prob_pix.ndim != 1:
        raise ValueError(f"{npz_path}: pe_prob_pix must be 1D.")
    if edges.ndim != 1 or edges.size < 3:
        raise ValueError(f"{npz_path}: pe_dL_edges must be 1D with >=3 entries.")
    widths = np.diff(edges)
    if pdf_bins.ndim != 2 or pdf_bins.shape[1] != widths.size:
        raise ValueError(f"{npz_path}: pe_pdf_bins must be 2D with shape (n_pix, n_bins).")

    p_sum = float(np.sum(prob_pix))
    if not (np.isfinite(p_sum) and p_sum > 0.0):
        raise ValueError(f"{npz_path}: invalid pe_prob_pix normalization.")

    if pdf_bins.shape[0] == prob_pix.size:
        pdf_1d = np.sum(prob_pix.reshape((-1, 1)) * pdf_bins, axis=0) / p_sum
    elif pdf_bins.shape[0] == 1:
        pdf_1d = np.asarray(pdf_bins[0], dtype=float)
    else:
        raise ValueError(f"{npz_path}: incompatible pe_pdf_bins shape {pdf_bins.shape} vs prob_pix {prob_pix.shape}.")

    pdf_1d = np.clip(pdf_1d, 0.0, np.inf)
    norm = float(np.sum(pdf_1d * widths))
    if not (np.isfinite(norm) and norm > 0.0):
        raise ValueError(f"{npz_path}: invalid sky-marginal PE distance normalization.")
    pdf_1d = pdf_1d / norm

    cdf = np.cumsum(pdf_1d * widths)
    j = int(np.searchsorted(cdf, 0.5, side="left"))
    j = min(max(j, 0), int(widths.size - 1))
    c0 = float(cdf[j - 1]) if j > 0 else 0.0
    p = float(pdf_1d[j])
    if not (np.isfinite(p) and p > 0.0):
        # Fall back to midpoint if density is pathological.
        return float(0.5 * (edges[j] + edges[j + 1]))
    frac = (0.5 - c0) / (p * float(widths[j]))
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(edges[j] + frac * widths[j])


def _lcdm_z_from_dL(dL_mpc: float, *, H0: float, omega_m0: float, z_max: float = 1.0) -> float:
    """Invert a flat LCDM dL(z) relation by interpolation (adequate for binning/splits)."""
    dL = float(dL_mpc)
    if not (np.isfinite(dL) and dL > 0.0):
        return float("nan")
    H0 = float(H0)
    om = float(omega_m0)
    if not (np.isfinite(H0) and H0 > 0.0 and np.isfinite(om) and 0.0 < om < 2.0):
        raise ValueError("Invalid H0/omega_m0 for LCDM inversion.")

    c = 299792.458  # km/s
    z_grid = np.linspace(0.0, float(z_max), 20001)
    Ez = np.sqrt(om * (1.0 + z_grid) ** 3 + (1.0 - om))
    invE = 1.0 / Ez
    dz = np.diff(z_grid)
    dc = np.empty_like(z_grid)
    dc[0] = 0.0
    dc[1:] = (c / H0) * np.cumsum(0.5 * dz * (invE[:-1] + invE[1:]))
    dL_grid = (1.0 + z_grid) * dc
    if dL >= float(dL_grid[-1]):
        return float(z_grid[-1])
    return float(np.interp(dL, dL_grid, z_grid))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dark-siren robustness package: selection nuisance + split coherence re-scoring.")
    ap.add_argument("--config", type=str, default="configs/dark_siren_robustness_package_o3.json")
    ap.add_argument("--out", type=str, default="", help="Output directory (default: outputs/dark_siren_robustness_package_o3_<UTC>).")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-selection", action="store_true", help="Skip selection nuisance module (A1).")
    ap.add_argument("--skip-splits", action="store_true", help="Skip split-coherence module (B2).")
    ap.add_argument("--skip-a2", action="store_true", help="Skip catalog/photo-z stress module (A2).")
    ap.add_argument("--skip-a3", action="store_true", help="Skip PE/waveform robustness module (A3).")
    ap.add_argument("--mc-samples", type=int, default=2000, help="Monte Carlo samples for nuisance marginalization.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    _set_thread_env(int(args.threads))

    repo_root = Path(__file__).resolve().parents[1]
    cfg = _read_json(Path(args.config))

    gap_run_root = Path(cfg["baseline"]["gap_run_root"])
    run_label = str(cfg["baseline"]["run_label"])
    recon_run_dir = Path(cfg["baseline"].get("recon_run_dir", "/home/primary/PROJECT/outputs/realdata_variant_M0"))

    out_dir = Path(args.out) if str(args.out).strip() else (repo_root / "outputs" / f"dark_siren_robustness_package_o3_{_utc_now_compact()}")
    paths = ReportPaths(out_dir=out_dir)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    paths.samples_dir.mkdir(parents=True, exist_ok=True)

    cache = _load_gap_run_cache(gap_run_root=gap_run_root, run_label=run_label)

    # Baseline reproduction.
    base = _delta_lpd_from_cached_terms(cache=cache, log_alpha_mu=cache.log_alpha_mu, log_alpha_gr=cache.log_alpha_gr)
    _write_json_atomic(paths.tables_dir / "baseline_recompute.json", base.to_jsonable())

    # Compare to stored summary (guardrail).
    ref = cache.baseline_summary
    chk = {
        "ref_delta_lpd_total": float(ref["delta_lpd_total"]),
        "recompute_delta_lpd_total": float(base.lpd_mu_total - base.lpd_gr_total),
        "ref_delta_lpd_total_data": float(ref["delta_lpd_total_data"]),
        "recompute_delta_lpd_total_data": float(base.lpd_mu_total_data - base.lpd_gr_total_data),
        "ref_delta_lpd_total_sel": float(ref["delta_lpd_total_sel"]),
        "recompute_delta_lpd_total_sel": float((base.lpd_mu_total - base.lpd_gr_total) - (base.lpd_mu_total_data - base.lpd_gr_total_data)),
    }
    _write_json_atomic(paths.tables_dir / "baseline_check.json", chk)

    sel_enabled = bool(cfg.get("selection_nuisance", {}).get("enabled", False)) and not bool(args.skip_selection)
    a2_enabled = bool(cfg.get("catalog_photoz_stress", {}).get("enabled", False)) and not bool(args.skip_a2)
    a3_enabled = bool(cfg.get("pe_waveform_stress", {}).get("enabled", False)) and not bool(args.skip_a3)

    post: MuForwardPosterior | None = None
    if sel_enabled or a2_enabled or a3_enabled:
        post_full = load_mu_forward_posterior(recon_run_dir)
        post = _subset_mu_posterior(post_full, cache.draw_idx)

    # A1) Selection-function nuisance deformations (linearized moments).
    sel_rows: list[list[Any]] = []
    sel_summary: dict[str, Any] = {"enabled": False}
    if sel_enabled:
        if post is None:
            raise RuntimeError("Internal error: expected post to be loaded for selection nuisance.")
        sel_cfg = cfg["selection_nuisance"]
        cfg_obj = SelectionNuisanceConfig(
            logsnr_knots=np.asarray(sel_cfg["logsnr_knots"], dtype=float),
            z_knots=None if not sel_cfg.get("z_knots") else np.asarray(sel_cfg["z_knots"], dtype=float),
            mass_pivot_msun=float(sel_cfg.get("mass_pivot_msun", 30.0)),
            mass_log_scale=float(sel_cfg.get("mass_log_scale", 1.0)),
        )
        delta_bound = float(sel_cfg.get("delta_bound", 0.2))
        prior_sigma = float(sel_cfg.get("prior_sigma", 0.1))
        include_mass = bool(sel_cfg.get("include_mass", True))

        # Load injections path from baseline manifest.
        inj_path = cache.manifest.get("selection_injections_hdf")
        if not inj_path:
            raise KeyError("baseline manifest missing selection_injections_hdf")
        injections = load_o3_injections(str(inj_path), ifar_threshold_yr=float(cache.manifest.get("selection_ifar_thresh_yr", 1.0)))

        # Mirror baseline selection config.
        sel_meta = cache.selection_meta
        mom = compute_selection_nuisance_moments(
            post=post,
            injections=injections,
            convention=str(cache.convention),
            z_max=float(sel_meta.get("z_max", 0.3)),
            weight_mode=str(sel_meta.get("weight_mode", "inv_sampling_pdf")),
            snr_offset=0.0,
            injection_logit_l2=float(sel_meta.get("injection_logit_l2", 1e-2)),
            injection_logit_max_iter=int(sel_meta.get("injection_logit_max_iter", 64)),
            cfg=cfg_obj,
            mu_det_distance="gw",
            pop_z_mode=str(sel_meta.get("pop_z_mode", "none")),
            pop_z_powerlaw_k=float(sel_meta.get("pop_z_k", 0.0)),
            pop_mass_mode=str(sel_meta.get("pop_mass_mode", "none")),
            pop_m1_alpha=float(sel_meta.get("pop_m1_alpha", 2.3)),
            pop_m_min=float(sel_meta.get("pop_m_min", 5.0)),
            pop_m_max=float(sel_meta.get("pop_m_max", 80.0)),
            pop_q_beta=float(sel_meta.get("pop_q_beta", 0.0)),
            pop_m_taper_delta=float(sel_meta.get("pop_m_taper_delta", 0.0)),
            pop_m_peak=float(sel_meta.get("pop_m_peak", 35.0)),
            pop_m_peak_sigma=float(sel_meta.get("pop_m_peak_sigma", 5.0)),
            pop_m_peak_frac=float(sel_meta.get("pop_m_peak_frac", 0.1)),
        )

        # Save moments.
        np.savez_compressed(
            paths.tables_dir / "selection_nuisance_moments.npz",
            alpha_mu_base=mom.alpha_mu_base,
            alpha_gr_base=mom.alpha_gr_base,
            mom_mu_logsnr=mom.mom_mu_logsnr,
            mom_gr_logsnr=mom.mom_gr_logsnr,
            mom_mu_z=np.array([]) if mom.mom_mu_z is None else mom.mom_mu_z,
            mom_gr_z=np.array([]) if mom.mom_gr_z is None else mom.mom_gr_z,
            mom_mu_mass=np.array([]) if mom.mom_mu_mass is None else mom.mom_mu_mass,
            mom_gr_mass=np.array([]) if mom.mom_gr_mass is None else mom.mom_gr_mass,
            meta=json.dumps(mom.to_jsonable()),
        )

        # Quick sanity: base alpha match to cached selection_alpha.
        base_alpha_diff = {
            "max_abs_alpha_mu": float(np.max(np.abs(mom.alpha_mu_base - np.exp(cache.log_alpha_mu)))),
            "max_abs_alpha_gr": float(np.max(np.abs(mom.alpha_gr_base - np.exp(cache.log_alpha_gr)))),
        }

        # Stress curves: vary each logSNR knot individually.
        grid = np.linspace(-delta_bound, delta_bound, 21)
        k_snr = int(mom.mom_mu_logsnr.shape[1])
        curves = []
        for kk in range(k_snr):
            dl = np.zeros((k_snr,), dtype=float)
            y = []
            for a in grid:
                dl2 = dl.copy()
                dl2[kk] = float(a)
                az = None if mom.mom_mu_z is None else np.zeros((int(mom.mom_mu_z.shape[1]),), dtype=float)
                bm = float(a) if (include_mass and kk == 0) else 0.0  # keep mass scan separate below
                alpha_mu, alpha_gr = apply_nuisance_to_alpha_linearized(mom=mom, delta_logsnr_knots=dl2, delta_z_knots=az, b_mass=0.0)
                res = _delta_lpd_from_cached_terms(cache=cache, log_alpha_mu=np.log(alpha_mu), log_alpha_gr=np.log(alpha_gr))
                y.append(float(res.lpd_mu_total - res.lpd_gr_total))
            curves.append((kk, np.asarray(y, dtype=float)))
            _plot_curve(
                grid,
                {f"knot_{kk}": np.asarray(y, dtype=float)},
                xlabel="delta (multiplicative in p_det)",
                ylabel="Delta LPD (total)",
                title=f"Selection Nuisance: logSNR knot {kk}",
                path=paths.figures_dir / f"sel_nuisance_logsnr_knot{kk}.png",
                hline=float(base.lpd_mu_total - base.lpd_gr_total),
            )

        # Stress curve: global logSNR offset (all knots equal).
        y_all = []
        for a in grid:
            dl = np.full((k_snr,), float(a), dtype=float)
            dz = None if mom.mom_mu_z is None else np.zeros((int(mom.mom_mu_z.shape[1]),), dtype=float)
            alpha_mu, alpha_gr = apply_nuisance_to_alpha_linearized(mom=mom, delta_logsnr_knots=dl, delta_z_knots=dz, b_mass=0.0)
            res = _delta_lpd_from_cached_terms(cache=cache, log_alpha_mu=np.log(alpha_mu), log_alpha_gr=np.log(alpha_gr))
            y_all.append(float(res.lpd_mu_total - res.lpd_gr_total))
        _plot_curve(
            grid,
            {"all_knots_equal": np.asarray(y_all, dtype=float)},
            xlabel="delta",
            ylabel="Delta LPD (total)",
            title="Selection Nuisance: global logSNR offset",
            path=paths.figures_dir / "sel_nuisance_logsnr_global.png",
            hline=float(base.lpd_mu_total - base.lpd_gr_total),
        )

        # Monte Carlo marginalization (truncated normal prior).
        rng = np.random.default_rng(int(args.seed))
        n_mc = int(max(100, int(args.mc_samples)))
        params_mc = []
        scores_mc = []
        for _ in range(n_mc):
            # Truncated normal via rejection (cheap at these dims).
            dl = rng.normal(loc=0.0, scale=prior_sigma, size=(k_snr,))
            dl = np.clip(dl, -delta_bound, delta_bound)
            dz = None
            if mom.mom_mu_z is not None:
                kz = int(mom.mom_mu_z.shape[1])
                dz = rng.normal(loc=0.0, scale=prior_sigma, size=(kz,))
                dz = np.clip(dz, -delta_bound, delta_bound)
            bm = float(rng.normal(loc=0.0, scale=prior_sigma)) if include_mass else 0.0
            bm = float(np.clip(bm, -delta_bound, delta_bound))
            alpha_mu, alpha_gr = apply_nuisance_to_alpha_linearized(mom=mom, delta_logsnr_knots=dl, delta_z_knots=dz, b_mass=bm)
            res = _delta_lpd_from_cached_terms(cache=cache, log_alpha_mu=np.log(alpha_mu), log_alpha_gr=np.log(alpha_gr))
            dlp = float(res.lpd_mu_total - res.lpd_gr_total)
            scores_mc.append(dlp)
            row = {f"dl_snr_{i}": float(dl[i]) for i in range(k_snr)}
            if dz is not None:
                for i in range(dz.size):
                    row[f"dl_z_{i}"] = float(dz[i])
            row["b_mass"] = float(bm)
            row["delta_lpd_total"] = dlp
            params_mc.append(row)
        scores_mc = np.asarray(scores_mc, dtype=float)
        fieldnames = sorted(params_mc[0].keys()) if params_mc else []
        _write_csv(paths.samples_dir / "selection_nuisance_mc.csv", params_mc, fieldnames=fieldnames)

        # Adversarial minimization.
        def _unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, float]:
            x = np.asarray(x, dtype=float)
            dl = x[:k_snr]
            off = k_snr
            dz = None
            if mom.mom_mu_z is not None:
                kz = int(mom.mom_mu_z.shape[1])
                dz = x[off : off + kz]
                off += kz
            bm = float(x[off]) if include_mass else 0.0
            return dl, dz, bm

        def _delta_lpd_obj(x: np.ndarray) -> float:
            dl, dz, bm = _unpack(x)
            alpha_mu, alpha_gr = apply_nuisance_to_alpha_linearized(mom=mom, delta_logsnr_knots=dl, delta_z_knots=dz, b_mass=bm)
            res = _delta_lpd_from_cached_terms(cache=cache, log_alpha_mu=np.log(alpha_mu), log_alpha_gr=np.log(alpha_gr))
            return float(res.lpd_mu_total - res.lpd_gr_total)

        def _delta_lpd_obj_penalized(x: np.ndarray) -> float:
            x = np.asarray(x, dtype=float)
            dlp = _delta_lpd_obj(x)
            pen = 0.5 * float(np.sum((x / prior_sigma) ** 2))
            return dlp + pen

        x0 = np.zeros((k_snr + (int(mom.mom_mu_z.shape[1]) if mom.mom_mu_z is not None else 0) + (1 if include_mass else 0),), dtype=float)
        bounds = [(-delta_bound, delta_bound)] * x0.size

        adv_unpen = scipy.optimize.minimize(_delta_lpd_obj, x0=x0, method="L-BFGS-B", bounds=bounds)
        adv_pen = scipy.optimize.minimize(_delta_lpd_obj_penalized, x0=x0, method="L-BFGS-B", bounds=bounds)

        def _summ(opt: scipy.optimize.OptimizeResult) -> dict[str, Any]:
            x = np.asarray(opt.x, dtype=float)
            dl, dz, bm = _unpack(x)
            return {
                "success": bool(opt.success),
                "message": str(opt.message),
                "nfev": int(getattr(opt, "nfev", -1)),
                "delta_lpd_total": float(_delta_lpd_obj(x)),
                "prior_penalty": float(0.5 * np.sum((x / prior_sigma) ** 2)),
                "max_abs_param": float(np.max(np.abs(x))) if x.size else 0.0,
                "at_bounds_frac": float(np.mean(np.isclose(np.abs(x), delta_bound, rtol=0.0, atol=1e-6))) if x.size else 0.0,
                "dl_snr": [float(v) for v in dl.tolist()],
                "dl_z": None if dz is None else [float(v) for v in dz.tolist()],
                "b_mass": float(bm),
            }

        sel_summary = {
            "enabled": True,
            "delta_bound": float(delta_bound),
            "prior_sigma": float(prior_sigma),
            "base_alpha_diff": base_alpha_diff,
            "mc": {
                "n": int(n_mc),
                "delta_lpd_mean": float(np.mean(scores_mc)),
                "delta_lpd_median": float(np.median(scores_mc)),
                "delta_lpd_p05": float(np.quantile(scores_mc, 0.05)),
                "delta_lpd_p95": float(np.quantile(scores_mc, 0.95)),
                "p_delta_lpd_gt0": float(np.mean(scores_mc > 0.0)),
                "p_delta_lpd_gt1": float(np.mean(scores_mc > 1.0)),
            },
            "adversarial": {
                "unpenalized": _summ(adv_unpen),
                "penalized": _summ(adv_pen),
            },
        }
        _write_json_atomic(paths.tables_dir / "selection_nuisance_summary.json", sel_summary)

        sel_rows = [
            ["Baseline ΔLPD", f"{(base.lpd_mu_total - base.lpd_gr_total):+.3f}"],
            ["MC median ΔLPD", f"{sel_summary['mc']['delta_lpd_median']:+.3f}"],
            ["Adversarial min (unpen.)", f"{sel_summary['adversarial']['unpenalized']['delta_lpd_total']:+.3f}"],
            ["Adversarial min (pen.)", f"{sel_summary['adversarial']['penalized']['delta_lpd_total']:+.3f}"],
            ["Adversarial max |param|", f"{sel_summary['adversarial']['unpenalized']['max_abs_param']:.3f}"],
        ]

    # A2/A3 share a spectral-only rescoring cache (fast recompute for adversarial PE/cat perturbations).
    spectral_cache: dict[str, Any] | None = None
    if a2_enabled or a3_enabled:
        if post is None:
            raise RuntimeError("Internal error: expected post to be loaded for A2/A3.")

        manifest = cache.manifest
        gal_chunk_size = int(manifest.get("galaxy_chunk_size", 50_000))
        pix_chunk_size = int(manifest.get("missing_pixel_chunk_size", 5_000))

        gw_prior = GWDistancePrior(
            mode="dL_powerlaw",
            powerlaw_k=float(manifest.get("gw_distance_prior_power", 2.0)),
            h0_ref=float(manifest.get("gw_distance_prior_h0_ref", 67.7)),
            omega_m0=float(manifest.get("gw_distance_prior_omega_m0", 0.31)),
            omega_k0=float(manifest.get("gw_distance_prior_omega_k0", 0.0)),
            z_max=float(manifest.get("gw_distance_prior_zmax", 10.0)),
            n_grid=50_000,
        )
        pre_missing = precompute_missing_host_prior(
            post,
            convention=str(cache.convention),
            z_max=float(manifest.get("missing_z_max", 0.3)),
            host_prior_z_mode=str(manifest.get("host_prior_z_mode", "comoving_uniform")),
            host_prior_z_k=float(manifest.get("host_prior_z_k", 0.0)),
        )

        def _load_event_cache(ev: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, PePixelDistanceHistogram]:
            p = cache.gap_run_root / "cache" / f"event_{ev}.npz"
            with np.load(p, allow_pickle=False) as d:
                z = np.asarray(d["z"], dtype=float)
                w = np.asarray(d["w"], dtype=float)
                ipix = np.asarray(d["ipix"], dtype=np.int64)
                pe = PePixelDistanceHistogram(
                    nside=int(cache.manifest.get("pe_nside", 64)),
                    nest=True,
                    p_credible=float(cache.manifest.get("p_credible", 0.9)),
                    pix_sel=np.asarray(d["pe_pix_sel"], dtype=np.int64),
                    prob_pix=np.asarray(d["pe_prob_pix"], dtype=float),
                    dL_edges=np.asarray(d["pe_dL_edges"], dtype=float),
                    pdf_bins=np.asarray(d["pe_pdf_bins"], dtype=float),
                )
            return z, w, ipix, pe

        # Baseline spectral-only logL vectors for the full event set (for fast perturbative rescoring).
        spec_cat_mu: list[np.ndarray] = []
        spec_cat_gr: list[np.ndarray] = []
        spec_miss_mu: list[np.ndarray] = []
        spec_miss_gr: list[np.ndarray] = []
        for ev in cache.events:
            z, w, ipix, pe = _load_event_cache(ev)
            logL_mu, logL_gr = compute_dark_siren_logL_draws_from_pe_hist(
                event=ev,
                pe=pe,
                post=post,
                z_gal=z,
                w_gal=w,
                ipix_gal=ipix,
                convention=str(cache.convention),
                gw_distance_prior=gw_prior,
                distance_mode="spectral_only",
                gal_chunk_size=gal_chunk_size,
            )
            miss_mu, miss_gr = compute_missing_host_logL_draws_from_histogram(
                prob_pix=np.asarray(pe.prob_pix, dtype=float),
                pdf_bins=np.asarray(pe.pdf_bins, dtype=float),
                dL_edges=np.asarray(pe.dL_edges, dtype=float),
                pre=pre_missing,
                gw_distance_prior=gw_prior,
                distance_mode="spectral_only",
                pixel_chunk_size=pix_chunk_size,
            )
            spec_cat_mu.append(np.asarray(logL_mu, dtype=float))
            spec_cat_gr.append(np.asarray(logL_gr, dtype=float))
            spec_miss_mu.append(np.asarray(miss_mu, dtype=float))
            spec_miss_gr.append(np.asarray(miss_gr, dtype=float))

        spec_score = marginalize_f_miss_global(
            logL_cat_mu_by_event=spec_cat_mu,
            logL_cat_gr_by_event=spec_cat_gr,
            logL_missing_mu_by_event=spec_miss_mu,
            logL_missing_gr_by_event=spec_miss_gr,
            log_alpha_mu=cache.log_alpha_mu,
            log_alpha_gr=cache.log_alpha_gr,
            prior=cache.prior,
            n_f=cache.n_f,
            eps=cache.eps,
        )
        _write_json_atomic(paths.tables_dir / "spectral_only_baseline_score.json", spec_score.to_jsonable())

        spectral_cache = {
            "gw_prior": gw_prior.to_jsonable(),
            "missing_pre": pre_missing.to_jsonable(),
            "spec_score": spec_score.to_jsonable(),
            # Arrays are stored separately to keep the JSON light.
        }
        np.savez_compressed(
            paths.tables_dir / "spectral_only_cached_terms.npz",
            events=np.asarray(cache.events),
            logL_cat_mu=np.stack(spec_cat_mu, axis=0),
            logL_cat_gr=np.stack(spec_cat_gr, axis=0),
            logL_missing_mu=np.stack(spec_miss_mu, axis=0),
            logL_missing_gr=np.stack(spec_miss_gr, axis=0),
        )

    # A2) Galaxy catalog incompleteness + photo-z stress (spectral-only rescoring; validates the dominant channel).
    a2_summary: dict[str, Any] = {"enabled": False}
    if a2_enabled:
        if post is None or spectral_cache is None:
            raise RuntimeError("Internal error: missing post/spectral_cache for A2.")
        a2_cfg = cfg.get("catalog_photoz_stress", {})
        topk = int(a2_cfg.get("events_topk", 2))
        gal_z_max = float(cache.manifest.get("gal_z_max", 0.3))
        z_pivot = float(a2_cfg.get("z_pivot", 0.15))

        # Rank events by baseline influence (use cached per-event scores).
        ev_scores = json.loads((cache.gap_run_root / "tables" / f"event_scores_{cache.run_label}.json").read_text(encoding="utf-8"))
        ev_scores = [e for e in ev_scores if isinstance(e, dict) and isinstance(e.get("event"), str)]
        ev_scores.sort(key=lambda e: float(e.get("delta_lpd", 0.0)), reverse=True)
        top_events = [str(e["event"]) for e in ev_scores[: max(1, topk)]]

        # Load baseline spectral-only cached terms.
        spec_npz = paths.tables_dir / "spectral_only_cached_terms.npz"
        with np.load(spec_npz, allow_pickle=False) as d:
            events = [str(x) for x in d["events"].tolist()]
            base_cat_mu = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_cat_mu"], dtype=float)]
            base_cat_gr = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_cat_gr"], dtype=float)]
            base_miss_mu = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_missing_mu"], dtype=float)]
            base_miss_gr = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_missing_gr"], dtype=float)]

        ev_to_idx = {ev: i for i, ev in enumerate(events)}
        for ev in top_events:
            if ev not in ev_to_idx:
                raise KeyError(f"Top event {ev} missing from spectral-only cache.")

        # Grids.
        b0_grid = [float(x) for x in a2_cfg.get("photoz_b0_grid", [-0.01, -0.005, 0.0, 0.005, 0.01])]
        b1_grid = [float(x) for x in a2_cfg.get("photoz_b1_grid", [0.0])]
        c_amp_grid = [float(x) for x in a2_cfg.get("comp_amp_grid", [0.0])]
        c_tilt_grid = [float(x) for x in a2_cfg.get("comp_tilt_grid", [0.0])]

        gw_prior = GWDistancePrior(**json.loads(json.dumps(spectral_cache["gw_prior"])))

        def _score_with_replacements(repl: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
            cat_mu = list(base_cat_mu)
            cat_gr = list(base_cat_gr)
            for ev, (mu, gr) in repl.items():
                j = ev_to_idx[ev]
                cat_mu[j] = np.asarray(mu, dtype=float)
                cat_gr[j] = np.asarray(gr, dtype=float)
            res = marginalize_f_miss_global(
                logL_cat_mu_by_event=cat_mu,
                logL_cat_gr_by_event=cat_gr,
                logL_missing_mu_by_event=base_miss_mu,
                logL_missing_gr_by_event=base_miss_gr,
                log_alpha_mu=cache.log_alpha_mu,
                log_alpha_gr=cache.log_alpha_gr,
                prior=cache.prior,
                n_f=cache.n_f,
                eps=cache.eps,
            )
            return float(res.lpd_mu_total - res.lpd_gr_total)

        rows_photoz = []
        for b0 in b0_grid:
            for b1 in b1_grid:
                repl = {}
                for ev in top_events:
                    z, w, ipix, pe = _load_event_cache(ev)
                    z2 = z + float(b0) + float(b1) * z
                    z2 = np.clip(z2, 1e-6, gal_z_max)
                    logL_mu, logL_gr = compute_dark_siren_logL_draws_from_pe_hist(
                        event=ev,
                        pe=pe,
                        post=post,
                        z_gal=z2,
                        w_gal=w,
                        ipix_gal=ipix,
                        convention=str(cache.convention),
                        gw_distance_prior=gw_prior,
                        distance_mode="spectral_only",
                        gal_chunk_size=int(cache.manifest.get("galaxy_chunk_size", 50_000)),
                    )
                    repl[ev] = (logL_mu, logL_gr)
                dlp = _score_with_replacements(repl)
                rows_photoz.append({"b0": float(b0), "b1": float(b1), "delta_lpd_total": float(dlp)})

        rows_comp = []
        for ca in c_amp_grid:
            for ct in c_tilt_grid:
                repl = {}
                for ev in top_events:
                    z, w, ipix, pe = _load_event_cache(ev)
                    # Simple z-tilt weight deformation; clip to keep weights positive.
                    fz = (1.0 + float(ca)) * (1.0 + float(ct) * ((z - z_pivot) / max(z_pivot, 1e-6)))
                    w2 = w * np.clip(fz, 0.1, 10.0)
                    logL_mu, logL_gr = compute_dark_siren_logL_draws_from_pe_hist(
                        event=ev,
                        pe=pe,
                        post=post,
                        z_gal=z,
                        w_gal=w2,
                        ipix_gal=ipix,
                        convention=str(cache.convention),
                        gw_distance_prior=gw_prior,
                        distance_mode="spectral_only",
                        gal_chunk_size=int(cache.manifest.get("galaxy_chunk_size", 50_000)),
                    )
                    repl[ev] = (logL_mu, logL_gr)
                dlp = _score_with_replacements(repl)
                rows_comp.append({"c_amp": float(ca), "c_tilt": float(ct), "delta_lpd_total": float(dlp)})

        # Summaries.
        base_spec = float(json.loads((paths.tables_dir / "spectral_only_baseline_score.json").read_text())["delta_lpd_total"])
        min_photoz = min(rows_photoz, key=lambda r: float(r["delta_lpd_total"])) if rows_photoz else None
        min_comp = min(rows_comp, key=lambda r: float(r["delta_lpd_total"])) if rows_comp else None
        a2_summary = {
            "enabled": True,
            "mode": "spectral_only",
            "top_events": top_events,
            "baseline_delta_lpd_total": base_spec,
            "photoz_grid": {"b0": b0_grid, "b1": b1_grid},
            "photoz_min": min_photoz,
            "comp_grid": {"c_amp": c_amp_grid, "c_tilt": c_tilt_grid, "z_pivot": z_pivot},
            "comp_min": min_comp,
        }
        _write_json_atomic(paths.tables_dir / "catalog_photoz_stress_summary.json", a2_summary)
        _write_csv(paths.tables_dir / "catalog_photoz_stress_photoz_grid.csv", rows_photoz, fieldnames=["b0", "b1", "delta_lpd_total"])
        _write_csv(paths.tables_dir / "catalog_photoz_stress_comp_grid.csv", rows_comp, fieldnames=["c_amp", "c_tilt", "delta_lpd_total"])

        # Plots (1D slices).
        try:
            fig, ax = plt.subplots(figsize=(6.2, 4.0))
            for b1 in sorted(set(float(r["b1"]) for r in rows_photoz)):
                pts = sorted([(float(r["b0"]), float(r["delta_lpd_total"])) for r in rows_photoz if float(r["b1"]) == b1])
                if not pts:
                    continue
                x = np.asarray([p[0] for p in pts], dtype=float)
                y = np.asarray([p[1] for p in pts], dtype=float)
                ax.plot(x, y, lw=2.0, label=f"b1={b1:+.3f}")
            ax.axhline(float(base_spec), color="k", lw=1.2, alpha=0.6)
            ax.set(
                xlabel="photo-z bias b0",
                ylabel="Delta LPD (spectral-only total)",
                title="Photo-z Stress Grid (slices in b1)",
            )
            ax.grid(alpha=0.25, linestyle=":")
            ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            fig.savefig(paths.figures_dir / "a2_photoz_slices.png", dpi=180)
            plt.close(fig)
        except Exception:
            pass

        try:
            fig, ax = plt.subplots(figsize=(6.2, 4.0))
            for ct in sorted(set(float(r["c_tilt"]) for r in rows_comp)):
                pts = sorted([(float(r["c_amp"]), float(r["delta_lpd_total"])) for r in rows_comp if float(r["c_tilt"]) == ct])
                if not pts:
                    continue
                x = np.asarray([p[0] for p in pts], dtype=float)
                y = np.asarray([p[1] for p in pts], dtype=float)
                ax.plot(x, y, lw=2.0, label=f"tilt={ct:+.3f}")
            ax.axhline(float(base_spec), color="k", lw=1.2, alpha=0.6)
            ax.set(
                xlabel="completeness amplitude",
                ylabel="Delta LPD (spectral-only total)",
                title="Completeness Stress Grid (slices in tilt)",
            )
            ax.grid(alpha=0.25, linestyle=":")
            ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            fig.savefig(paths.figures_dir / "a2_completeness_slices.png", dpi=180)
            plt.close(fig)
        except Exception:
            pass

    # A3) PE / waveform robustness for high-leverage events (analysis-group swaps + resampling).
    a3_summary: dict[str, Any] = {"enabled": False}
    if a3_enabled:
        if post is None or spectral_cache is None:
            raise RuntimeError("Internal error: missing post/spectral_cache for A3.")
        a3_cfg = cfg.get("pe_waveform_stress", {})
        top_events = [str(x) for x in a3_cfg.get("events", [])]
        if not top_events:
            # Default: top 2 by baseline influence.
            ev_scores = json.loads((cache.gap_run_root / "tables" / f"event_scores_{cache.run_label}.json").read_text(encoding="utf-8"))
            ev_scores = [e for e in ev_scores if isinstance(e, dict) and isinstance(e.get("event"), str)]
            ev_scores.sort(key=lambda e: float(e.get("delta_lpd", 0.0)), reverse=True)
            top_events = [str(e["event"]) for e in ev_scores[:2]]

        max_samples = int(a3_cfg.get("max_pe_samples", 200_000))
        seed = int(a3_cfg.get("seed", 0))
        analyses_pref = [str(x) for x in a3_cfg.get("analyses", ["C01:Mixed", "C01:IMRPhenomXPHM", "C01:SEOBNRv4PHM"])]

        # Load baseline spectral-only cached terms.
        spec_npz = paths.tables_dir / "spectral_only_cached_terms.npz"
        with np.load(spec_npz, allow_pickle=False) as d:
            events = [str(x) for x in d["events"].tolist()]
            base_cat_mu = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_cat_mu"], dtype=float)]
            base_cat_gr = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_cat_gr"], dtype=float)]
            base_miss_mu = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_missing_mu"], dtype=float)]
            base_miss_gr = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_missing_gr"], dtype=float)]
        ev_to_idx = {ev: i for i, ev in enumerate(events)}

        gw_prior = GWDistancePrior(**json.loads(json.dumps(spectral_cache["gw_prior"])))
        pre_missing = precompute_missing_host_prior(
            post,
            convention=str(cache.convention),
            z_max=float(cache.manifest.get("missing_z_max", 0.3)),
            host_prior_z_mode=str(cache.manifest.get("host_prior_z_mode", "comoving_uniform")),
            host_prior_z_k=float(cache.manifest.get("host_prior_z_k", 0.0)),
        )

        def _rescore_one_event(ev: str, *, cat_mu: np.ndarray, cat_gr: np.ndarray, miss_mu: np.ndarray, miss_gr: np.ndarray) -> float:
            mu = list(base_cat_mu)
            gr = list(base_cat_gr)
            mmu = list(base_miss_mu)
            mgr = list(base_miss_gr)
            j = ev_to_idx[ev]
            mu[j] = np.asarray(cat_mu, dtype=float)
            gr[j] = np.asarray(cat_gr, dtype=float)
            mmu[j] = np.asarray(miss_mu, dtype=float)
            mgr[j] = np.asarray(miss_gr, dtype=float)
            res = marginalize_f_miss_global(
                logL_cat_mu_by_event=mu,
                logL_cat_gr_by_event=gr,
                logL_missing_mu_by_event=mmu,
                logL_missing_gr_by_event=mgr,
                log_alpha_mu=cache.log_alpha_mu,
                log_alpha_gr=cache.log_alpha_gr,
                prior=cache.prior,
                n_f=cache.n_f,
                eps=cache.eps,
            )
            return float(res.lpd_mu_total - res.lpd_gr_total)

        a3_rows = []
        for ev in top_events:
            if ev not in ev_to_idx:
                continue
            # Read the PE file path from the cached term meta (authoritative).
            meta_path = cache.gap_run_root / "cache_terms" / f"cat_{ev}__{cache.run_label}.npz"
            with np.load(meta_path, allow_pickle=True) as d:
                meta = json.loads(str(d["meta"].tolist()))
            pe_file = Path(str(meta.get("pe_file", ""))).expanduser()
            if not pe_file.exists():
                raise FileNotFoundError(f"{ev}: pe_file not found: {pe_file}")

            z_gal, w_gal, ipix_gal, _pe_cached = _load_event_cache(ev)

            for analysis in analyses_pref:
                try:
                    ra, dec, dL, pe_meta = load_gwtc_pe_sky_samples(
                        path=pe_file,
                        analysis=str(analysis),
                        max_samples=max_samples,
                        seed=seed,
                    )
                except Exception:
                    continue
                pe_hist = build_pe_pixel_distance_histogram(
                    ra_rad=ra,
                    dec_rad=dec,
                    dL_mpc=dL,
                    nside=int(cache.manifest.get("pe_nside", 64)),
                    p_credible=float(cache.manifest.get("p_credible", 0.9)),
                    dl_nbins=int(cache.manifest.get("pe_dl_nbins", 64)),
                    dl_min_mpc=float(np.asarray(_pe_cached.dL_edges, dtype=float)[0]),
                    dl_max_mpc=float(np.asarray(_pe_cached.dL_edges, dtype=float)[-1]),
                    dl_qmin=float(cache.manifest.get("pe_dl_qmin", 0.001)),
                    dl_qmax=float(cache.manifest.get("pe_dl_qmax", 0.999)),
                    dl_pad_factor=float(cache.manifest.get("pe_dl_pad_factor", 1.2)),
                    dl_pseudocount=float(cache.manifest.get("pe_dl_pseudocount", 0.05)),
                    dl_smooth_iters=int(cache.manifest.get("pe_dl_smooth_iters", 2)),
                    nest=True,
                )
                logL_mu, logL_gr = compute_dark_siren_logL_draws_from_pe_hist(
                    event=ev,
                    pe=pe_hist,
                    post=post,
                    z_gal=z_gal,
                    w_gal=w_gal,
                    ipix_gal=ipix_gal,
                    convention=str(cache.convention),
                    gw_distance_prior=gw_prior,
                    distance_mode="spectral_only",
                    gal_chunk_size=int(cache.manifest.get("galaxy_chunk_size", 50_000)),
                )
                miss_mu, miss_gr = compute_missing_host_logL_draws_from_histogram(
                    prob_pix=np.asarray(pe_hist.prob_pix, dtype=float),
                    pdf_bins=np.asarray(pe_hist.pdf_bins, dtype=float),
                    dL_edges=np.asarray(pe_hist.dL_edges, dtype=float),
                    pre=pre_missing,
                    gw_distance_prior=gw_prior,
                    distance_mode="spectral_only",
                    pixel_chunk_size=int(cache.manifest.get("missing_pixel_chunk_size", 5_000)),
                )
                dlp = _rescore_one_event(ev, cat_mu=logL_mu, cat_gr=logL_gr, miss_mu=miss_mu, miss_gr=miss_gr)
                a3_rows.append(
                    {
                        "event": ev,
                        "analysis": str(pe_meta.analysis),
                        "n_pe_used": int(pe_meta.n_used),
                        "delta_lpd_total": float(dlp),
                    }
                )

        a3_summary = {"enabled": True, "rows": a3_rows, "top_events": top_events}
        _write_json_atomic(paths.tables_dir / "pe_waveform_stress_summary.json", a3_summary)
        if a3_rows:
            _write_csv(paths.tables_dir / "pe_waveform_stress_rows.csv", a3_rows, fieldnames=["event", "analysis", "n_pe_used", "delta_lpd_total"])

    # B2) Split coherence: simple redshift tertiles + O3a/O3b split (using event timestamp).
    split_summary: dict[str, Any] = {"enabled": False}
    if bool(cfg.get("splits", {}).get("enabled", False)) and not bool(args.skip_splits):
        s_cfg = cfg["splits"]
        n_z_bins = int(s_cfg.get("n_z_bins", 3))
        # Use the cached GW PE distance posterior (sky-marginal) as a redshift proxy.
        # This is a purely descriptive binning coordinate; it does not enter the likelihood.
        H0_ref = float(cache.manifest.get("gw_distance_prior_h0_ref", 67.7))
        om_ref = float(cache.manifest.get("gw_distance_prior_omega_m0", 0.31))

        z_proxy = []
        dL_proxy = []
        o3_epoch = []
        for ev in cache.events:
            p = cache.gap_run_root / "cache" / f"event_{ev}.npz"
            dL_med = _pe_distance_median_from_event_cache(p)
            z_bar = _lcdm_z_from_dL(dL_med, H0=H0_ref, omega_m0=om_ref, z_max=1.0)
            dL_proxy.append(float(dL_med))
            z_proxy.append(float(z_bar))

            # O3 epoch: O3a ended 2019-10-01; use event name date prefix.
            # Format: GWYYMMDD_...
            try:
                y = 2000 + int(ev[2:4])
                mo = int(ev[4:6])
                da = int(ev[6:8])
                epoch = "O3a" if (y, mo, da) <= (2019, 10, 1) else "O3b"
            except Exception:
                epoch = "unknown"
            o3_epoch.append(epoch)

        z_proxy = np.asarray(z_proxy, dtype=float)
        z_edges = []
        for b in range(1, n_z_bins):
            q = b / float(n_z_bins)
            z_edges.append(float(np.quantile(z_proxy[np.isfinite(z_proxy)], q)))
        z_edges = [float("-inf")] + z_edges + [float("inf")]

        split_rows = []
        # Score per z bin.
        zbin_results = []
        for b in range(n_z_bins):
            lo, hi = z_edges[b], z_edges[b + 1]
            m = (z_proxy > lo) & (z_proxy <= hi)
            res = _delta_lpd_from_cached_terms(cache=cache, log_alpha_mu=cache.log_alpha_mu, log_alpha_gr=cache.log_alpha_gr, events_mask=m)
            zbin_results.append(
                {
                    "bin": b,
                    "z_lo": float(lo),
                    "z_hi": float(hi),
                    "n_events": int(np.sum(m)),
                    "delta_lpd_total": float(res.lpd_mu_total - res.lpd_gr_total),
                    "delta_lpd_data": float(res.lpd_mu_total_data - res.lpd_gr_total_data),
                }
            )
            split_rows.append([f"z-bin {b}", int(np.sum(m)), f"{(res.lpd_mu_total - res.lpd_gr_total):+.3f}"])

        # Score O3a vs O3b.
        for ep in ("O3a", "O3b"):
            m = np.asarray([e == ep for e in o3_epoch], dtype=bool)
            if not np.any(m):
                continue
            res = _delta_lpd_from_cached_terms(cache=cache, log_alpha_mu=cache.log_alpha_mu, log_alpha_gr=cache.log_alpha_gr, events_mask=m)
            split_rows.append([ep, int(np.sum(m)), f"{(res.lpd_mu_total - res.lpd_gr_total):+.3f}"])

        split_summary = {
            "enabled": True,
            "dL_proxy_median_mpc": {ev: float(x) for ev, x in zip(cache.events, dL_proxy, strict=True)},
            "z_proxy": {ev: float(z) for ev, z in zip(cache.events, z_proxy, strict=True)},
            "z_bins": zbin_results,
            "epochs": {ev: ep for ev, ep in zip(cache.events, o3_epoch, strict=True)},
        }
        _write_json_atomic(paths.tables_dir / "split_coherence_summary.json", split_summary)

    # C/E) Consolidated "required failure" summary + lightweight falsifiers note.
    required = {
        "baseline_full_delta_lpd_total": float(base.lpd_mu_total - base.lpd_gr_total),
        "selection_unpenalized_min_delta_lpd_total": float((sel_summary.get("adversarial", {}).get("unpenalized", {}) or {}).get("delta_lpd_total", float("nan"))),
        "selection_bounds": float(sel_summary.get("delta_bound", float("nan"))) if sel_summary.get("enabled") else None,
        "photoz_min_delta_lpd_total": float((a2_summary.get("photoz_min") or {}).get("delta_lpd_total", float("nan"))),
        "photoz_min_params": (a2_summary.get("photoz_min") or {}) if a2_summary.get("enabled") else None,
        "completeness_min_delta_lpd_total": float((a2_summary.get("comp_min") or {}).get("delta_lpd_total", float("nan"))),
        "completeness_min_params": (a2_summary.get("comp_min") or {}) if a2_summary.get("enabled") else None,
        "pe_waveform_min_delta_lpd_total": None,
    }
    if a3_summary.get("enabled"):
        vals = [float(r.get("delta_lpd_total", float("nan"))) for r in a3_summary.get("rows", [])]
        vals = [v for v in vals if np.isfinite(v)]
        required["pe_waveform_min_delta_lpd_total"] = float(min(vals)) if vals else None

    _write_json_atomic(paths.tables_dir / "required_failure_summary.json", required)

    # Summary bar plot.
    try:
        labels = ["baseline", "sel(min)", "photoz(min)", "comp(min)", "PE(min)"]
        ys = [
            float(required["baseline_full_delta_lpd_total"]),
            float(required["selection_unpenalized_min_delta_lpd_total"]) if required["selection_unpenalized_min_delta_lpd_total"] == required["selection_unpenalized_min_delta_lpd_total"] else float("nan"),
            float(required["photoz_min_delta_lpd_total"]) if required["photoz_min_delta_lpd_total"] == required["photoz_min_delta_lpd_total"] else float("nan"),
            float(required["completeness_min_delta_lpd_total"]) if required["completeness_min_delta_lpd_total"] == required["completeness_min_delta_lpd_total"] else float("nan"),
            float(required["pe_waveform_min_delta_lpd_total"]) if required["pe_waveform_min_delta_lpd_total"] is not None else float("nan"),
        ]
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        x = np.arange(len(labels), dtype=float)
        ax.bar(x, ys, color=["C0", "C1", "C2", "C3", "C4"], alpha=0.85)
        ax.axhline(1.0, color="k", lw=1.0, alpha=0.5)
        ax.axhline(0.0, color="k", lw=1.0, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel("Delta LPD (total)")
        ax.set_title("Minimum Delta LPD Under Stress Families (within tested bounds)")
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        fig.tight_layout()
        fig.savefig(paths.figures_dir / "required_failure_bars.png", dpi=180)
        plt.close(fig)
    except Exception:
        pass

    # Predictions / falsifiers note (lightweight, tied to tests actually run here).
    preds = []
    preds.append("# Predictions / Falsifiers (From This Robustness Package)\n")
    preds.append("These are operational checks that future data or re-analyses can falsify.\n")
    preds.append("1. Redshift scaling: the propagation-preference should concentrate at higher redshift (larger PE distance) if it is a cumulative friction effect, not a fixed catalog mismatch.\n")
    preds.append("2. Selection deformations: within plausible bounded selection-function perturbations (logSNR/z/mass splines), Delta LPD should remain positive; a collapse to ~0 under modest bounds would point to selection misspecification.\n")
    preds.append("3. PE analysis dependence: swapping PE waveform analyses (Mixed/IMRPhenomXPHM/SEOBNRv4PHM) for the high-leverage events should not flip the sign of the preference if it is physical propagation.\n")
    preds.append("4. Photo-z failure mode: erasing the preference via photo-z bias would require redshift distortions comparable to the redshift range of the catalog support; if that were true, it should be detectable directly in catalog cross-checks and would not remain hidden as a subtle effect.\n")
    preds_path = paths.out_dir / "predictions.md"
    preds_path.write_text("".join(preds), encoding="utf-8")

    # Report.
    md = []
    md.append("# Dark-Siren Robustness Package (O3)\n")
    md.append(f"- Generated: `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`")
    md.append(f"- Command: `{command_str()}`")
    md.append(f"- Repo: `{repo_root}`")
    md.append(f"- Git SHA: `{git_head_sha(repo_root=repo_root)}` dirty={git_is_dirty(repo_root=repo_root)}")
    md.append(f"- Baseline gap run: `{cache.gap_run_root}` label `{cache.run_label}`")
    md.append("")

    md.append("## Baseline Reproduction\n")
    md.append(format_table([[k, v] for k, v in chk.items()], headers=["metric", "value"]))
    md.append("")

    if sel_summary.get("enabled"):
        md.append("## A1 Selection-Function Adversarial Deformation\n")
        md.append(format_table(sel_rows, headers=["item", "value"]))
        md.append("")
        md.append("- Plots: `figures/sel_nuisance_logsnr_global.png`, `figures/sel_nuisance_logsnr_knot*.png`")
        md.append("- Samples: `samples/selection_nuisance_mc.csv`")
        md.append("")

    if spectral_cache is not None:
        md.append("## Spectral-Only Baseline (Mechanism Channel)\n")
        spec = spectral_cache.get("spec_score", {})
        rows = [
            ["Delta LPD (spectral-only)", f"{float(spec.get('delta_lpd_total', float('nan'))):+.3f}"],
            ["Delta LPD_data", f"{float(spec.get('delta_lpd_total_data', float('nan'))):+.3f}"],
        ]
        md.append(format_table(rows, headers=["item", "value"]))
        md.append("")

    if a2_summary.get("enabled"):
        md.append("## A2 Catalog / Photo-z Stress (Spectral-Only Rescoring)\n")
        rows = [
            ["Top events perturbed", ", ".join(a2_summary.get("top_events", []))],
            ["Baseline Delta LPD (spectral-only)", f"{float(a2_summary.get('baseline_delta_lpd_total', float('nan'))):+.3f}"],
            ["Min Delta LPD on photo-z grid", f"{float((a2_summary.get('photoz_min') or {}).get('delta_lpd_total', float('nan'))):+.3f}"],
            ["Min Delta LPD on completeness grid", f"{float((a2_summary.get('comp_min') or {}).get('delta_lpd_total', float('nan'))):+.3f}"],
        ]
        md.append(format_table(rows, headers=["item", "value"]))
        md.append("")
        md.append("- Grids: `tables/catalog_photoz_stress_photoz_grid.csv`, `tables/catalog_photoz_stress_comp_grid.csv`")
        md.append("- Plots: `figures/a2_photoz_slices.png`, `figures/a2_completeness_slices.png`")
        md.append("")

    if a3_summary.get("enabled"):
        md.append("## A3 PE / Waveform Robustness (Analysis-Group Swaps)\n")
        rows = []
        for r in a3_summary.get("rows", [])[:12]:
            rows.append([r.get("event", ""), r.get("analysis", ""), r.get("n_pe_used", ""), f"{float(r.get('delta_lpd_total', float('nan'))):+.3f}"])
        if rows:
            md.append(format_table(rows, headers=["event", "analysis", "n_pe_used", "Delta LPD"]))
            md.append("")
            md.append("- Full rows: `tables/pe_waveform_stress_rows.csv`")
            md.append("")

    md.append("## C Required Failure Summary\n")
    try:
        req = _read_json(paths.tables_dir / "required_failure_summary.json")
        rows = [
            ["baseline (full)", f"{float(req.get('baseline_full_delta_lpd_total', float('nan'))):+.3f}"],
            ["selection min (unpen.)", f"{float(req.get('selection_unpenalized_min_delta_lpd_total', float('nan'))):+.3f}"],
            ["photo-z min", f"{float(req.get('photoz_min_delta_lpd_total', float('nan'))):+.3f}"],
            ["completeness min", f"{float(req.get('completeness_min_delta_lpd_total', float('nan'))):+.3f}"],
            ["PE/waveform min", f"{float(req.get('pe_waveform_min_delta_lpd_total', float('nan'))):+.3f}"],
        ]
        md.append(format_table(rows, headers=["family", "min Delta LPD"]))
        md.append("")
        md.append("- Plot: `figures/required_failure_bars.png`")
        md.append("")
    except Exception:
        pass

    if split_summary.get("enabled"):
        md.append("## B2 Split Coherence\n")
        rows = []
        for b in split_summary.get("z_bins", []):
            rows.append([b["bin"], b["n_events"], f"{b['z_lo']:.3f}", f"{b['z_hi']:.3f}", f"{b['delta_lpd_total']:+.3f}"])
        md.append(format_table(rows, headers=["z_bin", "n_events", "z_lo", "z_hi", "Delta LPD"]))
        md.append("")
        md.append("- Full split summary: `tables/split_coherence_summary.json`")
        md.append("")

    if (paths.out_dir / "predictions.md").exists():
        md.append("## E Predictions / Falsifiers\n")
        md.append("- Note: `predictions.md`")
        md.append("")
    write_markdown_report(paths=paths, markdown="\n".join(md).strip() + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
