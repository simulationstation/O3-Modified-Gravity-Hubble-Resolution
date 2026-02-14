#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

# Allow running without `pip install -e .` by adding `src/` to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists():
    sys.path.insert(0, str(_SRC_ROOT))

from entropy_horizon_recon.dark_siren_gap_lpd import BetaPrior, MarginalizedFMissResult, marginalize_f_miss_global
from entropy_horizon_recon.dark_sirens_incompleteness import precompute_missing_host_prior
from entropy_horizon_recon.dark_sirens_pe import PePixelDistanceHistogram
from entropy_horizon_recon.gw_distance_priors import GWDistancePrior
from entropy_horizon_recon.sirens import MuForwardPosterior, load_mu_forward_posterior, predict_dL_em, predict_r_gw_em


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _set_thread_env(n: int) -> None:
    n = int(max(1, n))
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(n))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(n))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _events_from_cache_terms(cache_terms_dir: Path, *, run_label: str) -> list[str]:
    # cat_<EVENT>__<LABEL>.npz
    out: list[str] = []
    for p in sorted(cache_terms_dir.glob(f"cat_*__{run_label}.npz")):
        stem = p.stem
        if not stem.startswith("cat_"):
            continue
        ev = stem[len("cat_") :].split(f"__{run_label}", 1)[0]
        if ev:
            out.append(ev)
    if not out:
        raise FileNotFoundError(f"No cached cat terms found under {cache_terms_dir} for run_label={run_label}")
    return out


@dataclass(frozen=True)
class GapRunCache:
    gap_run_root: Path
    run_label: str
    events: list[str]
    log_alpha_mu: np.ndarray
    log_alpha_gr: np.ndarray
    prior: BetaPrior
    n_f: int
    eps: float
    convention: str
    draw_idx: list[int]
    manifest: dict[str, Any]
    baseline_summary: dict[str, Any]


def _load_gap_run_cache(*, gap_run_root: Path, run_label: str) -> GapRunCache:
    gap_run_root = gap_run_root.expanduser().resolve()
    if not gap_run_root.exists():
        raise FileNotFoundError(gap_run_root)

    cache_terms_dir = gap_run_root / "cache_terms"
    tables_dir = gap_run_root / "tables"

    baseline_summary = _read_json(gap_run_root / f"summary_{run_label}.json")
    manifest = _read_json(gap_run_root / "manifest.json")

    events = _events_from_cache_terms(cache_terms_dir, run_label=run_label)

    sel_npz = tables_dir / f"selection_alpha_{run_label}.npz"
    with np.load(sel_npz, allow_pickle=True) as d:
        log_alpha_mu = np.asarray(d["log_alpha_mu"], dtype=float)
        log_alpha_gr = np.asarray(d["log_alpha_gr"], dtype=float)

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
        raise ValueError("Missing draw_idx in baseline summary; cannot subset recon posterior.")

    return GapRunCache(
        gap_run_root=gap_run_root,
        run_label=str(run_label),
        events=events,
        log_alpha_mu=log_alpha_mu,
        log_alpha_gr=log_alpha_gr,
        prior=prior,
        n_f=n_f,
        eps=eps,
        convention=convention,
        draw_idx=draw_idx,
        manifest=manifest,
        baseline_summary=baseline_summary,
    )


def _load_event_cache(
    *, gap_run_root: Path, ev: str, manifest: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PePixelDistanceHistogram, dict[str, Any]]:
    p = gap_run_root / "cache" / f"event_{ev}.npz"
    with np.load(p, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"].tolist())) if "meta" in d.files else {}
        z = np.asarray(d["z"], dtype=float)
        w = np.asarray(d["w"], dtype=float)
        ipix = np.asarray(d["ipix"], dtype=np.int64)
        pe = PePixelDistanceHistogram(
            nside=int(manifest.get("pe_nside", 64)),
            nest=True,
            p_credible=float(manifest.get("p_credible", 0.9)),
            pix_sel=np.asarray(d["pe_pix_sel"], dtype=np.int64),
            prob_pix=np.asarray(d["pe_prob_pix"], dtype=float),
            dL_edges=np.asarray(d["pe_dL_edges"], dtype=float),
            pdf_bins=np.asarray(d["pe_pdf_bins"], dtype=float),
        )
    return z, w, ipix, pe, meta


def _pe_sky_marginal_pdf_1d(pe: PePixelDistanceHistogram) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(pe.dL_edges, dtype=float)
    widths = np.diff(edges)
    if widths.size <= 0:
        raise ValueError("Invalid PE edges.")
    p_pix = np.asarray(pe.prob_pix, dtype=float)
    pdf_bins = np.asarray(pe.pdf_bins, dtype=float)
    p_sum = float(np.sum(p_pix))
    if not (np.isfinite(p_sum) and p_sum > 0.0):
        raise ValueError("Invalid prob_pix sum.")
    if int(pdf_bins.shape[0]) == int(p_pix.size):
        pdf_1d = np.sum(p_pix.reshape((-1, 1)) * pdf_bins, axis=0) / p_sum
    elif int(pdf_bins.shape[0]) == 1:
        pdf_1d = np.asarray(pdf_bins[0], dtype=float)
    else:
        raise ValueError("Incompatible pdf_bins shape.")
    pdf_1d = np.clip(np.asarray(pdf_1d, dtype=float), 0.0, np.inf)
    norm = float(np.sum(pdf_1d * widths))
    if not (np.isfinite(norm) and norm > 0.0):
        raise ValueError("Invalid spectral-only pdf normalization.")
    return edges, pdf_1d / norm


def _norm_pdf(edges: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    pdf = np.asarray(pdf, dtype=float)
    widths = np.diff(edges)
    norm = float(np.sum(pdf * widths))
    if not (np.isfinite(norm) and norm > 0.0):
        raise ValueError("Cannot normalize pdf.")
    return pdf / norm


def _delta_lpd_from_terms(
    *,
    logL_cat_mu_by_event: list[np.ndarray],
    logL_cat_gr_by_event: list[np.ndarray],
    logL_missing_mu_by_event: list[np.ndarray],
    logL_missing_gr_by_event: list[np.ndarray],
    log_alpha_mu: np.ndarray,
    log_alpha_gr: np.ndarray,
    prior: BetaPrior,
    n_f: int,
    eps: float,
) -> MarginalizedFMissResult:
    return marginalize_f_miss_global(
        logL_cat_mu_by_event=logL_cat_mu_by_event,
        logL_cat_gr_by_event=logL_cat_gr_by_event,
        logL_missing_mu_by_event=logL_missing_mu_by_event,
        logL_missing_gr_by_event=logL_missing_gr_by_event,
        log_alpha_mu=log_alpha_mu,
        log_alpha_gr=log_alpha_gr,
        prior=prior,
        n_f=int(n_f),
        eps=float(eps),
    )


def _summary_stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "p16": float("nan"), "p50": float("nan"), "p84": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(x)),
        "p16": float(np.quantile(x, 0.16)),
        "p50": float(np.quantile(x, 0.50)),
        "p84": float(np.quantile(x, 0.84)),
        "max": float(np.max(x)),
    }


def _parse_csv_floats(s: str) -> list[float]:
    out: list[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _beta_ab_from_mean_kappa(mean: float, kappa: float) -> tuple[float, float]:
    mean = float(mean)
    kappa = float(kappa)
    if not (0.0 < mean < 1.0):
        raise ValueError("Beta mean must be in (0,1)")
    if not (np.isfinite(kappa) and kappa > 0.0):
        raise ValueError("Beta kappa must be >0")
    a = mean * kappa
    b = (1.0 - mean) * kappa
    if not (np.isfinite(a) and np.isfinite(b) and a > 0.0 and b > 0.0):
        raise ValueError("Invalid Beta parameters from mean/kappa.")
    return float(a), float(b)


def _smooth_pz_hist(z_cent: np.ndarray, *, z0: float, beta: float) -> np.ndarray:
    """Return an unnormalised smooth p(z) weight curve evaluated at bin centres.

    p(z) ∝ z^2 exp(-(z/z0)^beta)
    """
    z_cent = np.asarray(z_cent, dtype=float)
    z0 = float(z0)
    beta = float(beta)
    if not (np.isfinite(z0) and z0 > 0.0):
        raise ValueError("z0 must be >0")
    if not (np.isfinite(beta) and beta > 0.0):
        raise ValueError("beta must be >0")
    w = np.clip(z_cent, 0.0, np.inf) ** 2 * np.exp(-np.clip(z_cent / z0, 0.0, np.inf) ** beta)
    w = np.asarray(w, dtype=float)
    if not np.any(np.isfinite(w) & (w > 0.0)):
        return np.ones_like(z_cent, dtype=float)
    return np.where(np.isfinite(w), w, 0.0)


def _subset_post(post_full: MuForwardPosterior, idx: list[int]) -> MuForwardPosterior:
    ii = np.asarray(idx, dtype=int)
    if ii.size == 0:
        raise ValueError("Empty draw index list.")
    n = int(post_full.H0.size)
    imax = int(np.max(ii))
    if imax >= n:
        raise IndexError(
            "draw_idx references posterior draws beyond the provided recon posterior. "
            f"Need >= {imax+1} draws but recon posterior has n_draws={n}. "
            "Pass the correct --recon-run-dir for this gap run."
        )
    return MuForwardPosterior(
        x_grid=post_full.x_grid,
        logmu_x_samples=post_full.logmu_x_samples[ii],
        z_grid=post_full.z_grid,
        H_samples=post_full.H_samples[ii],
        H0=post_full.H0[ii],
        omega_m0=post_full.omega_m0[ii],
        omega_k0=post_full.omega_k0[ii],
        sigma8_0=post_full.sigma8_0[ii] if post_full.sigma8_0 is not None else None,
    )


def run_suite(
    *,
    cache: GapRunCache,
    recon_run_dir: Path,
    truth: Literal["mg", "gr", "dial"],
    n_rep: int,
    seed: int,
    truth_z_mode: Literal["catalog_hist", "missing_prior", "catalog_miss_mix"],
    cat_mode: Literal["catalog", "toy_uniform_z", "global_pz", "smooth_prior", "event_hist_swap"],
    sigma_floor_frac: float,
    z_hist_nbins: int,
    n_f: int,
    truth_alpha: float = 1.0,
    truth_fmiss: float | None = None,
    truth_catalog_sampler: Literal["weighted", "prob_only", "uniform"] = "weighted",
    smooth_prior_z0: float = 0.10,
    smooth_prior_beta: float = 1.5,
    diag_splits_nrep: int = 0,
) -> dict[str, Any]:
    """Run MG/GR truth injection suite and return sample rows + summary.

    Synthetic PE density is constructed as a *posterior-like* histogram:
      p(dL | d) ∝ L(d | dL) * π_PE(dL)
    using the same analytic gw_prior used by the scoring code as π_PE(dL).
    This ensures the pipeline's prior-removal (divide by π) recovers a likelihood-like shape.
    """
    if truth not in ("mg", "gr", "dial"):
        raise ValueError("truth must be 'mg', 'gr', or 'dial'")
    if truth_z_mode not in ("catalog_hist", "missing_prior", "catalog_miss_mix", "global_pz", "smooth_prior", "event_hist_swap"):
        raise ValueError("truth_z_mode must be catalog_hist/missing_prior/catalog_miss_mix/global_pz/smooth_prior/event_hist_swap")
    if cat_mode not in ("catalog", "toy_uniform_z", "global_pz", "smooth_prior", "event_hist_swap"):
        raise ValueError("cat_mode must be catalog/toy_uniform_z/global_pz/smooth_prior/event_hist_swap")
    if truth_catalog_sampler not in ("weighted", "prob_only", "uniform"):
        raise ValueError("truth_catalog_sampler must be weighted/prob_only/uniform")
    truth_alpha = float(truth_alpha)
    if not (np.isfinite(truth_alpha) and truth_alpha >= 0.0):
        raise ValueError("truth_alpha must be finite and >=0")
    if truth_fmiss is not None:
        truth_fmiss = float(truth_fmiss)
        if not (0.0 <= truth_fmiss <= 1.0):
            raise ValueError("truth_fmiss must be in [0,1]")
    diag_splits_nrep = int(max(0, diag_splits_nrep))

    n_rep = int(n_rep)
    seed = int(seed)
    rng = np.random.default_rng(seed)

    # Load recon posterior draws used by the baseline scoring.
    post_full = load_mu_forward_posterior(recon_run_dir)
    post = _subset_post(post_full, cache.draw_idx)

    z_grid_post = np.asarray(post.z_grid, dtype=float)
    dL_em_grid = predict_dL_em(post, z_eval=z_grid_post)
    _, R_grid = predict_r_gw_em(post, z_eval=z_grid_post, convention=str(cache.convention))

    # Shared priors / missing-host precompute.
    manifest = cache.manifest
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

    # Precompute per-event z histograms and PE edges / scale estimates.
    z_max = float(manifest.get("gal_z_max", 0.3))
    z_edges = np.linspace(0.0, z_max, int(z_hist_nbins) + 1)
    z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])

    ev_weight_hist_raw: dict[str, np.ndarray] = {}
    ev_weight_hist_norm: dict[str, np.ndarray] = {}
    ev_prob_hist_norm: dict[str, np.ndarray] = {}
    ev_hist_sum: dict[str, float] = {}
    ev_pe_edges: dict[str, np.ndarray] = {}
    ev_pe_sigma: dict[str, float] = {}
    ev_host_mu_w: dict[str, np.ndarray] = {}
    ev_host_gr_w: dict[str, np.ndarray] = {}
    ev_sky_area_deg2: dict[str, float] = {}

    from entropy_horizon_recon.dark_sirens_incompleteness import _host_prior_matrix_from_precompute  # noqa: SLF001

    for ev in cache.events:
        z_gal, w_gal, ipix_gal, pe, meta = _load_event_cache(gap_run_root=cache.gap_run_root, ev=ev, manifest=manifest)

        try:
            ev_sky_area_deg2[ev] = float(meta.get("sky_area_deg2", float("nan")))
        except Exception:
            ev_sky_area_deg2[ev] = float("nan")

        # Build z histogram from catalog weights and sky probability (spectral-only grouping).
        npix = int(12 * int(pe.nside) * int(pe.nside))
        pix_to_row = np.full((npix,), -1, dtype=np.int32)
        pix_to_row[np.asarray(pe.pix_sel, dtype=np.int64)] = np.arange(int(pe.pix_sel.size), dtype=np.int32)
        row = pix_to_row[ipix_gal]
        good = (row >= 0) & np.isfinite(z_gal) & (z_gal > 0.0) & np.isfinite(w_gal) & (w_gal > 0.0)
        prob = np.asarray(pe.prob_pix, dtype=float)[row[good]] if np.any(good) else np.asarray([], dtype=float)
        weight = (w_gal[good] * prob) if np.any(good) else np.asarray([], dtype=float)
        hist_raw, _ = np.histogram(np.clip(z_gal[good], 0.0, z_max), bins=z_edges, weights=weight)
        hist_raw = np.asarray(hist_raw, dtype=float)
        s_raw = float(np.sum(hist_raw))
        if not (np.isfinite(s_raw) and s_raw > 0.0):
            hist_raw = np.ones_like(hist_raw, dtype=float)
            s_raw = float(np.sum(hist_raw))
        ev_weight_hist_raw[ev] = hist_raw
        ev_weight_hist_norm[ev] = hist_raw / s_raw
        ev_hist_sum[ev] = float(s_raw)

        # "Uniform galaxy weights" proxy histogram (still respects sky probability).
        hist_prob, _ = np.histogram(np.clip(z_gal[good], 0.0, z_max), bins=z_edges, weights=prob)
        hist_prob = np.asarray(hist_prob, dtype=float)
        s_prob = float(np.sum(hist_prob))
        if not (np.isfinite(s_prob) and s_prob > 0.0):
            hist_prob = np.ones_like(hist_prob, dtype=float)
            s_prob = float(np.sum(hist_prob))
        ev_prob_hist_norm[ev] = hist_prob / s_prob

        edges, pdf_1d = _pe_sky_marginal_pdf_1d(pe)
        ev_pe_edges[ev] = edges
        widths = np.diff(edges)
        dmid = 0.5 * (edges[:-1] + edges[1:])
        m = np.isfinite(pdf_1d) & (pdf_1d > 0.0)
        if np.any(m):
            mean = float(np.sum(pdf_1d[m] * dmid[m] * widths[m]))
            var = float(np.sum(pdf_1d[m] * (dmid[m] - mean) ** 2 * widths[m]))
            sig = float(np.sqrt(max(var, 0.0)))
        else:
            sig = float("nan")
        ev_pe_sigma[ev] = sig

        # Precompute missing-host factors on this event's dL grid.
        host_mu = _host_prior_matrix_from_precompute(pre_missing, dL_grid=dmid, model="mu", gw_distance_prior=gw_prior)
        host_gr = _host_prior_matrix_from_precompute(pre_missing, dL_grid=dmid, model="gr", gw_distance_prior=gw_prior)
        ev_host_mu_w[ev] = np.asarray(host_mu, dtype=float) * widths.reshape((1, -1))
        ev_host_gr_w[ev] = np.asarray(host_gr, dtype=float) * widths.reshape((1, -1))

    # Precompute distances and 1/pi(dL) factors at the z-histogram bin centers for all draws.
    n_draws = int(dL_em_grid.shape[0])
    dL_em_zc = np.vstack([np.interp(z_cent, z_grid_post, dL_em_grid[j]) for j in range(n_draws)])
    R_zc = np.vstack([np.interp(z_cent, z_grid_post, R_grid[j]) for j in range(n_draws)])
    dL_gw_zc = dL_em_zc * R_zc
    inv_pi_em_zc = np.exp(-gw_prior.log_pi_dL(np.clip(dL_em_zc, 1e-6, np.inf)))
    inv_pi_gw_zc = np.exp(-gw_prior.log_pi_dL(np.clip(dL_gw_zc, 1e-6, np.inf)))

    # Precompute missing-prior p(z) for truth sampling (draw-dependent).
    base_z = np.asarray(pre_missing.base_z, dtype=float)  # (n_draws, n_zmiss)
    z_grid_miss = np.asarray(pre_missing.z_grid, dtype=float)  # (n_zmiss,)

    # Structure-destruction / catalog-removed controls: global 1-point p(z) and a smooth p(z) proxy.
    global_hist_raw = np.zeros((z_cent.size,), dtype=float)
    for ev in cache.events:
        global_hist_raw += np.asarray(ev_weight_hist_raw[ev], dtype=float)
    global_hist_raw = np.where(np.isfinite(global_hist_raw), global_hist_raw, 0.0)
    global_hist_raw = np.clip(global_hist_raw, 0.0, np.inf)
    if not (np.isfinite(np.sum(global_hist_raw)) and float(np.sum(global_hist_raw)) > 0.0):
        global_hist_raw = np.ones_like(global_hist_raw, dtype=float)
    global_hist_norm = global_hist_raw / float(np.sum(global_hist_raw))

    smooth_raw = _smooth_pz_hist(z_cent, z0=float(smooth_prior_z0), beta=float(smooth_prior_beta))
    smooth_raw = np.where(np.isfinite(smooth_raw), smooth_raw, 0.0)
    smooth_raw = np.clip(smooth_raw, 0.0, np.inf)
    if not (np.isfinite(np.sum(smooth_raw)) and float(np.sum(smooth_raw)) > 0.0):
        smooth_raw = np.ones_like(smooth_raw, dtype=float)
    smooth_norm = smooth_raw / float(np.sum(smooth_raw))

    # Deterministic event->event mapping for histogram swap controls.
    swap_map: dict[str, str] | None = None
    if cat_mode == "event_hist_swap" or truth_z_mode == "event_hist_swap":
        swap_rng = np.random.default_rng(int(seed) + 12345)
        perm = [str(x) for x in swap_rng.permutation(np.asarray(cache.events, dtype=object))]
        swap_map = {ev: perm[i] for i, ev in enumerate(cache.events)}

    # Draw indices for truth selection.
    truth_draws = rng.integers(low=0, high=n_draws, size=n_rep)

    rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    sky_area_vals = np.asarray([ev_sky_area_deg2.get(ev, float("nan")) for ev in cache.events], dtype=float)
    sky_area_med = float(np.nanmedian(sky_area_vals)) if np.any(np.isfinite(sky_area_vals)) else float("nan")
    for rep in range(n_rep):
        jtruth = int(truth_draws[rep])
        fmiss_truth_rep: float | None = None
        cat_mu_by_event: list[np.ndarray] = []
        cat_gr_by_event: list[np.ndarray] = []
        miss_mu_by_event: list[np.ndarray] = []
        miss_gr_by_event: list[np.ndarray] = []
        dL_truth_by_event: dict[str, float] = {}

        for ev in cache.events:
            # Sample z_true.
            if truth_z_mode in ("catalog_hist", "catalog_miss_mix", "event_hist_swap"):
                use_missing = False
                if truth_z_mode == "catalog_miss_mix":
                    if fmiss_truth_rep is None:
                        if truth_fmiss is None:
                            a, b = _beta_ab_from_mean_kappa(cache.prior.mean, cache.prior.kappa)
                            fmiss_truth_rep = float(rng.beta(a, b))
                        else:
                            fmiss_truth_rep = float(truth_fmiss)
                    use_missing = bool(float(rng.uniform(0.0, 1.0)) < float(fmiss_truth_rep))

                if not use_missing:
                    if truth_z_mode == "event_hist_swap":
                        assert swap_map is not None
                        ev0 = swap_map[ev]
                        pz = np.asarray(ev_weight_hist_norm[ev0], dtype=float)
                    else:
                        if truth_catalog_sampler == "weighted":
                            pz = np.asarray(ev_weight_hist_norm[ev], dtype=float)
                        elif truth_catalog_sampler == "prob_only":
                            pz = np.asarray(ev_prob_hist_norm[ev], dtype=float)
                        else:
                            pz = np.ones_like(ev_weight_hist_norm[ev], dtype=float) / float(ev_weight_hist_norm[ev].size)
                    bidx = int(rng.choice(np.arange(pz.size), p=pz))
                    z_lo = float(z_edges[bidx])
                    z_hi = float(z_edges[bidx + 1])
                    z_true = float(rng.uniform(z_lo, z_hi))
                else:
                    pz = np.asarray(base_z[jtruth], dtype=float)
                    pz = np.clip(pz, 0.0, np.inf)
                    s = float(np.sum(pz))
                    if not (np.isfinite(s) and s > 0.0):
                        raise RuntimeError("Invalid base_z truth sampler weights.")
                    pz = pz / s
                    k = int(rng.choice(np.arange(pz.size), p=pz))
                    if k == 0:
                        z_lo, z_hi = float(z_grid_miss[0]), float(z_grid_miss[1])
                    elif k >= z_grid_miss.size - 1:
                        z_lo, z_hi = float(z_grid_miss[-2]), float(z_grid_miss[-1])
                    else:
                        z_lo, z_hi = float(z_grid_miss[k - 1]), float(z_grid_miss[k + 1])
                    z_true = float(rng.uniform(min(z_lo, z_hi), max(z_lo, z_hi)))

            elif truth_z_mode == "global_pz":
                pz = np.asarray(global_hist_norm, dtype=float)
                bidx = int(rng.choice(np.arange(pz.size), p=pz))
                z_lo = float(z_edges[bidx])
                z_hi = float(z_edges[bidx + 1])
                z_true = float(rng.uniform(z_lo, z_hi))

            elif truth_z_mode == "smooth_prior":
                pz = np.asarray(smooth_norm, dtype=float)
                bidx = int(rng.choice(np.arange(pz.size), p=pz))
                z_lo = float(z_edges[bidx])
                z_hi = float(z_edges[bidx + 1])
                z_true = float(rng.uniform(z_lo, z_hi))

            else:
                pz = np.asarray(base_z[jtruth], dtype=float)
                pz = np.clip(pz, 0.0, np.inf)
                s = float(np.sum(pz))
                if not (np.isfinite(s) and s > 0.0):
                    raise RuntimeError("Invalid base_z truth sampler weights.")
                pz = pz / s
                k = int(rng.choice(np.arange(pz.size), p=pz))
                if k == 0:
                    z_lo, z_hi = float(z_grid_miss[0]), float(z_grid_miss[1])
                elif k >= z_grid_miss.size - 1:
                    z_lo, z_hi = float(z_grid_miss[-2]), float(z_grid_miss[-1])
                else:
                    z_lo, z_hi = float(z_grid_miss[k - 1]), float(z_grid_miss[k + 1])
                z_true = float(rng.uniform(min(z_lo, z_hi), max(z_lo, z_hi)))

            # Truth distances.
            dL_em_true = float(np.interp(z_true, z_grid_post, dL_em_grid[jtruth]))
            if truth == "gr":
                R_true = 1.0
            else:
                R_base = float(np.interp(z_true, z_grid_post, R_grid[jtruth]))
                R_true = 1.0 + float(truth_alpha) * (R_base - 1.0)
            dL_gw_true = float(dL_em_true * R_true)
            dL_truth_by_event[ev] = float(dL_gw_true)

            # Synthetic *posterior-like* sky-marginal dL density on this event's grid:
            #   p(dL|d) ∝ L(d|dL) * π_PE(dL)
            edges = np.asarray(ev_pe_edges[ev], dtype=float)
            widths = np.diff(edges)
            dmid = 0.5 * (edges[:-1] + edges[1:])
            sig0 = float(ev_pe_sigma[ev])
            sig = float(max(sig0, sigma_floor_frac * dL_gw_true)) if np.isfinite(sig0) else float(max(sigma_floor_frac * dL_gw_true, 1.0))

            L_like = np.exp(-0.5 * ((dmid - dL_gw_true) / max(sig, 1e-6)) ** 2)
            pi = np.exp(gw_prior.log_pi_dL(np.clip(dmid, 1e-6, np.inf)))
            pdf_post = _norm_pdf(edges, np.asarray(L_like, dtype=float) * np.asarray(pi, dtype=float))

            # Catalog z weighting for the scoring term.
            if cat_mode == "catalog":
                w_hist = np.asarray(ev_weight_hist_raw[ev], dtype=float)
            elif cat_mode == "toy_uniform_z":
                w_hist = np.ones_like(z_cent, dtype=float)
            elif cat_mode == "global_pz":
                # Preserve per-event total weight but remove event-specific z structure.
                w_hist = global_hist_norm * float(ev_hist_sum[ev])
            elif cat_mode == "smooth_prior":
                w_hist = smooth_norm * float(ev_hist_sum[ev])
            else:
                assert swap_map is not None
                w_hist = np.asarray(ev_weight_hist_raw[swap_map[ev]], dtype=float)

            def _logL_cat_from_dL(dL: np.ndarray, inv_pi: np.ndarray) -> np.ndarray:
                bidx = np.searchsorted(edges, dL, side="right") - 1
                valid = (bidx >= 0) & (bidx < pdf_post.size) & np.isfinite(dL) & (dL > 0.0)
                bidx = np.clip(bidx, 0, pdf_post.size - 1)
                post_at = pdf_post[bidx]
                post_at = np.where(valid, post_at, 0.0)
                term = post_at * inv_pi  # posterior / prior ≈ likelihood shape (up to const)
                L = np.clip(term @ w_hist, 1e-300, np.inf)
                return np.log(L)

            logL_mu = _logL_cat_from_dL(dL_gw_zc, inv_pi_gw_zc)
            logL_gr = _logL_cat_from_dL(dL_em_zc, inv_pi_em_zc)

            host_mu_w = ev_host_mu_w[ev]
            host_gr_w = ev_host_gr_w[ev]
            Lm = np.clip(host_mu_w @ pdf_post, 1e-300, np.inf)
            Lg = np.clip(host_gr_w @ pdf_post, 1e-300, np.inf)
            miss_mu = np.log(Lm)
            miss_gr = np.log(Lg)

            cat_mu_by_event.append(np.asarray(logL_mu, dtype=float))
            cat_gr_by_event.append(np.asarray(logL_gr, dtype=float))
            miss_mu_by_event.append(np.asarray(miss_mu, dtype=float))
            miss_gr_by_event.append(np.asarray(miss_gr, dtype=float))

        # Optional leverage-structure diagnostics: compute ΔLPD for simple event splits.
        if diag_splits_nrep > 0 and rep < diag_splits_nrep:
            # Distance tertiles by truth dL (per replicate).
            evs = list(cache.events)
            dL_arr = np.asarray([dL_truth_by_event.get(ev, float("nan")) for ev in evs], dtype=float)
            ord_idx = np.argsort(np.where(np.isfinite(dL_arr), dL_arr, np.inf))
            tert = int(max(1, len(evs) // 3))
            groups: dict[str, list[int]] = {
                "dist_low": [int(i) for i in ord_idx[:tert]],
                "dist_mid": [int(i) for i in ord_idx[tert : 2 * tert]],
                "dist_high": [int(i) for i in ord_idx[2 * tert :]],
            }
            if np.isfinite(sky_area_med):
                small = [i for i, ev in enumerate(evs) if np.isfinite(ev_sky_area_deg2.get(ev, float("nan"))) and float(ev_sky_area_deg2[ev]) <= sky_area_med]
                large = [i for i in range(len(evs)) if i not in set(small)]
                groups["sky_small"] = [int(i) for i in small]
                groups["sky_large"] = [int(i) for i in large]

            def _subset(arrs: list[np.ndarray], idxs: Sequence[int]) -> list[np.ndarray]:
                return [arrs[int(i)] for i in idxs]

            for gname, idxs in groups.items():
                if not idxs:
                    continue
                res_g = _delta_lpd_from_terms(
                    logL_cat_mu_by_event=_subset(cat_mu_by_event, idxs),
                    logL_cat_gr_by_event=_subset(cat_gr_by_event, idxs),
                    logL_missing_mu_by_event=_subset(miss_mu_by_event, idxs),
                    logL_missing_gr_by_event=_subset(miss_gr_by_event, idxs),
                    log_alpha_mu=cache.log_alpha_mu,
                    log_alpha_gr=cache.log_alpha_gr,
                    prior=cache.prior,
                    n_f=int(n_f),
                    eps=float(cache.eps),
                )
                d_tot_g = float(res_g.lpd_mu_total - res_g.lpd_gr_total)
                d_dat_g = float(res_g.lpd_mu_total_data - res_g.lpd_gr_total_data)
                split_rows.append(
                    {
                        "rep": int(rep),
                        "group": str(gname),
                        "n_ev": int(len(idxs)),
                        "delta_lpd_total": d_tot_g,
                        "delta_lpd_data": d_dat_g,
                        "delta_lpd_sel": float(d_tot_g - d_dat_g),
                    }
                )

        res = _delta_lpd_from_terms(
            logL_cat_mu_by_event=cat_mu_by_event,
            logL_cat_gr_by_event=cat_gr_by_event,
            logL_missing_mu_by_event=miss_mu_by_event,
            logL_missing_gr_by_event=miss_gr_by_event,
            log_alpha_mu=cache.log_alpha_mu,
            log_alpha_gr=cache.log_alpha_gr,
            prior=cache.prior,
            n_f=int(n_f),
            eps=float(cache.eps),
        )

        d_tot = float(res.lpd_mu_total - res.lpd_gr_total)
        d_dat = float(res.lpd_mu_total_data - res.lpd_gr_total_data)
        d_sel = float(d_tot - d_dat)
        try:
            f_post_mu = float(np.sum(np.asarray(res.posterior_mu, dtype=float) * np.asarray(res.f_grid, dtype=float)))
        except Exception:
            f_post_mu = float("nan")
        try:
            f_post_gr = float(np.sum(np.asarray(res.posterior_gr, dtype=float) * np.asarray(res.f_grid, dtype=float)))
        except Exception:
            f_post_gr = float("nan")
        rows.append(
            {
                "rep": int(rep),
                "truth_draw": int(jtruth),
                "truth_fmiss_rep": None if fmiss_truth_rep is None else float(fmiss_truth_rep),
                "f_post_mean_mu": f_post_mu,
                "f_post_mean_gr": f_post_gr,
                "delta_lpd_total": d_tot,
                "delta_lpd_data": d_dat,
                "delta_lpd_sel": d_sel,
            }
        )

    arr_tot = np.asarray([r["delta_lpd_total"] for r in rows], dtype=float)
    arr_dat = np.asarray([r["delta_lpd_data"] for r in rows], dtype=float)
    arr_sel = np.asarray([r["delta_lpd_sel"] for r in rows], dtype=float)
    arr_fmu = np.asarray([r.get("f_post_mean_mu", float("nan")) for r in rows], dtype=float)
    arr_fgr = np.asarray([r.get("f_post_mean_gr", float("nan")) for r in rows], dtype=float)
    summary = {
        "truth": truth,
        "truth_alpha": float(truth_alpha),
        "truth_fmiss": None if truth_fmiss is None else float(truth_fmiss),
        "truth_catalog_sampler": str(truth_catalog_sampler),
        "n_rep": int(n_rep),
        "seed": int(seed),
        "truth_z_mode": truth_z_mode,
        "cat_mode": cat_mode,
        "sigma_floor_frac": float(sigma_floor_frac),
        "z_hist_nbins": int(z_hist_nbins),
        "smooth_prior_z0": float(smooth_prior_z0),
        "smooth_prior_beta": float(smooth_prior_beta),
        "n_f": int(n_f),
        "delta_lpd_total": _summary_stats(arr_tot),
        "delta_lpd_data": _summary_stats(arr_dat),
        "delta_lpd_sel": _summary_stats(arr_sel),
        "f_post_mean_mu": _summary_stats(arr_fmu),
        "f_post_mean_gr": _summary_stats(arr_fgr),
    }
    if split_rows:
        summary["split_diag"] = {
            "enabled": True,
            "n_rep": int(min(diag_splits_nrep, n_rep)),
            "groups": sorted({str(r.get("group")) for r in split_rows}),
        }
    return {"rows": rows, "summary": summary, "split_rows": split_rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MG-truth causal-closure injection suite for dark-siren ΔLPD (spectral-only + selection).")
    ap.add_argument("--gap-run-root", type=str, default="/home/primary/PROJECT/outputs/dark_siren_o3_injection_logit_20260209_055801UTC")
    ap.add_argument("--run-label", type=str, default="M0_start101")
    ap.add_argument("--recon-run-dir", type=str, default="/home/primary/PROJECT/outputs/finalization/highpower_multistart_v2/M0_start101")
    ap.add_argument("--out", type=str, default="", help="Output directory (default: outputs/mg_truth_closure_<UTC>/).")
    ap.add_argument("--n-rep", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--z-hist-nbins", type=int, default=200)
    ap.add_argument("--sigma-floor-frac", type=float, default=0.05)
    ap.add_argument("--n-f", type=int, default=101)
    ap.add_argument(
        "--referee-suite",
        action="store_true",
        help="Run dose-response alpha dial + GR-truth generator variants + extra structure-destruction controls.",
    )
    ap.add_argument(
        "--alpha-grid",
        type=str,
        default="",
        help="Comma-separated truth alpha dial grid (runs truth='dial' suites; e.g. '0,0.25,0.5,0.75,1').",
    )
    ap.add_argument("--run-gr-variants", action="store_true", help="Run additional GR-truth generator variants (harder null).")
    ap.add_argument(
        "--run-structure-controls",
        action="store_true",
        help="Run additional catalog-removed / structure-destruction controls (global_pz, smooth_prior, event_hist_swap).",
    )
    ap.add_argument("--smooth-prior-z0", type=float, default=0.10, help="Smooth p(z) proxy scale z0 for control runs.")
    ap.add_argument("--smooth-prior-beta", type=float, default=1.5, help="Smooth p(z) proxy exponent beta for control runs.")
    ap.add_argument(
        "--diag-splits-nrep",
        type=int,
        default=0,
        help="If >0, compute ΔLPD diagnostics for distance tertiles and a sky-area split for the first N replicates of selected suites.",
    )
    args = ap.parse_args(argv)

    _set_thread_env(int(args.threads))

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out) if str(args.out).strip() else (repo_root / "outputs" / f"mg_truth_closure_{_utc_now_compact()}")
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cache = _load_gap_run_cache(gap_run_root=Path(args.gap_run_root), run_label=str(args.run_label))
    recon_run_dir = Path(args.recon_run_dir).expanduser().resolve()

    cfg_snapshot = {
        "gap_run_root": str(Path(args.gap_run_root).expanduser().resolve()),
        "run_label": str(args.run_label),
        "recon_run_dir": str(recon_run_dir),
        "n_rep": int(args.n_rep),
        "seed": int(args.seed),
        "threads": int(args.threads),
        "z_hist_nbins": int(args.z_hist_nbins),
        "sigma_floor_frac": float(args.sigma_floor_frac),
        "n_f": int(args.n_f),
        "referee_suite": bool(args.referee_suite),
        "alpha_grid": str(args.alpha_grid),
        "run_gr_variants": bool(args.run_gr_variants),
        "run_structure_controls": bool(args.run_structure_controls),
        "smooth_prior_z0": float(args.smooth_prior_z0),
        "smooth_prior_beta": float(args.smooth_prior_beta),
        "diag_splits_nrep": int(args.diag_splits_nrep),
    }
    _write_json(out_dir / "config_snapshot.json", cfg_snapshot)

    observed = float(cache.baseline_summary.get("delta_lpd_total", float("nan")))

    def _tok(x: float) -> str:
        s = f"{float(x):.6g}"
        return s.replace("-", "m").replace(".", "p")

    def _suite_tag(summ: dict[str, Any]) -> str:
        parts = [
            f"truth_{summ.get('truth')}",
            f"a{_tok(float(summ.get('truth_alpha', 1.0)))}",
            f"z_{summ.get('truth_z_mode')}",
            f"cat_{summ.get('cat_mode')}",
            f"sam_{summ.get('truth_catalog_sampler', 'weighted')}",
            f"sig{_tok(float(summ.get('sigma_floor_frac', 0.0)))}",
        ]
        tf = summ.get("truth_fmiss", None)
        if tf is not None:
            try:
                parts.append(f"f{_tok(float(tf))}")
            except Exception:
                parts.append("fNA")
        return "_".join(parts)

    # Always run the baseline 3-suite closure block (backwards-compatible with earlier outputs).
    basic_suites: list[dict[str, Any]] = []
    # 1) MG truth with full catalog structure (closure target).
    basic_suites.append(
        run_suite(
            cache=cache,
            recon_run_dir=recon_run_dir,
            truth="mg",
            n_rep=int(args.n_rep),
            seed=int(args.seed),
            truth_z_mode="catalog_hist",
            cat_mode="catalog",
            sigma_floor_frac=float(args.sigma_floor_frac),
            z_hist_nbins=int(args.z_hist_nbins),
            n_f=int(args.n_f),
            truth_alpha=1.0,
            truth_catalog_sampler="weighted",
            smooth_prior_z0=float(args.smooth_prior_z0),
            smooth_prior_beta=float(args.smooth_prior_beta),
            diag_splits_nrep=int(args.diag_splits_nrep),
        )
    )
    # 2) GR truth (control).
    basic_suites.append(
        run_suite(
            cache=cache,
            recon_run_dir=recon_run_dir,
            truth="gr",
            n_rep=int(args.n_rep),
            seed=int(args.seed) + 1,
            truth_z_mode="catalog_hist",
            cat_mode="catalog",
            sigma_floor_frac=float(args.sigma_floor_frac),
            z_hist_nbins=int(args.z_hist_nbins),
            n_f=int(args.n_f),
            truth_alpha=0.0,
            truth_catalog_sampler="weighted",
            smooth_prior_z0=float(args.smooth_prior_z0),
            smooth_prior_beta=float(args.smooth_prior_beta),
            diag_splits_nrep=int(args.diag_splits_nrep),
        )
    )
    # 3) MG truth but with catalog removed (toy uniform-z catalog term + missing-prior truth sampling).
    basic_suites.append(
        run_suite(
            cache=cache,
            recon_run_dir=recon_run_dir,
            truth="mg",
            n_rep=int(args.n_rep),
            seed=int(args.seed) + 2,
            truth_z_mode="missing_prior",
            cat_mode="toy_uniform_z",
            sigma_floor_frac=float(args.sigma_floor_frac),
            z_hist_nbins=int(args.z_hist_nbins),
            n_f=int(args.n_f),
            truth_alpha=1.0,
            truth_catalog_sampler="weighted",
            smooth_prior_z0=float(args.smooth_prior_z0),
            smooth_prior_beta=float(args.smooth_prior_beta),
        )
    )

    alpha_grid = _parse_csv_floats(args.alpha_grid) if str(args.alpha_grid).strip() else []
    if bool(args.referee_suite) and not alpha_grid:
        alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    run_gr_variants = bool(args.run_gr_variants) or bool(args.referee_suite)
    run_controls = bool(args.run_structure_controls) or bool(args.referee_suite)

    alpha_suites: list[dict[str, Any]] = []
    if alpha_grid:
        # Use the same seed for all alpha values to make the dial paired (same draws/z samples, only R_alpha changes).
        for a in alpha_grid:
            alpha_suites.append(
                run_suite(
                    cache=cache,
                    recon_run_dir=recon_run_dir,
                    truth="dial",
                    truth_alpha=float(a),
                    n_rep=int(args.n_rep),
                    seed=int(args.seed),
                    truth_z_mode="catalog_hist",
                    cat_mode="catalog",
                    sigma_floor_frac=float(args.sigma_floor_frac),
                    z_hist_nbins=int(args.z_hist_nbins),
                    n_f=int(args.n_f),
                    truth_catalog_sampler="weighted",
                    smooth_prior_z0=float(args.smooth_prior_z0),
                    smooth_prior_beta=float(args.smooth_prior_beta),
                )
            )

    gr_variant_suites: list[dict[str, Any]] = []
    if run_gr_variants:
        # GR-truth generator variants (all keep scoring definition fixed).
        # Focus on the dominant potential escape hatch: catalogue incompleteness / missing-host mixing.
        fmiss_grid = [0.0, 0.25, 0.5, 0.75, 0.9]
        variants: list[dict[str, Any]] = [
            {
                "name": f"gr_mix_fmiss_{_tok(f)}",
                "truth": "gr",
                "truth_z_mode": "catalog_miss_mix",
                "truth_fmiss": float(f),
                "truth_catalog_sampler": "weighted",
                "sigma_floor_frac": float(args.sigma_floor_frac),
            }
            for f in fmiss_grid
        ]
        variants.extend(
            [
                {
                    "name": "gr_truth_global_pz",
                    "truth": "gr",
                    "truth_z_mode": "global_pz",
                    "truth_fmiss": None,
                    "truth_catalog_sampler": "weighted",
                    "sigma_floor_frac": float(args.sigma_floor_frac),
                },
                {
                    "name": "gr_catalog_prob_only",
                    "truth": "gr",
                    "truth_z_mode": "catalog_hist",
                    "truth_fmiss": None,
                    "truth_catalog_sampler": "prob_only",
                    "sigma_floor_frac": float(args.sigma_floor_frac),
                },
                {
                    "name": "gr_sigma_hi",
                    "truth": "gr",
                    "truth_z_mode": "catalog_hist",
                    "truth_fmiss": None,
                    "truth_catalog_sampler": "weighted",
                    "sigma_floor_frac": 0.10,
                },
            ]
        )
        for i, v in enumerate(variants):
            gr_variant_suites.append(
                run_suite(
                    cache=cache,
                    recon_run_dir=recon_run_dir,
                    truth=v["truth"],
                    truth_alpha=0.0,
                    n_rep=int(args.n_rep),
                    seed=int(args.seed) + 50 + i,
                    truth_z_mode=v["truth_z_mode"],
                    cat_mode="catalog",
                    sigma_floor_frac=float(v["sigma_floor_frac"]),
                    z_hist_nbins=int(args.z_hist_nbins),
                    n_f=int(args.n_f),
                    truth_fmiss=v["truth_fmiss"],
                    truth_catalog_sampler=v["truth_catalog_sampler"],
                    smooth_prior_z0=float(args.smooth_prior_z0),
                    smooth_prior_beta=float(args.smooth_prior_beta),
                )
            )

    control_suites: list[dict[str, Any]] = []
    if run_controls:
        # Less-toy structure/geometry controls.
        # - global_pz/smooth_prior: remove event-specific catalogue structure in both truth sampling and scoring term
        # - event_hist_swap: destroy event-specific structure in the scoring term only (truth stays catalogue-based)
        control_defs: list[tuple[str, str]] = [
            ("global_pz", "global_pz"),
            ("smooth_prior", "smooth_prior"),
            ("event_hist_swap", "catalog_hist"),
        ]
        for i, (cat_mode, truth_z_mode) in enumerate(control_defs):
            control_suites.append(
                run_suite(
                    cache=cache,
                    recon_run_dir=recon_run_dir,
                    truth="mg",
                    truth_alpha=1.0,
                    n_rep=int(args.n_rep),
                    seed=int(args.seed) + 100 + i,
                    truth_z_mode=str(truth_z_mode),
                    cat_mode=cat_mode,
                    sigma_floor_frac=float(args.sigma_floor_frac),
                    z_hist_nbins=int(args.z_hist_nbins),
                    n_f=int(args.n_f),
                    truth_catalog_sampler="weighted",
                    smooth_prior_z0=float(args.smooth_prior_z0),
                    smooth_prior_beta=float(args.smooth_prior_beta),
                )
            )

    all_suites: list[dict[str, Any]] = []
    all_suites.extend(basic_suites)
    all_suites.extend(alpha_suites)
    all_suites.extend(gr_variant_suites)
    all_suites.extend(control_suites)

    # Write tables.
    for s in all_suites:
        summ = s["summary"]
        tag = _suite_tag(summ)
        _write_json(tables_dir / f"{tag}_summary.json", summ)
        _write_csv(
            tables_dir / f"{tag}_samples.csv",
            s["rows"],
            fieldnames=[
                "rep",
                "truth_draw",
                "truth_fmiss_rep",
                "f_post_mean_mu",
                "f_post_mean_gr",
                "delta_lpd_total",
                "delta_lpd_data",
                "delta_lpd_sel",
            ],
        )
        split_rows = s.get("split_rows", [])
        if isinstance(split_rows, list) and split_rows:
            _write_csv(
                tables_dir / f"{tag}_split_diag.csv",
                split_rows,
                fieldnames=["rep", "group", "n_ev", "delta_lpd_total", "delta_lpd_data", "delta_lpd_sel"],
            )

    # Build summary.json
    mg_full = basic_suites[0]["summary"]
    gr_full = basic_suites[1]["summary"]
    mg_toy = basic_suites[2]["summary"]
    mg_full_rows = np.asarray([r["delta_lpd_total"] for r in basic_suites[0]["rows"]], dtype=float)
    gr_full_rows = np.asarray([r["delta_lpd_total"] for r in basic_suites[1]["rows"]], dtype=float)
    p_mg_ge_obs = float(np.mean(mg_full_rows[np.isfinite(mg_full_rows)] >= observed)) if np.isfinite(observed) else float("nan")
    p_gr_ge_obs = float(np.mean(gr_full_rows[np.isfinite(gr_full_rows)] >= observed)) if np.isfinite(observed) else float("nan")

    alpha_summary_rows: list[dict[str, Any]] = []
    for s in alpha_suites:
        summ = s["summary"]
        arr = np.asarray([r["delta_lpd_total"] for r in s["rows"]], dtype=float)
        p_ge = float(np.mean(arr[np.isfinite(arr)] >= observed)) if np.isfinite(observed) else float("nan")
        alpha_summary_rows.append(
            {
                "alpha": float(summ.get("truth_alpha", float("nan"))),
                "delta_lpd_total_mean": float(summ["delta_lpd_total"]["mean"]),
                "delta_lpd_total_p50": float(summ["delta_lpd_total"]["p50"]),
                "p_ge_observed": p_ge,
            }
        )
    if alpha_summary_rows:
        alpha_summary_rows = sorted(alpha_summary_rows, key=lambda r: float(r["alpha"]))
        _write_csv(
            tables_dir / "alpha_dose_response_summary.csv",
            alpha_summary_rows,
            fieldnames=["alpha", "delta_lpd_total_mean", "delta_lpd_total_p50", "p_ge_observed"],
        )

    gr_variant_rows: list[dict[str, Any]] = []
    for s in gr_variant_suites:
        summ = s["summary"]
        arr = np.asarray([r["delta_lpd_total"] for r in s["rows"]], dtype=float)
        p_ge = float(np.mean(arr[np.isfinite(arr)] >= observed)) if np.isfinite(observed) else float("nan")
        gr_variant_rows.append(
            {
                "truth_z_mode": str(summ.get("truth_z_mode")),
                "truth_fmiss": summ.get("truth_fmiss", None),
                "truth_catalog_sampler": str(summ.get("truth_catalog_sampler")),
                "sigma_floor_frac": float(summ.get("sigma_floor_frac")),
                "delta_lpd_total_mean": float(summ["delta_lpd_total"]["mean"]),
                "delta_lpd_total_p50": float(summ["delta_lpd_total"]["p50"]),
                "p_ge_observed": p_ge,
            }
        )
    if gr_variant_rows:
        _write_csv(
            tables_dir / "gr_truth_generator_variants_summary.csv",
            gr_variant_rows,
            fieldnames=[
                "truth_z_mode",
                "truth_fmiss",
                "truth_catalog_sampler",
                "sigma_floor_frac",
                "delta_lpd_total_mean",
                "delta_lpd_total_p50",
                "p_ge_observed",
            ],
        )

    control_rows: list[dict[str, Any]] = []
    for s in control_suites:
        summ = s["summary"]
        arr = np.asarray([r["delta_lpd_total"] for r in s["rows"]], dtype=float)
        p_ge = float(np.mean(arr[np.isfinite(arr)] >= observed)) if np.isfinite(observed) else float("nan")
        control_rows.append(
            {
                "cat_mode": str(summ.get("cat_mode")),
                "truth_z_mode": str(summ.get("truth_z_mode")),
                "delta_lpd_total_mean": float(summ["delta_lpd_total"]["mean"]),
                "delta_lpd_total_p50": float(summ["delta_lpd_total"]["p50"]),
                "p_ge_observed": p_ge,
            }
        )
    if control_rows:
        _write_csv(
            tables_dir / "structure_destruction_controls_summary.csv",
            control_rows,
            fieldnames=["cat_mode", "truth_z_mode", "delta_lpd_total_mean", "delta_lpd_total_p50", "p_ge_observed"],
        )

    summary = {
        "outdir": str(out_dir),
        "observed_delta_lpd_total": observed,
        "p_mg_truth_ge_observed": p_mg_ge_obs,
        "p_gr_truth_ge_observed": p_gr_ge_obs,
        "mg_truth_catalog": mg_full,
        "gr_truth_catalog": gr_full,
        "mg_truth_catalog_removed": mg_toy,
        "alpha_dose_response": alpha_summary_rows,
        "gr_truth_generator_variants": gr_variant_rows,
        "structure_destruction_controls": control_rows,
    }
    _write_json(out_dir / "summary.json", summary)

    # Plots.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _arr(suite: dict[str, Any], key: str) -> np.ndarray:
            return np.asarray([r[key] for r in suite["rows"]], dtype=float)

        arr_mg = _arr(basic_suites[0], "delta_lpd_total")
        arr_gr = _arr(basic_suites[1], "delta_lpd_total")
        arr_toy = _arr(basic_suites[2], "delta_lpd_total")

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.hist(arr_gr[np.isfinite(arr_gr)], bins=40, alpha=0.55, label="GR truth (catalog)", color="C1")
        ax.hist(arr_mg[np.isfinite(arr_mg)], bins=40, alpha=0.55, label="MG truth (catalog)", color="C2")
        ax.hist(arr_toy[np.isfinite(arr_toy)], bins=40, alpha=0.35, label="MG truth (catalog removed)", color="C0")
        if np.isfinite(observed):
            ax.axvline(observed, color="k", lw=1.2, alpha=0.8, label=f"observed ΔLPD={observed:.3f}")
        ax.axvline(0.0, color="k", lw=1.0, alpha=0.4)
        ax.set(xlabel="ΔLPD (total)", ylabel="count", title="MG-Truth Causal-Closure Injection Suite")
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(figures_dir / "delta_lpd_hist_overlay.png", dpi=180)
        plt.close(fig)

        # Decomposition scatter: MG truth (catalog)
        xd = _arr(basic_suites[0], "delta_lpd_data")
        xs = _arr(basic_suites[0], "delta_lpd_sel")
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        ax.scatter(xd, xs, s=10, alpha=0.6)
        ax.axhline(0.0, color="k", lw=1.0, alpha=0.4)
        ax.axvline(0.0, color="k", lw=1.0, alpha=0.4)
        ax.set(xlabel="ΔLPD_data", ylabel="ΔLPD_sel", title="MG truth: data vs selection contributions")
        ax.grid(alpha=0.25, linestyle=":")
        fig.tight_layout()
        fig.savefig(figures_dir / "mg_truth_data_vs_sel_scatter.png", dpi=180)
        plt.close(fig)

        if alpha_summary_rows:
            al = np.asarray([r["alpha"] for r in alpha_summary_rows], dtype=float)
            mu = np.asarray([r["delta_lpd_total_mean"] for r in alpha_summary_rows], dtype=float)
            pge = np.asarray([r["p_ge_observed"] for r in alpha_summary_rows], dtype=float)

            fig, ax = plt.subplots(figsize=(6.2, 4.0))
            ax.plot(al, mu, marker="o", lw=1.5)
            ax.set(xlabel="truth alpha (R_alpha)", ylabel="mean ΔLPD_total", title="Dose–Response: mean ΔLPD vs alpha")
            ax.grid(alpha=0.25, linestyle=":")
            fig.tight_layout()
            fig.savefig(figures_dir / "dose_response_mean_delta_lpd_vs_alpha.png", dpi=180)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6.2, 4.0))
            ax.plot(al, pge, marker="o", lw=1.5)
            ax.set(xlabel="truth alpha (R_alpha)", ylabel="P(ΔLPD_total ≥ observed)", title="Dose–Response: tail prob vs alpha")
            ax.grid(alpha=0.25, linestyle=":")
            fig.tight_layout()
            fig.savefig(figures_dir / "dose_response_p_ge_observed_vs_alpha.png", dpi=180)
            plt.close(fig)

    except Exception:
        pass

    # report.md
    md: list[str] = []
    md.append("# MG-Truth Causal-Closure Suite (Dark Sirens)\n")
    md.append(f"- Generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`\n")
    md.append(f"- Gap run root: `{cfg_snapshot['gap_run_root']}` label `{cfg_snapshot['run_label']}`\n")
    md.append(f"- Recon run dir: `{cfg_snapshot['recon_run_dir']}`\n")
    md.append(f"- n_rep per suite: `{cfg_snapshot['n_rep']}`; threads: `{cfg_snapshot['threads']}`\n")
    md.append("\n")
    md.append("## Observed\n")
    md.append(f"- Observed ΔLPD_total (real data): `{observed:.6g}`\n")
    md.append("\n")
    md.append("## Suites\n")
    for summ in (mg_full, gr_full, mg_toy):
        tag = f"truth={summ['truth']} alpha={summ.get('truth_alpha', 1.0)} z_mode={summ['truth_z_mode']} cat_mode={summ['cat_mode']}"
        md.append(f"### {tag}\n")
        for k in ("delta_lpd_total", "delta_lpd_data", "delta_lpd_sel"):
            st = summ[k]
            md.append(
                f"- {k}: mean `{st['mean']:+.3f}`, p16/p50/p84 `{st['p16']:+.3f}` / `{st['p50']:+.3f}` / `{st['p84']:+.3f}`, max `{st['max']:+.3f}`\n"
            )
        md.append("\n")

    if np.isfinite(p_mg_ge_obs):
        md.append("## Closure Check\n")
        md.append(f"- Fraction of MG-truth catalog replicates with ΔLPD_total ≥ observed: `{p_mg_ge_obs:.4f}`\n")
        md.append(f"- Fraction of GR-truth catalog replicates with ΔLPD_total ≥ observed: `{p_gr_ge_obs:.4f}`\n")
        md.append("\n")

    if alpha_summary_rows:
        md.append("## Dose–Response (Truth Alpha Dial)\n")
        md.append("- Table: `tables/alpha_dose_response_summary.csv`\n")
        md.append("- Figures: `figures/dose_response_mean_delta_lpd_vs_alpha.png`, `figures/dose_response_p_ge_observed_vs_alpha.png`\n")
        md.append("\n")

    if gr_variant_rows:
        md.append("## GR-Truth Generator Variants (Harder Null)\n")
        md.append("- Table: `tables/gr_truth_generator_variants_summary.csv`\n")
        md.append("\n")

    if control_rows:
        md.append("## Structure-Destruction Controls\n")
        md.append("- Table: `tables/structure_destruction_controls_summary.csv`\n")
        md.append("\n")

    md.append("## Artifacts\n")
    md.append("- Summary: `summary.json`\n")
    md.append("- Figures: `figures/delta_lpd_hist_overlay.png`, `figures/mg_truth_data_vs_sel_scatter.png`\n")
    md.append("- Samples CSVs: `tables/truth_*_samples.csv` (one per suite tag)\n")

    (out_dir / "report.md").write_text("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
