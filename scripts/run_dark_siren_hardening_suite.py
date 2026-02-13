#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

# Ensure we import the *local* package from this repository (not a different checkout on sys.path).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from entropy_horizon_recon.dark_siren_gap_lpd import (  # noqa: E402
    BetaPrior,
    MarginalizedFMissResult,
    marginalize_f_miss_global,
)
from entropy_horizon_recon.dark_sirens_incompleteness import (  # noqa: E402
    compute_missing_host_logL_draws_from_histogram,
    precompute_missing_host_prior,
)
from entropy_horizon_recon.dark_sirens_pe import (  # noqa: E402
    PePixelDistanceHistogram,
    build_pe_pixel_distance_histogram,
    load_gwtc_pe_sky_samples,
)
from entropy_horizon_recon.dark_sirens_pe_fast import compute_dark_siren_logL_draws_from_pe_hist_fast  # noqa: E402
from entropy_horizon_recon.dark_sirens_selection import load_o3_injections  # noqa: E402
from entropy_horizon_recon.gw_distance_priors import GWDistancePrior  # noqa: E402
from entropy_horizon_recon.report import ReportPaths, format_table, write_markdown_report  # noqa: E402
from entropy_horizon_recon.repro import command_str, git_head_sha, git_is_dirty  # noqa: E402
from entropy_horizon_recon.selection_nuisance import (  # noqa: E402
    SelectionNuisanceConfig,
    apply_nuisance_to_alpha_linearized,
    apply_nuisance_to_alpha_logit_resummed,
    compute_selection_nuisance_moments,
)
from entropy_horizon_recon.sirens import (  # noqa: E402
    MuForwardPosterior,
    load_mu_forward_posterior,
    predict_dL_em,
    predict_r_gw_em,
)


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


def _delta_lpd_from_terms(
    *,
    logL_cat_mu_by_event: list[np.ndarray],
    logL_cat_gr_by_event: list[np.ndarray],
    logL_missing_mu_by_event: list[np.ndarray],
    logL_missing_gr_by_event: list[np.ndarray],
    log_alpha_mu: np.ndarray | None,
    log_alpha_gr: np.ndarray | None,
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


def _delta_lpd_from_cached_terms(
    *,
    cache: GapRunCache,
    log_alpha_mu: np.ndarray,
    log_alpha_gr: np.ndarray,
    events_mask: np.ndarray | None = None,
) -> MarginalizedFMissResult:
    if events_mask is None:
        return _delta_lpd_from_terms(
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
    return _delta_lpd_from_terms(
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


def _logmeanexp_1d(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        return float("-inf")
    m = float(np.max(x))
    if not np.isfinite(m):
        return float("-inf")
    s = float(np.mean(np.exp(np.clip(x - m, -700.0, 50.0))))
    if not np.isfinite(s) or s <= 0.0:
        return float("-inf")
    return float(m + np.log(s))


def _delta_lpd_fixed_f_miss(
    *,
    logL_cat_mu_by_event: list[np.ndarray],
    logL_cat_gr_by_event: list[np.ndarray],
    logL_missing_mu_by_event: list[np.ndarray],
    logL_missing_gr_by_event: list[np.ndarray],
    log_alpha_mu: np.ndarray,
    log_alpha_gr: np.ndarray,
    f_miss: float,
) -> tuple[float, float, float, float]:
    """Fast fixed-f_miss score (no marginalisation) for adversarial scans.

    Returns (lpd_mu_total, lpd_gr_total, lpd_mu_total_data, lpd_gr_total_data).
    """
    f = float(f_miss)
    f = float(np.clip(f, 1e-6, 1.0 - 1e-6))
    logf = float(np.log(f))
    log1mf = float(np.log1p(-f))

    n_ev = int(len(logL_cat_mu_by_event))
    if n_ev <= 0:
        raise ValueError("Empty event list in fixed-f score.")
    n_draws = int(np.asarray(logL_cat_mu_by_event[0], dtype=float).size)
    mu = np.zeros((n_draws,), dtype=float)
    gr = np.zeros((n_draws,), dtype=float)
    for cat_mu, cat_gr, miss_mu, miss_gr in zip(logL_cat_mu_by_event, logL_cat_gr_by_event, logL_missing_mu_by_event, logL_missing_gr_by_event, strict=True):
        cat_mu = np.asarray(cat_mu, dtype=float)
        cat_gr = np.asarray(cat_gr, dtype=float)
        miss_mu = np.asarray(miss_mu, dtype=float)
        miss_gr = np.asarray(miss_gr, dtype=float)
        mu = mu + np.logaddexp(log1mf + cat_mu, logf + miss_mu)
        gr = gr + np.logaddexp(log1mf + cat_gr, logf + miss_gr)

    # Selection-normalised totals.
    mu_sel = mu - float(n_ev) * np.asarray(log_alpha_mu, dtype=float)
    gr_sel = gr - float(n_ev) * np.asarray(log_alpha_gr, dtype=float)

    return _logmeanexp_1d(mu_sel), _logmeanexp_1d(gr_sel), _logmeanexp_1d(mu), _logmeanexp_1d(gr)


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
        return float(0.5 * (edges[j] + edges[j + 1]))
    frac = (0.5 - c0) / (p * float(widths[j]))
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(edges[j] + frac * widths[j])


def _pe_sky_marginal_pdf_1d(pe: PePixelDistanceHistogram) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(pe.dL_edges, dtype=float)
    widths = np.diff(edges)
    prob_pix = np.asarray(pe.prob_pix, dtype=float)
    pdf_bins = np.asarray(pe.pdf_bins, dtype=float)
    p_sum = float(np.sum(prob_pix))
    if not (np.isfinite(p_sum) and p_sum > 0.0):
        raise ValueError("Invalid prob_pix normalization while building sky-marginal pdf.")
    if int(pdf_bins.shape[0]) == int(prob_pix.size):
        pdf_1d = np.sum(prob_pix.reshape((-1, 1)) * pdf_bins, axis=0) / p_sum
    elif int(pdf_bins.shape[0]) == 1:
        pdf_1d = np.asarray(pdf_bins[0], dtype=float)
    else:
        raise ValueError("Incompatible pdf_bins shape for sky-marginal pdf.")
    pdf_1d = np.clip(np.asarray(pdf_1d, dtype=float), 0.0, np.inf)
    norm = float(np.sum(pdf_1d * widths))
    if not (np.isfinite(norm) and norm > 0.0):
        raise ValueError("Invalid sky-marginal pdf normalization.")
    pdf_1d = pdf_1d / norm
    return edges, pdf_1d


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if x.ndim != 1 or w.ndim != 1 or x.shape != w.shape:
        raise ValueError("weighted_quantile expects 1D x and w with matching shapes.")
    if not (0.0 <= float(q) <= 1.0):
        raise ValueError("q must be in [0,1].")
    m = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
    if not np.any(m):
        return float("nan")
    x = x[m]
    w = w[m]
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    w = w[order]
    cw = np.cumsum(w)
    tot = float(cw[-1])
    if not (np.isfinite(tot) and tot > 0.0):
        return float("nan")
    t = float(q) * tot
    j = int(np.searchsorted(cw, t, side="left"))
    j = min(max(j, 0), x.size - 1)
    return float(x[j])


def _parse_event_date(ev: str) -> date | None:
    # Expected: GWYYMMDD_HHMMSS
    s = str(ev).strip()
    if not s.startswith("GW") or "_" not in s:
        return None
    try:
        ymd = s[2:8]
        yy = int(ymd[0:2])
        mm = int(ymd[2:4])
        dd = int(ymd[4:6])
        year = 2000 + yy
        return date(year, mm, dd)
    except Exception:
        return None


def _ensure_gwosc_event_meta(*, events: list[str], cache_path: Path) -> dict[str, Any]:
    """Fetch and cache minimal GWOSC event metadata needed for detector/network/mass splits."""
    cache_path = cache_path.expanduser().resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {}
    if cache_path.exists():
        try:
            meta = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    meta = meta if isinstance(meta, dict) else {}

    # Import here to keep the script runnable in fully-offline environments when this section is skipped.
    from gwosc.api import fetch_event_json  # type: ignore

    updated = False
    for ev in events:
        if ev in meta and isinstance(meta.get(ev), dict) and meta[ev].get("detectors"):
            continue
        try:
            j = fetch_event_json(ev)
        except Exception:
            continue
        if not isinstance(j, dict) or "events" not in j:
            continue
        ev_key = None
        for k, v in (j.get("events") or {}).items():
            if isinstance(v, dict) and str(v.get("commonName", "")) == ev:
                ev_key = k
                break
        if ev_key is None:
            # Fallback: take the first entry.
            keys = list((j.get("events") or {}).keys())
            ev_key = keys[0] if keys else None
        if ev_key is None:
            continue
        v = (j.get("events") or {}).get(ev_key, {})
        if not isinstance(v, dict):
            continue
        strain = v.get("strain", [])
        dets = sorted({str(s.get("detector")) for s in strain if isinstance(s, dict) and s.get("detector")})
        # Mass proxies.
        def _f(k: str) -> float | None:
            try:
                x = v.get(k)
                return float(x) if x is not None and np.isfinite(float(x)) else None
            except Exception:
                return None

        meta[ev] = {
            "gwosc_key": str(ev_key),
            "GPS": _f("GPS"),
            "detectors": dets,
            "chirp_mass_source": _f("chirp_mass_source"),
            "chirp_mass": _f("chirp_mass"),
            "network_snr": _f("network_matched_filter_snr"),
            "far": _f("far"),
            "p_astro": _f("p_astro"),
            "catalog": str(v.get("catalog.shortName", "")),
        }
        updated = True

    if updated:
        cache_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def _noncircular_event_splits(
    *,
    cache: GapRunCache,
    cache_events_dir: Path,
    gw_prior: GWDistancePrior,
    epoch_boundary: date,
    gwosc_meta: dict[str, Any] | None,
    z_bin_n: int,
) -> dict[str, Any]:
    """Compute strictly non-circular split diagnostics using frozen assignments."""
    # Load per-event caches once.
    dL_med = {}
    z_eff = {}
    dets = {}
    mchirp = {}
    epoch = {}
    for ev in cache.events:
        p = cache_events_dir / f"event_{ev}.npz"
        dL_med[ev] = float(_pe_distance_median_from_event_cache(p))
        # Parse epoch.
        d = _parse_event_date(ev)
        if d is None:
            epoch[ev] = "unknown"
        else:
            epoch[ev] = "O3a" if d < epoch_boundary else "O3b"

        # Detector/mass meta from GWOSC (optional).
        if gwosc_meta and isinstance(gwosc_meta.get(ev), dict):
            dets[ev] = [str(x) for x in (gwosc_meta[ev].get("detectors") or [])]
            cm = gwosc_meta[ev].get("chirp_mass_source")
            if cm is None:
                cm = gwosc_meta[ev].get("chirp_mass")
            mchirp[ev] = float(cm) if cm is not None else float("nan")
        else:
            dets[ev] = []
            mchirp[ev] = float("nan")

        # Catalog-derived effective z posterior median under a fixed GR fiducial mapping.
        with np.load(p, allow_pickle=False) as dcache:
            z = np.asarray(dcache["z"], dtype=float)
            w = np.asarray(dcache["w"], dtype=float)
            ipix = np.asarray(dcache["ipix"], dtype=np.int64)
            pe = PePixelDistanceHistogram(
                nside=int(cache.manifest.get("pe_nside", 64)),
                nest=True,
                p_credible=float(cache.manifest.get("p_credible", 0.9)),
                pix_sel=np.asarray(dcache["pe_pix_sel"], dtype=np.int64),
                prob_pix=np.asarray(dcache["pe_prob_pix"], dtype=float),
                dL_edges=np.asarray(dcache["pe_dL_edges"], dtype=float),
                pdf_bins=np.asarray(dcache["pe_pdf_bins"], dtype=float),
            )
        edges, pdf_1d = _pe_sky_marginal_pdf_1d(pe)
        widths = np.diff(edges)
        # Fiducial dL(z) for binning (Planck-ish).
        H0 = float(gw_prior.h0_ref)
        om = float(gw_prior.omega_m0)
        c = 299792.458
        z_grid = np.linspace(0.0, float(np.max(z)), 5001)
        Ez = np.sqrt(om * (1.0 + z_grid) ** 3 + (1.0 - om))
        invE = 1.0 / Ez
        dz = np.diff(z_grid)
        dc = np.empty_like(z_grid)
        dc[0] = 0.0
        dc[1:] = (c / H0) * np.cumsum(0.5 * dz * (invE[:-1] + invE[1:]))
        dL_grid = (1.0 + z_grid) * dc
        dL_em = np.interp(np.clip(z, 0.0, z_grid[-1]), z_grid, dL_grid)

        # Evaluate p(dL|data) on bins.
        bin_idx = np.searchsorted(edges, dL_em, side="right") - 1
        valid = (bin_idx >= 0) & (bin_idx < widths.size) & np.isfinite(dL_em) & (dL_em > 0.0)
        pdf = np.zeros_like(dL_em)
        pdf[valid] = pdf_1d[bin_idx[valid]]

        log_pi = gw_prior.log_pi_dL(np.clip(dL_em, 1e-6, np.inf))
        inv_pi = np.zeros_like(dL_em)
        ok = np.isfinite(log_pi)
        inv_pi[ok] = np.exp(-log_pi[ok])

        # Sky weights per galaxy.
        npix = int(12 * int(pe.nside) * int(pe.nside))
        pix_to_row = np.full((npix,), -1, dtype=np.int32)
        pix_to_row[np.asarray(pe.pix_sel, dtype=np.int64)] = np.arange(int(pe.pix_sel.size), dtype=np.int32)
        row = pix_to_row[ipix]
        good = (row >= 0) & np.isfinite(z) & (z > 0.0) & np.isfinite(w) & (w > 0.0) & np.isfinite(pdf) & (pdf > 0.0)
        if not np.any(good):
            z_eff[ev] = float("nan")
        else:
            prob = np.asarray(pe.prob_pix, dtype=float)[row[good]]
            wt = w[good] * prob * pdf[good] * inv_pi[good]
            z_eff[ev] = _weighted_quantile(z[good], wt, 0.5)

    # Helper: build quantile bins on a scalar map.
    def _bins_from_values(vals: dict[str, float], n: int) -> list[tuple[float, float]]:
        xs = np.asarray([v for v in vals.values() if np.isfinite(v)], dtype=float)
        if xs.size == 0:
            return [(-np.inf, np.inf)]
        qs = np.quantile(xs, np.linspace(0.0, 1.0, int(n) + 1))
        bins: list[tuple[float, float]] = []
        for i in range(int(n)):
            lo = float(qs[i]) if i > 0 else float("-inf")
            hi = float(qs[i + 1]) if i + 1 < qs.size - 1 else float("inf")
            bins.append((lo, hi))
        return bins

    out: dict[str, Any] = {"enabled": True, "splits": {}}

    # Distance bins (by PE dL median).
    bins_dL = _bins_from_values(dL_med, int(z_bin_n))
    split_rows = []
    for bi, (lo, hi) in enumerate(bins_dL):
        m = np.asarray([(dL_med[ev] >= lo) and (dL_med[ev] < hi) for ev in cache.events], dtype=bool)
        split_rows.append({"bin": int(bi), "n_events": int(np.sum(m)), "lo": float(lo), "hi": float(hi), "events": [ev for ev, ok in zip(cache.events, m, strict=True) if ok]})
    out["splits"]["distance_bins"] = {"bins": split_rows}

    # Catalog-derived z bins (frozen under GR fiducial mapping).
    bins_z = _bins_from_values(z_eff, int(z_bin_n))
    split_rows = []
    for bi, (lo, hi) in enumerate(bins_z):
        m = np.asarray([(z_eff[ev] >= lo) and (z_eff[ev] < hi) for ev in cache.events], dtype=bool)
        split_rows.append({"bin": int(bi), "n_events": int(np.sum(m)), "lo": float(lo), "hi": float(hi), "events": [ev for ev, ok in zip(cache.events, m, strict=True) if ok]})
    out["splits"]["catalog_z_bins"] = {"bins": split_rows}

    # Network splits (HL vs HLV).
    split_rows = []
    m_hlv = np.asarray([("V1" in set(dets.get(ev, []))) for ev in cache.events], dtype=bool)
    split_rows.append({"bin": "HL", "n_events": int(np.sum(~m_hlv)), "events": [ev for ev, ok in zip(cache.events, ~m_hlv, strict=True) if ok]})
    split_rows.append({"bin": "HLV", "n_events": int(np.sum(m_hlv)), "events": [ev for ev, ok in zip(cache.events, m_hlv, strict=True) if ok]})
    out["splits"]["network"] = {"bins": split_rows}

    # Epoch splits (O3a vs O3b).
    m_o3b = np.asarray([epoch.get(ev) == "O3b" for ev in cache.events], dtype=bool)
    out["splits"]["epoch"] = {
        "bins": [
            {"bin": "O3a", "n_events": int(np.sum(~m_o3b)), "events": [ev for ev, ok in zip(cache.events, ~m_o3b, strict=True) if ok]},
            {"bin": "O3b", "n_events": int(np.sum(m_o3b)), "events": [ev for ev, ok in zip(cache.events, m_o3b, strict=True) if ok]},
        ]
    }

    # Chirp-mass bins (from GWOSC, if available).
    bins_m = _bins_from_values(mchirp, int(z_bin_n))
    split_rows = []
    for bi, (lo, hi) in enumerate(bins_m):
        m = np.asarray([(mchirp[ev] >= lo) and (mchirp[ev] < hi) for ev in cache.events], dtype=bool)
        split_rows.append({"bin": int(bi), "n_events": int(np.sum(m)), "lo": float(lo), "hi": float(hi), "events": [ev for ev, ok in zip(cache.events, m, strict=True) if ok]})
    out["splits"]["chirp_mass_bins"] = {"bins": split_rows}

    out["per_event"] = {
        ev: {"dL_med_mpc": float(dL_med[ev]), "z_eff_cat_gr": float(z_eff[ev]), "detectors": dets.get(ev, []), "chirp_mass_source": float(mchirp[ev]), "epoch": str(epoch[ev])}
        for ev in cache.events
    }
    return out


def _alpha_scan_linearized(
    *,
    cache: GapRunCache,
    alpha_grid: np.ndarray,
    events_mask: np.ndarray | None = None,
    n_f: int | None = None,
) -> dict[str, Any]:
    """Linearized amplitude dial: interpolate logL/log_alpha between GR (alpha=0) and template (alpha=1)."""
    alpha_grid = np.asarray(alpha_grid, dtype=float)
    if alpha_grid.ndim != 1 or alpha_grid.size < 3:
        raise ValueError("alpha_grid must be 1D with >=3 points.")
    if np.any(~np.isfinite(alpha_grid)):
        raise ValueError("alpha_grid must be finite.")
    if np.min(alpha_grid) < 0.0:
        raise ValueError("alpha_grid must be >=0.")

    m = None if events_mask is None else np.asarray(events_mask, dtype=bool)
    if m is not None and (m.ndim != 1 or m.size != len(cache.events)):
        raise ValueError("events_mask shape mismatch.")
    cat_mu0 = cache.logL_cat_mu_by_event if m is None else [a for a, ok in zip(cache.logL_cat_mu_by_event, m, strict=True) if ok]
    cat_gr0 = cache.logL_cat_gr_by_event if m is None else [a for a, ok in zip(cache.logL_cat_gr_by_event, m, strict=True) if ok]
    miss_mu0 = cache.logL_missing_mu_by_event if m is None else [a for a, ok in zip(cache.logL_missing_mu_by_event, m, strict=True) if ok]
    miss_gr0 = cache.logL_missing_gr_by_event if m is None else [a for a, ok in zip(cache.logL_missing_gr_by_event, m, strict=True) if ok]

    n_f_use = int(cache.n_f if n_f is None else n_f)

    # Baseline endpoints.
    out_rows = []
    dlp = []
    for a in alpha_grid:
        a = float(a)
        # Mu(alpha)=GR + alpha*(Mu-GR) for each term. This is a linearized dial used for coherence diagnostics,
        # not a physical recomputation of R_alpha(z) at the catalog level.
        cat_mu_a = [gr + a * (mu - gr) for mu, gr in zip(cat_mu0, cat_gr0, strict=True)]
        miss_mu_a = [gr + a * (mu - gr) for mu, gr in zip(miss_mu0, miss_gr0, strict=True)]
        log_alpha_mu_a = cache.log_alpha_gr + a * (cache.log_alpha_mu - cache.log_alpha_gr)
        res = _delta_lpd_from_terms(
            logL_cat_mu_by_event=cat_mu_a,
            logL_cat_gr_by_event=cat_gr0,
            logL_missing_mu_by_event=miss_mu_a,
            logL_missing_gr_by_event=miss_gr0,
            log_alpha_mu=log_alpha_mu_a,
            log_alpha_gr=cache.log_alpha_gr,
            prior=cache.prior,
            n_f=n_f_use,
            eps=cache.eps,
        )
        d = float(res.lpd_mu_total - res.lpd_gr_total)
        dlp.append(d)
        out_rows.append({"alpha": a, "delta_lpd_total": d})

    dlp_arr = np.asarray(dlp, dtype=float)
    amax = float(alpha_grid[int(np.nanargmax(dlp_arr))]) if dlp_arr.size else float("nan")
    # Discrete posterior over alpha proportional to exp(delta_lpd - max).
    ok = np.isfinite(dlp_arr)
    if np.any(ok):
        d0 = float(np.nanmax(dlp_arr[ok]))
        w = np.zeros_like(dlp_arr, dtype=float)
        w[ok] = np.exp(np.clip(dlp_arr[ok] - d0, -700.0, 50.0))
        wsum = float(np.sum(w))
        if wsum > 0.0:
            w = w / wsum
            amean = float(np.sum(w * alpha_grid))
            avar = float(np.sum(w * (alpha_grid - amean) ** 2))
            astd = float(np.sqrt(max(avar, 0.0)))
        else:
            amean, astd = float("nan"), float("nan")
    else:
        amean, astd = float("nan"), float("nan")

    return {
        "alpha_grid": [float(x) for x in alpha_grid.tolist()],
        "rows": out_rows,
        "alpha_hat_argmax": amax,
        "alpha_post_mean": amean,
        "alpha_post_sd": astd,
        "delta_lpd_at_alpha1": float(dlp_arr[np.where(np.isclose(alpha_grid, 1.0))[0][0]]) if np.any(np.isclose(alpha_grid, 1.0)) else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dark-siren hardening suite (O3): non-circular splits + permutation nulls + adversarial systematics.")
    ap.add_argument("--config", type=str, default="configs/dark_siren_hardening_suite_o3.json")
    ap.add_argument("--out", type=str, default="", help="Output directory (default: outputs/dark_siren_hardening_suite_o3_<UTC>).")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    _set_thread_env(int(args.threads))
    rng = np.random.default_rng(int(args.seed))

    repo_root = Path(__file__).resolve().parents[1]
    cfg = _read_json(Path(args.config))

    gap_run_root = Path(cfg["baseline"]["gap_run_root"])
    run_label = str(cfg["baseline"]["run_label"])
    recon_run_dir = Path(cfg["baseline"].get("recon_run_dir", "/home/primary/PROJECT/outputs/realdata_variant_M0"))

    out_dir = Path(args.out) if str(args.out).strip() else (repo_root / "outputs" / f"dark_siren_hardening_suite_o3_{_utc_now_compact()}")
    paths = ReportPaths(out_dir=out_dir)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    paths.samples_dir.mkdir(parents=True, exist_ok=True)

    cache = _load_gap_run_cache(gap_run_root=gap_run_root, run_label=run_label)

    # Baseline reproduction.
    base = _delta_lpd_from_cached_terms(cache=cache, log_alpha_mu=cache.log_alpha_mu, log_alpha_gr=cache.log_alpha_gr)
    _write_json_atomic(paths.tables_dir / "baseline_recompute.json", base.to_jsonable())
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

    # Load the reconstruction posterior subset.
    post_full = load_mu_forward_posterior(recon_run_dir)
    post = _subset_mu_posterior(post_full, cache.draw_idx)
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
    gal_chunk_size = int(manifest.get("galaxy_chunk_size", 50_000))
    pix_chunk_size = int(manifest.get("missing_pixel_chunk_size", 5_000))

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

    # ---- SECTION 3.1: Selection-function adversarial nuisance (existing implementation + required-failure scan)
    sel_summary: dict[str, Any] = {"enabled": False}
    if bool(cfg.get("selection_nuisance", {}).get("enabled", False)):
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

        inj_path = cache.manifest.get("selection_injections_hdf")
        if not inj_path:
            raise KeyError("baseline manifest missing selection_injections_hdf")
        injections = load_o3_injections(str(inj_path), ifar_threshold_yr=float(cache.manifest.get("selection_ifar_thresh_yr", 1.0)))

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

        k_snr = int(mom.mom_mu_logsnr.shape[1])
        include_z = mom.mom_mu_z is not None
        k_z = int(mom.mom_mu_z.shape[1]) if include_z else 0
        x0 = np.zeros((k_snr + k_z + (1 if include_mass else 0),), dtype=float)
        req_cfg = (sel_cfg.get("required_failure_scan") or {}) if isinstance(sel_cfg.get("required_failure_scan"), dict) else {}
        n_f_req = int(req_cfg.get("n_f_coarse", 101))
        f_fix = float(req_cfg.get("f_miss_fixed", float(cache.prior.mean)))

        def _unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, float]:
            x = np.asarray(x, dtype=float)
            dl = x[:k_snr]
            off = k_snr
            dz = None
            if include_z:
                dz = x[off : off + k_z]
                off += k_z
            bm = float(x[off]) if include_mass else 0.0
            return dl, dz, bm

        def _delta_lpd_obj(x: np.ndarray) -> float:
            # Objective (unpenalised): minimise Delta LPD under bounded nuisance deformation.
            x = np.asarray(x, dtype=float)
            dl, dz, bm = _unpack(x)
            alpha_mu, alpha_gr = apply_nuisance_to_alpha_logit_resummed(mom=mom, delta_logsnr_knots=dl, delta_z_knots=dz, b_mass=bm)
            if not (np.all(np.isfinite(alpha_mu)) and np.all(np.isfinite(alpha_gr))):
                return float("inf")
            lpd_mu, lpd_gr, _lpd_mu_data, _lpd_gr_data = _delta_lpd_fixed_f_miss(
                logL_cat_mu_by_event=cache.logL_cat_mu_by_event,
                logL_cat_gr_by_event=cache.logL_cat_gr_by_event,
                logL_missing_mu_by_event=cache.logL_missing_mu_by_event,
                logL_missing_gr_by_event=cache.logL_missing_gr_by_event,
                log_alpha_mu=np.log(alpha_mu),
                log_alpha_gr=np.log(alpha_gr),
                f_miss=f_fix,
            )
            return float(lpd_mu - lpd_gr)

        def _minimize(bound: float, *, req_cfg: dict[str, Any]) -> dict[str, Any]:
            bound = float(bound)
            bounds = [(-bound, bound)] * int(x0.size)
            maxiter = int(req_cfg.get("maxiter", 200))
            penalty_lambda = float(req_cfg.get("penalty_lambda", 1.0))
            ftol = float(req_cfg.get("ftol", 1e-6))
            xtol = float(req_cfg.get("xtol", 1e-4))
            n_random = int(req_cfg.get("n_random", 4000))
            seed = int(req_cfg.get("seed", args.seed))
            rng_local = np.random.default_rng(seed + int(round(bound * 10_000)))

            def _obj_pen(x: np.ndarray) -> float:
                v = float(_delta_lpd_obj(x))
                if not np.isfinite(v):
                    return float("inf")
                if penalty_lambda > 0.0 and np.isfinite(prior_sigma) and prior_sigma > 0.0:
                    v = v + float(penalty_lambda) * float(0.5 * np.sum((np.asarray(x, dtype=float) / float(prior_sigma)) ** 2))
                return float(v)

            # Coarse random search to avoid expensive numerical gradients through f_miss marginalisation.
            best_x = np.asarray(x0, dtype=float)
            best_obj = float(_obj_pen(best_x))
            for _ in range(max(0, n_random)):
                x = rng_local.uniform(low=-bound, high=bound, size=x0.size)
                v = float(_obj_pen(x))
                if np.isfinite(v) and v < best_obj:
                    best_obj = v
                    best_x = np.asarray(x, dtype=float)

            # Local refinement with a derivative-free bounded method.
            best_opt: scipy.optimize.OptimizeResult | None = None
            try:
                best_opt = scipy.optimize.minimize(
                    _obj_pen,
                    x0=np.asarray(best_x, dtype=float),
                    method="Powell",
                    bounds=bounds,
                    options={"maxiter": maxiter, "ftol": ftol, "xtol": xtol},
                )
            except Exception:
                best_opt = None

            if best_opt is None:
                # Fallback: just report the best random-search point.
                best_opt = scipy.optimize.OptimizeResult(x=best_x, fun=best_obj, success=False, message="random_search_only", nit=0, nfev=int(n_random) + 1)

            x = np.asarray(best_opt.x, dtype=float)
            # Enforce bounds for reporting (numerical safety).
            x = np.clip(x, -bound, bound)
            dl, dz, bm = _unpack(x)
            # Recompute both unpenalised Delta LPD and the penalised objective for reporting.
            dlp_fixed = float(_delta_lpd_obj(x))
            obj = float(_obj_pen(x))
            # Full marginalised score at the optimum point (production-like), for reporting.
            alpha_mu, alpha_gr = apply_nuisance_to_alpha_logit_resummed(mom=mom, delta_logsnr_knots=dl, delta_z_knots=dz, b_mass=bm)
            res_full = _delta_lpd_from_cached_terms(cache=cache, log_alpha_mu=np.log(alpha_mu), log_alpha_gr=np.log(alpha_gr))
            dlp_full = float(res_full.lpd_mu_total - res_full.lpd_gr_total)
            dlp_full_data = float(res_full.lpd_mu_total_data - res_full.lpd_gr_total_data)
            return {
                "bound": float(bound),
                "success": bool(best_opt.success),
                "message": str(getattr(best_opt, "message", "")),
                "nfev": int(getattr(best_opt, "nfev", -1)),
                "nit": int(getattr(best_opt, "nit", -1)),
                "n_random": int(n_random),
                "delta_lpd_total": float(dlp_full),
                "delta_lpd_total_data": float(dlp_full_data),
                "delta_lpd_total_fixed_f": float(dlp_fixed),
                "f_miss_fixed": float(f_fix),
                "objective_penalised": float(obj),
                "penalty_lambda": float(penalty_lambda),
                "prior_penalty": float(0.5 * np.sum((x / max(prior_sigma, 1e-12)) ** 2)),
                "max_abs_param": float(np.max(np.abs(x))) if x.size else 0.0,
                "at_bounds_frac": float(np.mean(np.isclose(np.abs(x), bound, rtol=0.0, atol=1e-6))) if x.size else 0.0,
                "dl_snr": [float(v) for v in dl.tolist()],
                "dl_z": None if dz is None else [float(v) for v in dz.tolist()],
                "b_mass": float(bm),
            }

        scan = []
        bounds_scan = [float(x) for x in req_cfg.get("delta_bounds", [delta_bound])]
        for b in bounds_scan:
            scan.append(_minimize(float(b), req_cfg=req_cfg))

        # Derive bound thresholds from the scan (coarse but deterministic).
        thresholds = [float(x) for x in req_cfg.get("thresholds", [1.0, 0.0])]
        req_bounds = {}
        for thr in thresholds:
            req = None
            for r in scan:
                if np.isfinite(float(r.get("delta_lpd_total", float("nan")))) and float(r["delta_lpd_total"]) <= float(thr):
                    req = float(r["bound"])
                    break
            req_bounds[str(thr)] = req
        _write_json_atomic(paths.tables_dir / "selection_required_failure_scan.json", {"scan": scan, "thresholds": req_cfg.get("thresholds", [1.0, 0.0])})

        sel_summary = {
            "enabled": True,
            "baseline_delta_lpd_total": float(base.lpd_mu_total - base.lpd_gr_total),
            "prior_sigma": float(prior_sigma),
            "scan": scan,
            "required_bounds": req_bounds,
        }
        _write_json_atomic(paths.tables_dir / "selection_nuisance_hardening_summary.json", sel_summary)

        try:
            fig, ax = plt.subplots(figsize=(6.4, 3.8))
            x = np.asarray([r["bound"] for r in scan], dtype=float)
            y = np.asarray([r["delta_lpd_total"] for r in scan], dtype=float)
            ax.plot(x, y, marker="o", lw=2.0)
            ax.axhline(1.0, color="k", lw=1.0, alpha=0.5)
            ax.axhline(0.0, color="k", lw=1.0, alpha=0.5)
            ax.set(xlabel="Selection nuisance bound (fractional)", ylabel="Adversarial min Delta LPD", title="Selection Deformation Required-Failure Scan")
            ax.grid(alpha=0.25, linestyle=":")
            fig.tight_layout()
            fig.savefig(paths.figures_dir / "selection_required_failure_scan.png", dpi=180)
            plt.close(fig)
        except Exception:
            pass

    # ---- Build spectral-only cached terms (used by several suites)
    spec_cat_mu: list[np.ndarray] = []
    spec_cat_gr: list[np.ndarray] = []
    spec_miss_mu: list[np.ndarray] = []
    spec_miss_gr: list[np.ndarray] = []
    for ev in cache.events:
        z, w, ipix, pe = _load_event_cache(ev)
        logL_mu, logL_gr = compute_dark_siren_logL_draws_from_pe_hist_fast(
            event=ev,
            pe=pe,
            z_grid_post=z_grid_post,
            dL_em_grid=dL_em_grid,
            R_grid=R_grid,
            z_gal=z,
            w_gal=w,
            ipix_gal=ipix,
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
    spec_score = _delta_lpd_from_terms(
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
    np.savez_compressed(
        paths.tables_dir / "spectral_only_cached_terms.npz",
        events=np.asarray(cache.events),
        logL_cat_mu=np.stack(spec_cat_mu, axis=0),
        logL_cat_gr=np.stack(spec_cat_gr, axis=0),
        logL_missing_mu=np.stack(spec_miss_mu, axis=0),
        logL_missing_gr=np.stack(spec_miss_gr, axis=0),
    )

    # ---- SECTION 1: Non-circular coherence validation (splits)
    noncirc_summary: dict[str, Any] = {"enabled": False}
    if bool(cfg.get("noncircular_splits", {}).get("enabled", False)):
        nc = cfg["noncircular_splits"]
        n_bins = int(nc.get("n_bins", 3))
        boundary = str(nc.get("epoch_boundary_date", "2019-11-01"))
        y, m, d = (int(x) for x in boundary.split("-"))
        epoch_boundary = date(y, m, d)
        gwosc_cache = Path(str(nc.get("gwosc_meta_cache", "data/cache/gw/gwosc_event_meta_gwtc3.json")))
        gwosc_meta = _ensure_gwosc_event_meta(events=cache.events, cache_path=REPO_ROOT / gwosc_cache)

        noncirc_summary = _noncircular_event_splits(
            cache=cache,
            cache_events_dir=cache.gap_run_root / "cache",
            gw_prior=gw_prior,
            epoch_boundary=epoch_boundary,
            gwosc_meta=gwosc_meta,
            z_bin_n=n_bins,
        )

        # Score each bin under the *full* cached terms and also a linearized alpha scan.
        alpha_grid = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0], dtype=float)
        alpha_global = _alpha_scan_linearized(cache=cache, alpha_grid=alpha_grid, events_mask=None, n_f=101)
        noncirc_summary["alpha_linearized_global"] = alpha_global

        scored: dict[str, Any] = {}
        for split_name, split in (noncirc_summary.get("splits") or {}).items():
            bins = split.get("bins") or []
            rows = []
            alpha_bins = []
            for b in bins:
                evs = [str(x) for x in (b.get("events") or [])]
                mask = np.asarray([ev in set(evs) for ev in cache.events], dtype=bool)
                n_in_bin = int(np.sum(mask))
                if n_in_bin <= 0:
                    rows.append(
                        {
                            "bin": b.get("bin"),
                            "n_events": 0,
                            "delta_lpd_total": float("nan"),
                            "delta_lpd_data": float("nan"),
                            "alpha_hat_argmax_linearized": float("nan"),
                            "alpha_post_mean_linearized": float("nan"),
                            "alpha_post_sd_linearized": float("nan"),
                        }
                    )
                    continue
                res = _delta_lpd_from_cached_terms(cache=cache, log_alpha_mu=cache.log_alpha_mu, log_alpha_gr=cache.log_alpha_gr, events_mask=mask)
                alpha_bin = _alpha_scan_linearized(cache=cache, alpha_grid=alpha_grid, events_mask=mask, n_f=101)
                rows.append(
                    {
                        "bin": b.get("bin"),
                        "n_events": n_in_bin,
                        "delta_lpd_total": float(res.lpd_mu_total - res.lpd_gr_total),
                        "delta_lpd_data": float(res.lpd_mu_total_data - res.lpd_gr_total_data),
                        "alpha_hat_argmax_linearized": float(alpha_bin.get("alpha_hat_argmax", float("nan"))),
                        "alpha_post_mean_linearized": float(alpha_bin.get("alpha_post_mean", float("nan"))),
                        "alpha_post_sd_linearized": float(alpha_bin.get("alpha_post_sd", float("nan"))),
                    }
                )
                alpha_bins.append(alpha_bin)
            scored[split_name] = {"rows": rows}

            # Coherence metric for alpha_hat across bins (linearized scan).
            try:
                a0 = float(alpha_global.get("alpha_post_mean", float("nan")))
                s0 = float(alpha_global.get("alpha_post_sd", float("nan")))
                vals = []
                for r in rows:
                    ai = float(r["alpha_post_mean_linearized"])
                    si = float(r["alpha_post_sd_linearized"])
                    if not (np.isfinite(ai) and np.isfinite(si) and np.isfinite(a0) and np.isfinite(s0)):
                        continue
                    denom = (si * si + s0 * s0) if (si * si + s0 * s0) > 0 else float("nan")
                    vals.append(((ai - a0) ** 2) / denom)
                scored[split_name]["coherence_metric_linearized"] = float(np.mean(vals)) if vals else float("nan")
            except Exception:
                scored[split_name]["coherence_metric_linearized"] = float("nan")

        noncirc_summary["scored_bins"] = scored
        _write_json_atomic(paths.tables_dir / "noncircular_splits_summary.json", noncirc_summary)

    # ---- SECTION 2: Permutation / null collapse tests (fast, cached-term based)
    perm_summary: dict[str, Any] = {"enabled": False}
    if bool(cfg.get("permutation_null", {}).get("enabled", False)):
        pcfg = cfg["permutation_null"]
        n_perm = int(pcfg.get("n_perm", 200))
        seed = int(pcfg.get("seed", args.seed))
        rngp = np.random.default_rng(seed)
        mode_cfg = pcfg.get("mode", "permute_catalog_terms_across_events")
        modes = [str(x) for x in mode_cfg] if isinstance(mode_cfg, list) else [str(mode_cfg)]
        n_f_coarse = int(pcfg.get("n_f_coarse", 101))
        n_ev = int(len(cache.events))

        runs: list[dict[str, Any]] = []

        for mode in modes:
            if mode == "permute_catalog_terms_across_events":
                obs = float(base.lpd_mu_total - base.lpd_gr_total)
                samples = []
                for _t in range(n_perm):
                    perm = rngp.permutation(n_ev)
                    cat_mu = [cache.logL_cat_mu_by_event[int(i)] for i in perm]
                    cat_gr = [cache.logL_cat_gr_by_event[int(i)] for i in perm]
                    miss_mu = cache.logL_missing_mu_by_event
                    miss_gr = cache.logL_missing_gr_by_event
                    res = _delta_lpd_from_terms(
                        logL_cat_mu_by_event=cat_mu,
                        logL_cat_gr_by_event=cat_gr,
                        logL_missing_mu_by_event=miss_mu,
                        logL_missing_gr_by_event=miss_gr,
                        log_alpha_mu=cache.log_alpha_mu,
                        log_alpha_gr=cache.log_alpha_gr,
                        prior=cache.prior,
                        n_f=n_f_coarse,
                        eps=cache.eps,
                    )
                    samples.append(float(res.lpd_mu_total - res.lpd_gr_total))
                arr = np.asarray(samples, dtype=float)
                p_tail = float((np.sum(arr >= obs) + 1.0) / (arr.size + 1.0)) if arr.size else float("nan")
                runs.append(
                    {
                        "mode": mode,
                        "n_perm": int(n_perm),
                        "n_f_coarse": int(n_f_coarse),
                        "seed": int(seed),
                        "observed_delta_lpd_total": float(obs),
                        "observed_delta_lpd_data": float(base.lpd_mu_total_data - base.lpd_gr_total_data),
                        "perm_mean_total": float(np.mean(arr)) if arr.size else float("nan"),
                        "perm_sd_total": float(np.std(arr)) if arr.size else float("nan"),
                        "p_perm_ge_observed_total": p_tail,
                    }
                )
                _write_csv(
                    paths.tables_dir / "permutation_null_samples.csv",
                    [{"mode": mode, "delta_lpd_total": float(x)} for x in samples],
                    fieldnames=["mode", "delta_lpd_total"],
                )
                try:
                    fig, ax = plt.subplots(figsize=(6.4, 3.8))
                    ax.hist(arr[np.isfinite(arr)], bins=30, color="C0", alpha=0.8)
                    ax.axvline(obs, color="C3", lw=2.0, label="observed")
                    ax.set(xlabel="Delta LPD (total)", ylabel="count", title="Permutation Null (catalog-term permutation)")
                    ax.grid(alpha=0.25, linestyle=":")
                    ax.legend(loc="best", frameon=False)
                    fig.tight_layout()
                    fig.savefig(paths.figures_dir / "permutation_null_hist.png", dpi=180)
                    plt.close(fig)
                except Exception:
                    pass
                continue

            if mode == "permute_missing_terms_across_events":
                # Swap missing-host logL vectors across events while keeping catalog terms fixed.
                # This targets whether event-specific missing-host structure is required to sustain the preference.
                obs = float(base.lpd_mu_total - base.lpd_gr_total)
                obs_data = float(base.lpd_mu_total_data - base.lpd_gr_total_data)
                samples_total = []
                samples_data = []
                for _t in range(n_perm):
                    perm = rngp.permutation(n_ev)
                    cat_mu = cache.logL_cat_mu_by_event
                    cat_gr = cache.logL_cat_gr_by_event
                    miss_mu = [cache.logL_missing_mu_by_event[int(i)] for i in perm]
                    miss_gr = [cache.logL_missing_gr_by_event[int(i)] for i in perm]
                    res = _delta_lpd_from_terms(
                        logL_cat_mu_by_event=cat_mu,
                        logL_cat_gr_by_event=cat_gr,
                        logL_missing_mu_by_event=miss_mu,
                        logL_missing_gr_by_event=miss_gr,
                        log_alpha_mu=cache.log_alpha_mu,
                        log_alpha_gr=cache.log_alpha_gr,
                        prior=cache.prior,
                        n_f=n_f_coarse,
                        eps=cache.eps,
                    )
                    samples_total.append(float(res.lpd_mu_total - res.lpd_gr_total))
                    samples_data.append(float(res.lpd_mu_total_data - res.lpd_gr_total_data))
                arr_total = np.asarray(samples_total, dtype=float)
                arr_data = np.asarray(samples_data, dtype=float)
                p_tail_total = float((np.sum(arr_total >= obs) + 1.0) / (arr_total.size + 1.0)) if arr_total.size else float("nan")
                p_tail_data = float((np.sum(arr_data >= obs_data) + 1.0) / (arr_data.size + 1.0)) if arr_data.size else float("nan")
                runs.append(
                    {
                        "mode": mode,
                        "n_perm": int(n_perm),
                        "n_f_coarse": int(n_f_coarse),
                        "seed": int(seed),
                        "observed_delta_lpd_total": float(obs),
                        "observed_delta_lpd_data": float(obs_data),
                        "perm_mean_total": float(np.mean(arr_total)) if arr_total.size else float("nan"),
                        "perm_sd_total": float(np.std(arr_total)) if arr_total.size else float("nan"),
                        "p_perm_ge_observed_total": p_tail_total,
                        "perm_mean_data": float(np.mean(arr_data)) if arr_data.size else float("nan"),
                        "perm_sd_data": float(np.std(arr_data)) if arr_data.size else float("nan"),
                        "p_perm_ge_observed_data": p_tail_data,
                        "notes": "Permutation null: swap missing-host terms across events (catalog terms fixed).",
                    }
                )
                _write_csv(
                    paths.tables_dir / "missing_host_swap_null_samples.csv",
                    [{"delta_lpd_total": float(a), "delta_lpd_data": float(b)} for a, b in zip(samples_total, samples_data, strict=True)],
                    fieldnames=["delta_lpd_total", "delta_lpd_data"],
                )
                try:
                    fig, ax = plt.subplots(figsize=(6.4, 3.8))
                    ax.hist(arr_total[np.isfinite(arr_total)], bins=30, color="C0", alpha=0.8)
                    ax.axvline(obs, color="C3", lw=2.0, label="observed")
                    ax.set(xlabel="Delta LPD (total)", ylabel="count", title="Null: missing-host term swap (across events)")
                    ax.grid(alpha=0.25, linestyle=":")
                    ax.legend(loc="best", frameon=False)
                    fig.tight_layout()
                    fig.savefig(paths.figures_dir / "missing_host_swap_null_hist.png", dpi=180)
                    plt.close(fig)
                except Exception:
                    pass
                continue

            if mode == "scramble_catalog_redshift_hist_spectral_only":
                # This null targets the dominant spectral channel by destroying the coherent mapping
                # between catalogue redshift structure and each event's distance posterior.
                from entropy_horizon_recon.dark_sirens_incompleteness import _host_prior_matrix_from_precompute  # noqa: SLF001
                from entropy_horizon_recon.dark_sirens_incompleteness import MissingHostPriorPrecompute  # noqa: SLF001

                z_max = float(cache.manifest.get("gal_z_max", 0.3))
                zhist_n = int(pcfg.get("z_hist_nbins", 200))
                z_edges = np.linspace(0.0, z_max, zhist_n + 1)
                z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])
                n_draws = int(dL_em_grid.shape[0])

                # Precompute distances and inverse priors at z-bin centres for all draws.
                dL_em_zc = np.vstack([np.interp(z_cent, z_grid_post, dL_em_grid[j]) for j in range(n_draws)])
                R_zc = np.vstack([np.interp(z_cent, z_grid_post, R_grid[j]) for j in range(n_draws)])
                dL_gw_zc = dL_em_zc * R_zc
                inv_pi_em_zc = np.exp(-gw_prior.log_pi_dL(np.clip(dL_em_zc, 1e-6, np.inf)))
                inv_pi_gw_zc = np.exp(-gw_prior.log_pi_dL(np.clip(dL_gw_zc, 1e-6, np.inf)))

                # Per-event precompute: weight hist + distance-bin pdf evaluation matrices.
                ev_w_hist: list[np.ndarray] = []
                ev_term_mu: list[np.ndarray] = []
                ev_term_gr: list[np.ndarray] = []
                ev_miss_mu: list[np.ndarray] = []
                ev_miss_gr: list[np.ndarray] = []
                ev_pdf_1d: list[np.ndarray] = []
                ev_dmid: list[np.ndarray] = []
                ev_widths: list[np.ndarray] = []

                for ev in cache.events:
                    z, w, ipix, pe = _load_event_cache(ev)
                    npix = int(12 * int(pe.nside) * int(pe.nside))
                    pix_to_row = np.full((npix,), -1, dtype=np.int32)
                    pix_to_row[np.asarray(pe.pix_sel, dtype=np.int64)] = np.arange(int(pe.pix_sel.size), dtype=np.int32)
                    row = pix_to_row[ipix]
                    good = (row >= 0) & np.isfinite(z) & (z > 0.0) & np.isfinite(w) & (w > 0.0)
                    prob = np.asarray(pe.prob_pix, dtype=float)[row[good]] if np.any(good) else np.asarray([], dtype=float)
                    weight = (w[good] * prob) if np.any(good) else np.asarray([], dtype=float)
                    hist_raw, _ = np.histogram(np.clip(z[good], 0.0, z_max), bins=z_edges, weights=weight)
                    hist_raw = np.asarray(hist_raw, dtype=float)
                    s_raw = float(np.sum(hist_raw))
                    if not (np.isfinite(s_raw) and s_raw > 0.0):
                        hist_raw = np.ones_like(hist_raw, dtype=float)
                    ev_w_hist.append(hist_raw)

                    edges, pdf_1d = _pe_sky_marginal_pdf_1d(pe)
                    widths = np.diff(edges)
                    dmid = 0.5 * (edges[:-1] + edges[1:])
                    ev_pdf_1d.append(np.asarray(pdf_1d, dtype=float))
                    ev_dmid.append(np.asarray(dmid, dtype=float))
                    ev_widths.append(np.asarray(widths, dtype=float))

                    def _pdf_at_dL(dL: np.ndarray) -> np.ndarray:
                        dL = np.asarray(dL, dtype=float)
                        bidx = np.searchsorted(edges, dL, side="right") - 1
                        valid = (bidx >= 0) & (bidx < pdf_1d.size) & np.isfinite(dL) & (dL > 0.0)
                        bidx = np.clip(bidx, 0, pdf_1d.size - 1)
                        pdf_at = pdf_1d[bidx]
                        return np.where(valid, pdf_at, 0.0)

                    pdf_mu = _pdf_at_dL(dL_gw_zc)  # (n_draws, n_zbins)
                    pdf_gr = _pdf_at_dL(dL_em_zc)
                    ev_term_mu.append(np.asarray(pdf_mu * inv_pi_gw_zc, dtype=float))
                    ev_term_gr.append(np.asarray(pdf_gr * inv_pi_em_zc, dtype=float))

                    # Missing-host terms (spectral-only): use host-prior matrices on this event's dL grid.
                    host_mu = _host_prior_matrix_from_precompute(pre_missing, dL_grid=dmid, model="mu", gw_distance_prior=gw_prior) * widths.reshape((1, -1))
                    host_gr = _host_prior_matrix_from_precompute(pre_missing, dL_grid=dmid, model="gr", gw_distance_prior=gw_prior) * widths.reshape((1, -1))
                    Lm = np.clip(host_mu @ pdf_1d, 1e-300, np.inf)
                    Lg = np.clip(host_gr @ pdf_1d, 1e-300, np.inf)
                    ev_miss_mu.append(np.log(Lm))
                    ev_miss_gr.append(np.log(Lg))

                # Cache the baseline catalog terms under the binned-z approximation (used by multiple nulls).
                cat_mu_base = []
                cat_gr_base = []
                for w_hist, tmu, tgr in zip(ev_w_hist, ev_term_mu, ev_term_gr, strict=True):
                    Lmu = np.clip(tmu @ np.asarray(w_hist, dtype=float), 1e-300, np.inf)
                    Lgr = np.clip(tgr @ np.asarray(w_hist, dtype=float), 1e-300, np.inf)
                    cat_mu_base.append(np.log(Lmu))
                    cat_gr_base.append(np.log(Lgr))

                def _score_for_w_hists(w_hists: list[np.ndarray], *, miss_mu: list[np.ndarray] | None = None, miss_gr: list[np.ndarray] | None = None) -> MarginalizedFMissResult:
                    cat_mu = []
                    cat_gr = []
                    for w_hist, tmu, tgr in zip(w_hists, ev_term_mu, ev_term_gr, strict=True):
                        Lmu = np.clip(tmu @ np.asarray(w_hist, dtype=float), 1e-300, np.inf)
                        Lgr = np.clip(tgr @ np.asarray(w_hist, dtype=float), 1e-300, np.inf)
                        cat_mu.append(np.log(Lmu))
                        cat_gr.append(np.log(Lgr))
                    return _delta_lpd_from_terms(
                        logL_cat_mu_by_event=cat_mu,
                        logL_cat_gr_by_event=cat_gr,
                        logL_missing_mu_by_event=ev_miss_mu if miss_mu is None else miss_mu,
                        logL_missing_gr_by_event=ev_miss_gr if miss_gr is None else miss_gr,
                        log_alpha_mu=cache.log_alpha_mu,
                        log_alpha_gr=cache.log_alpha_gr,
                        prior=cache.prior,
                        n_f=n_f_coarse,
                        eps=cache.eps,
                    )

                # Baseline score under this binned-z approximation (for internal p-value calibration).
                base_res = _score_for_w_hists(ev_w_hist)
                obs_total = float(base_res.lpd_mu_total - base_res.lpd_gr_total)
                obs_data = float(base_res.lpd_mu_total_data - base_res.lpd_gr_total_data)

                # Permutation ensemble.
                samples_total = []
                samples_data = []
                for _t in range(n_perm):
                    w_perm = [rngp.permutation(w) for w in ev_w_hist]
                    res = _score_for_w_hists(w_perm)
                    samples_total.append(float(res.lpd_mu_total - res.lpd_gr_total))
                    samples_data.append(float(res.lpd_mu_total_data - res.lpd_gr_total_data))

                arr_total = np.asarray(samples_total, dtype=float)
                arr_data = np.asarray(samples_data, dtype=float)
                p_tail_total = float((np.sum(arr_total >= obs_total) + 1.0) / (arr_total.size + 1.0)) if arr_total.size else float("nan")
                p_tail_data = float((np.sum(arr_data >= obs_data) + 1.0) / (arr_data.size + 1.0)) if arr_data.size else float("nan")

                runs.append(
                    {
                        "mode": mode,
                        "n_perm": int(n_perm),
                        "n_f_coarse": int(n_f_coarse),
                        "seed": int(seed),
                        "observed_delta_lpd_total": float(obs_total),
                        "observed_delta_lpd_data": float(obs_data),
                        "baseline_ref_delta_lpd_spectral_only_total": float(spec_score.lpd_mu_total - spec_score.lpd_gr_total),
                        "perm_mean_total": float(np.mean(arr_total)) if arr_total.size else float("nan"),
                        "perm_sd_total": float(np.std(arr_total)) if arr_total.size else float("nan"),
                        "p_perm_ge_observed_total": p_tail_total,
                        "perm_mean_data": float(np.mean(arr_data)) if arr_data.size else float("nan"),
                        "perm_sd_data": float(np.std(arr_data)) if arr_data.size else float("nan"),
                        "p_perm_ge_observed_data": p_tail_data,
                        "notes": "binned-z spectral-only approximation; use observed (approx) vs scrambled ensemble for structure-dependence.",
                    }
                )

                _write_csv(
                    paths.tables_dir / "redshift_scramble_null_samples.csv",
                    [{"delta_lpd_total": float(a), "delta_lpd_data": float(b)} for a, b in zip(samples_total, samples_data, strict=True)],
                    fieldnames=["delta_lpd_total", "delta_lpd_data"],
                )
                try:
                    fig, ax = plt.subplots(figsize=(6.4, 3.8))
                    ax.hist(arr_data[np.isfinite(arr_data)], bins=30, color="C0", alpha=0.8)
                    ax.axvline(obs_data, color="C3", lw=2.0, label="observed (approx)")
                    ax.set(xlabel="Delta LPD (data-only)", ylabel="count", title="Null: redshift-scramble (spectral-only)")
                    ax.grid(alpha=0.25, linestyle=":")
                    ax.legend(loc="best", frameon=False)
                    fig.tight_layout()
                    fig.savefig(paths.figures_dir / "redshift_scramble_null_hist.png", dpi=180)
                    plt.close(fig)
                except Exception:
                    pass
                continue

            if mode == "permute_catalog_redshift_hist_across_events_spectral_only":
                # Reassign entire binned-z weight histograms between events, keeping per-histogram totals intact.
                # This destroys event-by-event coherence between distance posteriors and catalogue redshift structure.
                from entropy_horizon_recon.dark_sirens_incompleteness import _host_prior_matrix_from_precompute  # noqa: SLF001

                z_max = float(cache.manifest.get("gal_z_max", 0.3))
                zhist_n = int(pcfg.get("z_hist_nbins", 200))
                z_edges = np.linspace(0.0, z_max, zhist_n + 1)
                z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])
                n_draws = int(dL_em_grid.shape[0])

                dL_em_zc = np.vstack([np.interp(z_cent, z_grid_post, dL_em_grid[j]) for j in range(n_draws)])
                R_zc = np.vstack([np.interp(z_cent, z_grid_post, R_grid[j]) for j in range(n_draws)])
                dL_gw_zc = dL_em_zc * R_zc
                inv_pi_em_zc = np.exp(-gw_prior.log_pi_dL(np.clip(dL_em_zc, 1e-6, np.inf)))
                inv_pi_gw_zc = np.exp(-gw_prior.log_pi_dL(np.clip(dL_gw_zc, 1e-6, np.inf)))

                ev_w_hist: list[np.ndarray] = []
                ev_term_mu: list[np.ndarray] = []
                ev_term_gr: list[np.ndarray] = []
                ev_miss_mu: list[np.ndarray] = []
                ev_miss_gr: list[np.ndarray] = []

                for ev in cache.events:
                    z, w, ipix, pe = _load_event_cache(ev)
                    npix = int(12 * int(pe.nside) * int(pe.nside))
                    pix_to_row = np.full((npix,), -1, dtype=np.int32)
                    pix_to_row[np.asarray(pe.pix_sel, dtype=np.int64)] = np.arange(int(pe.pix_sel.size), dtype=np.int32)
                    row = pix_to_row[ipix]
                    good = (row >= 0) & np.isfinite(z) & (z > 0.0) & np.isfinite(w) & (w > 0.0)
                    prob = np.asarray(pe.prob_pix, dtype=float)[row[good]] if np.any(good) else np.asarray([], dtype=float)
                    weight = (w[good] * prob) if np.any(good) else np.asarray([], dtype=float)
                    hist_raw, _ = np.histogram(np.clip(z[good], 0.0, z_max), bins=z_edges, weights=weight)
                    hist_raw = np.asarray(hist_raw, dtype=float)
                    s_raw = float(np.sum(hist_raw))
                    if not (np.isfinite(s_raw) and s_raw > 0.0):
                        hist_raw = np.ones_like(hist_raw, dtype=float)
                    ev_w_hist.append(hist_raw)

                    edges, pdf_1d = _pe_sky_marginal_pdf_1d(pe)

                    def _pdf_at_dL(dL: np.ndarray) -> np.ndarray:
                        dL = np.asarray(dL, dtype=float)
                        bidx = np.searchsorted(edges, dL, side="right") - 1
                        valid = (bidx >= 0) & (bidx < pdf_1d.size) & np.isfinite(dL) & (dL > 0.0)
                        bidx = np.clip(bidx, 0, pdf_1d.size - 1)
                        pdf_at = pdf_1d[bidx]
                        return np.where(valid, pdf_at, 0.0)

                    pdf_mu = _pdf_at_dL(dL_gw_zc)  # (n_draws, n_zbins)
                    pdf_gr = _pdf_at_dL(dL_em_zc)
                    ev_term_mu.append(np.asarray(pdf_mu * inv_pi_gw_zc, dtype=float))
                    ev_term_gr.append(np.asarray(pdf_gr * inv_pi_em_zc, dtype=float))

                    widths = np.diff(edges)
                    dmid = 0.5 * (edges[:-1] + edges[1:])
                    host_mu = _host_prior_matrix_from_precompute(pre_missing, dL_grid=dmid, model="mu", gw_distance_prior=gw_prior) * widths.reshape((1, -1))
                    host_gr = _host_prior_matrix_from_precompute(pre_missing, dL_grid=dmid, model="gr", gw_distance_prior=gw_prior) * widths.reshape((1, -1))
                    Lm = np.clip(host_mu @ pdf_1d, 1e-300, np.inf)
                    Lg = np.clip(host_gr @ pdf_1d, 1e-300, np.inf)
                    ev_miss_mu.append(np.log(Lm))
                    ev_miss_gr.append(np.log(Lg))

                def _score_for_w_hists(w_hists: list[np.ndarray]) -> MarginalizedFMissResult:
                    cat_mu = []
                    cat_gr = []
                    for w_hist, tmu, tgr in zip(w_hists, ev_term_mu, ev_term_gr, strict=True):
                        Lmu = np.clip(tmu @ np.asarray(w_hist, dtype=float), 1e-300, np.inf)
                        Lgr = np.clip(tgr @ np.asarray(w_hist, dtype=float), 1e-300, np.inf)
                        cat_mu.append(np.log(Lmu))
                        cat_gr.append(np.log(Lgr))
                    return _delta_lpd_from_terms(
                        logL_cat_mu_by_event=cat_mu,
                        logL_cat_gr_by_event=cat_gr,
                        logL_missing_mu_by_event=ev_miss_mu,
                        logL_missing_gr_by_event=ev_miss_gr,
                        log_alpha_mu=cache.log_alpha_mu,
                        log_alpha_gr=cache.log_alpha_gr,
                        prior=cache.prior,
                        n_f=n_f_coarse,
                        eps=cache.eps,
                    )

                base_res = _score_for_w_hists(ev_w_hist)
                obs_total = float(base_res.lpd_mu_total - base_res.lpd_gr_total)
                obs_data = float(base_res.lpd_mu_total_data - base_res.lpd_gr_total_data)

                samples_total = []
                samples_data = []
                for _t in range(n_perm):
                    perm = rngp.permutation(n_ev)
                    w_perm = [ev_w_hist[int(i)] for i in perm]
                    res = _score_for_w_hists(w_perm)
                    samples_total.append(float(res.lpd_mu_total - res.lpd_gr_total))
                    samples_data.append(float(res.lpd_mu_total_data - res.lpd_gr_total_data))

                arr_total = np.asarray(samples_total, dtype=float)
                arr_data = np.asarray(samples_data, dtype=float)
                p_tail_total = float((np.sum(arr_total >= obs_total) + 1.0) / (arr_total.size + 1.0)) if arr_total.size else float("nan")
                p_tail_data = float((np.sum(arr_data >= obs_data) + 1.0) / (arr_data.size + 1.0)) if arr_data.size else float("nan")

                runs.append(
                    {
                        "mode": mode,
                        "n_perm": int(n_perm),
                        "n_f_coarse": int(n_f_coarse),
                        "seed": int(seed),
                        "observed_delta_lpd_total": float(obs_total),
                        "observed_delta_lpd_data": float(obs_data),
                        "perm_mean_total": float(np.mean(arr_total)) if arr_total.size else float("nan"),
                        "perm_sd_total": float(np.std(arr_total)) if arr_total.size else float("nan"),
                        "p_perm_ge_observed_total": p_tail_total,
                        "perm_mean_data": float(np.mean(arr_data)) if arr_data.size else float("nan"),
                        "perm_sd_data": float(np.std(arr_data)) if arr_data.size else float("nan"),
                        "p_perm_ge_observed_data": p_tail_data,
                        "notes": "binned-z spectral-only approximation; cross-event reassignment of entire redshift-weight histograms.",
                    }
                )

                _write_csv(
                    paths.tables_dir / "catalog_hist_swap_null_samples.csv",
                    [{"delta_lpd_total": float(a), "delta_lpd_data": float(b)} for a, b in zip(samples_total, samples_data, strict=True)],
                    fieldnames=["delta_lpd_total", "delta_lpd_data"],
                )
                try:
                    fig, ax = plt.subplots(figsize=(6.4, 3.8))
                    ax.hist(arr_data[np.isfinite(arr_data)], bins=30, color="C0", alpha=0.8)
                    ax.axvline(obs_data, color="C3", lw=2.0, label="observed (approx)")
                    ax.set(xlabel="Delta LPD (data-only)", ylabel="count", title="Null: cross-event histogram swap (spectral-only)")
                    ax.grid(alpha=0.25, linestyle=":")
                    ax.legend(loc="best", frameon=False)
                    fig.tight_layout()
                    fig.savefig(paths.figures_dir / "catalog_hist_swap_null_hist.png", dpi=180)
                    plt.close(fig)
                except Exception:
                    pass
                continue

            if mode == "scramble_missing_host_prior_basez_spectral_only":
                # Recompute missing-host terms after scrambling the redshift structure of the host prior.
                # Keep PE distance posteriors and catalog (binned-z) terms fixed.
                from entropy_horizon_recon.dark_sirens_incompleteness import _host_prior_matrix_from_precompute  # noqa: SLF001
                from entropy_horizon_recon.dark_sirens_incompleteness import MissingHostPriorPrecompute  # noqa: SLF001

                z_max = float(cache.manifest.get("gal_z_max", 0.3))
                zhist_n = int(pcfg.get("z_hist_nbins", 200))
                z_edges = np.linspace(0.0, z_max, zhist_n + 1)
                z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])
                n_draws = int(dL_em_grid.shape[0])
                n_z_pre = int(pre_missing.z_grid.size)

                dL_em_zc = np.vstack([np.interp(z_cent, z_grid_post, dL_em_grid[j]) for j in range(n_draws)])
                R_zc = np.vstack([np.interp(z_cent, z_grid_post, R_grid[j]) for j in range(n_draws)])
                dL_gw_zc = dL_em_zc * R_zc
                inv_pi_em_zc = np.exp(-gw_prior.log_pi_dL(np.clip(dL_em_zc, 1e-6, np.inf)))
                inv_pi_gw_zc = np.exp(-gw_prior.log_pi_dL(np.clip(dL_gw_zc, 1e-6, np.inf)))

                ev_w_hist: list[np.ndarray] = []
                ev_term_mu: list[np.ndarray] = []
                ev_term_gr: list[np.ndarray] = []
                ev_pdf_1d: list[np.ndarray] = []
                ev_dmid: list[np.ndarray] = []
                ev_widths: list[np.ndarray] = []

                for ev in cache.events:
                    z, w, ipix, pe = _load_event_cache(ev)
                    npix = int(12 * int(pe.nside) * int(pe.nside))
                    pix_to_row = np.full((npix,), -1, dtype=np.int32)
                    pix_to_row[np.asarray(pe.pix_sel, dtype=np.int64)] = np.arange(int(pe.pix_sel.size), dtype=np.int32)
                    row = pix_to_row[ipix]
                    good = (row >= 0) & np.isfinite(z) & (z > 0.0) & np.isfinite(w) & (w > 0.0)
                    prob = np.asarray(pe.prob_pix, dtype=float)[row[good]] if np.any(good) else np.asarray([], dtype=float)
                    weight = (w[good] * prob) if np.any(good) else np.asarray([], dtype=float)
                    hist_raw, _ = np.histogram(np.clip(z[good], 0.0, z_max), bins=z_edges, weights=weight)
                    hist_raw = np.asarray(hist_raw, dtype=float)
                    s_raw = float(np.sum(hist_raw))
                    if not (np.isfinite(s_raw) and s_raw > 0.0):
                        hist_raw = np.ones_like(hist_raw, dtype=float)
                    ev_w_hist.append(hist_raw)

                    edges, pdf_1d = _pe_sky_marginal_pdf_1d(pe)
                    widths = np.diff(edges)
                    dmid = 0.5 * (edges[:-1] + edges[1:])
                    ev_pdf_1d.append(np.asarray(pdf_1d, dtype=float))
                    ev_dmid.append(np.asarray(dmid, dtype=float))
                    ev_widths.append(np.asarray(widths, dtype=float))

                    def _pdf_at_dL(dL: np.ndarray) -> np.ndarray:
                        dL = np.asarray(dL, dtype=float)
                        bidx = np.searchsorted(edges, dL, side="right") - 1
                        valid = (bidx >= 0) & (bidx < pdf_1d.size) & np.isfinite(dL) & (dL > 0.0)
                        bidx = np.clip(bidx, 0, pdf_1d.size - 1)
                        pdf_at = pdf_1d[bidx]
                        return np.where(valid, pdf_at, 0.0)

                    pdf_mu = _pdf_at_dL(dL_gw_zc)
                    pdf_gr = _pdf_at_dL(dL_em_zc)
                    ev_term_mu.append(np.asarray(pdf_mu * inv_pi_gw_zc, dtype=float))
                    ev_term_gr.append(np.asarray(pdf_gr * inv_pi_em_zc, dtype=float))

                # Baseline binned-z score under the standard missing-host prior.
                cat_mu_base = []
                cat_gr_base = []
                for w_hist, tmu, tgr in zip(ev_w_hist, ev_term_mu, ev_term_gr, strict=True):
                    Lmu = np.clip(tmu @ np.asarray(w_hist, dtype=float), 1e-300, np.inf)
                    Lgr = np.clip(tgr @ np.asarray(w_hist, dtype=float), 1e-300, np.inf)
                    cat_mu_base.append(np.log(Lmu))
                    cat_gr_base.append(np.log(Lgr))

                # Baseline missing-host terms.
                miss_mu_base = []
                miss_gr_base = []
                for dmid, widths, pdf_1d in zip(ev_dmid, ev_widths, ev_pdf_1d, strict=True):
                    host_mu = _host_prior_matrix_from_precompute(pre_missing, dL_grid=dmid, model="mu", gw_distance_prior=gw_prior) * widths.reshape((1, -1))
                    host_gr = _host_prior_matrix_from_precompute(pre_missing, dL_grid=dmid, model="gr", gw_distance_prior=gw_prior) * widths.reshape((1, -1))
                    Lm = np.clip(host_mu @ pdf_1d, 1e-300, np.inf)
                    Lg = np.clip(host_gr @ pdf_1d, 1e-300, np.inf)
                    miss_mu_base.append(np.log(Lm))
                    miss_gr_base.append(np.log(Lg))

                base_res = _delta_lpd_from_terms(
                    logL_cat_mu_by_event=cat_mu_base,
                    logL_cat_gr_by_event=cat_gr_base,
                    logL_missing_mu_by_event=miss_mu_base,
                    logL_missing_gr_by_event=miss_gr_base,
                    log_alpha_mu=cache.log_alpha_mu,
                    log_alpha_gr=cache.log_alpha_gr,
                    prior=cache.prior,
                    n_f=n_f_coarse,
                    eps=cache.eps,
                )
                obs_total = float(base_res.lpd_mu_total - base_res.lpd_gr_total)
                obs_data = float(base_res.lpd_mu_total_data - base_res.lpd_gr_total_data)

                samples_total = []
                samples_data = []
                for _t in range(n_perm):
                    perm_z = rngp.permutation(n_z_pre)
                    pre_scr = MissingHostPriorPrecompute(
                        z_grid=np.asarray(pre_missing.z_grid, dtype=float),
                        dL_em=np.asarray(pre_missing.dL_em, dtype=float),
                        dL_gw=np.asarray(pre_missing.dL_gw, dtype=float),
                        base_z=np.asarray(pre_missing.base_z, dtype=float)[:, perm_z],
                        ddLdz_em=np.asarray(pre_missing.ddLdz_em, dtype=float),
                        ddLdz_gw=np.asarray(pre_missing.ddLdz_gw, dtype=float),
                    )
                    miss_mu = []
                    miss_gr = []
                    for dmid, widths, pdf_1d in zip(ev_dmid, ev_widths, ev_pdf_1d, strict=True):
                        host_mu = _host_prior_matrix_from_precompute(pre_scr, dL_grid=dmid, model="mu", gw_distance_prior=gw_prior) * widths.reshape((1, -1))
                        host_gr = _host_prior_matrix_from_precompute(pre_scr, dL_grid=dmid, model="gr", gw_distance_prior=gw_prior) * widths.reshape((1, -1))
                        Lm = np.clip(host_mu @ pdf_1d, 1e-300, np.inf)
                        Lg = np.clip(host_gr @ pdf_1d, 1e-300, np.inf)
                        miss_mu.append(np.log(Lm))
                        miss_gr.append(np.log(Lg))

                    res = _delta_lpd_from_terms(
                        logL_cat_mu_by_event=cat_mu_base,
                        logL_cat_gr_by_event=cat_gr_base,
                        logL_missing_mu_by_event=miss_mu,
                        logL_missing_gr_by_event=miss_gr,
                        log_alpha_mu=cache.log_alpha_mu,
                        log_alpha_gr=cache.log_alpha_gr,
                        prior=cache.prior,
                        n_f=n_f_coarse,
                        eps=cache.eps,
                    )
                    samples_total.append(float(res.lpd_mu_total - res.lpd_gr_total))
                    samples_data.append(float(res.lpd_mu_total_data - res.lpd_gr_total_data))

                arr_total = np.asarray(samples_total, dtype=float)
                arr_data = np.asarray(samples_data, dtype=float)
                p_tail_total = float((np.sum(arr_total >= obs_total) + 1.0) / (arr_total.size + 1.0)) if arr_total.size else float("nan")
                p_tail_data = float((np.sum(arr_data >= obs_data) + 1.0) / (arr_data.size + 1.0)) if arr_data.size else float("nan")

                runs.append(
                    {
                        "mode": mode,
                        "n_perm": int(n_perm),
                        "n_f_coarse": int(n_f_coarse),
                        "seed": int(seed),
                        "observed_delta_lpd_total": float(obs_total),
                        "observed_delta_lpd_data": float(obs_data),
                        "perm_mean_total": float(np.mean(arr_total)) if arr_total.size else float("nan"),
                        "perm_sd_total": float(np.std(arr_total)) if arr_total.size else float("nan"),
                        "p_perm_ge_observed_total": p_tail_total,
                        "perm_mean_data": float(np.mean(arr_data)) if arr_data.size else float("nan"),
                        "perm_sd_data": float(np.std(arr_data)) if arr_data.size else float("nan"),
                        "p_perm_ge_observed_data": p_tail_data,
                        "notes": "binned-z spectral-only approximation; missing-host prior base_z(z) scrambled (distance posteriors fixed).",
                    }
                )
                _write_csv(
                    paths.tables_dir / "missing_host_prior_scramble_null_samples.csv",
                    [{"delta_lpd_total": float(a), "delta_lpd_data": float(b)} for a, b in zip(samples_total, samples_data, strict=True)],
                    fieldnames=["delta_lpd_total", "delta_lpd_data"],
                )
                try:
                    fig, ax = plt.subplots(figsize=(6.4, 3.8))
                    ax.hist(arr_total[np.isfinite(arr_total)], bins=30, color="C0", alpha=0.8)
                    ax.axvline(obs_total, color="C3", lw=2.0, label="observed (approx)")
                    ax.set(xlabel="Delta LPD (total)", ylabel="count", title="Null: missing-host prior scramble (spectral-only)")
                    ax.grid(alpha=0.25, linestyle=":")
                    ax.legend(loc="best", frameon=False)
                    fig.tight_layout()
                    fig.savefig(paths.figures_dir / "missing_host_prior_scramble_null_hist.png", dpi=180)
                    plt.close(fig)
                except Exception:
                    pass
                continue

            raise ValueError(f"Unknown permutation_null.mode: {mode}")

        perm_summary = {"enabled": True, "runs": runs}
        _write_json_atomic(paths.tables_dir / "permutation_null_summary.json", perm_summary)

    # ---- SECTION 3.2: Global catalog/photo-z stress (top-N events covering target leverage)
    global_a2_summary: dict[str, Any] = {"enabled": False}
    if bool(cfg.get("catalog_photoz_stress_global", {}).get("enabled", False)):
        gcfg = cfg["catalog_photoz_stress_global"]
        leverage_target = float(gcfg.get("target_leverage_fraction", 0.8))
        min_events = int(gcfg.get("min_events", 5))
        max_events = int(gcfg.get("max_events", 12))
        z_pivot = float(gcfg.get("z_pivot", 0.15))
        gal_z_max = float(cache.manifest.get("gal_z_max", 0.3))

        ev_scores = json.loads((cache.gap_run_root / "tables" / f"event_scores_{cache.run_label}.json").read_text(encoding="utf-8"))
        ev_scores = [e for e in ev_scores if isinstance(e, dict) and isinstance(e.get("event"), str)]
        ev_scores.sort(key=lambda e: float(e.get("delta_lpd", 0.0)), reverse=True)
        dsum = float(np.sum([float(e.get("delta_lpd", 0.0)) for e in ev_scores]))
        top_events = []
        acc = 0.0
        for e in ev_scores:
            if not np.isfinite(dsum) or dsum <= 0.0:
                break
            top_events.append(str(e["event"]))
            acc += float(e.get("delta_lpd", 0.0))
            if (acc / dsum) >= leverage_target and len(top_events) >= min_events:
                break
        top_events = top_events[: max(min_events, min(max_events, len(top_events)))]

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

        def _score_with_replacements(repl: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
            cat_mu = list(base_cat_mu)
            cat_gr = list(base_cat_gr)
            for ev, (mu, gr) in repl.items():
                j = ev_to_idx[ev]
                cat_mu[j] = np.asarray(mu, dtype=float)
                cat_gr[j] = np.asarray(gr, dtype=float)
            res = _delta_lpd_from_terms(
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

        b0_grid = [float(x) for x in gcfg.get("photoz_b0_grid", [0.0])]
        b1_grid = [float(x) for x in gcfg.get("photoz_b1_grid", [0.0])]
        c_amp_grid = [float(x) for x in gcfg.get("comp_amp_grid", [0.0])]
        c_tilt_grid = [float(x) for x in gcfg.get("comp_tilt_grid", [0.0])]

        rows_photoz = []
        for b0 in b0_grid:
            for b1 in b1_grid:
                repl = {}
                for ev in top_events:
                    z, w, ipix, pe = _load_event_cache(ev)
                    z2 = z + float(b0) + float(b1) * z
                    z2 = np.clip(z2, 1e-6, gal_z_max)
                    logL_mu, logL_gr = compute_dark_siren_logL_draws_from_pe_hist_fast(
                        event=ev,
                        pe=pe,
                        z_grid_post=z_grid_post,
                        dL_em_grid=dL_em_grid,
                        R_grid=R_grid,
                        z_gal=z2,
                        w_gal=w,
                        ipix_gal=ipix,
                        gw_distance_prior=gw_prior,
                        distance_mode="spectral_only",
                        gal_chunk_size=gal_chunk_size,
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
                    fz = (1.0 + float(ca)) * (1.0 + float(ct) * ((z - z_pivot) / max(z_pivot, 1e-6)))
                    w2 = w * np.clip(fz, 0.1, 10.0)
                    logL_mu, logL_gr = compute_dark_siren_logL_draws_from_pe_hist_fast(
                        event=ev,
                        pe=pe,
                        z_grid_post=z_grid_post,
                        dL_em_grid=dL_em_grid,
                        R_grid=R_grid,
                        z_gal=z,
                        w_gal=w2,
                        ipix_gal=ipix,
                        gw_distance_prior=gw_prior,
                        distance_mode="spectral_only",
                        gal_chunk_size=gal_chunk_size,
                    )
                    repl[ev] = (logL_mu, logL_gr)
                dlp = _score_with_replacements(repl)
                rows_comp.append({"c_amp": float(ca), "c_tilt": float(ct), "delta_lpd_total": float(dlp)})

        base_spec = float(spec_score.lpd_mu_total - spec_score.lpd_gr_total)
        min_photoz = min(rows_photoz, key=lambda r: float(r["delta_lpd_total"])) if rows_photoz else None
        min_comp = min(rows_comp, key=lambda r: float(r["delta_lpd_total"])) if rows_comp else None
        global_a2_summary = {
            "enabled": True,
            "mode": "spectral_only",
            "top_events": top_events,
            "target_leverage_fraction": float(leverage_target),
            "baseline_delta_lpd_total": float(base_spec),
            "photoz_grid": {"b0": b0_grid, "b1": b1_grid},
            "photoz_min": min_photoz,
            "comp_grid": {"c_amp": c_amp_grid, "c_tilt": c_tilt_grid, "z_pivot": z_pivot},
            "comp_min": min_comp,
        }
        _write_json_atomic(paths.tables_dir / "catalog_photoz_stress_global_summary.json", global_a2_summary)
        _write_csv(paths.tables_dir / "catalog_photoz_stress_global_photoz_grid.csv", rows_photoz, fieldnames=["b0", "b1", "delta_lpd_total"])
        _write_csv(paths.tables_dir / "catalog_photoz_stress_global_comp_grid.csv", rows_comp, fieldnames=["c_amp", "c_tilt", "delta_lpd_total"])

    # ---- SECTION 3.3: PE / waveform robustness (analysis swaps for top-leverage events)
    a3_summary: dict[str, Any] = {"enabled": False}
    if bool(cfg.get("pe_waveform_stress", {}).get("enabled", False)):
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

        a3_rows: list[dict[str, Any]] = []
        for ev in top_events:
            # Read the PE file path from the cached term meta (authoritative).
            meta_path = cache.gap_run_root / "cache_terms" / f"cat_{ev}__{cache.run_label}.npz"
            with np.load(meta_path, allow_pickle=True) as d:
                meta = json.loads(str(d["meta"].tolist()))
            pe_file = Path(str(meta.get("pe_file", ""))).expanduser()
            if not pe_file.exists():
                raise FileNotFoundError(f"{ev}: pe_file not found: {pe_file}")

            z_gal, w_gal, ipix_gal, pe_cached = _load_event_cache(ev)

            # Load baseline spectral-only cached terms for replacement scoring.
            spec_npz = paths.tables_dir / "spectral_only_cached_terms.npz"
            with np.load(spec_npz, allow_pickle=False) as d:
                events = [str(x) for x in d["events"].tolist()]
                base_cat_mu = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_cat_mu"], dtype=float)]
                base_cat_gr = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_cat_gr"], dtype=float)]
                base_miss_mu = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_missing_mu"], dtype=float)]
                base_miss_gr = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_missing_gr"], dtype=float)]
            ev_to_idx = {e: i for i, e in enumerate(events)}
            if ev not in ev_to_idx:
                continue

            def _rescore_one_event(*, cat_mu: np.ndarray, cat_gr: np.ndarray, miss_mu: np.ndarray, miss_gr: np.ndarray) -> float:
                mu = list(base_cat_mu)
                gr = list(base_cat_gr)
                mmu = list(base_miss_mu)
                mgr = list(base_miss_gr)
                j = ev_to_idx[ev]
                mu[j] = np.asarray(cat_mu, dtype=float)
                gr[j] = np.asarray(cat_gr, dtype=float)
                mmu[j] = np.asarray(miss_mu, dtype=float)
                mgr[j] = np.asarray(miss_gr, dtype=float)
                res = _delta_lpd_from_terms(
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

            for analysis in analyses_pref:
                try:
                    ra, dec, dL, pe_meta = load_gwtc_pe_sky_samples(path=pe_file, analysis=str(analysis), max_samples=max_samples, seed=seed)
                except Exception:
                    continue
                pe_hist = build_pe_pixel_distance_histogram(
                    ra_rad=ra,
                    dec_rad=dec,
                    dL_mpc=dL,
                    nside=int(cache.manifest.get("pe_nside", 64)),
                    p_credible=float(cache.manifest.get("p_credible", 0.9)),
                    dl_nbins=int(cache.manifest.get("pe_dl_nbins", 64)),
                    dl_min_mpc=float(np.asarray(pe_cached.dL_edges, dtype=float)[0]),
                    dl_max_mpc=float(np.asarray(pe_cached.dL_edges, dtype=float)[-1]),
                    dl_qmin=float(cache.manifest.get("pe_dl_qmin", 0.001)),
                    dl_qmax=float(cache.manifest.get("pe_dl_qmax", 0.999)),
                    dl_pad_factor=float(cache.manifest.get("pe_dl_pad_factor", 1.2)),
                    dl_pseudocount=float(cache.manifest.get("pe_dl_pseudocount", 0.05)),
                    dl_smooth_iters=int(cache.manifest.get("pe_dl_smooth_iters", 2)),
                    nest=True,
                )
                logL_mu, logL_gr = compute_dark_siren_logL_draws_from_pe_hist_fast(
                    event=ev,
                    pe=pe_hist,
                    z_grid_post=z_grid_post,
                    dL_em_grid=dL_em_grid,
                    R_grid=R_grid,
                    z_gal=z_gal,
                    w_gal=w_gal,
                    ipix_gal=ipix_gal,
                    gw_distance_prior=gw_prior,
                    distance_mode="spectral_only",
                    gal_chunk_size=gal_chunk_size,
                )
                miss_mu, miss_gr = compute_missing_host_logL_draws_from_histogram(
                    prob_pix=np.asarray(pe_hist.prob_pix, dtype=float),
                    pdf_bins=np.asarray(pe_hist.pdf_bins, dtype=float),
                    dL_edges=np.asarray(pe_hist.dL_edges, dtype=float),
                    pre=pre_missing,
                    gw_distance_prior=gw_prior,
                    distance_mode="spectral_only",
                    pixel_chunk_size=pix_chunk_size,
                )
                dlp = _rescore_one_event(cat_mu=logL_mu, cat_gr=logL_gr, miss_mu=miss_mu, miss_gr=miss_gr)
                a3_rows.append({"event": ev, "analysis": str(pe_meta.analysis), "n_pe_used": int(pe_meta.n_used), "delta_lpd_total": float(dlp)})

        a3_summary = {"enabled": True, "top_events": top_events, "rows": a3_rows}
        _write_json_atomic(paths.tables_dir / "pe_waveform_stress_summary.json", a3_summary)
        if a3_rows:
            _write_csv(paths.tables_dir / "pe_waveform_stress_rows.csv", a3_rows, fieldnames=["event", "analysis", "n_pe_used", "delta_lpd_total"])

    # ---- SECTION 4: Injection-based sanity checks (lightweight, spectral-only)
    # We support both MG-truth recovery and a GR-truth null calibration using the same fast generator.
    mg_inj_summary: dict[str, Any] = {"enabled": False}
    gr_inj_summary: dict[str, Any] = {"enabled": False}
    mg_enabled = bool(cfg.get("mg_injection_check", {}).get("enabled", False))
    gr_enabled = bool(cfg.get("gr_injection_check", {}).get("enabled", False))
    if mg_enabled or gr_enabled:
        # Use MG config as the default template, but allow the GR block to override.
        icfg = cfg.get("mg_injection_check", {}) if mg_enabled else cfg.get("gr_injection_check", {})
        n_rep_default = int(icfg.get("n_rep", 64))
        seed_default = int(icfg.get("seed", args.seed))
        zhist_n = int(icfg.get("z_hist_nbins", 200))
        sigma_floor = float(icfg.get("sigma_floor_frac", 0.05))

        # Precompute per-event spectral-only z-weight histograms (catalog weights only),
        # and per-event missing-host host-matrix factors so injections are fast.
        from entropy_horizon_recon.dark_sirens_incompleteness import _host_prior_matrix_from_precompute  # noqa: SLF001

        z_max = float(cache.manifest.get("gal_z_max", 0.3))
        z_edges = np.linspace(0.0, z_max, zhist_n + 1)
        z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])

        ev_weight_hist_raw: dict[str, np.ndarray] = {}
        ev_weight_hist_norm: dict[str, np.ndarray] = {}
        ev_pe_edges: dict[str, np.ndarray] = {}
        ev_pe_pdf: dict[str, dict[str, float]] = {}
        ev_host_mu_w: dict[str, np.ndarray] = {}
        ev_host_gr_w: dict[str, np.ndarray] = {}
        for ev in cache.events:
            z, w, ipix, pe = _load_event_cache(ev)
            npix = int(12 * int(pe.nside) * int(pe.nside))
            pix_to_row = np.full((npix,), -1, dtype=np.int32)
            pix_to_row[np.asarray(pe.pix_sel, dtype=np.int64)] = np.arange(int(pe.pix_sel.size), dtype=np.int32)
            row = pix_to_row[ipix]
            good = (row >= 0) & np.isfinite(z) & (z > 0.0) & np.isfinite(w) & (w > 0.0)
            prob = np.asarray(pe.prob_pix, dtype=float)[row[good]] if np.any(good) else np.asarray([], dtype=float)
            weight = (w[good] * prob) if np.any(good) else np.asarray([], dtype=float)
            hist_raw, _ = np.histogram(np.clip(z[good], 0.0, z_max), bins=z_edges, weights=weight)
            hist_raw = np.asarray(hist_raw, dtype=float)
            s_raw = float(np.sum(hist_raw))
            if not (np.isfinite(s_raw) and s_raw > 0.0):
                hist_raw = np.ones_like(hist_raw, dtype=float)
                s_raw = float(np.sum(hist_raw))
            ev_weight_hist_raw[ev] = hist_raw
            ev_weight_hist_norm[ev] = hist_raw / s_raw

            edges, pdf_1d = _pe_sky_marginal_pdf_1d(pe)
            ev_pe_edges[ev] = edges
            # approximate sigma from the sky-marginal PE pdf
            widths = np.diff(edges)
            dmid = 0.5 * (edges[:-1] + edges[1:])
            m = np.isfinite(pdf_1d) & (pdf_1d > 0.0)
            if np.any(m):
                mean = float(np.sum(pdf_1d[m] * dmid[m] * widths[m]))
                var = float(np.sum(pdf_1d[m] * (dmid[m] - mean) ** 2 * widths[m]))
                sig = float(np.sqrt(max(var, 0.0)))
            else:
                mean = float("nan")
                sig = float("nan")
            ev_pe_pdf[ev] = {"edges": edges, "sigma_mpc": sig, "mean_mpc": mean if np.isfinite(sig) else float("nan")}

            # Precompute missing-host factors on this event's dL grid (spectral-only projection uses these).
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

        def _run_injection_suite(*, truth: str, n_rep: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
            if truth not in ("mg", "gr"):
                raise ValueError("truth must be 'mg' or 'gr'.")
            n_rep = int(n_rep)
            seed = int(seed)
            rngi = np.random.default_rng(seed)

            # Draw indices for truth selection.
            truth_draws = rngi.integers(low=0, high=n_draws, size=n_rep)

            deltas: list[float] = []
            for rep in range(n_rep):
                jtruth = int(truth_draws[rep])
                cat_mu_by_event = []
                cat_gr_by_event = []
                miss_mu_by_event = []
                miss_gr_by_event = []
                for ev in cache.events:
                    # Sample z_true from the event's (catalog-only) z-weight histogram.
                    pz = np.asarray(ev_weight_hist_norm[ev], dtype=float)
                    b = int(rngi.choice(np.arange(pz.size), p=pz))
                    z_lo = float(z_edges[b])
                    z_hi = float(z_edges[b + 1])
                    z_true = float(rngi.uniform(z_lo, z_hi))

                    # Truth distances.
                    dL_em_true = float(np.interp(z_true, z_grid_post, dL_em_grid[jtruth]))
                    if truth == "mg":
                        R_true = float(np.interp(z_true, z_grid_post, R_grid[jtruth]))
                    else:
                        R_true = 1.0
                    dL_gw_true = float(dL_em_true * R_true)

                    # Synthetic sky-marginal distance likelihood: Gaussian in dL on the event's histogram support.
                    edges = np.asarray(ev_pe_edges[ev], dtype=float)
                    widths = np.diff(edges)
                    dmid = 0.5 * (edges[:-1] + edges[1:])
                    sig0 = float(ev_pe_pdf[ev].get("sigma_mpc", float("nan")))
                    sig = float(max(sig0, sigma_floor * dL_gw_true)) if np.isfinite(sig0) else float(max(sigma_floor * dL_gw_true, 1.0))
                    pdf = np.exp(-0.5 * ((dmid - dL_gw_true) / max(sig, 1e-6)) ** 2)
                    pdf = np.asarray(pdf, dtype=float)
                    norm = float(np.sum(pdf * widths))
                    if not (np.isfinite(norm) and norm > 0.0):
                        pdf = np.ones_like(pdf) / float(np.sum(widths))
                    else:
                        pdf = pdf / norm

                    w_hist = np.asarray(ev_weight_hist_raw[ev], dtype=float)
                    if w_hist.shape != z_cent.shape:
                        raise RuntimeError("Internal error: z-hist shape mismatch.")

                    def _logL_cat_from_dL(dL: np.ndarray, inv_pi: np.ndarray) -> np.ndarray:
                        dL = np.asarray(dL, dtype=float)
                        inv_pi = np.asarray(inv_pi, dtype=float)
                        bidx = np.searchsorted(edges, dL, side="right") - 1
                        valid = (bidx >= 0) & (bidx < pdf.size) & np.isfinite(dL) & (dL > 0.0)
                        bidx = np.clip(bidx, 0, pdf.size - 1)
                        pdf_at = pdf[bidx]
                        pdf_at = np.where(valid, pdf_at, 0.0)
                        term = pdf_at * inv_pi
                        L = np.clip(term @ w_hist, 1e-300, np.inf)
                        return np.log(L)

                    logL_mu = _logL_cat_from_dL(dL_gw_zc, inv_pi_gw_zc)
                    logL_gr = _logL_cat_from_dL(dL_em_zc, inv_pi_em_zc)

                    host_mu_w = ev_host_mu_w[ev]
                    host_gr_w = ev_host_gr_w[ev]
                    Lm = np.clip(host_mu_w @ pdf, 1e-300, np.inf)
                    Lg = np.clip(host_gr_w @ pdf, 1e-300, np.inf)
                    miss_mu = np.log(Lm)
                    miss_gr = np.log(Lg)

                    cat_mu_by_event.append(np.asarray(logL_mu, dtype=float))
                    cat_gr_by_event.append(np.asarray(logL_gr, dtype=float))
                    miss_mu_by_event.append(np.asarray(miss_mu, dtype=float))
                    miss_gr_by_event.append(np.asarray(miss_gr, dtype=float))

                res = _delta_lpd_from_terms(
                    logL_cat_mu_by_event=cat_mu_by_event,
                    logL_cat_gr_by_event=cat_gr_by_event,
                    logL_missing_mu_by_event=miss_mu_by_event,
                    logL_missing_gr_by_event=miss_gr_by_event,
                    log_alpha_mu=cache.log_alpha_mu,
                    log_alpha_gr=cache.log_alpha_gr,
                    prior=cache.prior,
                    n_f=101,
                    eps=cache.eps,
                )
                deltas.append(float(res.lpd_mu_total - res.lpd_gr_total))

            arr = np.asarray(deltas, dtype=float)
            summary = {
                "enabled": True,
                "truth": str(truth),
                "n_rep": int(n_rep),
                "seed": int(seed),
                "delta_lpd_mean": float(np.mean(arr)) if arr.size else float("nan"),
                "delta_lpd_p16": float(np.quantile(arr, 0.16)) if arr.size else float("nan"),
                "delta_lpd_p50": float(np.quantile(arr, 0.50)) if arr.size else float("nan"),
                "delta_lpd_p84": float(np.quantile(arr, 0.84)) if arr.size else float("nan"),
                "delta_lpd_max": float(np.max(arr)) if arr.size else float("nan"),
            }
            return arr, summary

        if mg_enabled:
            mcfg = cfg["mg_injection_check"]
            arr, summary = _run_injection_suite(truth="mg", n_rep=int(mcfg.get("n_rep", n_rep_default)), seed=int(mcfg.get("seed", seed_default)))
            mg_inj_summary = summary
            _write_json_atomic(paths.tables_dir / "mg_injection_check_summary.json", summary)
            _write_csv(paths.tables_dir / "mg_injection_check_samples.csv", [{"delta_lpd_total": float(x)} for x in arr.tolist()], fieldnames=["delta_lpd_total"])
            try:
                fig, ax = plt.subplots(figsize=(6.4, 3.8))
                ax.hist(arr[np.isfinite(arr)], bins=30, color="C2", alpha=0.8)
                ax.axvline(0.0, color="k", lw=1.0, alpha=0.6)
                ax.set(xlabel="Delta LPD (total)", ylabel="count", title="MG-Truth Injection Recovery (spectral-only)")
                ax.grid(alpha=0.25, linestyle=":")
                fig.tight_layout()
                fig.savefig(paths.figures_dir / "mg_injection_check_hist.png", dpi=180)
                plt.close(fig)
            except Exception:
                pass

        if gr_enabled:
            gcfg = cfg["gr_injection_check"]
            arr, summary = _run_injection_suite(truth="gr", n_rep=int(gcfg.get("n_rep", n_rep_default)), seed=int(gcfg.get("seed", seed_default)))
            gr_inj_summary = summary
            _write_json_atomic(paths.tables_dir / "gr_injection_check_summary.json", summary)
            _write_csv(paths.tables_dir / "gr_injection_check_samples.csv", [{"delta_lpd_total": float(x)} for x in arr.tolist()], fieldnames=["delta_lpd_total"])
            try:
                fig, ax = plt.subplots(figsize=(6.4, 3.8))
                ax.hist(arr[np.isfinite(arr)], bins=30, color="C1", alpha=0.8)
                ax.axvline(0.0, color="k", lw=1.0, alpha=0.6)
                ax.set(xlabel="Delta LPD (total)", ylabel="count", title="GR-Truth Injection Null (spectral-only)")
                ax.grid(alpha=0.25, linestyle=":")
                fig.tight_layout()
                fig.savefig(paths.figures_dir / "gr_injection_check_hist.png", dpi=180)
                plt.close(fig)
            except Exception:
                pass

    # ---- SECTION 5: External interpretability memo (lensing response mapping)
    if bool(cfg.get("interpretability", {}).get("enabled", False)):
        lens_path = repo_root / "outputs" / "hubble_tension_mg_lensing_refit_camb32_20260210_live" / "tables" / "summary.json"
        memo = []
        memo.append("# Interpretability Memo: Lensing Response Parameters\n\n")
        if lens_path.exists():
            j = _read_json(lens_path)
            stats = (j.get("fit_parameter_stats") or {})
            ratio_p50 = float((stats.get("mstar2_ratio_0") or {}).get("p50", float("nan")))
            tilt_p50 = float((stats.get("ell_tilt") or {}).get("p50", float("nan")))
            memo.append(f"- From the constrained Planck-lensing response refit, the median effective Planck-mass ratio is `M_*^2(0)/M_*^2(early) ≈ {ratio_p50:.3f}`.\n")
            memo.append(f"- The corresponding phenomenological multipole-tilt parameter has median `ell_tilt ≈ {tilt_p50:.3f}` (pivot `L=200`).\n\n")
            if np.isfinite(ratio_p50) and ratio_p50 > 0:
                dln = float(np.log(ratio_p50))
                memo.append(f"Interpreted in EFT-of-DE language, this corresponds to an integrated Planck-mass running of order `Δ ln M_*^2 ≈ {dln:+.3f}` between early times and today. In Horndeski/EFT notation this is sourced by `α_M(a) = d ln M_*^2 / d ln a`; the present analysis does not reconstruct `α_M(a)` directly, but the inferred integrated change is at the few×10% level.\n\n")
            memo.append("Caveats:\n")
            memo.append("- The refit is a constrained phenomenological response (amplitude + scale-tilt) applied to the lensing-reconstruction likelihood only; it is not a full Boltzmann evolution of modified perturbation equations.\n")
            memo.append("- A physically complete mapping to `(α_M, α_B, α_K, α_T)` requires specifying a covariant model and refitting TT/TE/EE plus lensing jointly.\n\n")
            memo.append("Qualitative consistency check:\n")
            memo.append("- The fitted response removes the baseline `C_L^{φφ}` suppression without extreme parameter excursions, and remains consistent with the primary-spectrum closure preference for `A_L > 1` noted in the Letter.\n")
        else:
            memo.append(f"Expected lensing refit summary not found at `{lens_path}`.\n")
        (paths.out_dir / "interpretability.md").write_text("".join(memo), encoding="utf-8")

    # ---- SECTION 6: Predictions / falsifiers note (tie to suite outputs)
    if bool(cfg.get("predictions", {}).get("enabled", False)):
        preds = []
        preds.append("# Predictions / Falsifiers (From This Hardening Suite)\n\n")
        preds.append("This note records checks that would falsify the modified-propagation interpretation.\n\n")
        preds.append("Predictions:\n")
        preds.append("1. As the siren sample grows (O4/O5), leverage concentration should decrease: the joint preference should not remain dominated by a single event.\n")
        preds.append("2. High-distance bins should continue to contribute disproportionately if the effect is cumulative in propagation distance.\n")
        preds.append("3. Detector/network splits should not induce sign flips; if the preference is physical propagation, it should persist across HL and HLV subsets.\n\n")
        preds.append("Falsifiers:\n")
        preds.append("1. If improved catalogue modelling (completeness/photo-z) within realistic external priors erases the preference (Delta LPD -> 0), the signal is likely systematic.\n")
        preds.append("2. If the permutation null does not degrade the preference (i.e., the observed Delta LPD is typical under scrambled event-catalog associations), the interpretation as coherent catalogue-redshift structure would weaken.\n")
        preds.append("3. If non-circular splits show strong incoherence (alpha_hat drift far beyond its posterior uncertainty), it would argue for model misspecification rather than a single universal propagation history.\n")
        (paths.out_dir / "predictions.md").write_text("".join(preds), encoding="utf-8")

    # ---- Report (compact; full details in JSON/CSV tables)
    md = []
    md.append("# Dark-Siren Hardening Suite (O3)\n")
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
        md.append("## 3.1 Selection Nuisance (Required-Failure Scan)\n")
        rows = [["baseline", f"{float(sel_summary.get('baseline_delta_lpd_total', float('nan'))):+.3f}"]]
        for r in sel_summary.get("scan", [])[:10]:
            rows.append([f"bound={float(r['bound']):.2f}", f"{float(r['delta_lpd_total']):+.3f}"])
        md.append(format_table(rows, headers=["case", "adversarial min ΔLPD"]))
        rb = sel_summary.get("required_bounds") or {}
        if rb:
            md.append(f"- Required bound for ΔLPD<1: `{rb.get('1.0')}`; for ΔLPD<0: `{rb.get('0.0')}`")
        md.append("- Plot: `figures/selection_required_failure_scan.png`")
        md.append("")

    md.append("## Spectral-Only Baseline (Mechanism Channel)\n")
    md.append(format_table([["ΔLPD (spectral-only)", f"{float(spec_score.lpd_mu_total - spec_score.lpd_gr_total):+.3f}"]], headers=["item", "value"]))
    md.append("")

    if noncirc_summary.get("enabled"):
        md.append("## 1.1 Non-Circular Splits (Frozen Assignments)\n")
        scored = noncirc_summary.get("scored_bins", {}) or {}
        for split_name, block in scored.items():
            rows = []
            for r in (block.get("rows") or [])[:12]:
                rows.append([r.get("bin"), r.get("n_events"), f"{float(r.get('delta_lpd_total', float('nan'))):+.3f}", f"{float(r.get('alpha_post_mean_linearized', float('nan'))):.2f}"])
            md.append(f"### {split_name}\n")
            if rows:
                md.append(format_table(rows, headers=["bin", "n", "ΔLPD", "alpha_hat (lin., mean)"]))
                md.append(f"- coherence_metric_linearized: `{float(block.get('coherence_metric_linearized', float('nan'))):.3f}`")
            md.append("")

    if perm_summary.get("enabled"):
        md.append("## 2. Permutation / Null Collapse Tests\n")
        runs = perm_summary.get("runs", []) or []
        for r in runs:
            mode = str(r.get("mode", ""))
            md.append(f"### {mode}\n")
            rows = [
                ["observed ΔLPD (total)", f"{float(r.get('observed_delta_lpd_total', float('nan'))):+.3f}"],
                ["observed ΔLPD (data-only)", f"{float(r.get('observed_delta_lpd_data', float('nan'))):+.3f}"],
                ["null mean (total)", f"{float(r.get('perm_mean_total', float('nan'))):+.3f}"],
                ["null sd (total)", f"{float(r.get('perm_sd_total', float('nan'))):.3f}"],
                ["p(null ≥ obs) total", f"{float(r.get('p_perm_ge_observed_total', float('nan'))):.4f}"],
            ]
            if "perm_mean_data" in r:
                rows += [
                    ["null mean (data-only)", f"{float(r.get('perm_mean_data', float('nan'))):+.3f}"],
                    ["null sd (data-only)", f"{float(r.get('perm_sd_data', float('nan'))):.3f}"],
                    ["p(null ≥ obs) data", f"{float(r.get('p_perm_ge_observed_data', float('nan'))):.4f}"],
                ]
            md.append(format_table(rows, headers=["item", "value"]))
            if mode == "permute_catalog_terms_across_events":
                md.append("- Plot: `figures/permutation_null_hist.png`")
            if mode == "scramble_catalog_redshift_hist_spectral_only":
                md.append("- Plot: `figures/redshift_scramble_null_hist.png`")
            if mode == "permute_missing_terms_across_events":
                md.append("- Plot: `figures/missing_host_swap_null_hist.png`")
            if mode == "permute_catalog_redshift_hist_across_events_spectral_only":
                md.append("- Plot: `figures/catalog_hist_swap_null_hist.png`")
            if mode == "scramble_missing_host_prior_basez_spectral_only":
                md.append("- Plot: `figures/missing_host_prior_scramble_null_hist.png`")
            md.append("")

    if global_a2_summary.get("enabled"):
        md.append("## 3.2 Global Catalogue/Photo-z Stress (Top-N Leverage Events)\n")
        rows = [
            ["top events", ", ".join(global_a2_summary.get("top_events", [])[:8]) + (" ..." if len(global_a2_summary.get("top_events", [])) > 8 else "")],
            ["baseline ΔLPD (spectral-only)", f"{float(global_a2_summary.get('baseline_delta_lpd_total', float('nan'))):+.3f}"],
            ["min ΔLPD on photo-z grid", f"{float((global_a2_summary.get('photoz_min') or {}).get('delta_lpd_total', float('nan'))):+.3f}"],
            ["min ΔLPD on completeness grid", f"{float((global_a2_summary.get('comp_min') or {}).get('delta_lpd_total', float('nan'))):+.3f}"],
        ]
        md.append(format_table(rows, headers=["item", "value"]))
        md.append("")

    if a3_summary.get("enabled"):
        md.append("## 3.3 PE / Waveform Robustness (Analysis Swaps)\n")
        rows = [["top events", ", ".join(a3_summary.get("top_events", [])[:6])]]
        vals = [float(r.get("delta_lpd_total", float("nan"))) for r in a3_summary.get("rows", [])]
        vals = [v for v in vals if np.isfinite(v)]
        rows.append(["min ΔLPD across swaps", f"{float(min(vals)):+.3f}" if vals else "nan"])
        md.append(format_table(rows, headers=["item", "value"]))
        if (paths.tables_dir / "pe_waveform_stress_rows.csv").exists():
            md.append("- Full rows: `tables/pe_waveform_stress_rows.csv`")
        md.append("")

    if mg_inj_summary.get("enabled"):
        md.append("## 4.1 Injection Consistency Check (MG truth; spectral-only)\n")
        rows = [
            ["n_rep", int(mg_inj_summary.get("n_rep", 0))],
            ["mean ΔLPD", f"{float(mg_inj_summary.get('delta_lpd_mean', float('nan'))):+.3f}"],
            ["p16/p50/p84", f"{float(mg_inj_summary.get('delta_lpd_p16', float('nan'))):+.3f} / {float(mg_inj_summary.get('delta_lpd_p50', float('nan'))):+.3f} / {float(mg_inj_summary.get('delta_lpd_p84', float('nan'))):+.3f}"],
        ]
        md.append(format_table(rows, headers=["item", "value"]))
        md.append("- Plot: `figures/mg_injection_check_hist.png`")
        md.append("")

    if gr_inj_summary.get("enabled"):
        md.append("## 4.2 Injection Null Calibration (GR truth; spectral-only)\n")
        rows = [
            ["n_rep", int(gr_inj_summary.get("n_rep", 0))],
            ["mean ΔLPD", f"{float(gr_inj_summary.get('delta_lpd_mean', float('nan'))):+.3f}"],
            ["p16/p50/p84", f"{float(gr_inj_summary.get('delta_lpd_p16', float('nan'))):+.3f} / {float(gr_inj_summary.get('delta_lpd_p50', float('nan'))):+.3f} / {float(gr_inj_summary.get('delta_lpd_p84', float('nan'))):+.3f}"],
            ["max ΔLPD", f"{float(gr_inj_summary.get('delta_lpd_max', float('nan'))):+.3f}"],
        ]
        md.append(format_table(rows, headers=["item", "value"]))
        md.append("- Plot: `figures/gr_injection_check_hist.png`")
        md.append("")

        # Calibrated p/Z table against the GR-truth injection null.
        try:
            from scipy.stats import norm as _norm  # type: ignore

            arr = []
            p_csv = paths.tables_dir / "gr_injection_check_samples.csv"
            if p_csv.exists():
                # Minimal CSV loader (single column).
                with p_csv.open("r", encoding="utf-8") as f:
                    next(f, None)
                    for line in f:
                        s = line.strip().split(",")[0]
                        try:
                            arr.append(float(s))
                        except Exception:
                            pass
            arr = [x for x in arr if np.isfinite(x)]
            n0 = int(len(arr))

            # Observed scores to compare (dominant-channel calibrated; see notes in the report).
            obs_baseline = float(base.lpd_mu_total - base.lpd_gr_total)
            obs_sel_min_20 = float("nan")
            for r in (sel_summary.get("scan") or []):
                if abs(float(r.get("bound", -1.0)) - 0.2) < 5e-4:
                    obs_sel_min_20 = float(r.get("delta_lpd_total", float("nan")))
                    break
            obs_photoz_min = float((global_a2_summary.get("photoz_min") or {}).get("delta_lpd_total", float("nan")))

            def _pz(obs: float) -> tuple[float, float]:
                if n0 <= 0 or not np.isfinite(obs):
                    return float("nan"), float("nan")
                # Conservative +1 smoothing.
                k = int(np.sum(np.asarray(arr, dtype=float) >= float(obs)))
                p = float((k + 1.0) / (n0 + 1.0))
                z = float(_norm.isf(p)) if (np.isfinite(p) and p > 0.0 and p < 1.0) else float("nan")
                return p, z

            rows = []
            for label, obs in [
                ("baseline (full)", obs_baseline),
                ("selection min (bound=0.20)", obs_sel_min_20),
                ("photo-z stressed min", obs_photoz_min),
            ]:
                p, z = _pz(obs)
                rows.append([label, f"{obs:+.3f}" if np.isfinite(obs) else "nan", f"{p:.4g}" if np.isfinite(p) else "nan", f"{z:.2f}" if np.isfinite(z) else "nan"])

            md.append("## 2.1 Calibrated Significance (GR-truth Injection Null)\n")
            md.append(format_table(rows, headers=["case", "ΔLPD", "p (one-sided)", "Z (one-sided)"]))
            md.append(f"- Notes: p/Z are computed against `n={n0}` GR-truth injections from Section 4.2 (spectral-only generator; dominant-channel calibration).")
            md.append("")

            _write_csv(
                paths.tables_dir / "calibrated_pz_table.csv",
                [{"case": r[0], "delta_lpd": r[1], "p_one_sided": r[2], "z_one_sided": r[3]} for r in rows],
                fieldnames=["case", "delta_lpd", "p_one_sided", "z_one_sided"],
            )
        except Exception:
            pass

    if (paths.out_dir / "interpretability.md").exists():
        md.append("## 5 Interpretability Memo\n")
        md.append("- Note: `interpretability.md`")
        md.append("")
    if (paths.out_dir / "predictions.md").exists():
        md.append("## 6 Predictions / Falsifiers\n")
        md.append("- Note: `predictions.md`")
        md.append("")

    # Executive memo (one-page integration note).
    try:
        memo = []
        memo.append("# Executive Memo: Dark-Siren Hardening Suite (O3)\n\n")
        memo.append(f"- Output: `{paths.out_dir.name}`\n")
        memo.append(f"- Baseline ΔLPD (full cached terms): `{float(base.lpd_mu_total - base.lpd_gr_total):+.3f}`\n")
        memo.append(f"- Spectral-only ΔLPD: `{float(spec_score.lpd_mu_total - spec_score.lpd_gr_total):+.3f}`\n\n")

        if perm_summary.get("enabled"):
            memo.append("## Structure-Dependence Nulls (Empirical)\n\n")
            for r in (perm_summary.get("runs") or []):
                mode = str(r.get("mode", ""))
                p = float(r.get("p_perm_ge_observed_total", float("nan")))
                memo.append(
                    f"- `{mode}`: observed `{float(r.get('observed_delta_lpd_total', float('nan'))):+.3f}`, "
                    f"null mean `{float(r.get('perm_mean_total', float('nan'))):+.3f} ± {float(r.get('perm_sd_total', float('nan'))):.3f}`, "
                    f"p `{p:.4f}`\n"
                )
            memo.append("\n")

        if noncirc_summary.get("enabled"):
            memo.append("## Non-Circular Coherence (Frozen Assignments)\n\n")
            scored = (noncirc_summary.get("scored_bins") or {})
            for split_name in ("distance_bins", "catalog_z_bins", "network", "epoch", "chirp_mass_bins"):
                block = scored.get(split_name) or {}
                cm = float(block.get("coherence_metric_linearized", float("nan")))
                memo.append(f"- `{split_name}` coherence metric (linearised α scan): `{cm:.3f}`\n")
            memo.append("\n")

        if sel_summary.get("enabled"):
            memo.append("## Required-Failure (Selection Deformation)\n\n")
            rb = sel_summary.get("required_bounds") or {}
            memo.append(f"- Required bound for ΔLPD<1: `{rb.get('1.0')}`; for ΔLPD<0: `{rb.get('0.0')}`\n\n")

        if global_a2_summary.get("enabled"):
            memo.append("## Catalogue/Photo-z Stress (Global Top-N)\n\n")
            memo.append(
                f"- Min ΔLPD on tested photo-z grid: `{float((global_a2_summary.get('photoz_min') or {}).get('delta_lpd_total', float('nan'))):+.3f}` "
                f"(thresholds: {global_a2_summary.get('thresholds')}).\n\n"
            )

        if gr_inj_summary.get("enabled"):
            memo.append("## GR-Truth Calibration\n\n")
            memo.append(
                f"- GR-truth injection null (spectral-only): mean `{float(gr_inj_summary.get('delta_lpd_mean', float('nan'))):+.3f}`, "
                f"max `{float(gr_inj_summary.get('delta_lpd_max', float('nan'))):+.3f}` over `n={int(gr_inj_summary.get('n_rep', 0))}`.\n"
            )
            pz = paths.tables_dir / "calibrated_pz_table.csv"
            if pz.exists():
                memo.append(f"- Calibrated p/Z table: `tables/{pz.name}`.\n")
            memo.append("\n")

        memo.append("## Falsifiers (Operational)\n\n")
        memo.append("- If improved catalogue modelling within realistic external priors drives ΔLPD→0, the MG-propagation interpretation weakens.\n")
        memo.append("- If non-circular splits show sign flips or strong incoherence beyond posterior uncertainty, it argues for misspecification.\n")
        memo.append("- If future larger samples remove the distance-concentration trend, a cumulative propagation effect is disfavoured.\n")

        (paths.out_dir / "executive_memo.md").write_text("".join(memo), encoding="utf-8")
    except Exception:
        pass

    write_markdown_report(paths=paths, markdown="\n".join(md).strip() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
