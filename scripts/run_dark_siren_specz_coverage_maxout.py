#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import healpy as hp  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from entropy_horizon_recon.dark_siren_gap_lpd import BetaPrior, marginalize_f_miss_global  # noqa: E402
from entropy_horizon_recon.dark_sirens import GalaxyIndex, load_gladeplus_index  # noqa: E402
from entropy_horizon_recon.dark_sirens_pe import PePixelDistanceHistogram  # noqa: E402
from entropy_horizon_recon.gw_distance_priors import GWDistancePrior  # noqa: E402
from entropy_horizon_recon.sirens import MuForwardPosterior, load_mu_forward_posterior, predict_dL_em, predict_r_gw_em  # noqa: E402


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _set_thread_env(n: int) -> None:
    n = int(max(1, n))
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[k] = str(n)


def _as_unit_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    cosd = np.cos(dec)
    return np.stack((cosd * np.cos(ra), cosd * np.sin(ra), np.sin(dec)), axis=1).astype(np.float32)


def _chord_from_radius_arcsec(r_arcsec: float) -> float:
    r_rad = float(r_arcsec) * (np.pi / 180.0) / 3600.0
    return float(2.0 * np.sin(0.5 * r_rad))


def _sep_arcsec_from_chord(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    d = np.clip(d, 0.0, 2.0)
    sep_rad = 2.0 * np.arcsin(np.clip(0.5 * d, 0.0, 1.0))
    return sep_rad * (180.0 / np.pi) * 3600.0


def _shift_ra_dec(*, ra_deg: np.ndarray, dec_deg: np.ndarray, dra_deg: float, ddec_deg: float) -> tuple[np.ndarray, np.ndarray]:
    ra = (np.asarray(ra_deg, dtype=float) + float(dra_deg)) % 360.0
    dec = np.asarray(dec_deg, dtype=float) + float(ddec_deg)
    dec = np.clip(dec, -90.0, 90.0)
    return ra, dec


def _random_rotation_control(*, ra_deg: np.ndarray, dec_deg: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A simple extra null: random RA offset + random Dec sign flip (deterministic by seed)."""
    rng = np.random.default_rng(int(seed))
    dra = float(rng.uniform(0.0, 360.0))
    flip = bool(rng.integers(0, 2))
    ra = (np.asarray(ra_deg, dtype=float) + dra) % 360.0
    dec = np.asarray(dec_deg, dtype=float)
    if flip:
        dec = -dec
    dec = np.clip(dec, -90.0, 90.0)
    return ra, dec


@dataclass(frozen=True)
class BaselineCache:
    gap_root: Path
    run_label: str
    events: list[str]
    log_alpha_mu: np.ndarray
    log_alpha_gr: np.ndarray
    prior: BetaPrior
    n_f: int
    eps: float
    draw_idx: list[int]


def _load_baseline_cache(*, gap_root: Path, run_label: str) -> BaselineCache:
    gap_root = gap_root.expanduser().resolve()
    summary = _read_json(gap_root / f"summary_{run_label}.json")
    mix = summary.get("mixture", {})
    mix_meta = mix.get("f_miss_meta", {})
    prior_meta = (mix_meta.get("prior") or {})
    grid_meta = (mix_meta.get("grid") or {})
    prior = BetaPrior(mean=float(prior_meta["mean"]), kappa=float(prior_meta["kappa"]))
    n_f = int(grid_meta.get("n", 401))
    eps = float(grid_meta.get("eps", 1e-6))
    draw_idx = [int(i) for i in summary.get("draw_idx", [])]
    if not draw_idx:
        raise ValueError("baseline summary missing draw_idx")

    with np.load(gap_root / "tables" / f"selection_alpha_{run_label}.npz", allow_pickle=True) as d:
        log_alpha_mu = np.asarray(d["log_alpha_mu"], dtype=float)
        log_alpha_gr = np.asarray(d["log_alpha_gr"], dtype=float)

    events = []
    for p in sorted((gap_root / "cache").glob("event_*.npz")):
        name = p.stem
        if name.startswith("event_"):
            events.append(name[len("event_") :])
    if not events:
        raise FileNotFoundError("No event_*.npz found under gap_root/cache")

    return BaselineCache(
        gap_root=gap_root,
        run_label=str(run_label),
        events=events,
        log_alpha_mu=log_alpha_mu,
        log_alpha_gr=log_alpha_gr,
        prior=prior,
        n_f=n_f,
        eps=eps,
        draw_idx=draw_idx,
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


def _load_event_cache(ev_npz: Path, *, pe_nside: int, p_credible: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PePixelDistanceHistogram]:
    with np.load(ev_npz, allow_pickle=False) as d:
        z = np.asarray(d["z"], dtype=float)
        w = np.asarray(d["w"], dtype=float)
        ipix = np.asarray(d["ipix"], dtype=np.int64)
        hpix_sel = np.asarray(d["hpix_sel"], dtype=np.int64)
        pe = PePixelDistanceHistogram(
            nside=int(pe_nside),
            nest=True,
            p_credible=float(p_credible),
            pix_sel=np.asarray(d["pe_pix_sel"], dtype=np.int64),
            prob_pix=np.asarray(d["pe_prob_pix"], dtype=float),
            dL_edges=np.asarray(d["pe_dL_edges"], dtype=float),
            pdf_bins=np.asarray(d["pe_pdf_bins"], dtype=float),
        )
    return z, w, ipix, hpix_sel, pe


def _gather_galaxies_with_global_idx(
    cat: GalaxyIndex,
    hpix: np.ndarray,
    *,
    z_max: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hpix = np.asarray(hpix, dtype=np.int64)
    hpix = np.unique(hpix)
    if hpix.size == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    offs = cat.hpix_offsets
    n = int(np.sum(offs[hpix + 1] - offs[hpix]))
    ra = np.empty(n, dtype=np.float32)
    dec = np.empty(n, dtype=np.float32)
    z = np.empty(n, dtype=np.float32)
    w = np.empty(n, dtype=np.float32)
    idx = np.empty(n, dtype=np.int64)

    pos = 0
    for p in hpix.tolist():
        a = int(offs[p])
        b = int(offs[p + 1])
        if b <= a:
            continue
        m = slice(pos, pos + (b - a))
        ra[m] = cat.ra_deg[a:b]
        dec[m] = cat.dec_deg[a:b]
        z[m] = cat.z[a:b]
        w[m] = cat.w[a:b]
        idx[m] = np.arange(a, b, dtype=np.int64)
        pos += b - a

    ra = ra[:pos]
    dec = dec[:pos]
    z = z[:pos]
    w = w[:pos]
    idx = idx[:pos]

    if z_max is not None:
        m = np.isfinite(z) & (z > 0.0) & (z <= float(z_max))
        ra, dec, z, w, idx = ra[m], dec[m], z[m], w[m], idx[m]
    return ra, dec, z, w, idx


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


def _build_fiducial_dL_of_z(*, h0: float, omega_m0: float, z_max: float, n: int = 5001) -> tuple[np.ndarray, np.ndarray]:
    c = 299792.458
    z_grid = np.linspace(0.0, float(z_max), int(n))
    Ez = np.sqrt(float(omega_m0) * (1.0 + z_grid) ** 3 + (1.0 - float(omega_m0)))
    invE = 1.0 / np.clip(Ez, 1e-12, np.inf)
    dz = np.diff(z_grid)
    dc = np.empty_like(z_grid)
    dc[0] = 0.0
    dc[1:] = (c / float(h0)) * np.cumsum(0.5 * dz * (invE[:-1] + invE[1:]))
    dL = (1.0 + z_grid) * dc
    return z_grid, dL


@dataclass
class SpecZCatalog:
    name: str
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    z: np.ndarray
    quality: np.ndarray | None
    source_meta: dict[str, Any]


def _load_specz_npz(path: Path, *, name: str) -> SpecZCatalog:
    with np.load(path, allow_pickle=True) as d:
        ra = np.asarray(d["ra_deg"], dtype=float)
        dec = np.asarray(d["dec_deg"], dtype=float)
        z = np.asarray(d["z"], dtype=float)
        q = np.asarray(d["quality"], dtype=float) if "quality" in d.files else None
        meta = json.loads(str(d["source_meta"].tolist())) if "source_meta" in d.files else {}
    m = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z)
    ra, dec, z = ra[m], dec[m], z[m]
    if q is not None:
        q = np.asarray(q, dtype=float)[m]
    return SpecZCatalog(name=str(name), ra_deg=ra, dec_deg=dec, z=z, quality=q, source_meta=meta)


def _filter_specz_catalog(cat: SpecZCatalog, *, min_quality: float | None) -> SpecZCatalog:
    if min_quality is None or cat.quality is None:
        return cat
    q = np.asarray(cat.quality, dtype=float)
    m = np.isfinite(q) & (q >= float(min_quality))
    if not np.any(m):
        return SpecZCatalog(name=cat.name, ra_deg=np.zeros((0,), dtype=float), dec_deg=np.zeros((0,), dtype=float), z=np.zeros((0,), dtype=float), quality=np.zeros((0,), dtype=float), source_meta=cat.source_meta)
    return SpecZCatalog(name=cat.name, ra_deg=cat.ra_deg[m], dec_deg=cat.dec_deg[m], z=cat.z[m], quality=q[m], source_meta=cat.source_meta)


def _combine_catalogs(cats: list[SpecZCatalog], *, z_max: float) -> dict[str, Any]:
    ra_all: list[np.ndarray] = []
    dec_all: list[np.ndarray] = []
    z_all: list[np.ndarray] = []
    src_all: list[np.ndarray] = []
    for cat in cats:
        if int(cat.ra_deg.size) == 0:
            continue
        m = np.isfinite(cat.z) & (cat.z > 0.0) & (cat.z <= float(z_max))
        if not np.any(m):
            continue
        ra_all.append(np.asarray(cat.ra_deg, dtype=float)[m])
        dec_all.append(np.asarray(cat.dec_deg, dtype=float)[m])
        z_all.append(np.asarray(cat.z, dtype=float)[m])
        src_all.append(np.full((int(np.count_nonzero(m)),), str(cat.name), dtype=object))
    if not ra_all:
        return {"ra_deg": np.zeros((0,)), "dec_deg": np.zeros((0,)), "z": np.zeros((0,)), "source": np.zeros((0,), dtype=object)}
    ra = np.concatenate(ra_all)
    dec = np.concatenate(dec_all)
    z = np.concatenate(z_all)
    src = np.concatenate(src_all)
    return {"ra_deg": ra, "dec_deg": dec, "z": z, "source": src}


def _build_tree(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple[cKDTree, np.ndarray]:
    xyz = _as_unit_xyz(ra_deg, dec_deg)
    # cKDTree stores a copy; keep xyz for chord->angle conversion if needed.
    tree = cKDTree(xyz)
    return tree, xyz


def _score_with_cat_replacements(
    *,
    base_cat_mu: list[np.ndarray],
    base_cat_gr: list[np.ndarray],
    base_miss_mu: list[np.ndarray],
    base_miss_gr: list[np.ndarray],
    baseline: BaselineCache,
    repl: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    cat_mu = list(base_cat_mu)
    cat_gr = list(base_cat_gr)
    for j, (mu, gr) in repl.items():
        cat_mu[int(j)] = np.asarray(mu, dtype=float)
        cat_gr[int(j)] = np.asarray(gr, dtype=float)
    res = marginalize_f_miss_global(
        logL_cat_mu_by_event=cat_mu,
        logL_cat_gr_by_event=cat_gr,
        logL_missing_mu_by_event=base_miss_mu,
        logL_missing_gr_by_event=base_miss_gr,
        log_alpha_mu=baseline.log_alpha_mu,
        log_alpha_gr=baseline.log_alpha_gr,
        prior=baseline.prior,
        n_f=baseline.n_f,
        eps=baseline.eps,
    )
    dlp_total = float(res.lpd_mu_total - res.lpd_gr_total)
    dlp_data = float(res.lpd_mu_total_data - res.lpd_gr_total_data)
    return dlp_total, dlp_data


def _spectral_only_logL_from_weight_hist(
    *,
    pe: PePixelDistanceHistogram,
    z_cent: np.ndarray,
    weight_hist: np.ndarray,
    z_grid_post: np.ndarray,
    dL_em_grid: np.ndarray,
    R_grid: np.ndarray,
    gw_prior: GWDistancePrior,
) -> tuple[np.ndarray, np.ndarray]:
    # Copy of the helper used in the existing sprint script, to keep scoring identical.
    z_cent = np.asarray(z_cent, dtype=float)
    weight_hist = np.asarray(weight_hist, dtype=float)
    if z_cent.ndim != 1 or weight_hist.ndim != 1 or z_cent.size != weight_hist.size:
        raise ValueError("z_cent and weight_hist must be 1D and same length.")
    m = np.isfinite(z_cent) & (z_cent > 0.0) & np.isfinite(weight_hist) & (weight_hist > 0.0)
    if not np.any(m):
        n_draws = int(dL_em_grid.shape[0])
        return np.full((n_draws,), -np.inf, dtype=float), np.full((n_draws,), -np.inf, dtype=float)
    z_cent = z_cent[m]
    w = weight_hist[m]

    edges, pdf_1d = _pe_sky_marginal_pdf_1d(pe)
    nb = int(edges.size - 1)

    n_draws = int(dL_em_grid.shape[0])
    out_mu = np.full((n_draws,), -np.inf, dtype=float)
    out_gr = np.full((n_draws,), -np.inf, dtype=float)
    for j in range(n_draws):
        dL_em = np.interp(z_cent, z_grid_post, np.asarray(dL_em_grid[j], dtype=float))
        R = np.interp(z_cent, z_grid_post, np.asarray(R_grid[j], dtype=float))
        dL_gw = dL_em * R

        def _logL(dL: np.ndarray) -> float:
            bin_idx = np.searchsorted(edges, dL, side="right") - 1
            valid = (bin_idx >= 0) & (bin_idx < nb) & np.isfinite(dL) & (dL > 0.0)
            if not np.any(valid):
                return float("-inf")
            pdf = pdf_1d[np.clip(bin_idx, 0, nb - 1)]
            pdf = np.where(valid, pdf, 0.0)
            inv_pi = np.exp(-gw_prior.log_pi_dL(np.clip(dL, 1e-6, np.inf)))
            t = w * pdf * inv_pi
            s = float(np.sum(t))
            return float(np.log(max(s, 1e-300)))

        out_gr[j] = _logL(dL_em)
        out_mu[j] = _logL(dL_gw)
    return out_mu, out_gr


def main() -> int:
    ap = argparse.ArgumentParser(description="Spec-z host-weight coverage maxout under strict false-match legitimacy constraints.")
    ap.add_argument("--config", required=True, help="Path to JSON config.")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--outdir", default=None, help="Optional output directory under outputs/.")
    args = ap.parse_args()

    _set_thread_env(int(args.threads))

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _read_json(cfg_path)

    out_root = Path(args.outdir).expanduser().resolve() if args.outdir else (REPO_ROOT / "outputs" / f"dark_siren_specz_coverage_maxout_{_utc_now_compact()}")
    fig_dir = out_root / "figures"
    tab_dir = out_root / "tables"
    raw_dir = out_root / "raw"
    for d in (fig_dir, tab_dir, raw_dir, out_root / "configs"):
        d.mkdir(parents=True, exist_ok=True)
    (out_root / "configs" / cfg_path.name).write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

    base_cfg = cfg["baseline"]
    gap_root = Path(base_cfg["gap_run_root"]).expanduser().resolve()
    run_label = str(base_cfg["run_label"])
    recon_run_dir = Path(base_cfg["recon_run_dir"]).expanduser().resolve()
    spectral_only_terms_npz = Path(base_cfg["spectral_only_terms_npz"]).expanduser().resolve()

    maxout_cfg = cfg["maxout"]
    z_max = float(maxout_cfg.get("z_max", 0.3))
    radii = [float(r) for r in maxout_cfg.get("radii_arcsec", [1, 2, 3, 5, 10, 15, 20, 30, 45, 60])]
    radii = sorted({float(r) for r in radii})
    k_grid = [int(k) for k in maxout_cfg.get("k_grid", [200, 1000, 5000, 20000, 50000, 100000, 200000])]
    k_grid = sorted({int(max(1, k)) for k in k_grid})
    k_max = int(max(k_grid))
    k_gain_min = float(maxout_cfg.get("k_gain_min_frac", 0.05))
    max_k_dedup = int(maxout_cfg.get("max_k_for_outputs", k_max))

    false_cfg = cfg.get("false_match", {}) or {}
    shift_ra_deg = float(false_cfg.get("shift_ra_deg", 0.5))
    shift_dec_deg = float(false_cfg.get("shift_dec_deg", 0.5))
    extra_random_control = bool(false_cfg.get("random_rotation_enabled", True))
    random_seed = int(false_cfg.get("random_seed", 12345))

    gate_cfg = cfg.get("legitimacy_gate", {}) or {}
    gate_median_max = float(gate_cfg.get("median_shift_over_true_max", 0.10))
    gate_event_max = float(gate_cfg.get("max_event_shift_over_true_max", 0.30))
    gate_events = [str(e) for e in gate_cfg.get("gate_events", [])]

    top_events = [str(e) for e in cfg.get("top_events", [])]
    if not top_events:
        raise SystemExit("Config missing top_events.")

    # Baseline cache for scoring (Phase 5).
    baseline = _load_baseline_cache(gap_root=gap_root, run_label=run_label)
    manifest = _read_json(gap_root / "manifest.json")
    pe_nside = int(manifest.get("pe_nside", 64))
    p_credible = float(manifest.get("p_credible", 0.9))

    # Candidate ranking uses a fiducial LCDM dL(z), same as sprint.
    ppc_cfg = cfg.get("ppc_residuals", {}) or {}
    z_grid_fid, dL_grid_fid = _build_fiducial_dL_of_z(
        h0=float(ppc_cfg.get("h0_ref", float(manifest.get("gw_distance_prior_h0_ref", 67.7)))),
        omega_m0=float(ppc_cfg.get("omega_m0", float(manifest.get("gw_distance_prior_omega_m0", 0.31)))),
        z_max=float(z_max),
        n=5001,
    )

    gw_prior = GWDistancePrior(
        mode="dL_powerlaw",
        powerlaw_k=float(manifest.get("gw_distance_prior_power", 2.0)),
        h0_ref=float(manifest.get("gw_distance_prior_h0_ref", 67.7)),
        omega_m0=float(manifest.get("gw_distance_prior_omega_m0", 0.31)),
        omega_k0=float(manifest.get("gw_distance_prior_omega_k0", 0.0)),
        z_max=float(manifest.get("gw_distance_prior_zmax", 10.0)),
        n_grid=50_000,
    )

    # Load GLADE+ luminosity-weighted index (used for candidate coordinates/weights).
    gl_cfg = cfg["glade"]
    cat_lumB = load_gladeplus_index(gl_cfg["index_lumB"])

    # --------------------------------------------
    # PHASE 3: build per-event host-weight candidates (top-Kmax)
    # --------------------------------------------
    cand_by_event: dict[str, dict[str, Any]] = {}
    capture_rows: list[dict[str, Any]] = []

    for ev in top_events:
        ev_npz = gap_root / "cache" / f"event_{ev}.npz"
        z_cat, w_cat, ipix_cat, hpix_sel, pe = _load_event_cache(ev_npz, pe_nside=pe_nside, p_credible=p_credible)
        ra, dec, z2, w2, idx_global = _gather_galaxies_with_global_idx(cat_lumB, hpix_sel, z_max=z_max)
        if z2.size != z_cat.size or w2.size != w_cat.size:
            raise RuntimeError(f"{ev}: mismatch between cache galaxy arrays and gathered lumB index arrays.")
        # Sky probability per galaxy.
        npix = int(hp.nside2npix(int(pe.nside)))
        pix_to_row = np.full((npix,), -1, dtype=np.int32)
        pix_to_row[np.asarray(pe.pix_sel, dtype=np.int64)] = np.arange(int(pe.pix_sel.size), dtype=np.int32)
        row = pix_to_row[ipix_cat]
        good = (row >= 0) & np.isfinite(z_cat) & (z_cat > 0.0) & np.isfinite(w_cat) & (w_cat > 0.0)
        prob = np.zeros_like(z_cat, dtype=float)
        prob[good] = np.asarray(pe.prob_pix, dtype=float)[row[good]]

        edges, pdf_1d = _pe_sky_marginal_pdf_1d(pe)
        widths = np.diff(edges)

        # Host-weight proxy for ranking: w * prob * pdf_1d(dL(z)) / pi(dL(z)).
        dL_em = np.interp(np.clip(z_cat, 0.0, z_grid_fid[-1]), z_grid_fid, dL_grid_fid)
        bin_idx = np.searchsorted(edges, dL_em, side="right") - 1
        valid = good & (bin_idx >= 0) & (bin_idx < widths.size) & np.isfinite(dL_em) & (dL_em > 0.0)
        pdf = np.zeros_like(dL_em, dtype=float)
        pdf[valid] = pdf_1d[bin_idx[valid]]
        inv_pi = np.exp(-gw_prior.log_pi_dL(np.clip(dL_em, 1e-6, np.inf)))
        weight = np.zeros_like(z_cat, dtype=float)
        weight[valid] = w_cat[valid] * prob[valid] * pdf[valid] * inv_pi[valid]

        tot_w = float(np.sum(weight))
        if not (np.isfinite(tot_w) and tot_w > 0.0):
            raise RuntimeError(f"{ev}: invalid host-weight proxy normalization.")

        k_use = min(int(k_max), int(weight.size))
        if k_use <= 0:
            raise RuntimeError(f"{ev}: no galaxies available for ranking.")
        if k_use == int(weight.size):
            idx_top = np.argsort(weight)[::-1].astype(np.int64, copy=False)
        else:
            idx_top = np.argpartition(weight, -k_use)[-k_use:]
            idx_top = idx_top[np.argsort(weight[idx_top])[::-1]].astype(np.int64, copy=False)

        w_sorted = weight[idx_top]
        cw = np.cumsum(w_sorted)
        cw_frac = cw / tot_w

        # Apply the K expansion rule: stop increasing when incremental gain < k_gain_min.
        k_allowed: list[int] = []
        prev = 0.0
        for k in k_grid:
            kk = min(int(k), int(idx_top.size))
            gain = float(cw_frac[kk - 1] - prev) if kk > 0 else 0.0
            if (not k_allowed) or (gain >= float(k_gain_min)):
                k_allowed.append(int(k))
                prev = float(cw_frac[kk - 1]) if kk > 0 else prev
            else:
                # A maxout scan should not assume monotone incremental gains at every K step.
                # If a particular K jump fails the gain threshold, skip it and keep trying larger K.
                continue

        capture_rows.append(
            {
                "event": ev,
                "n_gal_total": int(weight.size),
                "k_max_available": int(idx_top.size),
                "k_allowed": json.dumps(k_allowed),
                "cum_weight_frac_at_kmax": float(cw_frac[int(idx_top.size - 1)]) if idx_top.size else float("nan"),
            }
        )

        cand_by_event[ev] = {
            "pe": pe,
            "idx_top": idx_top,
            "ra_top": np.asarray(ra, dtype=float)[idx_top],
            "dec_top": np.asarray(dec, dtype=float)[idx_top],
            "weight_top": w_sorted,
            "cw_top": cw,
            "cw_frac_top": cw_frac,
            "weight_total": float(tot_w),
            "z_cat": np.asarray(z_cat, dtype=float),
            "w_cat": np.asarray(w_cat, dtype=float),
            "prob": np.asarray(prob, dtype=float),
            "k_allowed": k_allowed,
        }

        # Export a compact candidate table for audit (top max_k_dedup).
        k_out = min(int(max_k_dedup), int(idx_top.size))
        cand_rows = []
        for rank, gi in enumerate(idx_top[:k_out].tolist(), start=1):
            cand_rows.append(
                {
                    "event": ev,
                    "rank": int(rank),
                    "gal_index_in_event": int(gi),
                    "catalog_id": int(idx_global[gi]),
                    "ra_deg": float(np.asarray(ra, dtype=float)[gi]),
                    "dec_deg": float(np.asarray(dec, dtype=float)[gi]),
                    "catalog_z": float(z_cat[gi]),
                    "weight_proxy": float(weight[gi]),
                    "cumulative_weight_frac": float(cw_frac[rank - 1]),
                }
            )
        _write_csv(
            tab_dir / f"host_candidates_{ev}.csv",
            cand_rows,
            fieldnames=["event", "rank", "gal_index_in_event", "catalog_id", "ra_deg", "dec_deg", "catalog_z", "weight_proxy", "cumulative_weight_frac"],
        )

    _write_csv(tab_dir / "host_candidates_capture_summary.csv", capture_rows, fieldnames=list(capture_rows[0].keys()) if capture_rows else [])

    # --------------------------------------------
    # PHASE 1/2: ingest spec-z catalogs (local caches only)
    # --------------------------------------------
    cache_dir = (REPO_ROOT / str(cfg.get("specz_cache_dir", "data/cache/specz_catalogs"))).resolve()
    tiers_cfg = cfg["tiers"]
    tier_names = [str(t) for t in tiers_cfg.keys()]

    # Build per-tier combined KD-trees.
    tier_trees: dict[str, dict[str, Any]] = {}
    specz_manifest: dict[str, Any] = {"tiers": {}}
    for tier in tier_names:
        tcfg = tiers_cfg[tier] or {}
        sources = list(tcfg.get("sources", []))
        cats: list[SpecZCatalog] = []
        src_rows: list[dict[str, Any]] = []
        for s in sources:
            nm = str(s["name"])
            npz = cache_dir / str(s["npz"])
            if not npz.exists():
                src_rows.append({"name": nm, "npz": str(npz), "status": "missing"})
                continue
            cat = _load_specz_npz(npz, name=nm)
            cat = _filter_specz_catalog(cat, min_quality=s.get("min_quality"))
            src_rows.append(
                {
                    "name": nm,
                    "npz": str(npz),
                    "status": "loaded",
                    "n_rows": int(cat.ra_deg.size),
                    "min_quality": s.get("min_quality", None),
                }
            )
            cats.append(cat)

        comb = _combine_catalogs(cats, z_max=z_max)
        tree, xyz = _build_tree(np.asarray(comb["ra_deg"], dtype=float), np.asarray(comb["dec_deg"], dtype=float))
        tier_trees[tier] = {"tree": tree, "xyz": xyz, "z": np.asarray(comb["z"], dtype=float), "source": np.asarray(comb["source"], dtype=object)}
        specz_manifest["tiers"][tier] = {"sources": src_rows, "n_combined": int(np.asarray(comb["ra_deg"]).size)}

    _write_json(raw_dir / "specz_catalog_manifest.json", specz_manifest)

    # --------------------------------------------
    # PHASE 3/4: coverage scan + false-match controls
    # --------------------------------------------
    # coverage_grid_full rows: one per (event, tier, radius, k) with true/shifted values
    coverage_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []

    for tier in tier_names:
        tree = tier_trees[tier]["tree"]
        z_spec = tier_trees[tier]["z"]
        src_spec = tier_trees[tier]["source"]
        n_spec = int(z_spec.size)
        if n_spec == 0:
            continue

        for ev in top_events:
            data = cand_by_event[ev]
            ra0 = np.asarray(data["ra_top"], dtype=float)
            dec0 = np.asarray(data["dec_top"], dtype=float)
            w_sorted = np.asarray(data["weight_top"], dtype=float)
            cw = np.asarray(data["cw_top"], dtype=float)
            tot_w = float(data["weight_total"])
            k_allowed = list(data["k_allowed"])

            # Only consider k values that are allowed (rule) and <= available candidates.
            k_vals = [int(k) for k in k_allowed if int(k) <= int(ra0.size)]
            if not k_vals:
                continue

            def _query_control(ra_deg: np.ndarray, dec_deg: np.ndarray) -> dict[str, Any]:
                xyz = _as_unit_xyz(ra_deg, dec_deg)
                # k=2 to estimate conflict rate (2nd neighbour within radius).
                d, idx = tree.query(xyz, k=2, workers=int(args.threads))
                d = np.asarray(d, dtype=float)
                idx = np.asarray(idx, dtype=np.int64)
                # idx can be n_spec for missing when using distance_upper_bound (we don't). Here always finite.
                s1 = _sep_arcsec_from_chord(d[:, 0])
                s2 = _sep_arcsec_from_chord(d[:, 1])
                i1 = idx[:, 0]
                return {"sep1_arcsec": s1, "sep2_arcsec": s2, "z1": z_spec[i1], "src1": src_spec[i1]}

            controls: dict[str, dict[str, Any]] = {}
            controls["true"] = _query_control(ra0, dec0)
            ra_sr, dec_sr = _shift_ra_dec(ra_deg=ra0, dec_deg=dec0, dra_deg=shift_ra_deg, ddec_deg=0.0)
            ra_sd, dec_sd = _shift_ra_dec(ra_deg=ra0, dec_deg=dec0, dra_deg=0.0, ddec_deg=shift_dec_deg)
            controls["shift_ra"] = _query_control(ra_sr, dec_sr)
            controls["shift_dec"] = _query_control(ra_sd, dec_sd)
            if extra_random_control:
                ra_rr, dec_rr = _random_rotation_control(ra_deg=ra0, dec_deg=dec0, seed=(random_seed + hash((tier, ev)) % 1_000_000))
                controls["rand_rot"] = _query_control(ra_rr, dec_rr)

            # Precompute per-radius masks and cumulative matched weights for each control.
            # For each control, produce dict radius->matched_cumsum and conflict_cumsum.
            ctrl_match: dict[str, dict[float, np.ndarray]] = {}
            ctrl_conflict: dict[str, dict[float, np.ndarray]] = {}
            for ck, q in controls.items():
                sep1 = np.asarray(q["sep1_arcsec"], dtype=float)
                sep2 = np.asarray(q["sep2_arcsec"], dtype=float)
                ctrl_match[ck] = {}
                ctrl_conflict[ck] = {}
                for r in radii:
                    m = np.isfinite(sep1) & (sep1 <= float(r))
                    c = np.isfinite(sep2) & (sep2 <= float(r))
                    ctrl_match[ck][float(r)] = np.cumsum(w_sorted * m)
                    ctrl_conflict[ck][float(r)] = np.cumsum(w_sorted * c)

            # Emit coverage rows for each (radius, k) requested, at this tier.
            for r in radii:
                for k in k_vals:
                    kk = min(int(k), int(w_sorted.size))
                    w_topk = float(cw[kk - 1]) if kk > 0 else 0.0
                    row: dict[str, Any] = {
                        "event": ev,
                        "tier": tier,
                        "radius_arcsec": float(r),
                        "k": int(k),
                        "k_used": int(kk),
                        "weight_total": float(tot_w),
                        "weight_topk": float(w_topk),
                        "cum_weight_frac_at_k": float(data["cw_frac_top"][kk - 1]) if kk > 0 else float("nan"),
                    }
                    for ck in ("true", "shift_ra", "shift_dec"):
                        w_m = float(ctrl_match[ck][float(r)][kk - 1]) if kk > 0 else 0.0
                        row[f"frac_weight_{ck}"] = float(w_m / tot_w) if tot_w > 0 else float("nan")
                    # Optional extra control.
                    if "rand_rot" in controls:
                        w_m = float(ctrl_match["rand_rot"][float(r)][kk - 1]) if kk > 0 else 0.0
                        row["frac_weight_rand_rot"] = float(w_m / tot_w) if tot_w > 0 else float("nan")
                    coverage_rows.append(row)

            # Conflict-rate diagnostics at the max K for this event.
            k_gate = int(max(k_vals))
            kk = min(int(k_gate), int(w_sorted.size))
            for r in radii:
                w_m = float(ctrl_conflict["true"][float(r)][kk - 1]) if kk > 0 else 0.0
                conflict_rows.append(
                    {
                        "event": ev,
                        "tier": tier,
                        "radius_arcsec": float(r),
                        "k": int(k_gate),
                        "frac_weight_conflict_true": float(w_m / tot_w) if tot_w > 0 else float("nan"),
                    }
                )

    if not coverage_rows:
        raise SystemExit("No coverage rows produced; check spec-z catalogs and config.")

    _write_csv(
        tab_dir / "coverage_grid_full.csv",
        coverage_rows,
        fieldnames=list(coverage_rows[0].keys()),
    )
    if conflict_rows:
        _write_csv(tab_dir / "match_conflict_rates.csv", conflict_rows, fieldnames=list(conflict_rows[0].keys()))

    # --------------------------------------------
    # PHASE 0/4: legitimacy gate evaluation per tier+radius (top gate_events)
    # --------------------------------------------
    if not gate_events:
        gate_events = list(top_events[:3])

    gate_rows: list[dict[str, Any]] = []
    allowed_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    # Gate computed at the maximum common K across gate_events (within each tier).
    for tier in tier_names:
        for r in radii:
            # Find max common K among gate events.
            common_ks = None
            for ev in gate_events:
                ev_ks = set(int(k) for k in cand_by_event[ev]["k_allowed"])
                common_ks = ev_ks if common_ks is None else (common_ks & ev_ks)
            if not common_ks:
                continue
            k_gate = int(max(common_ks))
            ratios = []
            ratios_by_event: dict[str, float] = {}
            for ev in gate_events:
                rr = [x for x in coverage_rows if x["tier"] == tier and x["event"] == ev and float(x["radius_arcsec"]) == float(r) and int(x["k"]) == int(k_gate)]
                if not rr:
                    ratios_by_event[ev] = float("nan")
                    continue
                row = rr[0]
                ft = float(row.get("frac_weight_true", float("nan")))
                fsr = float(row.get("frac_weight_shift_ra", float("nan")))
                fsd = float(row.get("frac_weight_shift_dec", float("nan")))
                fs = 0.5 * (fsr + fsd)
                rat = float(fs / ft) if (np.isfinite(ft) and ft > 0.0 and np.isfinite(fs)) else float("inf")
                ratios.append(rat)
                ratios_by_event[ev] = rat
            ratios_arr = np.asarray(ratios, dtype=float)
            med = float(np.nanmedian(ratios_arr)) if ratios_arr.size else float("nan")
            mx = float(np.nanmax(ratios_arr)) if ratios_arr.size else float("nan")
            ok = bool(np.isfinite(med) and np.isfinite(mx) and med < gate_median_max and mx < gate_event_max)
            gate_rows.append({"tier": tier, "radius_arcsec": float(r), "k_gate": int(k_gate), "median_shift_over_true": med, "max_event_shift_over_true": mx, "ok": ok, "ratios_by_event": json.dumps(ratios_by_event, sort_keys=True)})
            if ok:
                allowed_rows.append({"tier": tier, "radius_arcsec": float(r), "k_gate": int(k_gate), "median_shift_over_true": med, "max_event_shift_over_true": mx})
            else:
                failures.append({"tier": tier, "radius_arcsec": float(r), "k_gate": int(k_gate), "median_shift_over_true": med, "max_event_shift_over_true": mx, "ratios_by_event": json.dumps(ratios_by_event, sort_keys=True)})

    _write_csv(tab_dir / "false_match_gate_by_radius.csv", gate_rows, fieldnames=list(gate_rows[0].keys()) if gate_rows else ["tier"])
    if failures:
        _write_csv(tab_dir / "false_match_failures.csv", failures, fieldnames=list(failures[0].keys()))

    # --------------------------------------------
    # PHASE 3: find max legitimate operating points per tier
    # --------------------------------------------
    op_rows: list[dict[str, Any]] = []
    best_points: dict[str, dict[str, Any]] = {}
    for tier in tier_names:
        # Allowed radii under gate.
        allowed_r = sorted({float(r["radius_arcsec"]) for r in allowed_rows if r["tier"] == tier})
        if not allowed_r:
            continue

        # Common K across gate_events (respect per-event k_allowed).
        common_ks = None
        for ev in gate_events:
            ev_ks = set(int(k) for k in cand_by_event[ev]["k_allowed"])
            common_ks = ev_ks if common_ks is None else (common_ks & ev_ks)
        if not common_ks:
            continue
        common_ks = sorted(common_ks)

        best = None
        for r in allowed_r:
            for k in common_ks:
                vals = []
                for ev in gate_events:
                    rr = [x for x in coverage_rows if x["tier"] == tier and x["event"] == ev and float(x["radius_arcsec"]) == float(r) and int(x["k"]) == int(k)]
                    if not rr:
                        continue
                    vals.append(float(rr[0].get("frac_weight_true", float("nan"))))
                if not vals:
                    continue
                med = float(np.nanmedian(np.asarray(vals, dtype=float)))
                if best is None or med > best[0]:
                    best = (med, float(r), int(k))
        if best is None:
            continue
        best_med, best_r, best_k = best
        best_points[tier] = {"radius_arcsec": float(best_r), "k": int(best_k), "median_frac_weight_true_gate_events": float(best_med), "gate_events": list(gate_events)}
        op_rows.append({"tier": tier, "best_radius_arcsec": float(best_r), "best_k": int(best_k), "median_frac_weight_true_gate_events": float(best_med), "gate_events": json.dumps(gate_events)})

        # Per-event max coverage (within allowed radii, at its max allowed K).
        for ev in gate_events:
            ev_ks = sorted(int(k) for k in cand_by_event[ev]["k_allowed"])
            if not ev_ks:
                continue
            k_ev = int(max(ev_ks))
            best_ev = None
            for r in allowed_r:
                rr = [x for x in coverage_rows if x["tier"] == tier and x["event"] == ev and float(x["radius_arcsec"]) == float(r) and int(x["k"]) == int(k_ev)]
                if not rr:
                    continue
                v = float(rr[0].get("frac_weight_true", float("nan")))
                if best_ev is None or v > best_ev[0]:
                    best_ev = (v, float(r))
            if best_ev is None:
                continue
            op_rows.append({"tier": tier, "event": ev, "best_radius_arcsec": float(best_ev[1]), "best_k": int(k_ev), "frac_weight_true": float(best_ev[0])})

    if op_rows:
        _write_csv(tab_dir / "max_coverage_operating_points.csv", op_rows, fieldnames=list(op_rows[0].keys()))
    _write_json(raw_dir / "best_points.json", best_points)

    # --------------------------------------------
    # PHASE 5: override scoring at best points (Tier A and Tier B only, if present)
    # --------------------------------------------
    scoring_cfg = cfg.get("scoring", {}) or {}
    do_scoring = bool(scoring_cfg.get("enabled", True))
    score_tiers = [t for t in ("A", "B") if t in best_points]

    score_rows: list[dict[str, Any]] = []
    if do_scoring and score_tiers:
        # Load baseline spectral-only cached terms.
        with np.load(spectral_only_terms_npz, allow_pickle=False) as d:
            events_all = [str(x) for x in d["events"].tolist()]
            base_cat_mu = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_cat_mu"], dtype=float)]
            base_cat_gr = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_cat_gr"], dtype=float)]
            base_miss_mu = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_missing_mu"], dtype=float)]
            base_miss_gr = [np.asarray(v, dtype=float) for v in np.asarray(d["logL_missing_gr"], dtype=float)]
        ev_to_idx = {e: i for i, e in enumerate(events_all)}

        # Load mu forward posterior and match draw set.
        post_full = load_mu_forward_posterior(str(recon_run_dir))
        post = _subset_mu_posterior(post_full, baseline.draw_idx)
        z_grid_post = np.asarray(post.z_grid, dtype=float)
        dL_em_grid = np.asarray(predict_dL_em(post, z_eval=z_grid_post), dtype=float)
        _, R_grid = predict_r_gw_em(post, z_eval=z_grid_post, convention=str(manifest.get("convention", "A")))
        R_grid = np.asarray(R_grid, dtype=float)

        # Baseline score for reference.
        base_res = marginalize_f_miss_global(
            logL_cat_mu_by_event=base_cat_mu,
            logL_cat_gr_by_event=base_cat_gr,
            logL_missing_mu_by_event=base_miss_mu,
            logL_missing_gr_by_event=base_miss_gr,
            log_alpha_mu=baseline.log_alpha_mu,
            log_alpha_gr=baseline.log_alpha_gr,
            prior=baseline.prior,
            n_f=baseline.n_f,
            eps=baseline.eps,
        )
        base_delta = float(base_res.lpd_mu_total - base_res.lpd_gr_total)
        base_delta_data = float(base_res.lpd_mu_total_data - base_res.lpd_gr_total_data)

        z_hist_nbins = int(scoring_cfg.get("z_hist_nbins", 400))
        z_edges = np.linspace(0.0, z_max, z_hist_nbins + 1)
        z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])

        for tier in score_tiers:
            best_r = float(best_points[tier]["radius_arcsec"])
            best_k = int(best_points[tier]["k"])
            # Build a combined KD-tree for this tier (already built).
            tree = tier_trees[tier]["tree"]
            z_spec = tier_trees[tier]["z"]
            src_spec = tier_trees[tier]["source"]

            repl: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            override_maps: dict[str, list[dict[str, Any]]] = {}

            for ev in top_events:
                if ev not in ev_to_idx:
                    continue
                data = cand_by_event[ev]
                idx_top = np.asarray(data["idx_top"], dtype=np.int64)
                pe = data["pe"]
                kk = min(int(best_k), int(idx_top.size))
                use = idx_top[:kk]
                ra = np.asarray(data["ra_top"], dtype=float)[:kk]
                dec = np.asarray(data["dec_top"], dtype=float)[:kk]

                # Query nearest spec-z for the top-K candidates.
                xyz = _as_unit_xyz(ra, dec)
                d, idx = tree.query(xyz, k=1, workers=int(args.threads))
                sep = _sep_arcsec_from_chord(np.asarray(d, dtype=float))
                idx = np.asarray(idx, dtype=np.int64)

                z_cat = np.asarray(data["z_cat"], dtype=float)
                wprob = np.asarray(data["w_cat"], dtype=float) * np.asarray(data["prob"], dtype=float)
                good_wprob = np.isfinite(z_cat) & (z_cat > 0.0) & (z_cat <= z_max) & np.isfinite(wprob) & (wprob > 0.0)
                hist, _ = np.histogram(np.clip(z_cat[good_wprob], 0.0, z_max), bins=z_edges, weights=wprob[good_wprob])
                hist = np.asarray(hist, dtype=float)

                applied: list[dict[str, Any]] = []
                for j, gi in enumerate(use.tolist()):
                    if not (np.isfinite(sep[j]) and float(sep[j]) <= float(best_r)):
                        continue
                    z_s = float(z_spec[int(idx[j])])
                    if not np.isfinite(z_s):
                        continue
                    wprob_gi = float(wprob[int(gi)])
                    if not (np.isfinite(wprob_gi) and wprob_gi > 0.0):
                        continue
                    # Remove from old bin.
                    b_old = int(np.searchsorted(z_edges, float(z_cat[int(gi)]), side="right") - 1)
                    if 0 <= b_old < hist.size:
                        hist[b_old] = max(0.0, float(hist[b_old] - wprob_gi))
                    # Add to new bin (or drop weight if out-of-support).
                    if 0.0 < z_s <= z_max:
                        b_new = int(np.searchsorted(z_edges, z_s, side="right") - 1)
                        if 0 <= b_new < hist.size:
                            hist[b_new] = float(hist[b_new] + wprob_gi)
                    applied.append({"gal_index_in_event": int(gi), "z_spec": float(z_s), "source": str(src_spec[int(idx[j])]), "sep_arcsec": float(sep[j])})

                logL_mu, logL_gr = _spectral_only_logL_from_weight_hist(
                    pe=pe,
                    z_cent=z_cent,
                    weight_hist=hist,
                    z_grid_post=z_grid_post,
                    dL_em_grid=dL_em_grid,
                    R_grid=R_grid,
                    gw_prior=gw_prior,
                )
                repl[int(ev_to_idx[ev])] = (logL_mu, logL_gr)
                override_maps[ev] = applied

            dlp_total, dlp_data = _score_with_cat_replacements(
                base_cat_mu=base_cat_mu,
                base_cat_gr=base_cat_gr,
                base_miss_mu=base_miss_mu,
                base_miss_gr=base_miss_gr,
                baseline=baseline,
                repl=repl,
            )
            score_rows.append(
                {
                    "tier": tier,
                    "radius_arcsec": float(best_r),
                    "k": int(best_k),
                    "delta_lpd_total": float(dlp_total),
                    "delta_lpd_data": float(dlp_data),
                    "baseline_delta_lpd_total": float(base_delta),
                    "baseline_delta_lpd_data": float(base_delta_data),
                }
            )
            # Save override maps for audit.
            p = raw_dir / "specz_override_maps_best_points" / f"specz_overrides_tier{tier}_r{int(round(best_r))}_k{int(best_k)}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"tier": tier, "radius_arcsec": float(best_r), "k": int(best_k), "overrides_by_event": override_maps}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if score_rows:
        _write_csv(tab_dir / "override_score_best_points.csv", score_rows, fieldnames=list(score_rows[0].keys()))

    # --------------------------------------------
    # Figures
    # --------------------------------------------
    try:
        # Heatmap: median frac_weight_true across gate_events as function of (radius, k) for each tier.
        for tier in tier_names:
            rs = sorted({float(x["radius_arcsec"]) for x in coverage_rows if x["tier"] == tier})
            ks = sorted({int(x["k"]) for x in coverage_rows if x["tier"] == tier})
            if not rs or not ks:
                continue
            mat = np.full((len(rs), len(ks)), np.nan)
            for i, r in enumerate(rs):
                for j, k in enumerate(ks):
                    vals = []
                    for ev in gate_events:
                        rr = [x for x in coverage_rows if x["tier"] == tier and x["event"] == ev and float(x["radius_arcsec"]) == float(r) and int(x["k"]) == int(k)]
                        if rr:
                            vals.append(float(rr[0].get("frac_weight_true", np.nan)))
                    if vals:
                        mat[i, j] = float(np.nanmedian(np.asarray(vals, dtype=float)))
            fig, ax = plt.subplots(figsize=(7.8, 3.6))
            im = ax.imshow(mat, origin="lower", aspect="auto", interpolation="nearest")
            ax.set_xticks(np.arange(len(ks)))
            ax.set_xticklabels([str(k) for k in ks], fontsize=7, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(rs)))
            ax.set_yticklabels([str(int(r)) for r in rs], fontsize=7)
            ax.set_xlabel("K (top host candidates)")
            ax.set_ylabel("radius (arcsec)")
            ax.set_title(f"Median spec-z weight coverage (gate events) tier={tier}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="median frac_weight_true")
            fig.tight_layout()
            fig.savefig(fig_dir / f"coverage_heatmap_tier{tier}.png", dpi=180)
            plt.close(fig)
    except Exception:
        pass

    # --------------------------------------------
    # Report + summary
    # --------------------------------------------
    summary = {
        "outdir": str(out_root),
        "top_events": list(top_events),
        "gate_events": list(gate_events),
        "radii_arcsec": [float(r) for r in radii],
        "k_grid": [int(k) for k in k_grid],
        "k_gain_min_frac": float(k_gain_min),
        "legitimacy_gate": {"median_shift_over_true_max": gate_median_max, "max_event_shift_over_true_max": gate_event_max},
        "best_points": best_points,
        "scoring_best_points_rows": score_rows,
    }
    _write_json(out_root / "summary.json", summary)

    # Short report markdown.
    rep = []
    rep.append("# Spec-z Coverage Maxout (Pre-O4)\n")
    rep.append(f"Output directory: `{out_root}`\n")
    rep.append("## Inputs\n")
    rep.append(f"- gap_root: `{gap_root}`\n")
    rep.append(f"- run_label: `{run_label}`\n")
    rep.append(f"- recon_run_dir: `{recon_run_dir}`\n")
    rep.append(f"- z_max: {z_max}\n")
    rep.append(f"- gate_events: {', '.join(gate_events)}\n")
    rep.append("\n## Legitimacy Gate\n")
    rep.append(f"- median(shifted/true) < {gate_median_max}\n")
    rep.append(f"- max_event(shifted/true) < {gate_event_max}\n")
    rep.append("\n## Best Legit Operating Points\n")
    if best_points:
        for t, bp in best_points.items():
            rep.append(f"- Tier {t}: r={bp['radius_arcsec']:.1f}\"  K={bp['k']}  median frac_weight_true={bp['median_frac_weight_true_gate_events']:.4f}\n")
    else:
        rep.append("- No tier produced a radius passing the gate under the configured thresholds.\n")
    if score_rows:
        rep.append("\n## Override Scoring At Best Points (Spectral-only + selection)\n")
        for r in score_rows:
            rep.append(f"- Tier {r['tier']}: r={r['radius_arcsec']:.1f}\" K={r['k']}  ΔLPD_total={r['delta_lpd_total']:.3f} (baseline {r['baseline_delta_lpd_total']:.3f})\n")
    rep.append("\n## Artifacts\n")
    rep.append(f"- coverage grid: `{(tab_dir / 'coverage_grid_full.csv')}`\n")
    rep.append(f"- gate by radius: `{(tab_dir / 'false_match_gate_by_radius.csv')}`\n")
    rep.append(f"- max points: `{(tab_dir / 'max_coverage_operating_points.csv')}`\n")
    rep.append(f"- specz manifest: `{(raw_dir / 'specz_catalog_manifest.json')}`\n")
    (out_root / "report.md").write_text("".join(rep), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
