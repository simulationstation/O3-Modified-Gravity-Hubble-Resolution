#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Prefer local source tree when available (important when another editable install is active).
_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if _REPO_SRC.is_dir():
    _src = str(_REPO_SRC)
    if _src not in sys.path:
        sys.path.insert(0, _src)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.camb_utils import CambFiducial, camb_clpp
from entropy_horizon_recon.ingest_planck_lensing_bandpowers import load_planck_lensing_bandpowers
from entropy_horizon_recon.likelihoods_planck_lensing_mg import PlanckLensingBandpowerMGLogLike
from entropy_horizon_recon.mg_lensing_response import MGLensingResponseParams
from entropy_horizon_recon.sirens import load_mu_forward_posterior


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _stats_1d(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    if x.size <= 0:
        return {
            "mean": float("nan"),
            "sd": float("nan"),
            "p16": float("nan"),
            "p50": float("nan"),
            "p84": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "p16": float(np.percentile(x, 16.0)),
        "p50": float(np.percentile(x, 50.0)),
        "p84": float(np.percentile(x, 84.0)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _plot_chi2_compare(
    *,
    chi2_baseline: np.ndarray,
    chi2_fit: np.ndarray,
    chi2_ref: float,
    out_path: Path,
) -> None:
    bins = min(40, max(12, int(np.sqrt(chi2_baseline.size))))
    plt.figure(figsize=(7.6, 4.8))
    plt.hist(chi2_baseline, bins=bins, alpha=0.55, label="Baseline MG posterior draws")
    plt.hist(chi2_fit, bins=bins, alpha=0.55, label="After MG-response refit")
    plt.axvline(float(chi2_ref), color="k", linestyle="--", linewidth=1.5, label="Planck-reference model")
    plt.xlabel(r"$\chi^2$ vs Planck lensing bandpowers")
    plt.ylabel("Count")
    plt.title("Lensing fit quality before and after MG-response refit")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_mstar_hist(*, ratio: np.ndarray, out_path: Path) -> None:
    bins = min(40, max(12, int(np.sqrt(ratio.size))))
    plt.figure(figsize=(7.2, 4.8))
    plt.hist(ratio, bins=bins, alpha=0.75)
    plt.axvline(1.0, color="k", linestyle="--", linewidth=1.2, label=r"$M_*^2(z=0)/M_*^2(\mathrm{hi\text{-}z)=1}$")
    plt.xlabel(r"$M_*^2(z=0)/M_*^2(\mathrm{hi\text{-}z})$")
    plt.ylabel("Count")
    plt.title("Best-fit effective Planck-mass ratio from lensing refit")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_bandpowers_overlay(
    *,
    ell: np.ndarray,
    data_clpp: np.ndarray,
    data_err: np.ndarray,
    model_ref: np.ndarray,
    baseline_q16: np.ndarray,
    baseline_q50: np.ndarray,
    baseline_q84: np.ndarray,
    fit_q16: np.ndarray,
    fit_q50: np.ndarray,
    fit_q84: np.ndarray,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8.2, 5.0))
    plt.errorbar(ell, data_clpp, yerr=data_err, fmt="o", capsize=3, label="Planck lensing bandpowers")
    plt.plot(ell, model_ref, "k--", linewidth=1.4, label="Planck-reference CAMB")
    plt.fill_between(ell, baseline_q16, baseline_q84, alpha=0.20, label="Baseline MG draws 68%")
    plt.plot(ell, baseline_q50, linewidth=1.8, label="Baseline MG median")
    plt.fill_between(ell, fit_q16, fit_q84, alpha=0.20, label="MG-response-refit 68%")
    plt.plot(ell, fit_q50, linewidth=2.0, label="MG-response-refit median")
    plt.xlabel(r"$L_{\mathrm{eff}}$")
    plt.ylabel(r"$C_L^{\phi\phi}$")
    plt.title("Planck lensing data vs baseline and MG-response-refit predictions")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run an MG-aware Planck-lensing consistency refit by applying a phenomenological "
            "MG response (effective Planck-mass running + slip/Poisson scaling + ell tilt) "
            "to draw-level CAMB C_L^{phi phi} predictions."
        )
    )
    ap.add_argument("--run-dir", required=True, help="Finished run directory with samples/mu_forward_posterior.npz.")
    ap.add_argument("--out", default=None, help="Output directory (default: outputs/hubble_tension_mg_lensing_refit_<UTCSTAMP>).")
    ap.add_argument("--draws", type=int, default=64, help="Posterior draws to evaluate.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for draw subsampling.")
    ap.add_argument("--heartbeat-sec", type=float, default=20.0, help="Heartbeat interval while evaluating CAMB.")
    ap.add_argument("--planck-lensing-dataset", default="consext8", choices=["consext8", "agr2"])

    ap.add_argument("--h0-planck-ref", type=float, default=67.4)
    ap.add_argument("--omega-m-planck", type=float, default=0.315)
    ap.add_argument("--omega-k-planck", type=float, default=0.0)
    ap.add_argument("--sigma8-planck", type=float, default=0.811)
    ap.add_argument("--sigma8-mode", choices=["sampled", "fixed"], default="sampled")
    ap.add_argument("--sigma8-fixed", type=float, default=0.811)

    ap.add_argument("--ombh2-fid", type=float, default=0.02237)
    ap.add_argument("--ns-fid", type=float, default=0.9649)
    ap.add_argument("--tau-fid", type=float, default=0.0544)
    ap.add_argument("--mnu-fid", type=float, default=0.06)
    ap.add_argument("--As-fid", type=float, default=2.1e-9)

    # MG response settings (mu0 and eta0 can be fixed externally if desired).
    ap.add_argument("--mu0", type=float, default=1.0)
    ap.add_argument("--eta0", type=float, default=1.0)
    ap.add_argument("--ell-pivot", type=float, default=200.0)
    ap.add_argument("--response-power", type=float, default=1.0)
    ap.add_argument("--clip-min", type=float, default=0.05)
    ap.add_argument("--clip-max", type=float, default=20.0)
    ap.add_argument("--fit-log10-mstar2-min", type=float, default=-0.6)
    ap.add_argument("--fit-log10-mstar2-max", type=float, default=0.6)
    ap.add_argument("--fit-ell-tilt-min", type=float, default=-0.6)
    ap.add_argument("--fit-ell-tilt-max", type=float, default=0.6)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"hubble_tension_mg_lensing_refit_{_utc_stamp()}"
    tab_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    post = load_mu_forward_posterior(args.run_dir)
    n_all = int(post.H0.size)
    if n_all <= 0:
        raise ValueError("No posterior draws in run_dir.")

    rng = np.random.default_rng(int(args.seed))
    n_use = min(int(args.draws), n_all)
    if n_use < n_all:
        idx = np.sort(rng.choice(n_all, size=n_use, replace=False))
    else:
        idx = np.arange(n_all, dtype=int)

    H0 = np.asarray(post.H0[idx], dtype=float)
    omega_m0 = np.asarray(post.omega_m0[idx], dtype=float)
    omega_k0 = np.asarray(post.omega_k0[idx], dtype=float)

    if args.sigma8_mode == "sampled" and post.sigma8_0 is not None:
        sigma8_0 = np.asarray(post.sigma8_0[idx], dtype=float)
        sigma8_source = "posterior"
    else:
        sigma8_0 = np.full(n_use, float(args.sigma8_fixed), dtype=float)
        sigma8_source = "fixed"

    repo_root = Path(__file__).resolve().parents[1]
    paths = DataPaths(repo_root=repo_root)
    lens = load_planck_lensing_bandpowers(paths=paths, dataset=args.planck_lensing_dataset)
    ell_eff = np.asarray(np.rint(lens.ell_eff), dtype=int)
    ell_eff = np.clip(ell_eff, 2, None)
    n_ell = int(ell_eff.size)

    fid = CambFiducial(
        ombh2=float(args.ombh2_fid),
        ns=float(args.ns_fid),
        tau=float(args.tau_fid),
        mnu=float(args.mnu_fid),
        As=float(args.As_fid),
    )

    clpp_draws = np.full((n_use, n_ell), np.nan, dtype=float)
    fail_mask = np.zeros(n_use, dtype=bool)
    t_last_hb = time.time()
    for i in range(n_use):
        try:
            clpp_draws[i] = camb_clpp(
                H0=float(H0[i]),
                omega_m0=float(omega_m0[i]),
                omega_k0=float(omega_k0[i]),
                sigma8_0=float(sigma8_0[i]),
                ell=ell_eff,
                fid=fid,
            )
        except Exception:
            fail_mask[i] = True
        now = time.time()
        if (now - t_last_hb) >= float(args.heartbeat_sec) or i == (n_use - 1):
            done = i + 1
            pct = 100.0 * float(done) / float(max(1, n_use))
            print(f"[heartbeat] CAMB draws_done={done}/{n_use} ({pct:.1f}%) failures={int(np.sum(fail_mask))}", flush=True)
            t_last_hb = now

    ok = ~fail_mask
    if not np.any(ok):
        raise RuntimeError("All CAMB draw evaluations failed; cannot continue.")

    clpp_ok = clpp_draws[ok]
    H0_ok = H0[ok]
    om_ok = omega_m0[ok]
    ok_ok = omega_k0[ok]
    s8_ok = sigma8_0[ok]
    idx_ok = idx[ok]

    model_ref = camb_clpp(
        H0=float(args.h0_planck_ref),
        omega_m0=float(args.omega_m_planck),
        omega_k0=float(args.omega_k_planck),
        sigma8_0=float(args.sigma8_planck),
        ell=ell_eff,
        fid=fid,
    )

    like = PlanckLensingBandpowerMGLogLike.from_data(
        ell_eff=ell_eff.astype(float),
        clpp=np.asarray(lens.clpp, dtype=float),
        cov=np.asarray(lens.cov, dtype=float),
        meta=dict(lens.meta),
    )

    chi2_ref = like.chi2(model_ref)
    chi2_baseline = np.array([like.chi2(clpp_ok[i]) for i in range(clpp_ok.shape[0])], dtype=float)

    fit_log10_mstar2 = np.full(clpp_ok.shape[0], np.nan, dtype=float)
    fit_ell_tilt = np.full(clpp_ok.shape[0], np.nan, dtype=float)
    chi2_fit = np.full(clpp_ok.shape[0], np.nan, dtype=float)
    clpp_fit = np.full_like(clpp_ok, np.nan)
    fit_success = np.zeros(clpp_ok.shape[0], dtype=bool)

    bounds = [
        (float(args.fit_log10_mstar2_min), float(args.fit_log10_mstar2_max)),
        (float(args.fit_ell_tilt_min), float(args.fit_ell_tilt_max)),
    ]

    t_last_hb = time.time()
    for i in range(clpp_ok.shape[0]):
        clpp_gr = clpp_ok[i]

        def objective(x: np.ndarray) -> float:
            params = MGLensingResponseParams(
                log10_mstar2_ratio_0=float(x[0]),
                mu0=float(args.mu0),
                eta0=float(args.eta0),
                ell_tilt=float(x[1]),
                ell_pivot=float(args.ell_pivot),
                response_power=float(args.response_power),
                clip_min=float(args.clip_min),
                clip_max=float(args.clip_max),
            )
            model = like.apply_mg_response(clpp_gr, params)
            return like.chi2(model)

        x0 = np.array([0.0, 0.0], dtype=float)
        res = minimize(objective, x0=x0, method="L-BFGS-B", bounds=bounds)

        if np.isfinite(res.fun):
            fit_success[i] = bool(res.success)
            fit_log10_mstar2[i] = float(res.x[0])
            fit_ell_tilt[i] = float(res.x[1])
            params_best = MGLensingResponseParams(
                log10_mstar2_ratio_0=float(res.x[0]),
                mu0=float(args.mu0),
                eta0=float(args.eta0),
                ell_tilt=float(res.x[1]),
                ell_pivot=float(args.ell_pivot),
                response_power=float(args.response_power),
                clip_min=float(args.clip_min),
                clip_max=float(args.clip_max),
            )
            clpp_fit[i] = like.apply_mg_response(clpp_gr, params_best)
            chi2_fit[i] = like.chi2(clpp_fit[i])

        now = time.time()
        if (now - t_last_hb) >= float(args.heartbeat_sec) or i == (clpp_ok.shape[0] - 1):
            done = i + 1
            pct = 100.0 * float(done) / float(max(1, clpp_ok.shape[0]))
            okn = int(np.sum(np.isfinite(chi2_fit[:done])))
            print(f"[heartbeat] fit draws_done={done}/{clpp_ok.shape[0]} ({pct:.1f}%) fit_ok={okn}", flush=True)
            t_last_hb = now

    fit_ok = np.isfinite(chi2_fit)
    if not np.any(fit_ok):
        raise RuntimeError("MG-response fit failed for all draws.")

    chi2_b_ok = chi2_baseline[fit_ok]
    chi2_f_ok = chi2_fit[fit_ok]
    clpp_b_ok = clpp_ok[fit_ok]
    clpp_f_ok = clpp_fit[fit_ok]
    fit_log10_ok = fit_log10_mstar2[fit_ok]
    fit_tilt_ok = fit_ell_tilt[fit_ok]
    H0_fit_ok = H0_ok[fit_ok]
    om_fit_ok = om_ok[fit_ok]
    ok_fit_ok = ok_ok[fit_ok]
    s8_fit_ok = s8_ok[fit_ok]
    idx_fit_ok = idx_ok[fit_ok]
    fit_success_ok = fit_success[fit_ok]

    mstar2_ratio = np.power(10.0, fit_log10_ok)
    mstar2_drop_pct = 100.0 * (1.0 - mstar2_ratio)
    dchi2_fit_minus_baseline = chi2_f_ok - chi2_b_ok

    ratio_b = clpp_b_ok / np.clip(model_ref.reshape((1, -1)), 1e-300, np.inf)
    ratio_f = clpp_f_ok / np.clip(model_ref.reshape((1, -1)), 1e-300, np.inf)
    ratio_b_q16 = np.percentile(ratio_b, 16.0, axis=0)
    ratio_b_q50 = np.percentile(ratio_b, 50.0, axis=0)
    ratio_b_q84 = np.percentile(ratio_b, 84.0, axis=0)
    ratio_f_q16 = np.percentile(ratio_f, 16.0, axis=0)
    ratio_f_q50 = np.percentile(ratio_f, 50.0, axis=0)
    ratio_f_q84 = np.percentile(ratio_f, 84.0, axis=0)
    clpp_b_q16 = np.percentile(clpp_b_ok, 16.0, axis=0)
    clpp_b_q50 = np.percentile(clpp_b_ok, 50.0, axis=0)
    clpp_b_q84 = np.percentile(clpp_b_ok, 84.0, axis=0)
    clpp_f_q16 = np.percentile(clpp_f_ok, 16.0, axis=0)
    clpp_f_q50 = np.percentile(clpp_f_ok, 50.0, axis=0)
    clpp_f_q84 = np.percentile(clpp_f_ok, 84.0, axis=0)

    j100 = int(np.argmin(np.abs(ell_eff - 100)))
    j300 = int(np.argmin(np.abs(ell_eff - 300)))
    data_clpp = np.asarray(lens.clpp, dtype=float)
    data_err = np.sqrt(np.clip(np.diag(np.asarray(lens.cov, dtype=float)), 0.0, np.inf))

    _plot_chi2_compare(chi2_baseline=chi2_b_ok, chi2_fit=chi2_f_ok, chi2_ref=float(chi2_ref), out_path=fig_dir / "chi2_baseline_vs_mgfit.png")
    _plot_mstar_hist(ratio=mstar2_ratio, out_path=fig_dir / "mstar2_ratio_fit_hist.png")
    _plot_bandpowers_overlay(
        ell=ell_eff.astype(float),
        data_clpp=data_clpp,
        data_err=data_err,
        model_ref=model_ref,
        baseline_q16=clpp_b_q16,
        baseline_q50=clpp_b_q50,
        baseline_q84=clpp_b_q84,
        fit_q16=clpp_f_q16,
        fit_q50=clpp_f_q50,
        fit_q84=clpp_f_q84,
        out_path=fig_dir / "clpp_overlay_baseline_vs_mgfit.png",
    )

    draw_table = np.column_stack(
        [
            idx_fit_ok,
            H0_fit_ok,
            om_fit_ok,
            ok_fit_ok,
            s8_fit_ok,
            fit_log10_ok,
            mstar2_ratio,
            mstar2_drop_pct,
            fit_tilt_ok,
            chi2_b_ok,
            chi2_f_ok,
            dchi2_fit_minus_baseline,
            fit_success_ok.astype(float),
        ]
    )
    np.savetxt(
        tab_dir / "draw_level_refit.csv",
        draw_table,
        delimiter=",",
        header=(
            "posterior_index,H0,omega_m0,omega_k0,sigma8_0,"
            "fit_log10_mstar2_ratio_0,fit_mstar2_ratio_0,fit_mstar2_drop_pct,"
            "fit_ell_tilt,chi2_baseline,chi2_mg_fit,delta_chi2_fit_minus_baseline,optimizer_success"
        ),
        comments="",
    )

    np.savez_compressed(
        tab_dir / "clpp_refit_draws.npz",
        ell_eff=ell_eff.astype(float),
        clpp_data=data_clpp,
        clpp_ref=model_ref,
        clpp_baseline=clpp_b_ok,
        clpp_fit=clpp_f_ok,
        chi2_baseline=chi2_b_ok,
        chi2_fit=chi2_f_ok,
    )

    summary: dict[str, Any] = {
        "created_utc": _utc_now(),
        "elapsed_sec": float(time.time() - t0),
        "run_dir": str(Path(args.run_dir).resolve()),
        "draws_requested": int(args.draws),
        "draws_total_available": int(n_all),
        "draws_used": int(n_use),
        "draws_camb_ok": int(np.sum(ok)),
        "draws_fit_ok": int(np.sum(fit_ok)),
        "sigma8_source": sigma8_source,
        "planck_lensing_dataset": str(args.planck_lensing_dataset),
        "planck_reference": {
            "H0": float(args.h0_planck_ref),
            "omega_m0": float(args.omega_m_planck),
            "omega_k0": float(args.omega_k_planck),
            "sigma8_0": float(args.sigma8_planck),
        },
        "camb_fiducial": asdict(fid),
        "mg_response_assumptions": {
            "mu0_fixed": float(args.mu0),
            "eta0_fixed": float(args.eta0),
            "ell_pivot": float(args.ell_pivot),
            "response_power": float(args.response_power),
            "clip_min": float(args.clip_min),
            "clip_max": float(args.clip_max),
            "fit_bounds": {
                "log10_mstar2_ratio_0": [float(args.fit_log10_mstar2_min), float(args.fit_log10_mstar2_max)],
                "ell_tilt": [float(args.fit_ell_tilt_min), float(args.fit_ell_tilt_max)],
            },
        },
        "parameter_stats_from_draws": {
            "H0": _stats_1d(H0_fit_ok),
            "omega_m0": _stats_1d(om_fit_ok),
            "omega_k0": _stats_1d(ok_fit_ok),
            "sigma8_0": _stats_1d(s8_fit_ok),
        },
        "fit_parameter_stats": {
            "log10_mstar2_ratio_0": _stats_1d(fit_log10_ok),
            "mstar2_ratio_0": _stats_1d(mstar2_ratio),
            "mstar2_drop_pct": _stats_1d(mstar2_drop_pct),
            "ell_tilt": _stats_1d(fit_tilt_ok),
        },
        "chi2": {
            "chi2_planck_ref_model": float(chi2_ref),
            "chi2_baseline_draws": _stats_1d(chi2_b_ok),
            "chi2_mg_refit_draws": _stats_1d(chi2_f_ok),
            "delta_chi2_fit_minus_baseline": _stats_1d(dchi2_fit_minus_baseline),
            "p_baseline_better_than_ref": float(np.mean(chi2_b_ok < float(chi2_ref))),
            "p_refit_better_than_ref": float(np.mean(chi2_f_ok < float(chi2_ref))),
            "p_refit_better_than_baseline": float(np.mean(chi2_f_ok < chi2_b_ok)),
            "optimizer_success_fraction": float(np.mean(fit_success_ok)),
        },
        "headline_multipoles": {
            "near_L100": {
                "ell_eff": float(ell_eff[j100]),
                "baseline_ratio_q50": float(ratio_b_q50[j100]),
                "baseline_delta_frac_pct_q50": float(100.0 * (ratio_b_q50[j100] - 1.0)),
                "refit_ratio_q50": float(ratio_f_q50[j100]),
                "refit_delta_frac_pct_q50": float(100.0 * (ratio_f_q50[j100] - 1.0)),
            },
            "near_L300": {
                "ell_eff": float(ell_eff[j300]),
                "baseline_ratio_q50": float(ratio_b_q50[j300]),
                "baseline_delta_frac_pct_q50": float(100.0 * (ratio_b_q50[j300] - 1.0)),
                "refit_ratio_q50": float(ratio_f_q50[j300]),
                "refit_delta_frac_pct_q50": float(100.0 * (ratio_f_q50[j300] - 1.0)),
            },
        },
        "scope_notes": {
            "modeled": [
                "Planck 2018 lensing bandpower likelihood",
                "Draw-level CAMB C_L^{phi phi} from MG posterior",
                "Phenomenological MG response with effective Planck-mass running and ell tilt",
            ],
            "not_modeled": [
                "Full Horndeski/EFT perturbation-equation Boltzmann evolution",
                "Primary TT/TE/EE and nuisance-parameter joint likelihood refit",
            ],
        },
    }
    _write_json_atomic(tab_dir / "summary.json", summary)

    md = [
        "# MG-Aware Planck Lensing Refit",
        "",
        f"- Generated UTC: `{summary['created_utc']}`",
        f"- Run dir: `{summary['run_dir']}`",
        f"- Draws CAMB ok: `{summary['draws_camb_ok']}/{summary['draws_used']}`",
        f"- Draws fit ok: `{summary['draws_fit_ok']}/{summary['draws_camb_ok']}`",
        "",
        "## Main Diagnostics",
        "",
        (
            "- Baseline draw chi2 median: "
            f"`{summary['chi2']['chi2_baseline_draws']['p50']:.3f}`"
        ),
        (
            "- MG-refit draw chi2 median: "
            f"`{summary['chi2']['chi2_mg_refit_draws']['p50']:.3f}`"
        ),
        (
            "- Planck-reference chi2: "
            f"`{summary['chi2']['chi2_planck_ref_model']:.3f}`"
        ),
        (
            "- Refit beats Planck-reference in "
            f"`{100.0 * summary['chi2']['p_refit_better_than_ref']:.1f}%` of draws."
        ),
        (
            "- Median fitted M_*^2 ratio (z=0 / high-z): "
            f"`{summary['fit_parameter_stats']['mstar2_ratio_0']['p50']:.3f}` "
            f"(drop `{summary['fit_parameter_stats']['mstar2_drop_pct']['p50']:+.2f}%`)."
        ),
        "",
        "## Artifacts",
        "",
        "- `tables/summary.json`",
        "- `tables/draw_level_refit.csv`",
        "- `tables/clpp_refit_draws.npz`",
        "- `figures/chi2_baseline_vs_mgfit.png`",
        "- `figures/mstar2_ratio_fit_hist.png`",
        "- `figures/clpp_overlay_baseline_vs_mgfit.png`",
    ]
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[done] wrote {tab_dir / 'summary.json'}", flush=True)
    print(f"[done] wrote {out_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
