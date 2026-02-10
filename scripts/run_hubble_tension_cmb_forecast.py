#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.camb_utils import CambFiducial, camb_clpp
from entropy_horizon_recon.ingest_planck_lensing_bandpowers import load_planck_lensing_bandpowers
from entropy_horizon_recon.likelihoods_planck_lensing_bandpowers import PlanckLensingBandpowerLogLike
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


def _plot_clpp_bandpowers(
    *,
    ell: np.ndarray,
    data_clpp: np.ndarray,
    data_err: np.ndarray,
    model_ref: np.ndarray,
    model_q16: np.ndarray,
    model_q50: np.ndarray,
    model_q84: np.ndarray,
    out_path: Path,
) -> None:
    plt.figure(figsize=(7.6, 4.8))
    plt.errorbar(
        ell,
        data_clpp,
        yerr=data_err,
        fmt="o",
        capsize=3,
        label="Planck lensing bandpowers",
    )
    plt.plot(ell, model_ref, "k--", linewidth=1.5, label="Planck-reference CAMB model")
    plt.fill_between(ell, model_q16, model_q84, alpha=0.25, linewidth=0.0, label="MG posterior 68% band")
    plt.plot(ell, model_q50, linewidth=2.0, label="MG posterior median")
    plt.xlabel(r"$L_{\mathrm{eff}}$")
    plt.ylabel(r"$C_L^{\phi\phi}$")
    plt.title("Planck lensing bandpowers vs MG-implied CMB lensing prediction")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_ratio_to_ref(
    *,
    ell: np.ndarray,
    ratio_q16: np.ndarray,
    ratio_q50: np.ndarray,
    ratio_q84: np.ndarray,
    out_path: Path,
) -> None:
    plt.figure(figsize=(7.6, 4.8))
    plt.fill_between(ell, ratio_q16, ratio_q84, alpha=0.25, linewidth=0.0, label="68% band")
    plt.plot(ell, ratio_q50, linewidth=2.0, label="median")
    plt.axhline(1.0, color="k", linewidth=1.0, alpha=0.6)
    plt.xlabel(r"$L_{\mathrm{eff}}$")
    plt.ylabel(r"$C_L^{\phi\phi}(\mathrm{MG}) / C_L^{\phi\phi}(\mathrm{Planck\ ref})$")
    plt.title(r"Fractional CMB lensing shift implied by MG posterior")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_chi2_hist(
    *,
    chi2_draws: np.ndarray,
    chi2_ref: float,
    chi2_template: float,
    out_path: Path,
) -> None:
    plt.figure(figsize=(7.2, 4.6))
    plt.hist(chi2_draws, bins=min(40, max(10, int(np.sqrt(chi2_draws.size)))), alpha=0.75, label="MG posterior draws")
    plt.axvline(float(chi2_ref), color="k", linestyle="--", linewidth=1.5, label="Planck-reference model")
    plt.axvline(float(chi2_template), color="tab:red", linestyle=":", linewidth=1.5, label="Template model")
    plt.xlabel(r"$\chi^2$ vs Planck lensing bandpowers")
    plt.ylabel("Count")
    plt.title(r"Goodness-of-fit distribution under MG posterior")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Forecast CMB lensing changes implied by an MG posterior run by propagating draw-level "
            "(H0, Om0, Ok0, sigma8) into CAMB C_L^{phi phi} and comparing to Planck 2018 lensing bandpowers."
        )
    )
    ap.add_argument("--run-dir", required=True, help="Finished run directory with samples/mu_forward_posterior.npz.")
    ap.add_argument("--out", default=None, help="Output directory (default: outputs/hubble_tension_cmb_forecast_<UTCSTAMP>).")
    ap.add_argument("--draws", type=int, default=256, help="Number of posterior draws to evaluate (subsampled without replacement).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed used for draw subsampling.")
    ap.add_argument("--heartbeat-sec", type=float, default=30.0, help="Progress heartbeat interval while evaluating CAMB.")
    ap.add_argument("--planck-lensing-dataset", default="consext8", choices=["consext8", "agr2"])
    ap.add_argument(
        "--model-mode",
        default="template_proxy",
        choices=["template_proxy", "camb"],
        help="Prediction engine: fast template scaling proxy (default) or full CAMB.",
    )

    ap.add_argument("--h0-planck-ref", type=float, default=67.4)
    ap.add_argument("--omega-m-planck", type=float, default=0.315)
    ap.add_argument("--omega-k-planck", type=float, default=0.0)
    ap.add_argument("--sigma8-planck", type=float, default=0.811)

    ap.add_argument("--sigma8-mode", choices=["sampled", "fixed"], default="sampled")
    ap.add_argument("--sigma8-fixed", type=float, default=0.811)
    ap.add_argument("--s8om-alpha", type=float, default=0.25, help="Alpha exponent for S8Om^alpha lensing-amplitude proxy.")

    ap.add_argument("--ombh2-fid", type=float, default=0.02237)
    ap.add_argument("--ns-fid", type=float, default=0.9649)
    ap.add_argument("--tau-fid", type=float, default=0.0544)
    ap.add_argument("--mnu-fid", type=float, default=0.06)
    ap.add_argument("--As-fid", type=float, default=2.1e-9)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"hubble_tension_cmb_forecast_{_utc_stamp()}"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

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

    s8 = sigma8_0 * np.sqrt(np.clip(omega_m0 / 0.3, 1e-12, np.inf))
    s8om = sigma8_0 * np.power(np.clip(omega_m0, 1e-12, np.inf), float(args.s8om_alpha))
    s8om_ref = float(args.sigma8_planck) * float(args.omega_m_planck) ** float(args.s8om_alpha)
    A_lens_proxy = np.square(s8om / s8om_ref)

    # Optional posterior r_d metadata for context.
    r_d_draws: np.ndarray | None = None
    npz_path = Path(args.run_dir) / "samples" / "mu_forward_posterior.npz"
    with np.load(npz_path, allow_pickle=False) as d:
        if "r_d_Mpc" in d.files:
            r_d_draws = np.asarray(d["r_d_Mpc"], dtype=float)[idx]

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
    if args.model_mode == "template_proxy":
        lens_like = PlanckLensingBandpowerLogLike.from_data(
            clpp=np.asarray(lens.clpp, dtype=float),
            cov=np.asarray(lens.cov, dtype=float),
            template_clpp=np.asarray(lens.template_clpp, dtype=float),
            alpha=float(args.s8om_alpha),
            s8om_fid=float(s8om_ref),
        )
        model_ref = lens_like.predict(omega_m0=float(args.omega_m_planck), sigma8_0=float(args.sigma8_planck))
        # Vectorized proxy forecast across draws.
        for i in range(n_use):
            clpp_draws[i] = lens_like.predict(omega_m0=float(omega_m0[i]), sigma8_0=float(sigma8_0[i]))
        print(f"[heartbeat] draws_done={n_use}/{n_use} (100.0%) failures=0", flush=True)
    else:
        model_ref = camb_clpp(
            H0=float(args.h0_planck_ref),
            omega_m0=float(args.omega_m_planck),
            omega_k0=float(args.omega_k_planck),
            sigma8_0=float(args.sigma8_planck),
            ell=ell_eff,
            fid=fid,
        )

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
                print(f"[heartbeat] draws_done={done}/{n_use} ({pct:.1f}%) failures={int(np.sum(fail_mask))}", flush=True)
                t_last_hb = now

    ok = ~fail_mask
    if not np.any(ok):
        raise RuntimeError("All CAMB evaluations failed; cannot build CMB forecast.")

    clpp_ok = clpp_draws[ok]
    H0_ok = H0[ok]
    om_ok = omega_m0[ok]
    ok_ok = omega_k0[ok]
    s8_ok = sigma8_0[ok]
    S8_ok = s8[ok]
    s8om_ok = s8om[ok]
    A_lens_ok = A_lens_proxy[ok]
    r_d_ok = r_d_draws[ok] if r_d_draws is not None else None

    ratio = clpp_ok / np.clip(model_ref.reshape((1, -1)), 1e-300, np.inf)
    ratio_q16 = np.percentile(ratio, 16.0, axis=0)
    ratio_q50 = np.percentile(ratio, 50.0, axis=0)
    ratio_q84 = np.percentile(ratio, 84.0, axis=0)
    model_q16 = np.percentile(clpp_ok, 16.0, axis=0)
    model_q50 = np.percentile(clpp_ok, 50.0, axis=0)
    model_q84 = np.percentile(clpp_ok, 84.0, axis=0)

    cov = np.asarray(lens.cov, dtype=float)
    cov_inv = np.linalg.inv(cov)
    data_clpp = np.asarray(lens.clpp, dtype=float)
    data_err = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

    def _chi2(model: np.ndarray) -> float:
        r = data_clpp - model
        return float(r.T @ cov_inv @ r)

    chi2_draws = np.array([_chi2(clpp_ok[i]) for i in range(clpp_ok.shape[0])], dtype=float)
    chi2_ref = _chi2(model_ref)
    chi2_template = _chi2(np.asarray(lens.template_clpp, dtype=float))
    delta_chi2_vs_ref = chi2_draws - chi2_ref

    # A compact per-L table for downstream plotting/reporting.
    ell_table: list[dict[str, float]] = []
    for j in range(n_ell):
        ell_table.append(
            {
                "ell_eff": float(ell_eff[j]),
                "clpp_planck_data": float(data_clpp[j]),
                "clpp_planck_ref_model": float(model_ref[j]),
                "clpp_mg_q16": float(model_q16[j]),
                "clpp_mg_q50": float(model_q50[j]),
                "clpp_mg_q84": float(model_q84[j]),
                "ratio_mg_to_ref_q16": float(ratio_q16[j]),
                "ratio_mg_to_ref_q50": float(ratio_q50[j]),
                "ratio_mg_to_ref_q84": float(ratio_q84[j]),
                "delta_frac_pct_q50": float(100.0 * (ratio_q50[j] - 1.0)),
            }
        )

    # Two human-readable anchor multipoles.
    j100 = int(np.argmin(np.abs(ell_eff - 100)))
    j300 = int(np.argmin(np.abs(ell_eff - 300)))

    summary: dict[str, Any] = {
        "created_utc": _utc_now(),
        "elapsed_sec": float(time.time() - t0),
        "run_dir": str(Path(args.run_dir).resolve()),
        "draws_requested": int(args.draws),
        "draws_total_available": int(n_all),
        "draws_used": int(n_use),
        "draws_successful": int(np.sum(ok)),
        "draws_failed": int(np.sum(fail_mask)),
        "model_mode": str(args.model_mode),
        "sigma8_source": sigma8_source,
        "planck_lensing_dataset": str(args.planck_lensing_dataset),
        "planck_reference": {
            "H0": float(args.h0_planck_ref),
            "omega_m0": float(args.omega_m_planck),
            "omega_k0": float(args.omega_k_planck),
            "sigma8_0": float(args.sigma8_planck),
            "s8om_alpha": float(args.s8om_alpha),
        },
        "camb_fiducial": asdict(fid),
        "posterior_parameter_stats": {
            "H0": _stats_1d(H0_ok),
            "omega_m0": _stats_1d(om_ok),
            "omega_k0": _stats_1d(ok_ok),
            "sigma8_0": _stats_1d(s8_ok),
            "S8": _stats_1d(S8_ok),
            "sigma8_omega_m_alpha": _stats_1d(s8om_ok),
            "A_lens_proxy": _stats_1d(A_lens_ok),
        },
        "r_d_mpc_stats": _stats_1d(r_d_ok) if r_d_ok is not None else None,
        "lensing_chi2": {
            "chi2_draws": _stats_1d(chi2_draws),
            "chi2_planck_ref_model": float(chi2_ref),
            "chi2_planck_template_model": float(chi2_template),
            "delta_chi2_draw_minus_ref": _stats_1d(delta_chi2_vs_ref),
            "p_draw_better_than_ref": float(np.mean(chi2_draws < chi2_ref)),
            "p_draw_worse_than_ref": float(np.mean(chi2_draws > chi2_ref)),
        },
        "ell_bandpower_summary": ell_table,
        "headline_multipoles": {
            "near_L100": {
                "ell_eff": float(ell_eff[j100]),
                "ratio_q16": float(ratio_q16[j100]),
                "ratio_q50": float(ratio_q50[j100]),
                "ratio_q84": float(ratio_q84[j100]),
                "delta_frac_pct_q50": float(100.0 * (ratio_q50[j100] - 1.0)),
            },
            "near_L300": {
                "ell_eff": float(ell_eff[j300]),
                "ratio_q16": float(ratio_q16[j300]),
                "ratio_q50": float(ratio_q50[j300]),
                "ratio_q84": float(ratio_q84[j300]),
                "delta_frac_pct_q50": float(100.0 * (ratio_q50[j300] - 1.0)),
            },
        },
        "assumptions_and_scope": {
            "what_is_modeled": [
                "Planck 2018 lensing bandpowers C_L^{phi phi}",
                (
                    "Draw-level propagation of (Omega_m0, sigma8_0) through template-scaled lensing proxy"
                    if args.model_mode == "template_proxy"
                    else "Draw-level propagation of (H0, Omega_m0, Omega_k0, sigma8_0) through CAMB"
                ),
            ],
            "what_is_held_fixed": [
                "Baryon density, tilt, optical depth, neutrino mass, and primordial-shape fiducials unless overridden",
            ],
            "not_included_here": [
                "Full modified-gravity perturbation equations for primary TT/TE/EE spectra",
                "A dedicated early-universe recombination-sector refit",
            ],
        },
    }
    _write_json_atomic(tab_dir / "summary.json", summary)

    np.savez_compressed(
        tab_dir / "clpp_draws.npz",
        ell_eff=ell_eff.astype(float),
        clpp_data=data_clpp,
        clpp_ref=model_ref,
        clpp_draws=clpp_ok,
        ratio_draws=ratio,
        chi2_draws=chi2_draws,
    )

    _plot_clpp_bandpowers(
        ell=ell_eff.astype(float),
        data_clpp=data_clpp,
        data_err=data_err,
        model_ref=model_ref,
        model_q16=model_q16,
        model_q50=model_q50,
        model_q84=model_q84,
        out_path=fig_dir / "clpp_bandpowers_vs_models.png",
    )
    _plot_ratio_to_ref(
        ell=ell_eff.astype(float),
        ratio_q16=ratio_q16,
        ratio_q50=ratio_q50,
        ratio_q84=ratio_q84,
        out_path=fig_dir / "clpp_ratio_to_ref.png",
    )
    _plot_chi2_hist(
        chi2_draws=chi2_draws,
        chi2_ref=chi2_ref,
        chi2_template=chi2_template,
        out_path=fig_dir / "clpp_chi2_distribution.png",
    )

    md = [
        "# CMB Forecast Under MG-Truth Assumption",
        "",
        f"- Generated UTC: `{summary['created_utc']}`",
        f"- Run dir: `{summary['run_dir']}`",
        f"- Draws evaluated: `{summary['draws_successful']}/{summary['draws_used']}`",
        f"- Planck lensing dataset: `{summary['planck_lensing_dataset']}`",
        "",
        "## Headline Predicted Lensing Shifts",
        "",
        (
            "- Near L~100 "
            f"(actual `{summary['headline_multipoles']['near_L100']['ell_eff']:.0f}`): "
            f"median `ΔC/C = {summary['headline_multipoles']['near_L100']['delta_frac_pct_q50']:+.2f}%` "
            f"(68%: `{100.0*(summary['headline_multipoles']['near_L100']['ratio_q16']-1.0):+.2f}%` "
            f"to `{100.0*(summary['headline_multipoles']['near_L100']['ratio_q84']-1.0):+.2f}%`)."
        ),
        (
            "- Near L~300 "
            f"(actual `{summary['headline_multipoles']['near_L300']['ell_eff']:.0f}`): "
            f"median `ΔC/C = {summary['headline_multipoles']['near_L300']['delta_frac_pct_q50']:+.2f}%` "
            f"(68%: `{100.0*(summary['headline_multipoles']['near_L300']['ratio_q16']-1.0):+.2f}%` "
            f"to `{100.0*(summary['headline_multipoles']['near_L300']['ratio_q84']-1.0):+.2f}%`)."
        ),
        "",
        "## Fit to Planck Lensing Bandpowers",
        "",
        f"- MG draw chi2 (median): `{summary['lensing_chi2']['chi2_draws']['p50']:.3f}`",
        f"- Planck-reference model chi2: `{summary['lensing_chi2']['chi2_planck_ref_model']:.3f}`",
        f"- P(draw beats reference): `{100.0*summary['lensing_chi2']['p_draw_better_than_ref']:.1f}%`",
        "",
        "## Derived Amplitude Proxies",
        "",
        f"- `A_lens_proxy` median: `{summary['posterior_parameter_stats']['A_lens_proxy']['p50']:.4f}`",
        f"- `A_lens_proxy` p16/p84: `{summary['posterior_parameter_stats']['A_lens_proxy']['p16']:.4f}` / `{summary['posterior_parameter_stats']['A_lens_proxy']['p84']:.4f}`",
        "",
        "## Scope Notes",
        "",
        (
            "- This forecast uses the fast template-scaling lensing proxy "
            "(calibrated in the codebase) for draw-level CMB lensing shifts."
            if args.model_mode == "template_proxy"
            else "- This forecast propagates draw-level cosmology through standard CAMB transfer functions."
        ),
        "- It is a lensing-focused CMB prediction target, not a full MG TT/TE/EE perturbation-sector refit.",
        "",
        "## Artifacts",
        "",
        "- `tables/summary.json`",
        "- `tables/clpp_draws.npz`",
        "- `figures/clpp_bandpowers_vs_models.png`",
        "- `figures/clpp_ratio_to_ref.png`",
        "- `figures/clpp_chi2_distribution.png`",
    ]
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[done] wrote {tab_dir / 'summary.json'}", flush=True)
    print(f"[done] wrote {out_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
