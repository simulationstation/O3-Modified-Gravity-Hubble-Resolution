#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from entropy_horizon_recon.constants import PhysicalConstants
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


def _omega_radiation(H0: float) -> float:
    # Omega_gamma*h^2 ~= 2.469e-5 (T_CMB=2.7255K), incl. neutrinos: factor ~1.691.
    # => Omega_r*h^2 ~= 4.180e-5
    h = float(H0) / 100.0
    if h <= 0.0:
        return float("nan")
    return 4.180e-5 / (h * h)


def _E2(z: np.ndarray, *, H0: float, omega_m0: float, omega_k0: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    omega_r0 = _omega_radiation(H0)
    omega_l0 = 1.0 - float(omega_m0) - float(omega_k0) - float(omega_r0)
    zp1 = 1.0 + z
    return (
        float(omega_r0) * zp1**4
        + float(omega_m0) * zp1**3
        + float(omega_k0) * zp1**2
        + float(omega_l0)
    )


def _comoving_angular_distance(
    *,
    H0: float,
    omega_m0: float,
    omega_k0: float,
    z_star: float,
    constants: PhysicalConstants,
    n_int: int,
) -> float:
    z = np.linspace(0.0, float(z_star), int(n_int))
    e2 = _E2(z, H0=float(H0), omega_m0=float(omega_m0), omega_k0=float(omega_k0))
    if np.any(~np.isfinite(e2)) or np.any(e2 <= 0.0):
        return float("nan")
    invE = 1.0 / np.sqrt(e2)
    Dc = float(constants.c_km_s / float(H0) * np.trapezoid(invE, z))

    ok = float(omega_k0)
    if abs(ok) < 1e-12:
        return Dc
    arg = math.sqrt(abs(ok)) * float(H0) * Dc / float(constants.c_km_s)
    if ok > 0.0:
        return float((constants.c_km_s / (float(H0) * math.sqrt(ok))) * math.sinh(arg))
    return float((constants.c_km_s / (float(H0) * math.sqrt(abs(ok)))) * math.sin(arg))


def _theta_star(
    *,
    H0: float,
    omega_m0: float,
    omega_k0: float,
    r_d_Mpc: float,
    z_star: float,
    constants: PhysicalConstants,
    n_int: int,
) -> float:
    Dm = _comoving_angular_distance(
        H0=float(H0),
        omega_m0=float(omega_m0),
        omega_k0=float(omega_k0),
        z_star=float(z_star),
        constants=constants,
        n_int=int(n_int),
    )
    if not np.isfinite(Dm) or Dm <= 0.0:
        return float("nan")
    return float(float(r_d_Mpc) / Dm)


def _infer_h0_from_theta(
    *,
    theta_obs: float,
    omega_m_assumed: float,
    omega_k_assumed: float,
    r_d_assumed: float,
    z_star: float,
    constants: PhysicalConstants,
    n_int: int,
    h0_lo: float = 40.0,
    h0_hi: float = 100.0,
) -> float:
    if not np.isfinite(theta_obs) or theta_obs <= 0.0:
        return float("nan")

    def f(h0: float) -> float:
        t = _theta_star(
            H0=float(h0),
            omega_m0=float(omega_m_assumed),
            omega_k0=float(omega_k_assumed),
            r_d_Mpc=float(r_d_assumed),
            z_star=float(z_star),
            constants=constants,
            n_int=int(n_int),
        )
        return float(t - theta_obs)

    flo = f(float(h0_lo))
    fhi = f(float(h0_hi))
    if np.isfinite(flo) and np.isfinite(fhi) and flo == 0.0:
        return float(h0_lo)
    if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi < 0.0:
        try:
            return float(brentq(f, float(h0_lo), float(h0_hi), maxiter=128))
        except Exception:
            pass

    # Fallback grid search when no bracket is found.
    grid = np.linspace(float(h0_lo), float(h0_hi), 121)
    vals = np.array([f(g) for g in grid], dtype=float)
    ok = np.isfinite(vals)
    if not np.any(ok):
        return float("nan")
    j = int(np.argmin(np.abs(vals[ok])))
    return float(grid[np.where(ok)[0][j]])


def _plot_h0_hist(*, h0_true: np.ndarray, h0_inferred: np.ndarray, out: Path) -> None:
    plt.figure(figsize=(7.2, 4.8))
    bins = min(50, max(12, int(np.sqrt(h0_true.size))))
    plt.hist(h0_true, bins=bins, alpha=0.55, label=r"$H_0$ true (MG posterior)")
    plt.hist(h0_inferred, bins=bins, alpha=0.55, label=r"$H_0$ inferred under GR from synthetic CMB")
    plt.xlabel(r"$H_0$ [km/s/Mpc]")
    plt.ylabel("Count")
    plt.title(r"Early-universe inference bias under MG-truth assumption")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def _plot_relief_hist(*, relief: np.ndarray, out: Path) -> None:
    plt.figure(figsize=(7.2, 4.8))
    bins = min(50, max(12, int(np.sqrt(relief.size))))
    plt.hist(relief, bins=bins, alpha=0.75)
    plt.axvline(np.mean(relief), color="k", linewidth=1.5, linestyle="--", label="mean")
    plt.xlabel("Relief fraction")
    plt.ylabel("Count")
    plt.title("Fraction of local-vs-Planck gap absorbed by GR mis-inference")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def _plot_rd_shift(*, rd_shift_pct: np.ndarray, out: Path) -> None:
    plt.figure(figsize=(7.2, 4.8))
    bins = min(50, max(12, int(np.sqrt(rd_shift_pct.size))))
    plt.hist(rd_shift_pct, bins=bins, alpha=0.75)
    plt.axvline(0.0, color="k", linewidth=1.0, alpha=0.7)
    plt.xlabel(r"Required $\Delta r_d / r_d$ for $H_0=H_0^{\mathrm{local}}$ [%]")
    plt.ylabel("Count")
    plt.title(r"How much sound-horizon shift is needed for full reconciliation")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Study whether early-universe GR inference can be biased under MG truth by forward-modeling "
            "compressed CMB observables (theta*, lensing proxy) from MG posterior draws and inverting under GR assumptions."
        )
    )
    ap.add_argument("--run-dir", required=True, help="Finished run directory with samples/mu_forward_posterior.npz.")
    ap.add_argument("--out", default=None, help="Output directory (default: outputs/hubble_tension_early_universe_bias_<UTCSTAMP>).")
    ap.add_argument("--draws", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--z-star", type=float, default=1089.0, help="Effective CMB decoupling redshift for compressed theta* modeling.")
    ap.add_argument("--n-int", type=int, default=6000, help="Integration points for distance-to-last-scattering calculations.")
    ap.add_argument("--theta-noise-frac", type=float, default=0.0, help="Fractional Gaussian noise for synthetic theta* observations.")
    ap.add_argument("--lensing-noise", type=float, default=0.0, help="Absolute Gaussian noise on synthetic lensing-amplitude proxy.")

    ap.add_argument("--h0-local-ref", type=float, default=73.0)
    ap.add_argument("--h0-planck-ref", type=float, default=67.4)
    ap.add_argument("--r-d-planck-ref", type=float, default=147.09)
    ap.add_argument("--omega-m-planck-ref", type=float, default=0.315)
    ap.add_argument("--omega-k-assumed", type=float, default=0.0)
    ap.add_argument("--sigma8-planck-ref", type=float, default=0.811)
    ap.add_argument("--lensing-alpha", type=float, default=0.25)

    ap.add_argument("--omega-m-assumed-mode", choices=["fixed_planck", "from_lensing_proxy", "true_draw"], default="fixed_planck")
    ap.add_argument("--r-d-assumed-mode", choices=["planck_fixed", "true_draw"], default="planck_fixed")
    ap.add_argument("--heartbeat-sec", type=float, default=30.0)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"hubble_tension_early_universe_bias_{_utc_stamp()}"
    tab_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    post = load_mu_forward_posterior(args.run_dir)
    n_all = int(post.H0.size)
    if n_all <= 0:
        raise ValueError("No posterior draws available.")
    if post.sigma8_0 is None:
        raise ValueError("Posterior is missing sigma8_0, required for lensing-proxy mode.")

    npz_path = Path(args.run_dir) / "samples" / "mu_forward_posterior.npz"
    with np.load(npz_path, allow_pickle=False) as d:
        if "r_d_Mpc" not in d.files:
            raise ValueError("Posterior file is missing r_d_Mpc; cannot perform theta* mapping.")
        r_d_all = np.asarray(d["r_d_Mpc"], dtype=float)

    rng = np.random.default_rng(int(args.seed))
    n_use = min(int(args.draws), n_all)
    if n_use < n_all:
        idx = np.sort(rng.choice(n_all, size=n_use, replace=False))
    else:
        idx = np.arange(n_all, dtype=int)

    H0_true = np.asarray(post.H0[idx], dtype=float)
    om_true = np.asarray(post.omega_m0[idx], dtype=float)
    ok_true = np.asarray(post.omega_k0[idx], dtype=float)
    s8_true = np.asarray(post.sigma8_0[idx], dtype=float)
    r_d_true = np.asarray(r_d_all[idx], dtype=float)

    constants = PhysicalConstants()
    s8om_ref = float(args.sigma8_planck_ref) * float(args.omega_m_planck_ref) ** float(args.lensing_alpha)
    A_lens_true = np.square((s8_true * np.power(np.clip(om_true, 1e-12, np.inf), float(args.lensing_alpha))) / s8om_ref)

    theta_true = np.full(n_use, np.nan, dtype=float)
    theta_obs = np.full(n_use, np.nan, dtype=float)
    A_lens_obs = np.full(n_use, np.nan, dtype=float)
    om_assumed = np.full(n_use, np.nan, dtype=float)
    r_d_assumed = np.full(n_use, np.nan, dtype=float)
    H0_inferred_gr = np.full(n_use, np.nan, dtype=float)
    r_d_needed_for_local = np.full(n_use, np.nan, dtype=float)
    r_d_shift_needed_frac = np.full(n_use, np.nan, dtype=float)

    t_last = time.time()
    for i in range(n_use):
        theta_t = _theta_star(
            H0=float(H0_true[i]),
            omega_m0=float(om_true[i]),
            omega_k0=float(ok_true[i]),
            r_d_Mpc=float(r_d_true[i]),
            z_star=float(args.z_star),
            constants=constants,
            n_int=int(args.n_int),
        )
        theta_true[i] = theta_t
        if np.isfinite(theta_t) and theta_t > 0.0:
            theta_obs[i] = float(theta_t * (1.0 + float(args.theta_noise_frac) * rng.normal()))
        else:
            continue

        A_lens_obs[i] = float(A_lens_true[i] + float(args.lensing_noise) * rng.normal())
        if args.omega_m_assumed_mode == "fixed_planck":
            om_assumed[i] = float(args.omega_m_planck_ref)
        elif args.omega_m_assumed_mode == "true_draw":
            om_assumed[i] = float(om_true[i])
        else:
            Aeff = max(float(A_lens_obs[i]), 1e-8)
            s8om_est = math.sqrt(Aeff) * s8om_ref
            om_est = (s8om_est / float(args.sigma8_planck_ref)) ** (1.0 / float(args.lensing_alpha))
            om_assumed[i] = float(np.clip(om_est, 0.05, 0.8))

        if args.r_d_assumed_mode == "planck_fixed":
            r_d_assumed[i] = float(args.r_d_planck_ref)
        else:
            r_d_assumed[i] = float(r_d_true[i])

        H0_inferred_gr[i] = _infer_h0_from_theta(
            theta_obs=float(theta_obs[i]),
            omega_m_assumed=float(om_assumed[i]),
            omega_k_assumed=float(args.omega_k_assumed),
            r_d_assumed=float(r_d_assumed[i]),
            z_star=float(args.z_star),
            constants=constants,
            n_int=int(args.n_int),
        )

        Dm_local = _comoving_angular_distance(
            H0=float(args.h0_local_ref),
            omega_m0=float(om_assumed[i]),
            omega_k0=float(args.omega_k_assumed),
            z_star=float(args.z_star),
            constants=constants,
            n_int=int(args.n_int),
        )
        if np.isfinite(Dm_local) and Dm_local > 0.0:
            rneed = float(theta_obs[i] * Dm_local)
            r_d_needed_for_local[i] = rneed
            if r_d_assumed[i] > 0.0:
                r_d_shift_needed_frac[i] = float(rneed / r_d_assumed[i] - 1.0)

        now = time.time()
        if (now - t_last) >= float(args.heartbeat_sec) or i == (n_use - 1):
            done = i + 1
            okn = int(np.sum(np.isfinite(H0_inferred_gr[:done])))
            pct = 100.0 * float(done) / float(max(1, n_use))
            print(f"[heartbeat] draws_done={done}/{n_use} ({pct:.1f}%) inferred_ok={okn}", flush=True)
            t_last = now

    ok_mask = np.isfinite(H0_inferred_gr)
    if not np.any(ok_mask):
        raise RuntimeError("No successful GR inversions.")

    H0_t = H0_true[ok_mask]
    H0_g = H0_inferred_gr[ok_mask]
    om_a = om_assumed[ok_mask]
    rd_t = r_d_true[ok_mask]
    rd_a = r_d_assumed[ok_mask]
    theta_t_ok = theta_true[ok_mask]
    theta_o_ok = theta_obs[ok_mask]
    Al_t_ok = A_lens_true[ok_mask]
    Al_o_ok = A_lens_obs[ok_mask]
    rd_shift_ok = r_d_shift_needed_frac[ok_mask]

    baseline_gap = abs(float(args.h0_local_ref) - float(args.h0_planck_ref))
    relief = 1.0 - np.abs(float(args.h0_local_ref) - H0_g) / baseline_gap
    closure_signed = (H0_g - float(args.h0_planck_ref)) / baseline_gap
    closure_clipped = np.clip(closure_signed, 0.0, 1.0)
    delta_h0 = H0_g - H0_t

    np.savetxt(
        tab_dir / "draw_level.csv",
        np.column_stack(
            [
                H0_t,
                H0_g,
                delta_h0,
                om_true[ok_mask],
                om_a,
                ok_true[ok_mask],
                rd_t,
                rd_a,
                theta_t_ok,
                theta_o_ok,
                Al_t_ok,
                Al_o_ok,
                relief,
                rd_shift_ok,
            ]
        ),
        delimiter=",",
        header=(
            "H0_true,H0_inferred_gr,delta_H0_inferred_minus_true,"
            "omega_m_true,omega_m_assumed,omega_k_true,r_d_true,r_d_assumed,"
            "theta_star_true,theta_star_obs,A_lens_true,A_lens_obs,"
            "relief_fraction,rd_shift_needed_frac_for_H0local"
        ),
        comments="",
    )

    _plot_h0_hist(h0_true=H0_t, h0_inferred=H0_g, out=fig_dir / "h0_true_vs_inferred_hist.png")
    _plot_relief_hist(relief=relief, out=fig_dir / "relief_fraction_hist.png")
    _plot_rd_shift(rd_shift_pct=100.0 * rd_shift_ok, out=fig_dir / "rd_shift_needed_hist.png")

    summary: dict[str, Any] = {
        "created_utc": _utc_now(),
        "elapsed_sec": float(time.time() - t0),
        "run_dir": str(Path(args.run_dir).resolve()),
        "draws_requested": int(args.draws),
        "draws_total_available": int(n_all),
        "draws_used": int(n_use),
        "draws_inferred_ok": int(np.sum(ok_mask)),
        "assumptions": {
            "z_star": float(args.z_star),
            "n_int": int(args.n_int),
            "theta_noise_frac": float(args.theta_noise_frac),
            "lensing_noise": float(args.lensing_noise),
            "omega_m_assumed_mode": str(args.omega_m_assumed_mode),
            "r_d_assumed_mode": str(args.r_d_assumed_mode),
            "omega_k_assumed": float(args.omega_k_assumed),
            "h0_local_ref": float(args.h0_local_ref),
            "h0_planck_ref": float(args.h0_planck_ref),
            "r_d_planck_ref": float(args.r_d_planck_ref),
            "omega_m_planck_ref": float(args.omega_m_planck_ref),
            "sigma8_planck_ref": float(args.sigma8_planck_ref),
            "lensing_alpha": float(args.lensing_alpha),
        },
        "posterior_true_stats": {
            "H0_true": _stats_1d(H0_t),
            "omega_m_true": _stats_1d(om_true[ok_mask]),
            "omega_k_true": _stats_1d(ok_true[ok_mask]),
            "sigma8_true": _stats_1d(s8_true[ok_mask]),
            "r_d_true": _stats_1d(rd_t),
            "theta_star_true": _stats_1d(theta_t_ok),
            "A_lens_true": _stats_1d(Al_t_ok),
        },
        "gr_inference_stats": {
            "H0_inferred": _stats_1d(H0_g),
            "delta_H0_inferred_minus_true": _stats_1d(delta_h0),
            "omega_m_assumed": _stats_1d(om_a),
            "r_d_assumed": _stats_1d(rd_a),
            "theta_star_obs": _stats_1d(theta_o_ok),
        },
        "tension_reconciliation": {
            "baseline_gap_local_minus_planck": float(baseline_gap),
            "relief_fraction": _stats_1d(relief),
            "gap_closure_signed_vs_planck_to_local": _stats_1d(closure_signed),
            "gap_closure_clipped_0_1": _stats_1d(closure_clipped),
            "remaining_gap_local_minus_inferred": _stats_1d(np.abs(float(args.h0_local_ref) - H0_g)),
            "p_inferred_above_planck_ref": float(np.mean(H0_g > float(args.h0_planck_ref))),
            "p_inferred_above_70": float(np.mean(H0_g > 70.0)),
            "p_inferred_above_72": float(np.mean(H0_g > 72.0)),
            "p_inferred_at_or_above_local_ref": float(np.mean(H0_g >= float(args.h0_local_ref))),
            "p_inferred_above_local_ref": float(np.mean(H0_g > float(args.h0_local_ref))),
        },
        "required_rd_shift_for_full_match": {
            "rd_shift_needed_frac": _stats_1d(rd_shift_ok),
            "rd_shift_needed_pct": _stats_1d(100.0 * rd_shift_ok),
            "note": "Fractional shift in assumed r_d needed (with current theta* obs and assumed Omega_m) for inferred H0 to match local reference.",
        },
        "scope_notes": {
            "modeled": [
                "Compressed CMB geometry observable theta* = r_d / D_M(z*)",
                "Optional CMB-lensing-amplitude proxy to set assumed Omega_m in GR inversion",
                "Draw-level MG-truth posterior from run_dir",
            ],
            "not_modeled": [
                "Full TT/TE/EE likelihood with nuisance/marginalization",
                "Full modified perturbation equations in early universe",
                "A Boltzmann-level MG transfer-function refit",
            ],
        },
    }
    _write_json_atomic(tab_dir / "summary.json", summary)

    md = [
        "# Early-Universe Inference Bias Study (MG Truth -> GR Inference)",
        "",
        f"- Generated UTC: `{summary['created_utc']}`",
        f"- Run dir: `{summary['run_dir']}`",
        f"- Draws inferred: `{summary['draws_inferred_ok']}/{summary['draws_used']}`",
        "",
        "## Main Result",
        "",
        (
            "- Inferred early-universe `H0` under GR assumptions: "
            f"`mean={summary['gr_inference_stats']['H0_inferred']['mean']:.3f}`, "
            f"`p16/p50/p84={summary['gr_inference_stats']['H0_inferred']['p16']:.3f}/"
            f"{summary['gr_inference_stats']['H0_inferred']['p50']:.3f}/"
            f"{summary['gr_inference_stats']['H0_inferred']['p84']:.3f}` km/s/Mpc."
        ),
        (
            "- Relief fraction of local-vs-Planck baseline: "
            f"`mean={summary['tension_reconciliation']['relief_fraction']['mean']:.3f}`, "
            f"`p16/p50/p84={summary['tension_reconciliation']['relief_fraction']['p16']:.3f}/"
            f"{summary['tension_reconciliation']['relief_fraction']['p50']:.3f}/"
            f"{summary['tension_reconciliation']['relief_fraction']['p84']:.3f}`."
        ),
        (
            "- Gap closure fraction (Planck->local, clipped to [0,1]): "
            f"`mean={summary['tension_reconciliation']['gap_closure_clipped_0_1']['mean']:.3f}`, "
            f"`p16/p50/p84={summary['tension_reconciliation']['gap_closure_clipped_0_1']['p16']:.3f}/"
            f"{summary['tension_reconciliation']['gap_closure_clipped_0_1']['p50']:.3f}/"
            f"{summary['tension_reconciliation']['gap_closure_clipped_0_1']['p84']:.3f}`."
        ),
        (
            "- Required sound-horizon shift for full local match: "
            f"`mean={summary['required_rd_shift_for_full_match']['rd_shift_needed_pct']['mean']:.2f}%`, "
            f"`p16/p50/p84={summary['required_rd_shift_for_full_match']['rd_shift_needed_pct']['p16']:.2f}%/"
            f"{summary['required_rd_shift_for_full_match']['rd_shift_needed_pct']['p50']:.2f}%/"
            f"{summary['required_rd_shift_for_full_match']['rd_shift_needed_pct']['p84']:.2f}%`."
        ),
        "",
        "## Assumption Modes",
        "",
        f"- Omega_m assumed mode: `{args.omega_m_assumed_mode}`",
        f"- r_d assumed mode: `{args.r_d_assumed_mode}`",
        "",
        "## Artifacts",
        "",
        "- `tables/summary.json`",
        "- `tables/draw_level.csv`",
        "- `figures/h0_true_vs_inferred_hist.png`",
        "- `figures/relief_fraction_hist.png`",
        "- `figures/rd_shift_needed_hist.png`",
    ]
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[done] wrote {tab_dir / 'summary.json'}", flush=True)
    print(f"[done] wrote {out_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
