#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.sirens import (
    MuForwardPosterior,
    load_mu_forward_posterior,
    predict_r_gw_em,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _parse_run_dirs(text: str) -> list[str]:
    vals = [t.strip() for t in str(text).split(",") if t.strip()]
    if not vals:
        raise ValueError("No run dirs provided.")
    return vals


def _subset_post(post: MuForwardPosterior, idx: np.ndarray) -> MuForwardPosterior:
    return MuForwardPosterior(
        x_grid=post.x_grid,
        logmu_x_samples=post.logmu_x_samples[idx],
        z_grid=post.z_grid,
        H_samples=post.H_samples[idx],
        H0=post.H0[idx],
        omega_m0=post.omega_m0[idx],
        omega_k0=post.omega_k0[idx],
        sigma8_0=(None if post.sigma8_0 is None else post.sigma8_0[idx]),
    )


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


def _plot_ratio_band(
    *,
    z: np.ndarray,
    q16: np.ndarray,
    q50: np.ndarray,
    q84: np.ndarray,
    out_path: Path,
    convention: str,
) -> None:
    plt.figure(figsize=(7.8, 4.8))
    plt.fill_between(z, q16, q84, alpha=0.25, linewidth=0.0, label="68% band")
    plt.plot(z, q50, linewidth=2.0, label="median")
    plt.axhline(1.0, color="k", linewidth=1.0, alpha=0.6, label="GR")
    plt.xlabel("z")
    plt.ylabel(r"$R(z)=D_L^{GW}/D_L^{EM}$")
    plt.title(f"Phase 4 Forecast: GW/EM Distance Ratio (convention {convention})")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Concrete forward test (Phase 4): forecast GW-vs-EM distance ratio "
            "R(z)=D_L^GW/D_L^EM from existing posterior runs."
        )
    )
    ap.add_argument(
        "--run-dirs",
        default=(
            "outputs/finalization/highpower_multistart_v2/M0_start101,"
            "outputs/finalization/highpower_multistart_v2/M0_start303,"
            "outputs/finalization/highpower_multistart_v2/M0_start404"
        ),
    )
    ap.add_argument("--out", default="outputs/forward_tests/phase4_distance_ratio")
    ap.add_argument("--draws-per-run", type=int, default=800)
    ap.add_argument("--seed", type=int, default=9204)
    ap.add_argument("--z-min", type=float, default=0.02)
    ap.add_argument("--z-max", type=float, default=0.62)
    ap.add_argument("--z-n", type=int, default=60)
    ap.add_argument("--convention", choices=["A", "B"], default="A")

    ap.add_argument("--gate-min-highz-median-deviation", type=float, default=0.05)
    ap.add_argument("--gate-min-direction-coherence-frac", type=float, default=0.9)
    ap.add_argument("--gate-require-gr-outside-68-highz", action="store_true", default=True)
    ap.add_argument(
        "--calibration-robust",
        action="store_true",
        help="Enable calibration-marginalized robustness gate for GW/EM distance-ratio diagnostics.",
    )
    ap.add_argument(
        "--calibration-samples",
        type=int,
        default=4000,
        help="Number of nuisance draws for calibration-robust gate.",
    )
    ap.add_argument("--calibration-seed", type=int, default=29404, help="RNG seed for calibration robustness sampling.")
    ap.add_argument(
        "--gate-min-robust-pass-frac",
        type=float,
        default=0.60,
        help="Calibration-robust gate passes if nuisance-marginal pass fraction >= this value.",
    )
    ap.add_argument(
        "--calib-gw-distance-frac-sigma",
        type=float,
        default=0.03,
        help="Stddev for multiplicative GW distance calibration bias fraction.",
    )
    ap.add_argument(
        "--calib-em-distance-frac-sigma",
        type=float,
        default=0.03,
        help="Stddev for multiplicative EM distance calibration bias fraction.",
    )
    ap.add_argument(
        "--calib-ratio-drift-per-z-sigma",
        type=float,
        default=0.03,
        help="Stddev for residual redshift drift in multiplicative ratio calibration per unit z.",
    )
    ap.add_argument("--fail-on-gate", action="store_true")
    args = ap.parse_args()

    if args.z_n < 3:
        raise ValueError("z-n must be >= 3.")
    if not (args.z_min > 0.0 and args.z_max > args.z_min):
        raise ValueError("Require 0 < z-min < z-max.")
    if not (0.0 <= float(args.gate_min_robust_pass_frac) <= 1.0):
        raise ValueError("gate-min-robust-pass-frac must be in [0,1].")
    if int(args.calibration_samples) < 100:
        raise ValueError("calibration-samples must be >= 100.")
    if float(args.calib_gw_distance_frac_sigma) < 0.0:
        raise ValueError("calib-gw-distance-frac-sigma must be >= 0.")
    if float(args.calib_em_distance_frac_sigma) < 0.0:
        raise ValueError("calib-em-distance-frac-sigma must be >= 0.")
    if float(args.calib_ratio_drift_per_z_sigma) < 0.0:
        raise ValueError("calib-ratio-drift-per-z-sigma must be >= 0.")

    out_dir = Path(args.out).resolve()
    tab_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    run_dirs = _parse_run_dirs(args.run_dirs)
    z_eval = np.linspace(float(args.z_min), float(args.z_max), int(args.z_n))
    key_z = [0.10, 0.30, 0.50, float(args.z_max)]
    key_i = [int(np.argmin(np.abs(z_eval - z0))) for z0 in key_z]

    all_ratio: list[np.ndarray] = []
    per_run: list[dict[str, Any]] = []
    for run in run_dirs:
        post = load_mu_forward_posterior(run)
        n_all = int(post.H0.size)
        n_use = min(int(args.draws_per_run), n_all)
        idx = np.sort(rng.choice(n_all, size=n_use, replace=False)) if n_use < n_all else np.arange(n_all, dtype=int)
        ps = _subset_post(post, idx)
        _, ratio = predict_r_gw_em(ps, z_eval=z_eval, convention=args.convention, allow_extrapolation=False)
        all_ratio.append(ratio)

        med = np.percentile(ratio, 50.0, axis=0)
        row = {
            "run_dir": str(Path(run).resolve()),
            "draws_total": int(n_all),
            "draws_used": int(n_use),
            "highz_median_ratio": float(med[-1]),
            "highz_median_deviation": float(med[-1] - 1.0),
            "keypoint_medians": {f"z{key_z[j]:.2f}": float(med[key_i[j]]) for j in range(len(key_z))},
        }
        per_run.append(row)

    ratio_all = np.vstack(all_ratio)
    q16 = np.percentile(ratio_all, 16.0, axis=0)
    q50 = np.percentile(ratio_all, 50.0, axis=0)
    q84 = np.percentile(ratio_all, 84.0, axis=0)
    mean = np.mean(ratio_all, axis=0)

    highz_med_dev = float(q50[-1] - 1.0)
    highz_gr_outside_68 = bool((1.0 < float(q16[-1])) or (1.0 > float(q84[-1])))

    signs = []
    z_ref_i = int(np.argmin(np.abs(z_eval - 0.50)))
    for row in per_run:
        d = float(row["keypoint_medians"][f"z{key_z[2]:.2f}"] - 1.0)
        signs.append(np.sign(d))
    signs = np.asarray(signs, dtype=float)
    pos = int(np.sum(signs > 0))
    neg = int(np.sum(signs < 0))
    coherence = float(max(pos, neg) / max(1, signs.size))

    checks = [
        {
            "name": "highz_median_deviation_min",
            "value": highz_med_dev,
            "threshold": float(args.gate_min_highz_median_deviation),
            "pass": abs(highz_med_dev) >= float(args.gate_min_highz_median_deviation),
        },
        {
            "name": "direction_coherence_fraction",
            "value": coherence,
            "threshold": float(args.gate_min_direction_coherence_frac),
            "pass": coherence >= float(args.gate_min_direction_coherence_frac),
        },
        {
            "name": "gr_outside_68_at_highz",
            "value": bool(highz_gr_outside_68),
            "pass": bool(highz_gr_outside_68) if bool(args.gate_require_gr_outside_68_highz) else True,
        },
    ]
    strict_pass = bool(all(bool(c["pass"]) for c in checks))

    calibration_robust_gate: dict[str, Any] | None = None
    if bool(args.calibration_robust):
        rng_cal = np.random.default_rng(int(args.calibration_seed))
        n_cal = int(args.calibration_samples)
        gw_frac = float(args.calib_gw_distance_frac_sigma) * rng_cal.normal(size=n_cal)
        em_frac = float(args.calib_em_distance_frac_sigma) * rng_cal.normal(size=n_cal)
        drift = float(args.calib_ratio_drift_per_z_sigma) * rng_cal.normal(size=n_cal)

        base_scale = np.clip((1.0 + gw_frac) / np.clip(1.0 + em_frac, 1e-6, np.inf), 1e-6, np.inf)
        z_mid = float(key_z[2])
        z_anchor = float(args.z_min)
        z_high = float(args.z_max)
        k_mid = np.clip(base_scale * (1.0 + drift * (z_mid - z_anchor)), 1e-6, np.inf)
        k_high = np.clip(base_scale * (1.0 + drift * (z_high - z_anchor)), 1e-6, np.inf)

        highz_q16_eff = k_high * float(q16[-1])
        highz_q50_eff = k_high * float(q50[-1])
        highz_q84_eff = k_high * float(q84[-1])
        highz_dev_eff = highz_q50_eff - 1.0
        gr_outside_eff = (1.0 < highz_q16_eff) | (1.0 > highz_q84_eff)

        z05_key = f"z{key_z[2]:.2f}"
        run_medians_z05 = np.asarray(
            [float(row["keypoint_medians"][z05_key]) for row in per_run],
            dtype=float,
        )
        d_mid = k_mid.reshape((-1, 1)) * run_medians_z05.reshape((1, -1)) - 1.0
        pos = np.sum(d_mid > 0.0, axis=1)
        neg = np.sum(d_mid < 0.0, axis=1)
        coherence_eff = np.maximum(pos, neg) / max(1, run_medians_z05.size)

        pass_dev = np.abs(highz_dev_eff) >= float(args.gate_min_highz_median_deviation)
        pass_coh = coherence_eff >= float(args.gate_min_direction_coherence_frac)
        pass_gr = gr_outside_eff if bool(args.gate_require_gr_outside_68_highz) else np.ones((n_cal,), dtype=bool)
        pass_all = pass_dev & pass_coh & pass_gr

        robust_pass_fraction = float(np.mean(pass_all))
        robust_pass = bool(robust_pass_fraction >= float(args.gate_min_robust_pass_frac))

        calibration_robust_gate = {
            "enabled": True,
            "mode": "catalog_calibration_marginalized_proxy",
            "pass": robust_pass,
            "pass_fraction": robust_pass_fraction,
            "pass_fraction_threshold": float(args.gate_min_robust_pass_frac),
            "checks": [
                {
                    "name": "highz_median_deviation_min",
                    "threshold": float(args.gate_min_highz_median_deviation),
                    "pass_fraction": float(np.mean(pass_dev)),
                },
                {
                    "name": "direction_coherence_fraction",
                    "threshold": float(args.gate_min_direction_coherence_frac),
                    "pass_fraction": float(np.mean(pass_coh)),
                },
                {
                    "name": "gr_outside_68_at_highz",
                    "enabled": bool(args.gate_require_gr_outside_68_highz),
                    "pass_fraction": float(np.mean(pass_gr)),
                },
            ],
            "nuisance_priors": {
                "gw_distance_frac_sigma": float(args.calib_gw_distance_frac_sigma),
                "em_distance_frac_sigma": float(args.calib_em_distance_frac_sigma),
                "ratio_drift_per_z_sigma": float(args.calib_ratio_drift_per_z_sigma),
            },
            "effective_metric_stats": {
                "highz_median_deviation_effective": _stats_1d(highz_dev_eff),
                "direction_coherence_effective": _stats_1d(coherence_eff),
                "gw_over_em_scale_effective": _stats_1d(k_high),
            },
        }

    with (tab_dir / "ratio_quantiles.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["z", "ratio_q16", "ratio_q50", "ratio_q84", "ratio_mean"])
        for i in range(z_eval.size):
            w.writerow([z_eval[i], q16[i], q50[i], q84[i], mean[i]])

    with (tab_dir / "per_run_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run_dir",
                "draws_total",
                "draws_used",
                "highz_median_ratio",
                "highz_median_deviation",
                "median_ratio_z0.10",
                "median_ratio_z0.30",
                "median_ratio_z0.50",
                f"median_ratio_z{float(args.z_max):.2f}",
            ]
        )
        for row in per_run:
            km = row["keypoint_medians"]
            w.writerow(
                [
                    row["run_dir"],
                    row["draws_total"],
                    row["draws_used"],
                    row["highz_median_ratio"],
                    row["highz_median_deviation"],
                    km["z0.10"],
                    km["z0.30"],
                    km["z0.50"],
                    km[f"z{float(args.z_max):.2f}"],
                ]
            )

    _plot_ratio_band(
        z=z_eval,
        q16=q16,
        q50=q50,
        q84=q84,
        out_path=fig_dir / "ratio_band.png",
        convention=args.convention,
    )

    summary = {
        "created_utc": _utc_now(),
        "phase": "phase4_distance_ratio",
        "mode": "forecast_envelope",
        "inputs": {
            "run_dirs": [str(Path(r).resolve()) for r in run_dirs],
            "draws_per_run": int(args.draws_per_run),
            "seed": int(args.seed),
            "z_min": float(args.z_min),
            "z_max": float(args.z_max),
            "z_n": int(args.z_n),
            "convention": str(args.convention),
        },
        "combined": {
            "ratio_stats_highz": _stats_1d(ratio_all[:, -1]),
            "highz_median_deviation": highz_med_dev,
            "highz_gr_outside_68": highz_gr_outside_68,
            "direction_coherence_fraction": coherence,
            "keypoint_quantiles": {
                f"z{key_z[j]:.2f}": {
                    "q16": float(q16[key_i[j]]),
                    "q50": float(q50[key_i[j]]),
                    "q84": float(q84[key_i[j]]),
                }
                for j in range(len(key_z))
            },
        },
        "per_run": per_run,
        "strict_gate": {
            "pass": strict_pass,
            "checks": checks,
        },
        "interpretation_note": (
            "This is a forward forecast envelope from posterior draws. "
            "It does not yet ingest an independent GW siren catalog likelihood."
        ),
    }
    if calibration_robust_gate is not None:
        summary["calibration_robust_gate"] = calibration_robust_gate
    _write_json_atomic(tab_dir / "summary.json", summary)

    lines = [
        "# Forward Test Phase 4: GW/EM Distance-Ratio Forecast",
        "",
        f"- Generated UTC: `{summary['created_utc']}`",
        f"- Convention: `{args.convention}`",
        f"- Runs: `{len(run_dirs)}`",
        "",
        "## Combined Keypoints",
    ]
    for kz, q in summary["combined"]["keypoint_quantiles"].items():
        lines.append(f"- `{kz}`: `q16/q50/q84 = {q['q16']:.4f}/{q['q50']:.4f}/{q['q84']:.4f}`")
    lines.extend(
        [
            "",
            "## Strict Gate",
            "",
            f"- Result: `{'PASS' if strict_pass else 'FAIL'}`",
        ]
    )
    for c in checks:
        lines.append(f"- `{c['name']}`: `{'PASS' if c['pass'] else 'FAIL'}` (value={c['value']})")
    if calibration_robust_gate is not None:
        lines.extend(
            [
                "",
                "## Calibration-Robust Gate",
                "",
                (
                    f"- Result: `{'PASS' if calibration_robust_gate['pass'] else 'FAIL'}` "
                    f"(pass fraction `{calibration_robust_gate['pass_fraction']:.3f}`)"
                ),
            ]
        )
        for c in calibration_robust_gate["checks"]:
            lines.append(f"- `{c['name']}`: pass fraction `{float(c['pass_fraction']):.3f}`")
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "- Forecast-only phase: independent GW catalog cross-check is still required for full test closure.",
            "",
            "## Artifacts",
            "",
            "- `tables/summary.json`",
            "- `tables/ratio_quantiles.csv`",
            "- `tables/per_run_summary.csv`",
            "- `figures/ratio_band.png`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {tab_dir / 'summary.json'}")
    print(f"[done] strict_gate={'PASS' if strict_pass else 'FAIL'}")

    if bool(args.fail_on_gate) and (not strict_pass):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
