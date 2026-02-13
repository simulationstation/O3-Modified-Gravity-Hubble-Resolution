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

from entropy_horizon_recon.sirens import load_mu_forward_posterior


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _parse_run_dirs(text: str) -> list[str]:
    vals = [t.strip() for t in str(text).split(",") if t.strip()]
    if not vals:
        raise ValueError("No run dirs provided.")
    return vals


def _select_indices(n_all: int, n_use: int, rng: np.random.Generator) -> np.ndarray:
    if n_use <= 0:
        raise ValueError("n_use must be positive.")
    if n_use < n_all:
        return np.sort(rng.choice(n_all, size=n_use, replace=False))
    return np.arange(n_all, dtype=int)


def _normal_cdf_approx(x: np.ndarray) -> np.ndarray:
    # Fast erf approximation (GELU-style) to avoid scipy dependency in lightweight gates.
    xx = np.asarray(x, dtype=float)
    t = np.sqrt(2.0 / np.pi) * (xx + 0.044715 * (xx**3))
    return 0.5 * (1.0 + np.tanh(t))


def _compute_consistency_metrics(
    *,
    s8_draws: np.ndarray,
    s8_obs: float,
    s8_obs_sigma: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    stats = _stats_1d(s8_draws)
    q_obs = float(np.mean(s8_draws <= float(s8_obs)))
    z_obs = float(
        (stats["mean"] - float(s8_obs))
        / np.sqrt(max(1e-16, stats["sd"] ** 2 + float(s8_obs_sigma) ** 2))
    )

    n_ppc = int(max(10000, min(200000, 50 * s8_draws.size)))
    idx = rng.integers(0, s8_draws.size, size=n_ppc)
    y_rep = s8_draws[idx] + float(s8_obs_sigma) * rng.normal(size=n_ppc)
    p_left = float(np.mean(y_rep <= float(s8_obs)))
    ppc_two_sided = float(2.0 * min(p_left, 1.0 - p_left))

    return {
        "s8_stats": stats,
        "s8_obs": float(s8_obs),
        "s8_obs_sigma": float(s8_obs_sigma),
        "obs_quantile_in_predictive": q_obs,
        "obs_consistency_z": z_obs,
        "ppc_two_sided_p": ppc_two_sided,
    }


def _plot_combined_hist(
    *,
    s8_combined: np.ndarray,
    s8_obs: float,
    s8_obs_sigma: float,
    s8_planck_ref: float | None,
    out_path: Path,
) -> None:
    plt.figure(figsize=(7.6, 4.8))
    bins = min(60, max(16, int(np.sqrt(s8_combined.size))))
    plt.hist(s8_combined, bins=bins, alpha=0.75, density=False, label="MG posterior predictive S8")
    plt.axvline(float(s8_obs), color="tab:red", linewidth=1.6, label=f"Target S8 = {s8_obs:.3f}")
    plt.axvspan(
        float(s8_obs - s8_obs_sigma),
        float(s8_obs + s8_obs_sigma),
        color="tab:red",
        alpha=0.15,
        label=f"Target ±1σ ({s8_obs_sigma:.3f})",
    )
    if s8_planck_ref is not None and np.isfinite(float(s8_planck_ref)):
        plt.axvline(float(s8_planck_ref), color="tab:green", linewidth=1.4, linestyle="--", label=f"Planck ref = {s8_planck_ref:.3f}")
    plt.xlabel("S8")
    plt.ylabel("Count")
    plt.title("Forward Test Phase 2: Growth/S8 Predictive Distribution")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Concrete forward test (Phase 2): posterior-predictive S8 consistency check."
    )
    ap.add_argument(
        "--run-dirs",
        default=(
            "outputs/finalization/highpower_multistart_v2/M0_start101,"
            "outputs/finalization/highpower_multistart_v2/M0_start303,"
            "outputs/finalization/highpower_multistart_v2/M0_start404"
        ),
        help="Comma-separated posterior run directories.",
    )
    ap.add_argument("--out", default="outputs/forward_tests/phase2_growth_s8", help="Output directory.")
    ap.add_argument("--draws-per-run", type=int, default=1200, help="Max posterior draws to use per run.")
    ap.add_argument("--seed", type=int, default=9102, help="RNG seed.")

    ap.add_argument("--s8-obs", type=float, default=0.770, help="Observed S8 target to test against.")
    ap.add_argument("--s8-obs-sigma", type=float, default=0.020, help="Observed S8 1-sigma uncertainty.")
    ap.add_argument("--s8-planck-ref", type=float, default=0.832, help="Optional Planck-like reference shown in plots.")

    ap.add_argument("--gate-max-abs-z", type=float, default=2.0, help="Pass if |z_obs| <= this value.")
    ap.add_argument(
        "--gate-obs-quantile-range",
        default="0.05,0.95",
        help="Pass if observed S8 quantile in predictive is within [lo,hi].",
    )
    ap.add_argument("--gate-min-ppc-two-sided", type=float, default=0.05, help="Pass if PPC two-sided p >= this value.")
    ap.add_argument(
        "--calibration-robust",
        action="store_true",
        help="Enable calibration-marginalized robustness gate for external S8 anchor assumptions.",
    )
    ap.add_argument(
        "--calibration-samples",
        type=int,
        default=4000,
        help="Number of nuisance draws for calibration-robust gate.",
    )
    ap.add_argument("--calibration-seed", type=int, default=29102, help="RNG seed for calibration robustness sampling.")
    ap.add_argument(
        "--gate-min-robust-pass-frac",
        type=float,
        default=0.60,
        help="Calibration-robust gate passes if nuisance-marginal pass fraction >= this value.",
    )
    ap.add_argument(
        "--calib-s8-obs-add-sigma",
        type=float,
        default=0.020,
        help="Stddev for additive S8 anchor calibration bias.",
    )
    ap.add_argument(
        "--calib-s8-obs-frac-sigma",
        type=float,
        default=0.020,
        help="Stddev for fractional S8 anchor calibration bias.",
    )
    ap.add_argument(
        "--calib-s8-obs-sigma-frac-sigma",
        type=float,
        default=0.25,
        help="Stddev for multiplicative uncertainty on S8 observational sigma.",
    )
    ap.add_argument("--fail-on-gate", action="store_true", help="Return nonzero exit if strict gate fails.")
    args = ap.parse_args()

    run_dirs = _parse_run_dirs(args.run_dirs)
    q_tokens = [t.strip() for t in str(args.gate_obs_quantile_range).split(",") if t.strip()]
    if len(q_tokens) != 2:
        raise ValueError("gate-obs-quantile-range must be 'lo,hi'.")
    q_lo = float(q_tokens[0])
    q_hi = float(q_tokens[1])
    if not (0.0 <= q_lo < q_hi <= 1.0):
        raise ValueError("gate-obs-quantile-range must satisfy 0 <= lo < hi <= 1.")
    if not (0.0 <= float(args.gate_min_robust_pass_frac) <= 1.0):
        raise ValueError("gate-min-robust-pass-frac must be in [0,1].")
    if int(args.calibration_samples) < 100:
        raise ValueError("calibration-samples must be >= 100.")
    if float(args.calib_s8_obs_add_sigma) < 0.0:
        raise ValueError("calib-s8-obs-add-sigma must be >= 0.")
    if float(args.calib_s8_obs_frac_sigma) < 0.0:
        raise ValueError("calib-s8-obs-frac-sigma must be >= 0.")
    if float(args.calib_s8_obs_sigma_frac_sigma) < 0.0:
        raise ValueError("calib-s8-obs-sigma-frac-sigma must be >= 0.")

    out_dir = Path(args.out).resolve()
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    per_run_rows: list[dict[str, Any]] = []
    s8_all: list[np.ndarray] = []

    for run in run_dirs:
        post = load_mu_forward_posterior(run)
        if post.sigma8_0 is None:
            raise ValueError(f"{run}: missing sigma8_0 in posterior; cannot run S8 test.")

        n_all = int(post.H0.size)
        idx = _select_indices(n_all, int(args.draws_per_run), rng)
        om = np.asarray(post.omega_m0[idx], dtype=float)
        s8 = np.asarray(post.sigma8_0[idx], dtype=float) * np.sqrt(np.clip(om / 0.3, 1e-12, np.inf))
        metrics = _compute_consistency_metrics(
            s8_draws=s8,
            s8_obs=float(args.s8_obs),
            s8_obs_sigma=float(args.s8_obs_sigma),
            rng=rng,
        )
        s8_all.append(s8)
        row = {
            "run_dir": str(Path(run).resolve()),
            "draws_total": int(n_all),
            "draws_used": int(idx.size),
            "metrics": metrics,
        }
        per_run_rows.append(row)

    s8_combined = np.concatenate(s8_all)
    combined = _compute_consistency_metrics(
        s8_draws=s8_combined,
        s8_obs=float(args.s8_obs),
        s8_obs_sigma=float(args.s8_obs_sigma),
        rng=rng,
    )

    checks = [
        {
            "name": "abs_z_below_threshold",
            "value": abs(float(combined["obs_consistency_z"])),
            "threshold": float(args.gate_max_abs_z),
            "pass": abs(float(combined["obs_consistency_z"])) <= float(args.gate_max_abs_z),
        },
        {
            "name": "obs_quantile_in_range",
            "value": float(combined["obs_quantile_in_predictive"]),
            "range": [float(q_lo), float(q_hi)],
            "pass": float(q_lo) <= float(combined["obs_quantile_in_predictive"]) <= float(q_hi),
        },
        {
            "name": "ppc_two_sided_above_min",
            "value": float(combined["ppc_two_sided_p"]),
            "threshold": float(args.gate_min_ppc_two_sided),
            "pass": float(combined["ppc_two_sided_p"]) >= float(args.gate_min_ppc_two_sided),
        },
    ]
    gate_pass = bool(all(bool(c["pass"]) for c in checks))

    calibration_robust_gate: dict[str, Any] | None = None
    if bool(args.calibration_robust):
        rng_cal = np.random.default_rng(int(args.calibration_seed))
        n_cal = int(args.calibration_samples)
        s8_obs_base = float(args.s8_obs)
        s8_sigma_base = float(args.s8_obs_sigma)
        s8_mean = float(combined["s8_stats"]["mean"])
        s8_sd = float(combined["s8_stats"]["sd"])
        s8_sd_eff = max(1e-12, s8_sd)

        d_add = float(args.calib_s8_obs_add_sigma) * rng_cal.normal(size=n_cal)
        d_frac = float(args.calib_s8_obs_frac_sigma) * rng_cal.normal(size=n_cal)
        sig_scale = np.clip(
            1.0 + float(args.calib_s8_obs_sigma_frac_sigma) * rng_cal.normal(size=n_cal),
            0.2,
            5.0,
        )

        s8_obs_eff = (s8_obs_base + d_add) * (1.0 + d_frac)
        s8_sigma_eff = np.clip(s8_sigma_base * sig_scale, 1e-8, np.inf)

        q_eff = np.mean(s8_combined.reshape((1, -1)) <= s8_obs_eff.reshape((-1, 1)), axis=1)
        z_eff = (s8_mean - s8_obs_eff) / np.sqrt((s8_sd_eff**2) + (s8_sigma_eff**2))
        p_left = _normal_cdf_approx((s8_obs_eff - s8_mean) / np.sqrt((s8_sd_eff**2) + (s8_sigma_eff**2)))
        ppc_eff = 2.0 * np.minimum(p_left, 1.0 - p_left)

        pass_abs_z = np.abs(z_eff) <= float(args.gate_max_abs_z)
        pass_quant = (q_eff >= float(q_lo)) & (q_eff <= float(q_hi))
        pass_ppc = ppc_eff >= float(args.gate_min_ppc_two_sided)
        pass_all = pass_abs_z & pass_quant & pass_ppc

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
                    "name": "abs_z_below_threshold",
                    "threshold": float(args.gate_max_abs_z),
                    "pass_fraction": float(np.mean(pass_abs_z)),
                },
                {
                    "name": "obs_quantile_in_range",
                    "range": [float(q_lo), float(q_hi)],
                    "pass_fraction": float(np.mean(pass_quant)),
                },
                {
                    "name": "ppc_two_sided_above_min",
                    "threshold": float(args.gate_min_ppc_two_sided),
                    "pass_fraction": float(np.mean(pass_ppc)),
                    "note": "Uses normal-CDF approximation under nuisance sampling.",
                },
            ],
            "nuisance_priors": {
                "s8_obs_additive_sigma": float(args.calib_s8_obs_add_sigma),
                "s8_obs_fractional_sigma": float(args.calib_s8_obs_frac_sigma),
                "s8_obs_sigma_scale_sigma": float(args.calib_s8_obs_sigma_frac_sigma),
            },
            "effective_anchor_stats": {
                "s8_obs_effective": _stats_1d(s8_obs_eff),
                "s8_obs_sigma_effective": _stats_1d(s8_sigma_eff),
                "obs_consistency_z_effective": _stats_1d(z_eff),
                "obs_quantile_effective": _stats_1d(q_eff),
                "ppc_two_sided_effective": _stats_1d(ppc_eff),
            },
        }

    summary: dict[str, Any] = {
        "created_utc": _utc_now(),
        "phase": "phase2_growth_s8",
        "inputs": {
            "run_dirs": [str(Path(r).resolve()) for r in run_dirs],
            "draws_per_run": int(args.draws_per_run),
            "seed": int(args.seed),
            "s8_obs": float(args.s8_obs),
            "s8_obs_sigma": float(args.s8_obs_sigma),
            "s8_planck_ref": float(args.s8_planck_ref),
        },
        "per_run": per_run_rows,
        "combined": {
            "draws_total": int(s8_combined.size),
            **combined,
        },
        "strict_gate": {
            "pass": gate_pass,
            "checks": checks,
        },
    }
    if calibration_robust_gate is not None:
        summary["calibration_robust_gate"] = calibration_robust_gate
    _write_json_atomic(tab_dir / "summary.json", summary)

    with (tab_dir / "per_run_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run_dir",
                "draws_total",
                "draws_used",
                "s8_mean",
                "s8_sd",
                "s8_p16",
                "s8_p50",
                "s8_p84",
                "obs_quantile_in_predictive",
                "obs_consistency_z",
                "ppc_two_sided_p",
            ]
        )
        for row in per_run_rows:
            m = row["metrics"]
            s = m["s8_stats"]
            w.writerow(
                [
                    row["run_dir"],
                    row["draws_total"],
                    row["draws_used"],
                    s["mean"],
                    s["sd"],
                    s["p16"],
                    s["p50"],
                    s["p84"],
                    m["obs_quantile_in_predictive"],
                    m["obs_consistency_z"],
                    m["ppc_two_sided_p"],
                ]
            )

    _plot_combined_hist(
        s8_combined=s8_combined,
        s8_obs=float(args.s8_obs),
        s8_obs_sigma=float(args.s8_obs_sigma),
        s8_planck_ref=float(args.s8_planck_ref),
        out_path=fig_dir / "s8_predictive_hist.png",
    )

    cstats = combined["s8_stats"]
    lines = [
        "# Forward Test Phase 2: Growth/S8",
        "",
        f"- Generated UTC: `{summary['created_utc']}`",
        f"- Runs: `{len(run_dirs)}`",
        f"- Combined draws: `{summary['combined']['draws_total']}`",
        "",
        "## Combined Predictive Summary",
        "",
        f"- `S8 mean={cstats['mean']:.4f}`",
        f"- `S8 p16/p50/p84={cstats['p16']:.4f}/{cstats['p50']:.4f}/{cstats['p84']:.4f}`",
        f"- Observed target: `{float(args.s8_obs):.4f} +/- {float(args.s8_obs_sigma):.4f}`",
        f"- Observed quantile in predictive: `{combined['obs_quantile_in_predictive']:.4f}`",
        f"- Consistency z-score: `{combined['obs_consistency_z']:.4f}`",
        f"- PPC two-sided p-value: `{combined['ppc_two_sided_p']:.4f}`",
        "",
        "## Strict Gate",
        "",
        f"- Result: `{'PASS' if gate_pass else 'FAIL'}`",
    ]
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
            lines.append(
                f"- `{c['name']}`: pass fraction `{float(c['pass_fraction']):.3f}`"
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `tables/summary.json`",
            "- `tables/per_run_summary.csv`",
            "- `figures/s8_predictive_hist.png`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {tab_dir / 'summary.json'}")
    print(f"[done] strict_gate={'PASS' if gate_pass else 'FAIL'}")

    if bool(args.fail_on_gate) and (not gate_pass):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
