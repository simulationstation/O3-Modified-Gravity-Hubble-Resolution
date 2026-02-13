#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _stats_1d(x: np.ndarray) -> dict[str, float]:
    xx = np.asarray(x, dtype=float)
    if xx.size <= 0:
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
        "mean": float(np.mean(xx)),
        "sd": float(np.std(xx, ddof=1)) if xx.size > 1 else 0.0,
        "p16": float(np.percentile(xx, 16.0)),
        "p50": float(np.percentile(xx, 50.0)),
        "p84": float(np.percentile(xx, 84.0)),
        "min": float(np.min(xx)),
        "max": float(np.max(xx)),
    }


def _plot_chi2_bar(
    *,
    chi2_ref: float,
    chi2_baseline_p50: float,
    chi2_refit_p50: float,
    out_path: Path,
) -> None:
    labels = ["Planck ref", "Baseline MG", "MG refit"]
    vals = [float(chi2_ref), float(chi2_baseline_p50), float(chi2_refit_p50)]
    colors = ["#4d4d4d", "#cc4c4c", "#3a7f3a"]

    plt.figure(figsize=(6.8, 4.4))
    plt.bar(labels, vals, color=colors)
    plt.ylabel(r"$\chi^2$")
    plt.title("Phase 3 Proxy: Lensing Fit Quality")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_suppression_bar(
    *,
    lensing_l100_pct: float,
    lensing_l300_pct: float,
    s8_delta_pct: float,
    out_path: Path,
) -> None:
    labels = [r"CLpp L~100", r"CLpp L~300", "S8 vs Planck"]
    vals = [float(lensing_l100_pct), float(lensing_l300_pct), float(s8_delta_pct)]
    colors = ["#cc4c4c", "#cc4c4c", "#3366cc"]

    plt.figure(figsize=(7.2, 4.4))
    plt.axhline(0.0, color="k", linewidth=1.0, alpha=0.6)
    plt.bar(labels, vals, color=colors)
    plt.ylabel("Percent shift")
    plt.title("Phase 3 Proxy: Suppression Direction Consistency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Concrete forward test (Phase 3 proxy): "
            "cross-channel mu-Sigma consistency audit using existing baseline lensing, "
            "MG-refit lensing, and Phase-2 growth/S8 outputs."
        )
    )
    ap.add_argument(
        "--cmb-baseline-summary",
        default="outputs/hubble_tension_cmb_forecast_camb64_20260210_analyticrefresh/tables/summary.json",
    )
    ap.add_argument(
        "--mg-refit-summary",
        default="outputs/hubble_tension_mg_lensing_refit_camb32_20260210_live/tables/summary.json",
    )
    ap.add_argument(
        "--growth-summary",
        default="outputs/forward_tests/phase2_growth_s8/tables/summary.json",
        help="Phase-2 growth summary JSON.",
    )
    ap.add_argument("--out", default="outputs/forward_tests/phase3_mu_sigma")

    ap.add_argument("--s8-planck-ref", type=float, default=0.832)
    ap.add_argument("--gate-max-lensing-l100-pct", type=float, default=-5.0)
    ap.add_argument("--gate-max-lensing-l300-pct", type=float, default=-3.0)
    ap.add_argument("--gate-min-chi2-improvement", type=float, default=20.0)
    ap.add_argument("--gate-min-refit-better-frac", type=float, default=0.8)
    ap.add_argument("--gate-require-s8-below-planck", action="store_true", default=True)
    ap.add_argument("--gate-max-abs-s8-z", type=float, default=2.5)
    ap.add_argument(
        "--calibration-robust",
        action="store_true",
        help="Enable calibration-marginalized robustness gate for cross-channel summary metrics.",
    )
    ap.add_argument(
        "--calibration-samples",
        type=int,
        default=4000,
        help="Number of nuisance draws for calibration-robust gate.",
    )
    ap.add_argument("--calibration-seed", type=int, default=29303, help="RNG seed for calibration robustness sampling.")
    ap.add_argument(
        "--gate-min-robust-pass-frac",
        type=float,
        default=0.60,
        help="Calibration-robust gate passes if nuisance-marginal pass fraction >= this value.",
    )
    ap.add_argument(
        "--calib-lensing-shift-pct-sigma",
        type=float,
        default=3.0,
        help="Stddev for additive lensing suppression calibration shift (percent units).",
    )
    ap.add_argument(
        "--calib-s8-planck-ref-add-sigma",
        type=float,
        default=0.015,
        help="Stddev for additive calibration shift in S8 Planck reference anchor.",
    )
    ap.add_argument(
        "--calib-refit-better-frac-sigma",
        type=float,
        default=0.03,
        help="Stddev for calibration uncertainty on p_refit_better_than_ref.",
    )
    ap.add_argument(
        "--calib-chi2-improvement-frac-sigma",
        type=float,
        default=0.10,
        help="Stddev for fractional calibration uncertainty on chi2 improvement.",
    )
    ap.add_argument(
        "--calib-s8-obs-z-add-sigma",
        type=float,
        default=0.35,
        help="Stddev for additive calibration uncertainty on S8 observed-consistency z metric.",
    )
    ap.add_argument("--fail-on-gate", action="store_true")
    args = ap.parse_args()

    if not (0.0 <= float(args.gate_min_robust_pass_frac) <= 1.0):
        raise ValueError("gate-min-robust-pass-frac must be in [0,1].")
    if int(args.calibration_samples) < 100:
        raise ValueError("calibration-samples must be >= 100.")
    if float(args.calib_lensing_shift_pct_sigma) < 0.0:
        raise ValueError("calib-lensing-shift-pct-sigma must be >= 0.")
    if float(args.calib_s8_planck_ref_add_sigma) < 0.0:
        raise ValueError("calib-s8-planck-ref-add-sigma must be >= 0.")
    if float(args.calib_refit_better_frac_sigma) < 0.0:
        raise ValueError("calib-refit-better-frac-sigma must be >= 0.")
    if float(args.calib_chi2_improvement_frac_sigma) < 0.0:
        raise ValueError("calib-chi2-improvement-frac-sigma must be >= 0.")
    if float(args.calib_s8_obs_z_add_sigma) < 0.0:
        raise ValueError("calib-s8-obs-z-add-sigma must be >= 0.")

    cmb_path = Path(args.cmb_baseline_summary).resolve()
    refit_path = Path(args.mg_refit_summary).resolve()
    growth_path = Path(args.growth_summary).resolve()
    for p in (cmb_path, refit_path, growth_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    cmb = _read_json(cmb_path)
    refit = _read_json(refit_path)
    growth = _read_json(growth_path)

    out_dir = Path(args.out).resolve()
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    lens_l100 = float(cmb["headline_multipoles"]["near_L100"]["delta_frac_pct_q50"])
    lens_l300 = float(cmb["headline_multipoles"]["near_L300"]["delta_frac_pct_q50"])
    chi2_baseline_p50 = float(cmb["lensing_chi2"]["chi2_draws"]["p50"])
    chi2_ref = float(cmb["lensing_chi2"]["chi2_planck_ref_model"])

    chi2_refit_p50 = float(refit["chi2"]["chi2_mg_refit_draws"]["p50"])
    refit_better_frac = float(refit["chi2"]["p_refit_better_than_ref"])
    mstar2_ratio_p50 = float(refit["fit_parameter_stats"]["mstar2_ratio_0"]["p50"])
    mstar2_drop_p50 = float(refit["fit_parameter_stats"]["mstar2_drop_pct"]["p50"])

    s8_mean = float(growth["combined"]["s8_stats"]["mean"])
    s8_sd = float(growth["combined"]["s8_stats"]["sd"])
    s8_obs = float(growth["combined"]["s8_obs"])
    s8_obs_sigma = float(growth["combined"]["s8_obs_sigma"])
    s8_ppc = float(growth["combined"]["ppc_two_sided_p"])
    s8_obs_z = float(growth["combined"]["obs_consistency_z"])

    s8_delta_vs_planck = float(s8_mean - float(args.s8_planck_ref))
    s8_delta_vs_planck_pct = float(100.0 * s8_delta_vs_planck / float(args.s8_planck_ref))
    chi2_improvement = float(chi2_baseline_p50 - chi2_refit_p50)

    checks = [
        {
            "name": "lensing_suppression_L100_present",
            "value": lens_l100,
            "threshold": float(args.gate_max_lensing_l100_pct),
            "pass": lens_l100 <= float(args.gate_max_lensing_l100_pct),
        },
        {
            "name": "lensing_suppression_L300_present",
            "value": lens_l300,
            "threshold": float(args.gate_max_lensing_l300_pct),
            "pass": lens_l300 <= float(args.gate_max_lensing_l300_pct),
        },
        {
            "name": "refit_improves_chi2_materially",
            "value": chi2_improvement,
            "threshold": float(args.gate_min_chi2_improvement),
            "pass": chi2_improvement >= float(args.gate_min_chi2_improvement),
        },
        {
            "name": "refit_beats_reference_fraction",
            "value": refit_better_frac,
            "threshold": float(args.gate_min_refit_better_frac),
            "pass": refit_better_frac >= float(args.gate_min_refit_better_frac),
        },
        {
            "name": "s8_below_planck_reference",
            "value": s8_mean,
            "threshold": float(args.s8_planck_ref),
            "pass": (s8_mean < float(args.s8_planck_ref)) if bool(args.gate_require_s8_below_planck) else True,
        },
        {
            "name": "s8_obs_z_not_extreme",
            "value": abs(s8_obs_z),
            "threshold": float(args.gate_max_abs_s8_z),
            "pass": abs(s8_obs_z) <= float(args.gate_max_abs_s8_z),
        },
    ]
    strict_pass = bool(all(bool(c["pass"]) for c in checks))

    calibration_robust_gate: dict[str, Any] | None = None
    if bool(args.calibration_robust):
        rng_cal = np.random.default_rng(int(args.calibration_seed))
        n_cal = int(args.calibration_samples)

        lens_shift = float(args.calib_lensing_shift_pct_sigma) * rng_cal.normal(size=n_cal)
        l100_eff = lens_l100 + lens_shift
        l300_eff = lens_l300 + lens_shift
        chi2_scale = np.clip(
            1.0 + float(args.calib_chi2_improvement_frac_sigma) * rng_cal.normal(size=n_cal),
            0.05,
            5.0,
        )
        chi2_improvement_eff = chi2_improvement * chi2_scale
        refit_frac_eff = np.clip(
            refit_better_frac + float(args.calib_refit_better_frac_sigma) * rng_cal.normal(size=n_cal),
            0.0,
            1.0,
        )
        s8_planck_eff = float(args.s8_planck_ref) + float(args.calib_s8_planck_ref_add_sigma) * rng_cal.normal(size=n_cal)
        s8_obs_z_eff = s8_obs_z + float(args.calib_s8_obs_z_add_sigma) * rng_cal.normal(size=n_cal)

        pass_l100 = l100_eff <= float(args.gate_max_lensing_l100_pct)
        pass_l300 = l300_eff <= float(args.gate_max_lensing_l300_pct)
        pass_chi2 = chi2_improvement_eff >= float(args.gate_min_chi2_improvement)
        pass_refit_frac = refit_frac_eff >= float(args.gate_min_refit_better_frac)
        pass_s8_below = (s8_mean < s8_planck_eff) if bool(args.gate_require_s8_below_planck) else np.ones((n_cal,), dtype=bool)
        pass_s8_obs_z = np.abs(s8_obs_z_eff) <= float(args.gate_max_abs_s8_z)

        pass_all = pass_l100 & pass_l300 & pass_chi2 & pass_refit_frac & pass_s8_below & pass_s8_obs_z
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
                    "name": "lensing_suppression_L100_present",
                    "threshold": float(args.gate_max_lensing_l100_pct),
                    "pass_fraction": float(np.mean(pass_l100)),
                },
                {
                    "name": "lensing_suppression_L300_present",
                    "threshold": float(args.gate_max_lensing_l300_pct),
                    "pass_fraction": float(np.mean(pass_l300)),
                },
                {
                    "name": "refit_improves_chi2_materially",
                    "threshold": float(args.gate_min_chi2_improvement),
                    "pass_fraction": float(np.mean(pass_chi2)),
                },
                {
                    "name": "refit_beats_reference_fraction",
                    "threshold": float(args.gate_min_refit_better_frac),
                    "pass_fraction": float(np.mean(pass_refit_frac)),
                },
                {
                    "name": "s8_below_planck_reference",
                    "enabled": bool(args.gate_require_s8_below_planck),
                    "threshold": float(args.s8_planck_ref),
                    "pass_fraction": float(np.mean(pass_s8_below)),
                },
                {
                    "name": "s8_obs_z_not_extreme",
                    "threshold": float(args.gate_max_abs_s8_z),
                    "pass_fraction": float(np.mean(pass_s8_obs_z)),
                },
            ],
            "nuisance_priors": {
                "lensing_shift_pct_sigma": float(args.calib_lensing_shift_pct_sigma),
                "s8_planck_ref_add_sigma": float(args.calib_s8_planck_ref_add_sigma),
                "refit_better_frac_sigma": float(args.calib_refit_better_frac_sigma),
                "chi2_improvement_frac_sigma": float(args.calib_chi2_improvement_frac_sigma),
                "s8_obs_z_add_sigma": float(args.calib_s8_obs_z_add_sigma),
            },
            "effective_metric_stats": {
                "l100_effective_pct": _stats_1d(l100_eff),
                "l300_effective_pct": _stats_1d(l300_eff),
                "chi2_improvement_effective": _stats_1d(chi2_improvement_eff),
                "refit_better_frac_effective": _stats_1d(refit_frac_eff),
                "s8_planck_ref_effective": _stats_1d(s8_planck_eff),
                "s8_obs_z_effective": _stats_1d(s8_obs_z_eff),
            },
        }

    _plot_chi2_bar(
        chi2_ref=chi2_ref,
        chi2_baseline_p50=chi2_baseline_p50,
        chi2_refit_p50=chi2_refit_p50,
        out_path=fig_dir / "chi2_comparison.png",
    )
    _plot_suppression_bar(
        lensing_l100_pct=lens_l100,
        lensing_l300_pct=lens_l300,
        s8_delta_pct=s8_delta_vs_planck_pct,
        out_path=fig_dir / "suppression_direction.png",
    )

    summary = {
        "created_utc": _utc_now(),
        "phase": "phase3_mu_sigma",
        "mode": "proxy_consistency_audit",
        "inputs": {
            "cmb_baseline_summary": str(cmb_path),
            "mg_refit_summary": str(refit_path),
            "growth_summary": str(growth_path),
            "s8_planck_ref": float(args.s8_planck_ref),
        },
        "lensing_baseline": {
            "delta_frac_pct_q50_L100": lens_l100,
            "delta_frac_pct_q50_L300": lens_l300,
            "chi2_p50": chi2_baseline_p50,
            "chi2_planck_ref_model": chi2_ref,
        },
        "lensing_refit": {
            "chi2_p50": chi2_refit_p50,
            "p_refit_better_than_ref": refit_better_frac,
            "chi2_improvement_vs_baseline_p50": chi2_improvement,
            "mstar2_ratio_0_p50": mstar2_ratio_p50,
            "mstar2_drop_pct_p50": mstar2_drop_p50,
        },
        "growth_phase2": {
            "s8_mean": s8_mean,
            "s8_sd": s8_sd,
            "s8_obs": s8_obs,
            "s8_obs_sigma": s8_obs_sigma,
            "s8_obs_consistency_z": s8_obs_z,
            "s8_ppc_two_sided_p": s8_ppc,
            "s8_delta_vs_planck": s8_delta_vs_planck,
            "s8_delta_vs_planck_pct": s8_delta_vs_planck_pct,
        },
        "strict_gate": {
            "pass": strict_pass,
            "checks": checks,
        },
        "interpretation_note": (
            "This phase is a proxy consistency audit across existing outputs. "
            "It is not yet a fully tied mu-Sigma Boltzmann-sector joint fit."
        ),
    }
    if calibration_robust_gate is not None:
        summary["calibration_robust_gate"] = calibration_robust_gate
    _write_json_atomic(tab_dir / "summary.json", summary)

    lines = [
        "# Forward Test Phase 3: mu-Sigma Proxy Consistency Audit",
        "",
        f"- Generated UTC: `{summary['created_utc']}`",
        "",
        "## Key Metrics",
        "",
        f"- Baseline CLpp suppression: `L~100 {lens_l100:+.2f}%`, `L~300 {lens_l300:+.2f}%`",
        f"- Lensing chi2 median: baseline `{chi2_baseline_p50:.3f}` -> refit `{chi2_refit_p50:.3f}`",
        f"- Refit beats Planck-ref fraction: `{100.0 * refit_better_frac:.1f}%`",
        f"- Growth S8 mean: `{s8_mean:.4f}` (Planck ref `{float(args.s8_planck_ref):.4f}`; delta `{s8_delta_vs_planck:+.4f}`)",
        f"- Growth observed-consistency z: `{s8_obs_z:.3f}`",
        "",
        "## Strict Gate",
        "",
        f"- Result: `{'PASS' if strict_pass else 'FAIL'}`",
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
            lines.append(f"- `{c['name']}`: pass fraction `{float(c['pass_fraction']):.3f}`")
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "- This is a proxy cross-channel consistency check. A full tied mu-Sigma/EFT fit remains Phase-3 target work.",
            "",
            "## Artifacts",
            "",
            "- `tables/summary.json`",
            "- `figures/chi2_comparison.png`",
            "- `figures/suppression_direction.png`",
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
