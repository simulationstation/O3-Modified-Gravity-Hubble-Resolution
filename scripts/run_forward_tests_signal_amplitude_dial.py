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


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _normal_cdf_approx(x: np.ndarray | float) -> np.ndarray:
    xx = np.asarray(x, dtype=float)
    # Fast erf approximation (same style used in Phase-2 robust gate).
    t = np.sqrt(2.0 / np.pi) * (xx + 0.044715 * (xx**3))
    return 0.5 * (1.0 + np.tanh(t))


def _parse_alpha_grid(args: argparse.Namespace) -> np.ndarray:
    if str(args.alpha_grid).strip():
        vals = [float(t.strip()) for t in str(args.alpha_grid).split(",") if t.strip()]
        if not vals:
            raise ValueError("alpha-grid is empty.")
        grid = np.asarray(sorted(set(vals)), dtype=float)
    else:
        if float(args.alpha_step) <= 0.0:
            raise ValueError("alpha-step must be positive when alpha-grid is not provided.")
        n = int(np.floor((float(args.alpha_max) - float(args.alpha_min)) / float(args.alpha_step))) + 1
        grid = float(args.alpha_min) + float(args.alpha_step) * np.arange(max(1, n), dtype=float)
        grid = grid[grid <= float(args.alpha_max) + 1e-12]
    if np.any(grid < 0.0):
        raise ValueError("alpha values must be >= 0.")
    return grid


def _checks_by_name(summary: dict[str, Any], gate_key: str = "strict_gate") -> dict[str, dict[str, Any]]:
    gate = summary.get(gate_key, {})
    checks = gate.get("checks", [])
    out: dict[str, dict[str, Any]] = {}
    if isinstance(checks, list):
        for c in checks:
            if isinstance(c, dict) and isinstance(c.get("name"), str):
                out[str(c["name"])] = c
    return out


def _min_alpha(rows: list[dict[str, Any]], key: str) -> float | None:
    for r in rows:
        if bool(r.get(key, False)):
            return float(r["alpha"])
    return None


def _plot_pass_matrix(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    alpha = np.asarray([float(r["alpha"]) for r in rows], dtype=float)
    names = [
        "phase2_pass",
        "phase3_pass",
        "phase4_pass",
        "phase5_full_pass",
        "m2_supported",
        "m2_supported_with_material_relief_linear",
    ]
    y = np.zeros((len(names), alpha.size), dtype=float)
    for i, nm in enumerate(names):
        y[i, :] = np.asarray([1.0 if bool(r.get(nm, False)) else 0.0 for r in rows], dtype=float)

    plt.figure(figsize=(9.2, 4.8))
    for i, nm in enumerate(names):
        plt.step(alpha, y[i] + i, where="post", label=nm)
    plt.yticks(np.arange(len(names), dtype=float), names)
    plt.xlabel("Signal amplitude dial alpha")
    plt.ylabel("Pass state (offset rows)")
    plt.title("Forward Gates vs Signal-Amplitude Dial")
    plt.grid(alpha=0.25, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_metric_curves(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        return
    alpha = np.asarray([float(r["alpha"]) for r in rows], dtype=float)
    relief = np.asarray([float(r["relief_mean_linear"]) for r in rows], dtype=float)
    s8 = np.asarray([float(r["s8_mean_eff"]) for r in rows], dtype=float)
    highz = np.asarray([float(r["highz_median_deviation_eff"]) for r in rows], dtype=float)

    fig, ax = plt.subplots(1, 3, figsize=(12.0, 3.8))
    ax[0].plot(alpha, s8, linewidth=2.0)
    ax[0].set_title("S8 mean (effective)")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("S8")
    ax[0].grid(alpha=0.25, linestyle=":")

    ax[1].plot(alpha, highz, linewidth=2.0)
    ax[1].set_title("High-z R median - 1")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("Delta ratio")
    ax[1].grid(alpha=0.25, linestyle=":")

    ax[2].plot(alpha, relief, linewidth=2.0)
    ax[2].axhline(0.30, color="tab:red", linestyle="--", linewidth=1.2, label="material threshold")
    ax[2].set_title("Relief (linear-coupled)")
    ax[2].set_xlabel("alpha")
    ax[2].set_ylabel("Relief fraction")
    ax[2].grid(alpha=0.25, linestyle=":")
    ax[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Signal-amplitude dial sweep over existing forward-test summaries. "
            "This is a lightweight gate-sensitivity emulator, not a re-inference."
        )
    )
    ap.add_argument("--phase1-summary", default="outputs/forward_tests/phase1_closure/tables/summary.json")
    ap.add_argument("--phase2-summary", default="outputs/forward_tests/phase2_growth_s8/tables/summary.json")
    ap.add_argument("--phase3-summary", default="outputs/forward_tests/phase3_mu_sigma/tables/summary.json")
    ap.add_argument("--phase4-summary", default="outputs/forward_tests/phase4_distance_ratio/tables/summary.json")
    ap.add_argument("--phase5-summary", default="outputs/forward_tests/phase5_external_constraints/tables/summary.json")
    ap.add_argument("--phase6-summary", default="outputs/forward_tests/phase6_model_selection/tables/summary.json")
    ap.add_argument("--out", default="outputs/forward_tests/signal_amplitude_dial")

    ap.add_argument(
        "--alpha-grid",
        default="",
        help="Comma-separated alpha values. If set, overrides alpha-min/max/step.",
    )
    ap.add_argument("--alpha-min", type=float, default=0.0)
    ap.add_argument("--alpha-max", type=float, default=3.0)
    ap.add_argument("--alpha-step", type=float, default=0.01)
    ap.add_argument("--fail-on-no-solution", action="store_true")
    args = ap.parse_args()

    paths = [
        Path(args.phase1_summary).resolve(),
        Path(args.phase2_summary).resolve(),
        Path(args.phase3_summary).resolve(),
        Path(args.phase4_summary).resolve(),
        Path(args.phase5_summary).resolve(),
        Path(args.phase6_summary).resolve(),
    ]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    p1 = _read_json(paths[0])
    p2 = _read_json(paths[1])
    p3 = _read_json(paths[2])
    p4 = _read_json(paths[3])
    p5 = _read_json(paths[4])
    p6 = _read_json(paths[5])

    alpha_grid = _parse_alpha_grid(args)
    out_dir = Path(args.out).resolve()
    tab_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    p1_pass = bool(p1.get("strict_gate", {}).get("pass", False))

    p2_checks = _checks_by_name(p2, "strict_gate")
    c2_abs = p2_checks["abs_z_below_threshold"]
    c2_quant = p2_checks["obs_quantile_in_range"]
    c2_ppc = p2_checks["ppc_two_sided_above_min"]
    p2_abs_z_th = float(c2_abs["threshold"])
    p2_q_lo = float(c2_quant["range"][0])
    p2_q_hi = float(c2_quant["range"][1])
    p2_ppc_min = float(c2_ppc["threshold"])

    s8_mean = float(p2["combined"]["s8_stats"]["mean"])
    s8_sd = max(1e-12, float(p2["combined"]["s8_stats"]["sd"]))
    s8_obs = float(p2["combined"]["s8_obs"])
    s8_obs_sigma = float(p2["combined"]["s8_obs_sigma"])
    s8_planck_ref = float(p2["inputs"]["s8_planck_ref"])

    p3_checks = _checks_by_name(p3, "strict_gate")
    p3_l100_th = float(p3_checks["lensing_suppression_L100_present"]["threshold"])
    p3_l300_th = float(p3_checks["lensing_suppression_L300_present"]["threshold"])
    p3_chi2_th = float(p3_checks["refit_improves_chi2_materially"]["threshold"])
    p3_refit_frac_th = float(p3_checks["refit_beats_reference_fraction"]["threshold"])
    p3_s8_planck_th = float(p3_checks["s8_below_planck_reference"]["threshold"])
    p3_s8_z_th = float(p3_checks["s8_obs_z_not_extreme"]["threshold"])

    lens_l100 = float(p3["lensing_baseline"]["delta_frac_pct_q50_L100"])
    lens_l300 = float(p3["lensing_baseline"]["delta_frac_pct_q50_L300"])
    chi2_improvement = float(p3["lensing_refit"]["chi2_improvement_vs_baseline_p50"])
    refit_frac = float(p3["lensing_refit"]["p_refit_better_than_ref"])

    p4_checks = _checks_by_name(p4, "strict_gate")
    p4_dev_th = float(p4_checks["highz_median_deviation_min"]["threshold"])
    p4_coh_th = float(p4_checks["direction_coherence_fraction"]["threshold"])
    p4_require_outside = True
    if "gr_outside_68_at_highz" in p4_checks:
        # Check has no explicit threshold, but may be disabled in some runs.
        p4_require_outside = bool(p4_checks["gr_outside_68_at_highz"].get("pass", True)) or bool(
            p4_checks["gr_outside_68_at_highz"].get("value", True)
        )

    highz_dev = float(p4["combined"]["highz_median_deviation"])
    coherence = float(p4["combined"]["direction_coherence_fraction"])
    zmax_key = f"z{float(p4['inputs']['z_max']):.2f}"
    keyq = p4["combined"]["keypoint_quantiles"]
    if zmax_key not in keyq:
        zmax_key = sorted(keyq.keys())[-1]
    q16_highz = float(keyq[zmax_key]["q16"])
    q84_highz = float(keyq[zmax_key]["q84"])

    dep_names = {"phase3_dependency_pass", "phase4_dependency_pass"}
    p5_strict_checks = p5.get("strict_gate", {}).get("checks", [])
    p5_available_checks = p5.get("available_data_gate", {}).get("checks", [])

    p5_fixed_full = bool(
        all(bool(c.get("pass", False)) for c in p5_strict_checks if str(c.get("name")) not in dep_names)
    )
    p5_fixed_available = bool(
        all(bool(c.get("pass", False)) for c in p5_available_checks if str(c.get("name")) not in dep_names)
    )

    relief_mean_base = float(p6["proxy_metrics"]["h0_relief_anchor"]["mean"])
    relief_threshold = float(p6["proxy_metrics"]["h0_relief_anchor"]["material_threshold"])

    rows: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        a = float(alpha)

        # Phase 2 effective metrics under amplitude dial:
        s8_mean_eff = s8_planck_ref + a * (s8_mean - s8_planck_ref)
        s8_z_eff = (s8_mean_eff - s8_obs) / np.sqrt(s8_sd**2 + s8_obs_sigma**2 + 1e-12)
        s8_q_eff = float(_normal_cdf_approx((s8_obs - s8_mean_eff) / max(1e-12, s8_sd)))
        s8_ppc_left = float(_normal_cdf_approx((s8_obs - s8_mean_eff) / np.sqrt(s8_sd**2 + s8_obs_sigma**2 + 1e-12)))
        s8_ppc_eff = float(2.0 * min(s8_ppc_left, 1.0 - s8_ppc_left))

        p2_pass_abs = abs(s8_z_eff) <= p2_abs_z_th
        p2_pass_quant = (s8_q_eff >= p2_q_lo) and (s8_q_eff <= p2_q_hi)
        p2_pass_ppc = s8_ppc_eff >= p2_ppc_min
        p2_pass = bool(p2_pass_abs and p2_pass_quant and p2_pass_ppc)

        # Phase 3 effective metrics:
        lens_l100_eff = a * lens_l100
        lens_l300_eff = a * lens_l300
        chi2_improvement_eff = a * chi2_improvement
        refit_frac_eff = float(np.clip(0.5 + a * (refit_frac - 0.5), 0.0, 1.0))

        p3_pass_l100 = lens_l100_eff <= p3_l100_th
        p3_pass_l300 = lens_l300_eff <= p3_l300_th
        p3_pass_chi2 = chi2_improvement_eff >= p3_chi2_th
        p3_pass_refit = refit_frac_eff >= p3_refit_frac_th
        p3_pass_s8 = s8_mean_eff < p3_s8_planck_th
        p3_pass_s8z = abs(s8_z_eff) <= p3_s8_z_th
        p3_pass = bool(p3_pass_l100 and p3_pass_l300 and p3_pass_chi2 and p3_pass_refit and p3_pass_s8 and p3_pass_s8z)

        # Phase 4 effective metrics:
        highz_dev_eff = a * highz_dev
        coherence_eff = coherence if a > 0.0 else 0.0
        q16_eff = 1.0 + a * (q16_highz - 1.0)
        q84_eff = 1.0 + a * (q84_highz - 1.0)
        gr_outside_eff = bool((1.0 < q16_eff) or (1.0 > q84_eff))

        p4_pass_dev = abs(highz_dev_eff) >= p4_dev_th
        p4_pass_coh = coherence_eff >= p4_coh_th
        p4_pass_gr = gr_outside_eff if p4_require_outside else True
        p4_pass = bool(p4_pass_dev and p4_pass_coh and p4_pass_gr)

        # Phase 5 with fixed non-dependency checks + dialed dependencies:
        p5_full_pass = bool(p5_fixed_full and p3_pass and p4_pass)
        p5_available_pass = bool(p5_fixed_available and p3_pass and p4_pass)

        # Phase 6 decisions:
        m2_supported = bool(p1_pass and p2_pass and p3_pass and p4_pass and p5_full_pass)
        m2_supported_available = bool(p1_pass and p2_pass and p3_pass and p4_pass and p5_available_pass)

        # Relief modes:
        relief_mean_fixed = relief_mean_base
        relief_mean_linear = a * relief_mean_base
        m2_material_fixed = bool(m2_supported and (relief_mean_fixed >= relief_threshold))
        m2_material_linear = bool(m2_supported and (relief_mean_linear >= relief_threshold))

        rows.append(
            {
                "alpha": a,
                "s8_mean_eff": s8_mean_eff,
                "s8_obs_z_eff": s8_z_eff,
                "s8_obs_quantile_eff": s8_q_eff,
                "s8_ppc_two_sided_eff": s8_ppc_eff,
                "lens_L100_eff_pct": lens_l100_eff,
                "lens_L300_eff_pct": lens_l300_eff,
                "chi2_improvement_eff": chi2_improvement_eff,
                "refit_better_frac_eff": refit_frac_eff,
                "highz_median_deviation_eff": highz_dev_eff,
                "direction_coherence_eff": coherence_eff,
                "gr_outside_68_highz_eff": gr_outside_eff,
                "phase2_pass": p2_pass,
                "phase3_pass": p3_pass,
                "phase4_pass": p4_pass,
                "phase5_full_pass": p5_full_pass,
                "phase5_available_pass": p5_available_pass,
                "m2_supported": m2_supported,
                "m2_supported_available_data": m2_supported_available,
                "relief_mean_fixed": relief_mean_fixed,
                "relief_mean_linear": relief_mean_linear,
                "m2_supported_with_material_relief_fixed": m2_material_fixed,
                "m2_supported_with_material_relief_linear": m2_material_linear,
            }
        )

    alpha_min_phase2 = _min_alpha(rows, "phase2_pass")
    alpha_min_phase3 = _min_alpha(rows, "phase3_pass")
    alpha_min_phase4 = _min_alpha(rows, "phase4_pass")
    alpha_min_m2 = _min_alpha(rows, "m2_supported")
    alpha_min_m2_material_fixed = _min_alpha(rows, "m2_supported_with_material_relief_fixed")
    alpha_min_m2_material_linear = _min_alpha(rows, "m2_supported_with_material_relief_linear")

    summary = {
        "created_utc": _utc_now(),
        "mode": "signal_amplitude_dial_emulator",
        "inputs": {
            "phase1_summary": str(paths[0]),
            "phase2_summary": str(paths[1]),
            "phase3_summary": str(paths[2]),
            "phase4_summary": str(paths[3]),
            "phase5_summary": str(paths[4]),
            "phase6_summary": str(paths[5]),
            "alpha_grid": [float(x) for x in alpha_grid.tolist()],
        },
        "assumptions": {
            "phase2_s8_mean": "s8_eff = s8_planck + alpha * (s8_obsModel - s8_planck)",
            "phase3_lensing_and_fit": (
                "lensing suppressions and chi2 improvement scale linearly with alpha; "
                "refit-better fraction interpolates from 0.5 at alpha=0 to observed at alpha=1."
            ),
            "phase4_distance_ratio": "R-1 scales linearly with alpha at all z.",
            "phase5_dependencies": "Non-dependency checks fixed; phase3/phase4 dependency checks follow dialed passes.",
            "material_relief_modes": {
                "fixed": "relief mean fixed at current value.",
                "linear": "relief mean scales as alpha * current value.",
            },
            "note": "This is a gate-sensitivity emulator, not a full posterior re-inference at each alpha.",
        },
        "thresholds": {
            "material_relief_threshold": relief_threshold,
            "phase2_abs_z_threshold": p2_abs_z_th,
            "phase2_quantile_range": [p2_q_lo, p2_q_hi],
            "phase2_ppc_min": p2_ppc_min,
            "phase3_l100_threshold_pct": p3_l100_th,
            "phase3_l300_threshold_pct": p3_l300_th,
            "phase3_chi2_improvement_threshold": p3_chi2_th,
            "phase3_refit_fraction_threshold": p3_refit_frac_th,
            "phase4_highz_deviation_threshold": p4_dev_th,
            "phase4_direction_coherence_threshold": p4_coh_th,
        },
        "alpha_solutions": {
            "phase2_pass_min_alpha": alpha_min_phase2,
            "phase3_pass_min_alpha": alpha_min_phase3,
            "phase4_pass_min_alpha": alpha_min_phase4,
            "m2_supported_min_alpha": alpha_min_m2,
            "m2_material_relief_min_alpha_fixed_relief": alpha_min_m2_material_fixed,
            "m2_material_relief_min_alpha_linear_relief": alpha_min_m2_material_linear,
        },
        "baseline": {
            "phase1_pass": p1_pass,
            "phase5_non_dependency_full_fixed_pass": p5_fixed_full,
            "phase5_non_dependency_available_fixed_pass": p5_fixed_available,
            "relief_mean_base": relief_mean_base,
        },
    }
    _write_json_atomic(tab_dir / "summary.json", summary)

    with (tab_dir / "alpha_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "alpha",
            "s8_mean_eff",
            "s8_obs_z_eff",
            "s8_obs_quantile_eff",
            "s8_ppc_two_sided_eff",
            "lens_L100_eff_pct",
            "lens_L300_eff_pct",
            "chi2_improvement_eff",
            "refit_better_frac_eff",
            "highz_median_deviation_eff",
            "direction_coherence_eff",
            "gr_outside_68_highz_eff",
            "phase2_pass",
            "phase3_pass",
            "phase4_pass",
            "phase5_full_pass",
            "phase5_available_pass",
            "m2_supported",
            "m2_supported_available_data",
            "relief_mean_fixed",
            "relief_mean_linear",
            "m2_supported_with_material_relief_fixed",
            "m2_supported_with_material_relief_linear",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    _plot_pass_matrix(rows, fig_dir / "gate_pass_matrix.png")
    _plot_metric_curves(rows, fig_dir / "dial_metrics.png")

    lines = [
        "# Signal Amplitude Dial Sweep",
        "",
        f"- Generated UTC: `{summary['created_utc']}`",
        f"- Alpha range: `{alpha_grid[0]:.3f}` to `{alpha_grid[-1]:.3f}` with `{alpha_grid.size}` samples",
        "",
        "## Minimum Alpha Solutions",
        "",
        f"- Phase2 pass min alpha: `{alpha_min_phase2}`",
        f"- Phase3 pass min alpha: `{alpha_min_phase3}`",
        f"- Phase4 pass min alpha: `{alpha_min_phase4}`",
        f"- M2 support min alpha: `{alpha_min_m2}`",
        f"- M2 + material relief min alpha (fixed relief): `{alpha_min_m2_material_fixed}`",
        f"- M2 + material relief min alpha (linear relief coupling): `{alpha_min_m2_material_linear}`",
        "",
        "## Notes",
        "",
        "- This sweep uses lightweight scaling assumptions to map gate sensitivity quickly.",
        "- Use this as a targeting guide for full re-inference runs.",
        "",
        "## Artifacts",
        "",
        "- `tables/summary.json`",
        "- `tables/alpha_sweep.csv`",
        "- `figures/gate_pass_matrix.png`",
        "- `figures/dial_metrics.png`",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {tab_dir / 'summary.json'}")
    print(f"[done] m2_supported_min_alpha={alpha_min_m2}")
    print(f"[done] m2_material_relief_min_alpha_linear={alpha_min_m2_material_linear}")

    if bool(args.fail_on_no_solution):
        no_solution = alpha_min_m2 is None
        if no_solution:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
