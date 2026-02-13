#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def _status(flag: bool, yes: str, no: str) -> str:
    return yes if bool(flag) else no


def _failing_gate_checks(phase_summary: dict[str, Any], gate_key: str) -> list[str]:
    gate = phase_summary.get(gate_key, {})
    checks = gate.get("checks", [])
    if not isinstance(checks, list):
        return []
    out: list[str] = []
    for c in checks:
        if not isinstance(c, dict):
            continue
        req = bool(c.get("gate_required", True))
        if req and (not bool(c.get("pass", False))):
            out.append(str(c.get("name")))
    return out


def _extract_pass_probability(
    phase_summary: dict[str, Any],
    *,
    gate_key: str,
    robust_fraction_key: str | None = None,
) -> float:
    rg = phase_summary.get("calibration_robust_gate", {})
    if isinstance(rg, dict):
        if robust_fraction_key:
            v = rg.get(str(robust_fraction_key))
            if isinstance(v, (int, float)) and np.isfinite(float(v)):
                return float(np.clip(float(v), 0.0, 1.0))
        v2 = rg.get("pass_fraction")
        if isinstance(v2, (int, float)) and np.isfinite(float(v2)):
            return float(np.clip(float(v2), 0.0, 1.0))
    gate = phase_summary.get(str(gate_key), {})
    return 1.0 if bool(gate.get("pass", False)) else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Concrete forward test (Phase 6 proxy): aggregate scorecard for "
            "M0..M3 using existing phase outputs without heavy re-inference."
        )
    )
    ap.add_argument(
        "--phase1-summary",
        default="outputs/forward_tests/phase1_closure/tables/summary.json",
    )
    ap.add_argument(
        "--phase2-summary",
        default="outputs/forward_tests/phase2_growth_s8/tables/summary.json",
    )
    ap.add_argument(
        "--phase3-summary",
        default="outputs/forward_tests/phase3_mu_sigma/tables/summary.json",
    )
    ap.add_argument(
        "--phase4-summary",
        default="outputs/forward_tests/phase4_distance_ratio/tables/summary.json",
    )
    ap.add_argument(
        "--phase5-summary",
        default="outputs/forward_tests/phase5_external_constraints/tables/summary.json",
    )
    ap.add_argument(
        "--relief-summary",
        default="outputs/recalibration_planckref_20260210/final_relief_rebased/final_relief_posterior_summary.json",
    )
    ap.add_argument(
        "--cmb-summary",
        default="outputs/hubble_tension_cmb_forecast_camb64_20260210_analyticrefresh/tables/summary.json",
    )
    ap.add_argument(
        "--mg-refit-summary",
        default="outputs/hubble_tension_mg_lensing_refit_camb32_20260210_live/tables/summary.json",
    )
    ap.add_argument("--out", default="outputs/forward_tests/phase6_model_selection")

    ap.add_argument("--gate-min-chi2-improvement-m3", type=float, default=20.0)
    ap.add_argument("--gate-min-relief-for-material", type=float, default=0.30)
    ap.add_argument(
        "--calibration-robust",
        action="store_true",
        help="Enable calibration-marginalized M2 stability score over phase-gate and anchor-fit nuisances.",
    )
    ap.add_argument(
        "--calibration-samples",
        type=int,
        default=20000,
        help="Number of nuisance draws for calibration-robust model-stability score.",
    )
    ap.add_argument("--calibration-seed", type=int, default=29606, help="RNG seed for calibration robustness sampling.")
    ap.add_argument(
        "--gate-min-bias-stability",
        type=float,
        default=0.60,
        help="Calibration-robust gate passes if M2 stability score >= this value.",
    )
    ap.add_argument(
        "--calib-relief-sigma",
        type=float,
        default=-1.0,
        help="If >=0, override relief calibration sigma. If <0, use relief summary sd.",
    )
    ap.add_argument(
        "--calib-chi2-improvement-frac-sigma",
        type=float,
        default=0.10,
        help="Stddev for fractional calibration uncertainty on chi2 improvement in M3 support checks.",
    )
    ap.add_argument("--fail-on-gate", action="store_true")
    args = ap.parse_args()

    if not (0.0 <= float(args.gate_min_bias_stability) <= 1.0):
        raise ValueError("gate-min-bias-stability must be in [0,1].")
    if int(args.calibration_samples) < 100:
        raise ValueError("calibration-samples must be >= 100.")
    if float(args.calib_chi2_improvement_frac_sigma) < 0.0:
        raise ValueError("calib-chi2-improvement-frac-sigma must be >= 0.")

    p1_path = Path(args.phase1_summary).resolve()
    p2_path = Path(args.phase2_summary).resolve()
    p3_path = Path(args.phase3_summary).resolve()
    p4_path = Path(args.phase4_summary).resolve()
    p5_path = Path(args.phase5_summary).resolve()
    rel_path = Path(args.relief_summary).resolve()
    cmb_path = Path(args.cmb_summary).resolve()
    refit_path = Path(args.mg_refit_summary).resolve()
    for p in (p1_path, p2_path, p3_path, p4_path, p5_path, rel_path, cmb_path, refit_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    p1 = _read_json(p1_path)
    p2 = _read_json(p2_path)
    p3 = _read_json(p3_path)
    p4 = _read_json(p4_path)
    p5 = _read_json(p5_path)
    rel = _read_json(rel_path)
    cmb = _read_json(cmb_path)
    refit = _read_json(refit_path)

    p1_pass = bool(p1.get("strict_gate", {}).get("pass", False))
    p2_pass = bool(p2.get("strict_gate", {}).get("pass", False))
    p3_pass = bool(p3.get("strict_gate", {}).get("pass", False))
    p4_pass = bool(p4.get("strict_gate", {}).get("pass", False))
    p5_pass_full = bool(p5.get("strict_gate", {}).get("pass", False))
    p5_pass_available = bool(p5.get("available_data_gate", {}).get("pass", p5_pass_full))

    relief_mean = float(rel["posterior_with_mc_calibration"]["mean"])
    relief_p50 = float(rel["posterior_with_mc_calibration"]["p50"])
    relief_material = bool(relief_mean >= float(args.gate_min_relief_for_material))

    chi2_baseline = float(cmb["lensing_chi2"]["chi2_draws"]["p50"])
    chi2_refit = float(refit["chi2"]["chi2_mg_refit_draws"]["p50"])
    chi2_ref_model = float(cmb["lensing_chi2"]["chi2_planck_ref_model"])
    chi2_improvement = float(chi2_baseline - chi2_refit)
    m3_material_fit_gain = bool(chi2_improvement >= float(args.gate_min_chi2_improvement_m3))

    # Proxy evidence statements (not a Bayesian evidence computation).
    m0_supported = bool((not p4_pass) and (not p2_pass))
    m1_supported = bool(p4_pass and (not p5_pass_full))
    m2_supported = bool(p1_pass and p2_pass and p3_pass and p4_pass and p5_pass_full)
    m2_supported_available = bool(p1_pass and p2_pass and p3_pass and p4_pass and p5_pass_available)
    m3_supported = bool(p3_pass and m3_material_fit_gain)

    if m2_supported:
        leading = "M2"
    elif m2_supported_available:
        leading = "M2_provisional"
    elif m3_supported:
        leading = "M3"
    elif m1_supported:
        leading = "M1"
    else:
        leading = "M0"

    failing_p5_checks = _failing_gate_checks(p5, "strict_gate")
    failing_p5_available_checks = _failing_gate_checks(p5, "available_data_gate")

    scorecard = {
        "M0": {
            "label": "GR/LCDM baseline",
            "proxy_status": _status(m0_supported, "supported_by_current_proxy_checks", "disfavored_by_current_proxy_checks"),
            "key_reasons": [
                f"phase4_distance_ratio_strict_pass={p4_pass}",
                f"phase2_growth_strict_pass={p2_pass}",
            ],
        },
        "M1": {
            "label": "GW-only propagation anomaly (untied scalar sector)",
            "proxy_status": _status(m1_supported, "partially_supported_proxy", "not_preferred_proxy"),
            "key_reasons": [
                f"phase4_distance_ratio_strict_pass={p4_pass}",
                f"phase5_external_constraints_strict_pass={p5_pass_full}",
            ],
        },
        "M2": {
            "label": "Universal tied MG model (target)",
            "proxy_status": _status(m2_supported, "supported_proxy", "not_supported_proxy"),
            "proxy_status_available_data_gate": _status(
                m2_supported_available,
                "supported_proxy_with_available_data",
                "not_supported_proxy_with_available_data",
            ),
            "key_reasons": [
                f"phase1_closure_strict_pass={p1_pass}",
                f"phase2_growth_strict_pass={p2_pass}",
                f"phase3_mu_sigma_strict_pass={p3_pass}",
                f"phase4_distance_ratio_strict_pass={p4_pass}",
                f"phase5_external_constraints_strict_pass={p5_pass_full}",
                f"phase5_external_constraints_available_data_pass={p5_pass_available}",
            ],
        },
        "M3": {
            "label": "Untied phenomenological multi-channel model",
            "proxy_status": _status(m3_supported, "best_current_proxy_fit", "not_preferred_proxy"),
            "key_reasons": [
                f"phase3_mu_sigma_strict_pass={p3_pass}",
                f"chi2_improvement_baseline_to_refit={chi2_improvement:.3f}",
            ],
        },
    }

    calibration_robust_decision: dict[str, Any] | None = None
    if bool(args.calibration_robust):
        rng_cal = np.random.default_rng(int(args.calibration_seed))
        n_cal = int(args.calibration_samples)

        p1_prob = 1.0 if p1_pass else 0.0
        p2_prob = _extract_pass_probability(p2, gate_key="strict_gate")
        p3_prob = _extract_pass_probability(p3, gate_key="strict_gate")
        p4_prob = _extract_pass_probability(p4, gate_key="strict_gate")
        p5_full_prob = _extract_pass_probability(p5, gate_key="strict_gate", robust_fraction_key="full_pass_fraction")
        p5_available_prob = _extract_pass_probability(
            p5,
            gate_key="available_data_gate",
            robust_fraction_key="available_data_pass_fraction",
        )

        p1_draw = rng_cal.random(size=n_cal) < float(np.clip(p1_prob, 0.0, 1.0))
        p2_draw = rng_cal.random(size=n_cal) < float(np.clip(p2_prob, 0.0, 1.0))
        p3_draw = rng_cal.random(size=n_cal) < float(np.clip(p3_prob, 0.0, 1.0))
        p4_draw = rng_cal.random(size=n_cal) < float(np.clip(p4_prob, 0.0, 1.0))
        p5_full_draw = rng_cal.random(size=n_cal) < float(np.clip(p5_full_prob, 0.0, 1.0))
        p5_avail_draw = rng_cal.random(size=n_cal) < float(np.clip(p5_available_prob, 0.0, 1.0))

        relief_sd_summary = float(rel["posterior_with_mc_calibration"].get("sd", 0.0))
        if float(args.calib_relief_sigma) >= 0.0:
            relief_sigma = float(args.calib_relief_sigma)
        else:
            relief_sigma = max(1e-8, abs(relief_sd_summary))
        relief_draw = relief_mean + relief_sigma * rng_cal.normal(size=n_cal)
        relief_material_draw = relief_draw >= float(args.gate_min_relief_for_material)

        chi2_scale = np.clip(
            1.0 + float(args.calib_chi2_improvement_frac_sigma) * rng_cal.normal(size=n_cal),
            0.05,
            5.0,
        )
        chi2_improvement_draw = chi2_improvement * chi2_scale
        m3_material_draw = chi2_improvement_draw >= float(args.gate_min_chi2_improvement_m3)

        m2_draw = p1_draw & p2_draw & p3_draw & p4_draw & p5_full_draw
        m2_available_draw = p1_draw & p2_draw & p3_draw & p4_draw & p5_avail_draw
        m3_draw = p3_draw & m3_material_draw
        m1_draw = p4_draw & (~p5_full_draw)
        m0_draw = (~p4_draw) & (~p2_draw)

        leading_m2 = m2_draw
        leading_m2p = (~leading_m2) & m2_available_draw
        leading_m3 = (~leading_m2) & (~leading_m2p) & m3_draw
        leading_m1 = (~leading_m2) & (~leading_m2p) & (~leading_m3) & m1_draw
        leading_m0 = (~leading_m2) & (~leading_m2p) & (~leading_m3) & (~leading_m1)

        leading_fracs = {
            "M2": float(np.mean(leading_m2)),
            "M2_provisional": float(np.mean(leading_m2p)),
            "M3": float(np.mean(leading_m3)),
            "M1": float(np.mean(leading_m1)),
            "M0": float(np.mean(leading_m0)),
        }
        leading_stability = max(leading_fracs.items(), key=lambda kv: kv[1])[0]

        m2_stability = float(np.mean(m2_draw))
        m2_stability_available = float(np.mean(m2_available_draw))
        m2_material_stability = float(np.mean(m2_draw & relief_material_draw))
        robust_pass = bool(m2_stability >= float(args.gate_min_bias_stability))

        calibration_robust_decision = {
            "enabled": True,
            "mode": "calibration_marginalized_proxy_scorecard",
            "pass": robust_pass,
            "bias_stability_score_m2": m2_stability,
            "bias_stability_score_m2_available_data": m2_stability_available,
            "bias_stability_score_m2_material_relief": m2_material_stability,
            "pass_threshold": float(args.gate_min_bias_stability),
            "leading_model_by_stability": leading_stability,
            "leading_model_fractions": leading_fracs,
            "phase_pass_probabilities": {
                "phase1_closure": float(p1_prob),
                "phase2_growth_s8": float(p2_prob),
                "phase3_mu_sigma": float(p3_prob),
                "phase4_distance_ratio": float(p4_prob),
                "phase5_external_constraints_full": float(p5_full_prob),
                "phase5_external_constraints_available_data": float(p5_available_prob),
            },
            "nuisance_priors": {
                "relief_sigma": float(relief_sigma),
                "chi2_improvement_frac_sigma": float(args.calib_chi2_improvement_frac_sigma),
            },
            "draw_stats": {
                "relief_draw": {
                    "mean": float(np.mean(relief_draw)),
                    "p16": float(np.percentile(relief_draw, 16.0)),
                    "p50": float(np.percentile(relief_draw, 50.0)),
                    "p84": float(np.percentile(relief_draw, 84.0)),
                },
                "chi2_improvement_draw": {
                    "mean": float(np.mean(chi2_improvement_draw)),
                    "p16": float(np.percentile(chi2_improvement_draw, 16.0)),
                    "p50": float(np.percentile(chi2_improvement_draw, 50.0)),
                    "p84": float(np.percentile(chi2_improvement_draw, 84.0)),
                },
            },
        }

    strict_pass = bool(m2_supported)
    material_relief_pass = bool(m2_supported and relief_material)
    summary = {
        "created_utc": _utc_now(),
        "phase": "phase6_model_selection",
        "mode": "proxy_scorecard",
        "inputs": {
            "phase1_summary": str(p1_path),
            "phase2_summary": str(p2_path),
            "phase3_summary": str(p3_path),
            "phase4_summary": str(p4_path),
            "phase5_summary": str(p5_path),
            "relief_summary": str(rel_path),
            "cmb_summary": str(cmb_path),
            "mg_refit_summary": str(refit_path),
            "calibration_robust": bool(args.calibration_robust),
            "calibration_samples": int(args.calibration_samples),
            "calibration_seed": int(args.calibration_seed),
            "gate_min_bias_stability": float(args.gate_min_bias_stability),
            "calib_relief_sigma": float(args.calib_relief_sigma),
            "calib_chi2_improvement_frac_sigma": float(args.calib_chi2_improvement_frac_sigma),
        },
        "proxy_metrics": {
            "phase_strict_pass": {
                "phase1_closure": p1_pass,
                "phase2_growth_s8": p2_pass,
                "phase3_mu_sigma": p3_pass,
                "phase4_distance_ratio": p4_pass,
                "phase5_external_constraints_full": p5_pass_full,
                "phase5_external_constraints_available_data": p5_pass_available,
            },
            "h0_relief_anchor": {
                "mean": relief_mean,
                "p50": relief_p50,
                "material_threshold": float(args.gate_min_relief_for_material),
                "material": relief_material,
            },
            "lensing_fit_quality": {
                "chi2_baseline_p50": chi2_baseline,
                "chi2_refit_p50": chi2_refit,
                "chi2_planck_ref_model": chi2_ref_model,
                "chi2_improvement_baseline_to_refit": chi2_improvement,
                "m3_material_fit_gain_threshold": float(args.gate_min_chi2_improvement_m3),
                "m3_material_fit_gain": m3_material_fit_gain,
            },
            "phase5_failing_checks_full": failing_p5_checks,
            "phase5_failing_checks_available_data": failing_p5_available_checks,
        },
        "model_scorecard": scorecard,
        "decision": {
            "leading_proxy_model": leading,
            "target_hypothesis_M2_supported": m2_supported,
            "target_hypothesis_M2_supported_with_available_data_gate": m2_supported_available,
            "target_hypothesis_M2_supported_with_material_relief": material_relief_pass,
            "target_hypothesis_blockers": failing_p5_checks if not m2_supported else [],
            "target_hypothesis_available_data_blockers": (
                failing_p5_available_checks if not m2_supported_available else []
            ),
            "note": (
                "This decision is a proxy scorecard from phase-level outputs and not a full "
                "Bayesian global model comparison with TT/TE/EE+BBN+recombination."
            ),
        },
        "strict_gate": {
            "pass": strict_pass,
            "definition": "Core proxy gate for M2 support.",
            "checks": [
                {
                    "name": "M2_proxy_supported",
                    "value": m2_supported,
                    "pass": m2_supported,
                }
            ],
        },
        "material_relief_gate": {
            "pass": material_relief_pass,
            "definition": "Core proxy gate plus material anchor-relief requirement.",
            "checks": [
                {
                    "name": "M2_proxy_supported",
                    "value": m2_supported,
                    "pass": m2_supported,
                },
                {
                    "name": "anchor_relief_material",
                    "value": relief_material,
                    "threshold": float(args.gate_min_relief_for_material),
                    "pass": relief_material,
                },
            ],
        },
    }
    if calibration_robust_decision is not None:
        summary["calibration_robust_gate"] = {
            "pass": bool(calibration_robust_decision["pass"]),
            "definition": "Calibration-marginalized M2 bias-stability gate.",
            "checks": [
                {
                    "name": "M2_bias_stability_score",
                    "value": float(calibration_robust_decision["bias_stability_score_m2"]),
                    "threshold": float(calibration_robust_decision["pass_threshold"]),
                    "pass": bool(calibration_robust_decision["pass"]),
                }
            ],
        }
        summary["calibration_robust_decision"] = calibration_robust_decision
        summary["decision"]["bias_stability_score_m2"] = float(calibration_robust_decision["bias_stability_score_m2"])
        summary["decision"]["leading_model_by_stability"] = str(calibration_robust_decision["leading_model_by_stability"])
        summary["decision"]["target_hypothesis_M2_stable_under_calibration"] = bool(
            calibration_robust_decision["pass"]
        )

    out_dir = Path(args.out).resolve()
    _write_json_atomic(out_dir / "tables" / "summary.json", summary)

    lines = [
        "# Forward Test Phase 6: Global Model-Selection Proxy",
        "",
        f"- Generated UTC: `{summary['created_utc']}`",
        "",
        "## Proxy Snapshot",
        "",
        (
            f"- Phase passes: `P1={p1_pass}`, `P2={p2_pass}`, `P3={p3_pass}`, "
            f"`P4={p4_pass}`, `P5_full={p5_pass_full}`, `P5_data={p5_pass_available}`"
        ),
        f"- Anchor relief mean: `{relief_mean:.4f}` (material threshold `{float(args.gate_min_relief_for_material):.2f}`)",
        f"- Lensing chi2 baseline->refit: `{chi2_baseline:.3f} -> {chi2_refit:.3f}` (improvement `{chi2_improvement:.3f}`)",
        "",
        "## Model Scorecard",
        "",
        f"- `M0`: `{scorecard['M0']['proxy_status']}`",
        f"- `M1`: `{scorecard['M1']['proxy_status']}`",
        f"- `M2`: `{scorecard['M2']['proxy_status']}`",
        f"- `M3`: `{scorecard['M3']['proxy_status']}`",
        "",
        "## Decision",
        "",
        f"- Leading proxy model: `{leading}`",
        f"- Target hypothesis (`M2`) supported: `{'YES' if m2_supported else 'NO'}`",
        (
            "- Target hypothesis (`M2`) supported under available-data gate: "
            f"`{'YES' if m2_supported_available else 'NO'}`"
        ),
        (
            "- Target hypothesis (`M2`) supported with material-relief requirement: "
            f"`{'YES' if material_relief_pass else 'NO'}`"
        ),
    ]
    if failing_p5_checks:
        lines.append(f"- Current blockers: `{', '.join(failing_p5_checks)}`")
    if failing_p5_available_checks and (failing_p5_available_checks != failing_p5_checks):
        lines.append(f"- Available-data blockers: `{', '.join(failing_p5_available_checks)}`")
    if calibration_robust_decision is not None:
        lines.extend(
            [
                "",
                "## Calibration-Robust Stability",
                "",
                (
                    f"- M2 bias-stability score: "
                    f"`{float(calibration_robust_decision['bias_stability_score_m2']):.3f}` "
                    f"(threshold `{float(calibration_robust_decision['pass_threshold']):.3f}`)"
                ),
                (
                    f"- M2 available-data stability: "
                    f"`{float(calibration_robust_decision['bias_stability_score_m2_available_data']):.3f}`"
                ),
                (
                    f"- M2 material-relief stability: "
                    f"`{float(calibration_robust_decision['bias_stability_score_m2_material_relief']):.3f}`"
                ),
                (
                    f"- Leading model by stability: "
                    f"`{str(calibration_robust_decision['leading_model_by_stability'])}`"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "- Proxy scorecard only; full model selection requires unified likelihood work not yet run.",
            "",
            "## Artifacts",
            "",
            "- `tables/summary.json`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {out_dir / 'tables' / 'summary.json'}")
    print(f"[done] strict_gate={'PASS' if strict_pass else 'FAIL'}")

    if bool(args.fail_on_gate) and (not strict_pass):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
