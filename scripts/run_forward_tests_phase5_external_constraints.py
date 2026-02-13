#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from entropy_horizon_recon.sirens import load_mu_forward_posterior


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_run_dirs(text: str) -> list[str]:
    vals = [t.strip() for t in str(text).split(",") if t.strip()]
    if not vals:
        raise ValueError("No run dirs provided.")
    return vals


def _parse_range(text: str) -> tuple[float, float]:
    tok = [t.strip() for t in str(text).split(",") if t.strip()]
    if len(tok) != 2:
        raise ValueError("Expected range formatted as 'lo,hi'.")
    lo = float(tok[0])
    hi = float(tok[1])
    if not (lo < hi):
        raise ValueError("Require lo < hi for range.")
    return lo, hi


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


def _collect_strings(x: Any) -> list[str]:
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        out: list[str] = []
        for v in x:
            if isinstance(v, str):
                out.append(v)
        return out
    return []


def _extract_phase_pass_probability(
    phase_summary: dict[str, Any],
    *,
    mode: str,
    gate_key: str = "strict_gate",
) -> float:
    if str(mode) == "robust_if_available":
        rg = phase_summary.get("calibration_robust_gate", {})
        if isinstance(rg, dict):
            pf = rg.get("pass_fraction")
            if isinstance(pf, (int, float)) and np.isfinite(float(pf)):
                return float(np.clip(float(pf), 0.0, 1.0))
    gate = phase_summary.get(str(gate_key), {})
    return 1.0 if bool(gate.get("pass", False)) else 0.0


def _primary_cmb_included(cmb: dict[str, Any], mg_refit: dict[str, Any]) -> tuple[bool, list[str]]:
    notes: list[str] = []
    n1 = _collect_strings(cmb.get("assumptions_and_scope", {}).get("not_included_here", []))
    n2 = _collect_strings(mg_refit.get("scope_notes", {}).get("not_modeled", []))
    notes.extend(n1)
    notes.extend(n2)
    low = " ".join(notes).lower()
    missing = ("tt/te/ee" in low) or ("primary" in low and "not" in low)
    return (not missing), notes


def _primary_from_planck_global(summary_path: Path) -> tuple[bool, dict[str, Any]]:
    if not summary_path.exists():
        return False, {"reason": "missing_summary", "summary_path": str(summary_path)}

    try:
        s = _read_json(summary_path)
    except Exception as e:
        return False, {"reason": "summary_read_error", "summary_path": str(summary_path), "error": str(e)}

    best = s.get("best_success") or s.get("best_any") or {}
    cfg = Path(str(best.get("config_path", ""))).resolve() if best else Path("")
    if (not str(cfg)) or (not cfg.exists()):
        return False, {
            "reason": "missing_best_config",
            "summary_path": str(summary_path),
            "config_path": str(cfg),
        }

    try:
        y = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    except Exception as e:
        return False, {
            "reason": "config_parse_error",
            "summary_path": str(summary_path),
            "config_path": str(cfg),
            "error": str(e),
        }
    if not isinstance(y, dict):
        return False, {
            "reason": "config_not_mapping",
            "summary_path": str(summary_path),
            "config_path": str(cfg),
        }

    lk = y.get("likelihood", {})
    if not isinstance(lk, dict):
        return False, {
            "reason": "missing_likelihood_mapping",
            "summary_path": str(summary_path),
            "config_path": str(cfg),
        }
    lk_keys = [str(k) for k in lk.keys()]
    req = {
        "planck_2018_lowl.TT",
        "planck_2018_lowl.EE",
        "planck_2018_highl_plik.TTTEEE",
    }
    has_req = all(r in lk_keys for r in req)
    has_lensing = "planck_2018_lensing.clik" in lk_keys

    params = y.get("params", {})
    has_bbn_proxy = False
    if isinstance(params, dict):
        pkeys = {str(k) for k in params.keys()}
        has_bbn_proxy = ("Y_p" in pkeys) and ("DHBBN" in pkeys)

    successes = int(s.get("successes", 0)) if isinstance(s, dict) else 0
    ok = bool(has_req and has_lensing and has_bbn_proxy and (successes > 0))
    details = {
        "summary_path": str(summary_path),
        "config_path": str(cfg),
        "likelihood_keys": lk_keys,
        "has_required_primary_terms": bool(has_req),
        "has_planck_lensing_term": bool(has_lensing),
        "has_bbn_proxy_params": bool(has_bbn_proxy),
        "summary_successes": int(successes),
    }
    return ok, details


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Concrete forward test (Phase 5): external-constraint stress check "
            "for non-local cosmology channels."
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
    ap.add_argument("--out", default="outputs/forward_tests/phase5_external_constraints")
    ap.add_argument(
        "--draws-per-run",
        type=int,
        default=1000,
        help="Used only for deterministic bookkeeping of per-run draw usage.",
    )
    ap.add_argument("--seed", type=int, default=9305)

    ap.add_argument(
        "--coverage-mode",
        choices=["observed", "assumed_highz"],
        default="observed",
        help=(
            "Coverage gate mode: 'observed' uses posterior z-grid max; "
            "'assumed_highz' uses assumed-zmax as effective coverage (assumption study)."
        ),
    )
    ap.add_argument(
        "--assumed-zmax",
        type=float,
        default=1100.0,
        help="Effective z_max used when --coverage-mode=assumed_highz.",
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
        "--cmb-baseline-summary",
        default="outputs/hubble_tension_cmb_forecast_camb64_20260210_analyticrefresh/tables/summary.json",
    )
    ap.add_argument(
        "--mg-refit-summary",
        default="outputs/hubble_tension_mg_lensing_refit_camb32_20260210_live/tables/summary.json",
    )
    ap.add_argument(
        "--planck-global-summary",
        default="outputs/planck_global_mg_refit_realmin60_20260210_live/summary.json",
    )

    ap.add_argument("--mstar2-ratio-range", default="0.8,1.2")
    ap.add_argument("--gate-min-zmax-bbn", type=float, default=30.0)
    ap.add_argument("--gate-min-zmax-recomb", type=float, default=1000.0)
    ap.add_argument("--gate-require-primary-cmb-modeled", action="store_true", default=True)
    ap.add_argument("--gate-require-phase3-pass", action="store_true", default=True)
    ap.add_argument("--gate-require-phase4-pass", action="store_true", default=True)

    ap.add_argument(
        "--calibration-robust",
        action="store_true",
        help="Enable calibration-marginalized robustness gate for non-local external-constraint checks.",
    )
    ap.add_argument(
        "--calibration-samples",
        type=int,
        default=4000,
        help="Number of nuisance draws for calibration-robust gate.",
    )
    ap.add_argument("--calibration-seed", type=int, default=29505, help="RNG seed for calibration robustness sampling.")
    ap.add_argument(
        "--gate-min-robust-pass-frac",
        type=float,
        default=0.60,
        help="Calibration-robust gate passes if nuisance-marginal pass fraction >= this value.",
    )
    ap.add_argument(
        "--calib-mstar-frac-sigma",
        type=float,
        default=0.10,
        help="Stddev for fractional calibration uncertainty on mstar2_ratio_0 p50.",
    )
    ap.add_argument(
        "--calibration-phase-probability-mode",
        choices=["strict", "robust_if_available"],
        default="robust_if_available",
        help="How phase3/phase4 dependency pass probabilities are sourced under calibration-robust mode.",
    )
    ap.add_argument("--fail-on-gate", action="store_true")
    args = ap.parse_args()

    if int(args.draws_per_run) <= 0:
        raise ValueError("draws-per-run must be positive.")
    if str(args.coverage_mode) == "assumed_highz" and float(args.assumed_zmax) <= 0.0:
        raise ValueError("assumed-zmax must be positive for coverage_mode=assumed_highz.")
    if not (0.0 <= float(args.gate_min_robust_pass_frac) <= 1.0):
        raise ValueError("gate-min-robust-pass-frac must be in [0,1].")
    if int(args.calibration_samples) < 100:
        raise ValueError("calibration-samples must be >= 100.")
    if float(args.calib_mstar_frac_sigma) < 0.0:
        raise ValueError("calib-mstar-frac-sigma must be >= 0.")

    run_dirs = _parse_run_dirs(args.run_dirs)
    mstar_lo, mstar_hi = _parse_range(args.mstar2_ratio_range)

    phase3_path = Path(args.phase3_summary).resolve()
    phase4_path = Path(args.phase4_summary).resolve()
    cmb_path = Path(args.cmb_baseline_summary).resolve()
    mg_path = Path(args.mg_refit_summary).resolve()
    planck_global_path = Path(args.planck_global_summary).resolve()
    for p in (phase3_path, phase4_path, cmb_path, mg_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    phase3 = _read_json(phase3_path)
    phase4 = _read_json(phase4_path)
    cmb = _read_json(cmb_path)
    mg_refit = _read_json(mg_path)

    out_dir = Path(args.out).resolve()
    tab_dir = out_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    per_run: list[dict[str, Any]] = []
    zmax_list: list[float] = []

    for run in run_dirs:
        post = load_mu_forward_posterior(run)
        n_all = int(post.H0.size)
        n_use = min(int(args.draws_per_run), n_all)
        if n_use < n_all:
            _ = np.sort(rng.choice(n_all, size=n_use, replace=False))
        zmax = float(np.max(post.z_grid))
        zmax_list.append(zmax)
        per_run.append(
            {
                "run_dir": str(Path(run).resolve()),
                "draws_total": int(n_all),
                "draws_used": int(n_use),
                "z_max": zmax,
            }
        )

    zmax_min = float(np.min(zmax_list))
    zmax_max = float(np.max(zmax_list))
    if str(args.coverage_mode) == "assumed_highz":
        zmax_eff_min = float(max(zmax_min, float(args.assumed_zmax)))
        zmax_eff_max = float(max(zmax_max, float(args.assumed_zmax)))
    else:
        zmax_eff_min = zmax_min
        zmax_eff_max = zmax_max

    bbn_mult = float(args.gate_min_zmax_bbn) / max(1e-12, zmax_min)
    recomb_mult = float(args.gate_min_zmax_recomb) / max(1e-12, zmax_min)
    coverage_reachable_without_assumption = bool(
        (zmax_min >= float(args.gate_min_zmax_bbn))
        and (zmax_min >= float(args.gate_min_zmax_recomb))
    )

    p3_pass = bool(phase3.get("strict_gate", {}).get("pass", False))
    p4_pass = bool(phase4.get("strict_gate", {}).get("pass", False))

    primary_local_ok, primary_scope_notes = _primary_cmb_included(cmb, mg_refit)
    primary_global_ok, primary_global_details = _primary_from_planck_global(planck_global_path)
    primary_cov_ok = bool(primary_local_ok or primary_global_ok)

    mstar_p50 = float(mg_refit["fit_parameter_stats"]["mstar2_ratio_0"]["p50"])

    direct_bbn = bool(zmax_eff_min >= float(args.gate_min_zmax_bbn))
    direct_recomb = bool(zmax_eff_min >= float(args.gate_min_zmax_recomb))
    early_time_coverage_ok = bool((direct_bbn and direct_recomb) or primary_global_ok)

    checks = [
        {
            "name": "mstar2_ratio_p50_in_range",
            "category": "physics",
            "gate_required": True,
            "value": mstar_p50,
            "range": [float(mstar_lo), float(mstar_hi)],
            "pass": float(mstar_lo) <= mstar_p50 <= float(mstar_hi),
        },
        {
            "name": "phase3_dependency_pass",
            "category": "dependency",
            "gate_required": True,
            "value": p3_pass,
            "pass": p3_pass if bool(args.gate_require_phase3_pass) else True,
        },
        {
            "name": "phase4_dependency_pass",
            "category": "dependency",
            "gate_required": True,
            "value": p4_pass,
            "pass": p4_pass if bool(args.gate_require_phase4_pass) else True,
        },
        {
            "name": "primary_cmb_modeled_in_scope",
            "category": "scope",
            "gate_required": True,
            "value": primary_cov_ok,
            "pass": primary_cov_ok if bool(args.gate_require_primary_cmb_modeled) else True,
        },
        {
            "name": "zmax_reaches_bbn_proxy_direct",
            "category": "coverage_diagnostic",
            "gate_required": False,
            "value": zmax_eff_min,
            "threshold": float(args.gate_min_zmax_bbn),
            "pass": direct_bbn,
        },
        {
            "name": "zmax_reaches_recomb_proxy_direct",
            "category": "coverage_diagnostic",
            "gate_required": False,
            "value": zmax_eff_min,
            "threshold": float(args.gate_min_zmax_recomb),
            "pass": direct_recomb,
        },
        {
            "name": "early_time_external_channel_present",
            "category": "coverage",
            "gate_required": False,
            "value": primary_global_ok,
            "pass": primary_global_ok,
        },
        {
            "name": "early_time_coverage_satisfied",
            "category": "coverage",
            "gate_required": True,
            "value": early_time_coverage_ok,
            "pass": early_time_coverage_ok,
        },
    ]

    strict_gate_checks = [c for c in checks if bool(c.get("gate_required", True))]
    strict_pass = bool(all(bool(c["pass"]) for c in strict_gate_checks))

    available_data_checks = [
        c for c in checks if bool(c.get("gate_required", True)) and str(c.get("category")) in {"physics", "dependency"}
    ]
    available_data_pass = bool(all(bool(c["pass"]) for c in available_data_checks))

    failing_by_category: dict[str, list[str]] = {}
    for c in strict_gate_checks:
        if not bool(c["pass"]):
            cat = str(c.get("category", "uncategorized"))
            failing_by_category.setdefault(cat, []).append(str(c["name"]))

    calibration_robust_gate: dict[str, Any] | None = None
    if bool(args.calibration_robust):
        rng_cal = np.random.default_rng(int(args.calibration_seed))
        n_cal = int(args.calibration_samples)

        mstar_eff = mstar_p50 * (1.0 + float(args.calib_mstar_frac_sigma) * rng_cal.normal(size=n_cal))

        p3_prob = _extract_phase_pass_probability(
            phase3,
            mode=str(args.calibration_phase_probability_mode),
            gate_key="strict_gate",
        )
        p4_prob = _extract_phase_pass_probability(
            phase4,
            mode=str(args.calibration_phase_probability_mode),
            gate_key="strict_gate",
        )
        p3_draw = rng_cal.random(size=n_cal) < float(np.clip(p3_prob, 0.0, 1.0))
        p4_draw = rng_cal.random(size=n_cal) < float(np.clip(p4_prob, 0.0, 1.0))

        pass_mstar = (mstar_eff >= float(mstar_lo)) & (mstar_eff <= float(mstar_hi))
        pass_p3 = p3_draw if bool(args.gate_require_phase3_pass) else np.ones((n_cal,), dtype=bool)
        pass_p4 = p4_draw if bool(args.gate_require_phase4_pass) else np.ones((n_cal,), dtype=bool)
        pass_primary = np.full(
            (n_cal,),
            bool(primary_cov_ok if bool(args.gate_require_primary_cmb_modeled) else True),
            dtype=bool,
        )
        pass_bbn_direct = np.full((n_cal,), bool(direct_bbn), dtype=bool)
        pass_recomb_direct = np.full((n_cal,), bool(direct_recomb), dtype=bool)
        pass_early_ext = np.full((n_cal,), bool(primary_global_ok), dtype=bool)
        pass_early_cov = np.full((n_cal,), bool(early_time_coverage_ok), dtype=bool)

        pass_by_name: dict[str, np.ndarray] = {
            "mstar2_ratio_p50_in_range": pass_mstar,
            "phase3_dependency_pass": pass_p3,
            "phase4_dependency_pass": pass_p4,
            "primary_cmb_modeled_in_scope": pass_primary,
            "zmax_reaches_bbn_proxy_direct": pass_bbn_direct,
            "zmax_reaches_recomb_proxy_direct": pass_recomb_direct,
            "early_time_external_channel_present": pass_early_ext,
            "early_time_coverage_satisfied": pass_early_cov,
        }

        full_required_names = [str(c["name"]) for c in checks if bool(c.get("gate_required", True))]
        available_required_names = [
            str(c["name"])
            for c in checks
            if bool(c.get("gate_required", True)) and str(c.get("category")) in {"physics", "dependency"}
        ]

        full_pass_draws = np.ones((n_cal,), dtype=bool)
        for nm in full_required_names:
            full_pass_draws &= pass_by_name[nm]

        available_pass_draws = np.ones((n_cal,), dtype=bool)
        for nm in available_required_names:
            available_pass_draws &= pass_by_name[nm]

        robust_full_frac = float(np.mean(full_pass_draws))
        robust_available_frac = float(np.mean(available_pass_draws))
        robust_threshold = float(args.gate_min_robust_pass_frac)
        robust_full_pass = bool(robust_full_frac >= robust_threshold)
        robust_available_pass = bool(robust_available_frac >= robust_threshold)

        robust_checks: list[dict[str, Any]] = []
        for c in checks:
            nm = str(c["name"])
            robust_checks.append(
                {
                    "name": nm,
                    "category": str(c.get("category", "")),
                    "gate_required": bool(c.get("gate_required", True)),
                    "pass_fraction": float(np.mean(pass_by_name[nm])),
                }
            )

        calibration_robust_gate = {
            "enabled": True,
            "mode": "catalog_calibration_marginalized_external_constraints",
            "pass": robust_full_pass,
            "pass_fraction": robust_full_frac,
            "full_pass_fraction": robust_full_frac,
            "available_data_pass": robust_available_pass,
            "available_data_pass_fraction": robust_available_frac,
            "pass_fraction_threshold": robust_threshold,
            "dependency_pass_probabilities": {
                "phase3": float(np.clip(p3_prob, 0.0, 1.0)),
                "phase4": float(np.clip(p4_prob, 0.0, 1.0)),
                "source_mode": str(args.calibration_phase_probability_mode),
            },
            "checks": robust_checks,
            "nuisance_priors": {
                "mstar_frac_sigma": float(args.calib_mstar_frac_sigma),
            },
            "effective_metric_stats": {
                "mstar2_ratio_0_p50_effective": _stats_1d(mstar_eff),
            },
        }

    with (tab_dir / "per_run_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_dir", "draws_total", "draws_used", "z_max"])
        for row in per_run:
            w.writerow([row["run_dir"], row["draws_total"], row["draws_used"], row["z_max"]])

    summary = {
        "created_utc": _utc_now(),
        "phase": "phase5_external_constraints",
        "mode": "external_constraint_stress_test",
        "inputs": {
            "run_dirs": [str(Path(r).resolve()) for r in run_dirs],
            "draws_per_run": int(args.draws_per_run),
            "seed": int(args.seed),
            "calibration_robust": bool(args.calibration_robust),
            "calibration_samples": int(args.calibration_samples),
            "calibration_seed": int(args.calibration_seed),
            "gate_min_robust_pass_frac": float(args.gate_min_robust_pass_frac),
            "calib_mstar_frac_sigma": float(args.calib_mstar_frac_sigma),
            "calibration_phase_probability_mode": str(args.calibration_phase_probability_mode),
            "coverage_mode": str(args.coverage_mode),
            "assumed_zmax": float(args.assumed_zmax),
            "phase3_summary": str(phase3_path),
            "phase4_summary": str(phase4_path),
            "cmb_baseline_summary": str(cmb_path),
            "mg_refit_summary": str(mg_path),
            "planck_global_summary": str(planck_global_path),
        },
        "external_constraint_scope": {
            "notes": [
                "Phase5 gate uses MG-range, dependency, primary-CMB scope, and early-time coverage channels.",
            ],
        },
        "coverage": {
            "coverage_mode": str(args.coverage_mode),
            "z_max_min_across_runs": zmax_min,
            "z_max_max_across_runs": zmax_max,
            "z_max_effective_min": zmax_eff_min,
            "z_max_effective_max": zmax_eff_max,
            "assumed_zmax": float(args.assumed_zmax),
            "gate_min_zmax_bbn": float(args.gate_min_zmax_bbn),
            "gate_min_zmax_recomb": float(args.gate_min_zmax_recomb),
            "reachability_diagnostic": {
                "coverage_gate_reachable_without_assumption": coverage_reachable_without_assumption,
                "required_multiplier_to_bbn_gate": bbn_mult,
                "required_multiplier_to_recomb_gate": recomb_mult,
            },
        },
        "scope_checks": {
            "primary_cmb_modeled": primary_cov_ok,
            "primary_cmb_modeled_local_pipeline": primary_local_ok,
            "primary_cmb_modeled_planck_global": primary_global_ok,
            "source_notes": primary_scope_notes,
            "planck_global_details": primary_global_details,
        },
        "dependency_checks": {
            "phase3_strict_gate_pass": p3_pass,
            "phase4_strict_gate_pass": p4_pass,
        },
        "mg_refit_snapshot": {
            "mstar2_ratio_0_p50": mstar_p50,
            "mstar2_drop_pct_p50": float(mg_refit["fit_parameter_stats"]["mstar2_drop_pct"]["p50"]),
            "mstar2_ratio_gate_range": [float(mstar_lo), float(mstar_hi)],
        },
        "per_run": per_run,
        "strict_gate": {
            "pass": strict_pass,
            "definition": "Full gate with physics + dependency + scope + coverage checks.",
            "checks": strict_gate_checks,
            "diagnostic_checks": [c for c in checks if not bool(c.get("gate_required", True))],
        },
        "available_data_gate": {
            "pass": available_data_pass,
            "definition": "Data-available gate with physics + dependency checks only.",
            "checks": available_data_checks,
        },
        "failing_checks_by_category": failing_by_category,
        "interpretation_note": (
            "This phase is a proxy stress test with explicit assumptions. "
            "It is not a full TT/TE/EE+BBN+recombination joint likelihood analysis."
        ),
    }
    if calibration_robust_gate is not None:
        summary["calibration_robust_gate"] = calibration_robust_gate

    _write_json_atomic(tab_dir / "summary.json", summary)

    lines = [
        "# Forward Test Phase 5: External-Constraint Stress Test",
        "",
        f"- Generated UTC: `{summary['created_utc']}`",
        "",
        "## Core Checks",
        "",
        f"- `mstar2_ratio_0` p50: `{mstar_p50:.6f}` (allowed range `{mstar_lo:.3f}` to `{mstar_hi:.3f}`)",
        (
            f"- Posterior z-max across runs: `min={zmax_min:.3f}`, `max={zmax_max:.3f}` "
            f"(effective min/max: `{zmax_eff_min:.3f}` / `{zmax_eff_max:.3f}`, mode=`{args.coverage_mode}`)"
        ),
        (
            f"- Coverage reachability (without assumptions): "
            f"`{'YES' if coverage_reachable_without_assumption else 'NO'}` "
            f"(x{bbn_mult:.1f} to BBN gate, x{recomb_mult:.1f} to recomb gate)"
        ),
        (
            f"- Primary CMB coverage: `{'YES' if primary_cov_ok else 'NO'}` "
            f"(local_pipeline={'YES' if primary_local_ok else 'NO'}, "
            f"planck_global={'YES' if primary_global_ok else 'NO'})"
        ),
        f"- Phase dependencies: `phase3={'PASS' if p3_pass else 'FAIL'}`, `phase4={'PASS' if p4_pass else 'FAIL'}`",
        "",
        "## Strict Gate",
        "",
        f"- Full result: `{'PASS' if strict_pass else 'FAIL'}`",
        f"- Data-available result: `{'PASS' if available_data_pass else 'FAIL'}`",
    ]

    if calibration_robust_gate is not None:
        lines.extend(
            [
                (
                    f"- Calibration-robust full result: "
                    f"`{'PASS' if calibration_robust_gate['pass'] else 'FAIL'}` "
                    f"(pass fraction `{float(calibration_robust_gate['full_pass_fraction']):.3f}`)"
                ),
                (
                    f"- Calibration-robust data-available result: "
                    f"`{'PASS' if calibration_robust_gate['available_data_pass'] else 'FAIL'}` "
                    f"(pass fraction `{float(calibration_robust_gate['available_data_pass_fraction']):.3f}`)"
                ),
            ]
        )

    for c in checks:
        lines.append(
            f"- `{c['name']}` [{c['category']}]: "
            f"`{'PASS' if c['pass'] else 'FAIL'}` (value={c['value']}, required={'YES' if bool(c.get('gate_required', True)) else 'NO'})"
        )

    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "- This is a scoped external-constraint proxy test, not a full early-time constrained MG global fit.",
            "",
            "## Artifacts",
            "",
            "- `tables/summary.json`",
            "- `tables/per_run_summary.csv`",
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
