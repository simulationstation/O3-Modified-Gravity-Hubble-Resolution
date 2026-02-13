#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _get_path(d: dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing path '{dotted}' (stuck at '{part}')")
        cur = cur[part]
    return cur


def _collect_metrics(sources: dict[str, Path]) -> dict[str, float]:
    planck = _read_json(sources["planck_refit_summary"])
    anchor = _read_json(sources["early_anchor_summary"])
    relief = _read_json(sources["final_relief_summary"])
    cmb = _read_json(sources["cmb_forecast_summary"])
    mg = _read_json(sources["mg_lensing_refit_summary"])

    metrics: dict[str, float] = {
        "planck_refit.successes": float(planck["successes"]),
        "planck_refit.failures": float(planck["failures"]),
        "planck_refit.best_success_objective": float(planck["best_success"]["objective"]),
        "anchor.H0_p50": float(anchor["H0"]["p50"]),
        "anchor.omegam_p50": float(anchor["omegam"]["p50"]),
        "anchor.Alens_p50": float(anchor["Alens"]["p50"]),
        "relief.anchor_gr_mean_mc": float(relief["posterior_with_mc_calibration"]["mean"]),
        "relief.anchor_gr_p50_mc": float(relief["posterior_with_mc_calibration"]["p50"]),
        "relief.anchor_gr_p84_mc": float(relief["posterior_with_mc_calibration"]["p84"]),
        "cmb_baseline.delta_frac_pct_q50_L100": float(cmb["headline_multipoles"]["near_L100"]["delta_frac_pct_q50"]),
        "cmb_baseline.delta_frac_pct_q50_L300": float(cmb["headline_multipoles"]["near_L300"]["delta_frac_pct_q50"]),
        "cmb_baseline.chi2_p50": float(cmb["lensing_chi2"]["chi2_draws"]["p50"]),
        "cmb_baseline.p_draw_better_than_ref": float(cmb["lensing_chi2"]["p_draw_better_than_ref"]),
        "mg_refit.chi2_refit_p50": float(mg["chi2"]["chi2_mg_refit_draws"]["p50"]),
        "mg_refit.p_refit_better_than_ref": float(mg["chi2"]["p_refit_better_than_ref"]),
        "mg_refit.mstar2_ratio_0_p50": float(mg["fit_parameter_stats"]["mstar2_ratio_0"]["p50"]),
        "mg_refit.refit_delta_frac_pct_q50_L100": float(mg["headline_multipoles"]["near_L100"]["refit_delta_frac_pct_q50"]),
        "mg_refit.refit_delta_frac_pct_q50_L300": float(mg["headline_multipoles"]["near_L300"]["refit_delta_frac_pct_q50"]),
    }
    return metrics


def _resolve_sources(manifest: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    srcs = manifest.get("baseline_sources", {})
    required = [
        "planck_refit_summary",
        "early_anchor_summary",
        "final_relief_summary",
        "cmb_forecast_summary",
        "mg_lensing_refit_summary",
    ]
    for key in required:
        raw = srcs.get(key)
        if not isinstance(raw, str) or not raw:
            raise KeyError(f"Manifest missing baseline_sources.{key}")
        p = (repo_root / raw).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Missing source for {key}: {p}")
        out[key] = p
    return out


def _freeze_manifest(manifest: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    checks = manifest.get("checks", {})
    if not isinstance(checks, dict):
        raise ValueError("Manifest field 'checks' must be a mapping.")

    for key, value in metrics.items():
        row = checks.get(key)
        if not isinstance(row, dict):
            continue
        row["expected"] = float(value)

    manifest["frozen_utc"] = _utc_now()
    manifest["frozen_note"] = "Expected values populated by run_forward_tests_phase0_baseline_freeze.py"
    return manifest


def _check_manifest(manifest: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    checks = manifest.get("checks", {})
    if not isinstance(checks, dict):
        raise ValueError("Manifest field 'checks' must be a mapping.")

    rows: list[dict[str, Any]] = []
    n_fail = 0

    for key, spec in checks.items():
        if not isinstance(spec, dict):
            rows.append({"metric": key, "status": "invalid_spec", "reason": "spec is not an object"})
            n_fail += 1
            continue

        expected = spec.get("expected")
        tol = spec.get("abs_tol", 0.0)
        actual = metrics.get(key)

        if actual is None:
            rows.append({"metric": key, "status": "missing_actual"})
            n_fail += 1
            continue
        if expected is None:
            rows.append({"metric": key, "status": "missing_expected", "actual": actual})
            n_fail += 1
            continue

        exp = float(expected)
        t = float(tol)
        diff = float(actual - exp)
        ok = math.isfinite(diff) and abs(diff) <= t
        if not ok:
            n_fail += 1

        rows.append(
            {
                "metric": key,
                "status": "pass" if ok else "fail",
                "actual": float(actual),
                "expected": exp,
                "abs_diff": abs(diff),
                "abs_tol": t,
            }
        )

    return {
        "checked_utc": _utc_now(),
        "n_total": len(rows),
        "n_fail": int(n_fail),
        "n_pass": int(len(rows) - n_fail),
        "ok": bool(n_fail == 0),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Phase-0 baseline freeze/check for forward tests. Reads baseline outputs, "
            "writes a snapshot, and optionally freezes/checks manifest expectations."
        )
    )
    ap.add_argument("--manifest", default="forward_tests_manifest.json", help="Manifest JSON path.")
    ap.add_argument(
        "--out",
        default="outputs/forward_tests/phase0_baseline_freeze/baseline_snapshot.json",
        help="Snapshot JSON output path.",
    )
    ap.add_argument("--freeze-manifest", action="store_true", help="Populate manifest checks.*.expected from current baseline.")
    ap.add_argument("--check-manifest", action="store_true", help="Run check report against manifest expected values.")
    ap.add_argument(
        "--check-out",
        default="outputs/forward_tests/phase0_baseline_freeze/baseline_check.json",
        help="Check report output path.",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = _read_json(manifest_path)
    sources = _resolve_sources(manifest, repo_root)
    metrics = _collect_metrics(sources)

    snapshot = {
        "created_utc": _utc_now(),
        "manifest": str(manifest_path),
        "sources": {k: str(v) for k, v in sources.items()},
        "metrics": metrics,
    }
    snapshot_path = (repo_root / args.out).resolve()
    _write_json(snapshot_path, snapshot)
    print(f"[phase0] wrote snapshot: {snapshot_path}")

    if args.freeze_manifest:
        frozen = _freeze_manifest(manifest, metrics)
        _write_json(manifest_path, frozen)
        print(f"[phase0] updated manifest expected values: {manifest_path}")

    if args.check_manifest:
        report = _check_manifest(manifest, metrics)
        check_path = (repo_root / args.check_out).resolve()
        _write_json(check_path, report)
        status = "PASS" if report["ok"] else "FAIL"
        print(f"[phase0] check={status} wrote: {check_path}")
        return 0 if report["ok"] else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
