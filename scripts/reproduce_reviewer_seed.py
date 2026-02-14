#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@dataclass(frozen=True)
class Headline:
    label: str
    value: Any


def _fmt_float(x: float, ndp: int = 3) -> str:
    return f"{float(x):.{int(ndp)}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a reviewer seed reproduction report (headlines only).")
    ap.add_argument(
        "--out",
        default="",
        help="Output directory. Default: outputs/reviewer_seed_<timestamp>/",
    )
    args = ap.parse_args()

    artifacts = REPO_ROOT / "artifacts"
    if not artifacts.exists():
        raise SystemExit(f"Missing artifacts directory: {artifacts}")

    out_dir = Path(args.out) if args.out else (REPO_ROOT / "outputs" / f"reviewer_seed_{_utc_now_compact()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(artifacts / "manifest.json")

    # --- Dark siren: hardening suite (full pipeline baseline + calibrated p/Z) ---
    hard = artifacts / "o3" / "dark_siren_hardening_suite"
    baseline = _read_json(hard / "tables" / "baseline_recompute.json")
    pz_rows = _read_csv_rows(hard / "tables" / "calibrated_pz_table.csv")
    pz_baseline = next((r for r in pz_rows if (r.get("case") or "").startswith("baseline")), None)
    if not pz_baseline:
        raise SystemExit("Could not locate baseline row in calibrated_pz_table.csv")

    # --- Dark siren: spec-z coverage maxout (spectral-only + selection) ---
    specz = _read_json(artifacts / "o3" / "specz_coverage_maxout" / "summary.json")

    # --- Hubble: relief posterior ---
    hub_relief = _read_json(artifacts / "hubble" / "final_relief_posterior" / "final_relief_posterior_summary.json")

    # --- Hubble: Planck lensing refit (MG response) ---
    lens = _read_json(artifacts / "hubble" / "mg_lensing_refit" / "tables" / "summary.json")

    # --- Hubble: joint transfer bias fit ---
    joint = _read_json(artifacts / "hubble" / "joint_transfer_bias_fit" / "tables" / "summary.json")

    # Summarize the key numbers in one place.
    headlines: list[Headline] = []

    headlines.append(Headline("O3 dark siren ΔLPD_tot (full; selection-normalized)", baseline["delta_lpd_total"]))
    headlines.append(Headline("O3 dark siren ΔLPD_data (full)", baseline["delta_lpd_total_data"]))
    headlines.append(
        Headline(
            "O3 calibrated p (one-sided; GR-truth spectral-only null)",
            float(pz_baseline["p_one_sided"]),
        )
    )
    headlines.append(
        Headline(
            "O3 calibrated Z (one-sided; GR-truth spectral-only null)",
            float(pz_baseline["z_one_sided"]) if (pz_baseline.get("z_one_sided") or "").lower() != "nan" else None,
        )
    )

    # Spec-z maxout: Tier A best point.
    bestA = specz["best_points"]["A"]
    bestA_row = next((r for r in specz["scoring_best_points_rows"] if r.get("tier") == "A"), None)
    if not bestA_row:
        raise SystemExit("Missing Tier-A scoring_best_points_rows in specz maxout summary.json")
    headlines.append(
        Headline(
            "Spec-z maxout operating point (Tier A)",
            {"radius_arcsec": bestA["radius_arcsec"], "k": bestA["k"]},
        )
    )
    headlines.append(Headline("Spec-z maxout median host-weight anchored (Tier A; gate events)", bestA["median_frac_weight_true_gate_events"]))
    headlines.append(Headline("Spec-z override ΔLPD_tot (spectral-only+sel; Tier A)", bestA_row["delta_lpd_total"]))
    headlines.append(Headline("Spec-z override ΔLPD_data (spectral-only; Tier A)", bestA_row["delta_lpd_data"]))

    # Hubble relief posterior.
    mc = hub_relief["posterior_with_mc_calibration"]
    headlines.append(Headline("Hubble transfer-bias relief posterior mean (MC-calibrated)", mc["mean"]))
    headlines.append(Headline("Hubble transfer-bias relief posterior p16/p50/p84", [mc["p16"], mc["p50"], mc["p84"]]))

    # Planck lensing refit.
    chi2 = lens["chi2"]
    headlines.append(Headline("Planck lensing χ² (reference model)", chi2["chi2_planck_ref_model"]))
    headlines.append(Headline("Planck lensing χ² median (MG-aware refit)", chi2["chi2_mg_refit_draws"]["p50"]))
    headlines.append(Headline("Planck lensing baseline suppression near L~100 (median ΔC/C, %)", lens["headline_multipoles"]["near_L100"]["baseline_delta_frac_pct_q50"]))
    headlines.append(Headline("Planck lensing refit residual near L~100 (median ΔC/C, %)", lens["headline_multipoles"]["near_L100"]["refit_delta_frac_pct_q50"]))

    # Joint fit headline (as stored).
    # This is a light pointer; the detailed model interpretation lives in the paper.
    headlines.append(Headline("Joint transfer fit: log BF(transfer vs no-transfer)", joint.get("log_bayes_factor_transfer_vs_no_transfer")))

    summary = {
        "created_utc": _utc_now(),
        "git_sha": manifest.get("git_sha"),
        "artifacts_manifest": str((artifacts / "manifest.json").as_posix()),
        "headlines": {h.label: h.value for h in headlines},
        "artifact_paths": {
            "o3_dark_siren_hardening_suite": str(hard.as_posix()),
            "o3_specz_coverage_maxout": str((artifacts / "o3" / "specz_coverage_maxout").as_posix()),
            "hubble_final_relief_posterior": str((artifacts / "hubble" / "final_relief_posterior").as_posix()),
            "hubble_mg_lensing_refit": str((artifacts / "hubble" / "mg_lensing_refit").as_posix()),
            "hubble_joint_transfer_bias_fit": str((artifacts / "hubble" / "joint_transfer_bias_fit").as_posix()),
        },
    }

    _write_json_atomic(out_dir / "summary.json", summary)

    md_lines: list[str] = []
    md_lines.append("# Reviewer Seed Reproduction (Headlines)")
    md_lines.append("")
    md_lines.append(f"- Created: `{summary['created_utc']}`")
    md_lines.append(f"- Git SHA: `{summary['git_sha']}`")
    md_lines.append("")
    md_lines.append("## Headline Numbers")
    md_lines.append("")
    md_lines.append("| item | value |")
    md_lines.append("| --- | --- |")

    def cell(v: Any) -> str:
        if isinstance(v, float):
            return _fmt_float(v, 3)
        return f"`{v}`" if not isinstance(v, (list, dict)) else f"`{json.dumps(v)}`"

    for h in headlines:
        md_lines.append(f"| {h.label} | {cell(h.value)} |")

    md_lines.append("")
    md_lines.append("## Artifact Pointers")
    md_lines.append("")
    for k, v in summary["artifact_paths"].items():
        md_lines.append(f"- `{k}`: `{v}`")

    md_lines.append("")
    md_lines.append("## Notes")
    md_lines.append("")
    md_lines.append("- This target is a **headline verification** run: it reads the curated artifacts in `artifacts/` and writes `summary.json` + this `report.md`.")
    md_lines.append("- For full reruns (data acquisition + scoring), see `README_reproduce.md`.")
    md_lines.append("")

    _write_text(out_dir / "report.md", "\n".join(md_lines) + "\n")

    print(f"[done] wrote {out_dir / 'summary.json'}")
    print(f"[done] wrote {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

