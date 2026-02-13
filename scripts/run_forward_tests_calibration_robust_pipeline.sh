#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

py="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "$py" ]; then
  echo "ERROR: missing python executable: $py" >&2
  exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

timestamp="$(date -u +%Y%m%d_%H%M%SUTC)"
out_root="${1:-outputs/forward_tests/calibration_robust_${timestamp}}"
mkdir -p "$out_root"

log="$out_root/run.log"
summary="$out_root/summary.json"
latest_summary="outputs/forward_tests/calibration_robust_latest.json"

echo "[calib-robust] start $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log"
echo "[calib-robust] out_root=$out_root" | tee -a "$log"

p2_out="$out_root/phase2_growth_s8"
p3_out="$out_root/phase3_mu_sigma"
p4_out="$out_root/phase4_distance_ratio"
p5_out="$out_root/phase5_external_constraints"
p6_out="$out_root/phase6_model_selection"

echo "[phase2] calibration-robust" | tee -a "$log"
"$py" scripts/run_forward_tests_phase2_growth_s8.py \
  --out "$p2_out" \
  --calibration-robust \
  --calibration-samples 5000 \
  --gate-min-robust-pass-frac 0.60 | tee -a "$log"

echo "[phase3] calibration-robust" | tee -a "$log"
"$py" scripts/run_forward_tests_phase3_mu_sigma.py \
  --growth-summary "$p2_out/tables/summary.json" \
  --out "$p3_out" \
  --calibration-robust \
  --calibration-samples 5000 \
  --gate-min-robust-pass-frac 0.60 | tee -a "$log"

echo "[phase4] calibration-robust" | tee -a "$log"
"$py" scripts/run_forward_tests_phase4_distance_ratio.py \
  --out "$p4_out" \
  --calibration-robust \
  --calibration-samples 5000 \
  --gate-min-robust-pass-frac 0.60 | tee -a "$log"

echo "[phase5] calibration-robust" | tee -a "$log"
"$py" scripts/run_forward_tests_phase5_external_constraints.py \
  --phase3-summary "$p3_out/tables/summary.json" \
  --phase4-summary "$p4_out/tables/summary.json" \
  --out "$p5_out" \
  --calibration-robust \
  --calibration-samples 5000 \
  --gate-min-robust-pass-frac 0.60 | tee -a "$log"

echo "[phase6] calibration-robust stability scorecard" | tee -a "$log"
"$py" scripts/run_forward_tests_phase6_model_selection.py \
  --phase2-summary "$p2_out/tables/summary.json" \
  --phase3-summary "$p3_out/tables/summary.json" \
  --phase4-summary "$p4_out/tables/summary.json" \
  --phase5-summary "$p5_out/tables/summary.json" \
  --out "$p6_out" \
  --calibration-robust \
  --calibration-samples 20000 \
  --gate-min-bias-stability 0.60 | tee -a "$log"

OUT_ROOT="$out_root" "$py" - <<'PY' > "$summary"
import json
import os
from pathlib import Path

def load(path: str):
    return json.loads(Path(path).read_text())

root = Path(os.environ["OUT_ROOT"]).resolve()
p2 = load(str(root / "phase2_growth_s8" / "tables" / "summary.json"))
p3 = load(str(root / "phase3_mu_sigma" / "tables" / "summary.json"))
p4 = load(str(root / "phase4_distance_ratio" / "tables" / "summary.json"))
p5 = load(str(root / "phase5_external_constraints" / "tables" / "summary.json"))
p6 = load(str(root / "phase6_model_selection" / "tables" / "summary.json"))

row = {
    "out_root": str(root),
    "phase2_strict_pass": bool(p2["strict_gate"]["pass"]),
    "phase2_robust_pass": bool(p2.get("calibration_robust_gate", {}).get("pass", False)),
    "phase2_robust_pass_fraction": float(p2.get("calibration_robust_gate", {}).get("pass_fraction", 0.0)),
    "phase3_strict_pass": bool(p3["strict_gate"]["pass"]),
    "phase3_robust_pass": bool(p3.get("calibration_robust_gate", {}).get("pass", False)),
    "phase3_robust_pass_fraction": float(p3.get("calibration_robust_gate", {}).get("pass_fraction", 0.0)),
    "phase4_strict_pass": bool(p4["strict_gate"]["pass"]),
    "phase4_robust_pass": bool(p4.get("calibration_robust_gate", {}).get("pass", False)),
    "phase4_robust_pass_fraction": float(p4.get("calibration_robust_gate", {}).get("pass_fraction", 0.0)),
    "phase5_strict_pass": bool(p5["strict_gate"]["pass"]),
    "phase5_available_pass": bool(p5["available_data_gate"]["pass"]),
    "phase5_robust_pass": bool(p5.get("calibration_robust_gate", {}).get("pass", False)),
    "phase5_robust_pass_fraction": float(p5.get("calibration_robust_gate", {}).get("full_pass_fraction", 0.0)),
    "phase5_robust_available_pass_fraction": float(
        p5.get("calibration_robust_gate", {}).get("available_data_pass_fraction", 0.0)
    ),
    "phase6_strict_pass": bool(p6["strict_gate"]["pass"]),
    "phase6_leading_model": str(p6["decision"]["leading_proxy_model"]),
    "phase6_target_M2_supported": bool(p6["decision"]["target_hypothesis_M2_supported"]),
    "phase6_material_relief_gate": bool(p6["material_relief_gate"]["pass"]),
    "phase6_bias_stability_score_m2": float(p6.get("decision", {}).get("bias_stability_score_m2", 0.0)),
    "phase6_calibration_robust_pass": bool(p6.get("calibration_robust_gate", {}).get("pass", False)),
    "phase6_leading_model_by_stability": str(p6.get("decision", {}).get("leading_model_by_stability", "n/a")),
}
print(json.dumps({"row": row}, indent=2, sort_keys=True))
PY

cp "$summary" "$latest_summary"

echo "[calib-robust] summary=$summary"
echo "[calib-robust] latest_summary=$latest_summary"
echo "[calib-robust] log=$log"
echo "[calib-robust] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
