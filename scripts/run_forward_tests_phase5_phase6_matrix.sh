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
out_root="${1:-outputs/forward_tests/phase5_phase6_matrix_${timestamp}}"
mkdir -p "$out_root"

log="$out_root/run.log"
manifest="$out_root/manifest.json"
summary="$out_root/summary.json"
latest_summary="outputs/forward_tests/phase5_phase6_matrix_latest.json"

cat > "$manifest" <<JSON
{
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "repo_root": "$repo_root",
  "python_bin": "$py",
  "thread_env": {
    "OMP_NUM_THREADS": "${OMP_NUM_THREADS}",
    "MKL_NUM_THREADS": "${MKL_NUM_THREADS}",
    "OPENBLAS_NUM_THREADS": "${OPENBLAS_NUM_THREADS}",
    "NUMEXPR_NUM_THREADS": "${NUMEXPR_NUM_THREADS}"
  },
  "scenarios": [
    {
      "id": "baseline",
      "phase5_out": "outputs/forward_tests/phase5_external_constraints",
      "phase6_out": "outputs/forward_tests/phase6_model_selection",
      "phase5_args": []
    },
    {
      "id": "assumed_highz",
      "phase5_out": "outputs/forward_tests/phase5_external_constraints_assumed_highz",
      "phase6_out": "outputs/forward_tests/phase6_model_selection_assumed_highz",
      "phase5_args": ["--coverage-mode", "assumed_highz", "--assumed-zmax", "1100"]
    }
  ]
}
JSON

run_phase5() {
  local out="$1"
  shift
  echo "[phase5] out=$out args=$*" | tee -a "$log"
  "$py" scripts/run_forward_tests_phase5_external_constraints.py --out "$out" "$@" | tee -a "$log"
}

run_phase6() {
  local out="$1"
  local phase5_summary="$2"
  echo "[phase6] out=$out phase5_summary=$phase5_summary" | tee -a "$log"
  "$py" scripts/run_forward_tests_phase6_model_selection.py \
    --phase5-summary "$phase5_summary" \
    --out "$out" | tee -a "$log"
}

echo "[matrix] start $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log"

# Scenario 1: baseline
run_phase5 "outputs/forward_tests/phase5_external_constraints"
run_phase6 \
  "outputs/forward_tests/phase6_model_selection" \
  "outputs/forward_tests/phase5_external_constraints/tables/summary.json"

# Scenario 2: assumed high-z coverage sensitivity
run_phase5 "outputs/forward_tests/phase5_external_constraints_assumed_highz" \
  --coverage-mode assumed_highz \
  --assumed-zmax 1100
run_phase6 \
  "outputs/forward_tests/phase6_model_selection_assumed_highz" \
  "outputs/forward_tests/phase5_external_constraints_assumed_highz/tables/summary.json"

echo "[matrix] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$log"

"$py" - <<'PY' > "$summary"
import json
from pathlib import Path

def load(path: str):
    return json.loads(Path(path).read_text())

def check_pass(summary: dict, name: str):
    for section in ("strict_gate", "available_data_gate"):
        gate = summary.get(section, {})
        checks = gate.get("checks", [])
        if isinstance(checks, list):
            for c in checks:
                if isinstance(c, dict) and str(c.get("name")) == name:
                    return bool(c.get("pass", False))
    return None

scenarios = [
    ("baseline", "outputs/forward_tests/phase5_external_constraints/tables/summary.json", "outputs/forward_tests/phase6_model_selection/tables/summary.json"),
    ("assumed_highz", "outputs/forward_tests/phase5_external_constraints_assumed_highz/tables/summary.json", "outputs/forward_tests/phase6_model_selection_assumed_highz/tables/summary.json"),
]

rows = []
for sid, p5p, p6p in scenarios:
    p5 = load(p5p)
    p6 = load(p6p)
    rows.append(
        {
            "scenario": sid,
            "phase5_full_pass": bool(p5["strict_gate"]["pass"]),
            "phase5_available_data_pass": bool(p5["available_data_gate"]["pass"]),
            "phase5_primary_cmb_modeled": bool(p5.get("scope_checks", {}).get("primary_cmb_modeled", False)),
            "phase5_early_time_coverage_satisfied": check_pass(p5, "early_time_coverage_satisfied"),
            "phase6_pass": bool(p6["strict_gate"]["pass"]),
            "phase6_leading_model": str(p6["decision"]["leading_proxy_model"]),
            "phase6_target_M2_supported": bool(p6["decision"]["target_hypothesis_M2_supported"]),
            "phase6_target_M2_supported_available_data": bool(p6["decision"]["target_hypothesis_M2_supported_with_available_data_gate"]),
            "phase6_target_M2_supported_material_relief": bool(p6["decision"].get("target_hypothesis_M2_supported_with_material_relief", False)),
            "phase6_full_blockers": list(p6["decision"]["target_hypothesis_blockers"]),
        }
    )

print(json.dumps({"rows": rows}, indent=2, sort_keys=True))
PY

cp "$summary" "$latest_summary"

echo "[matrix] summary=$summary"
echo "[matrix] log=$log"
echo "[matrix] latest_summary=$latest_summary"
