#!/usr/bin/env bash
set -euo pipefail
cd "/home/primary/PROJECT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH=src
echo "[job] started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
.venv/bin/python scripts/run_joint_transfer_bias_fit.py \
  --out "outputs/joint_transfer_bias_fit_full_20260209_061958UTC" \
  --draws-total "8192" \
  --theta-samples "32768" \
  --theta-chunk "512" \
  --workers "48" \
  --seed "0" \
  --o3-delta-lpd "3.669945265" \
  --heartbeat-sec "60" \
  --resume \
  ${MAX_CHUNKS:+--max-chunks "0"} \
  
echo "[job] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
