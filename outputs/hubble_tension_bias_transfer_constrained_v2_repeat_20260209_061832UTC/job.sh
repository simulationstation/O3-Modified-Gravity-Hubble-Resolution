#!/usr/bin/env bash
set -euo pipefail
cd /home/primary/PROJECT
echo "[job] started \2026-02-09T06:18:32Z"
PYTHONPATH=src .venv/bin/python scripts/run_hubble_tension_bias_transfer_sweep.py \
  --out-root "outputs/hubble_tension_bias_transfer_constrained_v2_repeat_20260209_061832UTC" \
  --run-dirs "outputs/finalization/highpower_multistart_v2/M0_start101,outputs/finalization/highpower_multistart_v2/M0_start202,outputs/finalization/highpower_multistart_v2/M0_start303,outputs/finalization/highpower_multistart_v2/M0_start404,outputs/finalization/highpower_multistart_v2/M0_start505" \
  --highz-bias-fracs=-0.003,0.0,0.003 \
  --local-biases=-0.25,0.0,0.25 \
  --draws 8192 \
  --n-rep 100000 \
  --seed0 19000 \
  --z-max 0.62 \
  --z-n 320 \
  --z-anchors "0.2,0.35,0.5,0.62" \
  --sigma-highz-frac 0.01 \
  --local-mode external \
  --h0-local-ref 73.0 \
  --h0-local-sigma 1.0 \
  --h0-planck-ref 67.4 \
  --h0-planck-sigma 0.5 \
  --omega-m-planck 0.315 \
  --gr-omega-mode sample \
  --gr-omega-fixed 0.315 \
  --heartbeat-sec 30
echo "[job] finished \2026-02-09T06:18:32Z"
