#!/usr/bin/env bash
set -euo pipefail
cd "/home/primary/O3-Modified-Gravity-Hubble-Resolution"
echo "[job] started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
PYTHONPATH=src .venv/bin/python scripts/run_hubble_tension_early_universe_bias.py \
  --run-dir "outputs/finalization/highpower_multistart_v2/M0_start101" \
  --out "outputs/hubble_tension_early_universe_bias_fixed_smoke_20260210_062143UTC" \
  --draws "256" \
  --seed "0" \
  --theta-noise-frac "0.0" \
  --lensing-noise "0.0" \
  --n-int "6000" \
  --omega-m-assumed-mode "fixed_planck" \
  --r-d-assumed-mode planck_fixed \
  --heartbeat-sec "10"
echo "[job] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
