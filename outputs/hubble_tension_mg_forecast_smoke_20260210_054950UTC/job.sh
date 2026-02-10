#!/usr/bin/env bash
set -euo pipefail
cd "/home/primary/O3-Modified-Gravity-Hubble-Resolution"
echo "[job] started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[job] phase=smoke out_root=outputs/hubble_tension_mg_forecast_smoke_20260210_054950UTC run_dir=outputs/finalization/highpower_multistart_v2/M0_start101"
PYTHONPATH=src .venv/bin/python scripts/run_hubble_tension_mg_forecast.py   --run-dir "outputs/finalization/highpower_multistart_v2/M0_start101"   --out "outputs/hubble_tension_mg_forecast_smoke_20260210_054950UTC"   --draws "1024"   --seed 0   --z-max "0.62"   --z-n "160"   --z-anchors "0.2,0.35,0.5,0.62"   --n-rep "500"   --sigma-highz-frac "0.01"   --local-mode "external"   --h0-local-ref "73.0"   --h0-local-sigma "1.0"   --h0-planck-ref "67.4"   --h0-planck-sigma "0.5"   --omega-m-planck "0.315"   --gr-omega-mode "sample"   --gr-omega-fixed "0.315"   --heartbeat-sec "10"   --resume
echo "[job] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
