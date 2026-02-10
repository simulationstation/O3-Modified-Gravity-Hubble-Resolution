#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/launch_hubble_tension_early_universe_bias_single_nohup.sh <phase> [mode]
  scripts/launch_hubble_tension_early_universe_bias_single_nohup.sh <out_root> <phase> [mode]

Phases:
  smoke | pilot | full

Modes:
  fixed   - Omega_m fixed to Planck reference in GR inversion
  lensing - Omega_m inferred from lensing-amplitude proxy

Examples:
  scripts/launch_hubble_tension_early_universe_bias_single_nohup.sh smoke fixed
  scripts/launch_hubble_tension_early_universe_bias_single_nohup.sh pilot lensing
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ "$#" -eq 0 ]; then
  usage
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

timestamp="$(date -u +%Y%m%d_%H%M%SUTC)"
phase="pilot"
mode="fixed"
out_root=""

if [ "${1:-}" = "smoke" ] || [ "${1:-}" = "pilot" ] || [ "${1:-}" = "full" ]; then
  phase="$1"
  mode="${2:-fixed}"
  out_root="outputs/hubble_tension_early_universe_bias_${mode}_${phase}_${timestamp}"
else
  out_root="$1"
  phase="${2:-pilot}"
  mode="${3:-fixed}"
fi

case "$phase" in
  smoke|pilot|full) ;;
  *)
    echo "ERROR: unknown phase '$phase' (expected smoke|pilot|full)." >&2
    exit 2
    ;;
esac

case "$mode" in
  fixed|lensing) ;;
  *)
    echo "ERROR: unknown mode '$mode' (expected fixed|lensing)." >&2
    exit 2
    ;;
esac

if [ ! -x ".venv/bin/python" ]; then
  echo "ERROR: missing .venv/bin/python (create venv + install deps first)." >&2
  exit 2
fi

n_total="$(nproc)"
if [ "$n_total" -lt 1 ]; then
  n_total=1
fi
cpuset="${CPUSET:-0-$((n_total - 1))}"

run_dir="${RUN_DIR:-outputs/finalization/highpower_multistart_v2/M0_start101}"
draws=1024
heartbeat_sec=30
theta_noise_frac="${THETA_NOISE_FRAC:-0.0}"
lensing_noise="${LENSING_NOISE:-0.0}"
n_int="${N_INT:-6000}"
seed="${SEED:-0}"

case "$phase" in
  smoke)
    draws=256
    heartbeat_sec=10
    ;;
  pilot)
    draws=1024
    heartbeat_sec=30
    ;;
  full)
    draws=1600
    heartbeat_sec=60
    ;;
esac

omega_mode="fixed_planck"
if [ "$mode" = "lensing" ]; then
  omega_mode="from_lensing_proxy"
fi

mkdir -p "$out_root"
job_sh="$out_root/job.sh"
run_log="$out_root/run.log"
pid_file="$out_root/pid.txt"
manifest="$out_root/launcher_manifest.json"

cat > "$manifest" <<JSON
{
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "repo_root": "$repo_root",
  "out_root": "$out_root",
  "phase": "$phase",
  "mode": "$mode",
  "run_dir": "$run_dir",
  "cpuset": "$cpuset",
  "n_total_cores": $n_total,
  "draws": $draws,
  "seed": $seed,
  "theta_noise_frac": $theta_noise_frac,
  "lensing_noise": $lensing_noise,
  "n_int": $n_int,
  "omega_m_assumed_mode": "$omega_mode"
}
JSON

cat > "$job_sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$repo_root"
echo "[job] started \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PYTHONPATH=src .venv/bin/python scripts/run_hubble_tension_early_universe_bias.py \\
  --run-dir "$run_dir" \\
  --out "$out_root" \\
  --draws "$draws" \\
  --seed "$seed" \\
  --theta-noise-frac "$theta_noise_frac" \\
  --lensing-noise "$lensing_noise" \\
  --n-int "$n_int" \\
  --omega-m-assumed-mode "$omega_mode" \\
  --r-d-assumed-mode planck_fixed \\
  --heartbeat-sec "$heartbeat_sec"
echo "[job] finished \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF
chmod +x "$job_sh"

env \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PYTHONUNBUFFERED=1 \
  setsid taskset -c "$cpuset" bash "$job_sh" > "$run_log" 2>&1 < /dev/null &

pid="$!"
echo "$pid" > "$pid_file"

echo "[launcher] started phase=$phase mode=$mode"
echo "[launcher] out_root=$out_root"
echo "[launcher] pid=$pid (written to $pid_file)"
echo "[launcher] run_log=$run_log"
echo "[launcher] status: tail -n 80 \"$run_log\""
