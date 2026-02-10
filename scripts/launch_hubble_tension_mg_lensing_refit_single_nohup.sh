#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/launch_hubble_tension_mg_lensing_refit_single_nohup.sh <phase>
  scripts/launch_hubble_tension_mg_lensing_refit_single_nohup.sh <out_root> <phase>

Phases:
  smoke | pilot | full

Examples:
  scripts/launch_hubble_tension_mg_lensing_refit_single_nohup.sh smoke
  scripts/launch_hubble_tension_mg_lensing_refit_single_nohup.sh pilot
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
out_root=""

if [ "${1:-}" = "smoke" ] || [ "${1:-}" = "pilot" ] || [ "${1:-}" = "full" ]; then
  phase="$1"
  out_root="outputs/hubble_tension_mg_lensing_refit_${phase}_${timestamp}"
else
  out_root="$1"
  phase="${2:-pilot}"
fi

case "$phase" in
  smoke|pilot|full) ;;
  *)
    echo "ERROR: unknown phase '$phase' (expected smoke|pilot|full)." >&2
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

# Keep a conservative default to avoid host-level resource fights.
default_max_core=31
if [ "$n_total" -le $((default_max_core + 1)) ]; then
  cpuset_default="0-$((n_total - 1))"
else
  cpuset_default="0-$default_max_core"
fi
cpuset="${CPUSET:-$cpuset_default}"

run_dir="${RUN_DIR:-outputs/finalization/highpower_multistart_v2/M0_start101}"
draws=64
heartbeat_sec=20
seed="${SEED:-0}"

case "$phase" in
  smoke)
    draws=8
    heartbeat_sec=8
    ;;
  pilot)
    draws=64
    heartbeat_sec=20
    ;;
  full)
    draws=160
    heartbeat_sec=30
    ;;
esac

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
  "run_dir": "$run_dir",
  "cpuset": "$cpuset",
  "n_total_cores": $n_total,
  "draws": $draws,
  "seed": $seed
}
JSON

cat > "$job_sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$repo_root"
echo "[job] started \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PYTHONPATH=src .venv/bin/python scripts/run_hubble_tension_mg_lensing_refit.py \\
  --run-dir "$run_dir" \\
  --out "$out_root" \\
  --draws "$draws" \\
  --seed "$seed" \\
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

echo "[launcher] started phase=$phase"
echo "[launcher] out_root=$out_root"
echo "[launcher] pid=$pid (written to $pid_file)"
echo "[launcher] run_log=$run_log"
echo "[launcher] status: tail -n 80 \"$run_log\""
