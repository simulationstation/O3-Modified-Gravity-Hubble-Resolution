#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/launch_planck_global_mg_refit_single_nohup.sh <phase>
  scripts/launch_planck_global_mg_refit_single_nohup.sh <out_root> <phase>

Phases:
  smoke | pilot | full

Examples:
  scripts/launch_planck_global_mg_refit_single_nohup.sh smoke
  scripts/launch_planck_global_mg_refit_single_nohup.sh pilot

Environment controls:
  MODE=multistart|single          (default: multistart)
  CPUSET=0-31                     (default: conservative cap <= 32 logical cores)
  THREADS_PER_WORKER=1            (default: 1)
  WORKERS=<n>                     (default: auto from CPUSET)
  RESTARTS=<n>                    (default: phase default)
  MAX_EVALS=<n>                   (default: phase default)
  RHOEND=<float>                  (default: phase default)
  MONITOR_SEC=<seconds>           (default: 20)
  PRINT_PROGRESS=1|0              (default: 0)
  RESUME=1|0                      (default: 1)
  COVMAT_MODE=fixed|auto          (default: fixed)
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
  out_root="outputs/planck_global_mg_refit_${phase}_${timestamp}"
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

if [ ! -x ".venv/bin/cobaya-run" ]; then
  echo "ERROR: missing .venv/bin/cobaya-run (install Cobaya first)." >&2
  exit 2
fi
if [ ! -x ".venv/bin/python" ]; then
  echo "ERROR: missing .venv/bin/python (create venv + install deps first)." >&2
  exit 2
fi

config_src="configs/planck_2018_camb_mg_alens_minimize_pilot.yaml"
if [ ! -f "$config_src" ]; then
  echo "ERROR: missing config '$config_src'." >&2
  exit 2
fi

n_total="$(nproc)"
if [ "$n_total" -lt 1 ]; then
  n_total=1
fi
default_max_core=31
if [ "$n_total" -le $((default_max_core + 1)) ]; then
  cpuset_default="0-$((n_total - 1))"
else
  cpuset_default="0-$default_max_core"
fi
cpuset="${CPUSET:-$cpuset_default}"
mode="${MODE:-multistart}"
threads_per_worker="${THREADS_PER_WORKER:-1}"
workers="${WORKERS:-0}"
restarts="${RESTARTS:-0}"
max_evals="${MAX_EVALS:-0}"
rhoend="${RHOEND:--1}"
monitor_sec="${MONITOR_SEC:-20}"
print_progress="${PRINT_PROGRESS:-0}"
resume="${RESUME:-1}"
covmat_mode="${COVMAT_MODE:-fixed}"
covmat_file="${COVMAT_FILE:-external/cobaya_packages/data/planck_supp_data_and_covmats/covmats/base_Alens_plikHM_TTTEEE_lowE.covmat}"

case "$mode" in
  multistart|single) ;;
  *)
    echo "ERROR: unknown MODE '$mode' (expected multistart|single)." >&2
    exit 2
    ;;
esac

if [ "$mode" = "single" ]; then
  workers=1
  restarts=1
fi

mkdir -p "$out_root"
job_sh="$out_root/job.sh"
run_log="$out_root/run.log"
pid_file="$out_root/pid.txt"
manifest="$out_root/launcher_manifest.json"

worker_slots="$(
  .venv/bin/python - "$cpuset" "$threads_per_worker" <<'PY'
import sys
cpuset = sys.argv[1]
tpw = max(1, int(sys.argv[2]))
cores = set()
for part in cpuset.split(","):
    p = part.strip()
    if not p:
        continue
    if "-" in p:
        a, b = p.split("-", 1)
        a = int(a.strip())
        b = int(b.strip())
        if b < a:
            a, b = b, a
        for c in range(a, b + 1):
            cores.add(c)
    else:
        cores.add(int(p))
print(max(1, len(sorted(cores)) // tpw))
PY
)"

cat > "$manifest" <<JSON
{
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "repo_root": "$repo_root",
  "phase": "$phase",
  "mode": "$mode",
  "out_root": "$out_root",
  "cpuset": "$cpuset",
  "worker_slots_from_cpuset": $worker_slots,
  "threads_per_worker": $threads_per_worker,
  "workers_requested": "$workers",
  "restarts_requested": "$restarts",
  "max_evals_requested": "$max_evals",
  "rhoend_requested": "$rhoend",
  "monitor_sec": "$monitor_sec",
  "print_progress": "$print_progress",
  "resume": "$resume",
  "covmat_mode": "$covmat_mode",
  "covmat_file": "$covmat_file",
  "n_total_cores": $n_total,
  "config_source": "$config_src"
}
JSON

cat > "$job_sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$repo_root"
echo "[job] started \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[job] mode=$mode phase=$phase"
echo "[job] cpuset=$cpuset worker_slots=$worker_slots threads_per_worker=$threads_per_worker"
echo "[job] workers=$workers restarts=$restarts max_evals=$max_evals rhoend=$rhoend"
echo "[job] monitor_sec=$monitor_sec print_progress=$print_progress resume=$resume"
echo "[job] covmat_mode=$covmat_mode covmat_file=$covmat_file"
cmd=(
  .venv/bin/python
  scripts/run_planck_global_mg_refit_multistart.py
  --base-config "$config_src"
  --out "$out_root"
  --phase "$phase"
  --cpuset "$cpuset"
  --threads-per-worker "$threads_per_worker"
  --workers "$workers"
  --restarts "$restarts"
  --max-evals "$max_evals"
  --rhoend "$rhoend"
  --monitor-sec "$monitor_sec"
  --packages-path external/cobaya_packages
  --covmat-mode "$covmat_mode"
  --covmat-file "$covmat_file"
)
if [ "$print_progress" = "1" ]; then
  cmd+=(--print-progress)
else
  cmd+=(--no-print-progress)
fi
if [ "$resume" = "1" ]; then
  cmd+=(--resume)
else
  cmd+=(--no-resume)
fi
PYTHONPATH=src "\${cmd[@]}"
echo "[job] finished \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF
chmod +x "$job_sh"

env \
  PYTHONUNBUFFERED=1 \
  setsid bash "$job_sh" > "$run_log" 2>&1 < /dev/null &

pid="$!"
echo "$pid" > "$pid_file"

echo "[launcher] started phase=$phase"
echo "[launcher] mode=$mode"
echo "[launcher] out_root=$out_root"
echo "[launcher] pid=$pid (written to $pid_file)"
echo "[launcher] run_log=$run_log"
echo "[launcher] status: tail -n 80 \"$run_log\""
echo "[launcher] progress: tail -n 80 \"$out_root/monitor/monitor.log\""
