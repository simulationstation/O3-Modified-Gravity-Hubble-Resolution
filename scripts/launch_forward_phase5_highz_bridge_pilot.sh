#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/launch_forward_phase5_highz_bridge_pilot.sh
  scripts/launch_forward_phase5_highz_bridge_pilot.sh <out_root>

Environment overrides:
  PROJECT_ROOT=/home/primary/PROJECT
  PYTHON_BIN=/home/primary/PROJECT/.venv/bin/python
  CPU_CORES=64                 # keep <=128 per project constraint
  Z_MAX_CAP=1.2
  MU_WALKERS=32
  MU_STEPS=220
  MU_BURN=80
  MU_DRAWS=220
  MU_PROCS=32
  MU_CHUNK_STEPS=20
  GP_WALKERS=32
  GP_STEPS=180
  GP_BURN=60
  GP_PROCS=16
  MAX_RSS_MB=1024
  SEED=77201

This launcher starts a single low-impact pilot run in the background (nohup-like)
and writes pid/log/manifest under out_root.
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

project_root="${PROJECT_ROOT:-/home/primary/PROJECT}"
py="${PYTHON_BIN:-$project_root/.venv/bin/python}"
recon_py="$project_root/scripts/run_realdata_recon.py"

if [ ! -x "$py" ]; then
  echo "ERROR: missing python executable: $py" >&2
  exit 2
fi
if [ ! -f "$recon_py" ]; then
  echo "ERROR: missing run script: $recon_py" >&2
  exit 2
fi

timestamp="$(date -u +%Y%m%d_%H%M%SUTC)"
out_root="${1:-outputs/forward_tests/highz_bridge_pilot_${timestamp}}"
mkdir -p "$out_root"

cpu_cores="${CPU_CORES:-64}"
z_max_cap="${Z_MAX_CAP:-1.2}"
mu_walkers="${MU_WALKERS:-32}"
mu_steps="${MU_STEPS:-220}"
mu_burn="${MU_BURN:-80}"
mu_draws="${MU_DRAWS:-220}"
mu_procs="${MU_PROCS:-32}"
mu_chunk_steps="${MU_CHUNK_STEPS:-20}"
gp_walkers="${GP_WALKERS:-32}"
gp_steps="${GP_STEPS:-180}"
gp_burn="${GP_BURN:-60}"
gp_procs="${GP_PROCS:-16}"
max_rss_mb="${MAX_RSS_MB:-1024}"
seed="${SEED:-77201}"

if [ "$cpu_cores" -gt 128 ]; then
  echo "ERROR: CPU_CORES must be <=128 (got $cpu_cores)." >&2
  exit 2
fi

manifest="$out_root/launcher_manifest.json"
job="$out_root/job.sh"
log="$out_root/run.log"
pid_file="$out_root/pid.txt"

cat > "$manifest" <<JSON
{
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "repo_root": "$repo_root",
  "project_root": "$project_root",
  "python_bin": "$py",
  "run_script": "$recon_py",
  "out_root": "$out_root",
  "config": {
    "cpu_cores": $cpu_cores,
    "z_max_cap": $z_max_cap,
    "mu_walkers": $mu_walkers,
    "mu_steps": $mu_steps,
    "mu_burn": $mu_burn,
    "mu_draws": $mu_draws,
    "mu_procs": $mu_procs,
    "mu_chunk_steps": $mu_chunk_steps,
    "gp_walkers": $gp_walkers,
    "gp_steps": $gp_steps,
    "gp_burn": $gp_burn,
    "gp_procs": $gp_procs,
    "max_rss_mb": $max_rss_mb,
    "seed": $seed
  }
}
JSON

cat > "$job" <<JOB
#!/usr/bin/env bash
set -euo pipefail
cd "$repo_root"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "[job] start \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$py" "$recon_py" \
  --out "$repo_root/$out_root/highz_bridge_recon" \
  --seed $seed \
  --cpu-cores $cpu_cores \
  --z-max-cap $z_max_cap \
  --mu-walkers $mu_walkers --mu-steps $mu_steps --mu-burn $mu_burn --mu-draws $mu_draws \
  --mu-procs $mu_procs \
  --mu-chunk-steps $mu_chunk_steps \
  --gp-walkers $gp_walkers --gp-steps $gp_steps --gp-burn $gp_burn --gp-procs $gp_procs \
  --run-mapping-variants \
  --include-rsd --rsd-mode dr12+dr16_fsbao \
  --include-planck-lensing-clpp --clpp-backend scaled --clpp-dataset consext8 \
  --max-rss-mb $max_rss_mb \
  --skip-ablations
echo "[job] done \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
JOB

chmod +x "$job"

setsid bash "$job" > "$log" 2>&1 < /dev/null &
pid="$!"
echo "$pid" > "$pid_file"

echo "[launcher] started"
echo "[launcher] out_root=$out_root"
echo "[launcher] pid=$pid"
echo "[launcher] run_log=$log"
echo "[launcher] manifest=$manifest"
echo "[launcher] tail: tail -n 80 $log"
