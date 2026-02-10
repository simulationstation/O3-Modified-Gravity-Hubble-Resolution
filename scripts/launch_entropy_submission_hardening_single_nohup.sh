#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/launch_entropy_submission_hardening_single_nohup.sh
  scripts/launch_entropy_submission_hardening_single_nohup.sh <out_root>

Environment controls (optional):
  PROJECT_ROOT=/home/primary/PROJECT
  CPU_CORES=256

  LONG_WALKERS=256
  LONG_STEPS=1200
  LONG_BURN=400
  LONG_DRAWS=800
  LONG_GP_WALKERS=128
  LONG_GP_STEPS=500
  LONG_GP_BURN=160
  LONG_MU_CHUNK_STEPS=20
  LONG_TIMING_EVERY=20
  LONG_CHECKPOINT_EVERY=0
  MG_UNBIASED=0

  ABL_GP_WALKERS=96
  ABL_GP_STEPS=800
  ABL_GP_BURN=250

  SBC_N=24
  SBC_PROCS=24
  SBC_WALKERS=96
  SBC_STEPS=700
  SBC_BURN=240
  SBC_DRAWS=400
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

timestamp="$(date -u +%Y%m%d_%H%M%SUTC)"
out_root="${1:-outputs/entropy_submission_hardening_${timestamp}}"

project_root="${PROJECT_ROOT:-/home/primary/PROJECT}"
py="$project_root/.venv/bin/python"

if [ ! -x "$py" ]; then
  echo "ERROR: missing python executable: $py" >&2
  exit 2
fi
if [ ! -f "$project_root/scripts/run_realdata_recon.py" ]; then
  echo "ERROR: missing $project_root/scripts/run_realdata_recon.py" >&2
  exit 2
fi

cpu_cores="${CPU_CORES:-256}"

long_walkers="${LONG_WALKERS:-256}"
long_steps="${LONG_STEPS:-1200}"
long_burn="${LONG_BURN:-400}"
long_draws="${LONG_DRAWS:-800}"
long_gp_walkers="${LONG_GP_WALKERS:-128}"
long_gp_steps="${LONG_GP_STEPS:-500}"
long_gp_burn="${LONG_GP_BURN:-160}"
long_mu_chunk_steps="${LONG_MU_CHUNK_STEPS:-20}"
long_timing_every="${LONG_TIMING_EVERY:-20}"
long_checkpoint_every="${LONG_CHECKPOINT_EVERY:-20}"
mg_unbiased="${MG_UNBIASED:-0}"
mg_unbiased_norm="$(printf '%s' "$mg_unbiased" | tr '[:upper:]' '[:lower:]')"
mg_unbiased_flag=""
mg_unbiased_json="false"
if [ "$mg_unbiased_norm" = "1" ] || [ "$mg_unbiased_norm" = "true" ] || [ "$mg_unbiased_norm" = "yes" ] || [ "$mg_unbiased_norm" = "on" ]; then
  mg_unbiased_flag="--mg-unbiased"
  mg_unbiased_json="true"
fi

abl_gp_walkers="${ABL_GP_WALKERS:-96}"
abl_gp_steps="${ABL_GP_STEPS:-800}"
abl_gp_burn="${ABL_GP_BURN:-250}"

sbc_n="${SBC_N:-24}"
sbc_procs="${SBC_PROCS:-24}"
sbc_walkers="${SBC_WALKERS:-96}"
sbc_steps="${SBC_STEPS:-700}"
sbc_burn="${SBC_BURN:-240}"
sbc_draws="${SBC_DRAWS:-400}"

if [ "$long_checkpoint_every" -gt 0 ]; then
  echo "[launcher] checkpointing enabled (interval=$long_checkpoint_every)." >&2
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
  "project_root": "$project_root",
  "out_root": "$out_root",
  "cpu_cores": $cpu_cores,
  "long": {
    "walkers": $long_walkers,
    "steps": $long_steps,
    "burn": $long_burn,
    "draws": $long_draws,
    "gp_walkers": $long_gp_walkers,
    "gp_steps": $long_gp_steps,
    "gp_burn": $long_gp_burn,
    "mu_chunk_steps": $long_mu_chunk_steps,
    "timing_every": $long_timing_every,
    "checkpoint_every": $long_checkpoint_every,
    "mg_unbiased": $mg_unbiased_json
  },
  "ablation": {
    "gp_walkers": $abl_gp_walkers,
    "gp_steps": $abl_gp_steps,
    "gp_burn": $abl_gp_burn
  },
  "synthetic": {
    "sbc_n": $sbc_n,
    "sbc_procs": $sbc_procs,
    "walkers": $sbc_walkers,
    "steps": $sbc_steps,
    "burn": $sbc_burn,
    "draws": $sbc_draws
  }
}
JSON

cat > "$job_sh" <<JOB
#!/usr/bin/env bash
set -euo pipefail
cd "$repo_root"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$project_root/src"

collector="$repo_root/scripts/collect_entropy_submission_hardening_table.sh"

echo "[job] start \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[job] out_root=$out_root"

run_stage() {
  local name="\$1"
  shift
  echo "[stage] \$name START \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "\$@"
  echo "[stage] \$name END \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "\$collector" "$out_root"
}

run_stage realdata_long_mapvar \
  "$py" "$project_root/scripts/run_realdata_recon.py" \
  --out "$repo_root/$out_root/realdata_long_mapvar_256c" \
  --seed 60221 \
  --cpu-cores $cpu_cores \
  --mu-walkers $long_walkers --mu-steps $long_steps --mu-burn $long_burn --mu-draws $long_draws \
  --mu-procs $long_walkers \
  --mu-chunk-steps $long_mu_chunk_steps \
  --gp-walkers $long_gp_walkers --gp-steps $long_gp_steps --gp-burn $long_gp_burn \
  --gp-procs $long_gp_walkers \
  --spline-bootstrap 96 \
  --max-rss-mb 1024 \
  --skip-ablations \
  --run-mapping-variants \
  --include-rsd --rsd-mode dr12+dr16_fsbao \
  --include-planck-lensing-clpp --clpp-backend scaled --clpp-dataset consext8 \
  --include-fullshape-pk --pk-dataset shapefit_lrgz1_ngc_mono \
  $mg_unbiased_flag \
  --checkpoint-every $long_checkpoint_every \
  --timing-log "$repo_root/$out_root/realdata_long_mapvar_256c/timing.jsonl" \
  --timing-every $long_timing_every

run_stage ablation_suite \
  "$py" "$project_root/scripts/run_ablation_suite.py" \
  --out "$repo_root/$out_root/ablation_suite" \
  --seed 60222 \
  --z-max 0.62 \
  --n-knots 18 \
  --n-grid 220 \
  --gp-walkers $abl_gp_walkers \
  --gp-steps $abl_gp_steps \
  --gp-burn $abl_gp_burn \
  --cpu-cores 1

run_stage synthetic_closure_bh_sbc \
  "$py" "$project_root/scripts/run_synthetic_closure.py" \
  --out "$repo_root/$out_root/synthetic_closure_bh_sbc" \
  --seed 60223 \
  --model bh \
  --z-max 0.62 \
  --walkers $sbc_walkers --steps $sbc_steps --burn $sbc_burn --draws $sbc_draws \
  --procs $sbc_walkers \
  --cpu-cores $cpu_cores \
  --sbc-n $sbc_n \
  --sbc-procs $sbc_procs \
  --noise-scale-range 0.9 1.1 \
  --max-rss-mb 1024

echo "[job] done \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"\$collector" "$out_root"
JOB

chmod +x "$job_sh"

setsid bash "$job_sh" > "$run_log" 2>&1 < /dev/null &
pid="$!"
echo "$pid" > "$pid_file"

# initial empty table for convenience
scripts/collect_entropy_submission_hardening_table.sh "$out_root" >/dev/null 2>&1 || true

echo "[launcher] started"
echo "[launcher] out_root=$out_root"
echo "[launcher] pid=$pid"
echo "[launcher] run_log=$run_log"
echo "[launcher] summary: $out_root/submission_hardening_summary.md"
echo "[launcher] live: tail -n 80 $run_log"
