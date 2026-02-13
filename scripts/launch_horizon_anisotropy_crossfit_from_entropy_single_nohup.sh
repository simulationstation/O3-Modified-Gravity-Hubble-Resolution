#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/launch_horizon_anisotropy_crossfit_from_entropy_single_nohup.sh [phase]
  scripts/launch_horizon_anisotropy_crossfit_from_entropy_single_nohup.sh <entropy_run_root> [phase]
  scripts/launch_horizon_anisotropy_crossfit_from_entropy_single_nohup.sh <entropy_summary_json> [phase]

Phases:
  smoke | pilot | full

Examples:
  scripts/launch_horizon_anisotropy_crossfit_from_entropy_single_nohup.sh pilot
  scripts/launch_horizon_anisotropy_crossfit_from_entropy_single_nohup.sh outputs/entropy_submission_hardening_20260210_175840UTC full

Notes:
  - The entropy run must be complete enough to contain:
      <run>/realdata_long_mapvar_256c/tables/summary.json
    or pass that summary.json path directly.
  - Set DRY_RUN=1 to write manifest/job.sh without launching.
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [ ! -x ".venv/bin/python" ]; then
  echo "ERROR: missing .venv/bin/python" >&2
  exit 2
fi
if [ ! -f "scripts/run_horizon_anisotropy_scan.py" ]; then
  echo "ERROR: missing scripts/run_horizon_anisotropy_scan.py" >&2
  exit 2
fi

phase="pilot"
entropy_ref=""

if [ "${1:-}" = "smoke" ] || [ "${1:-}" = "pilot" ] || [ "${1:-}" = "full" ]; then
  phase="$1"
else
  entropy_ref="${1:-}"
  phase="${2:-pilot}"
fi

case "$phase" in
  smoke|pilot|full) ;;
  *)
    echo "ERROR: unknown phase '$phase' (expected smoke|pilot|full)." >&2
    exit 2
    ;;
esac

if [ -z "$entropy_ref" ]; then
  entropy_ref="$(ls -dt outputs/entropy_submission_hardening_* 2>/dev/null | head -n1 || true)"
fi
if [ -z "$entropy_ref" ]; then
  echo "ERROR: no entropy run found under outputs/entropy_submission_hardening_*." >&2
  exit 2
fi

summary_json=""
if [ -f "$entropy_ref" ]; then
  summary_json="$entropy_ref"
elif [ -f "$entropy_ref/tables/summary.json" ]; then
  summary_json="$entropy_ref/tables/summary.json"
elif [ -f "$entropy_ref/realdata_long_mapvar_256c/tables/summary.json" ]; then
  summary_json="$entropy_ref/realdata_long_mapvar_256c/tables/summary.json"
fi

if [ -z "$summary_json" ] || [ ! -f "$summary_json" ]; then
  echo "ERROR: could not locate entropy summary.json from '$entropy_ref'." >&2
  echo "Expected one of:" >&2
  echo "  <path>/tables/summary.json" >&2
  echo "  <path>/realdata_long_mapvar_256c/tables/summary.json" >&2
  exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required by this launcher." >&2
  exit 2
fi

z_min="$(jq -r '.z_domain.z_min // empty' "$summary_json")"
z_max="$(jq -r '.z_domain.z_max // empty' "$summary_json")"
subset="cosmology"
cov_kind="$(jq -r '.settings.sn_cov_kind // "stat+sys"' "$summary_json")"
z_column="$(jq -r '.settings.sn_z_column // "zHD"' "$summary_json")"

if [ -z "$z_min" ] || [ -z "$z_max" ]; then
  echo "ERROR: summary missing z_domain in $summary_json" >&2
  exit 2
fi

n_total="$(nproc)"
if [ "$n_total" -lt 1 ]; then
  n_total=1
fi
cpuset="${CPUSET:-0-$((n_total - 1))}"

# Resource defaults
mu_procs="${MU_PROCS:-1}"
train_axis_jobs="${TRAIN_AXIS_JOBS:-32}"
test_mu_procs="${TEST_MU_PROCS:-64}"

# Phase knobs (based on previous HAC runs)
kfold=5
train_axes_nside=4
train_mu_sampler="emcee"
train_mu_walkers=24
train_mu_knots=2
train_mu_steps=250
train_mu_burn=80
train_mu_draws=100
test_mu_knots=6
test_mu_steps=5000
test_mu_burn=1500
test_mu_draws=1500
test_parallel_fore_aft=1
test_min_sn_per_side=80

case "$phase" in
  smoke)
    kfold=2
    train_axes_nside=2
    train_axis_jobs="${TRAIN_AXIS_JOBS:-8}"
    train_mu_steps=120
    train_mu_burn=40
    train_mu_draws=60
    test_mu_steps=800
    test_mu_burn=250
    test_mu_draws=250
    test_mu_procs="${TEST_MU_PROCS:-16}"
    ;;
  pilot)
    kfold=5
    ;;
  full)
    kfold=5
    train_axis_jobs="${TRAIN_AXIS_JOBS:-64}"
    test_mu_steps=7000
    test_mu_burn=2200
    test_mu_draws=2200
    test_mu_procs="${TEST_MU_PROCS:-96}"
    ;;
esac

timestamp="$(date -u +%Y%m%d_%H%M%SUTC)"
out_root="${OUT_ROOT:-outputs/horizon_anisotropy_from_entropy_${phase}_${timestamp}}"
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
  "cpuset": "$cpuset",
  "n_total_cores": $n_total,
  "entropy_ref": "$entropy_ref",
  "entropy_summary_json": "$summary_json",
  "z_min": $z_min,
  "z_max": $z_max,
  "subset": "$subset",
  "cov_kind": "$cov_kind",
  "z_column": "$z_column",
  "kfold": $kfold,
  "train_axes_nside": $train_axes_nside,
  "train_axis_jobs": $train_axis_jobs,
  "mu_procs": $mu_procs,
  "test_mu_procs": $test_mu_procs,
  "train_mu_sampler": "$train_mu_sampler",
  "train_mu_walkers": $train_mu_walkers,
  "train_mu_knots": $train_mu_knots,
  "train_mu_steps": $train_mu_steps,
  "train_mu_burn": $train_mu_burn,
  "train_mu_draws": $train_mu_draws,
  "test_mu_knots": $test_mu_knots,
  "test_mu_steps": $test_mu_steps,
  "test_mu_burn": $test_mu_burn,
  "test_mu_draws": $test_mu_draws,
  "test_parallel_fore_aft": $( [ "$test_parallel_fore_aft" -eq 1 ] && echo true || echo false ),
  "test_min_sn_per_side": $test_min_sn_per_side
}
JSON

cat > "$job_sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$repo_root"
echo "[job] started \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python scripts/run_horizon_anisotropy_scan.py \\
  --mode crossfit \\
  --out "$out_root" \\
  --seed 123 \\
  --subset "$subset" --cov-kind "$cov_kind" --z-column "$z_column" \\
  --z-min "$z_min" --z-max "$z_max" \\
  --match-z bin_downsample --match-z-bin-width 0.05 --match-mode survey_z \\
  --sn-like-bin-width 0.05 --sn-like-min-per-bin 20 \\
  --kfold "$kfold" --split-z-bin-width 0.05 \\
  --train-axes-nside "$train_axes_nside" --train-axes-nest --train-skip-antipodes \\
  --train-axis-jobs "$train_axis_jobs" \\
  --train-mu-sampler "$train_mu_sampler" \\
  --train-mu-walkers "$train_mu_walkers" \\
  --train-mu-knots "$train_mu_knots" --train-mu-steps "$train_mu_steps" --train-mu-burn "$train_mu_burn" --train-mu-draws "$train_mu_draws" \\
  --train-mu-procs 1 \\
  --mu-sampler ptemcee --pt-ntemps 6 --pt-tmax 20 \\
  --mu-walkers 48 --mu-procs "$mu_procs" \\
  --test-mu-knots "$test_mu_knots" --test-mu-steps "$test_mu_steps" --test-mu-burn "$test_mu_burn" --test-mu-draws "$test_mu_draws" \\
  --test-mu-procs "$test_mu_procs" \\
  --test-min-sn-per-side "$test_min_sn_per_side" \\
  $( [ "$test_parallel_fore_aft" -eq 1 ] && echo "--test-parallel-fore-aft" )
echo "[job] finished \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF
chmod +x "$job_sh"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[launcher] dry-run only (DRY_RUN=1)"
  echo "[launcher] entropy_ref=$entropy_ref"
  echo "[launcher] summary_json=$summary_json"
  echo "[launcher] out_root=$out_root"
  echo "[launcher] job_sh=$job_sh"
  echo "[launcher] manifest=$manifest"
  exit 0
fi

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
echo "[launcher] entropy_ref=$entropy_ref"
echo "[launcher] summary_json=$summary_json"
echo "[launcher] out_root=$out_root"
echo "[launcher] pid=$pid"
echo "[launcher] run_log=$run_log"
echo "[launcher] status: tail -n 80 \"$run_log\""
