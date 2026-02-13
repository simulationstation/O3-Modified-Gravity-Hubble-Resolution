#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-outputs/forward_tests/hero_waveform_consistency_rift_latest}"
JOBS_DIR="$ROOT/jobs"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$ROOT/queue_run_${STAMP}.log"

if [[ ! -d "$JOBS_DIR" ]]; then
  echo "[error] jobs dir not found: $JOBS_DIR" >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-256}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "[start] $(date -Is)" | tee -a "$MASTER_LOG"
echo "[root] $ROOT" | tee -a "$MASTER_LOG"
echo "[omp] OMP_NUM_THREADS=$OMP_NUM_THREADS" | tee -a "$MASTER_LOG"

n_total=0
n_ok=0
n_fail=0

while IFS= read -r job; do
  [[ -z "$job" ]] && continue
  n_total=$((n_total+1))
  name="$(basename "$job")"
  rundir="$job/rundir"
  cmd="$rundir/command-single.sh"
  log="$rundir/run_local_${STAMP}.log"

  echo "[job-start] $(date -Is) $name" | tee -a "$MASTER_LOG"
  if [[ ! -x "$cmd" ]]; then
    chmod +x "$cmd" || true
  fi

  set +e
  (
    cd "$rundir"
    bash ./command-single.sh
  ) >"$log" 2>&1
  rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    # Some RIFT runs return rc=0 while internally skipping/failed analysis.
    if rg -q 'FAILED ANALYSIS|Nyquist frequency error|Internal function call failed|Traceback \(most recent call last\)|XLAL Error' "$log"; then
      n_fail=$((n_fail+1))
      echo "[job-fail] $(date -Is) $name rc=0(log-failure) log=$log" | tee -a "$MASTER_LOG"
    else
      n_ok=$((n_ok+1))
      echo "[job-ok] $(date -Is) $name log=$log" | tee -a "$MASTER_LOG"
    fi
  else
    n_fail=$((n_fail+1))
    echo "[job-fail] $(date -Is) $name rc=$rc log=$log" | tee -a "$MASTER_LOG"
    # continue to next job; do not abort entire queue
  fi
done < <(find "$JOBS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

echo "[done] $(date -Is) total=$n_total ok=$n_ok fail=$n_fail" | tee -a "$MASTER_LOG"
