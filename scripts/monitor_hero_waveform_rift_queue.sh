#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-outputs/forward_tests/hero_waveform_consistency_rift_latest}"
PID_FILE="$ROOT/queue_nohup.pid"
LATEST_QUEUE_LOG="$(ls -1t "$ROOT"/queue_run_*.log 2>/dev/null | head -n 1 || true)"

if [[ -n "$LATEST_QUEUE_LOG" ]]; then
  total=$(grep -c '^\[job-start\]' "$LATEST_QUEUE_LOG" || true)
  ok=$(grep -c '^\[job-ok\]' "$LATEST_QUEUE_LOG" || true)
  fail=$(grep -c '^\[job-fail\]' "$LATEST_QUEUE_LOG" || true)
  running=$((total - ok - fail))
else
  total=0; ok=0; fail=0; running=0
fi

echo "root: $ROOT"
echo "queue_log: ${LATEST_QUEUE_LOG:-none}"
if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "queue_pid: $pid (running)"
    ps -p "$pid" -o pid,etime,%cpu,%mem,cmd --no-headers
  else
    echo "queue_pid: $pid (not running)"
  fi
else
  echo "queue_pid: none"
fi

echo "progress: total_started=$total ok=$ok fail=$fail in_flight=$running"

if [[ -n "$LATEST_QUEUE_LOG" ]]; then
  last_start=$(grep '^\[job-start\]' "$LATEST_QUEUE_LOG" | tail -n 1 || true)
  if [[ -n "$last_start" ]]; then
    current_job=$(echo "$last_start" | awk '{print $3}')
    echo "current_job: $current_job"
    stamp=$(basename "$LATEST_QUEUE_LOG" | sed -E 's/^queue_run_([0-9_]+)\.log$/\1/')
    runlog="$ROOT/jobs/$current_job/rundir/run_local_${stamp}.log"
    if [[ -f "$runlog" ]]; then
      echo "current_log: $runlog"
      echo "----- current log tail -----"
      tail -n 20 "$runlog" || true
    else
      echo "current_log: (not created yet)"
    fi
  fi
  echo "----- queue log tail -----"
  tail -n 20 "$LATEST_QUEUE_LOG" || true
fi
