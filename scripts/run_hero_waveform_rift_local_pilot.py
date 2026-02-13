#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RIFT_ROOT = ROOT / "outputs" / "forward_tests" / "hero_waveform_consistency_rift_latest"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run one local RIFT ILE pilot directly from command-single.sh with reduced workload."
    )
    ap.add_argument("--rift-root", type=Path, default=DEFAULT_RIFT_ROOT)
    ap.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Job id under <rift-root>/jobs. If omitted, choose first job.",
    )
    ap.add_argument("--n-max", type=int, default=20000, help="Override --n-max for pilot run.")
    ap.add_argument("--n-eff", type=int, default=10, help="Override --n-eff for pilot run.")
    ap.add_argument(
        "--events-to-analyze",
        type=int,
        default=1,
        help="Override --n-events-to-analyze for pilot run.",
    )
    ap.add_argument(
        "--cpu-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop GPU/vectorized flags for a CPU-only pilot run.",
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def choose_job(rift_root: Path, job_id: str | None) -> Path:
    jobs_root = rift_root / "jobs"
    jobs = sorted([p for p in jobs_root.iterdir() if p.is_dir()])
    if not jobs:
        raise FileNotFoundError(f"No jobs found in {jobs_root}")
    if job_id is None:
        return jobs[0]
    sel = jobs_root / job_id
    if not sel.exists():
        raise FileNotFoundError(f"Job not found: {sel}")
    return sel


def build_cmd(cmd_single_path: Path, n_max: int, n_eff: int, n_events: int, cpu_only: bool) -> str:
    lines = cmd_single_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    payload = ""
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        payload = ln
        break
    if not payload:
        raise ValueError(f"No executable command found in {cmd_single_path}")

    payload = re.sub(r"--n-max\s+\d+", f"--n-max {int(n_max)}", payload)
    payload = re.sub(r"--n-eff\s+\d+", f"--n-eff {int(n_eff)}", payload)
    payload = re.sub(r"--n-events-to-analyze\s+\d+", f"--n-events-to-analyze {int(n_events)}", payload)
    if cpu_only:
        for flag in ["--gpu", "--vectorized", "--force-xpy"]:
            payload = payload.replace(f" {flag}", "")
            payload = payload.replace(f"{flag} ", "")
            payload = payload.replace(flag, "")
    return payload


def main() -> None:
    args = parse_args()
    rift_root = args.rift_root.resolve()
    job_dir = choose_job(rift_root, args.job_id)
    rundir = job_dir / "rundir"
    cmd_single = rundir / "command-single.sh"
    if not cmd_single.exists():
        raise FileNotFoundError(f"Missing {cmd_single}")

    cmd = build_cmd(
        cmd_single,
        n_max=args.n_max,
        n_eff=args.n_eff,
        n_events=args.events_to_analyze,
        cpu_only=args.cpu_only,
    )

    print(f"[job] {job_dir.name}")
    print(f"[rundir] {rundir}")
    print(f"[cmd] {cmd}")
    if args.dry_run:
        return

    log = rundir / "pilot_local.log"
    with log.open("a", encoding="utf-8") as f:
        f.write(f"[pilot-start] job={job_dir.name}\n")
        f.write(cmd + "\n")
        f.flush()
        proc = subprocess.run(
            ["bash", "-lc", cmd],
            cwd=str(rundir),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        f.write(f"[pilot-end] rc={proc.returncode}\n")

    print(f"[done] rc={proc.returncode} log={log}")


if __name__ == "__main__":
    main()
