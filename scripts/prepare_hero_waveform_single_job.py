#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import os
import re
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "outputs" / "forward_tests" / "hero_waveform_consistency_prod_latest"
VENV_BIN = Path("/home/primary/PROJECT/.venv/bin")
BILBY_PIPE_ANALYSIS = VENV_BIN / "bilby_pipe_analysis"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare a strict single-job waveform run (one job at a time, one launcher process, "
            "all requested cores assigned to that single job)."
        )
    )
    ap.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help=f"Production run root. Default: {DEFAULT_RUN_ROOT}",
    )
    ap.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Specific job id under <run-root>/jobs. If omitted, first non-success job is selected.",
    )
    ap.add_argument(
        "--cores",
        type=int,
        default=os.cpu_count() or 1,
        help="Cores to assign to this one job (default: visible CPU count).",
    )
    return ap.parse_args()


def iter_job_dirs(run_root: Path) -> Iterable[Path]:
    jobs_dir = run_root / "jobs"
    if not jobs_dir.exists():
        raise FileNotFoundError(f"Missing jobs dir: {jobs_dir}")
    for p in sorted(jobs_dir.iterdir()):
        if p.is_dir():
            yield p


def job_finished_ok(job_dir: Path) -> bool:
    exit_code = job_dir / "exit.code"
    if exit_code.exists():
        try:
            return int(exit_code.read_text(encoding="utf-8").strip()) == 0
        except Exception:
            pass
    return any((job_dir / "run_out" / "result").glob("*result.hdf5"))


def choose_job(run_root: Path, requested_job_id: str | None) -> Path:
    jobs = list(iter_job_dirs(run_root))
    if requested_job_id is not None:
        job_dir = run_root / "jobs" / requested_job_id
        if job_dir not in jobs:
            raise FileNotFoundError(f"Job not found: {job_dir}")
        return job_dir

    pending = [j for j in jobs if not job_finished_ok(j)]
    if not pending:
        raise RuntimeError("No pending/failed jobs found; everything appears finished.")
    return pending[0]


def replace_key(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^({re.escape(key)})=.*$")
    replaced, n = pattern.subn(rf"\1={value}", text, count=1)
    if n == 0:
        raise ValueError(f"Missing key in config: {key}")
    return replaced


def patch_sampler_kwargs(text: str, npool: int) -> str:
    pattern = re.compile(r"(?m)^sampler-kwargs=(.*)$")
    m = pattern.search(text)
    if m is None:
        raise ValueError("Missing sampler-kwargs in config.")
    raw = m.group(1).strip()
    data = ast.literal_eval(raw)
    if not isinstance(data, dict):
        raise ValueError("sampler-kwargs is not a dict literal.")
    data["npool"] = int(npool)
    data["resume"] = True
    value = repr(data)
    return pattern.sub(f"sampler-kwargs={value}", text, count=1)


def patch_config(config_path: Path, cores: int) -> None:
    original = config_path.read_text(encoding="utf-8")
    backup = config_path.with_suffix(config_path.suffix + ".bak_single")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")

    text = original
    text = replace_key(text, "request-cpus", str(cores))
    text = replace_key(text, "n-parallel", "1")
    text = patch_sampler_kwargs(text, npool=cores)
    config_path.write_text(text, encoding="utf-8")


def extract_data_dump(job_run_sh: Path) -> str:
    blob = job_run_sh.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'--data-dump-file "([^"]+)"', blob)
    if m is None:
        raise ValueError(f"Could not find --data-dump-file in {job_run_sh}")
    return m.group(1)


def write_single_runner(job_dir: Path, config_path: Path, data_dump: str, cores: int) -> Path:
    if cores < 1:
        raise ValueError("cores must be >= 1")
    core_range = f"0-{cores - 1}"
    run_root = job_dir.parents[1]
    lock_path = run_root / ".single_waveform.lock"
    pid_path = run_root / "single_waveform_active.pid"

    run_single = job_dir / "run_single_all_cores.sh"
    run_single.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                f'LOCK="{lock_path}"',
                f'PIDFILE="{pid_path}"',
                'if [ -e "$LOCK" ]; then',
                '  echo "[error] lock exists: $LOCK" >&2',
                "  exit 90",
                "fi",
                'echo "$$" > "$LOCK"',
                'echo "$$" > "$PIDFILE"',
                'cleanup() { rm -f "$LOCK" "$PIDFILE"; }',
                "trap cleanup EXIT INT TERM",
                "",
                f'cd "{job_dir}"',
                "export OMP_NUM_THREADS=1",
                "export MKL_NUM_THREADS=1",
                "export OPENBLAS_NUM_THREADS=1",
                "export NUMEXPR_NUM_THREADS=1",
                "",
                'echo "[start] $(date -u +%Y-%m-%dT%H:%M:%SZ)" > status_single.log',
                'echo "[cores] using taskset ' + core_range + '" >> status_single.log',
                "set +e",
                (
                    f'taskset -c {core_range} "{BILBY_PIPE_ANALYSIS}" '
                    f'"{config_path}" --data-dump-file "{data_dump}" > run_single.log 2>&1'
                ),
                "ec=$?",
                'echo "[end] $(date -u +%Y-%m-%dT%H:%M:%SZ) exit=$ec" >> status_single.log',
                'echo "$ec" > exit.single.code',
                "exit $ec",
                "",
            ]
        ),
        encoding="utf-8",
    )
    os.chmod(run_single, 0o775)
    return run_single


def any_active_wave_jobs(run_root: Path) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    run_key = str(run_root.resolve())
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        cmdline = pid_dir / "cmdline"
        try:
            cmd = cmdline.read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore").strip()
        except Exception:
            continue
        if not cmd:
            continue
        if "bilby_pipe_analysis" in cmd and run_key in cmd:
            matches.append((int(pid_dir.name), cmd))
        elif "/tmp/wave_resume_worker_v2.sh" in cmd and run_key in cmd:
            matches.append((int(pid_dir.name), cmd))
        elif "/tmp/wave_queue_controller_v2.py" in cmd or "/tmp/wave_queue_controller_v3.py" in cmd:
            matches.append((int(pid_dir.name), cmd))
    return sorted(matches)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()

    if not BILBY_PIPE_ANALYSIS.exists():
        raise FileNotFoundError(f"Missing bilby_pipe_analysis: {BILBY_PIPE_ANALYSIS}")
    if args.cores < 1:
        raise ValueError("--cores must be >= 1")

    active = any_active_wave_jobs(run_root)
    if active:
        raise RuntimeError(
            "Active waveform processes detected; refuse to prepare start state until fully idle:\n"
            + "\n".join(f"{pid} {cmd}" for pid, cmd in active)
        )

    job_dir = choose_job(run_root, args.job_id)
    config_path = job_dir / "config_complete.ini"
    run_sh = job_dir / "run.sh"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not run_sh.exists():
        raise FileNotFoundError(f"Missing run.sh: {run_sh}")

    patch_config(config_path, cores=args.cores)
    data_dump = extract_data_dump(run_sh)
    runner = write_single_runner(job_dir, config_path, data_dump, cores=args.cores)

    print("[ok] single-job mode prepared")
    print(f"[ok] run_root: {run_root}")
    print(f"[ok] job_id: {job_dir.name}")
    print(f"[ok] config patched: {config_path}")
    print(f"[ok] runner script: {runner}")
    print(f"[ok] start command: bash {runner}")
    print("[ok] safety: n-parallel=1 and lock file prevents concurrent waveform jobs")


if __name__ == "__main__":
    main()

