#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "outputs" / "forward_tests" / "hero_waveform_consistency_prod_latest"
BILBY_PIPE_ANALYSIS = Path("/home/primary/PROJECT/.venv/bin/bilby_pipe_analysis")
VENV_PYTHON = Path("/home/primary/PROJECT/.venv/bin/python")
ACTIVE_CFG_RE = re.compile(r"(/[^ ]+/jobs/([^/ ]+)/config_complete\.ini)")
PROGRESS_RE = re.compile(
    r"(?P<it>\d+)it\s+\[[^\n]*?ncall:(?P<ncall>[0-9.e+\-]+)[^\n]*?dlogz:(?P<dlogz>[0-9.]+)>"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run all remaining (non-success) hero waveform jobs sequentially, one at a time, "
            "using all requested cores per job."
        )
    )
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--cores", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument(
        "--stall-minutes",
        type=float,
        default=25.0,
        help="Mark a run stalled if sampler progress makes no progress for this many minutes.",
    )
    ap.add_argument(
        "--stall-grace-minutes",
        type=float,
        default=10.0,
        help="Do not apply stall logic until this many minutes after launch.",
    )
    ap.add_argument(
        "--max-restarts-per-job",
        type=int,
        default=2,
        help="Auto-restart attempts per job for retryable non-success exits.",
    )
    ap.add_argument(
        "--restart-backoff-sec",
        type=int,
        default=5,
        help="Sleep this long between automatic retries.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show remaining queue and exit without launching jobs.",
    )
    return ap.parse_args()


def iter_job_dirs(run_root: Path) -> list[Path]:
    jobs_dir = run_root / "jobs"
    if not jobs_dir.exists():
        raise FileNotFoundError(f"Missing jobs dir: {jobs_dir}")
    return sorted([p for p in jobs_dir.iterdir() if p.is_dir()])


def job_finished_ok(job_dir: Path) -> bool:
    exit_path = job_dir / "exit.code"
    if exit_path.exists():
        try:
            if int(exit_path.read_text(encoding="utf-8").strip()) == 0:
                return True
        except Exception:
            pass
    return any((job_dir / "run_out" / "result").glob("*result.hdf5"))


def read_active_jobs(run_root: Path) -> dict[str, int]:
    out = subprocess.check_output(["bash", "-lc", "ps -eo pid=,ppid=,args="], text=True)
    run_key = str(run_root.resolve())
    per_job: dict[str, list[tuple[int, int]]] = {}
    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        args = parts[2]
        if "bilby_pipe_analysis" not in args:
            continue
        if run_key not in args:
            continue
        m = ACTIVE_CFG_RE.search(args)
        if not m:
            continue
        job_id = m.group(2)
        per_job.setdefault(job_id, []).append((pid, ppid))

    chosen: dict[str, int] = {}
    for job_id, recs in per_job.items():
        ppid1 = [pid for pid, ppid in recs if ppid == 1]
        if ppid1:
            chosen[job_id] = min(ppid1)
        else:
            chosen[job_id] = min(pid for pid, _ in recs)
    return chosen


def replace_key(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^({re.escape(key)})=.*$")
    new_text, n = pattern.subn(rf"\1={value}", text, count=1)
    if n == 0:
        raise ValueError(f"Missing key in config: {key}")
    return new_text


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
    data["print_method"] = "interval-60"
    data["check_point_delta_t"] = 120
    return pattern.sub(f"sampler-kwargs={repr(data)}", text, count=1)


def patch_config(config_path: Path, cores: int) -> None:
    text = config_path.read_text(encoding="utf-8")
    text = replace_key(text, "request-cpus", str(cores))
    text = replace_key(text, "n-parallel", "1")
    text = patch_sampler_kwargs(text, npool=cores)
    config_path.write_text(text, encoding="utf-8")


def extract_data_dump(job_dir: Path) -> str:
    run_sh = job_dir / "run.sh"
    if not run_sh.exists():
        raise FileNotFoundError(f"Missing run.sh for {job_dir.name}: {run_sh}")
    blob = run_sh.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'--data-dump-file "([^"]+)"', blob)
    if m is None:
        raise ValueError(f"Could not parse --data-dump-file from {run_sh}")
    return m.group(1)


def _find_resume_pickle(job_dir: Path) -> Path | None:
    files = sorted((job_dir / "run_out" / "result").glob("*_resume.pickle"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _read_resume_progress(resume_pickle: Path) -> dict | None:
    if not resume_pickle.exists():
        return None
    code = r"""
import json
import pickle
import sys

obj = pickle.load(open(sys.argv[1], "rb"))
sampler = obj[0] if isinstance(obj, tuple) and len(obj) > 0 else obj
it = int(getattr(sampler, "it", -1))
ncall = int(getattr(sampler, "ncall", -1))
eff = float(getattr(sampler, "eff", float("nan")))
print(json.dumps({"it": it, "ncall": ncall, "eff": eff}))
"""
    try:
        out = subprocess.check_output(
            [str(VENV_PYTHON), "-c", code, str(resume_pickle)],
            text=True,
            timeout=20,
        )
        data = json.loads(out.strip())
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def _read_log_progress(run_log: Path) -> dict | None:
    if not run_log.exists():
        return None
    try:
        size = run_log.stat().st_size
        start = max(0, size - 600_000)
        with run_log.open("rb") as f:
            f.seek(start)
            blob = f.read().decode("utf-8", errors="ignore")
    except Exception:
        return None
    matches = list(PROGRESS_RE.finditer(blob))
    if not matches:
        return None
    m = matches[-1]
    try:
        it = int(m.group("it"))
        ncall = int(float(m.group("ncall")))
        dlogz = float(m.group("dlogz"))
    except Exception:
        return None
    return {"it": it, "ncall": ncall, "dlogz": dlogz}


def run_job(
    job_dir: Path,
    cores: int,
    poll_sec: int,
    stall_minutes: float,
    stall_grace_minutes: float,
) -> tuple[int, str]:
    config_path = job_dir / "config_complete.ini"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config for {job_dir.name}: {config_path}")
    patch_config(config_path, cores=cores)
    data_dump = extract_data_dump(job_dir)

    core_range = f"0-{cores - 1}"
    run_log = job_dir / "run.log"
    exit_path = job_dir / "exit.code"
    pid_path = job_dir / "pid.txt"
    exit_path.unlink(missing_ok=True)
    stall_sec = max(0.0, stall_minutes) * 60.0
    grace_sec = max(0.0, stall_grace_minutes) * 60.0

    with run_log.open("a", encoding="utf-8", errors="ignore") as lf:
        lf.write(
            f"[queue-launch] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"cores={core_range} print=interval-60\n"
        )
        lf.flush()

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"

        cmd = [
            "taskset",
            "-c",
            core_range,
            str(BILBY_PIPE_ANALYSIS),
            str(config_path),
            "--data-dump-file",
            data_dump,
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
        pid_path.write_text(str(proc.pid), encoding="utf-8")
        start_ts = time.time()
        last_progress_ts = start_ts
        last_it = -1
        last_ncall = -1
        last_source = "none"
        reason = "completed"

        while True:
            rc = proc.poll()
            if rc is not None:
                break

            # Progress source 1: dynesty progress lines in run.log (most responsive).
            log_prog = _read_log_progress(run_log)
            if log_prog is not None:
                it = int(log_prog.get("it", -1))
                ncall = int(log_prog.get("ncall", -1))
                if it > last_it or ncall > last_ncall:
                    last_it = it
                    last_ncall = ncall
                    last_progress_ts = time.time()
                    last_source = "run_log"
                    print(
                        f"[queue] progress {job_dir.name}: "
                        f"it={it} ncall={ncall} dlogz={log_prog.get('dlogz'):.3f} source=run_log"
                    )
                    lf.flush()

            # Progress source 2: resume checkpoint (coarser cadence, but independent confirmation).
            resume_pickle = _find_resume_pickle(job_dir)
            if resume_pickle is not None:
                prog = _read_resume_progress(resume_pickle)
                if prog is not None:
                    it = int(prog.get("it", -1))
                    ncall = int(prog.get("ncall", -1))
                    if it > last_it or ncall > last_ncall:
                        last_it = it
                        last_ncall = ncall
                        last_progress_ts = time.time()
                        last_source = "resume_pickle"
                        print(
                            f"[queue] progress {job_dir.name}: "
                            f"it={it} ncall={ncall} eff={prog.get('eff'):.4f} source=resume_pickle"
                        )
                        lf.flush()

            now = time.time()
            if stall_sec > 0 and (now - start_ts) >= grace_sec and (now - last_progress_ts) >= stall_sec:
                reason = "stalled_no_sampler_progress"
                print(
                    f"[queue] stall detected for {job_dir.name}: "
                    f"no sampler progress for {stall_minutes:.1f}m; "
                    f"last_it={last_it} last_ncall={last_ncall} source={last_source}; terminating pid={proc.pid}"
                )
                try:
                    proc.terminate()
                    proc.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=30)
                rc = proc.returncode if proc.returncode is not None else 124
                break

            time.sleep(poll_sec)

    exit_path.write_text(str(rc), encoding="utf-8")
    return rc, reason


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    if args.cores < 1:
        raise ValueError("--cores must be >= 1")
    if args.poll_sec < 1:
        raise ValueError("--poll-sec must be >= 1")
    if not BILBY_PIPE_ANALYSIS.exists():
        raise FileNotFoundError(f"Missing bilby_pipe_analysis: {BILBY_PIPE_ANALYSIS}")
    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"Missing venv python: {VENV_PYTHON}")
    if args.max_restarts_per_job < 0:
        raise ValueError("--max-restarts-per-job must be >= 0")

    jobs = iter_job_dirs(run_root)
    targets = [j for j in jobs if not job_finished_ok(j)]
    print(f"[queue] run_root={run_root}")
    print(
        f"[queue] cores={args.cores} poll_sec={args.poll_sec} "
        f"stall_minutes={args.stall_minutes} stall_grace_minutes={args.stall_grace_minutes} "
        f"max_restarts_per_job={args.max_restarts_per_job}"
    )
    print(f"[queue] initial_remaining={len(targets)}")
    if not targets:
        print("[queue] nothing to do")
        return
    if args.dry_run:
        print("[queue] dry-run targets:")
        for idx, jd in enumerate(targets, start=1):
            print(f"[queue] {idx}/{len(targets)} {jd.name}")
        return

    done_ok = 0
    done_fail = 0
    skipped_now_ok = 0
    auto_retries = 0

    for idx, job_dir in enumerate(targets, start=1):
        job_id = job_dir.name
        print(f"[queue] target {idx}/{len(targets)}: {job_id}")

        while True:
            active = read_active_jobs(run_root)
            if not active:
                break
            active_list = ",".join(sorted(active.keys()))
            print(f"[queue] waiting for active job(s): {active_list}")
            time.sleep(args.poll_sec)

        if job_finished_ok(job_dir):
            skipped_now_ok += 1
            print(f"[queue] skip already successful: {job_id}")
            continue

        attempt = 0
        retryable_codes = {124, 137, 143, -9, -15, 15}
        while True:
            print(f"[queue] start: {job_id} attempt={attempt + 1}")
            rc, reason = run_job(
                job_dir,
                cores=args.cores,
                poll_sec=args.poll_sec,
                stall_minutes=args.stall_minutes,
                stall_grace_minutes=args.stall_grace_minutes,
            )
            ok = job_finished_ok(job_dir)
            if ok:
                done_ok += 1
                print(f"[queue] done ok: {job_id} rc={rc} reason={reason}")
                break

            retryable = (reason == "stalled_no_sampler_progress") or (rc in retryable_codes)
            if retryable and attempt < args.max_restarts_per_job:
                attempt += 1
                auto_retries += 1
                print(
                    f"[queue] retrying {job_id}: rc={rc} reason={reason} "
                    f"attempt={attempt}/{args.max_restarts_per_job}"
                )
                if args.restart_backoff_sec > 0:
                    time.sleep(args.restart_backoff_sec)
                continue

            done_fail += 1
            print(f"[queue] done fail: {job_id} rc={rc} reason={reason}")
            break

    remaining = [j.name for j in iter_job_dirs(run_root) if not job_finished_ok(j)]
    print("[queue] complete")
    print(
        f"[queue] summary started={len(targets)} done_ok={done_ok} "
        f"done_fail={done_fail} skipped_now_ok={skipped_now_ok} "
        f"auto_retries={auto_retries} remaining={len(remaining)}"
    )
    if remaining:
        print("[queue] remaining_jobs=" + ",".join(remaining))


if __name__ == "__main__":
    main()
