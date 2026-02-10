#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_cpuset(cpuset: str) -> list[int]:
    cores: set[int] = set()
    for part in cpuset.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a_raw, b_raw = chunk.split("-", 1)
            a = int(a_raw.strip())
            b = int(b_raw.strip())
            if b < a:
                a, b = b, a
            for c in range(a, b + 1):
                cores.add(c)
        else:
            cores.add(int(chunk))
    if not cores:
        raise ValueError(f"Failed to parse cpuset '{cpuset}'.")
    return sorted(cores)


def _partition_core_groups(cores: list[int], threads_per_worker: int) -> list[str]:
    if threads_per_worker <= 0:
        raise ValueError("threads_per_worker must be >= 1")
    n_groups = len(cores) // threads_per_worker
    if n_groups <= 0:
        raise ValueError(
            f"cpuset has {len(cores)} cores, insufficient for threads_per_worker={threads_per_worker}."
        )
    groups: list[str] = []
    for i in range(n_groups):
        start = i * threads_per_worker
        grp = cores[start : start + threads_per_worker]
        groups.append(",".join(str(c) for c in grp))
    return groups


def _phase_defaults(phase: str, workers: int) -> tuple[int, float, int]:
    if phase == "smoke":
        return (80, 0.20, max(2, workers))
    if phase == "pilot":
        # Keep pilot fast enough to yield completed restarts quickly.
        return (150, 0.20, max(workers, 8))
    if phase == "full":
        return (2000, 0.10, max(2 * workers, 16))
    raise ValueError(f"Unknown phase '{phase}'")


_RE_MIN = re.compile(r"-log\((?:posterior|likelihood)\)\s+minimized to\s+([0-9eE+.\-]+)")
_RE_OBJ = re.compile(r"Objective value f\(xmin\)\s*=\s*([0-9eE+.\-]+)")
_RE_FAIL_REASON = re.compile(r"Finished unsuccessfully\. Reason:\s*(.+)")


def _parse_log_metrics(log_path: Path) -> tuple[float | None, str | None]:
    if not log_path.exists():
        return (None, None)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    ms = list(_RE_MIN.finditer(text))
    if ms:
        return (float(ms[-1].group(1)), None)
    os_ = list(_RE_OBJ.finditer(text))
    objective = float(os_[-1].group(1)) if os_ else None
    fr = list(_RE_FAIL_REASON.finditer(text))
    reason = fr[-1].group(1).strip() if fr else None
    return (objective, reason)


@dataclass
class ActiveRun:
    restart_id: int
    slot_id: int
    core_group: str
    run_dir: Path
    config_path: Path
    output_prefix: Path
    log_path: Path
    process: subprocess.Popen[bytes]
    started_unix: float


def _build_run_config(
    *,
    base_cfg: dict[str, Any],
    output_prefix: str,
    packages_path: str,
    max_evals: int,
    rhoend: float,
    seed: int,
    print_progress: bool,
    covmat_mode: str,
    covmat_file: str,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    sampler = cfg.setdefault("sampler", {}).setdefault("minimize", {})
    sampler["method"] = "bobyqa"
    sampler["max_evals"] = int(max_evals)
    sampler["best_of"] = 1
    sampler["seed"] = int(seed)

    override = sampler.setdefault("override_bobyqa", {})
    override["rhoend"] = float(rhoend)
    override["print_progress"] = bool(print_progress)
    # Keep pybobyqa diagnostics disabled by default to avoid huge artifacts per restart.
    override["do_logging"] = False

    if covmat_mode == "fixed":
        sampler["covmat"] = str(covmat_file)
    else:
        sampler["covmat"] = "auto"

    cfg["packages_path"] = str(packages_path)
    cfg["output"] = str(output_prefix)
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run a parallel multistart Cobaya minimize campaign for the Planck+MG global refit. "
            "Designed to fill an assigned CPU set while providing heartbeat progress logs."
        )
    )
    ap.add_argument("--base-config", default="configs/planck_2018_camb_mg_alens_minimize_pilot.yaml")
    ap.add_argument("--out", default=None, help="Output directory (default: outputs/planck_global_mg_refit_multistart_<UTC>).")
    ap.add_argument("--phase", choices=["smoke", "pilot", "full"], default="pilot")
    ap.add_argument("--cpuset", required=True, help="CPU set for worker assignment, e.g. 0-31 or 0-15,32-47.")
    ap.add_argument("--threads-per-worker", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0, help="Parallel workers. 0 means use all worker slots from cpuset.")
    ap.add_argument("--restarts", type=int, default=0, help="Total multistart attempts. 0 means phase default.")
    ap.add_argument("--max-evals", type=int, default=0, help="Per-restart eval cap. 0 means phase default.")
    ap.add_argument("--rhoend", type=float, default=-1.0, help="BOBYQA rhoend. Negative means phase default.")
    ap.add_argument("--seed-base", type=int, default=1001)
    ap.add_argument("--monitor-sec", type=float, default=20.0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--print-progress", action="store_true", default=False)
    ap.add_argument("--no-print-progress", dest="print_progress", action="store_false")
    ap.add_argument("--packages-path", default="external/cobaya_packages")
    ap.add_argument("--cobaya-run-bin", default=".venv/bin/cobaya-run")
    ap.add_argument("--covmat-mode", choices=["fixed", "auto"], default="fixed")
    ap.add_argument(
        "--covmat-file",
        default="external/cobaya_packages/data/planck_supp_data_and_covmats/covmats/base_Alens_plikHM_TTTEEE_lowE.covmat",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    base_config_path = (repo_root / args.base_config).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Missing base config: {base_config_path}")

    cobaya_run_bin = (repo_root / args.cobaya_run_bin).resolve()
    if not cobaya_run_bin.exists():
        raise FileNotFoundError(f"Missing cobaya-run binary: {cobaya_run_bin}")

    if str(args.covmat_mode) == "fixed":
        covmat_path = (repo_root / str(args.covmat_file)).resolve()
        if not covmat_path.exists():
            raise FileNotFoundError(
                f"covmat file not found in fixed mode: {covmat_path}\n"
                "Set --covmat-mode auto or provide --covmat-file that exists."
            )

    cores = _parse_cpuset(args.cpuset)
    core_groups = _partition_core_groups(cores, int(args.threads_per_worker))
    worker_slots = len(core_groups)
    workers = int(args.workers) if int(args.workers) > 0 else worker_slots
    workers = max(1, min(workers, worker_slots))

    max_evals_def, rhoend_def, restarts_def = _phase_defaults(args.phase, workers)
    max_evals = int(args.max_evals) if int(args.max_evals) > 0 else int(max_evals_def)
    rhoend = float(args.rhoend) if float(args.rhoend) >= 0.0 else float(rhoend_def)
    restarts = int(args.restarts) if int(args.restarts) > 0 else int(restarts_def)

    out_dir = Path(args.out) if args.out else (repo_root / "outputs" / f"planck_global_mg_refit_multistart_{args.phase}_{_utc_stamp()}")
    out_dir = out_dir.resolve()
    monitor_dir = out_dir / "monitor"
    restart_root = out_dir / "restarts"
    table_dir = out_dir / "tables"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    restart_root.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    monitor_log = monitor_dir / "monitor.log"
    heartbeat_json = monitor_dir / "heartbeat.json"
    manifest_json = out_dir / "manifest.json"
    summary_json = out_dir / "summary.json"
    summary_csv = table_dir / "restart_summary.csv"

    with base_config_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    if not isinstance(base_cfg, dict):
        raise ValueError("Base config must be a YAML mapping at top level.")

    manifest = {
        "created_utc": _utc_now(),
        "repo_root": str(repo_root),
        "phase": str(args.phase),
        "base_config": str(base_config_path),
        "output_dir": str(out_dir),
        "cpuset": str(args.cpuset),
        "cores": cores,
        "core_groups": core_groups,
        "threads_per_worker": int(args.threads_per_worker),
        "workers": int(workers),
        "restarts": int(restarts),
        "max_evals": int(max_evals),
        "rhoend": float(rhoend),
        "seed_base": int(args.seed_base),
        "resume": bool(args.resume),
        "packages_path": str(args.packages_path),
        "covmat_mode": str(args.covmat_mode),
        "covmat_file": str(args.covmat_file),
        "print_progress": bool(args.print_progress),
    }
    _write_json_atomic(manifest_json, manifest)

    # Resume completed runs if requested.
    completed: dict[int, dict[str, Any]] = {}
    if bool(args.resume):
        for rid in range(restarts):
            result_path = restart_root / f"r{rid:03d}" / "result.json"
            if result_path.exists():
                try:
                    rec = _load_json(result_path)
                except Exception:
                    continue
                if bool(rec.get("finished", False)):
                    completed[rid] = rec

    pending = [rid for rid in range(restarts) if rid not in completed]
    results: list[dict[str, Any]] = [completed.get(i, {}) for i in range(restarts)]
    active: list[ActiveRun] = []
    slot_to_restart: dict[int, int] = {}

    t0 = time.time()
    last_heartbeat = 0.0

    def _append_monitor(msg: str) -> None:
        line = f"[{_utc_now()}] {msg}"
        with monitor_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def _emit_heartbeat(force: bool = False) -> None:
        nonlocal last_heartbeat
        now = time.time()
        if (not force) and (now - last_heartbeat < float(args.monitor_sec)):
            return
        last_heartbeat = now
        done_count = sum(1 for r in results if r)
        success_count = sum(1 for r in results if r and bool(r.get("success")))
        fail_count = sum(1 for r in results if r and (not bool(r.get("success"))))
        best_obj = None
        best_id = None
        for i, r in enumerate(results):
            if not r:
                continue
            obj = r.get("objective")
            if obj is None:
                continue
            objf = float(obj)
            if best_obj is None or objf < best_obj:
                best_obj = objf
                best_id = i
        hb = {
            "timestamp_utc": _utc_now(),
            "elapsed_sec": float(now - t0),
            "restarts_total": int(restarts),
            "restarts_done": int(done_count),
            "running": int(len(active)),
            "pending": int(len(pending)),
            "successes": int(success_count),
            "failures": int(fail_count),
            "best_objective": best_obj,
            "best_restart_id": best_id,
            "active": [
                {
                    "restart_id": int(a.restart_id),
                    "pid": int(a.process.pid),
                    "core_group": str(a.core_group),
                    "elapsed_sec": float(now - a.started_unix),
                }
                for a in active
            ],
        }
        _write_json_atomic(heartbeat_json, hb)
        _append_monitor(
            "heartbeat "
            + f"done={done_count}/{restarts} running={len(active)} pending={len(pending)} "
            + f"success={success_count} fail={fail_count} "
            + (f"best_obj={best_obj:.6f} rid={best_id}" if best_obj is not None else "best_obj=NA")
        )

    _append_monitor(
        f"start phase={args.phase} workers={workers} restarts={restarts} max_evals={max_evals} "
        + f"rhoend={rhoend:.4f} cpuset={args.cpuset} threads_per_worker={args.threads_per_worker}"
    )
    if completed:
        _append_monitor(f"resume: reused {len(completed)} completed restarts from prior run.")

    while pending or active:
        # Fill free worker slots.
        used_slots = set(slot_to_restart.keys())
        free_slots = [i for i in range(workers) if i not in used_slots]
        while pending and free_slots:
            rid = pending.pop(0)
            slot = free_slots.pop(0)
            core_group = core_groups[slot]

            run_dir = restart_root / f"r{rid:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg_path = run_dir / "run_config.yaml"
            log_path = run_dir / "run.log"
            output_prefix = run_dir / "minimize"
            lock_path = run_dir / "minimize.minimize.input.yaml.locked"

            cfg = _build_run_config(
                base_cfg=base_cfg,
                output_prefix=str(output_prefix),
                packages_path=str(args.packages_path),
                max_evals=max_evals,
                rhoend=rhoend,
                seed=int(args.seed_base) + int(rid),
                print_progress=bool(args.print_progress),
                covmat_mode=str(args.covmat_mode),
                covmat_file=str(args.covmat_file),
            )
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            if lock_path.exists():
                lock_path.unlink()

            env = os.environ.copy()
            threads = str(args.threads_per_worker)
            env["OMP_NUM_THREADS"] = threads
            env["MKL_NUM_THREADS"] = threads
            env["OPENBLAS_NUM_THREADS"] = threads
            env["NUMEXPR_NUM_THREADS"] = threads
            env["PYTHONUNBUFFERED"] = "1"

            log_fh = log_path.open("wb")
            cmd = [
                "taskset",
                "-c",
                core_group,
                str(cobaya_run_bin),
                "--no-mpi",
                "-p",
                str(args.packages_path),
                "-f",
                str(cfg_path),
            ]
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            # Keep handle attached to process object for cleanup when process exits.
            setattr(proc, "_codex_log_fh", log_fh)

            active.append(
                ActiveRun(
                    restart_id=rid,
                    slot_id=slot,
                    core_group=core_group,
                    run_dir=run_dir,
                    config_path=cfg_path,
                    output_prefix=output_prefix,
                    log_path=log_path,
                    process=proc,
                    started_unix=time.time(),
                )
            )
            slot_to_restart[slot] = rid
            _append_monitor(
                f"launch rid={rid} slot={slot} cores={core_group} pid={proc.pid} cfg={cfg_path.relative_to(repo_root)}"
            )

        # Poll active jobs.
        still_active: list[ActiveRun] = []
        for ar in active:
            rc = ar.process.poll()
            if rc is None:
                still_active.append(ar)
                continue

            log_fh = getattr(ar.process, "_codex_log_fh", None)
            if log_fh is not None:
                try:
                    log_fh.flush()
                    log_fh.close()
                except Exception:
                    pass

            objective, reason = _parse_log_metrics(ar.log_path)
            minimum_path = ar.run_dir / "minimize.minimum.txt"
            success = (int(rc) == 0) and minimum_path.exists()

            rec = {
                "restart_id": int(ar.restart_id),
                "slot_id": int(ar.slot_id),
                "core_group": str(ar.core_group),
                "pid": int(ar.process.pid),
                "return_code": int(rc),
                "success": bool(success),
                "objective": objective,
                "reason": reason if reason else (None if success else f"exit_code_{rc}"),
                "run_dir": str(ar.run_dir),
                "config_path": str(ar.config_path),
                "log_path": str(ar.log_path),
                "minimum_path": str(minimum_path),
                "finished": True,
                "started_utc": datetime.fromtimestamp(ar.started_unix, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "finished_utc": _utc_now(),
                "runtime_sec": float(time.time() - ar.started_unix),
            }
            _write_json_atomic(ar.run_dir / "result.json", rec)
            results[ar.restart_id] = rec
            slot_to_restart.pop(ar.slot_id, None)
            _append_monitor(
                f"finish rid={ar.restart_id} rc={rc} success={success} "
                + (f"objective={objective:.6f}" if objective is not None else "objective=NA")
                + (f" reason='{rec['reason']}'" if (not success and rec["reason"]) else "")
            )

        active = still_active
        _emit_heartbeat(force=False)
        time.sleep(1.0)

    _emit_heartbeat(force=True)

    # Final summary and best-run pointers.
    clean_results = [r for r in results if r]
    succeeded = [r for r in clean_results if bool(r.get("success"))]

    def _best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        rows2 = [r for r in rows if r.get("objective") is not None]
        if not rows2:
            return None
        return min(rows2, key=lambda r: float(r["objective"]))

    best_success = _best(succeeded)
    best_any = _best(clean_results)

    if best_success is not None:
        best_dir = Path(str(best_success["run_dir"]))
        link = out_dir / "best_success"
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(str(Path("restarts") / best_dir.name), link, target_is_directory=True)
    if best_any is not None:
        best_any_dir = Path(str(best_any["run_dir"]))
        link = out_dir / "best_any"
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(str(Path("restarts") / best_any_dir.name), link, target_is_directory=True)

    # CSV export.
    fields = [
        "restart_id",
        "slot_id",
        "core_group",
        "pid",
        "return_code",
        "success",
        "objective",
        "reason",
        "runtime_sec",
        "run_dir",
        "log_path",
        "minimum_path",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in sorted(clean_results, key=lambda x: int(x.get("restart_id", 10**9))):
            writer.writerow({k: r.get(k) for k in fields})

    summary = {
        "finished_utc": _utc_now(),
        "phase": str(args.phase),
        "elapsed_sec": float(time.time() - t0),
        "restarts_total": int(restarts),
        "restarts_completed": int(len(clean_results)),
        "successes": int(len(succeeded)),
        "failures": int(len(clean_results) - len(succeeded)),
        "workers": int(workers),
        "threads_per_worker": int(args.threads_per_worker),
        "cpuset": str(args.cpuset),
        "max_evals": int(max_evals),
        "rhoend": float(rhoend),
        "best_success": best_success,
        "best_any": best_any,
        "artifacts": {
            "manifest": str(manifest_json),
            "heartbeat": str(heartbeat_json),
            "monitor_log": str(monitor_log),
            "summary_csv": str(summary_csv),
        },
    }
    _write_json_atomic(summary_json, summary)

    _append_monitor(
        "done "
        + f"completed={len(clean_results)}/{restarts} success={len(succeeded)} fail={len(clean_results) - len(succeeded)} "
        + (
            f"best_success_rid={best_success['restart_id']} best_success_obj={float(best_success['objective']):.6f}"
            if best_success and best_success.get("objective") is not None
            else "best_success=NA"
        )
    )
    print(f"[summary] {summary_json}", flush=True)

    # Do not hard-fail the orchestration when all starts are unsuccessful;
    # downstream can inspect summary.json and choose to continue with higher max_evals.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
