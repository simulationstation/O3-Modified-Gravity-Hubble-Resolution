#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC_ROOT = ROOT / "outputs" / "forward_tests" / "hero_waveform_consistency_prod_latest"
DEFAULT_OUT_ROOT = ROOT / "outputs" / "forward_tests" / "hero_waveform_consistency_rift_latest"
DEFAULT_VENV = ROOT / ".venv_rift"


@dataclass
class SourceJob:
    job_id: str
    source_job_dir: str
    event: str
    waveform: str
    seed: int
    trigger_time: float
    ifos: list[str]
    fmin: float
    lmax: int
    source_status: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare a RIFT workflow matrix from hero waveform production jobs "
            "in a separate output tree."
        )
    )
    ap.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    ap.add_argument(
        "--selection",
        choices=["failed", "not_ok", "all"],
        default="not_ok",
        help="Choose which source jobs to map into RIFT jobs.",
    )
    ap.add_argument(
        "--prepare-now",
        action="store_true",
        help="Run each generated prepare script now.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of jobs processed.",
    )
    ap.add_argument(
        "--force-clean",
        action="store_true",
        help="If set, remove existing out-root before generating jobs.",
    )
    return ap.parse_args()


def parse_ini_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def infer_lmax(waveform: str) -> int:
    wf = waveform.upper()
    if "PHM" in wf or "XHM" in wf or wf.endswith("HM"):
        return 4
    return 2


def parse_ifos(detectors_field: str) -> list[str]:
    # Expected format from source config: [H1,L1,V1]
    cleaned = detectors_field.strip().strip("[]")
    if not cleaned:
        return []
    return [x.strip() for x in cleaned.split(",") if x.strip()]


def source_status(job_dir: Path) -> str:
    result_files = list((job_dir / "run_out" / "result").glob("*result.hdf5"))
    if result_files:
        return "finished_ok"
    exit_path = job_dir / "exit.code"
    if exit_path.exists():
        try:
            ec = int(exit_path.read_text(encoding="utf-8").strip())
        except Exception:
            ec = None
        if ec == 0:
            return "finished_ok"
        return "finished_fail"
    return "unknown"


def select_jobs(src_root: Path, selection: str) -> list[SourceJob]:
    jobs_dir = src_root / "jobs"
    if not jobs_dir.exists():
        raise FileNotFoundError(f"Missing source jobs dir: {jobs_dir}")

    out: list[SourceJob] = []
    for job_dir in sorted(p for p in jobs_dir.iterdir() if p.is_dir()):
        cfg = job_dir / "config_complete.ini"
        if not cfg.exists():
            continue
        kv = parse_ini_kv(cfg)
        stat = source_status(job_dir)

        if selection == "failed" and stat != "finished_fail":
            continue
        if selection == "not_ok" and stat == "finished_ok":
            continue

        waveform = kv.get("waveform-approximant", "")
        seed = int(float(kv.get("sampling-seed", "0")))
        trigger_time = float(kv.get("trigger-time", "0"))
        ifos = parse_ifos(kv.get("detectors", ""))
        fmin = float(kv.get("minimum-frequency", "20"))
        lmax = infer_lmax(waveform)
        event = job_dir.name.split("__")[0]
        out.append(
            SourceJob(
                job_id=job_dir.name,
                source_job_dir=str(job_dir.resolve()),
                event=event,
                waveform=waveform,
                seed=seed,
                trigger_time=trigger_time,
                ifos=ifos,
                fmin=fmin,
                lmax=lmax,
                source_status=stat,
            )
        )
    return out


def write_script(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o775)


def make_prepare_script(job: SourceJob, job_out_dir: Path, venv: Path) -> str:
    ifo_csv = ",".join(job.ifos) if job.ifos else "H1,L1,V1"
    fmin_int = int(round(job.fmin))
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f'ROOT="{ROOT}"',
            f'VENV="{venv}"',
            f'JOB_OUT="{job_out_dir}"',
            'if [[ ! -x "${VENV}/bin/util_RIFT_pseudo_pipe.py" ]]; then',
            '  echo "[error] missing RIFT executable: ${VENV}/bin/util_RIFT_pseudo_pipe.py" >&2',
            "  exit 2",
            "fi",
            'if [[ -d "${JOB_OUT}/rundir" ]]; then',
            '  echo "[skip] rundir already exists: ${JOB_OUT}/rundir"',
            "  exit 0",
            "fi",
            'source "${VENV}/bin/activate"',
            'cd "${JOB_OUT}"',
            'util_RIFT_pseudo_pipe.py \\',
            f'  --event-time {job.trigger_time:.3f} \\',
            f'  --manual-ifo-list {ifo_csv} \\',
            f'  --approx {job.waveform} \\',
            f'  --fmin {fmin_int} \\',
            f'  --l-max {job.lmax} \\',
            '  --use-rundir "${JOB_OUT}/rundir" \\',
            "  > \"${JOB_OUT}/prepare.log\" 2>&1",
            'echo "[ok] prepared ${JOB_OUT}/rundir"',
            "",
        ]
    )


def make_submit_script(job_out_dir: Path) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f'RUNDIR="{job_out_dir}/rundir"',
            'DAG="${RUNDIR}/marginalize_intrinsic_parameters_BasicIterationWorkflow.dag"',
            'if [[ ! -f "${DAG}" ]]; then',
            '  echo "[error] missing DAG: ${DAG}" >&2',
            "  exit 2",
            "fi",
            'cd "${RUNDIR}"',
            'condor_submit_dag -force "${DAG}"',
            "",
        ]
    )


def prepare_now(job_dirs: list[Path]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for job_dir in job_dirs:
        prep = job_dir / "prepare_rift.sh"
        proc = subprocess.run(
            ["bash", str(prep)],
            cwd=str(job_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        (job_dir / "prepare_run.log").write_text(proc.stdout, encoding="utf-8")
        rows.append({"job_id": job_dir.name, "returncode": proc.returncode})
    return rows


def main() -> None:
    args = parse_args()
    src_root = args.src_root.resolve()
    out_root = args.out_root.resolve()
    venv = args.venv.resolve()

    jobs = select_jobs(src_root, args.selection)
    if args.limit is not None:
        jobs = jobs[: max(0, args.limit)]

    if args.force_clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    jobs_dir = out_root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    written_job_dirs: list[Path] = []
    for job in jobs:
        job_out_dir = jobs_dir / job.job_id
        job_out_dir.mkdir(parents=True, exist_ok=True)
        (job_out_dir / "source_job.json").write_text(
            json.dumps(asdict(job), indent=2),
            encoding="utf-8",
        )
        write_script(job_out_dir / "prepare_rift.sh", make_prepare_script(job, job_out_dir, venv))
        write_script(job_out_dir / "submit_dag.sh", make_submit_script(job_out_dir))
        written_job_dirs.append(job_out_dir)

    prep_results: list[dict[str, str | int]] = []
    if args.prepare_now:
        prep_results = prepare_now(written_job_dirs)

    run_all = out_root / "prepare_all.sh"
    run_all_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'ROOT="{out_root}"',
        'for d in "${ROOT}"/jobs/*; do',
        '  [[ -d "$d" ]] || continue',
        '  echo "[prepare] $(basename "$d")"',
        '  bash "$d/prepare_rift.sh" || true',
        "done",
        "",
    ]
    write_script(run_all, "\n".join(run_all_lines))

    summary = {
        "mode": "hero_waveform_rift_prepare",
        "src_root": str(src_root),
        "out_root": str(out_root),
        "venv": str(venv),
        "selection": args.selection,
        "jobs_total": len(jobs),
        "jobs": [asdict(j) for j in jobs],
        "prepare_now": args.prepare_now,
        "prepare_results": prep_results,
        "notes": [
            "This step prepares RIFT run directories and DAG submit scripts; it does not submit DAGs.",
            "Use jobs/*/submit_dag.sh after validating local.cache and PSD availability for your environment.",
        ],
    }
    (out_root / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[ok] out_root: {out_root}")
    print(f"[ok] jobs prepared: {len(jobs)}")
    if args.prepare_now:
        n_ok = sum(1 for r in prep_results if int(r["returncode"]) == 0)
        print(f"[ok] prepare-now success: {n_ok}/{len(prep_results)}")
    print(f"[ok] batch prepare script: {run_all}")


if __name__ == "__main__":
    main()
