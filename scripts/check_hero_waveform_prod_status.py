#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Status summary for hero waveform production matrix.")
    ap.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Run root (e.g. outputs/forward_tests/hero_waveform_consistency_prod_latest).",
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON.")
    return ap.parse_args()


def proc_config_set(run_root: Path) -> set[str]:
    cmd = (
        "ps -eo args | "
        "rg -i '/home/primary/PROJECT/.venv/bin/bilby_pipe_analysis .*"
        + str(run_root).replace("/", "\\/")
        + "' | rg -v 'rg -i' | "
        "awk '{for(i=1;i<=NF;i++){if($i ~ /config_complete\\.ini$/){print $i}}}'"
    )
    out = subprocess.check_output(["bash", "-lc", cmd], text=True)
    return {line.strip() for line in out.splitlines() if line.strip()}


_PROG_RE = re.compile(
    r"(?P<it>\d+)it\s+\[(?P<elapsed>[0-9:]+)\s+bound:(?P<bound>\d+)\s+nc:\s*(?P<nc>\d+)\s+"
    r"ncall:(?P<ncall>[0-9.e+\-]+)\s+eff:(?P<eff>[0-9.]+)%.*?dlogz:(?P<dlogz>[0-9.]+)>",
    re.DOTALL,
)


def parse_progress(run_log: Path) -> dict | None:
    if not run_log.exists():
        return None
    try:
        blob = run_log.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    blob = blob.replace("\r", "\n")
    matches = list(_PROG_RE.finditer(blob))
    if not matches:
        return None
    m = matches[-1]
    ncall_raw = m.group("ncall")
    return {
        "it": int(m.group("it")),
        "elapsed": m.group("elapsed"),
        "bound": int(m.group("bound")),
        "nc": int(m.group("nc")),
        "ncall": ncall_raw,
        "ncall_value": float(ncall_raw),
        "eff_pct": float(m.group("eff")),
        "dlogz": float(m.group("dlogz")),
        "source": "run_log",
    }


def _parse_resume_progress_via_venv(resume_pickle: Path) -> dict | None:
    venv_py = Path("/home/primary/PROJECT/.venv/bin/python")
    if not venv_py.exists():
        return None
    code = r"""
import json
import pickle
import sys

p = sys.argv[1]
obj = pickle.load(open(p, "rb"))
sampler = obj[0] if isinstance(obj, tuple) and len(obj) > 0 else obj
it = getattr(sampler, "it", None)
ncall = getattr(sampler, "ncall", None)
eff = getattr(sampler, "eff", None)
nbound = getattr(sampler, "nbound", None)
if it is None or ncall is None:
    raise SystemExit(2)
out = {
    "it": int(it),
    "elapsed": "?",
    "bound": int(nbound) if nbound is not None else -1,
    "nc": -1,
    "ncall": f"{float(ncall):.3g}",
    "ncall_value": float(ncall),
    "eff_pct": float(eff) if eff is not None else float("nan"),
    "dlogz": float("nan"),
    "source": "resume_pickle",
}
print(json.dumps(out))
"""
    try:
        raw = subprocess.check_output([str(venv_py), "-c", code, str(resume_pickle)], text=True)
        return json.loads(raw.strip())
    except Exception:
        return None


def parse_resume_progress(resume_pickle: Path) -> dict | None:
    if not resume_pickle.exists():
        return None
    try:
        import pickle

        obj = pickle.load(open(resume_pickle, "rb"))
        sampler = obj[0] if isinstance(obj, tuple) and len(obj) > 0 else obj
        it = getattr(sampler, "it", None)
        ncall = getattr(sampler, "ncall", None)
        eff = getattr(sampler, "eff", None)
        nbound = getattr(sampler, "nbound", None)
        if it is None or ncall is None:
            return None
        return {
            "it": int(it),
            "elapsed": "?",
            "bound": int(nbound) if nbound is not None else -1,
            "nc": -1,
            "ncall": f"{float(ncall):.3g}",
            "ncall_value": float(ncall),
            "eff_pct": float(eff) if eff is not None else float("nan"),
            "dlogz": float("nan"),
            "source": "resume_pickle",
        }
    except ModuleNotFoundError:
        return _parse_resume_progress_via_venv(resume_pickle)
    except Exception:
        return _parse_resume_progress_via_venv(resume_pickle)


def parse_sampler_caps(config_path: Path) -> dict:
    out = {"maxiter": None, "maxcall": None, "dlogz_target": None}
    if not config_path.exists():
        return out
    try:
        text = config_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return out
    m = re.search(r"(?m)^sampler-kwargs=(.*)$", text)
    if not m:
        return out
    try:
        kwargs = ast.literal_eval(m.group(1).strip())
    except Exception:
        return out
    if not isinstance(kwargs, dict):
        return out
    try:
        if kwargs.get("maxiter") is not None:
            out["maxiter"] = int(kwargs["maxiter"])
    except Exception:
        pass
    try:
        if kwargs.get("maxcall") is not None:
            out["maxcall"] = float(kwargs["maxcall"])
    except Exception:
        pass
    try:
        if kwargs.get("dlogz") is not None:
            out["dlogz_target"] = float(kwargs["dlogz"])
    except Exception:
        pass
    return out


def completion_metrics(progress: dict | None, caps: dict) -> dict | None:
    if progress is None:
        return None
    out = {
        "it_done": progress["it"],
        "it_total": caps.get("maxiter"),
        "it_frac": None,
        "ncall_done": progress["ncall_value"],
        "ncall_total": caps.get("maxcall"),
        "ncall_frac": None,
        "dlogz_current": progress["dlogz"],
        "dlogz_target": caps.get("dlogz_target"),
        "dlogz_frac": None,
    }
    it_total = out["it_total"]
    if isinstance(it_total, int) and it_total > 0:
        out["it_frac"] = min(1.0, max(0.0, out["it_done"] / it_total))
    ncall_total = out["ncall_total"]
    if isinstance(ncall_total, (int, float)) and ncall_total > 0:
        out["ncall_frac"] = min(1.0, max(0.0, out["ncall_done"] / ncall_total))
    dlogz_target = out["dlogz_target"]
    dlogz_current = out["dlogz_current"]
    if isinstance(dlogz_target, (int, float)) and dlogz_target > 0 and dlogz_current > 0:
        out["dlogz_frac"] = min(1.0, max(0.0, dlogz_target / dlogz_current))
    return out


def pct(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{100.0 * x:.1f}%"


def fmt_count(x: float | int | None) -> str:
    if x is None:
        return "?"
    return f"{int(round(float(x))):,}"


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    job_dirs = sorted((run_root / "jobs").glob("*"))
    active_cfg = proc_config_set(run_root)

    rows = []
    counts = {"running": 0, "finished_ok": 0, "finished_fail": 0, "unknown": 0}
    for jd in job_dirs:
        cfg = jd / "config_complete.ini"
        exit_path = jd / "exit.code"
        run_log = jd / "run.log"
        result_files = sorted((jd / "run_out" / "result").glob("*result.hdf5"))
        progress_log = parse_progress(run_log)
        progress = progress_log
        caps = parse_sampler_caps(cfg)
        status = "unknown"
        exit_code = None
        if str(cfg) in active_cfg:
            status = "running"
        elif result_files:
            status = "finished_ok"
        elif exit_path.exists():
            try:
                exit_code = int(exit_path.read_text(encoding="utf-8").strip())
            except Exception:
                exit_code = None
            status = "finished_ok" if exit_code == 0 else "finished_fail"

        if status == "running":
            resume_files = sorted((jd / "run_out" / "result").glob("*_resume.pickle"))
            if resume_files:
                resume_progress = parse_resume_progress(resume_files[0])
                if resume_progress is not None:
                    if progress_log is not None:
                        # Use whichever source is more advanced; log and checkpoint can lag each other.
                        resume_newer = (
                            (resume_progress.get("it", -1) > progress_log.get("it", -1))
                            or (
                                resume_progress.get("ncall_value", -1.0)
                                > progress_log.get("ncall_value", -1.0)
                            )
                        )
                        if resume_newer:
                            # Keep these from run-log if available, as resume pickles may not store them directly.
                            if "elapsed" in progress_log:
                                resume_progress["elapsed"] = progress_log["elapsed"]
                            if "dlogz" in progress_log:
                                resume_progress["dlogz"] = progress_log["dlogz"]
                            if "nc" in progress_log and progress_log["nc"] >= 0:
                                resume_progress["nc"] = progress_log["nc"]
                            resume_progress["source"] = "resume_pickle+run_log"
                            progress = resume_progress
                        else:
                            progress_log["source"] = "run_log+resume_pickle"
                            progress = progress_log
                    else:
                        progress = resume_progress

        completion = completion_metrics(progress, caps)
        counts[status] += 1
        rows.append(
            {
                "job": jd.name,
                "status": status,
                "exit_code": exit_code,
                "result_files": len(result_files),
                "progress": progress,
                "caps": caps,
                "completion": completion,
            }
        )

    summary = {
        "run_root": str(run_root),
        "jobs_total": len(job_dirs),
        "counts": counts,
        "active_config_count": len(active_cfg),
        "jobs_success_frac": (counts["finished_ok"] / len(job_dirs)) if job_dirs else 0.0,
        "jobs_terminal_frac": (
            (counts["finished_ok"] + counts["finished_fail"]) / len(job_dirs)
        )
        if job_dirs
        else 0.0,
    }

    if args.json:
        print(json.dumps({"summary": summary, "rows": rows}, indent=2))
        return

    print(f"run_root: {summary['run_root']}")
    print(f"jobs_total: {summary['jobs_total']}")
    print(
        "counts: "
        f"running={counts['running']} "
        f"finished_ok={counts['finished_ok']} "
        f"finished_fail={counts['finished_fail']} "
        f"unknown={counts['unknown']}"
    )
    print(
        "matrix_progress: "
        f"successful={counts['finished_ok']}/{summary['jobs_total']} ({pct(summary['jobs_success_frac'])}) "
        f"terminal={counts['finished_ok'] + counts['finished_fail']}/{summary['jobs_total']} "
        f"({pct(summary['jobs_terminal_frac'])})"
    )
    print(f"active_config_count: {summary['active_config_count']}")
    for row in rows:
        prog = row["progress"]
        completion = row["completion"]
        if prog is None:
            print(f"{row['job']}: {row['status']} (no progress line yet)")
            continue
        comp_text = ""
        if completion is not None:
            comp_text = (
                " "
                f"iter={completion['it_done']}/{fmt_count(completion['it_total'])}({pct(completion['it_frac'])}) "
                f"ncall={fmt_count(completion['ncall_done'])}/{fmt_count(completion['ncall_total'])}({pct(completion['ncall_frac'])}) "
                f"dlogz={completion['dlogz_current']:.3f}->{completion['dlogz_target'] if completion['dlogz_target'] is not None else '?'}"
                f"({pct(completion['dlogz_frac'])})"
            )
        print(
            f"{row['job']}: {row['status']} "
            f"it={prog['it']} ncall={prog['ncall']} eff={prog['eff_pct']:.1f}% "
            f"dlogz={prog['dlogz']:.3f} elapsed={prog['elapsed']}"
            f" source={prog.get('source', 'unknown')}{comp_text}"
        )


if __name__ == "__main__":
    main()
