#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from gwpy.timeseries import TimeSeries


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RIFT_ROOT = ROOT / "outputs" / "forward_tests" / "hero_waveform_consistency_rift_latest"
SCAFFOLD_EVENTS = (
    ROOT / "outputs" / "forward_tests" / "hero_waveform_consistency_scaffold_20260210" / "events"
)
LAL_PATH2CACHE = ROOT / ".venv_rift" / "bin" / "lal_path2cache"
PSD_ASCII2XML = ROOT / ".venv_rift" / "bin" / "convert_psd_ascii2xml"


CHANNEL_RE = re.compile(r"--channel-name\s+([A-Z0-9]+)=([^\s]+)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Populate RIFT jobs with local frame/cache/PSD assets derived from existing "
            "bilby data dumps (no IGWN datafind/gstlal dependency)."
        )
    )
    ap.add_argument("--rift-root", type=Path, default=DEFAULT_RIFT_ROOT)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing generated local assets in each event directory.",
    )
    ap.add_argument(
        "--patch-local-rate",
        action="store_true",
        default=True,
        help="Patch fmax/srate in generated job command files to match 2048 Hz local dumps.",
    )
    return ap.parse_args()


def find_data_dump(event: str) -> Path:
    ddir = SCAFFOLD_EVENTS / event / "run_out" / "data"
    if not ddir.exists():
        raise FileNotFoundError(f"Missing scaffold data dir for {event}: {ddir}")
    cands = sorted(ddir.glob("*_generation_data_dump.pickle"))
    if not cands:
        raise FileNotFoundError(f"No generation data dump found in {ddir}")
    return cands[0]


def infer_event_from_job_id(job_id: str) -> str:
    return job_id.split("__")[0]


def parse_channel_map(rundir: Path) -> dict[str, str]:
    ile_sub = rundir / "ILE.sub"
    if not ile_sub.exists():
        return {}
    txt = ile_sub.read_text(encoding="utf-8", errors="ignore")
    found = dict(CHANNEL_RE.findall(txt))
    return {k: v for k, v in found.items() if k in {"H1", "L1", "V1"}}


def ensure_tools() -> None:
    if not LAL_PATH2CACHE.exists():
        raise FileNotFoundError(f"Missing lal_path2cache: {LAL_PATH2CACHE}")
    if not PSD_ASCII2XML.exists():
        raise FileNotFoundError(f"Missing convert_psd_ascii2xml: {PSD_ASCII2XML}")


def write_frame(
    out_path: Path,
    td: np.ndarray,
    start: float,
    fs: float,
    ifo: str,
    channel_suffix: str,
) -> None:
    channel = f"{ifo}:{channel_suffix}"
    series = TimeSeries(td, t0=start, sample_rate=fs, unit="strain", name=channel, channel=channel)
    series.write(str(out_path), format="gwf")


def write_psd_xml(out_dir: Path, ifo: str, freq: np.ndarray, psd: np.ndarray) -> Path:
    asc = out_dir / f"{ifo}-psd.txt"
    mask = np.isfinite(freq) & np.isfinite(psd) & (freq >= 0)
    dat = np.column_stack([freq[mask], psd[mask]])
    np.savetxt(asc, dat, fmt="%.9e")

    cmd = [
        str(PSD_ASCII2XML),
        "--fname-psd-ascii",
        str(asc.name),
        "--ifo",
        ifo,
        "--prefix-out",
        "local",
        "--conventional-postfix",
    ]
    subprocess.run(cmd, cwd=str(out_dir), check=True)
    out = out_dir / f"{ifo}-psd.xml.gz"
    if not out.exists():
        raise FileNotFoundError(f"Expected PSD XML missing: {out}")
    return out


def write_cache(out_dir: Path, frame_paths: list[Path]) -> Path:
    payload = "\n".join(str(p) for p in frame_paths) + "\n"
    proc = subprocess.run(
        [str(LAL_PATH2CACHE)],
        input=payload,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    cache = out_dir / "local.cache"
    cache.write_text(proc.stdout, encoding="utf-8")
    return cache


def patch_text_local_rate(path: Path) -> None:
    if not path.exists():
        return
    txt = path.read_text(encoding="utf-8", errors="ignore")
    txt = txt.replace("--fmax 1700", "--fmax 1000")
    txt = txt.replace("--srate 4096", "--srate 2048")
    path.write_text(txt, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_tools()
    rift_root = args.rift_root.resolve()
    jobs_root = rift_root / "jobs"
    if not jobs_root.exists():
        raise FileNotFoundError(f"Missing jobs root: {jobs_root}")

    by_event: dict[str, list[Path]] = {}
    for job_dir in sorted(p for p in jobs_root.iterdir() if p.is_dir()):
        ev = infer_event_from_job_id(job_dir.name)
        by_event.setdefault(ev, []).append(job_dir)

    local_assets_root = rift_root / "local_assets"
    local_assets_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}

    for event, job_dirs in by_event.items():
        dump_path = find_data_dump(event)
        # parse a representative rundir for channel names
        rep_rundir = job_dirs[0] / "rundir"
        chan_map = parse_channel_map(rep_rundir)
        if not chan_map:
            # fallback defaults used by helper for O3
            chan_map = {
                "H1": "GDS-CALIB_STRAIN_CLEAN",
                "L1": "GDS-CALIB_STRAIN_CLEAN",
                "V1": "Hrec_hoft_16384Hz",
            }

        ev_dir = local_assets_root / event
        if ev_dir.exists() and args.force:
            shutil.rmtree(ev_dir)
        ev_dir.mkdir(parents=True, exist_ok=True)

        dd = pickle.load(open(dump_path, "rb"))
        ifos = dd.interferometers

        frame_paths: list[Path] = []
        psd_paths: dict[str, Path] = {}
        for ifo in ifos:
            if ifo.name not in chan_map:
                continue
            sd = ifo.strain_data
            psd = ifo.power_spectral_density
            td = np.asarray(sd.time_domain_strain, dtype=float)
            start = float(sd.start_time)
            fs = float(sd.sampling_frequency)
            dur = int(round(float(sd.duration)))

            letter = ifo.name[0]
            frame = ev_dir / f"{letter}-LOCAL{ifo.name}-{int(start)}-{dur}.gwf"
            write_frame(frame, td, start, fs, ifo.name, chan_map[ifo.name])
            frame_paths.append(frame)

            freq = np.asarray(psd.frequency_array, dtype=float)
            arr = np.asarray(psd.psd_array, dtype=float)
            psd_paths[ifo.name] = write_psd_xml(ev_dir, ifo.name, freq, arr)

        cache = write_cache(ev_dir, frame_paths)

        # push assets into each run dir
        for job_dir in job_dirs:
            rundir = job_dir / "rundir"
            if not rundir.exists():
                continue
            shutil.copy2(cache, rundir / "local.cache")
            for ifo_name, p in psd_paths.items():
                shutil.copy2(p, rundir / f"{ifo_name}-psd.xml.gz")

            if args.patch_local_rate:
                patch_text_local_rate(rundir / "args_ile.txt")
                patch_text_local_rate(rundir / "ILE.sub")
                patch_text_local_rate(rundir / "command-single.sh")

        summary[event] = {
            "data_dump": str(dump_path),
            "channel_map": chan_map,
            "n_jobs": len(job_dirs),
            "asset_dir": str(ev_dir),
            "cache_file": str(cache),
            "frame_files": [str(p) for p in frame_paths],
            "psd_files": {k: str(v) for k, v in psd_paths.items()},
        }

    out = {
        "mode": "hero_waveform_rift_local_assets",
        "rift_root": str(rift_root),
        "local_assets_root": str(local_assets_root),
        "events": summary,
        "notes": [
            "Frames and PSDs are derived from existing local bilby data dumps.",
            "This bypasses IGWN datafind auth and gstlal_reference_psd dependency.",
            "Local-rate patch sets fmax=1000 and srate=2048 in generated command files.",
        ],
    }
    out_path = rift_root / "local_assets_manifest.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[ok] wrote local assets manifest: {out_path}")
    print(f"[ok] events prepared: {len(summary)}")


if __name__ == "__main__":
    main()
