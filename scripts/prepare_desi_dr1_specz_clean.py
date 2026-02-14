#!/usr/bin/env python3
"""Prepare a compact DESI DR1 spectroscopic redshift cache for fast local crossmatch.

This script reads the DESI DR1 zcatalog (zall-pix-iron.fits) and writes a
compressed NPZ containing only the columns needed by the dark-siren spec-z
override tests: RA/Dec (deg), z, optional z_err, and an optional quality scalar.

It is designed to avoid loading the full FITS table into memory by processing
in chunks.

Default filtering (tunable via flags):
- require finite z
- ZCAT_PRIMARY == True (if column exists)
- ZWARN == 0 (if column exists)
- optional SPECTYPE == 'GALAXY' (if column exists)
- z_min < z <= z_max

The output NPZ format matches the cache expected by
`scripts/run_dark_siren_smoking_gun_next_sprint.py`:
  ra_deg, dec_deg, z, z_err (optional), quality (optional), source_meta (json)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pick_col(colnames: list[str], candidates: list[str]) -> str | None:
    s = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand.lower() in s:
            return s[cand.lower()]
    return None


def _as_str_array(x: np.ndarray) -> np.ndarray:
    # Convert a FITS string/bytes column to clean python strings.
    out = np.empty(x.shape, dtype=object)
    for i, v in enumerate(x.tolist()):
        if v is None:
            out[i] = ""
            continue
        if isinstance(v, (bytes, bytearray, np.bytes_)):
            try:
                out[i] = v.decode("utf-8", errors="replace").strip()
            except Exception:
                out[i] = str(v).strip()
        else:
            out[i] = str(v).strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare DESI DR1 spec-z clean cache (NPZ) for dark-siren crossmatch.")
    ap.add_argument("--fits", required=True, help="Path to DESI zcatalog FITS (e.g. zall-pix-iron.fits).")
    ap.add_argument("--out", required=True, help="Output NPZ path (e.g. data/cache/specz_catalogs/DESI_DR1_clean.npz).")
    ap.add_argument("--z-max", type=float, default=0.3, help="Keep rows with 0 < z <= z-max.")
    ap.add_argument("--z-min", type=float, default=0.0, help="Keep rows with z > z-min.")
    ap.add_argument("--require-primary", action="store_true", help="Require ZCAT_PRIMARY==True if present.")
    ap.add_argument("--require-zwarn0", action="store_true", help="Require ZWARN==0 if present.")
    ap.add_argument("--require-galaxy", action="store_true", help="Require SPECTYPE=='GALAXY' if present.")
    ap.add_argument("--chunk", type=int, default=2_000_000, help="Chunk size (rows).")
    args = ap.parse_args()

    fits_path = Path(args.fits).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not fits_path.exists():
        raise SystemExit(f"Missing FITS file: {fits_path}")

    z_max = float(args.z_max)
    z_min = float(args.z_min)
    chunk = int(max(100_000, args.chunk))

    ra_all: list[np.ndarray] = []
    dec_all: list[np.ndarray] = []
    z_all: list[np.ndarray] = []
    zerr_all: list[np.ndarray] = []
    q_all: list[np.ndarray] = []

    with fits.open(str(fits_path), memmap=True) as hdul:
        if len(hdul) < 2:
            raise RuntimeError("FITS file missing table HDU 1.")
        data = hdul[1].data
        colnames = [str(c) for c in data.dtype.names]

        ra_col = _pick_col(colnames, ["RA", "TARGET_RA", "RA_ICRS", "RAJ2000"])
        dec_col = _pick_col(colnames, ["DEC", "TARGET_DEC", "DE_ICRS", "DEJ2000"])
        z_col = _pick_col(colnames, ["Z", "z", "ZBEST"])
        zerr_col = _pick_col(colnames, ["ZERR", "zerr", "E_Z", "E_ZBEST"])
        zwarn_col = _pick_col(colnames, ["ZWARN", "z_warn", "ZWARNING"])
        primary_col = _pick_col(colnames, ["ZCAT_PRIMARY", "ZCAT_PRIMARY"])
        spectype_col = _pick_col(colnames, ["SPECTYPE", "SPECTYPE", "CLASS"])

        if not (ra_col and dec_col and z_col):
            raise RuntimeError(f"Could not locate required columns. Have ra={ra_col}, dec={dec_col}, z={z_col}. Columns={colnames[:40]} ...")

        n = int(len(data))
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            sl = slice(start, stop)
            ra = np.asarray(data[ra_col][sl], dtype=float)
            dec = np.asarray(data[dec_col][sl], dtype=float)
            z = np.asarray(data[z_col][sl], dtype=float)
            m = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z) & (z > z_min) & (z <= z_max)

            if args.require_primary and primary_col:
                try:
                    prim = np.asarray(data[primary_col][sl])
                    m = m & prim.astype(bool)
                except Exception:
                    m = m & False

            if args.require_zwarn0 and zwarn_col:
                try:
                    zw = np.asarray(data[zwarn_col][sl], dtype=float)
                    m = m & np.isfinite(zw) & (zw.astype(int) == 0)
                except Exception:
                    m = m & False

            if args.require_galaxy and spectype_col:
                try:
                    st = np.asarray(data[spectype_col][sl])
                    st = _as_str_array(st)
                    m = m & (st == "GALAXY")
                except Exception:
                    m = m & False

            if not np.any(m):
                continue

            ra_all.append(ra[m])
            dec_all.append(dec[m])
            z_all.append(z[m])

            if zerr_col:
                try:
                    zerr = np.asarray(data[zerr_col][sl], dtype=float)
                    zerr_all.append(zerr[m])
                except Exception:
                    pass

            # Encode a simple quality scalar: 1.0 for rows that pass filters.
            q_all.append(np.ones(int(np.count_nonzero(m)), dtype=float))

    ra_out = np.concatenate(ra_all) if ra_all else np.asarray([], dtype=float)
    dec_out = np.concatenate(dec_all) if dec_all else np.asarray([], dtype=float)
    z_out = np.concatenate(z_all) if z_all else np.asarray([], dtype=float)
    q_out = np.concatenate(q_all) if q_all else np.asarray([], dtype=float)
    zerr_out = np.concatenate(zerr_all) if zerr_all else None

    meta: dict[str, Any] = {
        "prepared_utc": _utc_now_iso(),
        "fits_path": str(fits_path),
        "filters": {
            "z_min": z_min,
            "z_max": z_max,
            "require_primary": bool(args.require_primary),
            "require_zwarn0": bool(args.require_zwarn0),
            "require_galaxy": bool(args.require_galaxy),
        },
        "columns": {
            "ra_col": ra_col,
            "dec_col": dec_col,
            "z_col": z_col,
            "zerr_col": zerr_col,
            "zwarn_col": zwarn_col,
            "primary_col": primary_col,
            "spectype_col": spectype_col,
        },
        "n_rows_kept": int(ra_out.size),
    }

    payload = {
        "ra_deg": ra_out,
        "dec_deg": dec_out,
        "z": z_out,
        "quality": q_out,
        "source_meta": np.asarray(json.dumps(meta), dtype=object),
    }
    if zerr_out is not None and int(zerr_out.size) == int(z_out.size):
        payload["z_err"] = zerr_out

    np.savez_compressed(str(out_path), **payload)
    print(f"Wrote {out_path} with n={int(ra_out.size)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
