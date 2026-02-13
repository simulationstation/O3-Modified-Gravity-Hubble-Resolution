#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _stats_1d(vals: list[float]) -> dict[str, float]:
    x = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if x.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "p16": float("nan"),
            "p50": float("nan"),
            "p84": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "p16": float(np.percentile(x, 16.0)),
        "p50": float(np.percentile(x, 50.0)),
        "p84": float(np.percentile(x, 84.0)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _recompute_relief(local_ref: float, planck_ref: float, inferred_h0: float) -> float:
    baseline_gap = abs(float(local_ref) - float(planck_ref))
    if not np.isfinite(baseline_gap) or baseline_gap <= 0.0:
        return float("nan")
    return 1.0 - abs(float(local_ref) - float(inferred_h0)) / baseline_gap


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Rebase bias-transfer sweep relief metrics to a new Planck-like H0 reference "
            "using each case's stored summary.json."
        )
    )
    ap.add_argument("--input-csv", required=True, help="Input sweep_results.csv")
    ap.add_argument("--output-csv", required=True, help="Output CSV path")
    ap.add_argument("--h0-planck-ref-new", type=float, required=True, help="New Planck-like H0 reference")
    ap.add_argument(
        "--h0-local-ref-override",
        type=float,
        default=float("nan"),
        help="Optional override for local H0 reference (default: use each case summary value).",
    )
    ap.add_argument(
        "--overwrite-relief",
        action="store_true",
        help="If set, overwrite anchor_gr_relief/posterior_h0_relief columns with rebased values.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing/invalid case summaries. Default is skip with NaN output for that row.",
    )
    args = ap.parse_args()

    in_csv = Path(args.input_csv).resolve()
    out_csv = Path(args.output_csv).resolve()
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {in_csv}")

    rows_out: list[dict[str, Any]] = []
    skipped = 0
    errors: list[str] = []

    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        in_fields = list(reader.fieldnames or [])
        for row in reader:
            out = dict(row)
            summary_path = Path(str(row.get("out_dir", ""))) / "tables" / "summary.json"
            anchor_gr_old = _to_float(row.get("anchor_gr_relief", float("nan")))
            post_old = _to_float(row.get("posterior_h0_relief", float("nan")))

            anchor_gr_new = float("nan")
            anchor_mg_new = float("nan")
            post_new = float("nan")
            baseline_new = float("nan")
            local_ref_used = float("nan")
            planck_ref_old = float("nan")
            case_ok = False

            if summary_path.exists():
                try:
                    s = json.loads(summary_path.read_text(encoding="utf-8"))
                    refs = dict(s.get("references", {}))
                    proj = dict(s.get("h0_tension_projection", {}))
                    post = dict(s.get("h0_posterior_mg_truth", {}))

                    local_ref_summary = _to_float(refs.get("h0_local_ref", float("nan")))
                    local_ref_override = _to_float(args.h0_local_ref_override)
                    if np.isfinite(local_ref_override):
                        local_ref_used = float(local_ref_override)
                    else:
                        local_ref_used = float(local_ref_summary)

                    planck_ref_old = _to_float(refs.get("h0_planck_ref", float("nan")))
                    h0_gr_anchor = _to_float(proj.get("anchor_h0_gr_mean", float("nan")))
                    h0_mg_anchor = _to_float(proj.get("anchor_h0_mg_mean", float("nan")))
                    h0_p50 = _to_float(post.get("p50", float("nan")))

                    baseline_new = abs(float(local_ref_used) - float(args.h0_planck_ref_new))
                    anchor_gr_new = _recompute_relief(local_ref_used, float(args.h0_planck_ref_new), h0_gr_anchor)
                    anchor_mg_new = _recompute_relief(local_ref_used, float(args.h0_planck_ref_new), h0_mg_anchor)
                    post_new = _recompute_relief(local_ref_used, float(args.h0_planck_ref_new), h0_p50)
                    case_ok = True
                except Exception as e:  # pragma: no cover - defensive parsing
                    errors.append(f"{summary_path}: {e}")
            else:
                errors.append(f"{summary_path}: missing")

            if (not case_ok) and bool(args.strict):
                raise RuntimeError(f"Failed to rebase row '{row.get('label', '<unknown>')}'")
            if not case_ok:
                skipped += 1

            out["h0_planck_ref_old"] = f"{planck_ref_old:.9g}" if np.isfinite(planck_ref_old) else ""
            out["h0_planck_ref_new"] = f"{float(args.h0_planck_ref_new):.9g}"
            out["h0_local_ref_used"] = f"{local_ref_used:.9g}" if np.isfinite(local_ref_used) else ""
            out["baseline_gap_new"] = f"{baseline_new:.9g}" if np.isfinite(baseline_new) else ""
            out["anchor_gr_relief_old"] = f"{anchor_gr_old:.9g}" if np.isfinite(anchor_gr_old) else ""
            out["anchor_gr_relief_rebased"] = f"{anchor_gr_new:.9g}" if np.isfinite(anchor_gr_new) else ""
            out["anchor_mg_relief_rebased"] = f"{anchor_mg_new:.9g}" if np.isfinite(anchor_mg_new) else ""
            out["posterior_h0_relief_old"] = f"{post_old:.9g}" if np.isfinite(post_old) else ""
            out["posterior_h0_relief_rebased"] = f"{post_new:.9g}" if np.isfinite(post_new) else ""

            if bool(args.overwrite_relief):
                out["anchor_gr_relief"] = out["anchor_gr_relief_rebased"]
                out["posterior_h0_relief"] = out["posterior_h0_relief_rebased"]

            rows_out.append(out)

    extra = [
        "h0_planck_ref_old",
        "h0_planck_ref_new",
        "h0_local_ref_used",
        "baseline_gap_new",
        "anchor_gr_relief_old",
        "anchor_gr_relief_rebased",
        "anchor_mg_relief_rebased",
        "posterior_h0_relief_old",
        "posterior_h0_relief_rebased",
    ]
    out_fields = list(in_fields)
    for col in extra:
        if col not in out_fields:
            out_fields.append(col)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_out)

    anch_old = [_to_float(r.get("anchor_gr_relief_old", float("nan"))) for r in rows_out]
    anch_new = [_to_float(r.get("anchor_gr_relief_rebased", float("nan"))) for r in rows_out]
    post_old = [_to_float(r.get("posterior_h0_relief_old", float("nan"))) for r in rows_out]
    post_new = [_to_float(r.get("posterior_h0_relief_rebased", float("nan"))) for r in rows_out]

    summary = {
        "created_utc": _utc_now(),
        "input_csv": str(in_csv),
        "output_csv": str(out_csv),
        "rows_total": int(len(rows_out)),
        "rows_skipped": int(skipped),
        "h0_planck_ref_new": float(args.h0_planck_ref_new),
        "h0_local_ref_override": (
            float(args.h0_local_ref_override) if np.isfinite(_to_float(args.h0_local_ref_override)) else None
        ),
        "overwrite_relief": bool(args.overwrite_relief),
        "anchor_gr_relief_old_stats": _stats_1d(anch_old),
        "anchor_gr_relief_rebased_stats": _stats_1d(anch_new),
        "posterior_h0_relief_old_stats": _stats_1d(post_old),
        "posterior_h0_relief_rebased_stats": _stats_1d(post_new),
        "errors_preview": errors[:20],
        "errors_total": int(len(errors)),
    }
    _write_json_atomic(out_csv.with_suffix(out_csv.suffix + ".summary.json"), summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
