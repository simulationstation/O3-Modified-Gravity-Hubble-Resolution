#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _alpha_logistic(z: float, alpha_low: np.ndarray, alpha_high: np.ndarray, z_t: np.ndarray, width: np.ndarray) -> np.ndarray:
    return alpha_high + (alpha_low - alpha_high) / (1.0 + np.exp((z - z_t) / np.maximum(width, 1e-6)))


def _parse_float_list(text: str) -> list[float]:
    vals = [float(tok.strip()) for tok in str(text).split(",") if tok.strip()]
    if not vals:
        raise ValueError("Expected at least one float value.")
    return vals


@dataclass(frozen=True)
class Grids:
    alpha_low: np.ndarray
    alpha_high: np.ndarray
    z_t: np.ndarray
    width: np.ndarray


def _build_grid(args: argparse.Namespace) -> Grids:
    alpha_low = np.arange(args.alpha_low_min, args.alpha_low_max + 1e-12, args.alpha_low_step, dtype=float)
    alpha_high = np.arange(args.alpha_high_min, args.alpha_high_max + 1e-12, args.alpha_high_step, dtype=float)
    z_t = np.arange(args.z_t_min, args.z_t_max + 1e-12, args.z_t_step, dtype=float)
    width = np.arange(args.width_min, args.width_max + 1e-12, args.width_step, dtype=float)
    if alpha_low.size == 0 or alpha_high.size == 0 or z_t.size == 0 or width.size == 0:
        raise ValueError("Grid construction produced an empty axis. Check min/max/step values.")
    return Grids(alpha_low=alpha_low, alpha_high=alpha_high, z_t=z_t, width=width)


def _top_rows(
    *,
    obj: np.ndarray,
    mask: np.ndarray,
    alpha_low: np.ndarray,
    alpha_high: np.ndarray,
    z_t: np.ndarray,
    width: np.ndarray,
    a2: np.ndarray,
    a3: np.ndarray,
    a4: np.ndarray,
    ar: np.ndarray,
    top_n: int,
) -> list[dict[str, float]]:
    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0:
        return []
    obj_flat = obj.ravel()[idx]
    ord_idx = idx[np.argsort(obj_flat)[: max(1, int(top_n))]]
    out: list[dict[str, float]] = []
    shp = mask.shape
    for flat in ord_idx:
        i, j, k, m = np.unravel_index(int(flat), shp)
        out.append(
            {
                "objective": float(obj[i, j, k, m]),
                "alpha_low_param": float(alpha_low[i, j, k, m]),
                "alpha_high_param": float(alpha_high[i, j, k, m]),
                "z_transition": float(z_t[i, j, k, m]),
                "width": float(width[i, j, k, m]),
                "alpha_phase2_eff": float(a2[i, j, k, m]),
                "alpha_phase3_eff": float(a3[i, j, k, m]),
                "alpha_phase4_eff": float(a4[i, j, k, m]),
                "alpha_relief_eff": float(ar[i, j, k, m]),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Quick non-linear bridge scan for alpha(z). "
            "Uses existing forward-test thresholds (no heavy re-inference)."
        )
    )
    ap.add_argument(
        "--dial-summary",
        default="outputs/forward_tests/signal_amplitude_dial_quick_fullfine_20260210/tables/summary.json",
        help="Summary JSON from the full-fine amplitude dial run.",
    )
    ap.add_argument(
        "--phase4-summary",
        default="outputs/forward_tests/phase4_distance_ratio/tables/summary.json",
    )
    ap.add_argument(
        "--phase6-summary",
        default="outputs/forward_tests/phase6_model_selection/tables/summary.json",
    )
    ap.add_argument("--out", default="outputs/forward_tests/nonlinear_bridge_quick")

    ap.add_argument("--z-phase2", type=float, default=0.50)
    ap.add_argument(
        "--z-phase3",
        type=float,
        default=-1.0,
        help="If <0, use phase4 z_max.",
    )
    ap.add_argument(
        "--z-phase4",
        type=float,
        default=-1.0,
        help="If <0, use phase4 z_max.",
    )
    ap.add_argument(
        "--z-relief-list",
        default="0.10,0.20,0.35",
        help="Comma-separated relief-effective redshifts to test.",
    )
    ap.add_argument("--top-n", type=int, default=8)

    ap.add_argument("--alpha-low-min", type=float, default=0.80)
    ap.add_argument("--alpha-low-max", type=float, default=3.00)
    ap.add_argument("--alpha-low-step", type=float, default=0.05)

    ap.add_argument("--alpha-high-min", type=float, default=0.00)
    ap.add_argument("--alpha-high-max", type=float, default=1.00)
    ap.add_argument("--alpha-high-step", type=float, default=0.05)

    ap.add_argument("--z-t-min", type=float, default=0.05)
    ap.add_argument("--z-t-max", type=float, default=0.60)
    ap.add_argument("--z-t-step", type=float, default=0.01)

    ap.add_argument("--width-min", type=float, default=0.03)
    ap.add_argument("--width-max", type=float, default=0.30)
    ap.add_argument("--width-step", type=float, default=0.02)
    args = ap.parse_args()

    dial = _read_json(Path(args.dial_summary).resolve())
    p4 = _read_json(Path(args.phase4_summary).resolve())
    p6 = _read_json(Path(args.phase6_summary).resolve())

    th = dial["alpha_solutions"]
    th2 = float(th["phase2_pass_min_alpha"])
    th3 = float(th["phase3_pass_min_alpha"])
    th4 = float(th["phase4_pass_min_alpha"])

    relief_mean_base = float(p6["proxy_metrics"]["h0_relief_anchor"]["mean"])
    relief_threshold = float(p6["proxy_metrics"]["h0_relief_anchor"]["material_threshold"])
    alpha_relief_target = relief_threshold / max(relief_mean_base, 1e-12)

    zmax = float(p4["inputs"]["z_max"])
    z_phase3 = float(args.z_phase3) if float(args.z_phase3) >= 0.0 else zmax
    z_phase4 = float(args.z_phase4) if float(args.z_phase4) >= 0.0 else zmax
    z_relief_list = _parse_float_list(args.z_relief_list)

    grids = _build_grid(args)
    a_low, a_high, z_t, width = np.meshgrid(
        grids.alpha_low,
        grids.alpha_high,
        grids.z_t,
        grids.width,
        indexing="ij",
    )

    a2 = _alpha_logistic(float(args.z_phase2), a_low, a_high, z_t, width)
    a3 = _alpha_logistic(z_phase3, a_low, a_high, z_t, width)
    a4 = _alpha_logistic(z_phase4, a_low, a_high, z_t, width)

    pass2 = a2 >= th2
    pass3 = a3 >= th3
    pass4 = a4 >= th4
    core_pass = pass2 & pass3 & pass4

    top_rows_all: list[dict[str, Any]] = []
    scenario_results: list[dict[str, Any]] = []
    target_core = max(th2, th3)
    obj_core = np.abs(a2 - target_core) + np.abs(a3 - target_core) + 0.25 * np.abs(a4 - th4)

    for z_relief in z_relief_list:
        ar = _alpha_logistic(float(z_relief), a_low, a_high, z_t, width)
        relief_pass = (relief_mean_base * ar) >= relief_threshold
        material_pass = core_pass & relief_pass

        obj_material = obj_core + np.abs(ar - alpha_relief_target)

        top_core = _top_rows(
            obj=obj_core,
            mask=core_pass,
            alpha_low=a_low,
            alpha_high=a_high,
            z_t=z_t,
            width=width,
            a2=a2,
            a3=a3,
            a4=a4,
            ar=ar,
            top_n=int(args.top_n),
        )
        top_material = _top_rows(
            obj=obj_material,
            mask=material_pass,
            alpha_low=a_low,
            alpha_high=a_high,
            z_t=z_t,
            width=width,
            a2=a2,
            a3=a3,
            a4=a4,
            ar=ar,
            top_n=int(args.top_n),
        )

        scenario_results.append(
            {
                "z_relief_eff": float(z_relief),
                "n_grid_total": int(core_pass.size),
                "n_core_pass": int(np.count_nonzero(core_pass)),
                "n_material_pass": int(np.count_nonzero(material_pass)),
                "core_pass_fraction": float(np.mean(core_pass)),
                "material_pass_fraction": float(np.mean(material_pass)),
                "top_core_solutions": top_core,
                "top_material_solutions": top_material,
            }
        )

        for r in top_material:
            rr = dict(r)
            rr["z_relief_eff"] = float(z_relief)
            rr["solution_type"] = "material"
            top_rows_all.append(rr)
        for r in top_core:
            rr = dict(r)
            rr["z_relief_eff"] = float(z_relief)
            rr["solution_type"] = "core"
            top_rows_all.append(rr)

    out_dir = Path(args.out).resolve()
    tab_dir = out_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "created_utc": _utc_now(),
        "mode": "nonlinear_bridge_quick_scan",
        "inputs": {
            "dial_summary": str(Path(args.dial_summary).resolve()),
            "phase4_summary": str(Path(args.phase4_summary).resolve()),
            "phase6_summary": str(Path(args.phase6_summary).resolve()),
            "z_phase2": float(args.z_phase2),
            "z_phase3": float(z_phase3),
            "z_phase4": float(z_phase4),
            "z_relief_list": [float(z) for z in z_relief_list],
        },
        "thresholds": {
            "phase2_min_alpha": th2,
            "phase3_min_alpha": th3,
            "phase4_min_alpha": th4,
            "relief_mean_base": relief_mean_base,
            "material_relief_threshold": relief_threshold,
            "alpha_relief_target_linear": alpha_relief_target,
        },
        "parameterization": {
            "alpha_z": "alpha_high + (alpha_low - alpha_high)/(1 + exp((z - z_t)/width))",
            "grid": {
                "alpha_low": {
                    "min": float(args.alpha_low_min),
                    "max": float(args.alpha_low_max),
                    "step": float(args.alpha_low_step),
                },
                "alpha_high": {
                    "min": float(args.alpha_high_min),
                    "max": float(args.alpha_high_max),
                    "step": float(args.alpha_high_step),
                },
                "z_t": {"min": float(args.z_t_min), "max": float(args.z_t_max), "step": float(args.z_t_step)},
                "width": {"min": float(args.width_min), "max": float(args.width_max), "step": float(args.width_step)},
                "n_total": int(a_low.size),
            },
        },
        "scenarios": scenario_results,
        "notes": [
            "This is a threshold-emulator bridge scan, not a full posterior re-inference.",
            "Use top material-pass solutions as seeds for future full runs.",
        ],
    }
    _write_json_atomic(tab_dir / "summary.json", summary)

    with (tab_dir / "top_solutions.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "solution_type",
            "z_relief_eff",
            "objective",
            "alpha_low_param",
            "alpha_high_param",
            "z_transition",
            "width",
            "alpha_phase2_eff",
            "alpha_phase3_eff",
            "alpha_phase4_eff",
            "alpha_relief_eff",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in top_rows_all:
            w.writerow(row)

    lines: list[str] = []
    lines.append("# Non-linear Bridge Quick Scan")
    lines.append("")
    lines.append(f"- Generated UTC: `{summary['created_utc']}`")
    lines.append(f"- Grid size: `{summary['parameterization']['grid']['n_total']}`")
    lines.append(f"- Core threshold target: `alpha >= {max(th2, th3):.2f}`")
    lines.append(f"- Material relief target (linear mode): `alpha_relief >= {alpha_relief_target:.2f}`")
    lines.append("")
    lines.append("## Scenario Counts")
    lines.append("")
    for s in scenario_results:
        lines.append(
            f"- z_relief={s['z_relief_eff']:.2f}: "
            f"core_pass={s['n_core_pass']}/{s['n_grid_total']}, "
            f"material_pass={s['n_material_pass']}/{s['n_grid_total']}"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `tables/summary.json`")
    lines.append("- `tables/top_solutions.csv`")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {tab_dir / 'summary.json'}")
    for s in scenario_results:
        print(
            "[scenario]"
            f" z_relief={s['z_relief_eff']:.2f}"
            f" core_pass={s['n_core_pass']}"
            f" material_pass={s['n_material_pass']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
