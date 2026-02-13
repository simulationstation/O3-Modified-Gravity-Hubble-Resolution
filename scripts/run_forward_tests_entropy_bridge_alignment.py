#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.sirens import load_mu_forward_posterior, predict_r_gw_em


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _parse_float_list(text: str) -> list[float]:
    vals = [float(tok.strip()) for tok in str(text).split(",") if tok.strip()]
    if not vals:
        raise ValueError("Expected at least one float value.")
    return vals


def _load_ratio_quantiles_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z: list[float] = []
    q16: list[float] = []
    q50: list[float] = []
    q84: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            z.append(float(row["z"]))
            q16.append(float(row["ratio_q16"]))
            q50.append(float(row["ratio_q50"]))
            q84.append(float(row["ratio_q84"]))
    if not z:
        raise ValueError(f"No rows in ratio quantiles CSV: {path}")
    return (
        np.asarray(z, dtype=float),
        np.asarray(q16, dtype=float),
        np.asarray(q50, dtype=float),
        np.asarray(q84, dtype=float),
    )


def _alpha_logistic(
    z: np.ndarray,
    alpha_low: np.ndarray,
    alpha_high: np.ndarray,
    z_t: np.ndarray,
    width: np.ndarray,
) -> np.ndarray:
    zz = z.reshape((1, -1))
    w = np.maximum(width, 1e-6).reshape((-1, 1))
    return alpha_high.reshape((-1, 1)) + (
        alpha_low.reshape((-1, 1)) - alpha_high.reshape((-1, 1))
    ) / (1.0 + np.exp((zz - z_t.reshape((-1, 1))) / w))


def _fit_logistic_random(
    *,
    z: np.ndarray,
    alpha_target: np.ndarray,
    sigma: np.ndarray,
    n_models: int,
    seed: int,
    alpha_low_min: float,
    alpha_low_max: float,
    alpha_high_min: float,
    alpha_high_max: float,
    z_t_min: float,
    z_t_max: float,
    width_min: float,
    width_max: float,
    top_k: int,
) -> dict[str, Any]:
    if n_models < 2000:
        raise ValueError("n_models must be >= 2000 for stable random fit.")
    if top_k < 20:
        raise ValueError("top_k must be >= 20.")
    rng = np.random.default_rng(int(seed))

    n_draw = int(n_models)
    alpha_low = rng.uniform(alpha_low_min, alpha_low_max, size=n_draw)
    alpha_high = rng.uniform(alpha_high_min, alpha_high_max, size=n_draw)
    z_t = rng.uniform(z_t_min, z_t_max, size=n_draw)
    width = rng.uniform(width_min, width_max, size=n_draw)

    valid = alpha_low >= alpha_high
    if not np.any(valid):
        raise ValueError("No valid random draws with alpha_low >= alpha_high.")
    alpha_low = alpha_low[valid]
    alpha_high = alpha_high[valid]
    z_t = z_t[valid]
    width = width[valid]

    pred = _alpha_logistic(
        z=np.asarray(z, dtype=float),
        alpha_low=alpha_low,
        alpha_high=alpha_high,
        z_t=z_t,
        width=width,
    )
    sig = np.clip(np.asarray(sigma, dtype=float), 1e-3, np.inf).reshape((1, -1))
    res = (pred - np.asarray(alpha_target, dtype=float).reshape((1, -1))) / sig
    obj = np.mean(res**2, axis=1)

    best_i = int(np.argmin(obj))
    best = {
        "objective": float(obj[best_i]),
        "alpha_low": float(alpha_low[best_i]),
        "alpha_high": float(alpha_high[best_i]),
        "z_transition": float(z_t[best_i]),
        "width": float(width[best_i]),
    }

    k = min(int(top_k), int(obj.size))
    order = np.argsort(obj)[:k]
    top_rows: list[dict[str, float]] = []
    for i in order:
        top_rows.append(
            {
                "objective": float(obj[i]),
                "alpha_low": float(alpha_low[i]),
                "alpha_high": float(alpha_high[i]),
                "z_transition": float(z_t[i]),
                "width": float(width[i]),
            }
        )

    z_t_top = np.asarray([row["z_transition"] for row in top_rows], dtype=float)
    width_top = np.asarray([row["width"] for row in top_rows], dtype=float)
    alpha_low_top = np.asarray([row["alpha_low"] for row in top_rows], dtype=float)
    alpha_high_top = np.asarray([row["alpha_high"] for row in top_rows], dtype=float)

    top_stats = {
        "z_transition_p16": float(np.percentile(z_t_top, 16.0)),
        "z_transition_p50": float(np.percentile(z_t_top, 50.0)),
        "z_transition_p84": float(np.percentile(z_t_top, 84.0)),
        "width_p16": float(np.percentile(width_top, 16.0)),
        "width_p50": float(np.percentile(width_top, 50.0)),
        "width_p84": float(np.percentile(width_top, 84.0)),
        "alpha_low_p50": float(np.percentile(alpha_low_top, 50.0)),
        "alpha_high_p50": float(np.percentile(alpha_high_top, 50.0)),
    }
    return {
        "best": best,
        "top_rows": top_rows,
        "top_stats": top_stats,
        "n_valid_models": int(obj.size),
    }


def _load_reference_bridge(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    all_rows: list[dict[str, float]] = []
    for sc in payload.get("scenarios", []):
        if not isinstance(sc, dict):
            continue
        for row in sc.get("top_material_solutions", []):
            if not isinstance(row, dict):
                continue
            all_rows.append(
                {
                    "objective": float(row.get("objective", np.inf)),
                    "alpha_low": float(row.get("alpha_low_param", np.nan)),
                    "alpha_high": float(row.get("alpha_high_param", np.nan)),
                    "z_transition": float(row.get("z_transition", np.nan)),
                    "width": float(row.get("width", np.nan)),
                }
            )
    if not all_rows:
        return None
    return min(all_rows, key=lambda r: float(r["objective"]))


def _plot_ratio_comparison(
    *,
    z: np.ndarray,
    ent_q16: np.ndarray,
    ent_q50: np.ndarray,
    ent_q84: np.ndarray,
    base_q16: np.ndarray,
    base_q50: np.ndarray,
    base_q84: np.ndarray,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8.2, 4.8))
    plt.fill_between(z, base_q16, base_q84, alpha=0.20, color="tab:blue", linewidth=0.0, label="Phase4 68% band")
    plt.plot(z, base_q50, color="tab:blue", linewidth=2.0, label="Phase4 median")
    plt.fill_between(z, ent_q16, ent_q84, alpha=0.24, color="tab:orange", linewidth=0.0, label="Entropy 68% band")
    plt.plot(z, ent_q50, color="tab:orange", linewidth=2.0, label="Entropy median")
    plt.axhline(1.0, color="k", linewidth=1.0, alpha=0.6, label="GR")
    plt.xlabel("z")
    plt.ylabel(r"$R(z)=D_L^{GW}/D_L^{EM}$")
    plt.title("Entropy-Derived GW/EM Ratio vs Phase4 Baseline")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_alpha_fit(
    *,
    z: np.ndarray,
    alpha_q16: np.ndarray,
    alpha_q50: np.ndarray,
    alpha_q84: np.ndarray,
    alpha_fit: np.ndarray,
    alpha_core: float,
    alpha_material: float | None,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8.2, 4.8))
    plt.fill_between(z, alpha_q16, alpha_q84, alpha=0.25, linewidth=0.0, label="effective alpha 68%")
    plt.plot(z, alpha_q50, linewidth=2.0, label="effective alpha median")
    plt.plot(z, alpha_fit, linewidth=2.0, linestyle="--", label="best logistic fit")
    plt.axhline(float(alpha_core), color="tab:red", linewidth=1.3, linestyle=":", label=f"core target={alpha_core:.2f}")
    if alpha_material is not None and np.isfinite(float(alpha_material)):
        plt.axhline(
            float(alpha_material),
            color="tab:purple",
            linewidth=1.2,
            linestyle=":",
            label=f"material target={float(alpha_material):.2f}",
        )
    plt.xlabel("z")
    plt.ylabel("alpha_eff(z)")
    plt.title("Entropy Effective-Alpha Profile and Bridge Fit")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Forward test: compare entropy-derived R(z) curve to the existing "
            "signal-amplitude bridge targets and fit an effective logistic alpha(z)."
        )
    )
    ap.add_argument(
        "--entropy-run-dir",
        default="outputs/entropy_submission_hardening_20260210_203502UTC/realdata_long_mapvar_256c",
    )
    ap.add_argument(
        "--phase4-ratio-csv",
        default="outputs/forward_tests/phase4_distance_ratio/tables/ratio_quantiles.csv",
    )
    ap.add_argument(
        "--dial-summary",
        default="outputs/forward_tests/signal_amplitude_dial_quick_fullfine_20260210/tables/summary.json",
    )
    ap.add_argument(
        "--bridge-summary",
        default="outputs/forward_tests/nonlinear_bridge_quick_20260210/tables/summary.json",
    )
    ap.add_argument("--out", default="outputs/forward_tests/entropy_bridge_alignment")
    ap.add_argument("--convention", choices=["A", "B"], default="A")
    ap.add_argument("--z-min", type=float, default=0.08)
    ap.add_argument("--z-max", type=float, default=0.62)
    ap.add_argument("--z-n", type=int, default=60)
    ap.add_argument(
        "--z-keypoints",
        default="0.10,0.20,0.26,0.30,0.50,0.62",
        help="Comma-separated z values for summary keypoints.",
    )
    ap.add_argument(
        "--alpha-denominator-floor",
        type=float,
        default=0.01,
        help="Ignore/clip points where baseline (R-1) is too close to zero.",
    )
    ap.add_argument(
        "--alpha-sigma-floor",
        type=float,
        default=0.03,
        help="Minimum effective sigma for weighted logistic fit.",
    )

    ap.add_argument("--fit-models", type=int, default=200000)
    ap.add_argument("--fit-seed", type=int, default=20260210)
    ap.add_argument("--fit-top-k", type=int, default=3000)
    ap.add_argument("--fit-alpha-low-min", type=float, default=0.4)
    ap.add_argument("--fit-alpha-low-max", type=float, default=3.2)
    ap.add_argument("--fit-alpha-high-min", type=float, default=0.0)
    ap.add_argument("--fit-alpha-high-max", type=float, default=1.4)
    ap.add_argument("--fit-zt-min", type=float, default=0.05)
    ap.add_argument("--fit-zt-max", type=float, default=0.60)
    ap.add_argument("--fit-width-min", type=float, default=0.02)
    ap.add_argument("--fit-width-max", type=float, default=0.30)
    args = ap.parse_args()

    if args.z_n < 20:
        raise ValueError("z-n must be >= 20.")
    if not (0.0 < args.z_min < args.z_max):
        raise ValueError("Require 0 < z-min < z-max.")
    if args.alpha_denominator_floor <= 0.0:
        raise ValueError("alpha-denominator-floor must be > 0.")
    if args.alpha_sigma_floor <= 0.0:
        raise ValueError("alpha-sigma-floor must be > 0.")

    out_dir = Path(args.out).resolve()
    tab_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    dial = _read_json(Path(args.dial_summary).resolve())
    alpha_solutions = dial.get("alpha_solutions", {})
    alpha_phase2 = float(alpha_solutions["phase2_pass_min_alpha"])
    alpha_phase3 = float(alpha_solutions["phase3_pass_min_alpha"])
    alpha_phase4 = float(alpha_solutions["phase4_pass_min_alpha"])
    alpha_material = alpha_solutions.get("m2_material_relief_min_alpha_linear_relief", None)
    alpha_material_f = float(alpha_material) if alpha_material is not None else None
    alpha_core = max(alpha_phase2, alpha_phase3)

    z_base, base_q16_raw, base_q50_raw, base_q84_raw = _load_ratio_quantiles_csv(Path(args.phase4_ratio_csv).resolve())
    z_eval = np.linspace(float(args.z_min), float(args.z_max), int(args.z_n))
    base_q16 = np.interp(z_eval, z_base, base_q16_raw)
    base_q50 = np.interp(z_eval, z_base, base_q50_raw)
    base_q84 = np.interp(z_eval, z_base, base_q84_raw)

    post = load_mu_forward_posterior(Path(args.entropy_run_dir).resolve())
    _, r_entropy = predict_r_gw_em(post, z_eval=z_eval, convention=args.convention, allow_extrapolation=False)
    ent_q16 = np.percentile(r_entropy, 16.0, axis=0)
    ent_q50 = np.percentile(r_entropy, 50.0, axis=0)
    ent_q84 = np.percentile(r_entropy, 84.0, axis=0)

    denom = np.clip(base_q50 - 1.0, float(args.alpha_denominator_floor), np.inf)
    alpha_q16 = (ent_q16 - 1.0) / denom
    alpha_q50 = (ent_q50 - 1.0) / denom
    alpha_q84 = (ent_q84 - 1.0) / denom
    alpha_sigma = np.maximum(0.5 * (alpha_q84 - alpha_q16), float(args.alpha_sigma_floor))

    fit_mask = np.isfinite(alpha_q50) & np.isfinite(alpha_sigma) & (denom > float(args.alpha_denominator_floor))
    if int(np.count_nonzero(fit_mask)) < 20:
        raise ValueError("Not enough valid points for logistic fit after denominator masking.")
    z_fit = z_eval[fit_mask]
    alpha_fit_target = alpha_q50[fit_mask]
    alpha_fit_sigma = alpha_sigma[fit_mask]

    fit = _fit_logistic_random(
        z=z_fit,
        alpha_target=alpha_fit_target,
        sigma=alpha_fit_sigma,
        n_models=int(args.fit_models),
        seed=int(args.fit_seed),
        alpha_low_min=float(args.fit_alpha_low_min),
        alpha_low_max=float(args.fit_alpha_low_max),
        alpha_high_min=float(args.fit_alpha_high_min),
        alpha_high_max=float(args.fit_alpha_high_max),
        z_t_min=float(args.fit_zt_min),
        z_t_max=float(args.fit_zt_max),
        width_min=float(args.fit_width_min),
        width_max=float(args.fit_width_max),
        top_k=int(args.fit_top_k),
    )
    best = fit["best"]
    alpha_fit_best = _alpha_logistic(
        z=z_eval,
        alpha_low=np.asarray([best["alpha_low"]], dtype=float),
        alpha_high=np.asarray([best["alpha_high"]], dtype=float),
        z_t=np.asarray([best["z_transition"]], dtype=float),
        width=np.asarray([best["width"]], dtype=float),
    )[0]

    z_keys = _parse_float_list(args.z_keypoints)
    keypoints: dict[str, dict[str, float]] = {}
    for z0 in z_keys:
        keypoints[f"z{z0:.2f}"] = {
            "entropy_R_q16": float(np.interp(z0, z_eval, ent_q16)),
            "entropy_R_q50": float(np.interp(z0, z_eval, ent_q50)),
            "entropy_R_q84": float(np.interp(z0, z_eval, ent_q84)),
            "phase4_R_q50": float(np.interp(z0, z_eval, base_q50)),
            "alpha_eff_q16": float(np.interp(z0, z_eval, alpha_q16)),
            "alpha_eff_q50": float(np.interp(z0, z_eval, alpha_q50)),
            "alpha_eff_q84": float(np.interp(z0, z_eval, alpha_q84)),
        }

    cross_core = bool(np.any(alpha_q50 >= float(alpha_core)))
    cross_material = bool(np.any(alpha_q50 >= float(alpha_material_f))) if alpha_material_f is not None else False
    alpha_max = float(np.max(alpha_q50))
    alpha_min = float(np.min(alpha_q50))

    ref_bridge = _load_reference_bridge(Path(args.bridge_summary).resolve())
    ref_delta: dict[str, float] | None = None
    if ref_bridge is not None:
        ref_delta = {
            "delta_alpha_low": float(best["alpha_low"] - ref_bridge["alpha_low"]),
            "delta_alpha_high": float(best["alpha_high"] - ref_bridge["alpha_high"]),
            "delta_z_transition": float(best["z_transition"] - ref_bridge["z_transition"]),
            "delta_width": float(best["width"] - ref_bridge["width"]),
        }

    with (tab_dir / "alpha_curve.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "z",
                "entropy_R_q16",
                "entropy_R_q50",
                "entropy_R_q84",
                "phase4_R_q16",
                "phase4_R_q50",
                "phase4_R_q84",
                "alpha_eff_q16",
                "alpha_eff_q50",
                "alpha_eff_q84",
                "alpha_fit_best",
            ]
        )
        for i in range(z_eval.size):
            w.writerow(
                [
                    z_eval[i],
                    ent_q16[i],
                    ent_q50[i],
                    ent_q84[i],
                    base_q16[i],
                    base_q50[i],
                    base_q84[i],
                    alpha_q16[i],
                    alpha_q50[i],
                    alpha_q84[i],
                    alpha_fit_best[i],
                ]
            )

    _plot_ratio_comparison(
        z=z_eval,
        ent_q16=ent_q16,
        ent_q50=ent_q50,
        ent_q84=ent_q84,
        base_q16=base_q16,
        base_q50=base_q50,
        base_q84=base_q84,
        out_path=fig_dir / "ratio_entropy_vs_phase4.png",
    )
    _plot_alpha_fit(
        z=z_eval,
        alpha_q16=alpha_q16,
        alpha_q50=alpha_q50,
        alpha_q84=alpha_q84,
        alpha_fit=alpha_fit_best,
        alpha_core=float(alpha_core),
        alpha_material=alpha_material_f,
        out_path=fig_dir / "alpha_effective_fit.png",
    )

    summary: dict[str, Any] = {
        "created_utc": _utc_now(),
        "mode": "entropy_bridge_alignment",
        "inputs": {
            "entropy_run_dir": str(Path(args.entropy_run_dir).resolve()),
            "phase4_ratio_csv": str(Path(args.phase4_ratio_csv).resolve()),
            "dial_summary": str(Path(args.dial_summary).resolve()),
            "bridge_summary": str(Path(args.bridge_summary).resolve()),
            "convention": str(args.convention),
            "z_min": float(args.z_min),
            "z_max": float(args.z_max),
            "z_n": int(args.z_n),
        },
        "gate_targets": {
            "phase2_min_alpha": float(alpha_phase2),
            "phase3_min_alpha": float(alpha_phase3),
            "phase4_min_alpha": float(alpha_phase4),
            "core_min_alpha": float(alpha_core),
            "material_min_alpha_linear": float(alpha_material_f) if alpha_material_f is not None else None,
        },
        "entropy_curve": {
            "alpha_eff_min_q50": alpha_min,
            "alpha_eff_max_q50": alpha_max,
            "crosses_core_target": cross_core,
            "crosses_material_target": cross_material,
            "keypoints": keypoints,
        },
        "logistic_fit": {
            "best": best,
            "top_stats": fit["top_stats"],
            "n_valid_models": int(fit["n_valid_models"]),
        },
        "reference_bridge": ref_bridge,
        "reference_bridge_delta_from_entropy_fit": ref_delta,
        "notes": [
            "alpha_eff is computed against the existing Phase4 baseline via (R_entropy-1)/(R_phase4-1).",
            "This is a fast alignment test using existing posteriors, not a fresh dark-siren hierarchical re-inference.",
        ],
    }
    _write_json_atomic(tab_dir / "summary.json", summary)

    lines: list[str] = []
    lines.append("# Entropy Bridge Alignment Test")
    lines.append("")
    lines.append(f"- Generated UTC: `{summary['created_utc']}`")
    lines.append(f"- Effective alpha range (q50): `{alpha_min:.3f}` to `{alpha_max:.3f}`")
    lines.append(f"- Core target alpha: `{alpha_core:.2f}` -> `{'PASS' if cross_core else 'FAIL'}`")
    if alpha_material_f is not None:
        lines.append(
            f"- Material target alpha: `{alpha_material_f:.2f}` -> `{'PASS' if cross_material else 'FAIL'}`"
        )
    lines.append("")
    lines.append("## Best Logistic Bridge Fit")
    lines.append("")
    lines.append(
        f"- Best `(alpha_low, alpha_high, z_t, width)` = "
        f"`({best['alpha_low']:.3f}, {best['alpha_high']:.3f}, {best['z_transition']:.3f}, {best['width']:.3f})`"
    )
    lines.append(f"- Weighted MSE objective: `{best['objective']:.4f}`")
    if ref_bridge is not None and ref_delta is not None:
        lines.append("")
        lines.append("## Versus Existing Nonlinear Bridge Reference")
        lines.append("")
        lines.append(
            f"- Reference `(alpha_low, alpha_high, z_t, width)` = "
            f"`({ref_bridge['alpha_low']:.3f}, {ref_bridge['alpha_high']:.3f}, "
            f"{ref_bridge['z_transition']:.3f}, {ref_bridge['width']:.3f})`"
        )
        lines.append(
            f"- Delta `(d_alpha_low, d_alpha_high, d_z_t, d_width)` = "
            f"`({ref_delta['delta_alpha_low']:.3f}, {ref_delta['delta_alpha_high']:.3f}, "
            f"{ref_delta['delta_z_transition']:.3f}, {ref_delta['delta_width']:.3f})`"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `tables/summary.json`")
    lines.append("- `tables/alpha_curve.csv`")
    lines.append("- `figures/ratio_entropy_vs_phase4.png`")
    lines.append("- `figures/alpha_effective_fit.png`")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {tab_dir / 'summary.json'}")
    print(f"[done] alpha_max_q50={alpha_max:.4f} core_target={alpha_core:.4f} cross_core={cross_core}")
    if alpha_material_f is not None:
        print(
            f"[done] material_target={alpha_material_f:.4f} cross_material={cross_material}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
