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

from entropy_horizon_recon.sirens import (
    MuForwardPosterior,
    load_mu_forward_posterior,
    predict_r_gw_em,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _parse_run_dirs(text: str) -> list[str]:
    vals = [t.strip() for t in str(text).split(",") if t.strip()]
    if not vals:
        raise ValueError("No run dirs provided.")
    return vals


def _parse_floats_csv(text: str) -> list[float]:
    vals = [float(t.strip()) for t in str(text).split(",") if t.strip()]
    if not vals:
        raise ValueError("No float values provided.")
    return vals


def _subset_post(post: MuForwardPosterior, idx: np.ndarray) -> MuForwardPosterior:
    return MuForwardPosterior(
        x_grid=post.x_grid,
        logmu_x_samples=post.logmu_x_samples[idx],
        z_grid=post.z_grid,
        H_samples=post.H_samples[idx],
        H0=post.H0[idx],
        omega_m0=post.omega_m0[idx],
        omega_k0=post.omega_k0[idx],
        sigma8_0=(None if post.sigma8_0 is None else post.sigma8_0[idx]),
    )


def _stats_1d(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    if x.size <= 0:
        return {
            "mean": float("nan"),
            "sd": float("nan"),
            "p16": float("nan"),
            "p50": float("nan"),
            "p84": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "p16": float(np.percentile(x, 16.0)),
        "p50": float(np.percentile(x, 50.0)),
        "p84": float(np.percentile(x, 84.0)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _plot_ratio_medians(*, z_probe: np.ndarray, per_run: list[dict[str, Any]], out_path: Path) -> None:
    plt.figure(figsize=(7.8, 4.8))
    for row in per_run:
        med = np.asarray(row["ratio_median_by_z"], dtype=float)
        plt.plot(z_probe, med, linewidth=1.8, label=Path(str(row["run_dir"])).name)
    plt.axhline(1.0, color="k", linewidth=1.0, alpha=0.6)
    plt.xlabel("z")
    plt.ylabel(r"$R(z)=D_L^{GW}/D_L^{EM}$")
    plt.title("Phase 1 Closure: Shared-Draw GW Ratio Curves")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Concrete forward test (Phase 1): shared-draw closure audit across "
            "GW ratio, growth proxy (S8), and lensing proxy amplitude."
        )
    )
    ap.add_argument(
        "--run-dirs",
        default=(
            "outputs/finalization/highpower_multistart_v2/M0_start101,"
            "outputs/finalization/highpower_multistart_v2/M0_start303,"
            "outputs/finalization/highpower_multistart_v2/M0_start404"
        ),
    )
    ap.add_argument("--out", default="outputs/forward_tests/phase1_closure")
    ap.add_argument("--draws-per-run", type=int, default=900)
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--convention", choices=["A", "B"], default="A")
    ap.add_argument("--z-probe", default="0.20,0.35,0.50,0.62")
    ap.add_argument("--s8om-fid", type=float, default=0.589)

    ap.add_argument("--gate-min-success-frac", type=float, default=0.995)
    ap.add_argument("--gate-min-direction-coherence", type=float, default=0.8)
    ap.add_argument("--fail-on-gate", action="store_true")
    args = ap.parse_args()

    if float(args.s8om_fid) <= 0.0:
        raise ValueError("s8om-fid must be positive.")
    if not (0.0 <= float(args.gate_min_success_frac) <= 1.0):
        raise ValueError("gate-min-success-frac must be in [0,1].")
    if not (0.0 <= float(args.gate_min_direction_coherence) <= 1.0):
        raise ValueError("gate-min-direction-coherence must be in [0,1].")

    run_dirs = _parse_run_dirs(args.run_dirs)
    z_probe = np.asarray(_parse_floats_csv(args.z_probe), dtype=float)
    if np.any(np.diff(z_probe) <= 0):
        raise ValueError("z-probe values must be strictly increasing.")
    if np.any(z_probe <= 0.0):
        raise ValueError("z-probe values must be > 0.")

    out_dir = Path(args.out).resolve()
    tab_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    per_run: list[dict[str, Any]] = []
    all_success = []
    direction_signs = []

    for run in run_dirs:
        post = load_mu_forward_posterior(run)
        if post.sigma8_0 is None:
            raise ValueError(f"{run}: missing sigma8_0; cannot run closure test.")
        if z_probe[-1] > float(np.max(post.z_grid)):
            raise ValueError(f"{run}: z-probe max exceeds posterior z-grid max.")

        n_all = int(post.H0.size)
        n_use = min(int(args.draws_per_run), n_all)
        idx = np.sort(rng.choice(n_all, size=n_use, replace=False)) if n_use < n_all else np.arange(n_all, dtype=int)
        ps = _subset_post(post, idx)

        _, ratio = predict_r_gw_em(
            ps,
            z_eval=z_probe,
            convention=str(args.convention),
            allow_extrapolation=False,
        )

        om = np.asarray(ps.omega_m0, dtype=float)
        sig8 = np.asarray(ps.sigma8_0, dtype=float)
        s8 = sig8 * np.sqrt(np.clip(om / 0.3, 1e-12, np.inf))
        s8om = sig8 * np.power(np.clip(om, 1e-12, np.inf), 0.25)
        lens_amp = np.power(s8om / float(args.s8om_fid), 2.0)

        ok_mask = (
            np.all(np.isfinite(ratio), axis=1)
            & np.all(ratio > 0.0, axis=1)
            & np.isfinite(s8)
            & (s8 > 0.0)
            & np.isfinite(lens_amp)
            & (lens_amp > 0.0)
            & np.isfinite(ps.H0)
            & (ps.H0 > 0.0)
            & np.isfinite(om)
            & (om > 0.0)
        )
        success_frac = float(np.mean(ok_mask))
        all_success.append(ok_mask)

        ratio_med = np.percentile(ratio[ok_mask], 50.0, axis=0) if np.any(ok_mask) else np.full_like(z_probe, np.nan)
        sign = float(np.sign(ratio_med[-1] - 1.0)) if np.isfinite(ratio_med[-1]) else 0.0
        direction_signs.append(sign)

        per_run.append(
            {
                "run_dir": str(Path(run).resolve()),
                "draws_total": int(n_all),
                "draws_used": int(n_use),
                "success_fraction": success_frac,
                "ratio_median_by_z": [float(v) for v in ratio_med.tolist()],
                "ratio_highz_median_minus_one": float(ratio_med[-1] - 1.0),
                "s8_stats": _stats_1d(s8[ok_mask]),
                "lensing_amp_proxy_stats": _stats_1d(lens_amp[ok_mask]),
                "h0_stats": _stats_1d(ps.H0[ok_mask]),
                "omega_m0_stats": _stats_1d(om[ok_mask]),
            }
        )

    ok_concat = np.concatenate(all_success) if all_success else np.asarray([], dtype=bool)
    success_global = float(np.mean(ok_concat)) if ok_concat.size > 0 else float("nan")

    sign_arr = np.asarray(direction_signs, dtype=float)
    pos = int(np.sum(sign_arr > 0))
    neg = int(np.sum(sign_arr < 0))
    direction_coherence = float(max(pos, neg) / max(1, sign_arr.size))

    checks = [
        {
            "name": "shared_draw_success_fraction",
            "value": success_global,
            "threshold": float(args.gate_min_success_frac),
            "pass": bool(np.isfinite(success_global)) and (success_global >= float(args.gate_min_success_frac)),
        },
        {
            "name": "ratio_direction_coherence_across_runs",
            "value": direction_coherence,
            "threshold": float(args.gate_min_direction_coherence),
            "pass": direction_coherence >= float(args.gate_min_direction_coherence),
        },
    ]
    strict_pass = bool(all(bool(c["pass"]) for c in checks))

    with (tab_dir / "per_run_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run_dir",
                "draws_total",
                "draws_used",
                "success_fraction",
                "ratio_median_z0.20",
                "ratio_median_z0.35",
                "ratio_median_z0.50",
                "ratio_median_z0.62",
                "s8_p50",
                "lensing_amp_proxy_p50",
            ]
        )
        for row in per_run:
            rm = row["ratio_median_by_z"]
            w.writerow(
                [
                    row["run_dir"],
                    row["draws_total"],
                    row["draws_used"],
                    row["success_fraction"],
                    rm[0],
                    rm[1],
                    rm[2],
                    rm[3],
                    row["s8_stats"]["p50"],
                    row["lensing_amp_proxy_stats"]["p50"],
                ]
            )

    _plot_ratio_medians(z_probe=z_probe, per_run=per_run, out_path=fig_dir / "ratio_median_by_run.png")

    summary = {
        "created_utc": _utc_now(),
        "phase": "phase1_closure",
        "mode": "shared_draw_closure_audit",
        "inputs": {
            "run_dirs": [str(Path(r).resolve()) for r in run_dirs],
            "draws_per_run": int(args.draws_per_run),
            "seed": int(args.seed),
            "convention": str(args.convention),
            "z_probe": [float(v) for v in z_probe.tolist()],
            "s8om_fid": float(args.s8om_fid),
        },
        "combined": {
            "success_fraction": success_global,
            "direction_coherence_fraction": direction_coherence,
            "direction_sign_counts": {"positive": pos, "negative": neg, "total_runs": int(sign_arr.size)},
        },
        "per_run": per_run,
        "strict_gate": {
            "pass": strict_pass,
            "checks": checks,
        },
        "interpretation_note": (
            "This phase verifies technical cross-channel evaluability from one draw set. "
            "It does not by itself prove physical tied-parameter sufficiency."
        ),
    }
    _write_json_atomic(tab_dir / "summary.json", summary)

    lines = [
        "# Forward Test Phase 1: Shared-Draw Closure Audit",
        "",
        f"- Generated UTC: `{summary['created_utc']}`",
        f"- Shared-draw success fraction: `{success_global:.4f}`",
        f"- Ratio-direction coherence across runs: `{direction_coherence:.3f}`",
        "",
        "## Strict Gate",
        "",
        f"- Result: `{'PASS' if strict_pass else 'FAIL'}`",
    ]
    for c in checks:
        lines.append(f"- `{c['name']}`: `{'PASS' if c['pass'] else 'FAIL'}` (value={c['value']})")
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "- This is a closure/evaluability test, not a full tied-EFT global fit.",
            "",
            "## Artifacts",
            "",
            "- `tables/summary.json`",
            "- `tables/per_run_summary.csv`",
            "- `figures/ratio_median_by_run.png`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {tab_dir / 'summary.json'}")
    print(f"[done] strict_gate={'PASS' if strict_pass else 'FAIL'}")

    if bool(args.fail_on_gate) and (not strict_pass):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

