#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


PHASES = {
    "phase1_closure": "Build tied GW->scalar closure model and propagation hooks.",
    "phase2_growth_s8": "Posterior-predictive growth/S8 forward test.",
    "phase3_mu_sigma": "Joint mu-Sigma lensing/growth consistency test.",
    "phase4_distance_ratio": "GW-vs-EM luminosity-distance ratio trend test.",
    "phase5_external_constraints": "Constraint stress test (non-local channels: BBN/recombination, primary CMB).",
    "phase6_model_selection": "Global model comparison (M0..M3).",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Create design-ready stub folders for forward-test phases.")
    ap.add_argument("--phase", required=True, choices=sorted(PHASES.keys()))
    ap.add_argument("--out-root", default="outputs/forward_tests")
    ap.add_argument("--status", default="design_ready", choices=["design_ready", "planned", "in_progress"])
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = (repo_root / args.out_root / args.phase).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    status_json = {
        "created_utc": _utc_now(),
        "phase": args.phase,
        "status": args.status,
        "goal": PHASES[args.phase],
        "inputs": [],
        "planned_outputs": [],
        "owner": "unassigned",
        "notes": [
            "Stub scaffold only. Fill inputs/outputs and wire concrete run scripts before execution.",
            "Designed to avoid accidental heavy compute."
        ],
    }
    _write_json(out_dir / "status.json", status_json)

    readme = [
        f"# {args.phase}",
        "",
        f"Goal: {PHASES[args.phase]}",
        "",
        "## What To Fill",
        "1. Exact input artifacts and selectors.",
        "2. Statistical model and likelihood terms.",
        "3. Pass/fail metrics and thresholds.",
        "4. Output table/figure schema.",
        "",
        "## Run Contract",
        "- Keep runs resumable.",
        "- Emit `tables/summary.json`.",
        "- Emit `report.md` with assumptions and caveats.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(f"[stub] created {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
