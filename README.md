# O3 Modified Gravity Hubble Resolution (Reviewer Seed)

This repository is a review-focused seed export of the code and *curated* result artifacts used for:

1. **GWTC-3 / O3 dark-siren posterior-predictive scoring** of a fixed modified-propagation history against an internal GR propagation baseline (selection-normalised, injection-calibrated).
2. **Follow-on Hubble-tension transfer / inference-bias analysis** conditioned on the same propagation deformation.

The intent is that a reviewer can:

- verify the headline numbers quickly (`make reproduce`), and
- rerun the main pipelines if they download the (large) public inputs.

---

## What This Repository Is Testing

The model path is not “free modified gravity”; it is a fixed propagation history derived upstream and then evaluated out-of-sample on dark sirens.

Conceptually:

- **Stage A (geometry → entropy slope)**: reconstruct an effective horizon/entropy slope deformation $\mu(A)$ from late-time cosmology in the “minimal running-$M_\ast$” embedding implemented in `src/entropy_horizon_recon/`.
- **Stage B ($\mu(A)$ → GW propagation)**: map to a GW propagation ratio $R(z)=d_L^{GW}(z)/d_L^{EM}(z)$.
- **Stage C (dark sirens)**: score the fixed $R(z)$ history against an internal GR baseline using a joint posterior-predictive log score $\Delta \mathrm{LPD}$ with explicit selection normalisation and injection-based calibration/hardening.
- **Stage D (Hubble tension)**: propagate the same deformation into late-time standard-ruler inversions to quantify the induced “inference wedge” (transfer-bias) on $H_0$.

---

## Included Contents

- `src/entropy_horizon_recon/`: core package (sirens, selection models, mapping utilities, lensing ingests).
- `scripts/`: runnable entrypoints for the O3 dark-siren hardening suite, spec-$z$ audits, and Hubble follow-on analyses.
- `update_paper/`: latest Hubble-tension manuscript source + PDF (`hubble_tension_hypothesis.tex/.pdf`).
- `entropy_slope_paper/`: entropy-slope letter source + PDF (context/foundation for the mapping).
- `CQG_PAPER/`: CQG-style manuscript source + PDF for the Hubble-tension implications paper (`hubble_tension_cqg.tex/.pdf`).
- `CQG_DARK_SIREN/`: CQG-style manuscript source + PDF for the O3 dark-siren scoring paper (`dark_siren_cqg.tex/.pdf`).
- `artifacts/`: **curated**, small “reviewer seed” outputs (tables/figures/reports). These are what `make reproduce` reads.

Notes:

- Large intermediate runs live under `outputs/` when generated locally; `outputs/` is git-ignored by design.
- Large public inputs (GWTC PE files, GLADE+ indices, DESI/SDSS spectroscopy dumps, etc.) are not vendored in git.

---

## Seed Replication (Headline Verification)

One command creates a timestamped reproduction folder containing `report.md` + `summary.json` with the key numbers:

```bash
make reproduce
```

This uses the curated `artifacts/` bundle and does **not** require downloading the full public inputs.

To rebuild the bundled CQG paper PDF (optional):

```bash
make cqg-paper
```

To rebuild the bundled CQG dark-siren PDF (optional):

```bash
make cqg-dark-siren
```

Output example:

- `outputs/reviewer_seed_<timestamp>/report.md`
- `outputs/reviewer_seed_<timestamp>/summary.json`

For full reruns, see `README_reproduce.md`.

---

## Headline Snapshot (From `artifacts/`)

These are the current curated “seed” headlines; `make reproduce` writes the same numbers to a timestamped `report.md`:

- O3 dark sirens (full score, selection-normalised): `ΔLPD_tot ≈ +3.670` (`ΔLPD_data ≈ +2.670`), with calibrated one-sided `p ≈ 0.00195` (`Z ≈ 2.89`) against the bundled GR-truth spectral-only null generator.
- MG-truth causal-closure suite (ancillary, spectral-only + selection): under MG-truth forward catalogs generated from the same fixed propagation template, the observed `ΔLPD_tot ≈ 3.670` lands at `p ≈ 0.25` in the MG-truth distribution but only `p ≈ 0.027` under a GR-truth distribution; removing catalog structure collapses the typical score scale to `ΔLPD_tot ~ O(1)`.
- Spec-$z$ override (strict shifted-sky gate; Tier A; `r = 10″`, `K = 20000`): median anchored host-weight proxy across the top-3 gate events `≈ 6.1%`, with `ΔLPD_tot ≈ 3.644` (non-decreasing with anchored coverage in the clean-radius regime).
- Hubble transfer-bias relief posterior (MC-calibrated): mean `≈ 0.246` with `p16/p50/p84 ≈ 0.205 / 0.240 / 0.277`.
- Planck lensing response refit: baseline MG projection suppresses `C_L^{\\phi\\phi}` near `L~100` by `≈ -17.6%` (median), while the constrained MG-aware response refit leaves `≈ -0.25%` residual (median) and achieves `χ²` median `≈ 8.06` versus Planck reference `≈ 9.04`.

---

## Key Result Artifacts (Vendored)

Dark sirens (O3):

- Hardening suite (baseline reproduction, selection nuisance scan, null/permutation tests, injection checks):
  - `artifacts/o3/dark_siren_hardening_suite/report.md`
  - `artifacts/o3/dark_siren_hardening_suite/tables/baseline_recompute.json`
  - `artifacts/o3/dark_siren_hardening_suite/tables/calibrated_pz_table.csv`
- MG-truth causal-closure suite (forward MG/GR truth catalogs, scored under the same ΔLPD definition):
  - `artifacts/o3/mg_truth_closure/report.md`
  - `artifacts/o3/mg_truth_closure/summary.json`
- Spec-$z$ “coverage maxout” audit (strict false-match gated):
  - `artifacts/o3/specz_coverage_maxout/summary.json`
  - `artifacts/o3/specz_coverage_maxout/tables/false_match_gate_by_radius.csv`
  - `artifacts/o3/specz_coverage_maxout/tables/override_score_best_points.csv`

Hubble follow-on:

- Transfer-bias relief posterior:
  - `artifacts/hubble/final_relief_posterior/final_relief_posterior_summary.json`
- Planck lensing response refit (amplitude + mild $\ell$-tilt):
  - `artifacts/hubble/mg_lensing_refit/tables/summary.json`
- Joint transfer fit summary:
  - `artifacts/hubble/joint_transfer_bias_fit/tables/summary.json`

Artifact provenance:

- `artifacts/manifest.json`

---

## Environment

Typical local setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[sirens,optical_bias]'
```

---

## Data Source DOI Manifest (Primary)

- GWTC-3 PE products: DOI `10.1103/PhysRevX.13.041039`
- GLADE+ galaxy catalogue: DOI `10.1093/mnras/stac1443`
- O3 search-sensitivity injections: DOI `10.5281/zenodo.7890437`
- Code + reproducibility archive for the Hubble-tension manuscript: DOI `10.5281/zenodo.18640608`

---

## Notes For Reviewers

- This repo is intentionally structured to keep the main O3 dark-siren scoring + the Hubble transfer analysis on the same fixed propagation history.
- The “seed” command (`make reproduce`) is a headline verification tool; it does not claim to replace a full rerun with downloaded public inputs.

---

## Ancillary: MG-Truth Causal-Closure Suite (Non-Seed)

This is a forward-model “closure” test: if the fixed propagation history used in the Hubble-tension analysis were true, does it *generically* produce a dark-siren ΔLPD excursion on the observed scale when analysed under a GR baseline, using the same catalog/selection machinery?

Command (writes to a timestamped `outputs/mg_truth_closure_<UTC>/` directory):

```bash
PYTHONPATH=src .venv/bin/python scripts/run_dark_siren_mg_truth_closure_suite.py --n-rep 512 --threads 8
```

Outputs:

- `outputs/mg_truth_closure_<UTC>/report.md`
- `outputs/mg_truth_closure_<UTC>/summary.json`
- `outputs/mg_truth_closure_<UTC>/figures/delta_lpd_hist_overlay.png`
