# Reproducibility Guide (Reviewer Seed + Optional Full Reruns)

This repository supports two “levels” of reproducibility:

1. **Seed / headline verification (fast, no large downloads)**: reads the curated `artifacts/` bundle and produces a timestamped `report.md` + `summary.json`.
2. **Full reruns (slow, requires public inputs)**: reruns parts of the O3 dark-siren and Hubble follow-on pipelines once you download the large public inputs (GWTC PE files, GLADE+ index products, spectroscopy catalogues, etc.).

## 0) Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[sirens,optical_bias]'
```

## 1) Seed Reproduction (Headlines)

```bash
make reproduce
```

Writes:

- `outputs/reviewer_seed_<timestamp>/summary.json`
- `outputs/reviewer_seed_<timestamp>/report.md`

This target does **not** rerun the full pipelines; it is a consistency snapshot for review.

## 2) Optional: Rerun The Spec-z “Coverage Maxout” Audit

This is the strongest currently-implemented public-data attack on the remaining photo-$z$ escape hatch, under strict shifted-sky false-match controls.

### Inputs you must provide

- A local GLADE+ HEALPix index directory (luminosity-weighted). This repo expects the prebuilt index format used by `src/entropy_horizon_recon/dark_sirens.py::load_gladeplus_index`.
- Local spec-$z$ catalog caches (NPZ) for the selected sources (DESI/SDSS/6dF/2dF/GAMA/2MRS). The maxout runner reads these from `data/cache/specz_catalogs/` by default.

### Run

Start from the template config in `configs/dark_siren_specz_coverage_maxout_o3.json` and edit:

- `glade.index_lumB`
- `specz_cache_dir`
- (optionally) the spec-$z$ source list (tiers A/B/C)

Then run:

```bash
PYTHONPATH=src python3 scripts/run_dark_siren_specz_coverage_maxout.py \
  --config configs/dark_siren_specz_coverage_maxout_o3.json \
  --out outputs/dark_siren_specz_coverage_maxout_rerun
```

The run writes `summary.json` + `report.md` + a full coverage grid and the strict false-match gate table.

## 3) Optional: O3 Dark-Siren Hardening Suite

The full hardening suite (`scripts/run_dark_siren_hardening_suite.py`) reproduces the calibrated O3 preference and runs the adversarial nuisance scans + structure-destruction nulls + injection checks that are referenced by `artifacts/o3/dark_siren_hardening_suite/`.

This suite requires GWTC PE products and GLADE+ index products on disk.

To run with the provided configuration:

```bash
PYTHONPATH=src python3 scripts/run_dark_siren_hardening_suite.py \
  --config configs/dark_siren_hardening_suite_o3_final.json \
  --threads 16 \
  --out outputs/dark_siren_hardening_suite_o3_rerun
```

Important:

- Some configs in `configs/` include absolute paths from the author’s environment. For an independent rerun, edit these paths to point at your local data locations.

## 4) Optional: Hubble Follow-on

The Hubble “transfer / inference-bias” scripts live in `scripts/run_hubble_tension_*.py` and expect the supporting intermediate outputs described in `README.md`.

These are best run via the detached launchers in `scripts/launch_*.sh` (they log `pid.txt` and `run.log` under the chosen output directory).

