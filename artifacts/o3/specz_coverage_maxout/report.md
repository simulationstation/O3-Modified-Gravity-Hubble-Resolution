# Spec-z Coverage Maxout (Pre-O4)
Output directory: `/home/primary/O3-Modified-Gravity-Hubble-Resolution/outputs/dark_siren_specz_coverage_maxout_20260214_090513UTC`
## Inputs
- gap_root: `/home/primary/PROJECT/outputs/dark_siren_o3_injection_logit_20260209_055801UTC`
- run_label: `M0_start101`
- recon_run_dir: `/home/primary/PROJECT/outputs/finalization/highpower_multistart_v2/M0_start101`
- z_max: 0.3
- gate_events: GW200308_173609, GW200220_061928, GW200219_094415

## Legitimacy Gate
- median(shifted/true) < 0.1
- max_event(shifted/true) < 0.3

## Best Legit Operating Points
- Tier A: r=10.0"  K=20000  median frac_weight_true=0.0614
- Tier B: r=10.0"  K=20000  median frac_weight_true=0.0615
- Tier C: r=10.0"  K=20000  median frac_weight_true=0.0646

## Override Scoring At Best Points (Spectral-only + selection)
- Tier A: r=10.0" K=20000  ΔLPD_total=3.644 (baseline 3.634)
- Tier B: r=10.0" K=20000  ΔLPD_total=3.644 (baseline 3.634)

## Artifacts
- coverage grid: `/home/primary/O3-Modified-Gravity-Hubble-Resolution/outputs/dark_siren_specz_coverage_maxout_20260214_090513UTC/tables/coverage_grid_full.csv`
- gate by radius: `/home/primary/O3-Modified-Gravity-Hubble-Resolution/outputs/dark_siren_specz_coverage_maxout_20260214_090513UTC/tables/false_match_gate_by_radius.csv`
- max points: `/home/primary/O3-Modified-Gravity-Hubble-Resolution/outputs/dark_siren_specz_coverage_maxout_20260214_090513UTC/tables/max_coverage_operating_points.csv`
- specz manifest: `/home/primary/O3-Modified-Gravity-Hubble-Resolution/outputs/dark_siren_specz_coverage_maxout_20260214_090513UTC/raw/specz_catalog_manifest.json`
