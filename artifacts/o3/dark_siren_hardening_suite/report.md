# Dark-Siren Hardening Suite (O3)

- Generated: `2026-02-13T19:40:42Z`
- Command: `scripts/run_dark_siren_hardening_suite.py --config configs/dark_siren_hardening_suite_o3_final.json --threads 16`
- Repo: `/home/primary/O3-Modified-Gravity-Hubble-Resolution`
- Git SHA: `4b310bfb0cac69cd2670456a0386639ba2868cd3` dirty=True
- Baseline gap run: `/home/primary/PROJECT/outputs/dark_siren_o3_injection_logit_20260209_055801UTC` label `M0_start101`

## Baseline Reproduction

| metric                         | value              |
| ------------------------------ | ------------------ |
| ref_delta_lpd_total            | 3.669945265393494  |
| recompute_delta_lpd_total      | 3.669945265393494  |
| ref_delta_lpd_total_data       | 2.6700319342049568 |
| recompute_delta_lpd_total_data | 2.6700319342049568 |
| ref_delta_lpd_total_sel        | 0.9999133311885373 |
| recompute_delta_lpd_total_sel  | 0.9999133311885373 |

## 3.1 Selection Nuisance (Required-Failure Scan)

| case       | adversarial min ΔLPD |
| ---------- | -------------------- |
| baseline   | +3.670               |
| bound=0.10 | +3.667               |
| bound=0.20 | +3.667               |
| bound=0.30 | +3.667               |
| bound=0.40 | +3.667               |
| bound=0.60 | +3.667               |
- Required bound for ΔLPD<1: `None`; for ΔLPD<0: `None`
- Plot: `figures/selection_required_failure_scan.png`

## Spectral-Only Baseline (Mechanism Channel)

| item                 | value  |
| -------------------- | ------ |
| ΔLPD (spectral-only) | +3.634 |

## 1.1 Non-Circular Splits (Frozen Assignments)

### distance_bins

| bin | n  | ΔLPD   | alpha_hat (lin., mean) |
| --- | -- | ------ | ---------------------- |
| 0   | 12 | -0.517 | 0.73                   |
| 1   | 12 | +0.566 | 1.15                   |
| 2   | 12 | +3.547 | 2.00                   |
- coherence_metric_linearized: `2.151`

### catalog_z_bins

| bin | n  | ΔLPD   | alpha_hat (lin., mean) |
| --- | -- | ------ | ---------------------- |
| 0   | 12 | -0.417 | 0.76                   |
| 1   | 12 | +0.003 | 0.91                   |
| 2   | 12 | +3.846 | 2.00                   |
- coherence_metric_linearized: `2.453`

### network

| bin | n  | ΔLPD   | alpha_hat (lin., mean) |
| --- | -- | ------ | ---------------------- |
| HL  | 11 | +0.122 | 0.95                   |
| HLV | 25 | +3.668 | 2.00                   |
- coherence_metric_linearized: `1.380`

### epoch

| bin | n  | ΔLPD   | alpha_hat (lin., mean) |
| --- | -- | ------ | ---------------------- |
| O3a | 0  | +nan   | nan                    |
| O3b | 36 | +3.670 | 2.00                   |
- coherence_metric_linearized: `0.000`

### chirp_mass_bins

| bin | n  | ΔLPD   | alpha_hat (lin., mean) |
| --- | -- | ------ | ---------------------- |
| 0   | 12 | -0.316 | 0.79                   |
| 1   | 12 | +0.200 | 0.99                   |
| 2   | 12 | +3.683 | 2.00                   |
- coherence_metric_linearized: `2.180`

## 2. Permutation / Null Collapse Tests

### scramble_catalog_redshift_hist_spectral_only

| item                      | value  |
| ------------------------- | ------ |
| observed ΔLPD (total)     | +3.645 |
| observed ΔLPD (data-only) | +2.649 |
| null mean (total)         | +3.381 |
| null sd (total)           | 0.147  |
| p(null ≥ obs) total       | 0.0498 |
| null mean (data-only)     | +2.270 |
| null sd (data-only)       | 0.135  |
| p(null ≥ obs) data        | 0.0249 |
- Plot: `figures/redshift_scramble_null_hist.png`

### permute_catalog_terms_across_events

| item                      | value  |
| ------------------------- | ------ |
| observed ΔLPD (total)     | +3.670 |
| observed ΔLPD (data-only) | +2.670 |
| null mean (total)         | +3.647 |
| null sd (total)           | 0.016  |
| p(null ≥ obs) total       | 0.0050 |
- Plot: `figures/permutation_null_hist.png`

### permute_missing_terms_across_events

| item                      | value  |
| ------------------------- | ------ |
| observed ΔLPD (total)     | +3.670 |
| observed ΔLPD (data-only) | +2.670 |
| null mean (total)         | +3.648 |
| null sd (total)           | 0.015  |
| p(null ≥ obs) total       | 0.0050 |
| null mean (data-only)     | +2.633 |
| null sd (data-only)       | 0.029  |
| p(null ≥ obs) data        | 0.0050 |
- Plot: `figures/missing_host_swap_null_hist.png`

### permute_catalog_redshift_hist_across_events_spectral_only

| item                      | value  |
| ------------------------- | ------ |
| observed ΔLPD (total)     | +3.645 |
| observed ΔLPD (data-only) | +2.649 |
| null mean (total)         | +3.668 |
| null sd (total)           | 0.038  |
| p(null ≥ obs) total       | 0.7164 |
| null mean (data-only)     | +2.670 |
| null sd (data-only)       | 0.033  |
| p(null ≥ obs) data        | 0.7264 |
- Plot: `figures/catalog_hist_swap_null_hist.png`

### scramble_missing_host_prior_basez_spectral_only

| item                      | value  |
| ------------------------- | ------ |
| observed ΔLPD (total)     | +3.645 |
| observed ΔLPD (data-only) | +2.649 |
| null mean (total)         | +3.645 |
| null sd (total)           | 0.000  |
| p(null ≥ obs) total       | 0.9453 |
| null mean (data-only)     | +2.649 |
| null sd (data-only)       | 0.000  |
| p(null ≥ obs) data        | 0.7363 |
- Plot: `figures/missing_host_prior_scramble_null_hist.png`

## 3.2 Global Catalogue/Photo-z Stress (Top-N Leverage Events)

| item                          | value                                                                               |
| ----------------------------- | ----------------------------------------------------------------------------------- |
| top events                    | GW200308_173609, GW200220_061928, GW200208_130117, GW200219_094415, GW191230_180458 |
| baseline ΔLPD (spectral-only) | +3.634                                                                              |
| min ΔLPD on photo-z grid      | +1.976                                                                              |
| min ΔLPD on completeness grid | +3.609                                                                              |

## 3.3 PE / Waveform Robustness (Analysis Swaps)

| item                  | value                            |
| --------------------- | -------------------------------- |
| top events            | GW200308_173609, GW200220_061928 |
| min ΔLPD across swaps | +3.625                           |
- Full rows: `tables/pe_waveform_stress_rows.csv`

## 4.1 Injection Consistency Check (MG truth; spectral-only)

| item        | value                    |
| ----------- | ------------------------ |
| n_rep       | 64                       |
| mean ΔLPD   | +2.716                   |
| p16/p50/p84 | +2.533 / +2.682 / +2.877 |
- Plot: `figures/mg_injection_check_hist.png`

## 4.2 Injection Null Calibration (GR truth; spectral-only)

| item        | value                    |
| ----------- | ------------------------ |
| n_rep       | 512                      |
| mean ΔLPD   | +2.643                   |
| p16/p50/p84 | +2.511 / +2.630 / +2.768 |
| max ΔLPD    | +3.056                   |
- Plot: `figures/gr_injection_check_hist.png`

## 2.1 Calibrated Significance (GR-truth Injection Null)

| case                       | ΔLPD   | p (one-sided) | Z (one-sided) |
| -------------------------- | ------ | ------------- | ------------- |
| baseline (full)            | +3.670 | 0.001949      | 2.89          |
| selection min (bound=0.20) | +3.667 | 0.001949      | 2.89          |
| photo-z stressed min       | +1.976 | 1             | nan           |
- Notes: p/Z are computed against `n=512` GR-truth injections from Section 4.2 (spectral-only generator; dominant-channel calibration).

## 5 Interpretability Memo

- Note: `interpretability.md`

## 6 Predictions / Falsifiers

- Note: `predictions.md`
