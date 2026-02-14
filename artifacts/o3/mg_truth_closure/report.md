# MG-Truth Causal-Closure Suite (Dark Sirens)

- Generated: `2026-02-14T19:42:18+00:00`

- Gap run root: `/home/primary/PROJECT/outputs/dark_siren_o3_injection_logit_20260209_055801UTC` label `M0_start101`

- Recon run dir: `/home/primary/PROJECT/outputs/finalization/highpower_multistart_v2/M0_start101`

- n_rep per suite: `512`; threads: `8`



## Observed

- Observed ΔLPD_total (real data): `3.66995`



## Suites

### truth=mg z_mode=catalog_hist cat_mode=catalog

- delta_lpd_total: mean `+3.523`, p16/p50/p84 `+3.266` / `+3.471` / `+3.786`, max `+4.974`

- delta_lpd_data: mean `+2.503`, p16/p50/p84 `+2.298` / `+2.467` / `+2.701`, max `+3.655`

- delta_lpd_sel: mean `+1.019`, p16/p50/p84 `+0.957` / `+1.010` / `+1.082`, max `+1.347`



### truth=gr z_mode=catalog_hist cat_mode=catalog

- delta_lpd_total: mean `+3.311`, p16/p50/p84 `+3.148` / `+3.289` / `+3.462`, max `+3.815`

- delta_lpd_data: mean `+2.346`, p16/p50/p84 `+2.206` / `+2.330` / `+2.474`, max `+2.778`

- delta_lpd_sel: mean `+0.965`, p16/p50/p84 `+0.930` / `+0.962` / `+1.002`, max `+1.128`



### truth=mg z_mode=missing_prior cat_mode=toy_uniform_z

- delta_lpd_total: mean `+0.621`, p16/p50/p84 `+0.283` / `+0.541` / `+0.948`, max `+3.495`

- delta_lpd_data: mean `-0.324`, p16/p50/p84 `-0.614` / `-0.389` / `-0.038`, max `+2.035`

- delta_lpd_sel: mean `+0.945`, p16/p50/p84 `+0.890` / `+0.931` / `+0.998`, max `+1.460`



## Closure Check

- Fraction of MG-truth catalog replicates with ΔLPD_total ≥ observed: `0.2500`



## Artifacts

- Summary: `summary.json`

- Figures: `figures/delta_lpd_hist_overlay.png`, `figures/mg_truth_data_vs_sel_scatter.png`

- Samples CSVs: `tables/truth_*_samples.csv`
