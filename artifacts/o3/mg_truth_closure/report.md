# MG-Truth Causal-Closure Suite (Dark Sirens)

- Generated: `2026-02-14T21:48:11+00:00`

- Gap run root: `/home/primary/PROJECT/outputs/dark_siren_o3_injection_logit_20260209_055801UTC` label `M0_start101`

- Recon run dir: `/home/primary/PROJECT/outputs/finalization/highpower_multistart_v2/M0_start101`

- n_rep per suite: `256`; threads: `8`



## Observed

- Observed ΔLPD_total (real data): `3.66995`



## Suites

### truth=mg alpha=1.0 z_mode=catalog_hist cat_mode=catalog

- delta_lpd_total: mean `+3.504`, p16/p50/p84 `+3.239` / `+3.442` / `+3.763`, max `+4.722`

- delta_lpd_data: mean `+2.487`, p16/p50/p84 `+2.266` / `+2.440` / `+2.682`, max `+3.460`

- delta_lpd_sel: mean `+1.017`, p16/p50/p84 `+0.953` / `+1.007` / `+1.083`, max `+1.262`



### truth=gr alpha=0.0 z_mode=catalog_hist cat_mode=catalog

- delta_lpd_total: mean `+3.313`, p16/p50/p84 `+3.153` / `+3.309` / `+3.468`, max `+3.995`

- delta_lpd_data: mean `+2.346`, p16/p50/p84 `+2.217` / `+2.343` / `+2.477`, max `+2.838`

- delta_lpd_sel: mean `+0.967`, p16/p50/p84 `+0.928` / `+0.964` / `+1.006`, max `+1.157`



### truth=mg alpha=1.0 z_mode=missing_prior cat_mode=toy_uniform_z

- delta_lpd_total: mean `+0.637`, p16/p50/p84 `+0.278` / `+0.576` / `+0.951`, max `+2.351`

- delta_lpd_data: mean `-0.309`, p16/p50/p84 `-0.615` / `-0.366` / `-0.032`, max `+1.073`

- delta_lpd_sel: mean `+0.947`, p16/p50/p84 `+0.891` / `+0.932` / `+0.996`, max `+1.278`



## Closure Check

- Fraction of MG-truth catalog replicates with ΔLPD_total ≥ observed: `0.2188`

- Fraction of GR-truth catalog replicates with ΔLPD_total ≥ observed: `0.0195`



## Dose–Response (Truth Alpha Dial)

- Table: `tables/alpha_dose_response_summary.csv`

- Figures: `figures/dose_response_mean_delta_lpd_vs_alpha.png`, `figures/dose_response_p_ge_observed_vs_alpha.png`



## GR-Truth Generator Variants (Harder Null)

- Table: `tables/gr_truth_generator_variants_summary.csv`



## Structure-Destruction Controls

- Table: `tables/structure_destruction_controls_summary.csv`



## Artifacts

- Summary: `summary.json`

- Figures: `figures/delta_lpd_hist_overlay.png`, `figures/mg_truth_data_vs_sel_scatter.png`

- Samples CSVs: `tables/truth_*_samples.csv` (one per suite tag)
