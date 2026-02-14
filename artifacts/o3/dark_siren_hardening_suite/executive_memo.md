# Executive Memo: Dark-Siren Hardening Suite (O3)

- Output: `dark_siren_hardening_suite_o3_20260213_192819UTC`
- Baseline ΔLPD (full cached terms): `+3.670`
- Spectral-only ΔLPD: `+3.634`

## Structure-Dependence Nulls (Empirical)

- `scramble_catalog_redshift_hist_spectral_only`: observed `+3.645`, null mean `+3.381 ± 0.147`, p `0.0498`
- `permute_catalog_terms_across_events`: observed `+3.670`, null mean `+3.647 ± 0.016`, p `0.0050`
- `permute_missing_terms_across_events`: observed `+3.670`, null mean `+3.648 ± 0.015`, p `0.0050`
- `permute_catalog_redshift_hist_across_events_spectral_only`: observed `+3.645`, null mean `+3.668 ± 0.038`, p `0.7164`
- `scramble_missing_host_prior_basez_spectral_only`: observed `+3.645`, null mean `+3.645 ± 0.000`, p `0.9453`

## Non-Circular Coherence (Frozen Assignments)

- `distance_bins` coherence metric (linearised α scan): `2.151`
- `catalog_z_bins` coherence metric (linearised α scan): `2.453`
- `network` coherence metric (linearised α scan): `1.380`
- `epoch` coherence metric (linearised α scan): `0.000`
- `chirp_mass_bins` coherence metric (linearised α scan): `2.180`

## Required-Failure (Selection Deformation)

- Required bound for ΔLPD<1: `None`; for ΔLPD<0: `None`

## Catalogue/Photo-z Stress (Global Top-N)

- Min ΔLPD on tested photo-z grid: `+1.976` (thresholds: None).

## GR-Truth Calibration

- GR-truth injection null (spectral-only): mean `+2.643`, max `+3.056` over `n=512`.
- Calibrated p/Z table: `tables/calibrated_pz_table.csv`.

## Falsifiers (Operational)

- If improved catalogue modelling within realistic external priors drives ΔLPD→0, the MG-propagation interpretation weakens.
- If non-circular splits show sign flips or strong incoherence beyond posterior uncertainty, it argues for misspecification.
- If future larger samples remove the distance-concentration trend, a cumulative propagation effect is disfavoured.
