# Joint Transfer-Bias Fit Summary

- Created UTC: `2026-02-10T05:50:00Z`
- Effective posterior draws: `512`
- Theta samples evaluated: `512`
- log Bayes factor (transfer vs no-transfer): `-0.4887`
- Relief posterior (mean / p16 / p50 / p84): `0.8545 / 0.8320 / 0.8593 / 0.8765`
- Dominant transfer term: `beta_ia_sn`

## Transfer Term Dominance (posterior-weighted mean abs loglike shift)

- `beta_ia_sn`: `1.004826`
- `beta_cc_cc`: `0.280849`
- `beta_bao_bao`: `0.347985`
- `delta_h0_ladder_local`: `0.925527`

## Parameter Posteriors

- `beta_ia` mean/p16/p50/p84/sd: `-0.0313392` / `-0.0705228` / `-0.0388342` / `0.0128693` / `0.0420269`
- `beta_cc` mean/p16/p50/p84/sd: `-0.0263529` / `-0.156258` / `-0.0149041` / `0.10634` / `0.146426`
- `delta_h0_ladder` mean/p16/p50/p84/sd: `0.545105` / `-0.36893` / `0.506209` / `1.3892` / `0.93151`
- `beta_bao` mean/p16/p50/p84/sd: `-0.00552368` / `-0.0222707` / `-0.00508026` / `0.0111357` / `0.018948`

## Relief Sensitivity (weighted corr / slope)

- `beta_ia` corr/slope: `-0.280634` / `-0.146439`
- `beta_cc` corr/slope: `-0.413711` / `-0.0619616`
- `delta_h0_ladder` corr/slope: `0.396683` / `0.00933899`
- `beta_bao` corr/slope: `0.0650814` / `0.0753249`

## Artifacts

- `tables/summary.json`
- `tables/theta_posterior_rows.csv`
- `figures/relief_posterior.png`
- `figures/transfer_param_posteriors.png`
