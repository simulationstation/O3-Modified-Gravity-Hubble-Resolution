# Joint Transfer-Bias Fit Summary

- Created UTC: `2026-02-10T05:49:33Z`
- Effective posterior draws: `64`
- Theta samples evaluated: `16`
- log Bayes factor (transfer vs no-transfer): `-0.8269`
- Relief posterior (mean / p16 / p50 / p84): `0.8371 / 0.8096 / 0.8390 / 0.8542`
- Dominant transfer term: `beta_ia_sn`

## Transfer Term Dominance (posterior-weighted mean abs loglike shift)

- `beta_ia_sn`: `0.955559`
- `beta_cc_cc`: `0.282909`
- `beta_bao_bao`: `0.329288`
- `delta_h0_ladder_local`: `0.604048`

## Parameter Posteriors

- `beta_ia` mean/p16/p50/p84/sd: `-0.0275542` / `-0.111582` / `-0.0313936` / `0.0177633` / `0.0556709`
- `beta_cc` mean/p16/p50/p84/sd: `-0.026287` / `-0.188683` / `-0.0710052` / `0.0905781` / `0.14071`
- `delta_h0_ladder` mean/p16/p50/p84/sd: `0.131046` / `-0.99495` / `0.178906` / `0.949158` / `0.712891`
- `beta_bao` mean/p16/p50/p84/sd: `0.000175157` / `-0.0275027` / `-0.00158313` / `0.0122497` / `0.022935`

## Relief Sensitivity (weighted corr / slope)

- `beta_ia` corr/slope: `-0.287309` / `-0.107537`
- `beta_cc` corr/slope: `-0.570407` / `-0.0844693`
- `delta_h0_ladder` corr/slope: `-0.182441` / `-0.00533258`
- `beta_bao` corr/slope: `0.300387` / `0.27291`

## Artifacts

- `tables/summary.json`
- `tables/theta_posterior_rows.csv`
- `figures/relief_posterior.png`
- `figures/transfer_param_posteriors.png`
