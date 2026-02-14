# Joint Transfer-Bias Fit Summary

- Created UTC: `2026-02-09T06:20:06Z`
- Effective posterior draws: `8000`
- Theta samples evaluated: `32768`
- log Bayes factor (transfer vs no-transfer): `-0.5331`
- Relief posterior (mean / p16 / p50 / p84): `0.8510 / 0.8313 / 0.8561 / 0.8701`
- Dominant transfer term: `delta_h0_ladder_local`

## Transfer Term Dominance (posterior-weighted mean abs loglike shift)

- `beta_ia_sn`: `0.880975`
- `beta_cc_cc`: `0.291155`
- `beta_bao_bao`: `0.375361`
- `delta_h0_ladder_local`: `0.909342`

## Parameter Posteriors

- `beta_ia` mean/p16/p50/p84/sd: `-0.0274364` / `-0.067498` / `-0.0295735` / `0.0130397` / `0.0413015`
- `beta_cc` mean/p16/p50/p84/sd: `-0.0334336` / `-0.171967` / `-0.0341209` / `0.105485` / `0.144545`
- `delta_h0_ladder` mean/p16/p50/p84/sd: `0.507493` / `-0.414479` / `0.502375` / `1.42931` / `0.929698`
- `beta_bao` mean/p16/p50/p84/sd: `-0.00702704` / `-0.0248135` / `-0.00724163` / `0.0109364` / `0.0189977`

## Relief Sensitivity (weighted corr / slope)

- `beta_ia` corr/slope: `-0.176334` / `-0.0877977`
- `beta_cc` corr/slope: `-0.471803` / `-0.067123`
- `delta_h0_ladder` corr/slope: `0.652155` / `0.0144252`
- `beta_bao` corr/slope: `-0.022552` / `-0.0244117`

## Artifacts

- `tables/summary.json`
- `tables/theta_posterior_rows.csv`
- `figures/relief_posterior.png`
- `figures/transfer_param_posteriors.png`
