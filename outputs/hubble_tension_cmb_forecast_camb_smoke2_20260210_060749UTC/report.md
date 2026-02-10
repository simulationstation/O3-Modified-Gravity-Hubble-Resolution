# CMB Forecast Under MG-Truth Assumption

- Generated UTC: `2026-02-10T06:07:58Z`
- Run dir: `/home/primary/O3-Modified-Gravity-Hubble-Resolution/outputs/finalization/highpower_multistart_v2/M0_start101`
- Draws evaluated: `1/1`
- Planck lensing dataset: `consext8`

## Headline Predicted Lensing Shifts

- Near L~100 (actual `106`): median `ΔC/C = -21.93%` (68%: `-21.93%` to `-21.93%`).
- Near L~300 (actual `286`): median `ΔC/C = -13.10%` (68%: `-13.10%` to `-13.10%`).

## Fit to Planck Lensing Bandpowers

- MG draw chi2 (median): `88.607`
- Planck-reference model chi2: `9.041`
- P(draw beats reference): `0.0%`

## Derived Amplitude Proxies

- `A_lens_proxy` median: `0.9256`
- `A_lens_proxy` p16/p84: `0.9256` / `0.9256`

## Scope Notes

- This forecast propagates draw-level cosmology through standard CAMB transfer functions.
- It is a lensing-focused CMB prediction target, not a full MG TT/TE/EE perturbation-sector refit.

## Artifacts

- `tables/summary.json`
- `tables/clpp_draws.npz`
- `figures/clpp_bandpowers_vs_models.png`
- `figures/clpp_ratio_to_ref.png`
- `figures/clpp_chi2_distribution.png`
