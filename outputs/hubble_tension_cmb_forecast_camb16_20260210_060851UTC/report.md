# CMB Forecast Under MG-Truth Assumption

- Generated UTC: `2026-02-10T06:10:46Z`
- Run dir: `/home/primary/O3-Modified-Gravity-Hubble-Resolution/outputs/finalization/highpower_multistart_v2/M0_start101`
- Draws evaluated: `16/16`
- Planck lensing dataset: `consext8`

## Headline Predicted Lensing Shifts

- Near L~100 (actual `106`): median `ΔC/C = -14.86%` (68%: `-22.14%` to `-1.62%`).
- Near L~300 (actual `286`): median `ΔC/C = -8.41%` (68%: `-13.73%` to `-4.99%`).

## Fit to Planck Lensing Bandpowers

- MG draw chi2 (median): `48.009`
- Planck-reference model chi2: `9.041`
- P(draw beats reference): `12.5%`

## Derived Amplitude Proxies

- `A_lens_proxy` median: `0.9379`
- `A_lens_proxy` p16/p84: `0.9057` / `0.9602`

## Scope Notes

- This forecast propagates draw-level cosmology through standard CAMB transfer functions.
- It is a lensing-focused CMB prediction target, not a full MG TT/TE/EE perturbation-sector refit.

## Artifacts

- `tables/summary.json`
- `tables/clpp_draws.npz`
- `figures/clpp_bandpowers_vs_models.png`
- `figures/clpp_ratio_to_ref.png`
- `figures/clpp_chi2_distribution.png`
