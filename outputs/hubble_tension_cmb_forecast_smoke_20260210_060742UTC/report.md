# CMB Forecast Under MG-Truth Assumption

- Generated UTC: `2026-02-10T06:07:44Z`
- Run dir: `/home/primary/O3-Modified-Gravity-Hubble-Resolution/outputs/finalization/highpower_multistart_v2/M0_start101`
- Draws evaluated: `16/16`
- Planck lensing dataset: `consext8`

## Headline Predicted Lensing Shifts

- Near L~100 (actual `106`): median `ΔC/C = -5.71%` (68%: `-11.03%` to `-2.08%`).
- Near L~300 (actual `286`): median `ΔC/C = -5.71%` (68%: `-11.03%` to `-2.08%`).

## Fit to Planck Lensing Bandpowers

- MG draw chi2 (median): `12.630`
- Planck-reference model chi2: `9.312`
- P(draw beats reference): `12.5%`

## Derived Amplitude Proxies

- `A_lens_proxy` median: `0.9429`
- `A_lens_proxy` p16/p84: `0.8897` / `0.9792`

## Scope Notes

- This forecast uses the fast template-scaling lensing proxy (calibrated in the codebase) for draw-level CMB lensing shifts.
- It is a lensing-focused CMB prediction target, not a full MG TT/TE/EE perturbation-sector refit.

## Artifacts

- `tables/summary.json`
- `tables/clpp_draws.npz`
- `figures/clpp_bandpowers_vs_models.png`
- `figures/clpp_ratio_to_ref.png`
- `figures/clpp_chi2_distribution.png`
