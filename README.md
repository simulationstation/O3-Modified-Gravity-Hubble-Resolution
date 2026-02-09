# O3 Modified Gravity Hubble Resolution

Standalone pipeline for the O3-conditioned Hubble-relief analysis, updated to the latest O3 dark-siren support point:
- `ΔLPD_total = +3.6699` (`ΔLPD_data = +2.6700`, `ΔLPD_sel = +0.9999`)
- source summary: `outputs/dark_siren_o3_injection_logit_20260209_055801UTC/summary_M0_start101.json` in the parent project

## Included here

- Core Hubble scripts:
  - `scripts/run_hubble_tension_mg_forecast.py`
  - `scripts/run_hubble_tension_mg_forecast_robustness_grid.py`
  - `scripts/run_hubble_tension_bias_transfer_sweep.py`
  - `scripts/run_hubble_tension_final_relief_posterior.py`
  - `scripts/run_joint_transfer_bias_fit.py`
- Detached launchers:
  - `scripts/launch_hubble_tension_mg_forecast_single_nohup.sh`
  - `scripts/launch_hubble_tension_mg_robustness_grid_single_nohup.sh`
  - `scripts/launch_hubble_tension_bias_transfer_sweep_single_nohup.sh`
  - `scripts/launch_joint_transfer_bias_fit_single_nohup.sh`
- Package code used by these scripts under `src/entropy_horizon_recon/`.
- Updated paper assets:
  - `update_paper/hubble_tension_hypothesis.tex`
  - `update_paper/hubble_tension_hypothesis.pdf`
  - `update_paper/hubble_tension_assets/`
- Refreshed outputs from this run:
  - `outputs/hubble_tension_bias_transfer_constrained_v2_20260209_061632UTC/`
  - `outputs/hubble_tension_bias_transfer_constrained_v2_repeat_20260209_061832UTC/`
  - `outputs/hubble_tension_final_relief_posterior_20260209_061950UTC/`
  - `outputs/joint_transfer_bias_fit_full_20260209_061958UTC/`

## Refreshed headline numbers

From `outputs/hubble_tension_final_relief_posterior_20260209_061950UTC/final_relief_posterior_summary.json`:
- Anchor-based GR-interpreted relief posterior (MC-calibrated):
  - mean `0.2459`
  - p16/p50/p84 `0.2048 / 0.2396 / 0.2767`
- High-z bias thresholds (linearized):
  - `10%` relief at `~ -1.20%`
  - `40%` relief at `~ +1.26%`

From `outputs/joint_transfer_bias_fit_full_20260209_061958UTC/tables/summary.json`:
- `o3_delta_lpd_metadata = 3.669945265`
- `log BF(transfer vs no-transfer) = -0.5331`
- Dominant transfer term: `delta_h0_ladder_local`

## Run notes

All long runs are designed for detached execution (`setsid` + `taskset`) and write:
- `pid.txt`
- `run.log`
- incremental status JSON

To rerun the full joint transfer fit on a 48-core host:

```bash
CPUSET=0-47 WORKERS=48 O3_DELTA_LPD=3.669945265 \
  scripts/launch_joint_transfer_bias_fit_single_nohup.sh full
```
