# High-power multistart v2 - Master Report

Generated: 2026-01-28T04:50:27.774489+00:00 (UTC)

## Run identity
- Output base: `outputs/finalization/highpower_multistart_v2`
- Git SHA: `2805e482e85ff2c3cf5c18cc9acedf4e856fc060` (dirty=True)
- Seeds: 101, 202, 303, 404, 505
- Command template: `scripts/run_realdata_recon.py --out outputs/finalization/highpower_multistart_v2/M0_start101 --seed 101 --mu-init-seed 101 --mu-sampler ptemcee --rsd-mode dr12+dr16_fsbao --include-lensing --cpu-cores 0 --mu-procs 0 --gp-procs 0 --save-chain outputs/finalization/highpower_multistart_v2/M0_start101/samples/mu_chain.npz --mu-steps 3000 --mu-burn 1000 --mu-draws 1600`

## Timeline
- Monitor start: 2026-01-28T01:30:30+00:00
- Seeds completed: 2026-01-28T04:40:18+00:00
- Wall time (monitor): 3.16 hours
- Per-seed summary.json mtime (UTC):
  - seed 101: 2026-01-28T04:39:51.053169+00:00
  - seed 202: 2026-01-28T04:40:10.537038+00:00
  - seed 303: 2026-01-28T04:39:33.814285+00:00
  - seed 404: 2026-01-28T04:39:44.004216+00:00
  - seed 505: 2026-01-28T04:39:38.192255+00:00

## Likelihood composition
- SN: Pantheon+ cosmology subset (stat+sys cov), z column zHD
  - z-range: 0.02-0.6200000000000001
  - raw SN count: 1322, binned-for-forward: 12
- Cosmic chronometers: BC03_all
  - count: 9
- BAO (distance-only):
  - included: ['desi_2024_bao_all']
  - dropped due to FSBAO: ['sdss_dr12_consensus_bao', 'sdss_dr16_lrg_bao_dmdh']
- FSBAO (correlated distance+f_sigma8):
  - datasets: ['sdss_dr12_consensus_fs', 'sdss_dr16_lrg_fsbao_dmdhfs8'] (n=15)
- RSD compilation: not used (include_rsd=false; FSBAO supplied growth constraints)
- CMB lensing: Planck lensing proxy (Gaussian on sigma8 Omega_m0^0.25), lensing_mode=gaussian_s8

## Sampler + model settings
- Mapping variant: M0
- mu(A) sampler: ptemcee, nt=8, walkers=64
- Steps/burn/draws: 3000 / 1000 / 1600
- mu knots/grid: 8 / 120
- n_logA grid: 140
- Growth mode: ode
- sigma8_0 prior: [0.6, 1.0] (from summary)
- Omega_m0 prior: default [0.2, 0.4] (not overridden in the command; see scripts/run_realdata_recon.py)
- FSBAO covariance: full
- BAO covariance: full
- Lensing exponent: 0.25 (sigma8 Omega_m0^alpha)
- m-weight mode: variance
- H(z) cross-checks: GP + spline were run (not skipped)
- Quick ablation: SN diagonal covariance (sn_diagonal_cov) included (see per-seed report.md)

## LogA-domain stability
- Strict overlap used for seeds 101/202/303/404; robust fallback used for seed 505.
- The logA-domain fallback prevents the earlier crash and is recorded in each summary.json.

## Outputs generated (per seed)
- report.md (human-readable report)
- tables/summary.json, tables/departure_stats.json, tables/proximity.json
- figures/: Hz_forward.png, Hz_gp.png, Hz_spline.png, Az_log.png, logmu_x.png, logmu_logA.png
- samples/: mu_chain.npz, mu_forward_posterior.npz, mu_forward_meta.json, logmu_logA_samples.npz, plus GP/spline samples + meta

## Per-seed results
| seed | H0_p50 | Omega_m0_p50 | r_d_p50 | S8_p50 | m_mean +/- std | P(m>0) | slope_mean +/- std | P(slope>0) | acc_frac | ESS_min | logA_min..max | logA_method | fallback |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 101 | 70.388 | 0.356289 | 144.451 | 0.828144 | -0.0972 +/- 0.1665 | 0.2425 | -0.7143 +/- 0.3540 | 0.0281 | 0.3034 | 19712 | 18.871..18.905 | strict | false |
| 202 | 71.035 | 0.359407 | 143.136 | 0.828919 | -0.1120 +/- 0.1648 | 0.2338 | -0.7081 +/- 0.3410 | 0.0219 | 0.3048 | 14769 | 18.882..18.888 | strict | false |
| 303 | 70.565 | 0.353599 | 144.016 | 0.828705 | -0.1072 +/- 0.1692 | 0.2331 | -0.7194 +/- 0.3528 | 0.0312 | 0.3014 | 13715 | 18.860..18.953 | strict | false |
| 404 | 70.536 | 0.352851 | 143.990 | 0.828020 | -0.0972 +/- 0.1755 | 0.2662 | -0.6695 +/- 0.3594 | 0.0387 | 0.3068 | 12120 | 18.855..18.943 | strict | false |
| 505 | 70.729 | 0.353891 | 143.878 | 0.828613 | -0.0978 +/- 0.1678 | 0.2425 | -0.6940 +/- 0.3526 | 0.0306 | 0.3013 | 16414 | 18.698..19.099 | robust | true |

## Posterior quantiles (16/50/84)
| seed | H0 (16/50/84) | Omega_m0 (16/50/84) | r_d (16/50/84) | sigma8_0 (16/50/84) | S8 (16/50/84) | sigma_cc,jit (16/50/84) | sigma_sn,jit (16/50/84) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 67.704/70.388/73.393 | 0.305/0.356/0.387 | 138.339/144.451/150.229 | 0.739/0.769/0.802 | 0.791/0.828/0.861 | 0.866/2.530/5.631 | 0.025/0.033/0.043 |
| 202 | 68.103/71.035/73.814 | 0.301/0.359/0.389 | 138.181/143.136/149.333 | 0.740/0.768/0.802 | 0.792/0.829/0.861 | 0.801/2.724/5.626 | 0.025/0.033/0.043 |
| 303 | 67.716/70.565/73.358 | 0.300/0.354/0.388 | 138.751/144.016/149.794 | 0.744/0.769/0.805 | 0.791/0.829/0.861 | 0.779/2.600/5.906 | 0.025/0.033/0.043 |
| 404 | 67.695/70.536/73.602 | 0.297/0.353/0.384 | 138.155/143.990/149.821 | 0.743/0.773/0.810 | 0.790/0.828/0.859 | 0.906/2.802/6.039 | 0.025/0.034/0.044 |
| 505 | 67.718/70.729/73.434 | 0.301/0.354/0.385 | 138.411/143.878/149.816 | 0.742/0.770/0.803 | 0.790/0.829/0.859 | 0.814/2.640/5.963 | 0.025/0.033/0.045 |

## Aggregate (across seeds)
- H0_p50: mean=70.650476, sd=0.246503
- Om0_p50: mean=0.355207, sd=0.002677
- rd_p50: mean=143.894240, sd=0.476891
- S8_p50: mean=0.828480, sd=0.000383
- sigma8_p50: mean=0.769553, sd=0.001876
- m_mean: mean=-0.102307, sd=0.006897
- m_std: mean=0.168776, sd=0.004113
- m_p_gt0: mean=0.243625, sd=0.013437
- slope_mean: mean=-0.701049, sd=0.020016
- slope_std: mean=0.351962, sd=0.006698
- slope_p_gt0: mean=0.030125, sd=0.006082
- acc_frac: mean=0.303525, sd=0.002329
- ess_min: mean=15346.135668, sd=2898.420650

## Diagnostics
- Acceptance fraction mean (per seed): ~0.301-0.307 (mean 0.3035)
- ESS_min range: 12120-19712 (mean 15346)
- Each seed logged the ptemcee warning: chain length < 50x integrated autocorrelation time (see run.log).
- No runtime errors detected in any run.log; all seeds produced tables and reports.

## Notes / limitations
- The mu(A) inversion uses the late-time matter-dominance approximation for (rho+p); interpret mu(A) accordingly.
- Seed 505 used the robust logA overlap fallback (see logA_domain in its summary).
- This report does not combine posteriors; it summarizes each seed's posterior summaries and departure statistics.

## Reproducibility
- Per-seed command is recorded in each tables/summary.json under command.
- Monitor log: status.jsonl records process counts and completion timestamps.