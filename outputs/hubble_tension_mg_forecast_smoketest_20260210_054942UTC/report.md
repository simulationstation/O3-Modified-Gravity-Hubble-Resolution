# Hubble-tension forecast under MG-posterior truth

- Run dir: `/home/primary/O3-Modified-Gravity-Hubble-Resolution/outputs/finalization/highpower_multistart_v2/M0_start101`
- Draws used: `128` / `1600`
- Anchor replicates per z: `100`
- H0 (MG truth posterior, p16/p50/p84): `68.054`, `70.556`, `73.448` km/s/Mpc
- Local reference: `73.000 ± 1.000` km/s/Mpc
- Injected local bias: `0.0000` km/s/Mpc
- Planck reference: `67.400 ± 0.500` km/s/Mpc
- Injected high-z fractional bias: `0.0000`
- Tension-relief fraction vs local-planck baseline: `0.564`
- Anchor-based relief (GR-interpreted high-z): `0.163`
- Anchor local-vs-high-z gap sigma (GR): `1.425`

## Anchor highlights (GR interpretation)

- z=0.200: inferred H0_GR p50=67.905, local-highz gap sigma=1.298, apparent bias p50=-2.137
- z=0.350: inferred H0_GR p50=67.635, local-highz gap sigma=1.538, apparent bias p50=-3.213

## Artifacts

- `tables/summary.json`
- `tables/expansion_profile_quantiles.json`
- `figures/h_ratio_vs_planck.png`
- `figures/h0_apparent_gr_bias_vs_z.png`
- `figures/anchor_h0_inference.png`
