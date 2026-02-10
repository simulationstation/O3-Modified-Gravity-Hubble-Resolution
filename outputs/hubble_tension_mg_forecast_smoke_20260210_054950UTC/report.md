# Hubble-tension forecast under MG-posterior truth

- Run dir: `/home/primary/O3-Modified-Gravity-Hubble-Resolution/outputs/finalization/highpower_multistart_v2/M0_start101`
- Draws used: `1024` / `1600`
- Anchor replicates per z: `500`
- H0 (MG truth posterior, p16/p50/p84): `67.701`, `70.392`, `73.352` km/s/Mpc
- Local reference: `73.000 ± 1.000` km/s/Mpc
- Injected local bias: `0.0000` km/s/Mpc
- Planck reference: `67.400 ± 0.500` km/s/Mpc
- Injected high-z fractional bias: `0.0000`
- Tension-relief fraction vs local-planck baseline: `0.534`
- Anchor-based relief (GR-interpreted high-z): `0.221`
- Anchor local-vs-high-z gap sigma (GR): `1.235`

## Anchor highlights (GR interpretation)

- z=0.200: inferred H0_GR p50=68.716, local-highz gap sigma=1.294, apparent bias p50=-1.694
- z=0.350: inferred H0_GR p50=68.103, local-highz gap sigma=1.427, apparent bias p50=-2.375
- z=0.500: inferred H0_GR p50=67.815, local-highz gap sigma=1.346, apparent bias p50=-2.412
- z=0.620: inferred H0_GR p50=69.044, local-highz gap sigma=0.920, apparent bias p50=-1.664

## Artifacts

- `tables/summary.json`
- `tables/expansion_profile_quantiles.json`
- `figures/h_ratio_vs_planck.png`
- `figures/h0_apparent_gr_bias_vs_z.png`
- `figures/anchor_h0_inference.png`
