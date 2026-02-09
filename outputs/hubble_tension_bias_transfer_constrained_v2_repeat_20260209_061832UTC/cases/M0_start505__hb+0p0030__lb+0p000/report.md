# Hubble-tension forecast under MG-posterior truth

- Run dir: `/home/primary/PROJECT/outputs/finalization/highpower_multistart_v2/M0_start505`
- Draws used: `1600` / `1600`
- Anchor replicates per z: `100000`
- H0 (MG truth posterior, p16/p50/p84): `67.718`, `70.729`, `73.434` km/s/Mpc
- Local reference: `73.000 ± 1.000` km/s/Mpc
- Injected local bias: `0.0000` km/s/Mpc
- Planck reference: `67.400 ± 0.500` km/s/Mpc
- Injected high-z fractional bias: `0.0030`
- Tension-relief fraction vs local-planck baseline: `0.594`
- Anchor-based relief (GR-interpreted high-z): `0.277`
- Anchor local-vs-high-z gap sigma (GR): `1.135`

## Anchor highlights (GR interpretation)

- z=0.200: inferred H0_GR p50=69.050, local-highz gap sigma=1.237, apparent bias p50=-1.617
- z=0.350: inferred H0_GR p50=68.332, local-highz gap sigma=1.344, apparent bias p50=-2.342
- z=0.500: inferred H0_GR p50=68.472, local-highz gap sigma=1.180, apparent bias p50=-2.251
- z=0.620: inferred H0_GR p50=69.280, local-highz gap sigma=0.856, apparent bias p50=-1.408

## Artifacts

- `tables/summary.json`
- `tables/expansion_profile_quantiles.json`
- `figures/h_ratio_vs_planck.png`
- `figures/h0_apparent_gr_bias_vs_z.png`
- `figures/anchor_h0_inference.png`
