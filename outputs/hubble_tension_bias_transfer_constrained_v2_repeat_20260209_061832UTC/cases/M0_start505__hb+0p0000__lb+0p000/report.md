# Hubble-tension forecast under MG-posterior truth

- Run dir: `/home/primary/PROJECT/outputs/finalization/highpower_multistart_v2/M0_start505`
- Draws used: `1600` / `1600`
- Anchor replicates per z: `100000`
- H0 (MG truth posterior, p16/p50/p84): `67.718`, `70.729`, `73.434` km/s/Mpc
- Local reference: `73.000 ± 1.000` km/s/Mpc
- Injected local bias: `0.0000` km/s/Mpc
- Planck reference: `67.400 ± 0.500` km/s/Mpc
- Injected high-z fractional bias: `0.0000`
- Tension-relief fraction vs local-planck baseline: `0.594`
- Anchor-based relief (GR-interpreted high-z): `0.239`
- Anchor local-vs-high-z gap sigma (GR): `1.200`

## Anchor highlights (GR interpretation)

- z=0.200: inferred H0_GR p50=68.859, local-highz gap sigma=1.299, apparent bias p50=-1.816
- z=0.350: inferred H0_GR p50=68.122, local-highz gap sigma=1.413, apparent bias p50=-2.543
- z=0.500: inferred H0_GR p50=68.255, local-highz gap sigma=1.247, apparent bias p50=-2.449
- z=0.620: inferred H0_GR p50=69.067, local-highz gap sigma=0.917, apparent bias p50=-1.614

## Artifacts

- `tables/summary.json`
- `tables/expansion_profile_quantiles.json`
- `figures/h_ratio_vs_planck.png`
- `figures/h0_apparent_gr_bias_vs_z.png`
- `figures/anchor_h0_inference.png`
