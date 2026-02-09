# Hubble-tension forecast under MG-posterior truth

- Run dir: `/home/primary/PROJECT/outputs/finalization/highpower_multistart_v2/M0_start202`
- Draws used: `1600` / `1600`
- Anchor replicates per z: `100000`
- H0 (MG truth posterior, p16/p50/p84): `68.103`, `71.035`, `73.814` km/s/Mpc
- Local reference: `73.000 ± 1.000` km/s/Mpc
- Injected local bias: `0.2500` km/s/Mpc
- Planck reference: `67.400 ± 0.500` km/s/Mpc
- Injected high-z fractional bias: `0.0000`
- Tension-relief fraction vs local-planck baseline: `0.649`
- Anchor-based relief (GR-interpreted high-z): `0.275`
- Anchor local-vs-high-z gap sigma (GR): `1.184`

## Anchor highlights (GR interpretation)

- z=0.200: inferred H0_GR p50=69.034, local-highz gap sigma=1.278, apparent bias p50=-1.927
- z=0.350: inferred H0_GR p50=68.256, local-highz gap sigma=1.397, apparent bias p50=-2.705
- z=0.500: inferred H0_GR p50=68.355, local-highz gap sigma=1.227, apparent bias p50=-2.659
- z=0.620: inferred H0_GR p50=69.185, local-highz gap sigma=0.913, apparent bias p50=-1.896

## Artifacts

- `tables/summary.json`
- `tables/expansion_profile_quantiles.json`
- `figures/h_ratio_vs_planck.png`
- `figures/h0_apparent_gr_bias_vs_z.png`
- `figures/anchor_h0_inference.png`
