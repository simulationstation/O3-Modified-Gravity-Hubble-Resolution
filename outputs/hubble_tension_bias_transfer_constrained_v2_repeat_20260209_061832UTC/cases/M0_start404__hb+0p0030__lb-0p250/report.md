# Hubble-tension forecast under MG-posterior truth

- Run dir: `/home/primary/PROJECT/outputs/finalization/highpower_multistart_v2/M0_start404`
- Draws used: `1600` / `1600`
- Anchor replicates per z: `100000`
- H0 (MG truth posterior, p16/p50/p84): `67.695`, `70.536`, `73.602` km/s/Mpc
- Local reference: `73.000 ± 1.000` km/s/Mpc
- Injected local bias: `-0.2500` km/s/Mpc
- Planck reference: `67.400 ± 0.500` km/s/Mpc
- Injected high-z fractional bias: `0.0030`
- Tension-relief fraction vs local-planck baseline: `0.560`
- Anchor-based relief (GR-interpreted high-z): `0.308`
- Anchor local-vs-high-z gap sigma (GR): `0.988`

## Anchor highlights (GR interpretation)

- z=0.200: inferred H0_GR p50=69.138, local-highz gap sigma=1.067, apparent bias p50=-1.532
- z=0.350: inferred H0_GR p50=68.504, local-highz gap sigma=1.177, apparent bias p50=-2.206
- z=0.500: inferred H0_GR p50=68.710, local-highz gap sigma=1.017, apparent bias p50=-2.070
- z=0.620: inferred H0_GR p50=69.481, local-highz gap sigma=0.748, apparent bias p50=-1.295

## Artifacts

- `tables/summary.json`
- `tables/expansion_profile_quantiles.json`
- `figures/h_ratio_vs_planck.png`
- `figures/h0_apparent_gr_bias_vs_z.png`
- `figures/anchor_h0_inference.png`
