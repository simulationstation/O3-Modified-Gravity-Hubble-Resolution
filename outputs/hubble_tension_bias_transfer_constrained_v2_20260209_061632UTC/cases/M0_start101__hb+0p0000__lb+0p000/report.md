# Hubble-tension forecast under MG-posterior truth

- Run dir: `/home/primary/PROJECT/outputs/finalization/highpower_multistart_v2/M0_start101`
- Draws used: `1600` / `1600`
- Anchor replicates per z: `100000`
- H0 (MG truth posterior, p16/p50/p84): `67.704`, `70.388`, `73.393` km/s/Mpc
- Local reference: `73.000 ± 1.000` km/s/Mpc
- Injected local bias: `0.0000` km/s/Mpc
- Planck reference: `67.400 ± 0.500` km/s/Mpc
- Injected high-z fractional bias: `0.0000`
- Tension-relief fraction vs local-planck baseline: `0.534`
- Anchor-based relief (GR-interpreted high-z): `0.208`
- Anchor local-vs-high-z gap sigma (GR): `1.225`

## Anchor highlights (GR interpretation)

- z=0.200: inferred H0_GR p50=68.635, local-highz gap sigma=1.333, apparent bias p50=-1.885
- z=0.350: inferred H0_GR p50=67.917, local-highz gap sigma=1.432, apparent bias p50=-2.623
- z=0.500: inferred H0_GR p50=68.023, local-highz gap sigma=1.265, apparent bias p50=-2.531
- z=0.620: inferred H0_GR p50=68.880, local-highz gap sigma=0.946, apparent bias p50=-1.683

## Artifacts

- `tables/summary.json`
- `tables/expansion_profile_quantiles.json`
- `figures/h_ratio_vs_planck.png`
- `figures/h0_apparent_gr_bias_vs_z.png`
- `figures/anchor_h0_inference.png`
