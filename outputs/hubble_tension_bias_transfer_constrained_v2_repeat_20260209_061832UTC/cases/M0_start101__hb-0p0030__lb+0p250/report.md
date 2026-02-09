# Hubble-tension forecast under MG-posterior truth

- Run dir: `/home/primary/PROJECT/outputs/finalization/highpower_multistart_v2/M0_start101`
- Draws used: `1600` / `1600`
- Anchor replicates per z: `100000`
- H0 (MG truth posterior, p16/p50/p84): `67.704`, `70.388`, `73.393` km/s/Mpc
- Local reference: `73.000 ± 1.000` km/s/Mpc
- Injected local bias: `0.2500` km/s/Mpc
- Planck reference: `67.400 ± 0.500` km/s/Mpc
- Injected high-z fractional bias: `-0.0030`
- Tension-relief fraction vs local-planck baseline: `0.534`
- Anchor-based relief (GR-interpreted high-z): `0.174`
- Anchor local-vs-high-z gap sigma (GR): `1.352`

## Anchor highlights (GR interpretation)

- z=0.200: inferred H0_GR p50=68.468, local-highz gap sigma=1.465, apparent bias p50=-2.088
- z=0.350: inferred H0_GR p50=67.702, local-highz gap sigma=1.574, apparent bias p50=-2.822
- z=0.500: inferred H0_GR p50=67.856, local-highz gap sigma=1.393, apparent bias p50=-2.718
- z=0.620: inferred H0_GR p50=68.705, local-highz gap sigma=1.055, apparent bias p50=-1.865

## Artifacts

- `tables/summary.json`
- `tables/expansion_profile_quantiles.json`
- `figures/h_ratio_vs_planck.png`
- `figures/h0_apparent_gr_bias_vs_z.png`
- `figures/anchor_h0_inference.png`
