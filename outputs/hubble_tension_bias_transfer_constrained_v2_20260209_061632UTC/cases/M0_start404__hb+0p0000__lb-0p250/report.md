# Hubble-tension forecast under MG-posterior truth

- Run dir: `/home/primary/PROJECT/outputs/finalization/highpower_multistart_v2/M0_start404`
- Draws used: `1600` / `1600`
- Anchor replicates per z: `100000`
- H0 (MG truth posterior, p16/p50/p84): `67.695`, `70.536`, `73.602` km/s/Mpc
- Local reference: `73.000 ± 1.000` km/s/Mpc
- Injected local bias: `-0.2500` km/s/Mpc
- Planck reference: `67.400 ± 0.500` km/s/Mpc
- Injected high-z fractional bias: `0.0000`
- Tension-relief fraction vs local-planck baseline: `0.560`
- Anchor-based relief (GR-interpreted high-z): `0.271`
- Anchor local-vs-high-z gap sigma (GR): `1.046`

## Anchor highlights (GR interpretation)

- z=0.200: inferred H0_GR p50=68.938, local-highz gap sigma=1.136, apparent bias p50=-1.730
- z=0.350: inferred H0_GR p50=68.319, local-highz gap sigma=1.229, apparent bias p50=-2.403
- z=0.500: inferred H0_GR p50=68.505, local-highz gap sigma=1.087, apparent bias p50=-2.272
- z=0.620: inferred H0_GR p50=69.268, local-highz gap sigma=0.794, apparent bias p50=-1.492

## Artifacts

- `tables/summary.json`
- `tables/expansion_profile_quantiles.json`
- `figures/h_ratio_vs_planck.png`
- `figures/h0_apparent_gr_bias_vs_z.png`
- `figures/anchor_h0_inference.png`
