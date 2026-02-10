# Forward Test Plan: O3 MG Hypothesis and Hubble-Tension Implications

## Scope
This plan turns the current interpretation into a falsifiable forward program.

Working hypothesis:
A single modified-gravity sector inferred from the O3 GW propagation anomaly can explain the observed Hubble-tension inference shifts, and should produce consistent, correlated signatures in growth/lensing and GW-vs-EM distance observables.

Date context: baseline artifacts in this repo as of February 10, 2026.

## Core Principle
Do not treat each channel independently. The same MG parameterization must jointly explain:
1. GW propagation behavior.
2. Late-time distance inference shifts relevant to H0 tension.
3. Structure-growth and lensing behavior.
4. External consistency bounds (CMB primary, BBN, early-time coverage bounds).

If separate channels require separate untied freedoms, the universal-MG interpretation is not supported.

## Phase 0: Freeze Baseline and Provenance
Goal: lock reference outputs before adding new tests.

Actions:
1. Freeze the current reference run set and paper numbers in a manifest:
   - `outputs/finalization/highpower_multistart_v2/M0_start101`
   - `outputs/planck_global_mg_refit_realmin60_20260210_live`
   - `outputs/recalibration_planckref_20260210`
   - `outputs/hubble_tension_cmb_forecast_camb64_20260210_analyticrefresh`
   - `outputs/hubble_tension_mg_lensing_refit_camb32_20260210_live`
2. Record fixed benchmark metrics (anchor H0, relief posterior, CLpp suppression, chi2 stats).
3. Add a `forward_tests_manifest.json` with run IDs, hashes, and expected headline values.

Pass gate:
Re-run of baseline extraction reproduces frozen metrics within numeric tolerance.

## Phase 1: Build a Single Closure Layer (GW -> scalar sector)
Goal: remove "loose coupling" between friction and lensing/growth channels.

Actions:
1. Define a minimal tied MG parameterization with explicit redshift behavior for GW friction and scalar response.
2. Implement two model tiers:
   - Tier A: minimal phenomenological tied model.
   - Tier B: restricted EFT-like tied model with priors enforcing stability/physicality.
3. Generate posterior draws from GW-informed parameters and propagate into all downstream observables.

Pass gate:
A single parameter vector can be evaluated across GW, H0-inference, lensing, and growth pipelines without channel-specific retuning.

## Phase 2: Growth/S8 Forward Test
Question: does GW-inferred friction predict the observed low S8 direction and scale?

Actions:
1. From Phase-1 draws, compute predicted growth observables (`fσ8(z)`, `S8`).
2. Compare against available weak-lensing and growth summaries using posterior predictive checks.
3. Quantify whether the required suppression is naturally produced or requires tensioning priors.

Primary metrics:
1. Posterior predictive p-value for measured S8.
2. Shift in joint fit quality relative to GR baseline.
3. Amount of prior volume compression needed to match growth data.

Fail conditions:
1. Predicted S8 remains Planck-like unless forcing extreme/unphysical parameters.
2. Required growth shift conflicts with GW-supported friction posterior.

## Phase 3: mu-Sigma Consistency Test
Question: can one tied MG model fit both matter growth and lensing response?

Actions:
1. Replace purely phenomenological lensing refit freedom with tied parameters from Phase 1.
2. Jointly fit Planck lensing bandpowers and growth-sensitive probes with shared parameters.
3. Compare against:
   - GR baseline.
   - Untied phenomenological MG refit (current flexible benchmark).

Primary metrics:
1. `Δχ2`, information criteria, and Bayes factors for tied vs untied models.
2. Fraction of draws where tied model beats GR reference.
3. Residual structure across multipoles and redshift bins.

Fail conditions:
1. Tied model cannot approach current refit quality without reintroducing untied free functions.
2. Improvements appear only in one probe while degrading others.

## Phase 4: GW-vs-EM Distance Test (not EM duality violation)
Question: does `D_L^GW / D_L^EM` follow the predicted redshift-dependent MG curve?

Actions:
1. Build a hierarchical pipeline for GW standard sirens plus EM distance anchors (with selection effects).
2. Estimate binned/continuous ratio `R(z) = D_L^GW / D_L^EM`.
3. Compare measured `R(z)` against the curve implied by the same Phase-1 MG posterior.

Primary metrics:
1. Consistency of inferred `R(z)` shape with predicted MG curve.
2. Evidence ratio for redshift-dependent MG ratio vs constant ratio vs GR.
3. Robustness under host-association and selection-uncertainty variants.

Fail conditions:
1. No coherent redshift trend in `R(z)`.
2. Preferred trend disagrees with GW-friction posterior shape.

## Phase 5: External Consistency Constraints
Question: does the model survive known non-H0 constraints?

Actions:
1. Apply non-local external constraints tied to effective Planck-mass evolution in the cosmological channel.
2. Check compatibility with early-time bounds (BBN/recombination-informed priors).
3. Run a primary-CMB consistency check (TT/TE/EE-aware, not lensing-only).

Primary metrics:
1. Surviving posterior mass after external constraints.
2. Shift in the H0-relief and growth/lensing fit quality after applying constraints.

Fail conditions:
1. External bounds eliminate most or all parameter mass that drives the claimed effects.
2. Surviving region no longer explains the forward signatures.

## Phase 6: Global Model Selection
Compare four nested hypotheses:
1. `M0`: GR/ΛCDM baseline.
2. `M1`: GW-only propagation anomaly (no scalar-sector tie).
3. `M2`: universal tied MG model (target hypothesis).
4. `M3`: untied phenomenological multi-channel model.

Decision rule:
Support for the hypothesis requires `M2` to outperform `M0` and `M1`, and approach `M3` without large extra flexibility.

## Execution Order (Recommended)
1. Phase 0 (baseline freeze).
2. Phase 1 (closure layer).
3. Phase 2 and Phase 3 in parallel.
4. Phase 4 once updated siren sample is stabilized.
5. Phase 5 external constraints.
6. Phase 6 final model comparison and paper-grade summary.

## Design Now (Immediate)
The following can be designed and scaffolded immediately without running heavy inference:
1. Freeze/check baseline metrics and drift:
   - `.venv/bin/python scripts/run_forward_tests_phase0_baseline_freeze.py --freeze-manifest`
   - `.venv/bin/python scripts/run_forward_tests_phase0_baseline_freeze.py --check-manifest`
2. Create design stubs for each forward-test phase:
   - `.venv/bin/python scripts/run_forward_tests_phase_stub.py --phase phase1_closure`
   - `.venv/bin/python scripts/run_forward_tests_phase_stub.py --phase phase2_growth_s8`
   - `.venv/bin/python scripts/run_forward_tests_phase_stub.py --phase phase3_mu_sigma`
   - `.venv/bin/python scripts/run_forward_tests_phase_stub.py --phase phase4_distance_ratio`
   - `.venv/bin/python scripts/run_forward_tests_phase_stub.py --phase phase5_external_constraints`
   - `.venv/bin/python scripts/run_forward_tests_phase_stub.py --phase phase6_model_selection`
3. Define pass/fail thresholds in each phase `status.json` and `README.md` before execution.
4. Run the concrete Phase-1 shared-draw closure audit:
   - `.venv/bin/python scripts/run_forward_tests_phase1_closure.py`
5. Run the concrete Phase-2 S8 forward test:
   - `.venv/bin/python scripts/run_forward_tests_phase2_growth_s8.py`
6. Run the concrete Phase-3 mu-Sigma proxy consistency audit:
   - `.venv/bin/python scripts/run_forward_tests_phase3_mu_sigma.py`
7. Run the concrete Phase-4 GW/EM distance-ratio forecast:
   - `.venv/bin/python scripts/run_forward_tests_phase4_distance_ratio.py`
8. Run the concrete Phase-5 external-constraint proxy stress test:
   - `.venv/bin/python scripts/run_forward_tests_phase5_external_constraints.py`
9. Run the concrete Phase-6 model-selection proxy aggregator:
   - `.venv/bin/python scripts/run_forward_tests_phase6_model_selection.py`
10. Optional assumption-study scenarios (explicitly non-baseline):
   - Assumed high-z coverage sensitivity:
     `.venv/bin/python scripts/run_forward_tests_phase5_external_constraints.py --coverage-mode assumed_highz --assumed-zmax 1100 --out outputs/forward_tests/phase5_external_constraints_assumed_highz`
   - Corresponding model-selection summary:
     `.venv/bin/python scripts/run_forward_tests_phase6_model_selection.py --phase5-summary outputs/forward_tests/phase5_external_constraints_assumed_highz/tables/summary.json --out outputs/forward_tests/phase6_model_selection_assumed_highz`
11. Reproduce the full phase5/phase6 scenario matrix in one command:
   - `scripts/run_forward_tests_phase5_phase6_matrix.sh`
12. Launch a capped-resource high-z bridge pilot (background job, default 64 cores):
   - `scripts/launch_forward_phase5_highz_bridge_pilot.sh`
13. Run the calibration-robust forward pipeline (separate outputs, single-threaded):
   - `scripts/run_forward_tests_calibration_robust_pipeline.sh`
14. Run the signal-amplitude dial sweep (lightweight gate-sensitivity emulator):
   - `.venv/bin/python scripts/run_forward_tests_signal_amplitude_dial.py --out outputs/forward_tests/signal_amplitude_dial`

## Deliverables
1. `forward_tests_manifest.json` (frozen baseline references and tolerances).
2. `outputs/forward_tests/phase1_closure/` (tied model implementation checks).
3. `outputs/forward_tests/phase2_growth_s8/` (predictive growth tests).
4. `outputs/forward_tests/phase3_mu_sigma/` (joint lensing-growth consistency).
5. `outputs/forward_tests/phase4_distance_ratio/` (`D_L^GW / D_L^EM` trend analysis).
6. `outputs/forward_tests/phase5_external_constraints/` (constraint stress test).
7. `outputs/forward_tests/phase6_model_selection/tables/summary.json` and a short markdown report.
8. Scenario matrix artifact:
   - `outputs/forward_tests/phase5_phase6_matrix_*/summary.json`
   - Latest pointer: `outputs/forward_tests/phase5_phase6_matrix_latest.json`
9. Calibration-robust artifact:
   - `outputs/forward_tests/calibration_robust_*/summary.json`
   - Latest pointer: `outputs/forward_tests/calibration_robust_latest.json`
10. Signal-amplitude dial artifact:
   - `outputs/forward_tests/signal_amplitude_dial/tables/summary.json`
   - `outputs/forward_tests/signal_amplitude_dial/tables/alpha_sweep.csv`

## Minimum Success Criteria
Claim is strengthened only if all are true:
1. One tied MG parameterization remains viable after external constraints.
2. It explains a meaningful fraction of H0 inference shift.
3. It predicts growth/lensing trends in the observed direction with competitive fit quality.
4. GW-vs-EM distance ratio trend is consistent with the same model.

Note:
- Phase-6 now reports both a core support gate and a material-relief gate (`target_hypothesis_M2_supported_with_material_relief`) so interpretation does not overstate support when anchor relief remains small.
- Phases 2-6 now also expose an optional `calibration_robust_gate` that marginalizes over explicit catalog/calibration nuisance terms; Phase-6 reports `bias_stability_score_m2` when run with `--calibration-robust`.
- The signal-amplitude dial tool is a lightweight emulator for gate sensitivity and should be used to target follow-up full re-inference runs.

## Stop Criteria
Stop and downgrade the universal-MG claim if any occur:
1. Cross-channel consistency fails without untied freedoms.
2. External constraints remove the effective parameter region.
3. Distance-ratio trend contradicts friction-implied evolution.
