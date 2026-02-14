# Predictions / Falsifiers (From This Hardening Suite)

This note records checks that would falsify the modified-propagation interpretation.

Predictions:
1. As the siren sample grows (O4/O5), leverage concentration should decrease: the joint preference should not remain dominated by a single event.
2. High-distance bins should continue to contribute disproportionately if the effect is cumulative in propagation distance.
3. Detector/network splits should not induce sign flips; if the preference is physical propagation, it should persist across HL and HLV subsets.

Falsifiers:
1. If improved catalogue modelling (completeness/photo-z) within realistic external priors erases the preference (Delta LPD -> 0), the signal is likely systematic.
2. If the permutation null does not degrade the preference (i.e., the observed Delta LPD is typical under scrambled event-catalog associations), the interpretation as coherent catalogue-redshift structure would weaken.
3. If non-circular splits show strong incoherence (alpha_hat drift far beyond its posterior uncertainty), it would argue for model misspecification rather than a single universal propagation history.
