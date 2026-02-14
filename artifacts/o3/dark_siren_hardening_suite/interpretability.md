# Interpretability Memo: Lensing Response Parameters

- From the constrained Planck-lensing response refit, the median effective Planck-mass ratio is `M_*^2(0)/M_*^2(early) ≈ 0.901`.
- The corresponding phenomenological multipole-tilt parameter has median `ell_tilt ≈ -0.145` (pivot `L=200`).

Interpreted in EFT-of-DE language, this corresponds to an integrated Planck-mass running of order `Δ ln M_*^2 ≈ -0.104` between early times and today. In Horndeski/EFT notation this is sourced by `α_M(a) = d ln M_*^2 / d ln a`; the present analysis does not reconstruct `α_M(a)` directly, but the inferred integrated change is at the few×10% level.

Caveats:
- The refit is a constrained phenomenological response (amplitude + scale-tilt) applied to the lensing-reconstruction likelihood only; it is not a full Boltzmann evolution of modified perturbation equations.
- A physically complete mapping to `(α_M, α_B, α_K, α_T)` requires specifying a covariant model and refitting TT/TE/EE plus lensing jointly.

Qualitative consistency check:
- The fitted response removes the baseline `C_L^{φφ}` suppression without extreme parameter excursions, and remains consistent with the primary-spectrum closure preference for `A_L > 1` noted in the Letter.
