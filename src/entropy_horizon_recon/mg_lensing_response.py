from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MGLensingResponseParams:
    """Phenomenological MG response for CMB lensing bandpowers.

    Parameters are defined so that GR is recovered at:
      - log10_mstar2_ratio_0 = 0
      - mu0 = 1
      - eta0 = 1
      - ell_tilt = 0

    `mstar2_ratio_0` is M_*^2(z=0) / M_*^2(high-z).
    Values below 1 imply stronger effective gravity at late times.
    """

    log10_mstar2_ratio_0: float = 0.0
    mu0: float = 1.0
    eta0: float = 1.0
    ell_tilt: float = 0.0
    ell_pivot: float = 200.0
    response_power: float = 1.0
    clip_min: float = 0.05
    clip_max: float = 20.0


def mstar2_ratio_0(params: MGLensingResponseParams) -> float:
    return float(10.0 ** float(params.log10_mstar2_ratio_0))


def geff_over_gr_from_mstar(params: MGLensingResponseParams) -> float:
    ratio = mstar2_ratio_0(params)
    if ratio <= 0.0 or not np.isfinite(ratio):
        raise ValueError("Non-positive or non-finite M_*^2 ratio.")
    return float(1.0 / ratio)


def sigma_lensing_factor(params: MGLensingResponseParams) -> float:
    """Effective lensing-potential amplitude factor Sigma/Sigma_GR."""
    mu0 = float(params.mu0)
    eta0 = float(params.eta0)
    if mu0 <= 0.0 or not np.isfinite(mu0):
        raise ValueError("mu0 must be positive and finite.")
    if eta0 <= 0.0 or not np.isfinite(eta0):
        raise ValueError("eta0 must be positive and finite.")
    g_mstar = geff_over_gr_from_mstar(params)
    sigma = g_mstar * mu0 * 0.5 * (1.0 + eta0)
    if sigma <= 0.0 or not np.isfinite(sigma):
        raise ValueError("Computed Sigma lensing factor is invalid.")
    return float(sigma)


def mg_lensing_response(ell: np.ndarray, params: MGLensingResponseParams) -> np.ndarray:
    """Multiplicative response R_L such that C_L^pp(MG) = R_L * C_L^pp(GR)."""
    ell = np.asarray(ell, dtype=float)
    if np.any(ell <= 0.0):
        raise ValueError("All ell values must be > 0.")
    if not np.isfinite(params.ell_pivot) or float(params.ell_pivot) <= 0.0:
        raise ValueError("ell_pivot must be positive and finite.")

    sigma = sigma_lensing_factor(params)
    amp = float(sigma ** float(params.response_power))
    scale = np.power(ell / float(params.ell_pivot), float(params.ell_tilt))
    resp = amp * scale
    return np.clip(resp, float(params.clip_min), float(params.clip_max))
