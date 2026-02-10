from __future__ import annotations

import numpy as np

from entropy_horizon_recon.mg_lensing_response import (
    MGLensingResponseParams,
    mg_lensing_response,
)


def test_mg_response_recovers_gr_when_parameters_are_unity() -> None:
    ell = np.array([20.0, 100.0, 300.0, 800.0], dtype=float)
    p = MGLensingResponseParams(
        log10_mstar2_ratio_0=0.0,
        mu0=1.0,
        eta0=1.0,
        ell_tilt=0.0,
        ell_pivot=200.0,
    )
    r = mg_lensing_response(ell, p)
    assert np.allclose(r, 1.0)


def test_mg_response_increases_for_smaller_planck_mass_ratio() -> None:
    ell = np.array([100.0, 300.0], dtype=float)
    p = MGLensingResponseParams(log10_mstar2_ratio_0=-0.2, mu0=1.0, eta0=1.0, ell_tilt=0.0)
    r = mg_lensing_response(ell, p)
    assert np.all(r > 1.0)


def test_mg_response_ell_tilt_changes_scale_dependence() -> None:
    ell = np.array([50.0, 200.0, 800.0], dtype=float)
    p = MGLensingResponseParams(ell_tilt=0.2, ell_pivot=200.0)
    r = mg_lensing_response(ell, p)
    assert r[0] < r[1] < r[2]
