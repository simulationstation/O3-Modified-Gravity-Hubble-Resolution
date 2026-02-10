from __future__ import annotations

import numpy as np

from entropy_horizon_recon.likelihoods_planck_lensing_mg import PlanckLensingBandpowerMGLogLike
from entropy_horizon_recon.mg_lensing_response import MGLensingResponseParams


def test_mg_like_prefers_correct_response_on_synthetic_data() -> None:
    ell = np.array([40.0, 90.0, 160.0, 280.0, 420.0], dtype=float)
    clpp_gr = np.array([1.00, 0.92, 0.75, 0.53, 0.41], dtype=float)
    cov = np.eye(clpp_gr.size) * 0.01

    p_true = MGLensingResponseParams(log10_mstar2_ratio_0=-0.1, ell_tilt=0.0)
    response_true = np.power(10.0, 0.1)
    clpp_obs = clpp_gr * response_true

    like = PlanckLensingBandpowerMGLogLike.from_data(ell_eff=ell, clpp=clpp_obs, cov=cov)
    chi2_gr = like.chi2(clpp_gr)
    model_true = like.apply_mg_response(clpp_gr, p_true)
    chi2_true = like.chi2(model_true)

    assert chi2_true < chi2_gr
