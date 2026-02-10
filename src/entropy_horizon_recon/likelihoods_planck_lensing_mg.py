from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mg_lensing_response import MGLensingResponseParams, mg_lensing_response


@dataclass(frozen=True)
class PlanckLensingBandpowerMGLogLike:
    """Planck 2018 lensing bandpower likelihood with a phenomenological MG response."""

    ell_eff: np.ndarray
    clpp: np.ndarray
    cov: np.ndarray
    cov_inv: np.ndarray
    meta: dict

    @classmethod
    def from_data(
        cls,
        *,
        ell_eff: np.ndarray,
        clpp: np.ndarray,
        cov: np.ndarray,
        meta: dict | None = None,
    ) -> "PlanckLensingBandpowerMGLogLike":
        ell_eff = np.asarray(ell_eff, dtype=float)
        clpp = np.asarray(clpp, dtype=float)
        cov = np.asarray(cov, dtype=float)
        if cov.shape != (clpp.size, clpp.size):
            raise ValueError("Lensing covariance shape mismatch.")
        if ell_eff.shape != clpp.shape:
            raise ValueError("ell_eff shape mismatch.")
        if not np.allclose(cov, cov.T, atol=1e-10):
            raise ValueError("Lensing covariance not symmetric.")
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Lensing covariance is singular.") from exc
        return cls(
            ell_eff=ell_eff,
            clpp=clpp,
            cov=cov,
            cov_inv=cov_inv,
            meta=dict(meta or {}),
        )

    def apply_mg_response(self, clpp_gr: np.ndarray, params: MGLensingResponseParams) -> np.ndarray:
        clpp_gr = np.asarray(clpp_gr, dtype=float)
        if clpp_gr.shape != self.clpp.shape:
            raise ValueError("GR model shape mismatch for MG response.")
        if np.any(~np.isfinite(clpp_gr)) or np.any(clpp_gr <= 0.0):
            raise ValueError("GR model contains invalid values.")
        response = mg_lensing_response(self.ell_eff, params)
        return clpp_gr * response

    def chi2(self, model: np.ndarray) -> float:
        model = np.asarray(model, dtype=float)
        if model.shape != self.clpp.shape:
            raise ValueError("Model shape mismatch for lensing bandpower chi2.")
        if np.any(~np.isfinite(model)):
            return float("inf")
        r = self.clpp - model
        return float(r.T @ self.cov_inv @ r)

    def loglike(self, model: np.ndarray) -> float:
        chi2 = self.chi2(model)
        if not np.isfinite(chi2):
            return -np.inf
        sign, logdet = np.linalg.slogdet(self.cov)
        if sign <= 0:
            return -np.inf
        ll = -0.5 * (chi2 + logdet + self.clpp.size * np.log(2.0 * np.pi))
        return float(ll)
