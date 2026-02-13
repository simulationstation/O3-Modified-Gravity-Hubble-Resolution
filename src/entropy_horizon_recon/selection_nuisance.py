from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dark_sirens_selection import (
    O3InjectionSet,
    _fit_injection_logit_model,  # noqa: SLF001
    _predict_injection_logit_pdet,  # noqa: SLF001
)
from .sirens import MuForwardPosterior, predict_dL_em, predict_r_gw_em
from .spline_basis import LinearSpline1D


def _chirp_mass_from_m1_m2(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    mt = np.clip(m1 + m2, 1e-12, np.inf)
    return (np.clip(m1 * m2, 1e-300, np.inf) ** (3.0 / 5.0)) / (mt ** (1.0 / 5.0))


@dataclass(frozen=True)
class SelectionNuisanceConfig:
    """Definition of the nuisance deformation family.

    All deformations are multiplicative in p_det:
      p_det' = clip(p_det * (1 + delta(logsnr)) * (1 + delta_z(z)) * (1 + b_mass * phi_mass), 0, 1)

    For fast adversarial scans we linearize around delta=0:
      p_det' ~ p_det * (1 + ...).
    """

    logsnr_knots: np.ndarray
    z_knots: np.ndarray | None = None
    mass_pivot_msun: float = 30.0
    mass_log_scale: float = 1.0

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "logsnr_knots": [float(x) for x in np.asarray(self.logsnr_knots, dtype=float).tolist()],
            "z_knots": None if self.z_knots is None else [float(x) for x in np.asarray(self.z_knots, dtype=float).tolist()],
            "mass_pivot_msun": float(self.mass_pivot_msun),
            "mass_log_scale": float(self.mass_log_scale),
        }


@dataclass(frozen=True)
class SelectionNuisanceMoments:
    """Precomputed linear moments for fast nuisance scans.

    For each posterior draw j:
      alpha(j) = E[w * p_det] / E[w]
      d_alpha/d(delta_k) = E[w * p_det * basis_k] / E[w]
    """

    cfg: SelectionNuisanceConfig
    alpha_mu_base: np.ndarray  # (n_draws,)
    alpha_gr_base: np.ndarray  # (n_draws,)
    mom_mu_logsnr: np.ndarray  # (n_draws, k_snr)
    mom_gr_logsnr: np.ndarray  # (n_draws, k_snr)
    mom_mu_z: np.ndarray | None = None  # (n_draws, k_z)
    mom_gr_z: np.ndarray | None = None
    mom_mu_mass: np.ndarray | None = None  # (n_draws,)
    mom_gr_mass: np.ndarray | None = None

    def to_jsonable(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "cfg": self.cfg.to_jsonable(),
            "n_draws": int(self.alpha_mu_base.size),
            "k_logsnr": int(self.mom_mu_logsnr.shape[1]),
            "k_z": None if self.mom_mu_z is None else int(self.mom_mu_z.shape[1]),
            "has_mass": bool(self.mom_mu_mass is not None),
        }
        return out


def compute_selection_nuisance_moments(
    *,
    post: MuForwardPosterior,
    injections: O3InjectionSet,
    convention: str,
    z_max: float,
    weight_mode: str,
    snr_offset: float,
    injection_logit_l2: float,
    injection_logit_max_iter: int,
    cfg: SelectionNuisanceConfig,
    mu_det_distance: str = "gw",
    pop_z_mode: str = "none",
    pop_z_powerlaw_k: float = 0.0,
    pop_mass_mode: str = "none",
    pop_m1_alpha: float = 2.3,
    pop_m_min: float = 5.0,
    pop_m_max: float = 80.0,
    pop_q_beta: float = 0.0,
    pop_m_taper_delta: float = 0.0,
    pop_m_peak: float = 35.0,
    pop_m_peak_sigma: float = 5.0,
    pop_m_peak_frac: float = 0.1,
    inj_mass_pdf_coords: str = "m1m2",
) -> SelectionNuisanceMoments:
    """Precompute linearized nuisance moments for the injection-logit selection proxy.

    This mirrors `compute_selection_alpha_from_injections`, but additionally returns
    per-draw moment vectors for the nuisance spline basis.
    """
    z_hi = float(min(float(z_max), float(post.z_grid[-1])))
    if z_hi <= 0.0:
        raise ValueError("z_max too small or invalid relative to posterior grid.")

    z = np.asarray(injections.z, dtype=float)
    dL_fid = np.asarray(injections.dL_mpc_fid, dtype=float)
    snr = np.asarray(injections.snr_net_opt, dtype=float)
    m1 = np.asarray(injections.m1_source, dtype=float)
    m2 = np.asarray(injections.m2_source, dtype=float)
    found_ifar = np.asarray(injections.found_ifar, dtype=bool)

    m = (
        np.isfinite(z)
        & (z > 0.0)
        & (z <= z_hi)
        & np.isfinite(dL_fid)
        & (dL_fid > 0.0)
        & np.isfinite(snr)
        & (snr > 0.0)
        & np.isfinite(m1)
        & (m1 > 0.0)
        & np.isfinite(m2)
        & (m2 > 0.0)
        & (m2 <= m1)
    )
    if not np.any(m):
        raise ValueError("No injections remain after basic cuts.")
    z = z[m]
    dL_fid = dL_fid[m]
    snr = snr[m]
    m1 = m1[m]
    m2 = m2[m]
    found_ifar = found_ifar[m]

    # Build injection importance weights w (match dark_sirens_selection).
    w = np.ones_like(z, dtype=float)
    mw = np.asarray(getattr(injections, "mixture_weight", np.ones_like(injections.z)), dtype=float)[m]
    if mw.shape != z.shape:
        raise ValueError("mixture_weight shape mismatch.")
    w = w * mw

    if str(weight_mode) == "none":
        pass
    elif str(weight_mode) == "inv_sampling_pdf":
        pdf = np.asarray(injections.sampling_pdf, dtype=float)[m]
        if not np.all(np.isfinite(pdf)) or np.any(pdf <= 0.0):
            raise ValueError("Invalid sampling_pdf for inv_sampling_pdf weighting.")
        w = w / pdf
    else:
        raise ValueError("Unknown weight_mode.")

    # Optional population factors (keep compatible with compute_selection_alpha_from_injections).
    if str(pop_z_mode) != "none":
        H0 = 67.7
        om0 = 0.31
        c = 299792.458
        z_grid = np.linspace(0.0, float(np.max(z)), 5001)
        Ez = np.sqrt(om0 * (1.0 + z_grid) ** 3 + (1.0 - om0))
        dc = (c / H0) * np.cumsum(np.concatenate([[0.0], 0.5 * (1.0 / Ez[1:] + 1.0 / Ez[:-1]) * np.diff(z_grid)]))
        dVdz = (c / (H0 * np.interp(z, z_grid, Ez))) * (np.interp(z, z_grid, dc) ** 2)
        base = dVdz / (1.0 + z)
        if str(pop_z_mode) == "comoving_uniform":
            w = w * base
        elif str(pop_z_mode) == "comoving_powerlaw":
            w = w * base * (1.0 + z) ** float(pop_z_powerlaw_k)
        else:
            raise ValueError("Unknown pop_z_mode.")

    if str(pop_mass_mode) != "none":
        if str(weight_mode) == "inv_sampling_pdf":
            if str(inj_mass_pdf_coords) == "m1m2":
                w = w / np.clip(m1, 1e-300, np.inf)
            elif str(inj_mass_pdf_coords) == "m1q":
                pass
            else:
                raise ValueError("Unknown inj_mass_pdf_coords.")

        alpha = float(pop_m1_alpha)
        mmin = float(pop_m_min)
        mmax = float(pop_m_max)
        beta_q = float(pop_q_beta)
        q = np.clip(m2 / m1, 1e-6, 1.0)

        if str(pop_mass_mode) == "powerlaw_q":
            good_m = (m1 >= mmin) & (m1 <= mmax) & (m2 >= mmin) & (m2 <= m1)
            w = w * good_m.astype(float) * (m1 ** (-alpha)) * (q ** beta_q)
        elif str(pop_mass_mode) == "powerlaw_q_smooth":
            delta = float(pop_m_taper_delta)
            if not (np.isfinite(delta) and delta > 0.0):
                raise ValueError("pop_m_taper_delta must be >0.")

            def _sig(x: np.ndarray) -> np.ndarray:
                return 0.5 * (1.0 + np.tanh(0.5 * x))

            t1 = _sig((m1 - mmin) / delta) * _sig((mmax - m1) / delta)
            t2 = _sig((m2 - mmin) / delta) * _sig((mmax - m2) / delta)
            taper = np.clip(t1 * t2, 0.0, 1.0)
            w = w * taper * (m1 ** (-alpha)) * (q ** beta_q)
        elif str(pop_mass_mode) == "powerlaw_peak_q_smooth":
            # Match the selection module's mixture in log space.
            from scipy.special import logsumexp

            delta = float(pop_m_taper_delta)
            if not (np.isfinite(delta) and delta > 0.0):
                raise ValueError("pop_m_taper_delta must be >0.")
            mp = float(pop_m_peak)
            sig = float(pop_m_peak_sigma)
            f_peak = float(pop_m_peak_frac)
            if not (np.isfinite(f_peak) and 0.0 <= f_peak <= 1.0):
                raise ValueError("pop_m_peak_frac must be in [0,1].")

            def _sig(x: np.ndarray) -> np.ndarray:
                return 0.5 * (1.0 + np.tanh(0.5 * x))

            t1 = _sig((m1 - mmin) / delta) * _sig((mmax - m1) / delta)
            t2 = _sig((m2 - mmin) / delta) * _sig((mmax - m2) / delta)
            taper = np.clip(t1 * t2, 1e-300, 1.0)

            log_q = beta_q * np.log(np.clip(q, 1e-300, np.inf))
            log_taper = np.log(taper)
            log_pl = -alpha * np.log(np.clip(m1, 1e-300, np.inf)) + log_q + log_taper
            log_peak = -0.5 * ((m1 - mp) / sig) ** 2 - np.log(sig) + log_q + log_taper
            if f_peak <= 0.0:
                log_mass = log_pl
            elif f_peak >= 1.0:
                log_mass = log_peak
            else:
                log_mass = logsumexp(
                    np.stack([np.log(1.0 - f_peak) + log_pl, np.log(f_peak) + log_peak], axis=0),
                    axis=0,
                )
            m_ok = np.isfinite(log_mass)
            log_mass = log_mass - float(np.nanmax(log_mass[m_ok]))
            w = w * np.exp(log_mass)
        else:
            raise ValueError("Unknown pop_mass_mode.")

    good_w = np.isfinite(w) & (w > 0.0)
    if not np.any(good_w):
        raise ValueError("All injections received zero/invalid weights.")
    if not np.all(good_w):
        z = z[good_w]
        dL_fid = dL_fid[good_w]
        snr = snr[good_w]
        m1 = m1[good_w]
        m2 = m2[good_w]
        w = w[good_w]
        found_ifar = found_ifar[good_w]

    wsum = float(np.sum(w))
    if not (np.isfinite(wsum) and wsum > 0.0):
        raise ValueError("Sum of injection weights is non-positive.")

    # Fit injection-logit model on fiducial effective SNR.
    beta, feat_mu, feat_sig = _fit_injection_logit_model(
        snr_eff=snr - float(snr_offset),
        z=z,
        m1=m1,
        m2=m2,
        found_ifar=found_ifar,
        l2=float(injection_logit_l2),
        max_iter=int(injection_logit_max_iter),
    )

    # Set up spline basis objects.
    snr_spline = LinearSpline1D(np.asarray(cfg.logsnr_knots, dtype=float))
    z_spline = None if cfg.z_knots is None else LinearSpline1D(np.asarray(cfg.z_knots, dtype=float))

    # Mass basis vector (fixed for injections).
    mc = _chirp_mass_from_m1_m2(m1, m2)
    phi_mass = (np.log(np.clip(mc, 1e-12, np.inf)) - np.log(float(cfg.mass_pivot_msun))) / float(cfg.mass_log_scale)
    phi_mass = np.asarray(phi_mass, dtype=float)

    z_grid_post = np.asarray(post.z_grid, dtype=float)
    dL_em_grid = predict_dL_em(post, z_eval=z_grid_post)  # (n_draws, n_z)
    _, R_grid = predict_r_gw_em(post, z_eval=None, convention=str(convention))
    dL_gw_grid = dL_em_grid * np.asarray(R_grid, dtype=float)

    n_draws = int(dL_em_grid.shape[0])
    alpha_mu_base = np.empty((n_draws,), dtype=float)
    alpha_gr_base = np.empty((n_draws,), dtype=float)
    mom_mu_logsnr = np.empty((n_draws, int(snr_spline.n_knots)), dtype=float)
    mom_gr_logsnr = np.empty((n_draws, int(snr_spline.n_knots)), dtype=float)
    mom_mu_z = None
    mom_gr_z = None
    if z_spline is not None:
        mom_mu_z = np.empty((n_draws, int(z_spline.n_knots)), dtype=float)
        mom_gr_z = np.empty((n_draws, int(z_spline.n_knots)), dtype=float)
    mom_mu_mass = np.empty((n_draws,), dtype=float)
    mom_gr_mass = np.empty((n_draws,), dtype=float)

    # Precompute z basis once (does not depend on draw/model).
    B_z = None if z_spline is None else z_spline.basis(z)

    mu_det_distance = str(mu_det_distance)
    if mu_det_distance not in ("gw", "em"):
        raise ValueError("mu_det_distance must be 'gw' or 'em'.")

    for j in range(n_draws):
        dL_em = np.interp(z, z_grid_post, dL_em_grid[j])
        dL_gw = np.interp(z, z_grid_post, dL_gw_grid[j])
        dL_mu_det = dL_gw if mu_det_distance == "gw" else dL_em

        snr_gr_eff = snr * (dL_fid / np.clip(dL_em, 1e-6, np.inf)) - float(snr_offset)
        snr_mu_eff = snr * (dL_fid / np.clip(dL_mu_det, 1e-6, np.inf)) - float(snr_offset)

        p_gr = _predict_injection_logit_pdet(
            snr_eff=snr_gr_eff,
            z=z,
            m1=m1,
            m2=m2,
            beta=beta,
            feat_mu=feat_mu,
            feat_sig=feat_sig,
        )
        p_mu = _predict_injection_logit_pdet(
            snr_eff=snr_mu_eff,
            z=z,
            m1=m1,
            m2=m2,
            beta=beta,
            feat_mu=feat_mu,
            feat_sig=feat_sig,
        )

        # Base alpha.
        alpha_gr_base[j] = float(np.sum(w * p_gr) / wsum)
        alpha_mu_base[j] = float(np.sum(w * p_mu) / wsum)

        # Linear moments: E[w * p_det * basis] / wsum.
        logsnr_gr = np.log(np.clip(snr_gr_eff, 1e-6, np.inf))
        logsnr_mu = np.log(np.clip(snr_mu_eff, 1e-6, np.inf))
        B_gr = snr_spline.basis(logsnr_gr)
        B_mu = snr_spline.basis(logsnr_mu)
        wp_gr = (w * p_gr).reshape((-1, 1))
        wp_mu = (w * p_mu).reshape((-1, 1))
        mom_gr_logsnr[j, :] = (wp_gr * B_gr).sum(axis=0) / wsum
        mom_mu_logsnr[j, :] = (wp_mu * B_mu).sum(axis=0) / wsum

        if B_z is not None and mom_gr_z is not None and mom_mu_z is not None:
            mom_gr_z[j, :] = (wp_gr * B_z).sum(axis=0) / wsum
            mom_mu_z[j, :] = (wp_mu * B_z).sum(axis=0) / wsum

        # Mild mass dependence moment.
        mom_gr_mass[j] = float(np.sum(w * p_gr * phi_mass) / wsum)
        mom_mu_mass[j] = float(np.sum(w * p_mu * phi_mass) / wsum)

    return SelectionNuisanceMoments(
        cfg=cfg,
        alpha_mu_base=np.asarray(alpha_mu_base, dtype=float),
        alpha_gr_base=np.asarray(alpha_gr_base, dtype=float),
        mom_mu_logsnr=np.asarray(mom_mu_logsnr, dtype=float),
        mom_gr_logsnr=np.asarray(mom_gr_logsnr, dtype=float),
        mom_mu_z=None if mom_mu_z is None else np.asarray(mom_mu_z, dtype=float),
        mom_gr_z=None if mom_gr_z is None else np.asarray(mom_gr_z, dtype=float),
        mom_mu_mass=np.asarray(mom_mu_mass, dtype=float),
        mom_gr_mass=np.asarray(mom_gr_mass, dtype=float),
    )


def apply_nuisance_to_alpha_linearized(
    *,
    mom: SelectionNuisanceMoments,
    delta_logsnr_knots: np.ndarray,
    delta_z_knots: np.ndarray | None = None,
    b_mass: float = 0.0,
    clip_alpha: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the linearized nuisance model and return (alpha_mu, alpha_gr)."""
    dl = np.asarray(delta_logsnr_knots, dtype=float)
    if dl.ndim != 1 or dl.size != int(mom.mom_mu_logsnr.shape[1]):
        raise ValueError("delta_logsnr_knots shape mismatch.")
    alpha_mu = mom.alpha_mu_base + mom.mom_mu_logsnr @ dl
    alpha_gr = mom.alpha_gr_base + mom.mom_gr_logsnr @ dl

    if delta_z_knots is not None:
        if mom.mom_mu_z is None or mom.mom_gr_z is None:
            raise ValueError("This moments object has no z basis; pass delta_z_knots=None.")
        dz = np.asarray(delta_z_knots, dtype=float)
        if dz.ndim != 1 or dz.size != int(mom.mom_mu_z.shape[1]):
            raise ValueError("delta_z_knots shape mismatch.")
        alpha_mu = alpha_mu + mom.mom_mu_z @ dz
        alpha_gr = alpha_gr + mom.mom_gr_z @ dz

    b = float(b_mass)
    if b != 0.0:
        if mom.mom_mu_mass is None or mom.mom_gr_mass is None:
            raise ValueError("This moments object has no mass moment.")
        alpha_mu = alpha_mu + b * mom.mom_mu_mass
        alpha_gr = alpha_gr + b * mom.mom_gr_mass

    if clip_alpha:
        alpha_mu = np.clip(alpha_mu, 1e-8, 1.0 - 1e-8)
        alpha_gr = np.clip(alpha_gr, 1e-8, 1.0 - 1e-8)
    return np.asarray(alpha_mu, dtype=float), np.asarray(alpha_gr, dtype=float)


def apply_nuisance_to_alpha_logit_resummed(
    *,
    mom: SelectionNuisanceMoments,
    delta_logsnr_knots: np.ndarray,
    delta_z_knots: np.ndarray | None = None,
    b_mass: float = 0.0,
    clip_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Nonlinear, positivity-preserving alpha mapping for bounded nuisance scans.

    We start from the same first-order alpha perturbation implied by the precomputed moments,
    but apply it in *logit(alpha)* space rather than alpha space. This:
    - matches the linearized response for small perturbations, and
    - avoids hard clipping pathologies when exploring larger bounds (e.g. >=30%),
      since alpha is guaranteed to remain in (0,1).

    This is a robustness tool; it is not a replacement for a full nonlinear recomputation of
    p_det under nuisance deformations at the injection level.
    """
    dl = np.asarray(delta_logsnr_knots, dtype=float)
    if dl.ndim != 1 or dl.size != int(mom.mom_mu_logsnr.shape[1]):
        raise ValueError("delta_logsnr_knots shape mismatch.")

    # First-order alpha perturbations from the stored moments.
    d_mu = mom.mom_mu_logsnr @ dl
    d_gr = mom.mom_gr_logsnr @ dl

    if delta_z_knots is not None:
        if mom.mom_mu_z is None or mom.mom_gr_z is None:
            raise ValueError("This moments object has no z basis; pass delta_z_knots=None.")
        dz = np.asarray(delta_z_knots, dtype=float)
        if dz.ndim != 1 or dz.size != int(mom.mom_mu_z.shape[1]):
            raise ValueError("delta_z_knots shape mismatch.")
        d_mu = d_mu + mom.mom_mu_z @ dz
        d_gr = d_gr + mom.mom_gr_z @ dz

    b = float(b_mass)
    if b != 0.0:
        if mom.mom_mu_mass is None or mom.mom_gr_mass is None:
            raise ValueError("This moments object has no mass moment.")
        d_mu = d_mu + b * mom.mom_mu_mass
        d_gr = d_gr + b * mom.mom_gr_mass

    eps = float(max(1e-12, clip_eps))
    a0_mu = np.clip(np.asarray(mom.alpha_mu_base, dtype=float), eps, 1.0 - eps)
    a0_gr = np.clip(np.asarray(mom.alpha_gr_base, dtype=float), eps, 1.0 - eps)

    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
        return np.log(p) - np.log1p(-p)

    def _expit(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        # Stable sigmoid.
        out = np.empty_like(x)
        m = x >= 0
        out[m] = 1.0 / (1.0 + np.exp(-x[m]))
        ex = np.exp(x[~m])
        out[~m] = ex / (1.0 + ex)
        return out

    # Map the linear alpha perturbation to a logit-space shift using the local derivative
    # d alpha / d logit = alpha * (1-alpha).
    denom_mu = np.clip(a0_mu * (1.0 - a0_mu), 1e-12, np.inf)
    denom_gr = np.clip(a0_gr * (1.0 - a0_gr), 1e-12, np.inf)
    x_mu = np.asarray(d_mu, dtype=float) / denom_mu
    x_gr = np.asarray(d_gr, dtype=float) / denom_gr

    a_mu = _expit(_logit(a0_mu) + x_mu)
    a_gr = _expit(_logit(a0_gr) + x_gr)
    a_mu = np.clip(a_mu, eps, 1.0 - eps)
    a_gr = np.clip(a_gr, eps, 1.0 - eps)
    return np.asarray(a_mu, dtype=float), np.asarray(a_gr, dtype=float)
