from __future__ import annotations

import numpy as np

from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.cosmology import build_background_from_H_grid
from entropy_horizon_recon.likelihoods import BaoLogLike, ChronometerLogLike, SNLogLike
from entropy_horizon_recon.recon_gp import reconstruct_H_gp
from entropy_horizon_recon.recon_spline import reconstruct_H_spline


def _h_lcdm(z: np.ndarray, *, h0: float, omega_m: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    e2 = omega_m * (1.0 + z) ** 3 + (1.0 - omega_m)
    return h0 * np.sqrt(e2)


def _mg_shift_ratio(z: np.ndarray) -> np.ndarray:
    """Smooth low-z MG-like enhancement used for synthetic injection tests."""
    z = np.asarray(z, dtype=float)
    return 1.0 + 0.05 * np.exp(-0.5 * ((z - 0.45) / 0.25) ** 2)


def _synthetic_likes_from_hubble(
    *,
    z_bg: np.ndarray,
    h_bg: np.ndarray,
    rd_true: float,
    constants: PhysicalConstants,
) -> tuple[SNLogLike, ChronometerLogLike, list[BaoLogLike]]:
    bg = build_background_from_H_grid(z_bg, h_bg, constants=constants)

    # SN synthetic magnitudes with a fixed absolute magnitude offset.
    z_sn = np.linspace(0.02, 1.0, 20)
    mu0 = 5.0 * np.log10(bg.Dl(z_sn))
    m_true = mu0 - 19.3
    cov_sn = np.diag(np.full(z_sn.size, 0.04**2))
    sn_like = SNLogLike.from_arrays(z=z_sn, m=m_true, cov=cov_sn)

    # Chronometer synthetic H(z).
    z_cc = np.linspace(0.0, 1.0, 9)
    h_cc = np.interp(z_cc, z_bg, h_bg)
    sigma_cc = np.full(z_cc.size, 2.0)
    cc_like = ChronometerLogLike.from_arrays(z=z_cc, H=h_cc, sigma_H=sigma_cc)

    # BAO synthetic points (DM/rd and DH/rd at two redshifts).
    z_bao = np.array([0.35, 0.35, 0.7, 0.7], dtype=float)
    obs_bao = np.array(["DM_over_rs", "DH_over_rs", "DM_over_rs", "DH_over_rs"])
    y_bao = np.array(
        [
            bg.Dm(np.array([0.35]))[0] / rd_true,
            bg.Dh(np.array([0.35]))[0] / rd_true,
            bg.Dm(np.array([0.7]))[0] / rd_true,
            bg.Dh(np.array([0.7]))[0] / rd_true,
        ],
        dtype=float,
    )
    cov_bao = np.diag(np.array([0.30**2, 0.20**2, 0.35**2, 0.25**2], dtype=float))
    bao_like = BaoLogLike.from_arrays(
        dataset="desi_2024_bao_all",
        z=z_bao,
        y=y_bao,
        obs=obs_bao,
        cov=cov_bao,
        constants=constants,
    )
    return sn_like, cc_like, [bao_like]


def _synthetic_likes() -> tuple[SNLogLike, ChronometerLogLike, list[BaoLogLike], PhysicalConstants]:
    constants = PhysicalConstants()
    h0_true = 70.0
    omega_m_true = 0.3
    rd_true = 147.0

    z_bg = np.linspace(0.0, 1.6, 900)
    h_bg = _h_lcdm(z_bg, h0=h0_true, omega_m=omega_m_true)
    sn_like, cc_like, bao_likes = _synthetic_likes_from_hubble(
        z_bg=z_bg,
        h_bg=h_bg,
        rd_true=rd_true,
        constants=constants,
    )
    return sn_like, cc_like, bao_likes, constants


def test_entropy_recon_spline_smoke_physical() -> None:
    sn_like, cc_like, bao_likes, constants = _synthetic_likes()
    z_knots = np.array([0.0, 0.25, 0.5, 0.8, 1.1], dtype=float)
    z_grid = np.linspace(0.0, 1.1, 80)

    post = reconstruct_H_spline(
        z_knots=z_knots,
        sn_like=sn_like,
        cc_like=cc_like,
        bao_likes=bao_likes,
        constants=constants,
        z_grid=z_grid,
        z_max_background=1.3,
        smooth_lambda=8.0,
        n_bootstrap=12,
        seed=7,
        r_d_init=147.0,
        monotone=True,
        monotone_tol=1e-3,
    )

    assert post.H_samples.shape == (12, z_grid.size)
    assert post.dH_dz_samples.shape == (12, z_grid.size)
    assert np.all(np.isfinite(post.H_samples))
    assert np.all(post.H_samples > 0.0)
    assert post.meta["bootstrap_success_fraction"] >= 0.8
    assert post.meta["monotone_fraction"] >= 0.95


def test_entropy_recon_spline_recovers_hubble_scale() -> None:
    sn_like, cc_like, bao_likes, constants = _synthetic_likes()
    z_knots = np.array([0.0, 0.25, 0.5, 0.8, 1.1], dtype=float)
    z_grid = np.linspace(0.0, 1.1, 80)

    post = reconstruct_H_spline(
        z_knots=z_knots,
        sn_like=sn_like,
        cc_like=cc_like,
        bao_likes=bao_likes,
        constants=constants,
        z_grid=z_grid,
        z_max_background=1.3,
        smooth_lambda=8.0,
        n_bootstrap=12,
        seed=9,
        r_d_init=147.0,
        monotone=True,
        monotone_tol=1e-3,
    )

    h_med = np.median(post.H_samples, axis=0)
    h_true = _h_lcdm(z_grid, h0=70.0, omega_m=0.3)

    z_mid = 0.6
    i_mid = int(np.argmin(np.abs(z_grid - z_mid)))
    frac_err_mid = abs(h_med[i_mid] - h_true[i_mid]) / h_true[i_mid]

    # Keep the threshold loose enough for deterministic CI stability.
    assert frac_err_mid < 0.15


def test_entropy_recon_spline_recovers_mg_shaped_shift() -> None:
    constants = PhysicalConstants()
    h0_true = 70.0
    omega_m_true = 0.3
    rd_true = 147.0

    z_bg = np.linspace(0.0, 1.6, 900)
    h_lcdm = _h_lcdm(z_bg, h0=h0_true, omega_m=omega_m_true)
    h_mg = h_lcdm * _mg_shift_ratio(z_bg)
    sn_like, cc_like, bao_likes = _synthetic_likes_from_hubble(
        z_bg=z_bg,
        h_bg=h_mg,
        rd_true=rd_true,
        constants=constants,
    )

    z_knots = np.array([0.0, 0.25, 0.5, 0.8, 1.1], dtype=float)
    z_grid = np.linspace(0.0, 1.1, 80)
    post = reconstruct_H_spline(
        z_knots=z_knots,
        sn_like=sn_like,
        cc_like=cc_like,
        bao_likes=bao_likes,
        constants=constants,
        z_grid=z_grid,
        z_max_background=1.3,
        smooth_lambda=8.0,
        n_bootstrap=12,
        seed=17,
        r_d_init=147.0,
        monotone=True,
        monotone_tol=1e-3,
    )

    h_med = np.median(post.H_samples, axis=0)
    h_lcdm_grid = _h_lcdm(z_grid, h0=h0_true, omega_m=omega_m_true)
    ratio_true = _mg_shift_ratio(z_grid)
    ratio_rec = h_med / h_lcdm_grid

    z_checks = [0.35, 0.6, 0.9]
    errs = []
    for zc in z_checks:
        i = int(np.argmin(np.abs(z_grid - zc)))
        errs.append(abs(ratio_rec[i] - ratio_true[i]))
    assert max(errs) < 0.06
    assert ratio_rec[int(np.argmin(np.abs(z_grid - 0.6)))] > 1.0


def test_entropy_recon_gp_and_spline_are_consistent() -> None:
    sn_like, cc_like, bao_likes, constants = _synthetic_likes()
    z_knots = np.array([0.0, 0.25, 0.5, 0.8, 1.1], dtype=float)
    z_grid = np.linspace(0.0, 1.1, 60)

    post_spline = reconstruct_H_spline(
        z_knots=z_knots,
        sn_like=sn_like,
        cc_like=cc_like,
        bao_likes=bao_likes,
        constants=constants,
        z_grid=z_grid,
        z_max_background=1.3,
        smooth_lambda=8.0,
        n_bootstrap=10,
        seed=21,
        r_d_init=147.0,
        monotone=True,
        monotone_tol=1e-3,
    )
    post_gp = reconstruct_H_gp(
        z_knots=z_knots,
        sn_like=sn_like,
        cc_like=cc_like,
        bao_likes=bao_likes,
        constants=constants,
        z_grid=z_grid,
        z_max_background=1.3,
        kernel="matern32",
        n_walkers=16,
        n_steps=90,
        n_burn=30,
        seed=21,
        n_processes=1,
        r_d_prior=(120.0, 170.0),
    )

    h_spline = np.median(post_spline.H_samples, axis=0)
    h_gp = np.median(post_gp.H_samples, axis=0)

    z_checks = [0.2, 0.5, 0.9]
    for zc in z_checks:
        i = int(np.argmin(np.abs(z_grid - zc)))
        frac = abs(h_gp[i] - h_spline[i]) / h_spline[i]
        assert frac < 0.12

    assert post_gp.meta["acceptance_fraction_mean"] > 0.02
