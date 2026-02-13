from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def logmeanexp_axis(logw: np.ndarray, *, axis: int) -> np.ndarray:
    """Stable log(mean(exp(logw))) along an axis."""
    x = np.asarray(logw, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    # Avoid NaN when all -inf.
    ok = np.isfinite(m)
    m = np.where(ok, m, 0.0)
    out = m + np.log(np.mean(np.exp(np.clip(x - m, -700.0, 50.0)), axis=axis, keepdims=True))
    out = np.where(ok, out, -np.inf)
    return np.squeeze(out, axis=axis)


def logsumexp_1d(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("logsumexp_1d expects 1D.")
    if x.size == 0:
        return float("-inf")
    m = float(np.max(x))
    if not np.isfinite(m):
        return float("-inf")
    s = float(np.sum(np.exp(np.clip(x - m, -700.0, 50.0))))
    if not (np.isfinite(s) and s > 0.0):
        return float("-inf")
    return float(m + np.log(s))


def trapz_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("trapz_weights expects 1D x with >=2 points.")
    if np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0.0):
        raise ValueError("x must be finite and strictly increasing.")
    dx = np.diff(x)
    w = np.empty_like(x)
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    if x.size > 2:
        w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return w


@dataclass(frozen=True)
class BetaPrior:
    mean: float
    kappa: float

    @property
    def alpha(self) -> float:
        m = float(self.mean)
        k = float(self.kappa)
        return float(m * k)

    @property
    def beta(self) -> float:
        m = float(self.mean)
        k = float(self.kappa)
        return float((1.0 - m) * k)

    def validate(self) -> None:
        if not (np.isfinite(self.mean) and 0.0 < float(self.mean) < 1.0):
            raise ValueError("BetaPrior.mean must be in (0,1).")
        if not (np.isfinite(self.kappa) and float(self.kappa) > 0.0):
            raise ValueError("BetaPrior.kappa must be > 0.")

    def logpdf(self, f: np.ndarray) -> np.ndarray:
        from scipy.special import betaln

        self.validate()
        f = np.asarray(f, dtype=float)
        a = float(self.alpha)
        b = float(self.beta)
        f = np.clip(f, 1e-300, 1.0 - 1e-300)
        return (a - 1.0) * np.log(f) + (b - 1.0) * np.log1p(-f) - float(betaln(a, b))


@dataclass(frozen=True)
class MarginalizedFMissResult:
    f_grid: np.ndarray  # (n_f,)
    posterior_mu: np.ndarray  # (n_f,)
    posterior_gr: np.ndarray  # (n_f,)
    log_integrand_mu: np.ndarray  # (n_f,)
    log_integrand_gr: np.ndarray  # (n_f,)
    lpd_mu_total: float
    lpd_gr_total: float
    lpd_mu_total_data: float
    lpd_gr_total_data: float

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "n_f": int(self.f_grid.size),
            "f_min": float(self.f_grid[0]) if self.f_grid.size else float("nan"),
            "f_max": float(self.f_grid[-1]) if self.f_grid.size else float("nan"),
            "f_post_mean_mu": float(np.sum(self.posterior_mu * self.f_grid)) if self.f_grid.size else float("nan"),
            "f_post_mean_gr": float(np.sum(self.posterior_gr * self.f_grid)) if self.f_grid.size else float("nan"),
            "lpd_mu_total": float(self.lpd_mu_total),
            "lpd_gr_total": float(self.lpd_gr_total),
            "delta_lpd_total": float(self.lpd_mu_total - self.lpd_gr_total),
            "lpd_mu_total_data": float(self.lpd_mu_total_data),
            "lpd_gr_total_data": float(self.lpd_gr_total_data),
            "delta_lpd_total_data": float(self.lpd_mu_total_data - self.lpd_gr_total_data),
        }


def _norm_logpost(log_int: np.ndarray) -> np.ndarray:
    log_int = np.asarray(log_int, dtype=float)
    if log_int.ndim != 1:
        raise ValueError("_norm_logpost expects 1D.")
    m = float(np.max(log_int))
    if not np.isfinite(m):
        return np.full_like(log_int, np.nan)
    w = np.exp(np.clip(log_int - m, -700.0, 50.0))
    s = float(np.sum(w))
    return (w / s) if s > 0 else np.full_like(w, np.nan)


def marginalize_f_miss_global(
    *,
    logL_cat_mu_by_event: list[np.ndarray],
    logL_cat_gr_by_event: list[np.ndarray],
    logL_missing_mu_by_event: list[np.ndarray],
    logL_missing_gr_by_event: list[np.ndarray],
    log_alpha_mu: np.ndarray | None,
    log_alpha_gr: np.ndarray | None,
    prior: BetaPrior,
    n_f: int = 401,
    eps: float = 1e-6,
) -> MarginalizedFMissResult:
    """Marginalize a *global* f_miss shared across events.

    This reproduces the pattern used in the production O3 dark-siren gap pipeline:
      - compute per-event mixture likelihoods on an f grid for each posterior draw
      - sum across events at the draw level
      - apply selection normalization at the draw level
      - take LPD = logmeanexp over draws at each f
      - integrate over f with a quadrature rule + explicit prior
    """
    if not (logL_cat_mu_by_event and logL_cat_gr_by_event):
        raise ValueError("Empty event list.")
    if not (len(logL_cat_mu_by_event) == len(logL_cat_gr_by_event) == len(logL_missing_mu_by_event) == len(logL_missing_gr_by_event)):
        raise ValueError("Event list length mismatch.")
    n_ev = int(len(logL_cat_mu_by_event))
    n_draws = int(np.asarray(logL_cat_mu_by_event[0], dtype=float).size)
    for arrs in (logL_cat_mu_by_event, logL_cat_gr_by_event, logL_missing_mu_by_event, logL_missing_gr_by_event):
        for a in arrs:
            if int(np.asarray(a).size) != n_draws:
                raise ValueError("All per-event logL arrays must have the same n_draws.")

    prior.validate()
    n_f = int(n_f)
    if n_f < 21:
        raise ValueError("n_f too small (need >=21).")
    eps = float(eps)
    if not (np.isfinite(eps) and 0.0 < eps < 0.1):
        raise ValueError("eps must be in (0,0.1).")

    f_grid = np.linspace(eps, 1.0 - eps, n_f, dtype=float)
    w_f = trapz_weights(f_grid)
    logw_f = np.log(np.clip(w_f, 1e-300, np.inf))
    logf = np.log(np.clip(f_grid, 1e-300, np.inf))
    log1mf = np.log1p(-np.clip(f_grid, 1e-300, 1.0))

    log_prior_f = prior.logpdf(f_grid)

    logL_mu_fd = np.zeros((n_f, n_draws), dtype=float)
    logL_gr_fd = np.zeros((n_f, n_draws), dtype=float)
    for i in range(n_ev):
        cat_mu = np.asarray(logL_cat_mu_by_event[i], dtype=float).reshape((1, -1))
        cat_gr = np.asarray(logL_cat_gr_by_event[i], dtype=float).reshape((1, -1))
        miss_mu = np.asarray(logL_missing_mu_by_event[i], dtype=float).reshape((1, -1))
        miss_gr = np.asarray(logL_missing_gr_by_event[i], dtype=float).reshape((1, -1))
        ev_mu = np.logaddexp(log1mf[:, None] + cat_mu, logf[:, None] + miss_mu)
        ev_gr = np.logaddexp(log1mf[:, None] + cat_gr, logf[:, None] + miss_gr)
        logL_mu_fd += ev_mu
        logL_gr_fd += ev_gr

    have_alpha = log_alpha_mu is not None and log_alpha_gr is not None
    if have_alpha:
        log_alpha_mu = np.asarray(log_alpha_mu, dtype=float).reshape((1, -1))
        log_alpha_gr = np.asarray(log_alpha_gr, dtype=float).reshape((1, -1))
        if log_alpha_mu.shape[1] != n_draws or log_alpha_gr.shape[1] != n_draws:
            raise ValueError("log_alpha arrays must match n_draws.")
        logL_mu_fd = logL_mu_fd - float(n_ev) * log_alpha_mu
        logL_gr_fd = logL_gr_fd - float(n_ev) * log_alpha_gr

    lpd_mu_f = logmeanexp_axis(logL_mu_fd, axis=1)  # (n_f,)
    lpd_gr_f = logmeanexp_axis(logL_gr_fd, axis=1)

    log_int_mu = log_prior_f + lpd_mu_f + logw_f
    log_int_gr = log_prior_f + lpd_gr_f + logw_f
    lpd_mu_total = logsumexp_1d(log_int_mu)
    lpd_gr_total = logsumexp_1d(log_int_gr)

    if have_alpha:
        assert log_alpha_mu is not None and log_alpha_gr is not None
        logL_mu_fd_data = logL_mu_fd + float(n_ev) * np.asarray(log_alpha_mu, dtype=float)
        logL_gr_fd_data = logL_gr_fd + float(n_ev) * np.asarray(log_alpha_gr, dtype=float)
        lpd_mu_f_data = logmeanexp_axis(logL_mu_fd_data, axis=1)
        lpd_gr_f_data = logmeanexp_axis(logL_gr_fd_data, axis=1)
        lpd_mu_total_data = logsumexp_1d(log_prior_f + lpd_mu_f_data + logw_f)
        lpd_gr_total_data = logsumexp_1d(log_prior_f + lpd_gr_f_data + logw_f)
    else:
        lpd_mu_total_data = float(lpd_mu_total)
        lpd_gr_total_data = float(lpd_gr_total)

    post_mu = _norm_logpost(log_int_mu)
    post_gr = _norm_logpost(log_int_gr)

    return MarginalizedFMissResult(
        f_grid=f_grid,
        posterior_mu=post_mu,
        posterior_gr=post_gr,
        log_integrand_mu=log_int_mu,
        log_integrand_gr=log_int_gr,
        lpd_mu_total=float(lpd_mu_total),
        lpd_gr_total=float(lpd_gr_total),
        lpd_mu_total_data=float(lpd_mu_total_data),
        lpd_gr_total_data=float(lpd_gr_total_data),
    )

