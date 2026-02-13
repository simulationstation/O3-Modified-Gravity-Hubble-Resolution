from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
from scipy.special import logsumexp

import healpy as hp

from .dark_sirens_pe import PePixelDistanceHistogram
from .gw_distance_priors import GWDistancePrior


def compute_dark_siren_logL_draws_from_pe_hist_fast(
    *,
    event: str,
    pe: PePixelDistanceHistogram,
    z_grid_post: np.ndarray,
    dL_em_grid: np.ndarray,
    R_grid: np.ndarray,
    z_gal: np.ndarray,
    w_gal: np.ndarray,
    ipix_gal: np.ndarray,
    gw_distance_prior: GWDistancePrior | None = None,
    distance_mode: Literal["full", "spectral_only", "prior_only", "sky_only"] = "full",
    gal_chunk_size: int = 50_000,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast dark-siren catalog logL vectors using precomputed dL_em(z) and R(z) grids.

    This matches `dark_sirens_pe.compute_dark_siren_logL_draws_from_pe_hist`, but avoids rebuilding
    FRW distance integrals for each chunk by interpolating from precomputed arrays:

      dL_em_grid[j, k] ~= dL_em(z_grid_post[k])  for draw j
      R_grid[j, k]     ~= R(z_grid_post[k])      for draw j

    The PE histogram evaluation and distance-prior removal are otherwise identical.
    """
    if not bool(pe.nest):
        raise ValueError("PE pixel histogram must use NESTED ordering.")
    z = np.asarray(z_gal, dtype=float)
    w = np.asarray(w_gal, dtype=float)
    ipix = np.asarray(ipix_gal, dtype=np.int64)
    if z.ndim != 1 or w.ndim != 1 or ipix.ndim != 1 or not (z.shape == w.shape == ipix.shape):
        raise ValueError("z_gal/w_gal/ipix_gal must be 1D arrays with matching shapes.")
    if z.size == 0:
        raise ValueError("No galaxies provided.")

    z_grid_post = np.asarray(z_grid_post, dtype=float)
    dL_em_grid = np.asarray(dL_em_grid, dtype=float)
    R_grid = np.asarray(R_grid, dtype=float)
    if z_grid_post.ndim != 1:
        raise ValueError("z_grid_post must be 1D.")
    if dL_em_grid.ndim != 2 or R_grid.ndim != 2:
        raise ValueError("dL_em_grid and R_grid must be 2D (n_draws, n_z).")
    if dL_em_grid.shape != R_grid.shape:
        raise ValueError("dL_em_grid and R_grid must have the same shape.")
    if dL_em_grid.shape[1] != z_grid_post.size:
        raise ValueError("dL_em_grid.shape[1] must match z_grid_post.size.")

    n_draws = int(dL_em_grid.shape[0])
    if n_draws <= 0:
        raise ValueError("Need at least one draw.")

    # Map pixels to row indices in the selected set.
    npix = int(hp.nside2npix(int(pe.nside)))
    pix_to_row = np.full((npix,), -1, dtype=np.int32)
    pix_to_row[np.asarray(pe.pix_sel, dtype=np.int64)] = np.arange(int(pe.pix_sel.size), dtype=np.int32)
    row = pix_to_row[ipix]
    good = (row >= 0) & np.isfinite(z) & (z > 0.0) & np.isfinite(w) & (w > 0.0)
    if not np.any(good):
        raise ValueError("All galaxies map outside the PE credible region (or have invalid z/w).")
    z = z[good]
    w = w[good]
    row = row[good].astype(np.int64, copy=False)
    prob = np.asarray(pe.prob_pix, dtype=float)[row]

    chunk = int(gal_chunk_size)
    if chunk <= 0:
        raise ValueError("gal_chunk_size must be positive.")
    if z.size > chunk and not np.all(z[1:] >= z[:-1]):
        order = np.argsort(z, kind="mergesort")
        z = z[order]
        w = w[order]
        row = row[order]
        prob = prob[order]

    prior = gw_distance_prior or GWDistancePrior()
    edges = np.asarray(pe.dL_edges, dtype=float)
    nb = int(edges.size - 1)
    widths = np.diff(edges)
    dL_mid = 0.5 * (edges[:-1] + edges[1:])

    pdf_flat: np.ndarray | None = None
    pdf_1d: np.ndarray | None = None
    lognorm_prior_only: float | None = None
    if distance_mode == "full":
        pdf_flat = np.asarray(pe.pdf_bins, dtype=float).reshape(-1)
    elif distance_mode == "spectral_only":
        p_pix = np.asarray(pe.prob_pix, dtype=float)
        pdf_bins = np.asarray(pe.pdf_bins, dtype=float)
        p_sum = float(np.sum(p_pix))
        if not (np.isfinite(p_sum) and p_sum > 0.0):
            raise ValueError("Invalid prob_pix sum while building spectral_only distance density.")
        if int(pdf_bins.shape[0]) == int(p_pix.size):
            pdf_1d = np.sum(p_pix.reshape((-1, 1)) * pdf_bins, axis=0) / p_sum
        elif int(pdf_bins.shape[0]) == 1:
            pdf_1d = np.asarray(pdf_bins[0], dtype=float)
        else:
            raise ValueError("Incompatible pdf_bins shape for spectral_only mode.")
        pdf_1d = np.clip(np.asarray(pdf_1d, dtype=float), 0.0, np.inf)
        norm = float(np.sum(pdf_1d * widths))
        if not (np.isfinite(norm) and norm > 0.0):
            raise ValueError("Invalid spectral_only distance density normalization.")
        pdf_1d = pdf_1d / norm
    elif distance_mode == "prior_only":
        log_pi_mid = np.asarray(prior.log_pi_dL(np.clip(dL_mid, 1e-6, np.inf)), dtype=float)
        t = log_pi_mid + np.log(np.clip(widths, 1e-300, np.inf))
        m = np.isfinite(t)
        if not np.any(m):
            raise ValueError("Invalid π(dL) normalization in prior_only mode.")
        t0 = float(np.max(t[m]))
        lognorm_prior_only = float(t0 + np.log(float(np.sum(np.exp(t[m] - t0)))))
    elif distance_mode == "sky_only":
        pass
    else:
        raise ValueError("distance_mode must be full/spectral_only/prior_only/sky_only.")

    if distance_mode == "sky_only":
        logL0 = float(np.log(float(np.sum(w * prob))))
        out = np.full((n_draws,), logL0, dtype=float)
        return out, out.copy()

    logL_mu = np.full((n_draws,), -np.inf, dtype=float)
    logL_gr = np.full((n_draws,), -np.inf, dtype=float)

    for a in range(0, z.size, chunk):
        b = min(z.size, a + chunk)
        z_c = np.asarray(z[a:b], dtype=float)
        w_c = np.asarray(w[a:b], dtype=float)
        row_c = np.asarray(row[a:b], dtype=np.int64)
        prob_c = np.asarray(prob[a:b], dtype=float)

        if z_c.size > 1 and np.all(z_c[1:] >= z_c[:-1]):
            is_new = np.empty(z_c.shape, dtype=bool)
            is_new[0] = True
            is_new[1:] = z_c[1:] != z_c[:-1]
            z_u = z_c[is_new]
            inv = np.cumsum(is_new, dtype=np.int64) - 1
        else:
            z_u, inv = np.unique(z_c, return_inverse=True)

        # Interpolate dL_em(z) and R(z) for each draw on this chunk's unique z support.
        dL_em_u = np.empty((n_draws, int(z_u.size)), dtype=float)
        R_u = np.empty((n_draws, int(z_u.size)), dtype=float)
        for j in range(n_draws):
            dL_em_u[j] = np.interp(z_u, z_grid_post, dL_em_grid[j])
            R_u[j] = np.interp(z_u, z_grid_post, R_grid[j])
        dL_gw_u = dL_em_u * R_u

        logw = np.log(np.clip(w_c, 1e-30, np.inf))[None, :]
        logprob = np.log(np.clip(prob_c, 1e-300, np.inf))[None, :]

        def _chunk_logL(dL: np.ndarray) -> np.ndarray:
            dL = np.asarray(dL, dtype=float)
            bin_idx = np.searchsorted(edges, dL, side="right") - 1
            valid = (bin_idx >= 0) & (bin_idx < nb) & np.isfinite(dL) & (dL > 0.0)
            if distance_mode == "prior_only":
                assert lognorm_prior_only is not None
                logprior = prior.log_pi_dL(np.clip(dL, 1e-6, np.inf))
                ok = valid & np.isfinite(logprior)
                logterm = logw + logprob - float(lognorm_prior_only)
                logterm = np.where(ok, logterm, -np.inf)
                return logsumexp(logterm, axis=1)
            if distance_mode == "full":
                assert pdf_flat is not None
                lin = row_c.reshape((1, -1)) * nb + np.clip(bin_idx, 0, nb - 1)
                pdf = pdf_flat[lin]
                pdf = np.where(valid, pdf, 0.0)
            else:
                assert pdf_1d is not None
                pdf = pdf_1d[np.clip(bin_idx, 0, nb - 1)]
                pdf = np.where(valid, pdf, 0.0)

            logpdf = np.log(np.clip(pdf, 1e-300, np.inf))
            logprior = prior.log_pi_dL(np.clip(dL, 1e-6, np.inf))
            logterm = logw + logprob + logpdf - logprior
            logterm = np.where(np.isfinite(logprior), logterm, -np.inf)
            return logsumexp(logterm, axis=1)

        def _chunk_logL_grouped_by_z(dL_u: np.ndarray, *, logweight_u: np.ndarray) -> np.ndarray:
            dL_u = np.asarray(dL_u, dtype=float)
            bin_idx = np.searchsorted(edges, dL_u, side="right") - 1
            valid = (bin_idx >= 0) & (bin_idx < nb) & np.isfinite(dL_u) & (dL_u > 0.0)
            if distance_mode == "prior_only":
                assert lognorm_prior_only is not None
                logprior = prior.log_pi_dL(np.clip(dL_u, 1e-6, np.inf))
                ok = valid & np.isfinite(logprior)
                logterm = logweight_u - float(lognorm_prior_only)
                logterm = np.where(ok, logterm, -np.inf)
                return logsumexp(logterm, axis=1)
            assert pdf_1d is not None
            pdf = pdf_1d[np.clip(bin_idx, 0, nb - 1)]
            pdf = np.where(valid, pdf, 0.0)
            logpdf = np.log(np.clip(pdf, 1e-300, np.inf))
            logprior = prior.log_pi_dL(np.clip(dL_u, 1e-6, np.inf))
            logterm = logweight_u + logpdf - logprior
            logterm = np.where(np.isfinite(logprior), logterm, -np.inf)
            return logsumexp(logterm, axis=1)

        if distance_mode in ("spectral_only", "prior_only"):
            weight_u = np.bincount(inv, weights=(w_c * prob_c), minlength=int(z_u.size)).astype(float, copy=False)
            logweight_u = np.log(np.clip(weight_u, 1e-300, np.inf))[None, :]
            logL_mu = np.logaddexp(logL_mu, _chunk_logL_grouped_by_z(dL_gw_u, logweight_u=logweight_u))
            logL_gr = np.logaddexp(logL_gr, _chunk_logL_grouped_by_z(dL_em_u, logweight_u=logweight_u))
        else:
            dL_em = dL_em_u[:, inv]
            dL_gw = dL_gw_u[:, inv]
            logL_mu = np.logaddexp(logL_mu, _chunk_logL(dL_gw))
            logL_gr = np.logaddexp(logL_gr, _chunk_logL(dL_em))

        if progress_cb is not None:
            try:
                progress_cb(
                    {
                        "event": str(event),
                        "stage": "gal_chunk_done",
                        "galaxies_done": int(b),
                        "galaxies_total": int(z.size),
                        "uniq_z": int(z_u.size),
                    }
                )
            except Exception:
                pass

    return np.asarray(logL_mu, dtype=float), np.asarray(logL_gr, dtype=float)

