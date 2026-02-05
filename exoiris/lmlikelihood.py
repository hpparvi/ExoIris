#  ExoIris: fast, flexible, and easy exoplanet transmission spectroscopy in Python.
#  Copyright (C) 2026 Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from numba import njit

@njit
def _chol_solve(l: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve L L^T x = b given lower-triangular L."""
    y = np.linalg.solve(l, b)
    return np.linalg.solve(l.T, y)


@njit
def marginalized_loglike_mbl1d(
        obs: np.ndarray,
        mod: np.ndarray,
        covs: np.ndarray,
        sigma: np.ndarray,
        tau: float = 1e6,
        drop_constant: bool = False,
) -> float:
    """Compute the analytically marginalized log-likelihood for a multiplicative baseline model.

    JIT-compiled implementation of the collapsed (marginalized) log-likelihood for a
    model where systematic trends multiply the physical signal:

        obs = mod(θ) · covs·a + ε

    where ε ~ N(0, diag(σ²)) and a ~ N(0, Λ). The baseline coefficients a are
    integrated out analytically, yielding a likelihood that depends only on the
    physical model mod(θ).

    This formulation is appropriate when the out-of-transit baseline level must be
    estimated from the data. The design matrix `covs` should include a constant
    column (ones) to capture the baseline flux level, with additional columns for
    systematic trends.

    The computation avoids explicit matrix inversion by exploiting the Woodbury
    identity and matrix determinant lemma, working with k×k matrices rather than
    n×n matrices. This makes the function efficient when k ≪ n.

    Parameters
    ----------
    obs : ndarray of shape (n,), dtype float64
        Observed flux values. Must be contiguous float64 array.
    mod : ndarray of shape (n,), dtype float64
        Physical transit model evaluated at the current parameters θ, normalized
        such that the out-of-transit level is unity. Must be contiguous float64.
    covs : ndarray of shape (n, k), dtype float64
        Design matrix for the multiplicative baseline. Should include a constant
        column (ones) as the first column to capture the baseline flux level.
        Additional columns represent systematic trends (e.g., airmass, detector
        position, PSF width). Trend columns should typically be mean-centered.
        Must be contiguous float64 with C ordering.
    sigma : ndarray of shape (n,), dtype float64
        Per-observation measurement uncertainties (standard deviations). All
        values must be strictly positive. Must be contiguous float64.
    tau : float, default 1e6
        Prior standard deviation for all baseline coefficients (Λ = τ²I).
    drop_constant : bool, default False
        If True, omit terms constant in θ (log|Σ| and n·log(2π)). Use this for
        MCMC sampling over θ when σ is fixed, as these terms only shift the
        log-posterior by a constant without affecting sampling.

    Returns
    -------
    float
        The marginalized log-likelihood value. If drop_constant is True, this
        omits θ-independent normalization terms.

    Raises
    ------
    numpy.linalg.LinAlgError
        If Lambda or the internal matrix K is not positive definite (Cholesky
        decomposition fails). No other validation is performed.

    Notes
    -----
    The marginalized likelihood is obtained by integrating over the baseline
    coefficients:

        L(θ) = ∫ p(obs | θ, a) p(a) da

    Defining Φ = diag(mod)·covs, the marginal distribution of obs is:

        obs ~ N(0, C)  where  C = Σ + τ²ΦΦᵀ

    Rather than inverting the n×n matrix C directly, the implementation uses:

        C⁻¹ = W − τ²W·Φ·K⁻¹·Φᵀ·W     (Woodbury identity)
        |C| = |Σ|·|K|                 (matrix determinant lemma)

    where W = Σ⁻¹ = diag(1/σ²) and K = I + τ²ΦᵀWΦ is a k×k matrix.

    The log-likelihood is:

        log L = −½ [obsᵀC⁻¹obs + log|C| + n·log(2π)]
              = −½ [(obsᵀWobs − τ²cᵀK⁻¹c) + log|Σ| + log|K| + n·log(2π)]

    where c = ΦᵀW·obs.

    With the isotropic prior Λ = τ²I, the Cholesky factorization of Λ is trivially
    L_Λ = τI, eliminating one matrix decomposition compared to the general case.
    All operations involve only k×k matrices, giving O(nk² + k³) complexity
    rather than O(n³).

    In the limit τ → ∞, the marginalized likelihood approaches the profile
    likelihood with a at its maximum likelihood estimate (ordinary least squares).
    Numerical stability requires finite τ; values around 10⁶ are effectively
    uninformative for normalized flux data while maintaining well-conditioned K.
    """
    n = obs.shape[0]
    ncov = covs.shape[1]
    tau2 = tau * tau
    phi = mod[:, None] * covs
    w = 1.0 / (sigma * sigma)
    wphi = w[:, None] * phi
    k = np.eye(ncov) + tau2 * (phi.T @ wphi)
    c = phi.T @ (w * obs)
    lk = np.linalg.cholesky(k)
    obswobs = np.dot(w * obs, obs)
    kinvc = _chol_solve(lk, c)
    quad = obswobs - tau2 * np.dot(c, kinvc)
    logdetk = 2.0 * np.sum(np.log(np.diag(lk)))
    if drop_constant:
        return -0.5 * (quad + logdetk)
    logdetsigma = np.sum(np.log(sigma * sigma))
    return -0.5 * (quad + logdetsigma + logdetk + n * np.log(2.0 * np.pi))

@njit
def marginalized_loglike_mbl2d(
        obs: np.ndarray,
        mod: np.ndarray,
        err: np.ndarray,
        covs: np.ndarray,
        mask: np.ndarray,
        tau: float = 1e6,
) -> float:
    nwl = obs.shape[0]
    ncov = covs.shape[1]
    tau2 = tau * tau
    w = 1.0 / (err * err)

    lnlike = 0.0
    for i in range(nwl):
        m = mask[i]
        phi = mod[i, m, None] * covs[m]
        wphi = w[i, m, None] * phi
        k = np.eye(ncov) + tau2 * (phi.T @ wphi)
        c = phi.T @ (w[i, m] * obs[i, m])
        lk = np.linalg.cholesky(k)
        obswobs = np.dot(w[i, m] * obs[i, m], obs[i, m])
        kinvc = _chol_solve(lk, c)
        quad = obswobs - tau2 * np.dot(c, kinvc)
        logdetk = 2.0 * np.sum(np.log(np.diag(lk)))
        logdetsigma = np.sum(np.log(err[i, m] * err[i, m]))
        lnlike += -0.5 * (quad + logdetsigma + logdetk + mask[i].sum() * np.log(2.0 * np.pi))
    return lnlike