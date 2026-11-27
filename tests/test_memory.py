"""Regression tests for memory leaks in the likelihood evaluation."""
import numpy as np
import pytest
from numba.core.runtime import rtsys, _nrt_python as nrt

from exoiris import ExoIris, TSData


def _make_analysis(noise_model: str, nldc: int = 3) -> ExoIris:
    rng = np.random.default_rng(0)
    time = np.linspace(-0.1, 0.1, 60)
    wavelength = np.linspace(1.0, 2.0, 20)
    fluxes = 1.0 + 1e-4 * rng.normal(size=(wavelength.size, time.size))
    data = TSData(time, wavelength, fluxes, np.full_like(fluxes, 1e-4), name='a')
    ei = ExoIris('test', ldmodel='quadratic-tri', data=data, nk=5, nldc=nldc, noise_model=noise_model)
    ei.set_prior('tc_00', 'NP', 0.0, 0.01)
    ei.set_prior('p', 'NP', 3.0, 1e-5)
    ei.set_prior('rho', 'UP', 1.0, 3.0)
    ei.set_radius_ratio_prior('UP', 0.05, 0.15)
    return ei


@pytest.mark.parametrize('noise_model', ['white_profiled', 'white_marginalized'])
def test_nonfinite_model_does_not_leak(noise_model):
    """A NaN transit model must not leak numba-allocated memory.

    Negative q1 makes the triangular quadratic limb darkening law return NaNs.
    These used to make the numba-compiled linear algebra routines raise, which
    leaked every array the routines had allocated.
    """
    ei = _make_analysis(noise_model)
    pvp = ei._tsa.ps.sample_from_prior(8)
    pvp[:4, ei._tsa._sl_ld] = -0.1

    nrt.memsys_enable_stats()
    try:
        ei._tsa.lnlikelihood(pvp)  # Warm-up: compile everything
        before = rtsys.get_allocation_stats()
        for _ in range(5):
            lnl = ei._tsa.lnlikelihood(pvp)
        after = rtsys.get_allocation_stats()
    finally:
        nrt.memsys_disable_stats()

    assert not np.isfinite(lnl[:4]).any()
    assert np.isfinite(lnl[4:]).all()
    assert (after.alloc - before.alloc) == (after.free - before.free)


def test_quadratic_tri_domain_is_rejected_before_model_evaluation(monkeypatch):
    """Coefficients outside [0, 1] must be rejected by the prior without evaluating the model."""
    ei = _make_analysis('white_profiled')
    tsa = ei._tsa
    pvp = tsa.ps.sample_from_prior(8)
    pvp[:4, tsa._sl_ld] = -0.1

    assert (tsa.lnprior(pvp)[:4] == -np.inf).all()
    assert np.isfinite(tsa.lnprior(pvp)[4:]).all()

    evaluated = []
    lnlikelihood = tsa.lnlikelihood
    monkeypatch.setattr(tsa, 'lnlikelihood', lambda pv: evaluated.append(len(pv)) or lnlikelihood(pv))

    lnp = tsa.lnposterior(pvp)
    assert evaluated == [4]
    assert (lnp[:4] == -np.inf).all()
    assert np.isfinite(lnp[4:]).all()

    # Scalar in, scalar out, with and without a likelihood evaluation.
    assert tsa.lnposterior(pvp[0]) == -np.inf
    assert np.isfinite(tsa.lnposterior(pvp[5]))
    assert np.allclose(lnp[4:], [tsa.lnposterior(pv) for pv in pvp[4:]])


def test_interpolated_overshoot_is_rejected():
    """Knot values inside the domain can still give interpolated coefficients outside it."""
    tsa = _make_analysis('white_profiled', nldc=5)._tsa
    pv = tsa.ps.sample_from_prior(1)
    pv[0, tsa._sl_ld] = 0.5
    pv[0, tsa._sl_ld][::2] = [0.0, 0.0, 1.0, 1.0, 1.0]   # A step in q1 makes the spline ring
    ldc = tsa._eval_ldc(pv)[0]
    assert ldc.min() < 0.0 or ldc.max() > 1.0
    assert tsa.lnprior(pv)[0] == -np.inf


def _unreleased(f, n: int = 5) -> int:
    """Number of numba allocations a repeatedly called function leaves unreleased."""
    nrt.memsys_enable_stats()
    try:
        f()  # Warm-up: compile everything
        before = rtsys.get_allocation_stats()
        for _ in range(n):
            f()
        after = rtsys.get_allocation_stats()
    finally:
        nrt.memsys_disable_stats()
    return (after.alloc - before.alloc) - (after.free - before.free)


def _baseline_inputs():
    rng = np.random.default_rng(1)
    nwl, npt = 12, 40
    covs = np.ascontiguousarray(np.vstack([np.ones(npt), np.linspace(-1, 1, npt)]).T)
    obs = 1.0 + 1e-4 * rng.normal(size=(nwl, npt))
    err = np.full((nwl, npt), 1e-4)
    return covs, obs, err


def test_nlstsq_handles_unsolvable_rows_without_leaking():
    from exoiris.tslpf import nlstsq

    covs, obs, _ = _baseline_inputs()
    mask = np.ones(obs.shape, bool)
    mask[2] = False        # A fully masked wavelength
    mask[4, ::3] = False   # A partially masked wavelength
    res = np.where(mask, obs, np.nan)
    wlmask = mask.all(1)
    with_nans = np.where(~wlmask)[0]

    x = nlstsq(covs, res, mask, wlmask, with_nans)
    assert np.isnan(x[:, 2]).all()
    assert np.isfinite(np.delete(x, 2, axis=1)).all()
    assert np.allclose(x[:, 4], np.linalg.lstsq(covs[mask[4]], obs[4, mask[4]], rcond=None)[0])
    assert np.allclose(x[:, 0], np.linalg.lstsq(covs, obs[0], rcond=None)[0])

    # A non-finite residual the mask does not know about must give NaNs rather than an exception.
    res_bad = res.copy()
    res_bad[0, 3] = np.inf
    assert np.isnan(nlstsq(covs, res_bad, mask, wlmask, with_nans)).all()

    assert _unreleased(lambda: nlstsq(covs, res, mask, wlmask, with_nans)) == 0
    assert _unreleased(lambda: nlstsq(covs, res_bad, mask, wlmask, with_nans)) == 0


@pytest.mark.parametrize('case', ['nan_model', 'zero_error', 'huge_model', 'underdetermined_row'])
def test_marginalized_likelihood_does_not_raise_or_leak(case):
    from exoiris.lmlikelihood import marginalized_loglike_mbl2d

    covs, obs, err = _baseline_inputs()
    mod = np.ones(obs.shape)
    mask = np.ones(obs.shape, bool)
    assert np.isfinite(marginalized_loglike_mbl2d(obs, mod, err, covs, mask))

    if case == 'nan_model':
        mod[1, 5] = np.nan
    elif case == 'zero_error':
        err[1, 5] = 0.0
    elif case == 'huge_model':
        mod *= 1e150
    else:
        mask[1, 1:] = False

    f = lambda: marginalized_loglike_mbl2d(obs, mod, err, covs, mask)
    if case != 'underdetermined_row':
        assert f() == -np.inf
    assert _unreleased(f) == 0


def test_cholesky_helpers_match_numpy():
    from exoiris.lmlikelihood import _cholesky, _chol_solve

    rng = np.random.default_rng(2)
    for n in (1, 2, 4, 7):
        a = rng.normal(size=(n, n))
        a = a @ a.T + n * np.eye(n)
        b = rng.normal(size=n)
        l, ok = _cholesky(a)
        assert ok
        assert np.allclose(l, np.linalg.cholesky(a))
        assert np.allclose(_chol_solve(l, b), np.linalg.solve(a, b))

    assert not _cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))[1]
    assert not _cholesky(np.array([[1.0, np.nan], [np.nan, 1.0]]))[1]
    assert not _cholesky(np.array([[np.inf, 0.0], [0.0, 1.0]]))[1]
