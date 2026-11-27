"""Tests for eccentric orbits, the spot priors, the saved GP kernels, and the plots with given axes."""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest
from celerite2 import terms
from matplotlib.pyplot import close, subplots
from pytransit.orbits import as_from_rhop, d_from_pkaiews

from exoiris import ExoIris, TSData, load_model
from meepmeep.backends.numba.utils import i_from_baew

RHO, P, B, K = 2.0, 3.0, 0.3, 0.1


def _make_analysis(data, noise_model: str = 'white_profiled', ecc: float = 0.0, name: str = 'test') -> ExoIris:
    ei = ExoIris(name, ldmodel='power-2', data=data, nk=4, noise_model=noise_model)
    ei.set_prior('tc_00', 'NP', 0.0, 0.01)
    ei.set_prior('p', 'NP', P, 1e-5)
    ei.set_prior('rho', 'NP', RHO, 0.02)
    ei.set_radius_ratio_prior('UP', 0.05, 0.15)
    if ecc > 0.0:
        ei.set_prior('secw', 'NP', 0.0, 1e-3)
        ei.set_prior('sesw', 'NP', np.sqrt(ecc), 1e-3)
    return ei


def _make_data(ecc: float = 0.0, seed: int = 0) -> TSData:
    """Simulated transit of a planet with the periastron towards the observer."""
    rng = np.random.default_rng(seed)
    time = np.linspace(-0.12, 0.12, 150)
    wavelength = np.linspace(1.0, 2.0, 6)
    flat = TSData(time, wavelength, np.ones((6, 150)), np.full((6, 150), 1e-4), name='a')
    tsa = _make_analysis(flat, ecc=ecc)._tsa
    pv = tsa.ps.mean_pv.copy()
    pv[tsa.ps.find_pid('rho')], pv[tsa.ps.find_pid('b')] = RHO, B
    pv[tsa._sl_rratios] = K
    fluxes = tsa.flux_model(pv, include_baseline=False)[0][0] + 1e-4 * rng.normal(size=(6, 150))
    return TSData(time, wavelength, fluxes, np.full((6, 150), 1e-4), name='a')


# -- Eccentric orbits ---------------------------------------------------------

def test_i_from_baew():
    aor = as_from_rhop(RHO, P)
    assert i_from_baew(B, aor, 0.0, 0.0) == pytest.approx(np.arccos(B / aor))
    for ecc, w in ((0.3, 0.5 * np.pi), (0.5, -1.0), (0.1, 2.5)):
        inc = i_from_baew(B, aor, ecc, w)
        # Sky-projected distance at the transit centre, where the true anomaly is pi/2 - w.
        r = aor * (1 - ecc**2) / (1 + ecc * np.sin(w))
        assert r * np.cos(inc) == pytest.approx(B)
    assert np.isnan(i_from_baew(2 * aor, aor, 0.0, 0.0))


@pytest.mark.parametrize('ecc', [0.0, 0.3])
def test_white_fit_follows_the_eccentricity(ecc):
    aor = as_from_rhop(RHO, P)
    w = 0.5 * np.pi
    t14 = d_from_pkaiews(P, K, aor, i_from_baew(B, aor, ecc, w), ecc, w, 1, 14)

    ei = _make_analysis(_make_data(ecc), ecc=ecc)
    ei.fit_white(niter=150)
    wa = ei._wa
    pv = wa._local_minimization.x
    assert ei.transit_duration == pytest.approx(t14, rel=0.03)
    assert ei.zero_epoch == pytest.approx(0.0, abs=1e-3)
    assert pv[wa.ps.find_pid('sesw')]**2 == pytest.approx(ecc, abs=0.01)
    assert np.sqrt(pv[wa.ps.find_pid('k2')]) == pytest.approx(K, rel=0.03)
    # The first three parameters keep their places.
    assert list(wa.ps.names[:3]) == ['p', 'rho', 'b']
    # The out-of-transit mask covers the transit and no more.
    in_transit = ~ei.data[0].transit_mask
    assert in_transit.sum() * np.diff(ei.data[0].time)[0] == pytest.approx(t14, rel=0.06)


def test_fit_starts_from_the_white_fit_eccentricity():
    ei = _make_analysis(_make_data(0.3), ecc=0.3)
    ei.fit_white(niter=150)
    ei.fit(niter=1, npop=2 * ei._tsa.ndim + 2, plot_convergence=False)
    pop = ei._tsa._de_population
    assert np.isfinite(ei._tsa.lnposterior(pop)).all()
    assert pop[:, ei.ps.find_pid('sesw')].mean() == pytest.approx(np.sqrt(0.3), abs=0.01)


# -- Spot priors --------------------------------------------------------------

@pytest.mark.parametrize('tphot, tmin', [(5000.0, 3000.0), (3500.0, 2500.0), (2000.0, 1200.0)])
def test_spot_temperature_prior_ends_at_the_photosphere(tphot, tmin):
    ei = _make_analysis(_make_data())
    ei.initialize_spots(tphot, 1.5)
    ei.add_spot(0)
    prior = ei.ps[ei.ps.find_pid('spt_01')].prior
    assert (prior.a, prior.b) == (tmin, tphot)


def test_spot_parameter_descriptions():
    ei = _make_analysis(_make_data())
    ei.initialize_spots(5000.0, 1.5)
    ei.add_spot(0)
    ei.add_spot(0)
    descriptions = [p.description for p in ei.ps if p.name.startswith('sp')]
    assert 'spot 01 center' in descriptions and 'spot 02 temperature' in descriptions
    assert not any('{' in d for d in descriptions)


# -- GP kernels ---------------------------------------------------------------

def test_custom_gp_kernel_survives_save(tmp_path, monkeypatch, recwarn):
    monkeypatch.chdir(tmp_path)
    ei = _make_analysis(_make_data(), noise_model='fixed_gp', name='kernel')
    ei.set_gp_kernel(terms.SHOTerm(sigma=2e-4, rho=0.05, Q=0.6) + terms.Matern32Term(sigma=1e-4, rho=0.01))
    pv = ei._tsa.ps.mean_pv.copy()
    pv[ei._tsa._sl_rratios] = K
    lnl = ei._tsa.lnlikelihood(pv)
    ei.save()
    assert not [w for w in recwarn if 'kernel' in str(w.message)]

    b = load_model('kernel.fits')
    assert isinstance(b._tsa._gp[0].kernel, terms.TermSum)
    assert b._tsa.lnlikelihood(pv) == pytest.approx(lnl, rel=1e-10)


def test_set_gp_kernel_for_one_data_set():
    a, b = _make_data(seed=0), _make_data(seed=1)
    b.name = 'b'
    ei = _make_analysis(a + b, noise_model='fixed_gp')
    before = ei._tsa._gp[0].kernel
    kernel = terms.SHOTerm(sigma=2e-4, rho=0.05, Q=0.6)
    ei.set_gp_kernel(kernel, 1)
    assert ei._tsa._gp[0].kernel is before
    assert ei._tsa._gp[1].kernel is kernel
    ei.set_gp_kernel(kernel)
    assert ei._tsa._gp[0].kernel is kernel


# -- Plots with given axes ----------------------------------------------------

def test_white_plots_with_given_axes():
    ei = _make_analysis(_make_data(), noise_model='fixed_gp')
    ei.fit_white(niter=50)
    fig, axs = subplots(1, 1, squeeze=False)
    assert ei.plot_white(axs=axs) is fig
    assert len(axs[0, 0].lines) > 0
    close(fig)

    ei.optimize_gp_hyperparameters(npop=10, niter=5)
    fig, axs = subplots(1, 1, squeeze=False)
    ei.plot_white_gp_predictions(axs=axs)
    assert len(axs[0, 0].lines) == 2
    close(fig)
