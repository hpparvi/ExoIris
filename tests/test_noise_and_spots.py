"""Tests for the free GP noise model setup and the spot model edge cases."""
import numpy as np
import pytest

from exoiris import ExoIris, TSData, load_model
from exoiris.spotmodel import spot_model


def _make_data() -> TSData:
    rng = np.random.default_rng(0)
    time = np.linspace(-0.1, 0.1, 40)
    wavelength = np.linspace(1.0, 2.0, 8)
    fluxes = 1.0 + 1e-4 * rng.normal(size=(wavelength.size, time.size))
    return TSData(time, wavelength, fluxes, np.full_like(fluxes, 1e-4), name='a')


def _make_analysis(noise_model: str) -> ExoIris:
    ei = ExoIris('test', ldmodel='power-2', data=_make_data(), nk=4, noise_model=noise_model)
    ei.set_prior('tc_00', 'NP', 0.0, 0.01)
    ei.set_prior('p', 'NP', 3.0, 1e-5)
    ei.set_prior('rho', 'UP', 1.0, 3.0)
    ei.set_radius_ratio_prior('UP', 0.05, 0.15)
    return ei


def _n_gp_parameters(ei: ExoIris) -> int:
    return sum(n.startswith('gp_log_') for n in ei.ps.names)


def _assert_gp_parameters_ok(ei: ExoIris):
    tsa = ei._tsa
    assert _n_gp_parameters(ei) == 2
    assert [ei.ps.names[i] for i in range(len(ei.ps))[tsa._sl_gp]] == ['gp_log_sigma', 'gp_log_rho']
    pvp = tsa.ps.sample_from_prior(4)
    assert pvp.shape[1] == len(ei.ps)
    assert np.isfinite(tsa.lnposterior(pvp)).all()


def test_free_gp_in_constructor():
    _assert_gp_parameters_ok(_make_analysis('free_gp'))


def test_free_gp_hyperparameters_are_log10():
    ei = _make_analysis('free_gp')
    tsa = ei._tsa
    pv = tsa.ps.sample_from_prior(1)[0]
    pv[tsa._sl_gp] = [-4.0, -1.0]
    lnl_free = tsa.lnlikelihood(pv)
    assert tsa._gp[0].kernel.sigma == pytest.approx(1e-4)
    assert tsa._gp[0].kernel.rho == pytest.approx(0.1)

    fixed = _make_analysis('fixed_gp')
    fixed.set_gp_hyperparameters(1e-4, 0.1)
    assert lnl_free == pytest.approx(fixed._tsa.lnlikelihood(pv[:len(fixed.ps)]))


@pytest.mark.parametrize('initial', ['white_profiled', 'fixed_gp', 'free_gp'])
def test_free_gp_set_afterwards(initial):
    ei = _make_analysis(initial)
    ei.set_noise_model('free_gp')
    _assert_gp_parameters_ok(ei)


def test_free_gp_survives_knot_changes():
    ei = _make_analysis('free_gp')
    ei.set_radius_ratio_knots(np.linspace(1.0, 2.0, 6))
    _assert_gp_parameters_ok(ei)
    ei.set_limb_darkening_knots(np.linspace(1.0, 2.0, 4))
    _assert_gp_parameters_ok(ei)


def test_free_gp_save_and_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ei = _make_analysis('free_gp')
    ei.save(overwrite=True)
    _assert_gp_parameters_ok(load_model('test.fits'))


def test_spot_model_edge_cases():
    x = np.linspace(-1.0, 1.0, 21)
    ref = spot_model(x, 0.0, 0.01, 0.5, 2.0)
    assert ref.max() == pytest.approx(0.01)
    np.testing.assert_allclose(np.interp(0.25, x, ref), 0.005, rtol=0.05)  # Half maximum at fwhm/2

    np.testing.assert_array_equal(spot_model(x, 0.0, 0.01, 0.0, 2.0), np.zeros(x.size))
    for fwhm, shape in ((0.5, 0.0), (0.0, 0.0), (-0.5, 2.0), (0.5, -1.0), (np.nan, 2.0), (0.5, np.nan)):
        assert np.isnan(spot_model(x, 0.0, 0.01, fwhm, shape)).all()

    # The limit of a narrowing spot is the zero-width result.
    assert spot_model(x[x != 0.0], 0.0, 0.01, 1e-6, 2.0).max() == 0.0
