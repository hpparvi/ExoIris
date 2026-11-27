"""Tests for the save file round trip, the LDTk parameter clipping, and the broadband GP likelihood."""
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits as pf

from exoiris import ExoIris, TSData, load_model
from exoiris.bbdata import BBData
from exoiris.ldtkld import LDTkLD


def _make_data() -> TSData:
    rng = np.random.default_rng(0)
    time = np.linspace(-0.1, 0.1, 40)
    wavelength = np.linspace(1.0, 2.0, 8)
    fluxes = 1.0 + 1e-4 * rng.normal(size=(wavelength.size, time.size))
    return TSData(time, wavelength, fluxes, np.full_like(fluxes, 1e-4), name='a')


def _make_analysis(data, name: str = 'test', noise_model: str = 'white_profiled') -> ExoIris:
    ei = ExoIris(name, ldmodel='power-2', data=data, nk=4, noise_model=noise_model)
    ei.set_prior('tc_00', 'NP', 0.0, 0.01)
    ei.set_prior('p', 'NP', 3.0, 1e-5)
    ei.set_prior('rho', 'UP', 1.0, 3.0)
    ei.set_radius_ratio_prior('UP', 0.05, 0.15)
    return ei


def _with_results(ei: ExoIris, npop: int = 6, nsteps: int = 5) -> ExoIris:
    """Give the analysis a DE population and MCMC chains without running the optimizer or sampler."""
    rng = np.random.default_rng(1)
    ei._tsa._de_population = rng.normal(size=(npop, ei.ndim))
    ei._tsa._de_imin = 3
    ei._tsa._mc_chains = rng.normal(size=(npop, nsteps, ei.ndim))
    return ei


def _hdu_names(fname) -> list[str]:
    with pf.open(fname) as hdul:
        return [h.name for h in hdul]


def test_save_and_load_preserve_chain_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ei = _with_results(_make_analysis(_make_data()))
    ei.save(overwrite=True)
    a = load_model('test.fits')
    assert np.allclose(a.mcmc_chains, ei.mcmc_chains)
    assert np.allclose(a.optimizer_population, ei.optimizer_population)
    assert a._tsa._de_imin == 3


def test_saved_chain_is_stored_step_by_step(tmp_path, monkeypatch):
    """The MCMC table keeps the layout of emcee's flat chain used by the earlier versions."""
    monkeypatch.chdir(tmp_path)
    ei = _with_results(_make_analysis(_make_data()))
    ei.save(overwrite=True)
    with pf.open('test.fits') as hdul:
        first = np.array(list(hdul['MCMC'].data[1]))
    assert np.allclose(first, ei.mcmc_chains[1, 0])


def test_load_and_save_keeps_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ei = _with_results(_make_analysis(_make_data()))
    ei.save(overwrite=True)
    a = load_model('test.fits')
    assert a._tsa.de is None and a._tsa.sampler is None
    a.save(overwrite=True)
    names = _hdu_names('test.fits')
    assert 'DE' in names and 'MCMC' in names
    b = load_model('test.fits')
    assert np.allclose(b.mcmc_chains, ei.mcmc_chains)
    assert np.allclose(b.optimizer_population, ei.optimizer_population)


def test_save_follows_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ei = _make_analysis(_make_data())
    (tmp_path / 'test.fits').write_text('existing')
    with pytest.raises(OSError):
        ei.save(overwrite=False)
    assert (tmp_path / 'test.fits').read_text() == 'existing'
    # The default is to overwrite.
    ei.save()
    assert 'K_KNOTS' in _hdu_names('test.fits')


def test_ldtk_clipping_does_not_modify_parameters():
    ei = _make_analysis(_make_data())
    tsa = ei._tsa
    # Stand in for an LDTkLD limb darkening model without creating one, which would need the
    # stellar spectrum library. _eval_ldc uses only the parameter limits of the model.
    ldm = LDTkLD.__new__(LDTkLD)
    ldm.sc = SimpleNamespace(client=SimpleNamespace(teffl=(4000.0, 6000.0), loggl=(4.0, 5.0), zl=(-1.0, 1.0)))
    tsa.ldmodel = ldm
    tsa._sl_ld = slice(0, 3)

    pvp = np.tile([9000.0, 9.0, 9.0, 1.0], (3, 1))
    pv0 = pvp.copy()
    ldp = tsa._eval_ldc(pvp)
    assert np.array_equal(pvp, pv0)
    assert np.allclose(ldp[:, 0, :], [6000.0, 5.0, 1.0])


def _make_bbdata(n_wl: int) -> BBData:
    rng = np.random.default_rng(0)
    time = np.linspace(-0.1, 0.1, 40)
    flux = 1.0 + 1e-4 * rng.normal(size=time.size)
    flux[5] = np.nan
    return BBData(time, np.linspace(0.5, 0.6, 50), np.ones(50), n_wl, flux, np.full(time.size, 1e-4), name='b')


@pytest.mark.parametrize('noise_model', ['white_profiled', 'white_marginalized', 'fixed_gp'])
def test_broadband_likelihood_is_independent_of_n_wl(noise_model):
    lnl = []
    for n_wl in (5, 20):
        ei = _make_analysis(_make_bbdata(n_wl), noise_model=noise_model)
        tsa = ei._tsa
        if noise_model == 'fixed_gp':
            ei.set_gp_hyperparameters(1e-4, 0.01)
            assert tsa._gp_flux[0].size == 39
        pv = tsa.ps.mean_pv.copy()
        pv[tsa._sl_rratios] = 0.1
        lnl.append(tsa.lnlikelihood(pv))
    assert np.isfinite(lnl).all()
    assert lnl[0] == pytest.approx(lnl[1], rel=1e-6)
