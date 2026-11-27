"""Tests for the broadband data class and the white light curve creation."""
import numpy as np
import pytest

from exoiris import BBData, ExoIris, TSData
from exoiris.wlpf import WhiteLPF

NWL = 7


def _make_bbdata(npt: int = 80, **kwargs) -> BBData:
    rng = np.random.default_rng(1)
    time = np.linspace(-0.1, 0.1, npt)
    flux = 1.0 + 0.01 * time + 1e-4 * rng.normal(size=npt)
    fwl = np.linspace(0.6, 0.9, 50)
    ftr = np.exp(-0.5 * ((fwl - 0.75) / 0.05) ** 2)
    return BBData(time=time, wavelength=fwl, transmission=ftr, n_wl=NWL, flux=flux,
                  errors=np.full(npt, 1e-4), name='bb', noise_group=1, epoch_group=1,
                  offset_group=1, n_baseline=2, **kwargs)


def _make_tsdata(npt: int = 80) -> TSData:
    rng = np.random.default_rng(2)
    time = np.linspace(-0.1, 0.1, npt)
    wavelength = np.linspace(1.0, 2.0, 12)
    fluxes = 1.0 + 1e-4 * rng.normal(size=(wavelength.size, npt))
    return TSData(time, wavelength, fluxes, np.full_like(fluxes, 1e-4), name='ts')


def _assert_consistent(d: BBData):
    """The 1D arrays used by the likelihood must match the 2D arrays."""
    assert isinstance(d, BBData)
    assert d.fluxes.shape == d.errors.shape == d.mask.shape == (NWL, d.time.size)
    assert d._bb_flux.shape == d._bb_errors.shape == d._bb_mask.shape == (d.time.size,)
    assert d.covs.shape[0] == d.transit_mask.size == d.time.size
    assert (d.mask == d.mask[0]).all()
    np.testing.assert_array_equal(d.fluxes, np.tile(d._bb_flux, (NWL, 1)))
    np.testing.assert_array_equal(d.errors, np.tile(d._bb_errors, (NWL, 1)))
    assert d._weights.size == NWL


@pytest.mark.parametrize('inplace', [True, False])
def test_crop_time(inplace):
    d = _make_bbdata()
    c = d.crop_time(-0.05, 0.05, inplace=inplace)
    assert (c is d) == inplace
    _assert_consistent(c)
    assert c.time.size < 80 and c.time.min() > -0.05 and c.time.max() < 0.05
    assert (c.noise_group, c.epoch_group, c.offset_group, c.n_baseline) == (1, 1, 1, 2)


@pytest.mark.parametrize('inplace', [True, False])
def test_remove_fully_masked(inplace):
    mask = np.ones(80, bool)
    mask[10:20] = False
    d = _make_bbdata(mask=mask)
    c = d.remove_fully_masked(inplace=inplace)
    assert (c is d) == inplace
    _assert_consistent(c)
    assert c.time.size == 70 and c.mask.all()


def test_bin_time():
    d = _make_bbdata()
    b = d.bin_time(nb=20)
    _assert_consistent(b)
    assert b.time.size == 20
    # The binning must match the binning of an ordinary time series.
    ref = TSData(d.time, d.wavelength[:2], d.fluxes[:2], d.errors[:2], name='ref').bin_time(nb=20)
    np.testing.assert_allclose(b._bb_flux, ref.fluxes[0], rtol=1e-12)
    np.testing.assert_allclose(b._bb_errors, ref.errors[0], rtol=1e-12)
    assert 0.4e-4 < b._bb_errors.mean() < 0.6e-4
    np.testing.assert_array_equal(b._weights, d._weights)
    _assert_consistent(d.bin(time_nb=20))


def test_partition_time():
    g = _make_bbdata().partition_time(((-0.1, -0.02), (0.02, 0.1)))
    assert g.size == 2
    for d in g.data:
        _assert_consistent(d)


def test_normalization_reaches_the_likelihood_arrays():
    d = _make_bbdata()
    assert abs(np.polyfit(d.time, d._bb_flux, 1)[0]) > 5e-3
    d.normalize_to_poly(1)
    _assert_consistent(d)
    assert abs(np.polyfit(d.time, d._bb_flux, 1)[0]) < 1e-3


def test_wavelength_methods_raise():
    d = _make_bbdata()
    for call in (lambda: d.bin_wavelength(nb=3), lambda: d.crop_wavelength(0.7, 0.8),
                 lambda: d.mask_outliers(), lambda: d.bin(wave_nb=3, time_nb=10)):
        with pytest.raises(NotImplementedError):
            call()


def test_save_load_roundtrip_with_mask(tmp_path):
    mask = np.ones(80, bool)
    mask[5] = False
    d = _make_bbdata(mask=mask).crop_time(-0.09, 0.09)
    d.save(tmp_path / 'bb.fits')
    r = BBData.load(tmp_path / 'bb.fits')
    _assert_consistent(r)
    np.testing.assert_array_equal(r._bb_mask, d._bb_mask)
    np.testing.assert_array_equal(r._bb_flux, d._bb_flux)


def test_white_light_curve_with_fully_masked_exposure():
    d = _make_tsdata()
    ref = d.create_white_light_curve()
    d.fluxes[:, 7] = np.nan
    d.fluxes[3, 9] = np.nan
    d._update_data_mask()
    wlc = d.create_white_light_curve()
    assert np.isnan(wlc[7])
    keep = np.ones(80, bool)
    keep[[7, 9]] = False
    np.testing.assert_allclose(wlc[keep], ref[keep], rtol=1e-12)
    assert np.isfinite(wlc[9])


def _make_analysis(data) -> ExoIris:
    ei = ExoIris('test', ldmodel='power-2', data=data, nk=4)
    ei.set_prior('tc_00', 'NP', 0.0, 0.01)
    ei.set_prior('tc_01', 'NP', 0.0, 0.01)
    ei.set_prior('p', 'NP', 3.0, 1e-5)
    ei.set_prior('rho', 'UP', 1.0, 3.0)
    ei.set_radius_ratio_prior('UP', 0.05, 0.15)
    return ei


def test_white_lpf_errors_and_masked_exposures():
    ts = _make_tsdata()
    ts.fluxes[:, 7] = np.nan
    ts._update_data_mask()
    mask = np.ones(80, bool)
    mask[3] = False
    bb = _make_bbdata(mask=mask)
    wa = WhiteLPF(_make_analysis(ts + bb)._tsa)
    assert wa.times[0].size == 79 and wa.times[1].size == 79
    np.testing.assert_allclose(wa.std_errors[0], 1e-4 / np.sqrt(12), rtol=1e-12)
    # The broadband errors must not be scaled down by the number of passband samples.
    np.testing.assert_allclose(wa.std_errors[1], 1e-4, rtol=1e-12)
    np.testing.assert_array_equal(wa.fluxes[1], bb._bb_flux[mask])


def test_likelihood_after_wrangling():
    bb = _make_bbdata(npt=120).crop_time(-0.08, 0.08).bin_time(nb=40)
    bb.normalize_to_poly(1)
    ei = _make_analysis(_make_tsdata() + bb)
    pvp = ei._tsa.ps.sample_from_prior(6)
    assert np.isfinite(ei._tsa.lnposterior(pvp)).all()
