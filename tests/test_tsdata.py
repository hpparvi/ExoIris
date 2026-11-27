import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy import allclose, array_equal

from exoiris.binning import Binning
from exoiris.ephemeris import Ephemeris
from exoiris.tsdata import TSData, TSDataGroup


def _make_tsdata(mask=None, name='test', nwl=4, npt=6, rng=None):
    wavelength = np.linspace(1.0, 1.3, nwl)
    time = np.linspace(0.0, 0.5, npt)
    if rng is None:
        fluxes = np.ones((nwl, npt))
    else:
        fluxes = 1.0 + 0.001 * rng.standard_normal((nwl, npt))
    errors = np.full((nwl, npt), 0.01)
    return TSData(time=time, wavelength=wavelength, fluxes=fluxes, errors=errors,
                  name=name, mask=mask)


# -- remove_fully_masked (pre-existing behavior, now non-destructive) -------

def test_remove_fully_masked_shapes():
    """Fully-masked row and column are dropped; scattered masked cells are kept."""
    nwl, npt = 4, 6
    mask = np.ones((nwl, npt), dtype=bool)
    mask[1, :] = False
    mask[:, 2] = False
    mask[0, 3] = False
    mask[3, 4] = False

    d = _make_tsdata(mask=mask)
    r = d.remove_fully_masked(inplace=False)

    assert r.nwl == 3
    assert r.npt == 5
    assert r.fluxes.shape == (3, 5)
    assert r.errors.shape == (3, 5)
    assert r.mask.shape == (3, 5)


def test_remove_fully_masked_preserves_axes_and_edges():
    nwl, npt = 4, 6
    mask = np.ones((nwl, npt), dtype=bool)
    mask[1, :] = False
    mask[:, 2] = False

    d = _make_tsdata(mask=mask)
    wl0, tm0 = d.wavelength.copy(), d.time.copy()
    wl_l0, wl_r0 = d._wl_l_edges.copy(), d._wl_r_edges.copy()
    tm_l0, tm_r0 = d._tm_l_edges.copy(), d._tm_r_edges.copy()
    tmask0 = d.transit_mask.copy()
    covs0 = d.covs.copy()

    kw = np.array([True, False, True, True])
    kt = np.array([True, True, False, True, True, True])

    r = d.remove_fully_masked(inplace=False)

    assert allclose(r.wavelength, wl0[kw])
    assert allclose(r._wl_l_edges, wl_l0[kw])
    assert allclose(r._wl_r_edges, wl_r0[kw])
    assert allclose(r.time, tm0[kt])
    assert allclose(r._tm_l_edges, tm_l0[kt])
    assert allclose(r._tm_r_edges, tm_r0[kt])
    assert array_equal(r.transit_mask, tmask0[kt])
    assert allclose(r.covs, covs0[kt])
    assert array_equal(r.mask, mask[kw][:, kt])


def test_remove_fully_masked_noop_when_nothing_fully_masked():
    nwl, npt = 4, 6
    mask = np.ones((nwl, npt), dtype=bool)
    mask[0, 1] = False
    mask[2, 4] = False

    d = _make_tsdata(mask=mask)
    r = d.remove_fully_masked(inplace=False)

    assert r.nwl == nwl
    assert r.npt == npt
    assert array_equal(r.mask, mask)


def test_remove_fully_masked_is_idempotent():
    nwl, npt = 4, 6
    mask = np.ones((nwl, npt), dtype=bool)
    mask[1, :] = False
    mask[:, 2] = False
    mask[0, 3] = False
    mask[3, 4] = False

    d = _make_tsdata(mask=mask)
    r1 = d.remove_fully_masked(inplace=False)
    r2 = r1.remove_fully_masked(inplace=False)

    assert r2.nwl == r1.nwl
    assert r2.npt == r1.npt
    assert array_equal(r2.mask, r1.mask)


# -- Non-destructive semantics -----------------------------------------------

def test_pristine_original_is_self():
    d = _make_tsdata()
    assert d._original is d
    assert d.history == []


def test_crop_wavelength_leaves_original_untouched():
    d = _make_tsdata()
    wl0 = d.wavelength.copy()
    fx0 = d.fluxes.copy()

    r = d.crop_wavelength(1.05, 1.25)

    assert r is not d
    assert allclose(d.wavelength, wl0)
    assert allclose(d.fluxes, fx0)
    assert r.wavelength.size < d.wavelength.size
    assert r._original is d
    assert len(r.history) == 1
    assert r.history[0]['method'] == 'crop_wavelength'


def test_crop_time_leaves_original_untouched():
    d = _make_tsdata()
    tm0 = d.time.copy()

    r = d.crop_time(0.1, 0.4)

    assert r is not d
    assert allclose(d.time, tm0)
    assert r.time.size < d.time.size
    assert r._original is d


def test_mask_outliers_leaves_original_untouched():
    rng = np.random.default_rng(0)
    d = _make_tsdata(rng=rng)
    mask0 = d.mask.copy()
    fx0 = d.fluxes.copy()

    # Inject a single outlier
    d.fluxes[0, 3] = 100.0
    r = d.mask_outliers(sigma=3.0)

    assert r is not d
    # Original's mask is unchanged (our injection only changed fluxes, not mask)
    assert array_equal(d.mask, mask0)
    assert not r.mask[0, 3]
    assert r._original is d
    assert r.history[0] == {'method': 'mask_outliers', 'args': [], 'kwargs': {'sigma': 3.0}}
    # Restore for cleanliness (test isolation)
    d.fluxes[:] = fx0


def test_normalize_to_median_leaves_original_untouched():
    rng = np.random.default_rng(1)
    d = _make_tsdata(rng=rng)
    fx0 = d.fluxes.copy()
    r = d.normalize_to_median(slice(0, 3))
    assert r is not d
    assert allclose(d.fluxes, fx0)
    assert r.history[0]['method'] == 'normalize_to_median'


def test_original_does_not_chain():
    """Repeated modifications preserve a single pristine anchor, not a linked list."""
    d0 = _make_tsdata()
    d1 = d0.crop_wavelength(1.05, 1.3)
    d2 = d1.crop_time(0.05, 0.45)
    d3 = d2.mask_outliers(sigma=5.0)

    assert d1._original is d0
    assert d2._original is d0
    assert d3._original is d0
    assert len(d3.history) == 3
    assert [e['method'] for e in d3.history] == ['crop_wavelength', 'crop_time', 'mask_outliers']


def test_history_property_is_defensive_copy():
    d = _make_tsdata()
    r = d.mask_outliers(sigma=5.0)
    h = r.history
    h.append({'method': 'hacked', 'args': [], 'kwargs': {}})
    assert len(r.history) == 1


# -- inplace deprecation -----------------------------------------------------

def test_crop_wavelength_inplace_warns_and_returns_new():
    d = _make_tsdata()
    with pytest.warns(DeprecationWarning, match='inplace'):
        r = d.crop_wavelength(1.05, 1.3, inplace=True)
    assert r is not d


def test_crop_time_inplace_warns():
    d = _make_tsdata()
    with pytest.warns(DeprecationWarning, match='inplace'):
        r = d.crop_time(0.05, 0.45, inplace=True)
    assert r is not d


def test_remove_fully_masked_inplace_warns():
    nwl, npt = 4, 6
    mask = np.ones((nwl, npt), dtype=bool)
    mask[2, :] = False
    mask[:, 0] = False
    d = _make_tsdata(mask=mask)
    with pytest.warns(DeprecationWarning, match='inplace'):
        r = d.remove_fully_masked(inplace=True)
    assert r is not d
    # Original is untouched
    assert d.nwl == nwl
    assert d.npt == npt


def test_inplace_false_emits_no_warning():
    d = _make_tsdata()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        d.crop_wavelength(1.05, 1.3, inplace=False)


# -- replay ------------------------------------------------------------------

def _assert_same_state(a: TSData, b: TSData):
    assert allclose(a.time, b.time)
    assert allclose(a.wavelength, b.wavelength)
    assert array_equal(a.mask, b.mask)
    assert allclose(a.fluxes[a.mask], b.fluxes[b.mask])
    assert allclose(a.errors[a.mask], b.errors[b.mask])


def test_replay_reproduces_crop_chain():
    d0 = _make_tsdata()
    d1 = d0.crop_wavelength(1.05, 1.3).crop_time(0.05, 0.45)

    reconstructed = d0.replay(d1.history)
    _assert_same_state(reconstructed, d1)
    assert reconstructed._original is d0
    assert [e['method'] for e in reconstructed.history] == [e['method'] for e in d1.history]


def test_replay_with_mask_outliers():
    rng = np.random.default_rng(2)
    d0 = _make_tsdata(rng=rng)
    d0.fluxes[1, 4] = 50.0  # guaranteed outlier
    d1 = d0.mask_outliers(sigma=3.0)
    reconstructed = d0.replay(d1.history)
    _assert_same_state(reconstructed, d1)


def test_replay_empty_history_is_identity():
    d = _make_tsdata()
    r = d.replay([])
    assert r is d


def test_replay_partition_time_segment():
    d0 = _make_tsdata(npt=10)
    tlims = ((0.0, 0.2), (0.25, 0.5))
    group = d0.partition_time(tlims)
    seg1 = group[1]
    assert seg1._original is d0
    assert seg1.history[-1]['method'] == 'partition_time'

    reconstructed = d0.replay(seg1.history)
    _assert_same_state(reconstructed, seg1)


# -- Encoder round-trip ------------------------------------------------------

def test_encoder_roundtrip_through_json():
    import json
    from exoiris.tsdata import _encode_history, _decode_history

    history = [
        {'method': 'mask_outliers', 'args': [], 'kwargs': {'sigma': 4.5}},
        {'method': 'crop_time', 'args': [0.1, 0.4], 'kwargs': {}},
        {'method': 'normalize_to_median', 'args': [slice(0, 5, 2)], 'kwargs': {}},
        {'method': 'mask_transit', 'args': [],
         'kwargs': {'ephemeris': Ephemeris(0.0, 3.5, 0.1),
                    't0': None, 'p': None, 't14': None, 'elims': None}},
        {'method': 'bin_wavelength', 'args': [],
         'kwargs': {'binning': Binning(1.0, 2.0, nb=10), 'estimate_errors': False}},
    ]
    encoded = json.dumps(_encode_history(history))
    decoded = _decode_history(json.loads(encoded))

    assert decoded[0] == history[0]
    assert decoded[1] == history[1]
    assert decoded[2]['args'][0] == slice(0, 5, 2)
    eph = decoded[3]['kwargs']['ephemeris']
    assert (eph.zero_epoch, eph.period, eph.duration) == (0.0, 3.5, 0.1)
    b = decoded[4]['kwargs']['binning']
    assert (b.xmin, b.xmax, b.nb) == (1.0, 2.0, 10)


# -- FITS round-trip ---------------------------------------------------------

def test_fits_roundtrip_preserves_original_and_history(tmp_path: Path):
    d0 = _make_tsdata()
    d1 = d0.mask_outliers(sigma=5.0)
    d2 = d1.crop_time(0.05, 0.45)

    fname = tmp_path / 'roundtrip.fits'
    d2.save(fname)

    loaded = TSData.load(fname)
    assert loaded._original is not loaded
    assert loaded._original is not None
    assert loaded._original.time.size == d0.time.size
    assert loaded._original.wavelength.size == d0.wavelength.size
    assert [e['method'] for e in loaded.history] == ['mask_outliers', 'crop_time']
    assert loaded.history[0]['kwargs']['sigma'] == 5.0

    # Arrays match current state (not the original)
    _assert_same_state(loaded, d2)

    # Replay from the loaded original reproduces the current state
    _assert_same_state(loaded._original.replay(loaded.history), d2)


def test_fits_roundtrip_pristine_has_no_original(tmp_path: Path):
    d = _make_tsdata()
    fname = tmp_path / 'pristine.fits'
    d.save(fname)

    loaded = TSData.load(fname)
    assert loaded._original is loaded
    assert loaded.history == []


# -- TSDataGroup -------------------------------------------------------------

def test_tsdatagroup_mask_transit_replaces_items_nondestructively():
    d1 = _make_tsdata(name='a')
    d2 = _make_tsdata(name='b')
    group = TSDataGroup([d1, d2])

    orig_tmask_a = d1.transit_mask.copy()
    orig_tmask_b = d2.transit_mask.copy()

    group.mask_transit(0.25, 3.0, 0.05)

    # Pristine originals untouched
    assert array_equal(d1.transit_mask, orig_tmask_a)
    assert array_equal(d2.transit_mask, orig_tmask_b)

    # Group's entries are new objects that point back to the pristines
    assert group.data[0] is not d1
    assert group.data[1] is not d2
    assert group.data[0]._original is d1
    assert group.data[1]._original is d2
    assert group.data[0]._dataset is group
