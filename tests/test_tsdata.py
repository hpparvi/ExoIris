import numpy as np
from numpy import allclose, array_equal

from exoiris.tsdata import TSData


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


# -- remove_fully_masked ------------------------------------------------------

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
