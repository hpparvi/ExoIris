import numpy as np
import pytest
from numpy import nan, isnan, isfinite, allclose, isclose

from exoiris.binning import Binning, CompoundBinning
from exoiris.bin1d import bin1d
from exoiris.bin2d import bin2d


def test_binning_nb():
    binning = Binning(0.5, 2.5, nb=4)
    assert binning.xmin == 0.5
    assert binning.xmax == 2.5
    assert binning.bw == 0.5
    assert binning.nb == 4
    assert binning.bins.shape == (4,2)


def test_binning_bw():
    binning = Binning(0.5, 2.5, bw=0.5)
    assert binning.xmin == 0.5
    assert binning.xmax == 2.5
    assert binning.bw == 0.5
    assert binning.nb == 4
    assert binning.bins.shape == (4,2)


def test_binning_r():
    binning = Binning(0.5, 2.5, r=10.0)
    assert binning.xmin == 0.5
    assert binning.xmax == 2.5
    assert binning.r == 10.0
    assert binning.bins.shape == (17,2)


def test_add_normal_binning():
    binning_1 = Binning(xmin=0.5, xmax=1.0, nb=5)
    binning_2 = Binning(xmin=1.0, xmax=2.0, nb=5)
    result = binning_1 + binning_2
    assert isinstance(result, CompoundBinning)


def test_add_compound_binning():
    binning = Binning(xmin=0.5, xmax=1.0, nb=5)
    compound = CompoundBinning([Binning(xmin=1.0, xmax=2.0, nb=5)])
    result = binning + compound
    assert isinstance(result, CompoundBinning)


def test_add_incompatible_type():
    binning = Binning(xmin=1.0, xmax=2.0, nb=5)
    with pytest.raises(TypeError):
        result = binning + "invalid type"


# =============================================================================
# bin1d tests
# =============================================================================

def test_bin1d_constant_values():
    """Binning constant values should return the same constant."""
    n_wl, n_tm = 4, 10
    wl_l = np.array([1.0, 1.5, 2.0, 2.5])
    wl_r = np.array([1.5, 2.0, 2.5, 3.0])
    wl_bins = np.array([[1.0, 2.0], [2.0, 3.0]])  # 2 bins, 2 channels each

    constant = 5.0
    v = np.full((n_wl, n_tm), constant)
    e = np.full((n_wl, n_tm), 0.1)

    bv, be = bin1d(v, e, wl_l, wl_r, wl_bins)

    assert bv.shape == (2, n_tm)
    assert allclose(bv, constant)


def test_bin1d_error_propagation():
    """Test error propagation with equal weights.

    With 2 channels per bin and equal weights (0.5 each),
    SE = sqrt(2 × 0.5² × 0.1²) = sqrt(2 × 0.25 × 0.01) = sqrt(0.005) ≈ 0.0707
    """
    n_wl, n_tm = 4, 10
    wl_l = np.array([1.0, 1.5, 2.0, 2.5])
    wl_r = np.array([1.5, 2.0, 2.5, 3.0])
    wl_bins = np.array([[1.0, 2.0], [2.0, 3.0]])

    v = np.ones((n_wl, n_tm))
    e = np.full((n_wl, n_tm), 0.1)

    bv, be = bin1d(v, e, wl_l, wl_r, wl_bins)

    # With normalized weights summing to 1, SE = sqrt(Σ w² σ²)
    # Two equal weights of 0.5 each: SE = sqrt(2 × 0.25 × 0.01) ≈ 0.0707
    expected_error = np.sqrt(2 * 0.5**2 * 0.1**2)
    assert allclose(be, expected_error, rtol=1e-10)


def test_bin1d_error_estimation():
    """Test error estimation from data scatter."""
    n_wl, n_tm = 4, 10
    wl_l = np.array([1.0, 1.5, 2.0, 2.5])
    wl_r = np.array([1.5, 2.0, 2.5, 3.0])
    wl_bins = np.array([[1.0, 2.0], [2.0, 3.0]])

    # Create data with known scatter
    np.random.seed(42)
    base_value = 1.0
    scatter = 0.1
    v = base_value + np.random.normal(0, scatter, (n_wl, n_tm))
    e = np.full((n_wl, n_tm), 0.001)  # Small input errors

    bv, be = bin1d(v, e, wl_l, wl_r, wl_bins, estimate_errors=True)

    # Estimated errors should reflect data scatter, not the small input errors
    # They should be on the order of scatter / sqrt(n_points)
    assert np.all(be > 0.001)  # Should be larger than input errors
    assert np.all(isfinite(be))


def test_bin1d_nan_handling():
    """Test that NaN values are properly excluded from binning."""
    n_wl, n_tm = 4, 10
    wl_l = np.array([1.0, 1.5, 2.0, 2.5])
    wl_r = np.array([1.5, 2.0, 2.5, 3.0])
    wl_bins = np.array([[1.0, 2.0], [2.0, 3.0]])

    v = np.ones((n_wl, n_tm))
    e = np.full((n_wl, n_tm), 0.1)

    # Set some values to NaN
    v[0, 0] = nan
    v[1, 5] = nan

    bv, be = bin1d(v, e, wl_l, wl_r, wl_bins)

    # Results should still be finite (NaNs excluded)
    assert np.all(isfinite(bv))
    assert np.all(isfinite(be))


# =============================================================================
# bin2d tests
# =============================================================================

def test_bin2d_constant_values():
    """Binning constant values should return the same constant with correct shape."""
    n_wl, n_tm = 4, 10
    wl_l = np.array([1.0, 1.5, 2.0, 2.5])
    wl_r = np.array([1.5, 2.0, 2.5, 3.0])
    tm_l = np.arange(0.0, 10.0, 1.0)
    tm_r = np.arange(1.0, 11.0, 1.0)
    wl_bins = np.array([[1.0, 2.0], [2.0, 3.0]])  # 2 wl bins
    tm_bins = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0], [6.0, 8.0], [8.0, 10.0]])  # 5 tm bins

    constant = 5.0
    v = np.full((n_wl, n_tm), constant)
    e = np.full((n_wl, n_tm), 0.1)

    bv, be, bn = bin2d(v, e, wl_l, wl_r, tm_l, tm_r, wl_bins, tm_bins)

    assert bv.shape == (2, 5)
    assert be.shape == (2, 5)
    assert bn.shape == (2, 5)
    assert allclose(bv, constant)


def test_bin2d_error_propagation():
    """Test error propagation with equal weights.

    With 4 points per bin (2 wl × 2 tm), each with equal weight:
    - Total weight = 4 × w where w is the overlap weight
    - After normalization, each weight = 0.25
    - SE = sqrt(Σ w² σ²) / Σw = sqrt(4 × w² × σ²) / (4w) = sqrt(4 × 0.01) × w / (4w) = 0.1 / 2 = 0.05
    """
    n_wl, n_tm = 4, 4
    wl_l = np.array([1.0, 1.5, 2.0, 2.5])
    wl_r = np.array([1.5, 2.0, 2.5, 3.0])
    tm_l = np.array([0.0, 1.0, 2.0, 3.0])
    tm_r = np.array([1.0, 2.0, 3.0, 4.0])
    wl_bins = np.array([[1.0, 2.0], [2.0, 3.0]])  # 2 wl bins, 2 channels each
    tm_bins = np.array([[0.0, 2.0], [2.0, 4.0]])  # 2 tm bins, 2 exposures each

    v = np.ones((n_wl, n_tm))
    e = np.full((n_wl, n_tm), 0.1)

    bv, be, bn = bin2d(v, e, wl_l, wl_r, tm_l, tm_r, wl_bins, tm_bins)

    # With 4 equal-weight points per bin, SE = sqrt(4 × w² × σ²) / (4w) = σ / 2 = 0.05
    expected_error = 0.05
    assert allclose(be, expected_error, rtol=1e-10)


def test_bin2d_error_estimation():
    """Test error estimation from data scatter."""
    n_wl, n_tm = 4, 10
    wl_l = np.array([1.0, 1.5, 2.0, 2.5])
    wl_r = np.array([1.5, 2.0, 2.5, 3.0])
    tm_l = np.arange(0.0, 10.0, 1.0)
    tm_r = np.arange(1.0, 11.0, 1.0)
    wl_bins = np.array([[1.0, 2.0], [2.0, 3.0]])
    tm_bins = np.array([[0.0, 5.0], [5.0, 10.0]])

    # Create data with known scatter
    np.random.seed(42)
    base_value = 1.0
    scatter = 0.1
    v = base_value + np.random.normal(0, scatter, (n_wl, n_tm))
    e = np.full((n_wl, n_tm), 0.001)  # Small input errors

    bv, be, bn = bin2d(v, e, wl_l, wl_r, tm_l, tm_r, wl_bins, tm_bins, estimate_errors=True)

    # Estimated errors should reflect data scatter, not small input errors
    assert np.all(be > 0.001)
    assert np.all(isfinite(be))


def test_bin2d_nan_handling():
    """Test that NaN values are properly excluded from binning."""
    n_wl, n_tm = 4, 10
    wl_l = np.array([1.0, 1.5, 2.0, 2.5])
    wl_r = np.array([1.5, 2.0, 2.5, 3.0])
    tm_l = np.arange(0.0, 10.0, 1.0)
    tm_r = np.arange(1.0, 11.0, 1.0)
    wl_bins = np.array([[1.0, 2.0], [2.0, 3.0]])
    tm_bins = np.array([[0.0, 5.0], [5.0, 10.0]])

    v = np.ones((n_wl, n_tm))
    e = np.full((n_wl, n_tm), 0.1)

    # Set some values to NaN
    v[0, 0] = nan
    v[1, 5] = nan
    v[2, 8] = nan

    bv, be, bn = bin2d(v, e, wl_l, wl_r, tm_l, tm_r, wl_bins, tm_bins)

    # Results should still be finite (NaNs excluded)
    assert np.all(isfinite(bv))
    assert np.all(isfinite(be))


def test_bin2d_point_count():
    """Test that bn correctly counts finite points per bin."""
    n_wl, n_tm = 4, 4
    wl_l = np.array([1.0, 1.5, 2.0, 2.5])
    wl_r = np.array([1.5, 2.0, 2.5, 3.0])
    tm_l = np.array([0.0, 1.0, 2.0, 3.0])
    tm_r = np.array([1.0, 2.0, 3.0, 4.0])
    wl_bins = np.array([[1.0, 2.0], [2.0, 3.0]])  # 2 wl bins
    tm_bins = np.array([[0.0, 2.0], [2.0, 4.0]])  # 2 tm bins

    v = np.ones((n_wl, n_tm))
    e = np.full((n_wl, n_tm), 0.1)

    bv, be, bn = bin2d(v, e, wl_l, wl_r, tm_l, tm_r, wl_bins, tm_bins)

    # Each bin should have 4 points (2 wl × 2 tm)
    assert np.all(bn == 4)

    # Now add some NaNs and verify count decreases
    v[0, 0] = nan  # Affects first wl bin, first tm bin
    bv, be, bn = bin2d(v, e, wl_l, wl_r, tm_l, tm_r, wl_bins, tm_bins)

    assert bn[0, 0] == 3  # One NaN in this bin
    assert bn[0, 1] == 4  # Unaffected
    assert bn[1, 0] == 4  # Unaffected
    assert bn[1, 1] == 4  # Unaffected


# =============================================================================
# Consistency test
# =============================================================================

def test_bin1d_bin2d_consistency():
    """When bin2d bins only in wavelength (single time bin), results should match bin1d."""
    n_wl, n_tm = 4, 10
    wl_l = np.array([1.0, 1.5, 2.0, 2.5])
    wl_r = np.array([1.5, 2.0, 2.5, 3.0])
    tm_l = np.arange(0.0, 10.0, 1.0)
    tm_r = np.arange(1.0, 11.0, 1.0)
    wl_bins = np.array([[1.0, 2.0], [2.0, 3.0]])

    # Random data
    np.random.seed(42)
    v = np.random.normal(1.0, 0.1, (n_wl, n_tm))
    e = np.full((n_wl, n_tm), 0.1)

    # bin1d result
    bv_1d, be_1d = bin1d(v, e, wl_l, wl_r, wl_bins)

    # bin2d with single time bin covering all times
    tm_bins_single = np.array([[tm_l[i], tm_r[i]] for i in range(n_tm)])
    bv_2d, be_2d, bn_2d = bin2d(v, e, wl_l, wl_r, tm_l, tm_r, wl_bins, tm_bins_single)

    # Results should match
    assert allclose(bv_1d, bv_2d)
    assert allclose(be_1d, be_2d)


