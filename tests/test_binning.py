import pytest
from exoiris.binning import Binning, CompoundBinning


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


