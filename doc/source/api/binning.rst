.. _api.binning

Binning data
============
.. currentmodule:: exoiris.binning

While ExoIris aims to be fast enough that binning of data is not usually necessary, we still may want to do it every now
or then. For this, ExoIris provides the `Binning` and `CompoundBinning` classes that can be used to define complex,
non-regular binning geometries. These can be used to bin spectroscopic light curve data sets stored as
`~exoiris.tsdata.TSData` objects using its `~exoiris.tsdata.TSData.bin_wavelength` method.

Binning
-------

The `Binning` class creates a homogeneous binning from `xmin` to `xmax` based on either a given number of bins (`nb`),
bin width (`bw`), or resolving power (`r`). Giving a number of bins divides the range into `nb` equally wide bins while
giving the bin width divides the range into bins that are as close to `bw` as possible. In both cases, the bin widths
will be constant. Giving the resolving power, the range will be divided into bins with widths following the x/dx relation.

.. autosummary::
    :toctree: api/

    Binning
    Binning.bins


CompoundBinning
---------------

The `CompoundBinning` class creates a heterogeneous binning by combining several `Binning` objects.

.. autosummary::
    :toctree: api/

    CompoundBinning
