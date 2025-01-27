.. _api.tsdata:

Data handling
=============
.. currentmodule:: exoiris.tsdata

The spectroscopic light curves are given to ExoIris as a `TSData` (Transmission Spectroscopy Data) or `TSDataSet`
object. `TSData` is a utility class that provides methods for data cleanup, binning, and cropping, while `TSDataSet`
is a container class that can contain many `TSData` objects.

Main classes
------------

.. autosummary::
    :toctree: api/

    TSData
    TSDataSet

Data wrangling
--------------

.. autosummary::
    :toctree: api/

    TSData.bin_wavelength
    TSData.bin_time
    TSData.crop_wavelength
    TSData.crop_time
    TSData.remove_outliers
    TSData.normalize_to_median
    TSData.normalize_to_poly
    TSData.partition_time

Masking
-------

.. autosummary::
    :toctree: api/

    TSData.mask_transit

I/O
---

.. autosummary::
    :toctree: api/

    TSData.export_fits
    TSData.import_fits
    TSDataSet.export_fits
    TSDataSet.import_fits


Plotting
--------

.. autosummary::
    :toctree: api/

    TSData.plot
    TSData.plot_white
    TSDataSet.plot
