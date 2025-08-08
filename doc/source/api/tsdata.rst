.. _api.tsdata:

Data handling
=============
.. currentmodule:: exoiris.tsdata

The spectroscopic light curves are given to ExoIris as a `TSData` (Transmission Spectroscopy Data) or `TSDataGroup`
object. `TSData` is a utility class that provides methods for data cleanup, binning, and cropping, and uncertainty
estimation, while `TSDataGroup` is a container class that can hold multiple `TSData` objects and provides aggregate
properties and methods.

Main classes
------------

.. autosummary::
    :toctree: api/

    TSData
    TSDataGroup

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

Uncertainty estimation
----------------------

.. autosummary::
    :toctree: api/

    TSData.estimate_average_uncertainties

Masking
-------

.. autosummary::
    :toctree: api/

    TSData.mask_transit
    TSDataGroup.mask_transit

I/O
---

.. autosummary::
    :toctree: api/

    TSData.export_fits
    TSData.import_fits
    TSDataGroup.export_fits
    TSDataGroup.import_fits


Plotting
--------

.. autosummary::
    :toctree: api/

    TSData.plot
    TSData.plot_white
    TSDataGroup.plot
