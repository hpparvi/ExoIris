.. _api.tsdata:

Spectroscopic light curves
==========================
.. currentmodule:: easyts.tsdata

The spectroscopic light curves are given to `EasyTS` as a `TSData` (Transmission Spectroscopy Data) or `TSDataSet`
object. `TSData` is a utility class that provides methods for data cleanup, binning, and cropping, while `TSDataSet`
is a container class that can contain many `TSData` objects.

Constructor
-----------

.. autosummary::
    :toctree: api/

    TSData
    TSDataSet

Data wrangling
--------------

.. autosummary::
    :toctree: api/

    TSData.bin_wavelength
    TSData.crop_wavelength
    TSData.remove_outliers

Plotting
--------

.. autosummary::
    :toctree: api/

    TSData.plot
    TSDataSet.plot
