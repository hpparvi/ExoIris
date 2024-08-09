.. _api.tsdata:

Spectroscopic light curves
==========================
.. currentmodule:: easyts

The spectroscopic light curves are given to `EasyTS` as a `TSData` (Transmission Spectroscopy Data)  object. `TSData`
is a utility class that provides methods for data cleanup, binning, cropping, and concatenation of multiple data sets.

Constructor
-----------

.. autosummary::
    :toctree: api/

    TSData

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