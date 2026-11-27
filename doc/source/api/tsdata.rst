.. _api.tsdata:

Data handling
=============
.. currentmodule:: exoiris.tsdata

The spectroscopic light curves are given to ExoIris as a `TSData` (Transmission Spectroscopy Data) or `TSDataGroup`
object. `TSData` is a utility class that provides methods for data cleanup, binning, and cropping, and uncertainty
estimation, while `TSDataGroup` is a container class that can hold multiple `TSData` objects and provides aggregate
properties and methods. Two `TSData` objects can be combined into a `TSDataGroup` with the ``+`` operator, and
further data sets can be added to the group the same way.

Broadband light curves are stored as `~exoiris.BBData` objects, a `TSData` subclass described in
:ref:`api.bbdata`.

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

    TSData.bin
    TSData.bin_wavelength
    TSData.bin_time
    TSData.crop_wavelength
    TSData.crop_time
    TSData.mask_outliers
    TSData.remove_fully_masked
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

Each `TSData` object carries two masks. The data mask, `TSData.mask`, is a 2D boolean array that marks the valid
flux values, and is updated by the methods that mask or remove data. The transit mask, `TSData.transit_mask`, is a
1D boolean array that marks the out-of-transit exposures, and is created by `TSData.mask_transit` from the
ephemeris. Wavelengths and exposures without any valid data can be dropped with `TSData.remove_fully_masked`.

.. autosummary::
    :toctree: api/

    TSData.mask_transit
    TSData.ephemeris
    TSDataGroup.mask_transit

White light curve
-----------------

.. autosummary::
    :toctree: api/

    TSData.create_white_light_curve

Data set properties
-------------------

.. autosummary::
    :toctree: api/

    TSData.bbox_wl
    TSData.bbox_tm

`TSDataGroup` gives access to the data and the grouping of all its data sets as lists with one element per data
set.

.. autosummary::
    :toctree: api/

    TSDataGroup.size
    TSDataGroup.names
    TSDataGroup.times
    TSDataGroup.wavelengths
    TSDataGroup.fluxes
    TSDataGroup.errors
    TSDataGroup.noise_groups
    TSDataGroup.n_noise_groups
    TSDataGroup.epoch_groups
    TSDataGroup.offset_groups
    TSDataGroup.n_baselines

I/O
---

A `TSData` or `TSDataGroup` object can be saved to a FITS file with its ``save`` method and read back with
``load``. The file stores the fluxes, uncertainties, masks, baseline covariates, and group assignments, so data
that has been cleaned and binned once can be reused without repeating the preparation. The ``export_fits`` and
``import_fits`` methods do the same for an `~astropy.io.fits.HDUList`, and are what ExoIris uses to embed the data
in a model file.

.. autosummary::
    :toctree: api/

    TSData.save
    TSData.load
    TSDataGroup.save
    TSDataGroup.load
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
    TSData.plot_baseline
    TSData.plot_mean_error
    TSDataGroup.plot
    TSDataGroup.plot_white
