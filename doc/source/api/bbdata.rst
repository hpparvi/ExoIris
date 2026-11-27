.. _api.bbdata:

Broadband photometry
====================

.. currentmodule:: exoiris

Broadband transit light curves can be modelled jointly with spectroscopic time series using the `BBData`
(BroadBand Data) class. A `BBData` object holds a single light curve observed through a filter with a known
transmission profile, and it can be combined with `~exoiris.tsdata.TSData` objects into a
`~exoiris.tsdata.TSDataGroup` like any other data set.

`BBData` is a subclass of `~exoiris.tsdata.TSData`. Internally, the filter passband is divided into ``n_wl``
wavelength elements, the transit model is evaluated for each of them, and the result is averaged over the passband
using the filter transmission as weights before it is compared with the observed light curve. The radius ratio and
limb darkening inside the passband come from the same wavelength-dependent models that describe the spectroscopic
data, so the photometry constrains the transmission spectrum at the wavelengths the filter covers.

Creating a broadband data set
-----------------------------

A `BBData` object is created from the light curve (times, fluxes, and uncertainties) and the filter transmission
profile. The filter wavelengths must be in the same units as the wavelengths of the spectroscopic data
(micrometres), and the transmission can have an arbitrary normalization. The part of the profile where the
transmission exceeds one per cent of its maximum defines the passband.

.. code-block:: python

    from exoiris import BBData, TSData, ExoIris

    spectroscopy = TSData(time, wavelength, fluxes, errors, name='nirspec')
    photometry = BBData(time=lc_time, wavelength=filter_wavelength, transmission=filter_transmission,
                        n_wl=10, flux=lc_flux, errors=lc_errors, name='i_band',
                        noise_group=1, epoch_group=1)

    ts = ExoIris('joint_analysis', ldmodel='power-2', data=spectroscopy + photometry)

The noise, epoch, and offset groups work as they do for `~exoiris.tsdata.TSData`: a light curve observed with a
different instrument should usually have its own noise group, and a light curve covering a different transit its
own epoch group.

The number of wavelength elements, ``n_wl``, sets how finely the passband is sampled. A handful of elements is
enough for a narrow filter, while a wide filter covering a region where the transmission spectrum or the limb
darkening changes quickly benefits from more. The model evaluation cost grows linearly with ``n_wl``.

.. autosummary::
    :toctree: api/

    BBData

.. note::

    Broadband data are supported by the white noise models (``white_profiled`` and ``white_marginalized``). The
    Gaussian process noise models do not average the model over the passband and should not be used with
    `BBData`.

Data wrangling
--------------

A broadband light curve can be cropped, binned, and partitioned in time, and the methods return `BBData` objects
that keep the filter passband. The methods that work along the wavelength axis (`~exoiris.tsdata.TSData.bin_wavelength`,
`~exoiris.tsdata.TSData.crop_wavelength`, and `~exoiris.tsdata.TSData.mask_outliers`) are not defined for broadband
data and raise a `NotImplementedError`, because the wavelength axis of a `BBData` object is only an internal
discretisation of the passband. Outliers can be masked with the ``mask`` argument when creating the object.

.. autosummary::
    :toctree: api/

    BBData.crop_time
    BBData.bin_time
    BBData.bin
    BBData.partition_time
    BBData.remove_fully_masked

White light curve
-----------------

.. autosummary::
    :toctree: api/

    BBData.create_white_light_curve

I/O
---

A `BBData` object can be saved to a FITS file together with its filter profile and read back with `BBData.load`.
Broadband data sets stored in a `~exoiris.tsdata.TSDataGroup` file or in an ExoIris model file are restored as
`BBData` objects automatically.

.. autosummary::
    :toctree: api/

    BBData.save
    BBData.load
    BBData.export_fits
    BBData.import_fits
