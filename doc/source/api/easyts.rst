.. _api.easyts:

Transmission spectroscopy with EasyTS
=====================================
.. currentmodule:: easyts

The `EasyTS` class is the core class of EasyTS used for transmission spectroscopy. It serves as the primary tool for
first fitting the white light curve model and subsequently fitting and sampling the spectroscopic light curve model.

Constructor
-----------

The `EasyTS` class is initialized by providing a name for the analysis, selecting a limb darkening model, and passing the
spectroscopic light curves as either a `~easyts.tsdata.TSData` or `~easyts.tsdata.TSDataSet` object.
Additional parameters can also be supplied to the initializer to refine the analysis, although these can be set
later if preferred.

The transmission spectroscopy transit model used in EasyTS, `pytransit.TSModel`, is based on PyTransit's
RoadRunner transit model (`pytransit.RRModel`). The RoadRunner model is an advanced transit model that can
efficiently use any radially symmetric function to model stellar limb darkening, as described in
`Parviainen (2020) <https://ui.adsabs.harvard.edu/link_gateway/2020MNRAS.499.1633P/PUB_HTML>`_.

Since `TSModel` is based on `RRModel`, it offers the same flexibility for modeling stellar limb darkening.
The `ldmodel` argument can be one of the following:

- a string representing one of the built-in limb darkening models supported by `RRModel`, such as
  `power-2` or `quadratic`,
- an object that is a subclass of the `pytransit.models.ldmodel.LDModel` limb darkening model class,
- a tuple of two functions, with the first returning the limb darkening profile as a function of
  :math:`\mu` and the second returning its integral over the stellar disk, or
- a single function that returns the stellar limb darkening profile as a function of
  :math:`\mu` , in which case its integral is computed numerically.


.. autosummary::
    :toctree: api/

    EasyTS

Model saving and loading
------------------------

An `EasyTS` instance can be saved into a fits file using the `EasyTS.save` method. This saves the model setup,
optimiser state, and the MCMC sampler state, so that the model can be recreated using the `load_model` function.

.. autosummary::
    :toctree: api/

    EasyTS.save
    load_model

Model setup
-----------

.. autosummary::
    :toctree: api/

    EasyTS.set_radius_ratio_knots
    EasyTS.add_radius_ratio_knots
    EasyTS.set_ldtk_prior
    EasyTS.set_radius_ratio_prior
    EasyTS.plot_setup
    EasyTS.print_parameters


First steps
-----------

.. autosummary::
    :toctree: api/

    EasyTS.fit_white
    EasyTS.plot_white
    EasyTS.plot_residuals
    EasyTS.normalize_baseline
    EasyTS.plot_baseline

Fitting and sampling
--------------------

.. autosummary::
    :toctree: api/

    EasyTS.fit
    EasyTS.sample

Accessing results
-----------------

.. autosummary::
    :toctree: api/

    EasyTS.get_transmission_spectrum
    EasyTS.posterior_samples
    EasyTS.plot_transmission_spectrum
    EasyTS.plot_residuals
    EasyTS.plot_limb_darkening_parameters

Noise model
-----------

.. autosummary::
    :toctree: api/

    EasyTS.set_noise_model
    EasyTS.set_gp_kernel
    EasyTS.set_gp_hyperparameters
    EasyTS.optimize_gp_hyperparameters
    EasyTS.gp

Model properties
----------------

.. autosummary::
    :toctree: api/

    EasyTS.name
    EasyTS.ps
    EasyTS.data
    EasyTS.time
    EasyTS.wavelength
    EasyTS.fluxes
    EasyTS.errors
    EasyTS.k_knots
    EasyTS.ndim
    EasyTS.nk
    EasyTS.nldp
    EasyTS.npb
    EasyTS.ldmodel
    EasyTS.optimizer_population
    EasyTS.mcmc_chains


Utility methods
---------------

.. autosummary::
    :toctree: api/

    EasyTS.create_initial_population
    EasyTS.reset_sampler