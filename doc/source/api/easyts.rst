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

An `EasyTS` instance can be saved to a FITS file using the `EasyTS.save` method. This stores the model setup,
optimiser state, and MCMC sampler state, allowing the model to be fully recreated later using the `load_model`
function.


.. autosummary::
    :toctree: api/

    EasyTS.save
    load_model

Light curve model setup
-----------------------

All parts of an EasyTS analysis can be modified after initialisation. This enables iterative analysis approaches
where a saved low-resolution analysis can be loaded as a new analysis using the `load_model` function. Parameters
like the radius ratio and limb darkening knots can be adjusted to increase the resolution of the estimated
transmission spectrum, and even the observational data can be changed to improve the data resolution.


.. autosummary::
    :toctree: api/

    EasyTS.set_data
    EasyTS.set_radius_ratio_knots
    EasyTS.add_radius_ratio_knots
    EasyTS.set_ldtk_prior
    EasyTS.set_radius_ratio_prior
    EasyTS.plot_setup
    EasyTS.print_parameters


Noise model setup
-----------------

The noise in the spectroscopic light curves can be modeled as either white noise or time-correlated noise
(using a Gaussian process, GP). The noise model is chosen with the `EasyTS.noise_model` method, and can be
set to either "white" or "fixed_gp." Selecting "fixed_gp" models the noise as a time-correlated Gaussian process
using the `celerite2` package. The corresponding `celerite2.GaussianProcess` object can be accessed directly
via the `EasyTS.gp` attribute.

.. autosummary::
    :toctree: api/

    EasyTS.set_noise_model
    EasyTS.set_gp_kernel
    EasyTS.set_gp_hyperparameters
    EasyTS.optimize_gp_hyperparameters
    EasyTS.gp


First steps
-----------

The first steps of a transmission spectroscopy analysis include fitting a white light curve and normalising the
spectroscopic light curves. `EasyTS` offers utility methods for both of these tasks, as well as for visualising the
results.

.. autosummary::
    :toctree: api/

    EasyTS.fit_white
    EasyTS.normalize_baseline
    EasyTS.plot_white
    EasyTS.plot_residuals
    EasyTS.plot_baseline


Fitting and sampling
--------------------

The main tasks of `EasyTS` are to fit a spectroscopic light curve model to the observations and then sample its parameter
posterior to obtain a posterior transmission spectrum estimate. The fitting is carried out using a Differential Evolution
global optimiser and the sampling with the `emcee` affine invariant ensemble sampler.

The DE optimiser works by clumping a population of `npop` parameter vectors near the global posterior mode over `niter`
iterations. The optimisation can be stopped when the ptp width of the population's log posterior distribution has
decreased below a desired threshold (by default 2), after which the MCMC sampling phase can be started.

Both of these methods can be called iteratively, in which case they start from the state they finished in the previous
call. At the first `EasyTS.sample` call, the sampler will start from the current DE optimiser population. Note that
loading a previous analysis with `load_model` also loads the sampler state, so calling `EasyTS.sample` after loading
a model continues the sampler from the saved sampler state. `EasyTS.reset_sampler` should be called after loading a
saved model if you want to change the setup, optimise the new setup, and then sample the posterior.

.. autosummary::
    :toctree: api/

    EasyTS.fit
    EasyTS.sample
    EasyTS.reset_sampler


Accessing the results
---------------------

The main results from an EasyTS analysis are the model parameter posterior samples and the transmission spectrum.
The transmission spectrum, represented as the planet-to-star area ratio as a function of wavelength, can be
retrieved as a Pandas `~pandas.DataFrame` using the `EasyTS.transmission_spectrum` attribute. Similarly, the
model posterior samples can be accessed using the `EasyTS.posterior_samples` attribute, also in the form of a
Pandas `~pandas.DataFrame`.


.. autosummary::
    :toctree: api/

    EasyTS.transmission_spectrum
    EasyTS.posterior_samples
    EasyTS.plot_fit
    EasyTS.plot_transmission_spectrum
    EasyTS.plot_residuals
    EasyTS.plot_limb_darkening_parameters

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
