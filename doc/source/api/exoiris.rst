.. _api.exoiris:

Transmission spectroscopy
=========================
.. currentmodule:: exoiris

The `ExoIris` class is the core class of ExoIris used for transmission spectroscopy. It serves as the primary tool for
first fitting the white light curve model and subsequently fitting and sampling the spectroscopic light curve model.

Constructor
-----------

The `ExoIris` class is initialized by providing a name for the analysis, selecting a limb darkening model, and passing the
spectroscopic light curves as either a `~exoiris.tsdata.TSData` or `~exoiris.tsdata.TSDataSet` object.
Additional parameters can also be supplied to the initializer to refine the analysis, although these can be set
later if preferred.

The transmission spectroscopy transit model used in ExoIris, `pytransit.TSModel`, is based on PyTransit's
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

    ExoIris

Model saving and loading
------------------------

An `ExoIris` instance can be saved to a FITS file using the `ExoIris.save` method. This stores the model setup,
optimiser state, and MCMC sampler state, allowing the model to be fully recreated later using the `load_model`
function.


.. autosummary::
    :toctree: api/

    ExoIris.save
    load_model

Light curve model setup
-----------------------

All parts of an ExoIris analysis can be modified after initialisation. This enables iterative analysis approaches
where a saved low-resolution analysis can be loaded as a new analysis using the `load_model` function. Parameters
like the radius ratio and limb darkening knots can be adjusted to increase the resolution of the estimated
transmission spectrum, and even the observational data can be changed to improve the data resolution.


.. autosummary::
    :toctree: api/

    ExoIris.set_data
    ExoIris.set_radius_ratio_knots
    ExoIris.add_radius_ratio_knots
    ExoIris.set_ldtk_prior
    ExoIris.set_radius_ratio_prior
    ExoIris.plot_setup
    ExoIris.print_parameters


Noise model setup
-----------------

The noise in the spectroscopic light curves can be modeled as either white noise or time-correlated noise
(using a Gaussian process, GP). The noise model is chosen with the `ExoIris.noise_model` method, and can be
set to either "white" or "fixed_gp." Selecting "fixed_gp" models the noise as a time-correlated Gaussian process
using the `celerite2` package. The corresponding `celerite2.GaussianProcess` object can be accessed directly
via the `ExoIris.gp` attribute.

.. autosummary::
    :toctree: api/

    ExoIris.set_noise_model
    ExoIris.set_gp_kernel
    ExoIris.set_gp_hyperparameters
    ExoIris.optimize_gp_hyperparameters
    ExoIris.gp


First steps
-----------

The first steps of a transmission spectroscopy analysis include fitting a white light curve and normalising the
spectroscopic light curves. `ExoIris` offers utility methods for both of these tasks, as well as for visualising the
results.

.. autosummary::
    :toctree: api/

    ExoIris.fit_white
    ExoIris.normalize_baseline
    ExoIris.plot_white
    ExoIris.plot_residuals
    ExoIris.plot_baseline


Fitting and sampling
--------------------

The main tasks of `ExoIris` are to fit a spectroscopic light curve model to the observations and then sample its parameter
posterior to obtain a posterior transmission spectrum estimate. The fitting is carried out using a Differential Evolution
global optimiser and the sampling with the `emcee` affine invariant ensemble sampler.

The DE optimiser works by clumping a population of `npop` parameter vectors near the global posterior mode over `niter`
iterations. The optimisation can be stopped when the ptp width of the population's log posterior distribution has
decreased below a desired threshold (by default 2), after which the MCMC sampling phase can be started.

Both of these methods can be called iteratively, in which case they start from the state they finished in the previous
call. At the first `ExoIris.sample` call, the sampler will start from the current DE optimiser population. Note that
loading a previous analysis with `load_model` also loads the sampler state, so calling `ExoIris.sample` after loading
a model continues the sampler from the saved sampler state. `ExoIris.reset_sampler` should be called after loading a
saved model if you want to change the setup, optimise the new setup, and then sample the posterior.

.. autosummary::
    :toctree: api/

    ExoIris.fit
    ExoIris.sample
    ExoIris.reset_sampler


Accessing the results
---------------------

The main results from an ExoIris analysis are the model parameter posterior samples and the transmission spectrum.
The transmission spectrum, represented as the planet-to-star area ratio as a function of wavelength, can be
retrieved as a Pandas `~pandas.DataFrame` using the `ExoIris.transmission_spectrum` attribute. Similarly, the
model posterior samples can be accessed using the `ExoIris.posterior_samples` attribute, also in the form of a
Pandas `~pandas.DataFrame`.


.. autosummary::
    :toctree: api/

    ExoIris.transmission_spectrum
    ExoIris.posterior_samples
    ExoIris.plot_fit
    ExoIris.plot_transmission_spectrum
    ExoIris.plot_residuals
    ExoIris.plot_limb_darkening_parameters

Utility methods
---------------

.. autosummary::
    :toctree: api/

    ExoIris.create_initial_population
