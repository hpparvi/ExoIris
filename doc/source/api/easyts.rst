.. _api.easyts:

Transmission spectroscopy with EasyTS
=====================================
.. currentmodule:: easyts

The `EasyTS` class is the main class for transmission spectroscopy.

Constructor
-----------

The `EasyTS` class is initialized by giving it a name, limb darkening model to use, and the spectroscopic light
curves as a `TSData` object detailed later. The initializer can also be given additional parameters that further
define the analysis, but these can also be set later.

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
    EasyTS.plot_transmission_spectrum
    EasyTS.plot_residuals
    EasyTS.plot_limb_darkening

Noise model
-----------

.. autosummary::
    :toctree: api/

    EasyTS.set_noise_model
    EasyTS.set_gp_kernel
    EasyTS.set_gp_hyperparameters
    EasyTS.optimize_gp_hyperparameters
    EasyTS.gp

Utility methods
---------------

.. autosummary::
    :toctree: api/

    EasyTS.create_initial_population