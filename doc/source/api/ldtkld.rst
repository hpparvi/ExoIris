.. _api.ldtkld:

Limb darkening with LDTk
========================
.. currentmodule:: exoiris

The `LDTkLD` class provides integration with the `LDTk <https://github.com/hpparvi/ldtk>`_ package for
theoretical limb darkening coefficient estimation from stellar atmosphere models. This class extends
PyTransit's LDTk integration to work with ExoIris's transmission spectroscopy framework.

`LDTkLD` can be used directly when initializing the `ExoIris` class by passing it as the `ldmodel` parameter.
This allows ExoIris to use physically motivated limb darkening priors based on the host star's properties
(effective temperature, surface gravity, and metallicity).

Main class
----------

.. autosummary::
    :toctree: api/

    LDTkLD
