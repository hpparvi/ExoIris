Broadband Photometry
====================

Transmission spectroscopy rarely happens in isolation. Most targets observed with a spectrograph also have
broadband transit light curves from ground-based follow-up or space-based photometric surveys, often at
wavelengths the spectrograph does not reach and at epochs that extend the time baseline by years. ExoIris
can model these light curves together with the spectroscopic time series, in the same model and with the
same parameters.

How It Works
------------

A broadband light curve is the spectroscopic signal seen through a filter. ExoIris treats it exactly as
such: the filter passband is divided into a small number of wavelength elements, the transit model is
evaluated for each element, and the result is averaged over the passband using the filter transmission
:math:`T` as weights,

.. math::

   F_\mathrm{bb}(t) = \frac{\sum_i T(\lambda_i)\, F(\lambda_i, t)}{\sum_i T(\lambda_i)}.

The radius ratio and limb darkening at each :math:`\lambda_i` come from the same wavelength-dependent
models that describe the spectroscopic data. The photometry therefore does not need a radius ratio or limb
darkening coefficients of its own. It constrains the transmission spectrum and the limb darkening profile
directly at the wavelengths the filter covers, with the correct weighting across a passband where both may
change.

Why It Matters
--------------

Broadband photometry adds information that spectroscopy alone often lacks. Light curves at shorter or
longer wavelengths extend the transmission spectrum beyond the spectrograph's range, anchoring a
scattering slope in the optical or the continuum level in the infrared. Light curves from other transits
tighten the orbital period and reveal transit timing variations. And because the orbital parameters are
shared across all the data, every additional light curve sharpens the geometry that the spectroscopic
transit depths depend on.

Broadband light curves use the same noise, epoch, and offset groups as the spectroscopic data, and their
baselines are modelled with the same covariate approach, so adding one to an analysis is no different from
adding another spectroscopic visit.
