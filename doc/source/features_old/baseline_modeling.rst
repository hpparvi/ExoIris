Baseline Modeling
=================

Spectroscopic transit observations are affected by instrumental systematics that produce time-dependent
variations in the measured flux. ExoIris provides a two-stage approach to baseline modelling that
balances simplicity with flexibility.

Pre-normalisation
-----------------

Before fitting, the data can be normalised by dividing out a low-order polynomial fitted to the
out-of-transit portions of each spectroscopic light curve:

.. code-block:: python

   iris.normalize_baseline(deg=1)

This removes the bulk of the baseline trend. The polynomial degree is restricted to 0 (constant) or 1
(linear) to avoid absorbing transit depth information into the baseline. A ``transit_mask`` must be defined
first so the normalisation knows which data points are out of transit.

Model-based Baseline
--------------------

During the fit, ExoIris models the remaining baseline variations as a multiplicative polynomial in time.
For each wavelength bin, the observed flux is modelled as:

.. math::

   F_\mathrm{obs}(\lambda, t) = F_\mathrm{transit}(\lambda, t) \times \left( \sum_{j=0}^{n} c_j(\lambda) \, \hat{t}^j \right)

where :math:`\hat{t}` is the centred and normalised time, and the polynomial coefficients
:math:`c_j(\lambda)` are determined by linear least squares at each likelihood evaluation. This approach
has no free parameters in the MCMC---the baseline coefficients are profiled out analytically, which
keeps the parameter space compact and speeds up convergence.

The baseline order is set per dataset when creating the ``TSData`` object via the ``n_baseline``
parameter (default is 1, i.e., a linear baseline).
