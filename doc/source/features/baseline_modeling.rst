Baseline Modeling
=================

Spectroscopic transit observations are affected by instrumental systematics that produce time-dependent
variations in the measured flux. Detector ramps, thermal settling, and pointing drifts all contribute
trends that must be accounted for without absorbing genuine transit depth information. ExoIris addresses
this with a two-stage approach that balances simplicity with flexibility.

Pre-normalisation
-----------------

Before fitting, the data can be normalised by dividing out a low-order polynomial fitted to the
out-of-transit portions of each spectroscopic light curve. This removes the bulk of the baseline trend
while keeping the polynomial degree low enough---constant or linear---to avoid distorting the transit
signal.

Model-based Baseline
--------------------

During the fit, ExoIris models the remaining baseline variations as a multiplicative polynomial in time.
For each wavelength channel, the observed flux is modelled as:

.. math::

   F_\mathrm{obs}(\lambda, t) = F_\mathrm{transit}(\lambda, t) \times \left( \sum_{j=0}^{n} c_j(\lambda) \, \hat{t}^j \right)

where :math:`\hat{t}` is the centred and normalised time. The key feature of this approach is that the
polynomial coefficients :math:`c_j(\lambda)` are determined by linear least squares at each likelihood
evaluation. They are profiled out analytically rather than sampled, which means the baseline model adds no
free parameters to the MCMC. This keeps the parameter space compact and speeds up convergence, while still
providing a flexible, per-wavelength baseline correction.
