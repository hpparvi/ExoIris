Baseline Modeling
=================

Spectroscopic transit observations are affected by instrumental systematics that produce time-dependent
variations in the measured flux. Detector ramps, thermal settling, and pointing drifts all contribute
trends that must be accounted for without absorbing genuine transit depth information. ExoIris models
these systematics as a multiplicative linear baseline in user-defined covariates.

Covariate Model
---------------

ExoIris describes the observed flux at each wavelength channel as

.. math::

   F_\mathrm{obs}(\lambda, t) = F_\mathrm{transit}(\lambda, t) \times \bigl[\boldsymbol{\Phi}(t) \cdot \mathbf{c}(\lambda)\bigr]

where :math:`\boldsymbol{\Phi}` is an :math:`n_t \times k` covariate (design) matrix and
:math:`\mathbf{c}(\lambda)` is a vector of per-wavelength baseline coefficients. The first column of
:math:`\boldsymbol{\Phi}` is always a column of ones that captures the out-of-transit flux level; the
remaining columns describe systematic trends.

The covariates can be:

- **Time polynomials** --- the default when no custom covariates are supplied. ``TSData`` automatically
  builds a design matrix with columns :math:`1, \hat{t}, \hat{t}^2, \ldots`, where :math:`\hat{t}` is
  centred and normalised time, up to the order set by ``n_baseline``.
- **Measured auxiliary variables** such as detector temperature, pointing drift, airmass, or
  PSF width, recorded simultaneously with the spectroscopic time series.
- **Common-mode systematics** extracted from the observations (e.g., wavelength-independent trends
  derived from the white light curve residuals).
- **Any user-supplied matrix** passed via the ``covs`` parameter of :class:`~exoiris.tsdata.TSData`.
  When ``covs`` is provided it replaces the default polynomial construction entirely, so it should
  include a constant (ones) column if an additive offset is desired.

Profiled Baseline
-----------------

When the noise model is set to ``white_profiled`` (the default), ExoIris estimates the baseline
coefficients :math:`\mathbf{c}(\lambda)` by ordinary linear least squares at every likelihood
evaluation. Concretely, given the current transit model :math:`F_\mathrm{transit}`, the residual
:math:`F_\mathrm{obs} / F_\mathrm{transit}` is regressed against :math:`\boldsymbol{\Phi}` to obtain
the best-fit coefficients, which are then multiplied back into the transit model. Because the
coefficients are determined analytically at each step, they are *profiled out* of the likelihood and
add no extra parameters to the MCMC sampler. This keeps the parameter space compact and speeds up
convergence.

Analytically Marginalized Baseline
-----------------------------------

Setting the noise model to ``white_marginalized`` activates Bayesian analytic marginalisation of the
baseline coefficients. Instead of plugging in point estimates, ExoIris integrates the coefficients out
under a broad isotropic Gaussian prior :math:`\mathbf{c} \sim \mathcal{N}(\mathbf{0},\,\tau^2 \mathbf{I})`
with :math:`\tau = 10^6`. The marginal likelihood is

.. math::

   \mathcal{L}(\theta) = \int p\!\left(F_\mathrm{obs} \mid \theta, \mathbf{c}\right)\, p(\mathbf{c})\, \mathrm{d}\mathbf{c}

which can be evaluated in closed form because the model is linear in :math:`\mathbf{c}`. Defining
:math:`\boldsymbol{\Phi}_m = \mathrm{diag}(F_\mathrm{transit}) \, \boldsymbol{\Phi}`, the marginal
covariance of the data is :math:`C = \Sigma + \tau^2 \boldsymbol{\Phi}_m \boldsymbol{\Phi}_m^\top`,
where :math:`\Sigma = \mathrm{diag}(\sigma^2)`. ExoIris evaluates :math:`C^{-1}` and
:math:`\lvert C \rvert` efficiently using the Woodbury matrix identity and the matrix determinant
lemma, working with :math:`k \times k` matrices rather than :math:`n_t \times n_t` matrices.

Compared to the profiled approach, marginalisation properly propagates baseline uncertainty into the
posterior distributions of all transit parameters. The transit model is evaluated *without* a baseline
factor; the likelihood accounts for the baseline analytically. This can yield more conservative---and
more accurate---uncertainty estimates, particularly when the covariate model is flexible or the
out-of-transit coverage is limited.
