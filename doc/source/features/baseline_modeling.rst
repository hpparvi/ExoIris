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
remaining columns describe systematic trends. The covariates can be time polynomials (the default),
measured auxiliary variables such as detector temperature or pointing drift, common-mode systematics
extracted from the white light curve residuals, or any user-supplied matrix that replaces the default
polynomial construction entirely.

Profiled Baseline
-----------------

In the default profiled mode, ExoIris estimates the baseline coefficients by ordinary linear least
squares at every likelihood evaluation. The residual between the data and the current transit model is
regressed against the design matrix, and the resulting best-fit coefficients are folded back into the
model. Because the coefficients are determined analytically at each step, they are *profiled out* of the
likelihood and add no extra parameters to the sampler. This keeps the parameter space compact and speeds
up convergence.

Analytically Marginalized Baseline
-----------------------------------

As an alternative, ExoIris can analytically marginalise the baseline coefficients under a broad Gaussian
prior instead of plugging in point estimates. The marginal likelihood integrates the coefficients out in
closed form,

.. math::

   \mathcal{L}(\theta) = \int p\!\left(F_\mathrm{obs} \mid \theta, \mathbf{c}\right)\, p(\mathbf{c})\, \mathrm{d}\mathbf{c}

because the model is linear in :math:`\mathbf{c}`. The resulting covariance is evaluated efficiently via
the Woodbury matrix identity and the matrix determinant lemma, keeping the cost proportional to the
number of covariates rather than the number of time points.

Compared to the profiled approach, marginalisation properly propagates baseline uncertainty into the
posterior distributions of all transit parameters. This can yield more conservative---and more
accurate---uncertainty estimates, particularly when the covariate model is flexible or the out-of-transit
coverage is limited.
