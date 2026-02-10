Retrieval Likelihood
====================

After fitting a transmission spectrum with ExoIris, the next step is often atmospheric retrieval:
comparing the measured spectrum to theoretical models to infer atmospheric properties. ExoIris provides
a statistically rigorous likelihood function for this purpose, based on a reduced-rank Gaussian
approximation that properly accounts for correlations between wavelength bins.

The Rank-Deficiency Problem
---------------------------

ExoIris parameterises the transmission spectrum using :math:`K` spline knots, which are then
interpolated to :math:`M` data wavelengths. Since :math:`M \gg K`, the posterior samples live in
a :math:`K`-dimensional subspace of the :math:`M`-dimensional wavelength space. The full
:math:`M \times M` covariance matrix is therefore singular, and a standard multivariate Gaussian
likelihood cannot be evaluated directly.

Karhunen-Loeve Decomposition
------------------------------

ExoIris solves this using a Karhunen-Loeve (principal component) decomposition of the posterior
covariance. The eigendecomposition identifies the :math:`K` significant principal components and
discards the near-zero eigenvalues that correspond to directions with no posterior variance. The
resulting reduced-rank likelihood is:

.. math::

   \ln \mathcal{L} = -\frac{1}{2} \left( \sum_{i=1}^{K} \frac{p_i^2}{\lambda_i} + \ln \prod_{i=1}^{K} \lambda_i + K \ln 2\pi \right)

where :math:`p_i` are the projections of the residuals onto the eigenvectors and :math:`\lambda_i` are
the corresponding eigenvalues.

Usage
-----

After MCMC sampling, create the likelihood function:

.. code-block:: python

   wavelengths = np.linspace(0.6, 5.0, 1000)
   lnlike = iris.create_loglikelihood_function(wavelengths, kind='depth', method='svd')

The returned ``lnlike`` object is callable and accepts a theoretical transmission spectrum evaluated at
the same wavelengths:

.. code-block:: python

   lnL = lnlike(theoretical_spectrum)

Three decomposition methods are available:

- **svd** (default): Full singular value decomposition of the centred posterior samples. Most accurate.
- **randomized_svd**: Randomised SVD approximation for very large sample sets.
- **eigh**: Eigendecomposition of the covariance matrix.

The ``kind`` parameter controls whether the likelihood operates on radius ratios (``'radius_ratio'``)
or transit depths (``'depth'``).
