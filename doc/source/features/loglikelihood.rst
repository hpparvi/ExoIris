Retrieval Likelihood
====================

After fitting a transmission spectrum with ExoIris, the next step is often atmospheric retrieval:
comparing the measured spectrum to theoretical models to infer atmospheric properties. This step requires a
likelihood function that properly accounts for the statistical structure of the ExoIris posterior, and a
naive approach fails in a subtle but important way.

The Rank-Deficiency Problem
---------------------------

ExoIris parameterises the transmission spectrum using :math:`K` spline knots, which are then
interpolated to :math:`M` data wavelengths. Since :math:`M \gg K`, the posterior samples live in
a :math:`K`-dimensional subspace of the :math:`M`-dimensional wavelength space. The full
:math:`M \times M` covariance matrix is therefore singular: it has only :math:`K` non-zero eigenvalues.
A standard multivariate Gaussian likelihood cannot be evaluated because the covariance matrix is not
invertible.

Karhunen-Loeve Decomposition
------------------------------

ExoIris solves this using a Karhunen-Loeve (principal component) decomposition of the posterior
covariance. The eigendecomposition identifies the :math:`K` significant principal components and
discards the near-zero eigenvalues that correspond to directions with no posterior variance. The
resulting reduced-rank likelihood is:

.. math::

   \ln \mathcal{L} = -\frac{1}{2} \left( \sum_{i=1}^{K} \frac{p_i^2}{\lambda_i} + \ln \prod_{i=1}^{K} \lambda_i + K \ln 2\pi \right)

where :math:`p_i` are the projections of the residuals onto the eigenvectors and :math:`\lambda_i` are
the corresponding eigenvalues. This gives a statistically rigorous likelihood that lives in the correct
subspace, ready to be plugged into any atmospheric retrieval framework.
