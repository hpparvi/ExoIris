#  ExoIris: fast, flexible, and easy exoplanet transmission spectroscopy in Python.
#  Copyright (C) 2025 Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
from typing import Literal

from numpy import full, cov, sqrt, sum
from numpy.linalg import eigh
from numpy.typing import ArrayLike


class LogLikelihood:
    def __init__(self, exoiris, wavelength: ArrayLike, kind: Literal['radius_ratio', 'depth'] = 'depth', eps: float = 1e-10):
        """Reduced-rank Gaussian log-likelihood.

        This class constructs a statistically correct reduced-rank Gaussian
        log-likelihood function for comparing a theoretical transmission spectrum to the
        posterior distribution inferred by ExoIris. Because the posterior
        samples of the transmission spectrum are generated from a spline with
        far fewer independent parameters than the number of wavelength bins, the
        empirical covariance matrix is rank-deficient or strongly ill-conditioned.
        Direct inversion of the full covariance is therefore numerically unstable
        and produces incorrect likelihoods.

        Parameters
        ----------
        exoiris
            The ExoIris model object from which posterior samples of the knot
            values and spline interpolation functions are obtained.

        wavelength
            Wavelength grid on which the radius ratio posterior samples and the
            theoretical spectra will be evaluated.

        kind
            The type of the spectrum. Can be either ``radius_ratio`` for a radius ratio
            spectrum, or ``depth`` for a transit depth spectrum.

        eps
            Relative tolerance factor used to determine which eigenvalues of
            the covariance matrix are considered significant. Eigenvalues smaller
            than ``eps * max_eigenvalue`` are discarded. Default is ``1e-10``.

        Attributes
        ----------
        k_posteriors : ndarray of shape (n_samples, n_wavelengths)
            Radius-ratio posterior samples evaluated on the wavelength grid.

        k_mean : ndarray of shape (n_wavelengths,)
            Posterior mean radius-ratio spectrum.

        k_cov : ndarray of shape (n_wavelengths, n_wavelengths)
            Empirical covariance matrix of the posterior samples.

        lambda_r : ndarray of shape (k,)
            Significant eigenvalues of the covariance matrix (``k`` = reduced
            dimensionality).

        u_r : ndarray of shape (n_wavelengths, k)
            Eigenvectors corresponding to the significant eigenvalues.

        sqrt_inv_lambda_r : ndarray of shape (k,)
            Factors used to whiten the reduced-rank representation.

        y_data : ndarray of shape (k,)
            Whitened reduced-rank representation of the posterior mean spectrum.

        Notes
        -----
         The class implements the reduced-rank likelihood method:

        1. Posterior samples of the spline knot values are evaluated on the
           user-specified wavelength grid to produce a matrix of radius-ratio
           samples, ``k_posteriors``.

        2. The empirical mean spectrum and covariance matrix are computed over
           these samples.

        3. An eigendecomposition of the covariance matrix is performed. All
           eigenvalues below ``eps * max(eigenvalue)`` are discarded, ensuring that
           only statistically meaningful directions (i.e., those supported by the
           spline parameterization and the data) are retained.

        4. The retained eigenvectors form an orthonormal basis for the true
           low-dimensional subspace in which the posterior distribution lives.
           Projection onto this basis, followed by whitening with
           ``lambda_r**(-1/2)``, yields a representation where the posterior is a
           standard multivariate normal with identity covariance.

        5. The log-likelihood of a theoretical spectrum ``x`` is evaluated in this
           reduced space as:

               log L = -0.5 * sum_i (y_data[i] - y_model[i])^2

           where ``y_data`` is the whitened reduced-space representation of the
           posterior mean spectrum, and ``y_model`` is the whitened projection of
           the model spectrum.

        - This reduced-rank formulation is mathematically equivalent to computing
          the Gaussian likelihood in knot space, and avoids the numerical
          instabilities associated with inverting a nearly singular covariance
          matrix in the oversampled wavelength space.

        - If ``x`` is provided as a scalar, it is broadcast to a constant spectrum
          over the wavelength grid. Otherwise, it must be an array of the same
          wavelength length.
        """
        self.model = m = exoiris
        self.wavelength = wavelength
        self.eps = eps

        if kind == 'radius_ratio':
            self.spectrum = m.radius_ratio_spectrum(wavelength)
        elif kind == 'depth':
            self.spectrum = m.area_ratio_spectrum(wavelength)
        else:
            raise ValueError('Unknown spectrum type: {}'.format(kind))

        self.spmean = self.spectrum.mean(0)
        self.spcov = cov(self.spectrum, rowvar=False)

        evals, u = eigh(self.spcov)
        tol = eps * evals.max()
        keep = evals > tol
        self.lambda_r, self.u_r = evals[keep], u[:, keep]
        self.sqrt_inv_lambda_r = 1.0 / sqrt(self.lambda_r)
        self.y_data = (self.u_r.T @ self.spmean) * self.sqrt_inv_lambda_r

    def __call__(self, x):
        if isinstance(x, float):
            x = full(self.wavelength.size, x)
        y_model = (self.u_r.T @ x) * self.sqrt_inv_lambda_r
        return -0.5*sum((self.y_data - y_model)**2)
