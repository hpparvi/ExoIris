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

from numpy import full, cov, sum, ndarray, log, pi, asarray
from numpy.linalg import eigh, svd
from sklearn.utils.extmath import randomized_svd


class LogLikelihood:
    def __init__(self, wavelength: ndarray, spectra: None | ndarray = None, spmean: None | ndarray = None,
                 spcov: None | ndarray = None, eps: float = 1e-10, method: Literal['svd', 'randomized_svd', 'eigh'] = 'svd',
                 n_max_samples: int = 10000, nk: int | None = None):
        """Reduced-rank Normal log-likelihood.

        This class constructs a statistically robust log-likelihood function for
        comparing a theoretical transmission spectrum to the posterior distribution
        inferred by ExoIris.

        Because the posterior samples are generated from a spline with $K$ knots
        but evaluated on $M$ wavelengths ($M \gg K$), the empirical covariance
        matrix is singular or strongly ill-conditioned. This class solves the
        rank-deficiency problem by projecting the model into the principal
        subspace of the posterior (Karhunen-Loève compression).

        Parameters
        ----------
        wavelength
            The wavelength grid with a shape (M,) on which the posterior samples and theoretical
            spectra are evaluated.
        spectra
            The posterior spectrum samples with shape (N_samples, M_wavelengths).
            If provided, ``spmean`` and ``spcov`` are computed automatically.
            Mutually exclusive with ``spmean`` and ``spcov``.
        spmean
            The pre-computed mean spectrum with shape (M,). Must be provided
            along with ``spcov`` if ``spectra`` is None.
        spcov
            The pre-computed covariance matrix with shape (M, M). Must be provided
            along with ``spmean`` if ``spectra`` is None.
        eps
            Relative tolerance factor used to determine which eigenvalues of
            the covariance matrix are considered significant. Eigenvalues smaller
            than ``eps * max_eigenvalue`` are discarded. Default is ``1e-10``.

        Notes
        -----
        This implementation follows the "Signal-to-Noise Eigenmode" formalism
        described by Tegmark et al. (1997) for analyzing rank-deficient
        cosmological datasets.

        The log-likelihood is evaluated as:

        .. math:: \ln \mathcal{L} = -\frac{1}{2} \left[ \sum_{i=1}^{K} \frac{p_i^2}{\lambda_i} + \sum_{i=1}^{K} \ln(\lambda_i) + K \ln(2\pi) \right]

        where $\lambda_i$ are the significant eigenvalues of the covariance
        matrix, and $p_i$ are the projections of the model residuals onto the
        corresponding eigenvectors (principal components).

        References
        ----------
        Tegmark, M., Taylor, A. N., & Heavens, A. F. (1997). Karhunen-Loève
        eigenvalue problems in cosmology: how should we tackle large data sets?
        *The Astrophysical Journal*, 480(1), 22.
        """
        self.wavelength = wavelength
        self.eps = eps

        if spectra is not None and (spmean is not None or spcov is not None):
            raise ValueError("Cannot specify both `spectra` and `spmean` and `spcov`.")

        if spectra is None and (spmean is None or spcov is None):
            raise ValueError("Must specify either `spectra` or both `spmean` and `spcov`.")

        if spectra is not None:
            spectra = spectra[:n_max_samples, :]
            self.spmean = spectra.mean(axis=0)

        if method == 'svd':
            _, sigma, evecs = svd(spectra - spectra.mean(0), full_matrices=False)
            evals = (sigma**2) / (spectra.shape[0] - 1)
            evecs = evecs.T
        elif method == 'randomized_svd':
            if nk is None:
                raise ValueError("Must specify `nk` when using `method='randomized_svd'`.")
            _, sigma, evecs = randomized_svd(spectra - spectra.mean(0),  n_components=nk, n_iter=5, random_state=0)
            evals = (sigma ** 2) / (spectra.shape[0] - 1)
            evecs = evecs.T
        elif method == 'eigh' or (spmean is not None and spcov is not None):
            if spectra is not None:
                self.spcov = cov(spectra, rowvar=False)
            else:
                self.spmean = spmean
                self.spcov = spcov
            evals, evecs = eigh(self.spcov)

        keep = evals > eps * evals.max()
        self.eigenvalues, self.eigenvectors = evals[keep], evecs[:, keep]
        self.log_det = sum(log(self.eigenvalues))
        self.log_twopi = self.eigenvalues.size * log(2*pi)

    def __call__(self, model: ndarray | float) -> ndarray:
        """Evaluate the log-likelihood of a model spectrum.

        Parameters
        ----------
        model : float or ndarray
            The theoretical model spectrum. If a float is provided, it is
            broadcast to a flat spectrum. If an array, it must match the
            wavelength grid size used during initialization.

        Returns
        -------
        float
            The natural log-likelihood $\ln \mathcal{L}$.
        """
        if isinstance(model, float):
            model = full(self.wavelength.size, model)
        else:
            model = asarray(model)

        # Project the residuals onto the eigenvectors (Basis Rotation)
        # and Compute the Mahalanobis Distance (Chi-Squared in Subspace).
        p = (self.spmean - model) @ self.eigenvectors
        chisq = sum(p**2 / self.eigenvalues)
        return -0.5 * (chisq + self.log_det + self.log_twopi)
