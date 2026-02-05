#  ExoIris: fast, flexible, and easy exoplanet transmission spectroscopy in Python.
#  Copyright (C) 2026 Hannu Parviainen
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

from numba import njit
from numpy import *

@njit
def bin1d(v, e, el, er, bins, estimate_errors: bool = False) -> tuple[ndarray, ndarray]:
    """Bin 2D spectrophotometry data with its uncertainties into predefined bins along the first axis.

        Parameters
        ----------
        v : ndarray
            A 2D spectrophotometry array.
        e : ndarray
            A 2D array of uncertainties associated with the spectrophotometry in `v`, matching the shape of `v`.
        el : ndarray
            A 1D array containing the left edges of the integration ranges for each spectral data point.
        er : ndarray
            A 1D array containing the right edges of the integration ranges for each spectral data point.
        bins : ndarray
            A 2D array containing the edges of the bins. These should be sorted in ascending order.
        estimate_errors: bool, optional.
            Should the uncertainties be estimated from the data? Default value is False.

        Returns
        -------
        tuple of ndarrays
            A tuple containing two 2D arrays:
            - The first array (`bv`) contains the binned values of the transmission spectrum.
            - The second array (`be`) contains the binned uncertainties.
    """
    nbins = len(bins)
    ndata = v.shape[0]
    bv = zeros((nbins, v.shape[1]))
    be = zeros((nbins, v.shape[1]))
    nonfin_weights = isfinite(v).astype('d')
    v = where(nonfin_weights, v, 0.0)
    e2 = where(nonfin_weights, e**2, 0.0)
    weights = zeros(v.shape)
    npt = zeros(v.shape[1])

    i = 0
    for ibin in range(nbins):
        npt[:] = 0
        bel, ber = bins[ibin]
        for i in range(i, ndata - 1):
            if el[i + 1] > bel:
                break
        il = i
        if er[i] > ber:
            weights[i, :] = ber - bel
            npt += 1
        else:
            weights[i, :] = er[i] - max(el[i], bel)
            npt += 1
            for i in range(i + 1, ndata):
                if er[i] < ber:
                    weights[i, :] = er[i] - el[i]
                    npt += 1

                else:
                    weights[i, :] = ber - el[i]
                    npt += 1
                    break
        ir = i

        weights[il:ir+1, :] *= nonfin_weights[il:ir+1, :]
        weights[il:ir+1, :] /= weights[il:ir+1, :].sum(0)
        npt += (nonfin_weights[il:ir+1, :]-1.0).sum(0)
        ws = sum(weights[il:ir+1, :], 0)
        ws = where(ws > 0, ws, nan)
        bv[ibin] = vmean = sum(weights[il:ir+1, :] * v[il:ir+1,:], 0) / ws

        if estimate_errors:
            be[ibin, :] = where(npt > 2, sqrt(sum(weights[il:ir+1, :] * (v[il:ir+1,:] - vmean)**2, 0) / ws) / sqrt(npt), nan)
        else:
            be[ibin] = sqrt(sum(weights[il:ir+1, :]**2 * e2[il:ir+1,:], 0)) / ws
    return bv, be