#  ExoIris: fast, flexible, and easy exoplanet transmission spectroscopy in Python.
#  Copyright (C) 2024 Hannu Parviainen
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
from numpy import (zeros, sum, sqrt, linspace, vstack, concatenate, floor, dot, ndarray, nan, asarray, tile)
from numpy._typing import ArrayLike
from pytransit import TSModel
from pytransit.orbits import i_from_ba

@njit
def bin2d(v, e, el, er, bins, estimate_errors: bool = False) -> tuple[ndarray, ndarray]:
    """Bin 2D exoplanet transmission spectrum data with its uncertainties into predefined bins in wavelength.

        Parameters
        ----------
        v : ndarray
            A 2D array of the exoplanet transmission spectrum data with a shape (n_wavelength, n_exposure).
        e : ndarray
            A 2D array of uncertainties associated with the spectrum data in `v`, matching the shape of `v`.
        el : ndarray
            A 1D array containing the left wavelength edges of the integration ranges for each spectral data point.
        er : ndarray
            A 1D array containing the right wavelength edges of the integration ranges for each spectral data point.
        bins : ndarray
            A 2D array containing the edges of the wavelength bins. These should be sorted in ascending order.
        estimate_errors: bool, optional.
            Should the uncertainties be estimated from the data. Default value is False.

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
    e2 = e**2
    weights = zeros(ndata)
    i = 0
    for ibin in range(nbins):
        weights[:] = 0.0
        npt = 0
        bel, ber = bins[ibin]
        for i in range(i, ndata - 1):
            if el[i + 1] > bel:
                break
        il = i
        if er[i] > ber:
            weights[i] = ber - bel
            npt += 1
        else:
            weights[i] = er[i] - bel
            npt += 1
            for i in range(i + 1, ndata):
                if er[i] < ber:
                    weights[i] = er[i] - el[i]
                    npt += 1
                else:
                    weights[i] = ber - el[i]
                    npt += 1
                    break
        ir = i
        ws = sum(weights)
        bv[ibin] = vmean = dot(weights[il:ir+1], v[il:ir+1,:]) / ws
        if estimate_errors:
            if npt > 1:
                be[ibin] = sqrt(dot(weights[il:ir+1], (v[il:ir+1,:] - vmean)**2) / ws) / sqrt(npt)
            else:
                be[ibin] = nan
        else:
            be[ibin] = sqrt(dot(weights[il:ir+1], e2[il:ir+1,:])) / ws
    return bv, be


def create_binning(ranges, bwidths):
    """
    Create a combined array of discretization edges for multiple ranges with specified bin widths.

    This function generates a single concatenated array of discretization bins for given ranges
    where each range can have a different bin width. Each range is divided into bins of the specified width,
    and the start and end points of these bins are stored.

    Parameters
    ----------
    ranges : list of tuples
        A list where each tuple contains two elements (start, end) defining the range over which to create bins.
    bwidths : list of float
        A list of bin widths corresponding to each range in `ranges`. Each width specifies the size of the bin
        for the corresponding range.

    Returns
    -------
    ndarray
        A 2D NumPy array where each row contains the start and end points of a bin.

    Notes
    -----
    The final bin widths may differ from the given ones if they do not perfectly divide the ranges.
    """
    bins = []
    for r, w in zip(ranges, bwidths):
        n = int(floor((r[1] - r[0]) / w))
        e = linspace(*r, num=n)
        bins.append(vstack([e[:-1], e[1:]]).T)
    return concatenate(bins)


def create_mock_model(ks: ArrayLike, times: ArrayLike = None, ldc: ArrayLike = None, t0: float = 0.0, p: float =2.0, a: float =8.0, b: float =0.0) -> ndarray:
    """Create a mock transmission spectrum observation using given parameters.

    Parameters
    ----------
    ks
        Array of radius ratios, one radius ratio per wavelength.
    times
        Array of time values to set the data points. If None, defaults to a
        linspace of 500 points in the range [-0.1, 0.1].
    ldc
        Array representing the limb darkening coefficients. If None, defaults to
        a tile of [0.4, 0.4] for each wavelength element.
    t0
        Transit center.
    p
        Orbital period.
    a
        Semi-major axis.
    b
        Impact parameter.

    Returns
    -------
    ndarray
        Mock spectrophotometric light curves.

    """
    ks = asarray(ks)
    if times is None:
        times = linspace(-0.1, 0.1, 500)
    if ldc is None:
        ldc = tile([0.4, 0.4], (1, ks.size, 1))
    inc = i_from_ba(b, a)

    m1 = TSModel('power-2', ng=100, nzin=50, nzlimb=50)
    m1.set_data(times)
    f1 = m1.evaluate(ks, ldc, t0, p, a, inc)[0]
    return f1
