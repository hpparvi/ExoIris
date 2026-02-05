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

from numpy import (linspace, vstack, concatenate, floor, ndarray, asarray, tile)
from numpy._typing import ArrayLike
from pytransit import TSModel
from pytransit.orbits import i_from_ba


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
