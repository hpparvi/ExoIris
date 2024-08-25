#  EasyTS: fast, flexible, and easy exoplanet transmission spectroscopy in Python.
#  Copyright (C) 2024 Hannu Parviainen
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import warnings
import numba
from typing import Union, Optional, Iterable

from astropy.stats import mad_std
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, setp
from matplotlib.ticker import LinearLocator
from numpy import isfinite, median, where, concatenate, all, zeros_like, diff, asarray, interp, arange, floor, ndarray
from scipy.ndimage import median_filter

from .util import bin2d
from .binning import Binning, CompoundBinning

class TSData:
    """
    TSData is a utility class representing transmission spectroscopy time series data with associated wavelength,
    fluxes, and errors. It provides methods for manipulating and analyzing the data.

    Attributes
    ----------
    time: 1D ndarray
        Array of time values.
    wavelength : 1D ndarray
        Array of wavelength values.
    fluxes : 2D ndarray
        2D array of flux values with a shape ``(nwl, npt)``, where ``nwl`` is the number of wavelengths and ``npt`` the
        number of exposures.
    errors : 2D ndarray
        2D Array of error values with a shape ``(nwl, npt)``, where ``nwl`` is the number of wavelengths and ``npt`` the
        number of exposures.
    """
    def __init__(self, time: Iterable, wavelength: Iterable, fluxes: Iterable, errors: Iterable,
                 name: str = "", wl_edges : Iterable | None = None):
        """
        Parameters
        ----------
        time : 1D array-like
            Array of time values.
        wavelength : 1D array-like
            Array of wavelength values.
        fluxes : 2D array-like
            2D array of flux values with a shape ``(nwl, npt)``, where ``nwl`` is the number of wavelengths and ``npt`` the
            number of exposures.
        errors : 2D array-like
            2D Array of error values with a shape ``(nwl, npt)``, where ``nwl`` is the number of wavelengths and ``npt`` the
            number of exposures.
        wl_edges : tuple of 1D array-like, optional
            Tuple containing left and right wavelength edges for each wavelength element.
        """
        time, wavelength, fluxes, errors = asarray(time), asarray(wavelength), asarray(fluxes), asarray(errors)

        if fluxes.shape[0] != wavelength.size:
            raise ValueError("The size of the flux array's first axis must match the size of the wavelength array.")

        m = all(isfinite(fluxes), axis=1)
        self.name = name
        self.time = time.copy()
        self.wavelength = wavelength[m]
        self.fluxes = fluxes[m]
        self.errors = errors[m]
        self._update()
        self.groups = [slice(0, self.nwl)]

        if wl_edges is None:
            dwl = zeros_like(self.wavelength)
            dwl[:-1] = diff(self.wavelength)
            dwl[-1] = dwl[-2]
            self._wl_l_edges = self.wavelength - 0.5 * dwl
            self._wl_r_edges = self.wavelength + 0.5 * dwl
        else:
            self._wl_l_edges = wl_edges[0]
            self._wl_r_edges = wl_edges[1]

    def _update(self):
        """Update the internal attributes."""
        self.nwl = self.wavelength.size
        self.npt = self.time.size
        self.wllims = self.wavelength.min(), self.wavelength.max()

    def crop_wavelength(self, lmin: float, lmax: float) -> None:
        """Crop the data to include only the wavelength range between lmin and lmax.

        Parameters
        ----------
        lmin : float
            The minimum wavelength value to crop.
        lmax : float
            The maximum wavelength value to crop.

        Note
        ----
        The data will be modified in place.
        """
        m = (self.wavelength > lmin) & (self.wavelength < lmax)
        self.wavelength = self.wavelength[m]
        self.fluxes = self.fluxes[m]
        self.errors = self.errors[m]
        self._wl_l_edges = self._wl_l_edges[m]
        self._wl_r_edges = self._wl_r_edges[m]
        self._update()

    def remove_outliers(self, sigma: float = 5.0):
        """Remove outliers along the wavelength axis.

        Replace outliers along the wavelength axis with the value of a 5-point running median filter. Outliers are
        defined as data points that deviate from the median by more than sigma times the median absolute deviation
        along the wavelength axis.

        Parameters
        ----------
        sigma : float, default = 5.0
            The number of standard deviations to use as the threshold for outliers.

        Note
        ----
        The data will be modified in place.
        """
        fm = median(self.fluxes, axis=0)
        fe = mad_std(self.fluxes, axis=0)
        self.fluxes = where(abs(self.fluxes - fm) / fe < sigma, self.fluxes, median_filter(self.fluxes, 5))

    def plot(self, ax=None, vmin: float = None, vmax: float = None, cmap=None, figsize=None, data=None) -> Figure:
        """Plot the data as a 2D image.

        Plot the data as a 2D image with time on the x-axis, wavelength and light curve index on the y-axis, and the
        flux as a color.

        Parameters
        ----------
        ax : Axes, optional
            The subplot axes on which to plot. If None, a new figure and axes will be created.

        vmin : float, optional
            The minimum value of the color scale.

        vmax : float, optional
            The maximum value of the color scale.

        cmap : str or colormap, optional
            The colormap to be used.

        figsize : tuple, optional
            The size of the figure in inches (width, height).

        data : ndarray, optional
            Dataset to plot instead of self.fluxes.

        Returns
        -------
        ax : Axes
            The subplot axes on which the plot was made.

        ax2 : SecondaryAxis
            The secondary y-axis used for the light curve index.

        """
        if ax is None:
            fig, ax = subplots(figsize=figsize)
        else:
            fig = ax.figure
        tref = floor(self.time.min())

        def forward(x):
            return interp(x, self.wavelength, arange(self.nwl))
        def inverse(x):
            return interp(x, arange(self.nwl), self.wavelength)

        data = data if data is not None else self.fluxes
        ax.pcolormesh(self.time - tref, self.wavelength, data, vmin=vmin, vmax=vmax, cmap=cmap)
        setp(ax, ylabel=r'Wavelength [$\mu$m]', xlabel=f'Time - {tref:.0f} [BJD]')
        ax.yaxis.set_major_locator(LinearLocator(10))
        ax.yaxis.set_major_formatter('{x:.2f}')

        if self.name != "":
            ax.set_title(self.name)

        ax2 = ax.secondary_yaxis('right', functions=(forward, inverse))
        ax2.set_ylabel('Light curve index')
        ax2.set_yticks(forward(ax.get_yticks()))
        ax2.yaxis.set_major_formatter('{x:.0f}')
        return fig

    def __add__(self, other):
        """Combine two transmission spectra along the wavelength axis.

        Parameters
        ----------
        other : TSData
            The TSData object to be added to the current TSData object.

        Returns
        -------
        TSData
            The resulting TSData object after adding the other TSData object.

        Raises
        ------
        ValueError
            If the wavelength ranges of the current TSData object and the other TSData object overlap.

        """
        return TSDataSet([self, other])

    def bin_wavelength(self, binning: Optional[Union[Binning, CompoundBinning]] = None,
                       nb: Optional[int] = None, bw: Optional[float] = None, r: Optional[float] = None):
        """Bin the data along the wavelength axis.

        Bin the data along the wavelength axis. If binning is not specified, a Binning object is created using the
        minimum and maximum values of the wavelength.

        Parameters
        ----------
        binning: Optional[Union[Binning, CompoundBinning]], optional
            The binning method to use. Default value is None.
        nb: int, optional
            Number of bins. Default value is None.
        bw: float, optional
            Bin width. Default value is None.
        r: float, optional
            Bin resolution. Default value is None.

        Returns
        -------
        TSData
            The binned data.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numba.NumbaPerformanceWarning)
            if binning is None:
                binning = Binning(self.wllims[0], self.wllims[1], nb=nb, bw=bw, r=r)
            bf, be = bin2d(self.fluxes, self.errors, self._wl_l_edges, self._wl_r_edges, binning.bins)
            return TSData(self.time, binning.bins.mean(1), bf, be, wl_edges=(binning.bins[:,0], binning.bins[:,1]),
                          name=self.name)


class TSDataSet:
    def __init__(self, data: Iterable[TSData]):
        self.data = list(data)
        self.time = data[0].time
        self.fluxes = concatenate([d.fluxes for d in data])
        self.errors = concatenate([d.errors for d in data])
        self.wavelength = concatenate([d.wavelength for d in data])
        self.ngroups = len(self.data)
        self.groups = []
        i = 0
        for d in data:
            self.groups.append(slice(i, i+d.nwl))
            i += d.nwl

    def __repr__(self):
        return f"TSDataSet with {self.ngroups} groups: {str(self.groups)}"

    def plot(self, ax=None, vmin: float = None, vmax: float = None, cmap=None, figsize=None, data: ndarray | None = None):
        if ax is None:
            fig, ax = subplots(self.ngroups, 1, figsize=figsize, sharex='all')
        else:
            fig = ax[0].get_figure()

        if data is None:
            for i, a in enumerate(ax):
                self.data[i].plot(ax=a, vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            for i, a in enumerate(ax):
                self.data[i].plot(ax=a, vmin=vmin, vmax=vmax, cmap=cmap, data=data[self.groups[i], :])

        setp(ax[:-1], xlabel='')
        return fig

    def __add__(self, other):
        if isinstance(other, TSData):
            return TSDataSet(self.data + [other])
        elif isinstance(other, TSDataSet):
            return TSDataSet(self.data + other.data)