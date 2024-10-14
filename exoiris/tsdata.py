#  ExoIris: fast, flexible, and easy exoplanet transmission spectroscopy in Python.
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

from collections.abc import Sequence

from typing import Union, Optional

from astropy.stats import mad_std
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, setp
from matplotlib.ticker import LinearLocator, FuncFormatter
from numpy import isfinite, median, where, concatenate, all, zeros_like, diff, asarray, interp, arange, floor, ndarray, \
    ceil, newaxis
from pytransit.orbits import fold
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
    def __init__(self, time: Sequence, wavelength: Sequence, fluxes: Sequence, errors: Sequence,
                 name: str = "", wl_edges : Sequence | None = None, tm_edges : Sequence | None = None):
        """
        Parameters
        ----------
        time
            Array of time values.
        wavelength
            Array of wavelength values.
        fluxes
            2D array of flux values with a shape ``(nwl, npt)``, where ``nwl`` is the number of wavelengths and ``npt`` the
            number of exposures.
        errors
            2D Array of error values with a shape ``(nwl, npt)``, where ``nwl`` is the number of wavelengths and ``npt`` the
            number of exposures.
        wl_edges
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
        self.ootmask = None
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

        if tm_edges is None:
            dt = zeros_like(self.time)
            dt[:-1] = diff(self.time)
            dt[-1] = dt[-2]
            self._tm_l_edges = self.time - 0.5 * dt
            self._tm_r_edges = self.time + 0.5 * dt
        else:
            self._tm_l_edges = tm_edges[0]
            self._tm_r_edges = tm_edges[1]

    def __repr__(self) -> str:
        return f"TSData Name:'{self.name}' [{self.wavelength[0]:.2f} - {self.wavelength[-1]:.2f}] nwl={self.nwl} npt={self.npt}"

    def calculate_ootmask(self, t0: float, p: float, t14: float):
        phase = fold(self.time, p, t0)
        self.ootmask = abs(phase) > 0.502 * t14

    def _update(self) -> None:
        """Update the internal attributes."""
        self.nwl = self.wavelength.size
        self.npt = self.time.size
        self.wllims = self.wavelength.min(), self.wavelength.max()

    def normalize(self, s: slice) -> None:
        """Normalize the light curves to the median flux of the given slice along the time axis.

        Parameters
        ----------
        s : slice
            A slice object representing the portion of the data to normalize.
        """
        n = median(self.fluxes[:, s], axis=1)[:, newaxis]
        self.fluxes[:,:] /= n
        self.errors[:,:] /= n

    def split_time(self, t: float, b: float) -> 'TSDataSet':
        """Split the data into two parts: (time < t-b) and (time > t+b).

        Parameters
        ----------
        t : float
            The threshold time value used to split the data.
        b : float
            The buffer time around the threshold `t` to exclude from the split range.
        """
        m1 = self.time < t-b
        m2 = self.time > t+b
        t1 = TSData(time=self.time[m1], wavelength=self.wavelength, fluxes=self.fluxes[:, m1], errors=self.errors[:, m1])
        t2 = TSData(time=self.time[m2], wavelength=self.wavelength, fluxes=self.fluxes[:, m2], errors=self.errors[:, m2])
        return t1 + t2

    def partition_time(self, tlims: tuple[tuple[float,float]]) -> 'TSDataSet':
        """Partition the data into n segments defined by tlims.

        Parameters
        ----------
        tlims
            The lower and upper time limits for each segment.
        """
        masks = [(self.time >= l[0]) & (self.time <= l[1]) for l in tlims]
        m = masks[0]
        d = TSData(time=self.time[m], wavelength=self.wavelength, fluxes=self.fluxes[:, m], errors=self.errors[:, m])
        for m in masks[1:]:
            d = d + TSData(time=self.time[m], wavelength=self.wavelength, fluxes=self.fluxes[:, m], errors=self.errors[:, m])
        return d

    def crop_wavelength(self, lmin: float, lmax: float) -> None:
        """Crop the data to include only the wavelength range between lmin and lmax.

        Parameters
        ----------
        lmin
            The minimum wavelength value to crop.
        lmax
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
        sigma
            The number of standard deviations to use as the threshold for outliers.

        Note
        ----
        The data will be modified in place.
        """
        fm = median(self.fluxes, axis=0)
        fe = mad_std(self.fluxes, axis=0)
        self.fluxes = where(abs(self.fluxes - fm) / fe < sigma, self.fluxes, median_filter(self.fluxes, 5))

    def plot(self, ax=None, vmin: float = None, vmax: float = None, cmap=None, figsize=None, data=None) -> Figure:
        """Plot the spectroscopic light curves as a 2D image.

        Plot the spectroscopic light curves as a 2D image with time on the x-axis, wavelength and light curve index
        on the y-axis, and the flux as a color.

        Parameters
        ----------
        ax
            The subplot axes on which to plot. If None, a new figure and axes will be created.

        vmin
            The minimum value of the color scale.

        vmax
            The maximum value of the color scale.

        cmap
            The colormap to be used.

        figsize
            The size of the figure in inches (width, height).

        data
            Dataset to plot instead of self.fluxes.

        Returns
        -------
        Figure

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

    def plot_white(self, ax=None, figsize=None) -> Axes:
        if ax is None:
            fig, ax = subplots(figsize=figsize)
        else:
            fig = ax.figure
        tref = floor(self.time.min())

        ax.plot(self.time, self.fluxes.mean(0))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p: f"{x-tref:.3f}"))
        setp(ax, xlabel=f'Time - {tref:.0f} [d]', ylabel='Normalized flux', xlim=[self.time[0]-0.003, self.time[-1]+0.003])
        return ax

    def __add__(self, other: 'TSData') -> 'TSDataSet':
        """Combine two transmission spectra along the wavelength axis.

        Parameters
        ----------
        other
            The TSData object to be added to the current TSData object.

        Returns
        -------
        TSDataSet
            The resulting TSDataSet object combining the two TSData objects.

        """
        return TSDataSet([self, other])

    def bin_wavelength(self, binning: Optional[Union[Binning, CompoundBinning]] = None,
                       nb: Optional[int] = None, bw: Optional[float] = None, r: Optional[float] = None,
                       estimate_errors: bool = False):
        """Bin the data along the wavelength axis.

        Bin the data along the wavelength axis. If binning is not specified, a Binning object is created using the
        minimum and maximum values of the wavelength.

        Parameters
        ----------
        binning
            The binning method to use. Default value is None.
        nb
            Number of bins. Default value is None.
        bw
            Bin width. Default value is None.
        r
            Bin resolution. Default value is None.
        estimate_errors
            Should the uncertainties be estimated from the data. Default value is False.

        Returns
        -------
        TSData
            The binned data.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numba.NumbaPerformanceWarning)
            if binning is None:
                binning = Binning(self.wllims[0], self.wllims[1], nb=nb, bw=bw, r=r)
            bf, be = bin2d(self.fluxes, self.errors, self._wl_l_edges, self._wl_r_edges,
                           binning.bins, estimate_errors=estimate_errors)
            if not all(isfinite(be)):
                warnings.warn('Error estimation failed for some bins, check the error array.')
            return TSData(self.time, binning.bins.mean(1), bf, be, wl_edges=(binning.bins[:,0], binning.bins[:,1]),
                          name=self.name, tm_edges=(self._tm_l_edges, self._tm_r_edges))


    def bin_time(self, binning: Optional[Union[Binning, CompoundBinning]] = None,
                       nb: Optional[int] = None, bw: Optional[float] = None, r: Optional[float] = None,
                       estimate_errors: bool = False):
        """Bin the data along the time axis.

        Bin the data along the time axis. If binning is not specified, a Binning object is created using the
        minimum and maximum time values.

        Parameters
        ----------
        binning
            The binning method to use. Default value is None.
        nb
            Number of bins. Default value is None.
        bw
            Bin width. Default value is None.
        r
            Bin resolution. Default value is None.
        estimate_errors
            Should the uncertainties be estimated from the data. Default value is False.

        Returns
        -------
        TSData
            The binned data.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numba.NumbaPerformanceWarning)
            if binning is None:
                binning = Binning(self.time.min(), self.time.max(), nb=nb, bw=bw, r=r)
            bf, be = bin2d(self.fluxes.T, self.errors.T, self._tm_l_edges, self._tm_r_edges,
                           binning.bins, estimate_errors=estimate_errors)
            return TSData(binning.bins.mean(1), self.wavelength, bf.T, be.T,
                          wl_edges=(self._wl_l_edges, self._wl_r_edges),
                          tm_edges=(binning.bins[:,0], binning.bins[:,1]),
                          name=self.name)


class TSDataSet:
    """A high-level data storage class that can contain multiple TSData objects."""
    def __init__(self, data: Sequence[TSData]):
        self.data: list[TSData] = list(data)
        self.time: list[ndarray] = [d.time for d in data]
        self.fluxes: list[ndarray] = [d.fluxes for d in data]
        self.errors: list[ndarray] = [d.errors for d in data]
        self.wavelength: list[ndarray] = [d.wavelength for d in data]
        self.wlmin = concatenate(self.wavelength).min()
        self.wlmax = concatenate(self.wavelength).max()
        self.ngroups: int = len(self.data)
        self.groups: list = []
        i = 0
        for d in data:
            self.groups.append(slice(i, i+d.nwl))
            i += d.nwl

    def calculate_ootmask(self, tc: float, p: float, t14: float):
        for d in self.data:
            d.calculate_ootmask(tc, p, t14)

    def __getitem__(self, index: int) -> TSData:
        return self.data[index]

    def __len__(self) -> int:
        return self.ngroups

    def __repr__(self):
        return f"TSDataSet with {self.ngroups} groups: {str(self.groups)}"

    def plot(self, ax=None, vmin: float = None, vmax: float = None, ncols: int = 1, cmap=None, figsize=None, data: ndarray | None = None):
        """Plot all the data sets.

        Parameters
        ----------
        ax
            The Axes object used for plotting. If None, a new set of subplots will be created.
        vmin
            The minimum value for the color mapping. Default is None.
        vmax
            The maximum value for the color mapping. Default is None.
        cmap
            The colormap used for mapping the data values to colors. Default is None.
        figsize
            The size of the figure created if `ax` is None. Default is None.
        data
            The data to be plotted. If None, the `self.data` attribute will be used.

        Returns
        -------
        Figure
            The Figure object that contains the subplots.

        """
        if ax is None:
            nrows = int(ceil(self.ngroups / ncols))
            fig, ax = subplots(nrows, ncols=ncols, figsize=figsize)
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