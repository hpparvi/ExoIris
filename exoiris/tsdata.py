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

import pandas as pd
from collections.abc import Sequence

from typing import Union, Optional

from astropy.io import fits as pf
from astropy.stats import mad_std
from astropy.utils import deprecated
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, setp
from matplotlib.ticker import LinearLocator, FuncFormatter
from numpy import isfinite, median, where, concatenate, all, zeros_like, diff, asarray, interp, arange, floor, ndarray, \
    ceil, newaxis, inf, array, ones, unique, poly1d, polyfit, nanpercentile
from numpy.ma.extras import atleast_2d
from pytransit.orbits import fold
from scipy.ndimage import median_filter

from .ephemeris import Ephemeris
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
    def __init__(self, time: Sequence, wavelength: Sequence, fluxes: Sequence, errors: Sequence, name: str,
                 noise_group: str = 'a', wl_edges : Sequence | None = None, tm_edges : Sequence | None = None,
                 ootmask: ndarray | None = None, ephemeris: Ephemeris | None = None) -> None:
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
        name
            Name for the data set.
        noise_group
            Noise group the data belongs to.
        wl_edges
            Tuple containing left and right wavelength edges for each wavelength element.
        wl_edges
            Tuple containing left and right time edges for each exposure.
        """
        time, wavelength, fluxes, errors = asarray(time), asarray(wavelength), asarray(fluxes), asarray(errors)

        if fluxes.shape[0] != wavelength.size:
            raise ValueError("The size of the flux array's first axis must match the size of the wavelength array.")

        if ootmask is not None and ootmask.size != time.size:
            raise ValueError("The size of the out-of-transit mask array must match the size of the time array.")

        m = all(isfinite(fluxes), axis=1)
        self.name: str = name
        self.time: ndarray = time.copy()
        self.wavelength: ndarray = wavelength[m]
        self.fluxes: ndarray = fluxes[m]
        self.errors: ndarray = errors[m]
        self.ootmask: ndarray = ootmask if ootmask is not None else ones(time.size, dtype=bool)
        self.ngid: int = 0
        self.ephemeris: Ephemeris | None = ephemeris
        self._noise_group: str = noise_group
        self._dataset: 'TSDataSet' | None = None
        self._update()

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

    def export_fits(self) -> pf.HDUList:
        time = pf.ImageHDU(self.time, name=f'time_{self.name}')
        wave = pf.ImageHDU(self.wavelength, name=f'wave_{self.name}')
        data = pf.ImageHDU(array([self.fluxes, self.errors]), name=f'data_{self.name}')
        ootm = pf.ImageHDU(self.ootmask.astype(int), name=f'ootm_{self.name}')
        data.header['ngroup'] = self.noise_group
        #TODO: export ephemeris
        return pf.HDUList([time, wave, data, ootm])

    @staticmethod
    def import_fits(name: str, hdul: pf.HDUList) -> 'TSData':
        time = hdul[f'TIME_{name}'].data
        wave = hdul[f'WAVE_{name}'].data
        data = hdul[f'DATA_{name}'].data
        ootm = hdul[f'OOTM_{name}'].data
        noise_group = hdul[f'DATA_{name}'].header['NGROUP']
        #TODO: import ephemeris
        return TSData(time, wave, data[0], data[1], name=name, noise_group=noise_group, ootmask=ootm)

    def __repr__(self) -> str:
        return f"TSData Name:'{self.name}' [{self.wavelength[0]:.2f} - {self.wavelength[-1]:.2f}] nwl={self.nwl} npt={self.npt}"

    @property
    def noise_group(self) -> str:
        return self._noise_group

    @noise_group.setter
    def noise_group(self, ng: str) -> None:
        self._noise_group = ng
        if self._dataset is not None:
            self._dataset._update_nids()

    def mask_transit(self, t0: float | None = None, p: float | None = None, t14: float | None = None,
                     elims: tuple[int, int] | None = None) -> None:
        if t0 and p and t14:
            self.ephemeris = Ephemeris(t0, p, t14)
            phase = fold(self.time, p, t0)
            self.ootmask = abs(phase) > 0.502 * t14
        elif elims is not None:
            self.ootmask = ones(self.fluxes.shape, bool)
            self.ootmask[:, elims[0]:elims[1]] = False
        else:
            raise ValueError("Transit masking requires either t0, pp, and t14, or transit limits in exposure indices.")

    def calculate_ootmask(self, t0: float, p: float, t14: float):
        phase = fold(self.time, p, t0)
        self.ootmask = abs(phase) > 0.502 * t14

    def _update(self) -> None:
        """Update the internal attributes."""
        self.nwl = self.wavelength.size
        self.npt = self.time.size
        self.wllims = self.wavelength.min(), self.wavelength.max()

    @deprecated(0.9, alternative='normalize_to_poly')
    def normalize_baseline(self, deg: int = 1) -> None:
        return self.normalize_to_poly(deg)

    def normalize_to_poly(self, deg: int = 1) -> None:
        """Normalize the baseline flux for each spectroscopic light curve.

        Normalize the baseline flux using a low-order polynomial fitted to the out-of-transit
        data for each spectroscopic light curve.

        Parameters
        ----------
        deg
            The degree of the fitted polynomial. Should be 0 or 1. Higher degrees are not allowed
            because they could affect the transit depths.

        Raises
        ------
        ValueError
            If `deg` is greater than 1.

        Notes
        -----
        This method normalizes the baseline of the fluxes for each planet. It fits a polynomial of degree
        `deg` to the out-of-transit data points and divides the fluxes by the fitted polynomial evaluated
        at each time point.
        """
        if deg > 1:
            raise ValueError("The degree of the fitted polynomial ('deg') should be 0 or 1. Higher degrees "
                             "are not allowed because they could affect the transit depths.")

        if self.ootmask is None:
            raise ValueError("The out-of-transit mask must be defined for normalization. "
                             "Call TSData.mask_transit(...) first.")

        for ipb in range(self.nwl):
            bl = poly1d(polyfit(self.time[self.ootmask], self.fluxes[ipb, self.ootmask], deg=deg))(self.time)
            self.fluxes[ipb, :] /= bl
            self.errors[ipb, :] /= bl

    @deprecated(0.9, alternative='normalize_to_median')
    def normalize_median(self, s: slice) -> None:
        """Normalize the light curves to the median flux of the given slice along the time axis.

        Parameters
        ----------
        s : slice
            A slice object representing the portion of the data to normalize.
        """
        self.normalize_to_median(s)

    def normalize_to_median(self, s: slice) -> None:
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
        t1 = TSData(time=self.time[m1], wavelength=self.wavelength, fluxes=self.fluxes[:, m1], errors=self.errors[:, m1],
                    noise_group=self.noise_group, ootmask=self.ootmask, ephemeris=self.ephemeris)
        t2 = TSData(time=self.time[m2], wavelength=self.wavelength, fluxes=self.fluxes[:, m2], errors=self.errors[:, m2],
                    noise_group=self.noise_group, ootmask=self.ootmask, ephemeris=self.ephemeris)
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
        d = TSData(time=self.time[m], wavelength=self.wavelength, fluxes=self.fluxes[:, m], errors=self.errors[:, m],
                   name=f'{self.name}_1', noise_group=self.noise_group, ootmask=self.ootmask, ephemeris=self.ephemeris)
        for i, m in enumerate(masks[1:]):
            d = d + TSData(time=self.time[m], wavelength=self.wavelength,
                           fluxes=self.fluxes[:, m], errors=self.errors[:, m],
                           name=f'{self.name}_{i+2}', noise_group=self.noise_group,
                           ootmask=self.ootmask, ephemeris=self.ephemeris)
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

    def crop_time(self, tmin: float, tmax: float) -> None:
        """Crop the data to include only the time range between lmin and lmax.

        Parameters
        ----------
        tmin
            The minimum time value to crop.
        tmax
            The maximum time value to crop.

        Note
        ----
        The data will be modified in place.
        """
        m = (self.time > tmin) & (self.time < tmax)
        self.time = self.time[m]
        self.fluxes = self.fluxes[:, m]
        self.errors = self.errors[:, m]
        self._tm_l_edges = self._tm_l_edges[m]
        self._tm_r_edges = self._tm_r_edges[m]
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

    def plot(self, ax=None, vmin: float = None, vmax: float = None, cmap=None, figsize=None, data=None,
             plims: tuple[float, float] | None = None) -> Figure:
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

        plims
            Percentile flux limits. Overrides vmin and vmax.

        Returns
        -------
        Figure

        """
        if ax is None:
            fig, ax = subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.figure
        tref = floor(self.time.min())

        def forward_x(x):
            return interp(x, self.wavelength, arange(self.nwl))
        def inverse_x(x):
            return interp(x, arange(self.nwl), self.wavelength)
        def forward_y(y):
            return interp(y, self.time-tref, arange(self.npt))
        def inverse_y(y):
            return interp(y, arange(self.npt), self.time-tref)

        data = data if data is not None else self.fluxes
        if plims is not None:
            vmin, vmax = nanpercentile(data, plims)

        ax.pcolormesh(self.time - tref, self.wavelength, data, vmin=vmin, vmax=vmax, cmap=cmap)

        if self.ephemeris is not None:
            [ax.axvline(tl-tref, ls='--', c='k') for tl in self.ephemeris.transit_limits(self.time.mean())]

        setp(ax, ylabel=r'Wavelength [$\mu$m]', xlabel=f'Time - {tref:.0f} [BJD]')
        ax.yaxis.set_major_locator(LinearLocator(10))
        ax.yaxis.set_major_formatter('{x:.2f}')

        if self.name != "":
            ax.set_title(self.name)

        axx2 = ax.secondary_yaxis('right', functions=(forward_x, inverse_x))
        axx2.set_ylabel('Light curve index')
        axx2.set_yticks(forward_x(ax.get_yticks()))
        axx2.yaxis.set_major_formatter('{x:.0f}')
        axy2 = ax.secondary_xaxis('top', functions=(forward_y, inverse_y))
        axy2.set_xlabel('Exposure index')
        axy2.set_xticks(forward_y(ax.get_xticks()))
        axy2.xaxis.set_major_formatter('{x:.0f}')
        fig.axx2 = axx2
        fig.axy2 = axy2
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

    def __add__(self, other: Union['TSData', 'TSDataSet']) -> 'TSDataSet':
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
        if isinstance(other, TSData):
            return TSDataSet([self, other])
        else:
            return TSDataSet([self]) + other

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
                          name=self.name, tm_edges=(self._tm_l_edges, self._tm_r_edges), noise_group=self.noise_group,
                          ootmask=self.ootmask, ephemeris=self.ephemeris)


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
                          name=self.name, noise_group=self.noise_group,
                          ootmask=self.ootmask, ephemeris=self.ephemeris)


class TSDataSet:
    """A high-level data storage class that can contain multiple TSData objects."""
    def __init__(self, data: Sequence[TSData]):
        self.data: list[TSData] = []
        self.wlmin: float = inf
        self.wlmax: float = -inf
        self.tmin: float = inf
        self.tmax: float = -inf
        self.ngids: ndarray = array([])
        for d in data:
            self._add_data(d)

    def _add_data(self, d: TSData) -> None:
        if d.name in self.names:
            raise ValueError('A TSData object with the same name already exists.')
        d._dataset = self
        self.data.append(d)
        self._update_nids()
        self.wlmin = min(self.wlmin, d.wavelength.min())
        self.wlmax = max(self.wlmax, d.wavelength.max())
        self.tmin = min(self.tmin, d.time.min())
        self.tmax = max(self.tmax, d.time.max())

    def _update_nids(self):
        ngs =  pd.Categorical(self.noise_groups)
        self.unique_noise_groups = list(ngs.categories)
        self.ngids = ngs.codes.astype(int)
        for i,d in enumerate(self.data):
            d.ngid = self.ngids[i]

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.data]

    @property
    def times(self) -> list[ndarray]:
        return [d.time for d in self.data]

    @property
    def wavelengths(self) -> list[ndarray]:
        return [d.wavelength for d in self.data]

    @property
    def fluxes(self) -> list[ndarray]:
        return [d.fluxes for d in self.data]

    @property
    def errors(self) -> list[ndarray]:
        return [d.errors for d in self.data]

    @property
    def noise_groups(self) -> list[str]:
        return [d.noise_group for d in self.data]

    @property
    def n_noise_groups(self) -> int:
        return len(set(self.noise_groups))

    @property
    def size(self) -> int:
        return len(self.data)

    def export_fits(self) -> pf.HDUList:
        ds = pf.ImageHDU(name=f'dataset')
        ds.header['ndata'] = self.size
        for i,n in enumerate(self.names):
            ds.header[f'name_{i}'] = n

        hdul = pf.HDUList([ds])
        for d in self.data:
            hdul += d.export_fits()
        return hdul

    @staticmethod
    def import_fits(hdul) -> 'TSDataSet':
        ds = hdul['DATASET']
        data = []
        for i in range(ds.header['NDATA']):
            name = ds.header[f'NAME_{i}']
            data.append(TSData.import_fits(name, hdul))
        return TSDataSet(data)

    def calculate_ootmask(self, tc: float, p: float, t14: float):
        for d in self.data:
            d.calculate_ootmask(tc, p, t14)

    def __getitem__(self, index: int) -> TSData:
        return self.data[index]

    def __len__(self) -> int:
        return self.size

    def __repr__(self):
        return f"TSDataSet with {self.size} groups"

    def plot(self, axs=None, vmin: float = None, vmax: float = None, ncols: int = 1, cmap=None, figsize=None, data: ndarray | None = None):
        """Plot all the data sets.

        Parameters
        ----------
        axs
            A 2D ndarray of Axes used for plotting. If None, a new set of subplots will be created.
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
        if axs is None:
            nrows = int(ceil(self.size / ncols))
            fig, axs = subplots(nrows, ncols=ncols, figsize=figsize, squeeze=False)
        else:
            axs = atleast_2d(axs)
            fig = axs.flat[0].get_figure()

        if data is None:
            for i in range(self.size):
                self.data[i].plot(ax=axs.flat[i], vmin=vmin, vmax=vmax, cmap=cmap)
        else:
            for i in range(self.size):
                self.data[i].plot(ax=axs.flat[i], vmin=vmin, vmax=vmax, cmap=cmap, data=data[i])

        setp(axs[:-1, :], xlabel='')
        return fig

    def __add__(self, other):
        if isinstance(other, TSData):
            return TSDataSet(self.data + [other])
        elif isinstance(other, TSDataSet):
            return TSDataSet(self.data + other.data)