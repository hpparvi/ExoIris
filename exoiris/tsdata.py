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
from numpy import isfinite, median, where, all, zeros_like, diff, asarray, interp, arange, floor, ndarray, \
    ceil, newaxis, inf, array, ones, poly1d, polyfit, nanpercentile, atleast_2d, nan, linspace, any
from pytransit.orbits import fold
from scipy.ndimage import median_filter
from scipy.signal import medfilt

from .ephemeris import Ephemeris
from .util import bin2d
from .binning import Binning, CompoundBinning

class TSData:
    """
    `TSData` is a utility class representing transmission spectroscopy time series data with associated wavelength,
    fluxes, and errors. It provides methods for manipulating and analyzing the data.
    """
    def __init__(self, time: Sequence, wavelength: Sequence, fluxes: Sequence, errors: Sequence, name: str,
                 noise_group: str = 'a', wl_edges : Sequence | None = None, tm_edges : Sequence | None = None,
                 ootmask: ndarray | None = None, ephemeris: Ephemeris | None = None, n_baseline: int = 1) -> None:
        """
        Parameters
        ----------
        time
            1D Array of time values.
        wavelength
            1D Array of wavelength values.
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

        if any(~isfinite(fluxes)) or any(~isfinite(errors)):
            raise ValueError("Fluxes or errors cannot have nonfinite values.")

        if fluxes.shape[0] != wavelength.size:
            raise ValueError("The size of the flux array's first axis must match the size of the wavelength array.")

        if ootmask is not None and ootmask.size != time.size:
            raise ValueError("The size of the out-of-transit mask array must match the size of the time array.")

        if n_baseline < 1:
            raise ValueError("n_baseline must be greater than zero.")

        m = all(isfinite(fluxes), axis=1)
        self.name: str = name
        self.time: ndarray = time.copy()
        self.wavelength: ndarray = wavelength[m]
        self.fluxes: ndarray = fluxes[m]
        self.errors: ndarray = errors[m]
        self.ootmask: ndarray = ootmask if ootmask is not None else ones(time.size, dtype=bool)
        self.ngid: int = 0
        self.ephemeris: Ephemeris | None = ephemeris
        self.n_baseline: int = n_baseline
        self._noise_group: str = noise_group
        self._dataset: Optional['TSDataSet'] = None
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
        """Generate a `~astropy.io.fits.HDUList` containing HDUs storing the data and metadata.

            Returns
            -------
            ~astropy.io.fits.HDUList
        """
        time = pf.ImageHDU(self.time, name=f'time_{self.name}')
        wave = pf.ImageHDU(self.wavelength, name=f'wave_{self.name}')
        data = pf.ImageHDU(array([self.fluxes, self.errors]), name=f'data_{self.name}')
        ootm = pf.ImageHDU(self.ootmask.astype(int), name=f'ootm_{self.name}')
        data.header['ngroup'] = self.noise_group
        data.header['nbasel'] = self.n_baseline
        #TODO: export ephemeris
        return pf.HDUList([time, wave, data, ootm])

    @staticmethod
    def import_fits(name: str, hdul: pf.HDUList) -> 'TSData':
        """Import a data set from a `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        name
            The name of the dataset to be imported from the `~astropy.io.fits.HDUList`.
        hdul
            The `~astropy.io.fits.HDUList` containing the data.

        Returns
        -------
        TSData
        """
        time = hdul[f'TIME_{name}'].data.astype('d')
        wave = hdul[f'WAVE_{name}'].data.astype('d')
        data = hdul[f'DATA_{name}'].data.astype('d')
        ootm = hdul[f'OOTM_{name}'].data.astype(bool)
        noise_group = hdul[f'DATA_{name}'].header['NGROUP']

        try:
            n_baseline = hdul[f'DATA_{name}'].header['NBASEL']
        except KeyError:
            n_baseline = 1

        #TODO: import ephemeris
        return TSData(time, wave, data[0], data[1], name=name, noise_group=noise_group, ootmask=ootm,
                      n_baseline=n_baseline)

    def __repr__(self) -> str:
        return f"TSData Name:'{self.name}' [{self.wavelength[0]:.2f} - {self.wavelength[-1]:.2f}] nwl={self.nwl} npt={self.npt}"

    @property
    def noise_group(self) -> str:
        """Noise group name."""
        return self._noise_group

    @noise_group.setter
    def noise_group(self, ng: str) -> None:
        self._noise_group = ng
        if self._dataset is not None:
            self._dataset._update_nids()

    def mask_transit(self, t0: float | None = None, p: float | None = None, t14: float | None = None,
                     ephemeris : Ephemeris | None = None, elims: tuple[int, int] | None = None) -> 'TSData':
        """Create a transit mask based on a given ephemeris or exposure index limits.

        Parameters
        ----------
        t0
            The zero-epoch time.
        p
            The orbital period of the planet.
        t14
            The duration of the full transit in days.
        ephemeris
            The ephemeris object containing transit timing information.
        elims
            The limits of the region to mask in exposure indices.
        """
        if (t0 and p and t14) or ephemeris is not None:
            if ephemeris is not None:
                self.ephemeris = ephemeris
            else:
                self.ephemeris = Ephemeris(t0, p, t14)
            phase = fold(self.time, self.ephemeris.period, self.ephemeris.zero_epoch)
            self.ootmask = abs(phase) > 0.502 * self.ephemeris.duration
        elif elims is not None:
            self.ootmask = ones(self.fluxes.shape, bool)
            self.ootmask[:, elims[0]:elims[1]] = False
        else:
            raise ValueError("Transit masking requires either t0, pp, and t14, ephemeris, or transit limits in exposure indices.")
        return self

    @deprecated('0.9', alternative='mask_transit')
    def calculate_ootmask(self, t0: float | None = None, p: float | None = None, t14: float | None = None):
        self.ephemeris = Ephemeris(t0, p, t14)
        phase = fold(self.time, p, t0)
        self.ootmask = abs(phase) > 0.502 * t14

    def _update(self) -> None:
        """Update the internal attributes."""
        self.nwl = self.wavelength.size
        self.npt = self.time.size
        self.wllims = self.wavelength.min(), self.wavelength.max()

    @deprecated('0.9', alternative='normalize_to_poly')
    def normalize_baseline(self, deg: int = 1) -> 'TSData':
        return self.normalize_to_poly(deg)

    def normalize_to_poly(self, deg: int = 1) -> 'TSData':
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
        return self

    @deprecated('0.9', alternative='normalize_to_median')
    def normalize_median(self, s: slice) -> None:
        """Normalize the light curves to the median flux of the given slice along the time axis.

        Parameters
        ----------
        s : slice
            A slice object representing the portion of the data to normalize.
        """
        self.normalize_to_median(s)

    def normalize_to_median(self, s: slice) -> 'TSData':
        """Normalize the light curves to the median flux of the given slice along the time axis.

        Parameters
        ----------
        s
            A slice object representing the portion of the data to normalize.
        """
        n = median(self.fluxes[:, s], axis=1)[:, newaxis]
        self.fluxes[:,:] /= n
        self.errors[:,:] /= n
        return self

    @deprecated('0.9', alternative='partition_time')
    def split_time(self, t: float, b: float) -> 'TSDataSet':
        """Split the data into two parts: (time < t-b) and (time > t+b).

        Parameters
        ----------
        t
            The threshold time value used to split the data.
        b
            The buffer time around the threshold `t` to exclude from the split range.
        """
        m1 = self.time < t-b
        m2 = self.time > t+b
        t1 = TSData(name=f'{self.name}_a', time=self.time[m1], wavelength=self.wavelength, fluxes=self.fluxes[:, m1], errors=self.errors[:, m1],
                    noise_group=self.noise_group, ootmask=self.ootmask[m1], ephemeris=self.ephemeris, n_baseline=self.n_baseline)
        t2 = TSData(name=f'{self.name}_b', time=self.time[m2], wavelength=self.wavelength, fluxes=self.fluxes[:, m2], errors=self.errors[:, m2],
                    noise_group=self.noise_group, ootmask=self.ootmask[m2], ephemeris=self.ephemeris, n_baseline=self.n_baseline)
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
        d = TSData(name=f'{self.name}_1', time=self.time[m], wavelength=self.wavelength, fluxes=self.fluxes[:, m],
                   errors=self.errors[:, m], noise_group=self.noise_group, ootmask=self.ootmask[m], ephemeris=self.ephemeris)
        for i, m in enumerate(masks[1:]):
            d = d + TSData(name=f'{self.name}_{i+2}', time=self.time[m], wavelength=self.wavelength,
                           fluxes=self.fluxes[:, m], errors=self.errors[:, m],
                           noise_group=self.noise_group,
                           ootmask=self.ootmask[m], ephemeris=self.ephemeris,
                           n_baseline=self.n_baseline)
        return d

    def crop_wavelength(self, lmin: float, lmax: float) -> 'TSData':
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
        return self

    def crop_time(self, tmin: float, tmax: float) -> 'TSData':
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
        self.ootmask = self.ootmask[m]
        self._tm_l_edges = self._tm_l_edges[m]
        self._tm_r_edges = self._tm_r_edges[m]
        self._update()
        return self

    def remove_outliers(self, sigma: float = 5.0) -> 'TSData':
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
        return self

    def remove_outliers_along_wavelength(self, sigma: float = 5.0, filter_width: int = 9, min_flux: float = 1e-6,
                                         plot: bool = True, ax = None, figsize=None):

        mean_flux = self.fluxes.mean(1)
        filtered_flux = medfilt(mean_flux, filter_width)
        residuals = mean_flux - filtered_flux
        residual_std = mad_std(residuals)

        mask_good = abs(residuals) < sigma*residual_std
        mask_good &= mean_flux > min_flux

        if plot:
            if ax is None:
                fig, ax = subplots(figsize=figsize, constrained_layout=True)
            else:
                fig = ax.figure
            ax.plot(self.wavelength, mean_flux, c='0.8')
            ax.plot(self.wavelength, where(mask_good, mean_flux, nan))
            ax.plot(self.wavelength, filtered_flux+5*mad_std(residuals), '--', c='0.85')
            ax.plot(self.wavelength, filtered_flux-5*mad_std(residuals), '--', c='0.85')
            setp(ax, xlabel='wavelength', ylabel='Flux', xlim=self.wavelength[[0,-1]])

        self.wavelength = self.wavelength[mask_good]
        self.fluxes = self.fluxes[mask_good, :]
        self.errors = self.errors[mask_good, :]
        self._wl_l_edges = self._wl_l_edges[mask_good]
        self._wl_r_edges = self._wl_r_edges[mask_good]
        self._update()

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
        ~matplotlib.figure.Figure
        """
        if ax is None:
            fig, ax = subplots(figsize=figsize, constrained_layout=True)
        else:
            fig = ax.figure
        tref = floor(self.time.min())

        def forward_y(y):
            return interp(y, self.wavelength, arange(self.nwl))
        def inverse_y(y):
            return interp(y, arange(self.nwl), self.wavelength)
        def forward_x(x):
            return interp(x, self.time-tref, arange(self.npt))
        def inverse_x(x):
            return interp(x, arange(self.npt), self.time-tref)

        data = data if data is not None else self.fluxes
        if plims is not None:
            vmin, vmax = nanpercentile(data, plims)

        ax.pcolormesh(self.time - tref, self.wavelength, data, vmin=vmin, vmax=vmax, cmap=cmap)

        if self.ephemeris is not None:
            [ax.axvline(tl-tref, ls='--', c='k') for tl in self.ephemeris.transit_limits(self.time.mean())]

        setp(ax, ylabel=r'Wavelength [$\mu$m]', xlabel=f'Time - {tref:.0f} [BJD]')
        ax.yaxis.set_major_locator(LinearLocator(10))
        ax.yaxis.set_major_formatter('{x:.2f}')
        ax.xaxis.set_major_locator(LinearLocator())
        ax.xaxis.set_major_formatter('{x:.3f}')

        if self.name != "":
            ax.set_title(self.name)

        axy2 = ax.secondary_yaxis('right', functions=(forward_y, inverse_y))
        axy2.set_ylabel('Light curve index')
        axy2.set_yticks(forward_y(ax.get_yticks()))
        axy2.yaxis.set_major_formatter('{x:.0f}')
        axx2 = ax.secondary_xaxis('top', functions=(forward_x, inverse_x))
        axx2.set_xlabel('Exposure index')
        axx2.xaxis.set_major_locator(LinearLocator())
        axx2.xaxis.set_major_formatter('{x:.0f}')
        fig.axx2 = axx2
        fig.axy2 = axy2
        return fig

    def plot_white(self, ax: Axes | None = None, figsize: tuple[float, float] | None = None) -> Figure:
        """Plot a white light curve.

        Parameters
        ----------
        ax
            The axes on which to plot. If None, a new figure and axes are created.
        figsize
            The size of the figure to create if `ax` is None. It should be a tuple in the format (width, height).

        Returns
        -------
        ~matplotlib.figure.Figure
        """
        if ax is None:
            fig, ax = subplots(figsize=figsize)
        else:
            fig = ax.figure
        tref = floor(self.time.min())

        ax.plot(self.time, self.fluxes.mean(0))
        if self.ephemeris is not None:
            [ax.axvline(tl, ls='--', c='k') for tl in self.ephemeris.transit_limits(self.time.mean())]

        if self.name != "":
            ax.set_title(self.name)

        def forward_x(x):
            return interp(x, self.time, arange(self.npt))
        def inverse_x(x):
            return interp(x, arange(self.npt), self.time)

        axx2 = ax.secondary_xaxis('top', functions=(forward_x, inverse_x))
        axx2.set_xlabel('Exposure index')
        axx2.xaxis.set_major_locator(LinearLocator())
        axx2.xaxis.set_major_formatter('{x:.0f}')

        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p: f"{x-tref:.3f}"))
        setp(ax, xlabel=f'Time - {tref:.0f} [BJD]', ylabel='Normalized flux', xlim=[self.time[0]-0.003, self.time[-1]+0.003])
        return fig

    def plot_baseline(self, ax: Axes | None = None, figsize: tuple[float, float] | None = None) -> Figure:
        """Plot the out-of-transit spectroscopic light curves before and after the normalization.

        Parameters
        ----------
        ax
            The axes on which to plot. If None, a new figure and axes are created.
        figsize
            The size of the figure to create if `ax` is None. It should be a tuple in the format (width, height).

        Returns
        -------
        ~matplotlib.figure.Figure
        """
        return self.plot(ax=ax, figsize=figsize, data=where(self.ootmask, self.fluxes, nan))

    def __add__(self, other: Union['TSData', 'TSDataSet']) -> 'TSDataSet':
        """Combine two transmission spectra along the wavelength axis.

        Parameters
        ----------
        other
            The TSData object to be added to the current TSData object.

        Returns
        -------
        TSDataSet
        """
        if isinstance(other, TSData):
            return TSDataSet([self, other])
        else:
            return TSDataSet([self]) + other

    def bin_wavelength(self, binning: Optional[Union[Binning, CompoundBinning]] = None,
                       nb: Optional[int] = None, bw: Optional[float] = None, r: Optional[float] = None,
                       estimate_errors: bool = False) -> 'TSData':
        """Bin the data along the wavelength axis.

        Bin the data along the wavelength axis. If binning is not specified, a Binning object is created using the
        minimum and maximum values of the wavelength.

        Parameters
        ----------
        binning
            The binning method to use.
        nb
            Number of bins.
        bw
            Bin width.
        r
            Bin resolution.
        estimate_errors
            Should the uncertainties be estimated from the data.

        Returns
        -------
        TSData
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
                          ootmask=self.ootmask, ephemeris=self.ephemeris, n_baseline=self.n_baseline)


    def bin_time(self, binning: Optional[Union[Binning, CompoundBinning]] = None,
                       nb: Optional[int] = None, bw: Optional[float] = None,
                       estimate_errors: bool = False) -> 'TSData':
        """Bin the data along the time axis.

        Bin the data along the time axis. If binning is not specified, a Binning object is created using the
        minimum and maximum time values.

        Parameters
        ----------
        binning
            The binning method to use.
        nb
            Number of bins.
        bw
            Bin width in seconds.
        estimate_errors
            Should the uncertainties be estimated from the data.

        Returns
        -------
        TSData
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numba.NumbaPerformanceWarning)
            if binning is None:
                binning = Binning(self.time.min(), self.time.max(), nb=nb, bw=bw/(24*60*60) if bw is not None else None)
            bf, be = bin2d(self.fluxes.T, self.errors.T, self._tm_l_edges, self._tm_r_edges,
                           binning.bins, estimate_errors=estimate_errors)
            d = TSData(binning.bins.mean(1), self.wavelength, bf.T, be.T,
                          wl_edges=(self._wl_l_edges, self._wl_r_edges),
                          tm_edges=(binning.bins[:,0], binning.bins[:,1]),
                          name=self.name, noise_group=self.noise_group,
                          ephemeris=self.ephemeris, n_baseline=self.n_baseline)
            if self.ephemeris is not None:
                d.mask_transit(ephemeris=self.ephemeris)
            return d

class TSDataSet:
    """`TSDataSet` is a high-level data storage class that can contain multiple `TSData` objects.
    """
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
        """List of data set names."""
        return [d.name for d in self.data]

    @property
    def times(self) -> list[ndarray]:
        """List of 1D time arrays."""
        return [d.time for d in self.data]

    @property
    def wavelengths(self) -> list[ndarray]:
        """List of 1D wavelength arrays."""
        return [d.wavelength for d in self.data]

    @property
    def fluxes(self) -> list[ndarray]:
        """List of 2D flux arrays."""
        return [d.fluxes for d in self.data]

    @property
    def errors(self) -> list[ndarray]:
        """List of 2D error arrays."""
        return [d.errors for d in self.data]

    @property
    def noise_groups(self) -> list[str]:
        """List of noise group names."""
        return [d.noise_group for d in self.data]

    @property
    def n_noise_groups(self) -> int:
        """Number of noise groups."""
        return len(set(self.noise_groups))

    @property
    def n_baselines(self) -> list[int]:
        """Number of baseline coefficients for each data set."""
        return [d.n_baseline for d in self.data]

    @property
    def size(self) -> int:
        """Number of data sets."""
        return len(self.data)

    def export_fits(self) -> pf.HDUList:
        """Export the dataset along with its metadata to a FITS HDU list.

        Returns
        -------
        ~astropy.io.fits.HDUList
        """
        ds = pf.ImageHDU(name=f'dataset')
        ds.header['ndata'] = self.size
        for i,n in enumerate(self.names):
            ds.header[f'name_{i}'] = n

        hdul = pf.HDUList([ds])
        for d in self.data:
            hdul += d.export_fits()
        return hdul

    @staticmethod
    def import_fits(hdul: pf.HDUList) -> 'TSDataSet':
        """Import all the data from a FITS HDU list.

        Parameters
        ----------
        hdul
            HDU list containing FITS data.

        Returns
        -------
        TSDataSet
        """
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

    def plot(self, axs=None, vmin: float = None, vmax: float = None, ncols: int = 1, cmap=None, figsize=None, data: ndarray | None = None) -> Figure:
        """Plot all the data sets.

        Parameters
        ----------
        axs
            A 2D ndarray of Axes used for plotting. If None, a new set of subplots will be created.
        vmin
            The minimum value for the color mapping.
        vmax
            The maximum value for the color mapping.
        ncols
            The number of columns in the subplot grid.
        cmap
            The colormap used for mapping the data values to colors.
        figsize
            The size of the figure created if `ax` is None.
        data
            The data to be plotted. If None, the `self.data` attribute will be used.

        Returns
        -------
        ~matplotlib.figure.Figure
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

    def plot_white(self, axs=None, ncols: int = 1, figsize=None) -> Figure:
        """Plot the white light curves.

        Parameters
        ----------
        axs
            A 2D ndarray of Axes used for plotting. If None, a new set of subplots will be created.
        ncols
            The number of columns in the subplot grid.
        figsize
            The size of the figure created if `ax` is None.

        Returns
        -------
        ~matplotlib.figure.Figure
        """
        if axs is None:
            nrows = int(ceil(self.size / ncols))
            fig, axs = subplots(nrows, ncols=ncols, figsize=figsize, squeeze=False)
        else:
            axs = atleast_2d(axs)
            fig = axs.flat[0].get_figure()
        for i in range(self.size):
            self.data[i].plot_white(ax=axs.flat[i])
        setp(axs[:-1, :], xlabel='')
        return fig

    def __add__(self, other):
        if isinstance(other, TSData):
            return TSDataSet(self.data + [other])
        elif isinstance(other, TSDataSet):
            return TSDataSet(self.data + other.data)