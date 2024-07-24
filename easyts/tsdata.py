import warnings
import numba
from typing import Union, Optional

from astropy.stats import mad_std
from matplotlib.pyplot import subplots, setp
from matplotlib.ticker import LinearLocator
from numpy import isfinite, median, where, concatenate, all, zeros_like, diff, asarray, interp, arange, floor
from scipy.ndimage import median_filter

from .util import bin2d
from .binning import Binning, CompoundBinning

class TSData:
    """
    TSData is a class representing transmission spectroscopy time series data with associated wavelength, fluxes, and
    errors. It provides methods for manipulating and analyzing the data.

    Parameters:
    - time : 1D array-like
        Array of time values.
    - wavelength: 1D array-like
        Array of wavelength values.
    - fluxes: 2D array-like
        2D array of flux values with a shape (nwl, npt), where nwl is the number of wavelengths and npt the
        number of exposures.
    - errors: 2D array-like
        2D Array of error values with a shape (nwl, npt), where nwl is the number of wavelengths and npt the
        number of exposures.
    - wl_edges : tuple of 1D array-like, optional
        Tuple containing left and right wavelength edges for each bin. Default is None.

    Methods:
    - crop_wavelength(lmin: float, lmax: float)
        Crop the data to include only the wavelength range between lmin and lmax. Updates the wavelength, fluxes, errors, and wavelength edges accordingly.

    - remove_outliers(sigma: float = 5.0)
        Remove outliers from the data. Outliers are defined as data points that deviate from the median by more than sigma times the median absolute deviation. Updates the fluxes using a median filter.

    - plot(ax=None, vmin: float = None, vmax: float = None, cmap=None, figsize=None)
        Plot the data as a pcolormesh plot. The time values are plotted on the x-axis, wavelength values on the y-axis, and the fluxes as the color map. Additional optional arguments can be used to customize the plot. Returns the matplotlib Axes objects.

    - __add__(self, other)
        Concatenate the data from two TSData objects along the wavelength axis. Returns a new TSData object with the concatenated data.

    - bin_wavelength(binning: Optional[Union[Binning, CompoundBinning]] = None, nb=None, bw=None, r=None)
        Bin the data along the wavelength axis. If binning is not specified, a Binning object is created using the minimum and maximum values of the wavelength. The binned fluxes and errors are computed using bin2d function. Returns a new TSData object with the binned data.
    """
    def __init__(self, time, wavelength, fluxes, errors, wl_edges = None):
        if fluxes.shape[0] != wavelength.size:
            raise ValueError()
        m = all(isfinite(fluxes), axis=1)
        self.time = time.copy()
        self.wavelength = wavelength[m]
        self.fluxes = fluxes[m]
        self.errors = errors[m]
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

    def crop_wavelength(self, lmin: float, lmax: float) -> None:
        """
        Crop the wavelength range of the spectrum.

        Parameters
        ----------
        lmin : float
            The minimum wavelength value to crop.

        lmax : float
            The maximum wavelength value to crop.

        """
        m = (self.wavelength > lmin) & (self.wavelength < lmax)
        self.wavelength = self.wavelength[m]
        self.fluxes = self.fluxes[m]
        self.errors = self.errors[m]
        self._wl_l_edges = self._wl_l_edges[m]
        self._wl_r_edges = self._wl_r_edges[m]
        self._update()

    def remove_outliers(self, sigma: float = 5.0):
        """Removes the outliers from the data.

        Parameters
        ----------
        sigma : float, optional
            The number of standard deviations to use as the threshold for outliers.
            Defaults to 5.0.

        Description
        -----------
        This method removes outliers from the 'fluxes' attribute of the object. An outlier is defined as
        a value that is more than sigma standard deviations away from the median of the fluxes.

        The method first calculates the median and the median absolute deviation (MAD) standard deviation
        of the fluxes along the wavelength axis. It then replaces the values that are more than sigma standard
        deviations away from the median with the median value. The replacement is done using a 5-point median filter.

        Note
        ----
        The 'fluxes' attribute of the object will be modified in place.

        """
        fm = median(self.fluxes, axis=0)
        fe = mad_std(self.fluxes, axis=0)
        self.fluxes = where(abs(self.fluxes - fm) / fe < sigma, self.fluxes, median_filter(self.fluxes, 5))

    def _update(self):
        self.nwl = self.wavelength.size
        self.npt = self.time.size
        self.wllims = self.wavelength[[0, -1]]

    def plot(self, ax=None, vmin: float = None, vmax: float = None, cmap=None, figsize=None):
        if ax is None:
            fig, ax = subplots(figsize=figsize)
        tref = floor(self.time.min())

        def forward(x):
            return interp(x, self.wavelength, arange(self.nwl))
        def inverse(x):
            return interp(x, arange(self.nwl), self.wavelength)

        ax.pcolormesh(self.time - tref, self.wavelength, self.fluxes, vmin=vmin, vmax=vmax, cmap=cmap)
        setp(ax, ylabel=r'Wavelength [$\mu$m]', xlabel=f'Time - {tref:.0f} [BJD]')
        ax.yaxis.set_major_locator(LinearLocator(10))
        ax.yaxis.set_major_formatter('{x:.2f}')

        ax2 = ax.secondary_yaxis('right', functions=(forward, inverse))
        ax2.set_ylabel('Light curve index')
        ax2.set_yticks(forward(ax.get_yticks()))
        ax2.yaxis.set_major_formatter('{x:.0f}')
        return ax, ax2

    def __add__(self, other):
        if self.wllims[1] > other.wllims[0]:
            raise ValueError('The wavelength ranges should not overlap.')
        fluxes = concatenate([self.fluxes, other.fluxes])
        errors = concatenate([self.errors, other.errors])
        wavelength = concatenate([self.wavelength, other.wavelength])
        wel = concatenate([self._wl_l_edges, other._wl_l_edges])
        wer = concatenate([self._wl_r_edges, other._wl_r_edges])
        return TSData(self.time, wavelength, fluxes, errors, wl_edges=(wel, wer))

    def bin_wavelength(self, binning: Optional[Union[Binning, CompoundBinning]] = None,
                       nb: Optional[int] = None, bw: Optional[float] = None, r: Optional[float] = None):
        """Bin the data along the wavelength axis.

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
            return TSData(self.time, binning.bins.mean(1), bf, be, wl_edges=(binning.bins[:,0], binning.bins[:,1]))
