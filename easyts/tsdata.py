from typing import Union, Optional

from astropy.stats import mad_std
from matplotlib.pyplot import subplots, setp
from matplotlib.ticker import LinearLocator
from numpy import isfinite, median, where, concatenate, all, zeros_like, diff, asarray, interp, arange, floor
from scipy.ndimage import median_filter

from .util import bin2d
from .binning import Binning, CompoundBinning

class TSData:
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

    def crop_wavelength(self, lmin: float, lmax: float):
        m = (self.wavelength > lmin) & (self.wavelength < lmax)
        self.wavelength = self.wavelength[m]
        self.fluxes = self.fluxes[m]
        self.errors = self.errors[m]
        self._wl_l_edges = self._wl_l_edges[m]
        self._wl_r_edges = self._wl_r_edges[m]
        self._update()

    def remove_outliers(self, sigma: float = 5.0):
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
        setp(ax, ylabel='Wavelength', xlabel=f'Time - {tref:.0f} [BJD]')
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

    def bin_wavelength(self, binning: Optional[Union[Binning, CompoundBinning]] = None, nb=None, bw=None, r=None):
        if binning is None:
            binning = Binning(self.wllims[0], self.wllims[1], nb=nb, bw=bw, r=r)
        bf, be = bin2d(self.fluxes, self.errors, self._wl_l_edges, self._wl_r_edges, binning.bins)
        return TSData(self.time, binning.bins.mean(1), bf, be, wl_edges=(binning.bins[:,0], binning.bins[:,1]))
