from astropy.stats import mad_std
from matplotlib.pyplot import subplots
from numpy import isfinite, median, where, concatenate, all
from scipy.ndimage import median_filter

from pytransit.utils import downsample_time_2d

class TSData:
    def __init__(self, time, wavelength, fluxes, errors):
        if fluxes.shape[0] != wavelength.size:
            raise ValueError()
        m = all(isfinite(fluxes), axis=1)
        self.time = time.copy()
        self.wavelength = wavelength[m]
        self.fluxes = fluxes[m]
        self.errors = errors[m]
        self._update()

    def crop_wavelength(self, lmin: float, lmax: float):
        m = (self.wavelength > lmin) & (self.wavelength < lmax)
        self.wavelength = self.wavelength[m]
        self.fluxes = self.fluxes[m]
        self.errors = self.errors[m]
        self._update()

    def remove_outliers(self, sigma: float = 5.0):
        fm = median(self.fluxes, axis=0)
        fe = mad_std(self.fluxes, axis=0)
        self.fluxes = where(abs(self.fluxes - fm) / fe < sigma, self.fluxes, median_filter(self.fluxes, 5))

    def _update(self):
        self.nwl = self.wavelength.size
        self.npt = self.time.size
        self.wllims = self.wavelength[[0, -1]]

    def plot(self, ax=None):
        if ax is None:
            fig, ax = subplots()
        ax.pcolormesh(self.time, self.wavelength, self.fluxes, )
        return ax

    def __add__(self, other):
        if self.wllims[1] > other.wllims[0]:
            raise ValueError('The wavelength ranges should not overlap.')
        fluxes = concatenate([self.fluxes, other.fluxes])
        errors = concatenate([self.errors, other.errors])
        wavelength = concatenate([self.wavelength, other.wavelength])
        return TSData(self.time, wavelength, fluxes, errors)

    def downsample(self, dwl: float, wlmin=None, wlmax=None):
        bwl, bfl, bfe = downsample_time_2d(self.wavelength, self.fluxes, inttime=dwl * 1e-3, tmin=wlmin, tmax=wlmax)
        return TSData(self.time, bwl, bfl, bfe)
