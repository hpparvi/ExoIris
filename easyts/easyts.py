from typing import Optional

import seaborn as sb

from matplotlib.pyplot import subplots, setp
from numpy import poly1d, polyfit, where, sqrt, clip, percentile, median, squeeze, floor, interp, linspace, ndarray
from numpy.random import normal
from pytransit.orbits import as_from_rhop, i_from_ba, fold, i_from_baew, d_from_pkaiews, epoch

from .tsdata import TSData
from .wlpf import WhiteLPF
from .tslpf import TSLPF

class EasyTS:
    def __init__(self, name: str, ldmodel, data: TSData, nk: int = None, nbl: int = None, nldc: int = None,
                 nthreads: int = 1, tmpars=None):

        self.data = data
        self._tsa = TSLPF(name, ldmodel, data.time, data.wavelength, data.fluxes, data.errors, nk=nk, nbl=nbl,
                          nldc=nldc, nthreads=nthreads, tmpars=tmpars)
        self._wa = None

        self.nthreads = nthreads
        self.wavelength = data.wavelength
        self.nk = self._tsa.nk
        self.nbl = self._tsa.nbl

        self.name = name
        self.time = self._tsa.time
        self.fluxes = self._tsa.flux
        self.original_fluxes = self._tsa._original_flux
        self.ootmask = None
        self.npb = self._tsa.npb
        self.de = None
        self.sampler = None

        self._tref = floor(self.time.min())
        self._extent = (self.time[0] - self._tref, self.time[-1] - self._tref, self.wavelength[0], self.wavelength[-1])

    def lnposterior(self, pvp):
        return squeeze(self._tsa.lnposterior(pvp))

    def set_prior(self, parameter, prior, *nargs):
        self._tsa.set_prior(parameter, prior, *nargs)

    def set_radius_ratio_prior(self, prior, *nargs):
        for l in self._tsa.kx_knots:
            self.set_prior(f'k_{l:08.5f}', prior, *nargs)

    def set_ldtk_prior(self, teff, logg, metal, dataset: str = 'visir-lowres', width: float = 50, uncertainty_multiplier: float = 10):
        self._tsa.set_ldtk_prior(teff, logg, metal, dataset, width, uncertainty_multiplier)

    @property
    def k_knots(self) -> ndarray:
        return self._tsa.kx_knots

    def set_k_knots(self, knot_wavelengths):
        self.nk = len(knot_wavelengths)
        self._tsa.set_k_knots(knot_wavelengths)

    @property
    def ps(self):
        return self._tsa.ps

    def print_parameters(self):
        self._tsa.print_parameters(1)

    def plot_setup(self, figsize=(13,4)):
        fig, axs = subplots(3, 1, figsize=figsize, sharex='all', sharey='all')
        axs[0].vlines(self._tsa.ld_knots, 0.1, 0.5, ec='k')
        axs[0].text(0.01, 0.90, 'Limb darkening knots', va='top', transform=axs[0].transAxes)
        axs[1].vlines(self._tsa.kx_knots, 0.1, 0.5, ec='k')
        axs[1].text(0.01, 0.90, 'Radius ratio knots', va='top', transform=axs[1].transAxes)
        axs[2].vlines(self.wavelength, 0.1, 0.5, ec='k')
        axs[2].text(0.01, 0.90, 'Wavelength bins', va='top', transform=axs[2].transAxes)
        sb.despine(ax=axs[0], top=False, bottom=True, right=False)
        sb.despine(ax=axs[1], top=True, bottom=True, right=False)
        sb.despine(ax=axs[2], top=True, bottom=False, right=False)
        setp(axs, xlim=(self.wavelength[0]-0.02, self.wavelength[-1]+0.02), yticks=[], ylim=(0, 0.9))
        setp(axs[-1], xlabel=r'Wavelength [$\mu$m]')
        fig.tight_layout()
        return fig

    def fit_white(self):
        self._wa = WhiteLPF(self._tsa)
        self._wa.optimize()
        pv = self._wa._local_minimization.x
        phase = fold(self.time, pv[1], pv[0])
        self.ootmask = abs(phase) > 0.502 * self._wa.transit_duration

    def plot_white(self):
        return self._wa.plot()

    def normalize_baseline(self, deg: int = 1):
        for ipb in range(self.npb):
            pl = poly1d(polyfit(self.time[self.ootmask], self.fluxes[ipb, self.ootmask], deg=deg))
            self.fluxes[ipb, :] /= pl(self.time)

    def plot_baseline(self, axs=None):
        if axs is None:
            fig, axs = subplots(1, 2, figsize=(13, 4), sharey='all')
        else:
            fig = None
        axs[0].imshow(where(self.ootmask, self.original_fluxes, 1), aspect='auto', origin='lower', extent=self._extent)
        axs[1].imshow(where(self.ootmask, self.fluxes, 1), aspect='auto', origin='lower', extent=self._extent)
        setp(axs, xlabel=f'Time - {self._tref:.0f} [BJD]')
        setp(axs[0], ylabel='Wavelength [$\mu$m]')

        if fig is not None:
            fig.tight_layout()
        return axs

    def fit(self, niter: int = 200, npop: int = 150, pool=None, lnpost=None):
        if self._tsa.de is None:
            pv0 = self._wa._local_minimization.x
            pvp = self._tsa.ps.sample_from_prior(npop)
            pvp[:, 0] = normal(pv0[2], 0.05, size=npop)
            pvp[:, 1] = normal(pv0[0], 1e-4, size=npop)
            pvp[:, 2] = normal(pv0[1], 1e-5, size=npop)
            pvp[:, 3] = clip(normal(pv0[3], 0.01, size=npop), 0.0, 1.0)
            pvp[:, self._tsa._sl_rratios] = normal(sqrt(pv0[4]), 0.001, size=(npop, self.nk))
            pvp[:, self._tsa._sl_baseline] = normal(1.0, 1e-5, size=(npop, self.nbl))
        else:
            pvp = None
        self._tsa.optimize_global(niter=niter, npop=npop, population=pvp, pool=pool, lnpost=lnpost,
                                  vectorize=(pool is None))
        self.de = self._tsa.de

    def sample(self, niter: int = 500, thin: int = 10, repeats: int = 1, pool=None, lnpost=None, leave=True, save=False, use_tqdm: bool = True):
        self._tsa.sample_mcmc(niter=niter, thin=thin, repeats=repeats, pool=pool, lnpost=lnpost, vectorize=(pool is None), leave=leave, save=save, use_tqdm=use_tqdm)
        self.sampler = self._tsa.sampler

    def plot_transmission_spectrum(self, ax=None):
        if ax is None:
            fig, ax = subplots()

        if self._tsa.sampler is None:
            pv = self._tsa.de.minimum_location
            ar = 1e2 * squeeze(self._tsa._eval_k(pv[self._tsa._sl_rratios])) ** 2
            ax.plot(self.wavelength, ar, c='k')
            ax.plot(self._tsa.kx_knots, 1e2 * pv[self._tsa._sl_rratios] ** 2, 'ok')
        else:
            df = self._tsa.posterior_samples()
            ar = 1e2 * self._tsa._eval_k(df.iloc[:, self._tsa._sl_rratios]) ** 2
            ax.fill_between(self.wavelength, *percentile(ar, [16, 84], axis=0), alpha=0.25)
            ax.plot(self.wavelength, median(ar, 0), c='k')
        setp(ax, ylabel='Transit depth [%]', xlabel='Wavelength', xlim=self.wavelength[[0, -1]])
        return ax

    def plot_residuals(self, ax=None, pmin=1, pmax=99):
        if ax is None:
            fig, ax = subplots()
        else:
            fig = None

        tc = self._wa.transit_center
        td = self._wa.transit_duration

        fmodel = squeeze(self._tsa.flux_model(self._tsa.de.minimum_location))
        residuals = self.fluxes - fmodel
        pp = percentile(residuals, [pmin, pmax])
        ax.imshow(residuals, aspect='auto', vmin=pp[0], vmax=pp[1], origin='lower', extent=self._extent)

        for i in range(2):
            ax.axvline(tc + (-1) ** i * 0.5 * td - self._tref, c='w', ymax=0.05, lw=5)
            ax.axvline(tc + (-1) ** i * 0.5 * td - self._tref, c='w', ymin=0.95, lw=5)
            ax.axvline(tc + (-1) ** i * 0.5 * td - self._tref, c='k', ymax=0.05, lw=1)
            ax.axvline(tc + (-1) ** i * 0.5 * td - self._tref, c='k', ymin=0.95, lw=1)

        setp(ax, xlabel=f'Time - {self._tref:.0f} [BJD]', ylabel='Wavelength [$\mu$m]')
        if fig is not None:
            fig.tight_layout()
        return ax
