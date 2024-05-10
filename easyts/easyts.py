from matplotlib.pyplot import subplots, setp
from numpy import poly1d, polyfit, where, sqrt, clip, percentile, median, squeeze, floor, interp, linspace
from numpy.random import normal
from pytransit.orbits import as_from_rhop, i_from_ba, fold, i_from_baew, d_from_pkaiews, epoch

from .wlpf import WhiteLPF
from .tslpf import TSLPF

class EasyTS:
    def __init__(self, name: str, ldmodel, wavelength, time, fluxes, errors, nk: int = None, nbl: int = None,
                 nthreads: int = 1, tmpars=None):

        self._tsa = TSLPF(name, ldmodel, time, wavelength, fluxes, errors, nk=nk, nbl=nbl, nthreads=nthreads, tmpars=tmpars)
        self._wa = None

        self.nthreads = nthreads
        self.wavelength = wavelength
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

    def set_prior(self, parameter, prior, *nargs):
        self._tsa.set_prior(parameter, prior, *nargs)

    def set_radius_ratio_prior(self, prior, *nargs):
        for ipb in range(self.nk):
            self.set_prior(f'k_{ipb + 1:03d}', prior, *nargs)

    def print_parameters(self):
        self._tsa.print_parameters(1)

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
            pvp[:, 4] = normal(pv0[0], 1e-4, size=npop)
            pvp[:, 5] = normal(pv0[1], 1e-5, size=npop)
            pvp[:, 6] = clip(normal(pv0[3], 0.01, size=npop), 0.0, 1.0)
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
