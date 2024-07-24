from typing import Optional, Callable, Any

import seaborn as sb
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import subplots, setp
from numpy import poly1d, polyfit, where, sqrt, clip, percentile, median, squeeze, floor, ndarray, array
from numpy.random import normal
from pytransit.orbits import fold
from pytransit.param import ParameterSet

from .tsdata import TSData
from .tslpf import TSLPF
from .wlpf import WhiteLPF


class EasyTS:
    def __init__(self, name: str, ldmodel, data: TSData, nk: int = None, nbl: int = None, nldc: int = None,
                 nthreads: int = 1, tmpars=None):
        """
        Parameters
        ----------
        name : str
            The name of the instance.
        ldmodel : object
            The model for the limb darkening.
        data : TSData
            The time-series data object.
        nk : int, optional
            The number of kernel samples.
        nbl : int, optional
            The number of bins for the light curve.
        nldc : int, optional
            The number of limb darkening coefficients.
        nthreads : int, optional
            The number of threads to use for computation.
        tmpars : object, optional
            Additional parameters.

        """
        self.data = data
        self._tsa = TSLPF(name, ldmodel, data.time, data.wavelength, data.fluxes, data.errors, nk=nk, nbl=nbl,
                          nldc=nldc, nthreads=nthreads, tmpars=tmpars)
        self._wa = None
        self.nthreads = nthreads
        self.wavelength = data.wavelength

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
        """Log posterior probability for a single parameter vector or an array of parameter vectors.

        Parameters
        ----------
        pvp : array_like
            The vector of parameter values or an array of parameter vectors with a shape [npv, np].

        Returns
        -------
        array_like
            The natural logarithm of the posterior probability.

        """
        return squeeze(self._tsa.lnposterior(pvp))

    def set_prior(self, parameter, prior, *nargs):
        self._tsa.set_prior(parameter, prior, *nargs)

    def set_radius_ratio_prior(self, prior, *nargs):
        """Set an identical prior on all radius ratio (k) knots.

        Parameters
        ----------
        prior : float
            The prior for the radius ratios.
        *nargs : float
            Additional arguments for the prior.

        """
        for l in self._tsa.kx_knots:
            self.set_prior(f'k_{l:08.5f}', prior, *nargs)

    def set_ldtk_prior(self, teff, logg, metal, dataset: str = 'visir-lowres', width: float = 50, uncertainty_multiplier: float = 10):
        """Sets priors on the limb darkening parameters using LDTk.

        Sets priors on the limb darkening parameters based on theoretical stellar models using LDTk.

        Parameters
        ----------
        teff : float
            The effective temperature in Kelvin.

        logg : float
            The surface gravity in cm/s^2.

        metal : float
            The metallicity.

        dataset : str, optional
            The name of the dataset. Default is 'visir-lowres'.

        width : float, optional
            The passband width in nanometers. Default is 50.

        uncertainty_multiplier : float, optional
            The uncertainty multiplier to adjust the width of the prior. Default is 10.

        """
        self._tsa.set_ldtk_prior(teff, logg, metal, dataset, width, uncertainty_multiplier)

    @property
    def k_knots(self) -> ndarray:
        """Get the radius ratio (k) knots."""
        return self._tsa.kx_knots

    @property
    def nk(self) -> int:
        """Get the number of radius ratio (k) knots."""
        return self._tsa.nk

    @property
    def nbl(self) -> int:
        """Get the number of baseline knots."""
        return self._tsa.nbl

    def add_k_knots(self, knot_wavelengths) -> None:
        """Add radius ratio (k) knots.

        Parameters
        ----------
        knot_wavelengths : array-like
            List or array of knot wavelengths to be added.
        """
        self._tsa.add_k_knots(knot_wavelengths)

    def set_k_knots(self, knot_wavelengths) -> None:
        """Set the radius ratio (k) knots.

        Parameters
        ----------
        knot_wavelengths : array-like
            List or array of knot wavelengths.
        """
        self._tsa.set_k_knots(knot_wavelengths)

    @property
    def ps(self) -> ParameterSet:
        """Get the model parameterization."""
        return self._tsa.ps

    def print_parameters(self):
        """Prints the model parameterization."""
        self._tsa.print_parameters(1)

    def plot_setup(self, figsize=(13,4)) -> Figure:
        """Plots the model setup with limb darkening knots, radius ratio knots, and data binning.

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure. Default is (13, 4).

        Returns
        -------
        Figure
        """
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

    def fit_white(self) -> None:
        """Fits a white light curve model and sets the out-of-transit mask."""
        self._wa = WhiteLPF(self._tsa)
        self._wa.optimize()
        pv = self._wa._local_minimization.x
        phase = fold(self.time, pv[1], pv[0])
        self.ootmask = abs(phase) > 0.502 * self._wa.transit_duration

    def plot_white(self) -> Figure:
        """Plots the white light curve data with the best-fit model.

        Returns
        -------
        Figure
        """
        return self._wa.plot()

    def normalize_baseline(self, deg: int = 1) -> None:
        """Nortmalize the baseline flux for each spectroscopic light curve.

        Normalize the baseline flux using a low-order polynomial fitted to the out-of-transit
        data for each spectroscopic light curve.

        Parameters
        ----------
        deg : int
            The degree of the fitted polynomial. Should be 0 or 1. Higher degrees are not allowed
            because they could affect the transit depths.

        Returns
        -------
        None

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
            raise ValueError("The degree of the fitted polynomial ('deg') should be 0 or 1. Higher degrees are not allowed because they could affect the transit depths.")
        for ipb in range(self.npb):
            pl = poly1d(polyfit(self.time[self.ootmask], self.fluxes[ipb, self.ootmask], deg=deg))
            self.fluxes[ipb, :] /= pl(self.time)

    def plot_baseline(self, axs: Optional[Axes] = None) -> Axes:
        """Plots the out-of-transit spectroscopic light curves before and after the normalization.

        Parameters
        ----------
        axs : matplotlib.axes.Axes object or None, optional

        Returns
        -------
        axs : matplotlib.axes.Axes object
        """
        if axs is None:
            fig, axs = subplots(1, 2, figsize=(13, 4), sharey='all')
        else:
            fig = None
        axs[0].imshow(where(self.ootmask, self.original_fluxes, 1), aspect='auto', origin='lower', extent=self._extent)
        axs[1].imshow(where(self.ootmask, self.fluxes, 1), aspect='auto', origin='lower', extent=self._extent)
        setp(axs, xlabel=f'Time - {self._tref:.0f} [BJD]')
        setp(axs[0], ylabel=r'Wavelength [$\mu$m]')

        if fig is not None:
            fig.tight_layout()
        return axs

    def fit(self, niter: int = 200, npop: int = 150, pool: Optional[Any] = None, lnpost: Optional[Callable]=None) -> None:
        """Fit the spectroscopic light curves jointly using Differential Evolution.

        Fit the spectroscopic light curves jointly for `niter` iterations using Differential Evolution.

        Parameters
        ----------
        niter : int, optional
            Number of iterations for optimization. Default is 200.
        npop : int, optional
            Population size for optimization. Default is 150.
        pool : multiprocessing.Pool, optional
            Multiprocessing pool for parallel optimization. Default is None.
        lnpost : callable, optional
            Log posterior function for optimization. Default is None.
        """
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
        """Sample the posterior distribution using the emcee MCMC sampler.

        Parameters
        ----------
        niter : int, optional
            Number of iterations in the MCMC sampling. Default is 500.
        thin : int, optional
            Thinning factor for the MCMC samples. Default is 10.
        repeats : int, optional
            Number of repeated iterations in the MCMC sampling. Default is 1.
        pool : object, optional
            Parallel processing pool object to use for parallelization. Default is None.
        lnpost : function, optional
            Log posterior function that takes a parameter vector as input and returns the log posterior probability.
            Default is None.
        leave : bool, optional
            Whether to leave the progress bar visible after sampling is finished. Default is True.
        save : bool, optional
            Whether to save the MCMC samples to disk. Default is False.
        use_tqdm : bool, optional
            Whether to use tqdm progress bar during sampling. Default is True.

        """
        self._tsa.sample_mcmc(niter=niter, thin=thin, repeats=repeats, pool=pool, lnpost=lnpost, vectorize=(pool is None), leave=leave, save=save, use_tqdm=use_tqdm)
        self.sampler = self._tsa.sampler

    def plot_transmission_spectrum(self, result: Optional[str] = None, ax: Axes = None, xscale: Optional[str] = None,
                                   xticks=None, ylim=None,  plot_resolution: bool = True) -> Figure:
        """Plots the transmission spectrum.

        Parameters
        ----------
        result : Optional[str]
            The type of result to plot. Can be 'fit', 'mcmc', or None. If None, the default behavior is to use 'mcmc' if
            the MCMC sampler has been run, otherwise 'fit'. Default is None.
        ax : Axes
            The matplotlib Axes object to plot on. If None, a new figure and axes will be created. Default is None.
        xscale : Optional[str]
            The scale of the x-axis. Can be 'linear', 'log', 'symlog', 'logit', or None. If None, the default behavior is to
            use the scale of the current axes. Default is None.
        xticks
            The tick locations for the x-axis. If None, the default behavior is to use the tick locations of the current axes.
        ylim
            The limits for the y-axis. If None, the default behavior is to use the limits of the current axes.
        plot_resolution : bool
            Whether to plot the resolution of the transmission spectrum as vertical lines. Default is True.

        Returns
        -------
        Figure
            The matplotlib Figure object of the plotted transmission spectrum.

        """
        if result is None:
            result = 'mcmc' if self._tsa.sampler is not None else 'fit'
        if result not in ('fit', 'mcmc'):
            raise ValueError("Result must be either 'fit', 'mcmc', or None")
        if result == 'mcmc' and self._tsa.sampler is None:
            raise ValueError("Cannot plot posterior solution before running the MCMC sampler.")

        fig, ax = subplots() if ax is None else (ax.get_figure(), ax)

        if result == 'fit':
            pv = self._tsa.de.minimum_location
            ar = 1e2 * squeeze(self._tsa._eval_k(pv[self._tsa._sl_rratios])) ** 2
            ax.plot(self.wavelength, ar, c='k')
            ax.plot(self._tsa.kx_knots, 1e2 * pv[self._tsa._sl_rratios] ** 2, 'k.')
        else:
            df = self._tsa.posterior_samples()
            ar = 1e2 * self._tsa._eval_k(df.iloc[:, self._tsa._sl_rratios]) ** 2
            ax.fill_between(self.wavelength, *percentile(ar, [16, 84], axis=0), alpha=0.25)
            ax.plot(self.wavelength, median(ar, 0), c='k')
            ax.plot(self.k_knots, 1e2*median(df.iloc[:, self._tsa._sl_rratios].values, 0)**2, 'k.')
        setp(ax, ylabel='Transit depth [%]', xlabel='Wavelength', xlim=self.wavelength[[0, -1]], ylim=ylim)

        if plot_resolution:
            yl = ax.get_ylim()
            ax.vlines(self.wavelength, yl[0], yl[0]+0.02*(yl[1]-yl[0]), ec='k')
        if xscale is not None:
            ax.set_xscale(xscale)
        if xticks is not None:
            ax.set_xticks(xticks, labels=xticks)
        return ax.get_figure()

    def plot_limb_darkening_parameters(self, result: Optional[str] = None, axs: Axes = None) -> Figure:
        """
        Parameters
        ----------
        result : str, optional
            The type of result to plot. Can be 'fit', 'mcmc', or None. If None, the default behavior is to use 'mcmc' if
            the MCMC sampler has been run, otherwise 'fit'. Default is None.
        axs : Array-like of matplotlib.axes.Axes with a shape [2], optional
            The axes to plot the limb darkening parameters on. If None, a new figure with subplots will be created.
            Default is None.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot of the limb darkening parameters.

        Raises
        ------
        ValueError
            If the limb darkening model is not supported.
            If the result is not 'fit', 'mcmc', or None.
            If the result is 'mcmc' and the MCMC sampler has not been run.

        Notes
        -----
        This method plots the limb darkening parameters for two-parameter limb darkening models. It supports only
        quadratic, quadratic-tri, power-2, and power-2-pm models.
        """
        if not self._tsa.ldmodel in ('quadratic', 'quadratic-tri', 'power-2', 'power-2-pm'):
            raise ValueError('Unsupportted limb darkening model: the plot supports only two-parameter limb darkening models at the moment.')

        if axs is None:
            fig, axs = subplots(1, 2, sharey='all', figsize=(13,4))
        else:
            fig = axs[0].get_figure()

        ldp1 = array([[p.prior.mean, p.prior.std] for p in self.ps[self._tsa._sl_ld][::2]])
        ldp2 = array([[p.prior.mean, p.prior.std] for p in self.ps[self._tsa._sl_ld][1::2]])

        if result is None:
            result = 'mcmc' if self._tsa.sampler is not None else 'fit'
        if result not in ('fit', 'mcmc'):
            raise ValueError("Result must be either 'fit', 'mcmc', or None")
        if result == 'mcmc' and self._tsa.sampler is None:
            raise ValueError("Cannot plot posterior solution before running the MCMC sampler.")

        if result == 'fit':
            pv = self._tsa.de.minimum_location
            ldc = self._tsa._eval_ldc(pv)[0]
            axs[0].plot(self._tsa.ld_knots, pv[self._tsa._sl_ld][0::2], 'ok')
            axs[0].plot(self.wavelength, ldc[:,0])
            axs[1].plot(self._tsa.ld_knots, pv[self._tsa._sl_ld][1::2], 'ok')
            axs[1].plot(self.wavelength, ldc[:,1])
        else:
            df = self._tsa.posterior_samples()
            ldc = df.iloc[:,self._tsa._sl_ld]

            ld1m = median(ldc.values[:,::2], 0)
            ld1e = ldc.values[:,::2].std(0)
            ld2m = median(ldc.values[:,1::2], 0)
            ld2e = ldc.values[:,1::2].std(0)

            ldc = self._tsa._eval_ldc(df.values)
            ld1p = percentile(ldc[:,:,0], [50, 16, 84], axis=0)
            ld2p = percentile(ldc[:,:,1], [50, 16, 84], axis=0)

            axs[0].fill_between(self._tsa.wavelength, ld1p[1], ld1p[2], alpha=0.5)
            axs[0].plot(self._tsa.wavelength, ld1p[0], 'k')
            axs[1].fill_between(self._tsa.wavelength, ld2p[1], ld2p[2], alpha=0.5)
            axs[1].plot(self._tsa.wavelength, ld2p[0], 'k')

            axs[0].errorbar(self._tsa.ld_knots, ld1m, ld1e, fmt='ok')
            axs[1].errorbar(self._tsa.ld_knots, ld2m, ld2e, fmt='ok')

        axs[0].plot(self._tsa.ld_knots, ldp1[:,0] + ldp1[:,1], ':', c='C0')
        axs[0].plot(self._tsa.ld_knots, ldp1[:,0] - ldp1[:,1], ':', c='C0')
        axs[1].plot(self._tsa.ld_knots, ldp2[:,0] + ldp2[:,1], ':', c='C0')
        axs[1].plot(self._tsa.ld_knots, ldp2[:,0] - ldp2[:,1], ':', c='C0')

        setp(axs, xlim=self.wavelength[[0,-1]], xlabel=r'Wavelength [$\mu$m]')
        setp(axs[0], ylabel='Limb darkening coefficient 1')
        setp(axs[1], ylabel='Limb darkening coefficient 2')
        fig.tight_layout()
        return fig

    def plot_residuals(self, result: Optional[str] = None, ax: Axes = None, pmin: float = 1, pmax: float = 99) -> Figure:
        """
        Parameters
        ----------
        result : Optional[str], default=None
            The result type to plot. Must be either 'fit', 'mcmc', or None.
        ax : Axes, default=None
            The axes object to plot on. If None, a new figure and axes will be created.
        pmin : float, default=1
            The lower percentile to use when setting the color scale of the residuals image.
        pmax : float, default=99
            The upper percentile to use when setting the color scale of the residuals image.

        Returns
        -------
        Figure
            The figure object containing the plotted residuals.

        Raises
        ------
        ValueError
            If result is not one of 'fit', 'mcmc', or None.
        ValueError
            If result is 'mcmc' but the MCMC sampler has not been run.

        """
        if result is None:
            result = 'mcmc' if self._tsa.sampler is not None else 'fit'
        if result not in ('fit', 'mcmc'):
            raise ValueError("Result must be either 'fit', 'mcmc', or None")
        if result == 'mcmc' and self._tsa.sampler is None:
            raise ValueError("Cannot plot posterior solution before running the MCMC sampler.")

        fig, ax = subplots() if ax is None else (ax.get_figure(), ax)

        tc = self._wa.transit_center
        td = self._wa.transit_duration

        pv = self._tsa.de.minimum_location if result == 'fit' else median(self._tsa.posterior_samples().values, 0)
        fmodel = squeeze(self._tsa.flux_model(pv))
        residuals = self.fluxes - fmodel
        pp = percentile(residuals, [pmin, pmax])
        ax.imshow(residuals, aspect='auto', vmin=pp[0], vmax=pp[1], origin='lower', extent=self._extent)

        for i in range(2):
            ax.axvline(tc + (-1) ** i * 0.5 * td - self._tref, c='w', ymax=0.05, lw=5)
            ax.axvline(tc + (-1) ** i * 0.5 * td - self._tref, c='w', ymin=0.95, lw=5)
            ax.axvline(tc + (-1) ** i * 0.5 * td - self._tref, c='k', ymax=0.05, lw=1)
            ax.axvline(tc + (-1) ** i * 0.5 * td - self._tref, c='k', ymin=0.95, lw=1)

        setp(ax, xlabel=f'Time - {self._tref:.0f} [BJD]', ylabel=r'Wavelength [$\mu$m]')
        fig.tight_layout()
        return fig

    def plot_fit(self, result: Optional[str] = None, figsize=None, res_args=None, trs_args=None) -> Figure:
        """Plots the final fit.

        Parameters
        ----------
        result : Optional[str]
            The result of the fit. Default is None.
        figsize : Optional[Tuple[float, float]]
            The size of the figure in inches. Default is None.
        res_args : Optional[Dict[str, Any]]
            Additional arguments for plotting residuals. Default is None.
        trs_args : Optional[Dict[str, Any]]
            Additional arguments for plotting transmission spectrum. Default is None.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The plotted figure containing the residual plot, transmission spectrum plot, and limb darkening parameters plot.
        """
        if trs_args is None:
            trs_args = {}
        if res_args is None:
            res_args = {}
        fig = Figure(figsize=figsize)
        gs = GridSpec(2,3)
        axr = fig.add_subplot(gs[1,0])
        axs = fig.add_subplot(gs[0,:])
        axl = fig.add_subplot(gs[1,1]), fig.add_subplot(gs[1,2])
        self.plot_residuals(result, axr, **res_args)
        self.plot_transmission_spectrum(result, ax=axs, **trs_args)
        self.plot_limb_darkening_parameters(result, axs=axl)
        fig.tight_layout()
        fig.align_ylabels()
        return fig
