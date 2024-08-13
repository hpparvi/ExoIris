from typing import Optional

from ldtk import BoxcarFilter, LDPSetCreator
from numba import njit, prange
from numpy import zeros, log, pi, linspace, inf, atleast_2d, newaxis, clip, arctan2, ones, floor, sum, concatenate, \
    sort, ndarray, zeros_like

from pytransit import TSModel, RRModel, LDTkLD, TransitAnalysis
from pytransit.lpf.logposteriorfunction import LogPosteriorFunction

from pytransit.orbits import as_from_rhop, i_from_ba, fold, i_from_baew, d_from_pkaiews, epoch
from pytransit.param import ParameterSet, UniformPrior as UP, NormalPrior as NP, GParameter
from scipy.interpolate import splev, splrep


@njit(parallel=True, cache=False)
def lnlike_normal(o, m, e):
    if m.ndim == 2:
        return -sum(log(e)) - 0.5 * o.size * log(2. * pi) - 0.5 * sum((o - m) ** 2 / e ** 2)
    if m.ndim == 3:
        npv = m.shape[0]
        lnlike = zeros(npv)
        for i in prange(npv):
            lnlike[i] = -sum(log(e)) - 0.5 * o.size * log(2. * pi) - 0.5 * sum((o - m[i]) ** 2 / e ** 2)
        return lnlike


def resample(x_new, x_old, y_old):
    return splev(x_new, splrep(x_old, y_old, s=0.0))


def add_knots(x_new, x_old):
    return sort(concatenate([x_new, x_old]))


class TSLPF(LogPosteriorFunction):
    def __init__(self, name: str, ldmodel, time: ndarray, wavelength: ndarray, fluxes: ndarray, errors: ndarray,
                 nk: int = None, nldc: int = 10, nthreads: int = 1, tmpars = None):
        super().__init__(name)
        self.time = time.copy()
        self.wavelength = wavelength.copy()
        self.flux = fluxes.copy()
        self.ferr = errors.copy()

        self.nthreads = nthreads
        self.npb = fluxes.shape[0]
        self.npt = fluxes.shape[1]
        self.nldc = nldc
        self.ndim = None
        self.nk = self.npb if nk is None else min(nk, self.npb)

        self.k_knots = linspace(wavelength[0], wavelength[-1], self.nk)
        self.ld_knots = linspace(wavelength[0], wavelength[-1], self.nldc)

        self._original_flux = fluxes.copy()
        self._ootmask = None
        self._npv = 1
        self._de_population: Optional[ndarray] = None
        self._mc_chains: Optional[ndarray] = None

        self.ldmodel = ldmodel
        self.tm = TSModel(ldmodel, nthreads=nthreads, **(tmpars or {}))
        self.tm.set_data(self.time)
        self._init_parameters()
        self.white_model = None

    def _init_parameters(self):
        self.ps = ParameterSet([])
        self._init_p_star()
        self._init_p_orbit()
        self._init_limb_darkening()
        self._init_p_radius_ratios()
        self.ps.freeze()
        self.ndim = len(self.ps)

    def _init_p_star(self):
        pstar = [GParameter('rho', 'stellar density', 'g/cm^3', UP(0.1, 25.0), (0, inf))]
        self.ps.add_global_block('star', pstar)

    def _init_limb_darkening(self):
        if isinstance(self.ldmodel, LDTkLD):
            pld = [GParameter('teff', 'stellar TEff', 'K', NP(*self.ldmodel.sc.teff), (0, inf)),
                   GParameter('logg', 'stellar log g', '', NP(*self.ldmodel.sc.logg), (0, inf)),
                   GParameter('metal', 'stellar metallicity', '', NP(*self.ldmodel.sc.metal), (-inf, inf))]
        else:
            pld = concatenate([
                [GParameter(f'ldc1_{l:08.5f}', fr'ldc1 at {l:08.5f} $\mu$m', '', UP(0, 1), bounds=(-inf, inf)),
                 GParameter(f'ldc2_{l:08.5f}', fr'ldc2 at {l:08.5f} $\mu$m', '', UP(0, 1), bounds=(-inf, inf))]
                for l in self.ld_knots])
        self.ps.add_global_block('limb_darkening', pld)
        self._start_ld = self.ps.blocks[-1].start
        self._sl_ld = self.ps.blocks[-1].slice

    def _init_p_orbit(self):
        ps = self.ps
        pp = [GParameter('tc', 'zero_epoch', '', NP(0.0, 0.1), (-inf, inf)),
              GParameter('p', 'period', 'd', NP(1.0, 1e-5), (0, inf)),
              GParameter('b', 'impact_parameter', 'R_s', UP(0.0, 1.0), (0, inf)),
              GParameter('secw', 'sqrt(e) cos(w)', '', NP(0.0, 1e-5), (-1, 1)),
              GParameter('sesw', 'sqrt(e) sin(w)', '', NP(0.0, 1e-5), (-1, 1))]
        ps.add_global_block('orbit', pp)
        self._start_orbit = ps.blocks[-1].start
        self._sl_orbit = ps.blocks[-1].slice

    def _init_p_radius_ratios(self):
        ps = self.ps
        pp = [GParameter(f'k_{k:08.5f}', fr'radius ratio at {k:08.5f} $\mu$m', 'A_s', UP(0.02, 0.2), (0, inf)) for k in self.k_knots]
        ps.add_global_block('radius_ratios', pp)
        self._start_rratios = ps.blocks[-1].start
        self._sl_rratios = ps.blocks[-1].slice

    def set_ldtk_prior(self, teff, logg, metal, dataset: str = 'visir-lowres', width: float = 50,
                       uncertainty_multiplier: float = 10):
        hw = 0.5 * width
        filters = [BoxcarFilter('a', wlc - hw, wlc + hw) for wlc in 1e3 * self.ld_knots]
        sc = LDPSetCreator(teff, logg, metal, filters=filters, dataset=dataset)
        ps = sc.create_profiles()

        match self.ldmodel:
            case 'power-2':
                ldc, lde = ps.coeffs_p2()
            case 'quadratic':
                ldc, lde = ps.coeffs_qd()
            case 'quadratic-triangular':
                ldc, lde = ps.coeffs_tq()
            case _:
                raise ValueError('Unsupported limb darkening model.')

        for i,l in enumerate(self.ld_knots):
            self.set_prior(f'ldc1_{l:08.5f}', 'NP', ldc[i, 0].round(3), (uncertainty_multiplier * lde[i, 0]).round(3))
            self.set_prior(f'ldc2_{l:08.5f}', 'NP', ldc[i, 1].round(3), (uncertainty_multiplier * lde[i, 1]).round(3))

    def add_k_knots(self, knot_wavelengths) -> None:
        """Add radius ratio (k) knots to the model.

        Parameters
        ----------
        knot_wavelengths : array-like
            An array of knot wavelengths to be added.
        """
        self.set_k_knots(concatenate([self.k_knots, knot_wavelengths]))

    def set_k_knots(self, knot_wavelengths) -> None:
        """Set the radius ratio (k) knot wavelengths for the model.

        Parameters
        ----------
        knot_wavelengths : array-like
            Array of knot wavelengths.

        """

        # Save the old variables
        # ----------------------
        xo = self.k_knots
        pso = self.ps
        deo = self._de_population
        mco = self._mc_chains
        slo = self._sl_rratios
        ndo = self.ndim

        xn = self.k_knots = sort(knot_wavelengths)
        self.nk = self.k_knots.size

        self._init_parameters()
        psn = self.ps
        sln = self._sl_rratios
        ndn = self.ndim

        # Set the priors back as they were
        # --------------------------------
        for po in pso:
            if po.name in psn.names:
                self.set_prior(po.name, po.prior)

        # Resample the DE parameter population
        # ------------------------------------
        if self.de is not None:
            den = zeros((deo.shape[0], ndn))

            # Copy the old parameter values
            # -----------------------------
            for pid_old, p in enumerate(pso):
                if p.name in psn.names:
                    pid_new = psn.find_pid(p.name)
                    den[:, pid_new] = deo[:, pid_old]

            # Resample the radius ratios
            # --------------------------
            for i in range(den.shape[0]):
                den[i, sln] = resample(xn, xo, deo[i, slo])

            self._de_population = den
            self.de = None

        # Resample the MCMC parameter population
        # --------------------------------------
        if self.sampler is not None:
            fmco = mco.reshape([-1, ndo])
            fmcn = zeros((fmco.shape[0], ndn))

            # Copy the old parameter values
            # -----------------------------
            for pid_old, p in enumerate(pso):
                if p.name in psn.names:
                    pid_new = psn.find_pid(p.name)
                    fmcn[:, pid_new] = fmco[:, pid_old]

            # Resample the radius ratios
            # --------------------------
            for i in range(fmcn.shape[0]):
                fmcn[i, sln] = resample(xn, xo, fmco[i, slo])

            self._mc_chains = fmcn.reshape([mco.shape[0], mco.shape[1], ndn])
            self.sampler = None


    def add_ld_knots(self, knot_wavelengths) -> None:
        """Add limb darkening knots to the model.

        Parameters
        ----------
        knot_wavelengths : array-like
            An array of knot wavelengths to be added.
        """
        self.set_ld_knots(concatenate([self.ld_knots, knot_wavelengths]))

    def set_ld_knots(self, knot_wavelengths) -> None:
        """Set the limb darkening knot wavelengths for the model.

        Parameters
        ----------
        knot_wavelengths : array-like
            Array of knot wavelengths.
        """
        xo = self.ld_knots
        xn = self.ld_knots = sort(knot_wavelengths)
        self.nldc = self.ld_knots.size

        pvpo = self.de.population.copy() if self.de is not None else None
        pso = self.ps
        sldo = self._sl_ld
        self._init_parameters()
        psn = self.ps
        sldn = self._sl_ld
        for po in pso:
            if po.name in psn.names:
                self.set_prior(po.name, po.prior)

        if self.de is not None:
            pvpn = self.create_pv_population(pvpo.shape[0])
            # Copy the old parameter values
            # -----------------------------
            for pid_old, p in enumerate(pso):
                if p.name in psn:
                    pid_new = psn.find_pid(p.name)
                    pvpn[:, pid_new] = pvpo[:, pid_old]

            # Resample the radius ratios
            # --------------------------
            for i in range(pvpn.shape[0]):
                pvpn[i, sldn] = resample(xn, xo, pvpo[i, sldo])

            self.de = None
            self._de_population = pvpn

    def _eval_k(self, pvp):
        """
        Evaluate the radius ratio model.

        Parameters
        ----------
        pvp : ndarray
            The input array of shape (npv, np), where npv is the number of parameter vectors
            and np is the number of parameters.

        Returns
        -------
        ks : ndarray
            The radius ratios of shape (npv, npb), where npb is the number of passbands.
        """
        if self.nk == self.npb:
            return pvp
        else:
            pvp = atleast_2d(pvp)
            ks = zeros((pvp.shape[0], self.npb))
            for ipv in range(pvp.shape[0]):
                ks[ipv,:] =  splev(self.wavelength, splrep(self.k_knots, pvp[ipv], s=0.0))
            return ks

    def _eval_ldc(self, pvp):
        if isinstance(self.ldmodel, LDTkLD):
            ldp = pvp[:, newaxis, self._sl_ld]
            ldp[:, 0, 0] = clip(ldp[:, 0, 0], *self.ldmodel.sc.client.teffl)
            ldp[:, 0, 1] = clip(ldp[:, 0, 1], *self.ldmodel.sc.client.loggl)
            ldp[:, 0, 2] = clip(ldp[:, 0, 2], *self.ldmodel.sc.client.zl)
            return ldp
        else:
            pvp = atleast_2d(pvp)
            ldk = pvp[:, self._sl_ld].reshape([pvp.shape[0], self.nldc, 2])
            ldp = zeros((pvp.shape[0], self.npb, 2))
            for ipv in range(pvp.shape[0]):
                ldp[ipv, :, 0] = splev(self.wavelength, splrep(self.ld_knots, ldk[ipv, :, 0], s=0.0))
                ldp[ipv, :, 1] = splev(self.wavelength, splrep(self.ld_knots, ldk[ipv, :, 1], s=0.0))
            return ldp

    def transit_model(self, pv, copy=True):
        """Evaluates the transit model for parameter vector pv.

        Parameters
        ----------
        pv : numpy.ndarray
            Array of transit parameters. Each row represents a set of transit parameters for a single transit event.
            The columns of the array should be in the following order:
            - Column 0: stellar density (g/cm^3)
            - Column 1: transit center time (T0)
            - Column 2: orbital period (P)
            - Column 3: impact parameter
            - Column 4: sqrt e cos w
            - Column 5: sqrt e sin w
            - Column 6: planet-to-star radius ratio (Rp/R_star)

        copy : bool, optional
            Whether to create a copy of the calculated values before returning the result. Default is True.

        Returns
        -------
        numpy.ndarray
            Array of model transit fluxes. Each element corresponds to the transit model flux for the corresponding set
            of transit parameters in the input array 'pv'.
        """
        pv = atleast_2d(pv)
        ldp = self._eval_ldc(pv)
        t0 = pv[:, 1]
        p = pv[:, 2]
        k = self._eval_k(pv[:, self._sl_rratios])
        aor = as_from_rhop(pv[:, 0], p)
        inc = i_from_ba(pv[:, 3], aor)
        ecc = pv[:, 4] ** 2 + pv[:, 5] ** 2
        w = arctan2(pv[:, 5], pv[:, 4])
        return self.tm.evaluate(k, ldp, t0, p, aor, inc, ecc, w, copy)

    def flux_model(self, pv):
        return self.transit_model(pv)

    def create_pv_population(self, npop: int = 50):
        """ Crate a parameter vector population.
        Parameters
        ----------
        npop : int, optional
            The number of parameter vectors in the population. Default is 50.

        Returns
        -------
        population : array_like
            An array of parameter vectors sampled from the prior distribution.
        """
        return self.ps.sample_from_prior(npop)

    def set_radius_ratio_limits(self, kmin, kmax):
        for ipb in range(self.nk):
            self.set_prior(f'k_{ipb + 1:03d}', 'UP', kmin, kmax)

    def lnlikelihood(self, pv):
        """Log likelihood for parameter vector pv.

        Parameters
        ----------
        pv : array-like
            The input parameter values for the flux model.

        Returns
        -------
        lnlike : float
            The logarithm of the likelihood value calculated using the normal distribution.

        """
        fmod = self.flux_model(pv)
        return lnlike_normal(self.flux, fmod, self.ferr)

    def optimize_global(self, niter=200, npop=50, population=None, pool=None, lnpost=None, vectorize=True,
                        label='Global optimisation', leave=False, plot_convergence: bool = True, use_tqdm: bool = True,
                        plot_parameters: tuple = (0, 2, 3, 4)):

        if population is None:
            if self._de_population is None:
                population = self.create_pv_population()
            else:
                population = self._de_population

        super().optimize_global(niter=niter, npop=npop, population=population, pool=pool, lnpost=lnpost,
                                vectorize=vectorize, label=label, leave=leave, plot_convergence=plot_convergence,
                                use_tqdm=use_tqdm, plot_parameters=plot_parameters)
        self._de_population = self.de.population.copy()

    def sample_mcmc(self, niter: int = 500, thin: int = 5, repeats: int = 1, npop: int = None, population=None,
                    label='MCMC sampling', reset=True, leave=True, save=False, use_tqdm: bool = True, pool=None,
                    lnpost=None, vectorize: bool = True):

        if population is None:
            if self._mc_chains is None:
                population = self._de_population.copy()
            else:
                population = self._mc_chains[:, -1, :].copy()

        super().sample_mcmc(niter, thin, repeats, npop=npop, population=population, label=label, reset=reset,
                            leave=leave, save=save, use_tqdm=use_tqdm, pool=pool, lnpost=lnpost, vectorize=vectorize)
        self._mc_chains = self.sampler.chain.copy()
