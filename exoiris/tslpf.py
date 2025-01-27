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

from copy import deepcopy
from typing import Optional, Literal

from ldtk import BoxcarFilter, LDPSetCreator   # noqa
from numba import njit, prange
from numpy import zeros, log, pi, linspace, inf, atleast_2d, newaxis, clip, arctan2, ones, floor, sum, concatenate, \
    sort, ndarray, zeros_like, array, tile, arange, squeeze, dstack
from numpy.random import default_rng
from celerite2 import GaussianProcess as GP, terms

from pytransit.lpf.logposteriorfunction import LogPosteriorFunction

from pytransit.orbits import as_from_rhop, i_from_ba, fold, i_from_baew, d_from_pkaiews, epoch
from pytransit.param import ParameterSet, UniformPrior as UP, NormalPrior as NP, GParameter
from scipy.interpolate import pchip_interpolate, splrep, splev, Akima1DInterpolator

from .tsmodel import TransmissionSpectroscopyModel as TSModel
from .tsdata import TSDataSet
from .ldtkld import LDTkLD

NM_WHITE = 0
NM_GP_FIXED = 1
NM_GP_FREE = 2

noise_models = dict(white=NM_WHITE, fixed_gp=NM_GP_FIXED, free_gp=NM_GP_FREE)

@njit(parallel=True, cache=False)
def lnlike_normal(o, m, e, f):
    if m.ndim == 2:
        return -sum(log(f[0]*e)) - 0.5 * o.size * log(2. * pi) - 0.5 * sum((o - m) ** 2 / (f[0]*e) ** 2)
    if m.ndim == 3:
        npv = m.shape[0]
        lnlike = zeros(npv)
        for i in prange(npv):
            lnlike[i] = -sum(log(f[i]*e)) - 0.5 * o.size * log(2. * pi) - 0.5 * sum((o - m[i]) ** 2 / (f[i]*e) ** 2)
        return lnlike


def ip_pchip(x, xk, yk):
    return pchip_interpolate(xk, yk, x)


def ip_bspline(x, xk, yk):
    return splev(x, splrep(xk, yk))


def ip_makima(x, xk, yk):
    return Akima1DInterpolator(xk, yk, method='makima', extrapolate=True)(x)


def add_knots(x_new, x_old):
    return sort(concatenate([x_new, x_old]))


def clean_knots(knots, min_distance, lmin=0, lmax=inf):
    """Clean the knot table by replacing groups of adjacent knots with a single knot at the group mean.

    Parameters
    ----------
    knots : numpy.ndarray
        An array of knots.

    min_distance : float
        The minimum distance between adjacent knots.

    lmin : float, optional
        The minimum value of knots to consider. Default is 0.

    lmax : float, optional
        The maximum value of knots to consider. Default is inf.

    Returns
    -------
    numpy.ndarray
        An array of cleaned knots, where adjacent knots that are less than `min_distance` apart are replaced
        by the mean value of the group.
    """
    i = 0
    nknots = []
    while i < knots.size:
        m = [i]
        if lmin <= knots[i] <= lmax:
            j = i+1
            while i < knots.size - 1 and knots[j]-knots[i] < min_distance:
                j += 1
                i += 1
                m.append(i)
        nknots.append(knots[m].mean())
        i += 1
    return array(nknots)


class TSLPF(LogPosteriorFunction):
    def __init__(self, name: str, ldmodel, data: TSDataSet, nk: int = 50, nldc: int = 10, nthreads: int = 1,
                 tmpars = None, noise_model: str = 'white',
                 interpolation: Literal['bspline', 'pchip', 'makima'] = 'bspline'):
        super().__init__(name)
        self._original_data: TSDataSet | None = None
        self.data: TSDataSet | None = None
        self.npb: list[int] | None= None
        self.npt: list[int] | None = None
        self.ndim: int | None = None
        self._baseline_models: list[ndarray] | None = None
        self.interpolation: str = interpolation

        self._ip = {'bspline': ip_bspline, 'pchip': ip_pchip, 'makima': ip_makima}[interpolation]

        self._gp: Optional[list[GP]] = None
        self._gp_time: Optional[list[ndarray]] = None
        self._gp_flux: Optional[list[ndarray]] = None

        self.set_noise_model(noise_model)

        self.ldmodel = ldmodel

        self.tms = [TSModel(ldmodel, nthreads=nthreads, **(tmpars or {})) for i in range(len(data))]
        self.set_data(data)

        if isinstance(ldmodel, LDTkLD):
            for tm in self.tms:
                tm.ldmodel = None
            self.ldmodel._init_interpolation(self.tms[0].mu)

        self.nthreads = nthreads
        self.nldc = nldc
        self.nk = nk

        self.k_knots = linspace(data.wlmin, data.wlmax, self.nk)

        if isinstance(ldmodel, LDTkLD):
            self.ld_knots = array([])
        else:
            self.ld_knots = linspace(data.wlmin, data.wlmax, self.nldc)

        self._ootmask = None
        self._npv = 1
        self._de_population: Optional[ndarray] = None
        self._de_imin: Optional[int] = None
        self._mc_chains: Optional[ndarray] = None

        self._init_parameters()

    @property
    def flux(self) -> list[ndarray]:
        return self.data.fluxes

    @property
    def times(self) -> list[ndarray]:
        return self.data.times

    @property
    def wavelengths(self) -> list[ndarray]:
        return self.data.wavelengths

    @property
    def errors(self) -> list[ndarray]:
        return self.data.errors

    def set_data(self, data: TSDataSet):
        self._original_data = deepcopy(data)
        self.data = data
        self.npb: list[int] = [f.shape[0] for f in self.flux]
        self.npt: list[int] = [f.shape[1] for f in self.flux]
        for i, time in enumerate(self.times):
            self.tms[i].set_data(time)
        if self._nm in (NM_GP_FIXED, NM_GP_FREE):
            self._init_gp()

    def _init_parameters(self) -> None:
        self.ps = ParameterSet([])
        self._init_p_star()
        self._init_p_orbit()
        self._init_p_limb_darkening()
        self._init_p_radius_ratios()
        self._init_p_noise()
        self._init_p_baseline()
        self.ps.freeze()
        self.ndim = len(self.ps)

    def set_noise_model(self, noise_model: str) -> None:
        """Sets the noise model for the analysis.

        Parameters
        ----------
        noise_model : str
            The noise model to be used. Must be one of the following: white, fixed_gp, free_gp.

        Raises
        ------
        ValueError
            If noise_model is not one of the specified options.
        """
        if noise_model not in noise_models.keys():
            raise ValueError('noise_model must be one of: white, fixed_gp, free_gp')
        self.noise_model = noise_model
        self._nm = noise_models[noise_model]
        if self._nm in (NM_GP_FIXED, NM_GP_FREE):
            self._init_gp()

    def _init_gp(self) -> None:
        """Initializes the Gaussian Process (GP) .

        This method initializes the necessary variables and sets up the GP for the given data.
        """
        self._gp_time = []
        self._gp_flux = []
        self._gp_ferr = []
        self._gp = []
        for d in self.data:
            self._gp_time.append((tile(d.time[newaxis, :], (d.nwl, 1)) + arange(d.nwl)[:, newaxis]).ravel())
            self._gp_flux.append(d.fluxes.ravel())
            self._gp_ferr.append(d.errors.ravel())
            self._gp.append(GP(terms.Matern32Term(sigma=self._gp_flux[-1].std(), rho=0.1)))
            self._gp[-1].compute(self._gp_time[-1], yerr=self._gp_ferr[-1], quiet=True)

    def set_gp_hyperparameters(self, sigma: float, rho: float) -> None:
        """Sets the Gaussian Process hyperparameters assuming a Matern32 kernel.

        Parameters
        ----------
        sigma : float
            The kernel amplitude parameter.

        rho : float
            The length scale parameter.

        Raises
        ------
        RuntimeError
            If the GP has not been initialized before setting the hyperparameters.
        """
        if self._gp is None:
            raise RuntimeError('The GP needs to be initialized before setting hyperparameters.')
        for i, gp in enumerate(self._gp):
            gp.kernel = terms.Matern32Term(sigma=sigma, rho=rho)
            gp.compute(self._gp_time[i], yerr=self._gp_ferr[i], quiet=True)

    def set_gp_kernel(self, kernel: terms.Term) -> None:
        """Sets the kernel for the Gaussian Process (GP) model and recomputes the GP.

        Parameters
        ----------
        kernel : terms.Term
            The kernel to be set for the GP.
        """
        for i, gp in enumerate(self._gp):
            gp.kernel = kernel
            gp.compute(self._gp_time[i], yerr=self._gp_ferr[i], quiet=True)

    def _init_p_star(self) -> None:
        pstar = [GParameter('rho', 'stellar density', 'g/cm^3', UP(0.1, 25.0), (0, inf))]
        self.ps.add_global_block('star', pstar)

    def _init_p_limb_darkening(self) -> None:
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

    def _init_p_noise(self):
        ps = self.ps
        pp = [GParameter(f'sigma_m_{i:02d}', f'Noise group {i} sigma multipler', '',
                         NP(1.0, 0.01), (0, inf)) for i in range(self.data.n_noise_groups)]
        ps.add_global_block('white_noise_multipliers', pp)
        self._start_wnm = ps.blocks[-1].start
        self._sl_wnm = ps.blocks[-1].slice

    def _init_p_baseline(self):
        ps = self.ps
        self.n_baselines = self.data.n_baselines
        self.baseline_knots = []
        pp = []
        for i, d in enumerate(self.data):
            if d.n_baseline== 1:
                self.baseline_knots.append([])
                pp.append(GParameter(f'bl_{i:02d}_c', 'baseline constant', '', NP(1.0, 1e-6), (0, inf)))
            elif d.n_baseline > 1:
                knots = linspace(d.wavelength.min(), d.wavelength.max(), d.n_baseline)
                self.baseline_knots.append(knots)
                pp.extend([GParameter(f'bl_{i:02d}_{k:08.5f}', fr'baseline at {k:08.5f} $\mu$m', '', NP(1.0, 1e-6), (0, inf)) for k in knots])
        ps.add_global_block('baseline_coefficients', pp)
        self._start_baseline = ps.blocks[-1].start
        self._sl_baseline = ps.blocks[-1].slice

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

        xn = self.k_knots = clean_knots(sort(knot_wavelengths), 1e-5)
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
        if self._de_population is not None:
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
                den[i, sln] = self._ip(xn, xo, deo[i, slo])

            self._de_population = den
            self.de = None

        # Resample the MCMC parameter population
        # --------------------------------------
        if self._mc_chains is not None:
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
                fmcn[i, sln] = self._ip(xn, xo, fmco[i, slo])

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
                pvpn[i, sldn] = self._ip(xn, xo, pvpo[i, sldo])

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
        pvp = atleast_2d(pvp)
        ks = [zeros((pvp.shape[0], npb)) for npb in self.npb]
        for ids in range(self.data.size):
            for ipv in range(pvp.shape[0]):
                ks[ids][ipv,:] =  self._ip(self.wavelengths[ids], self.k_knots, pvp[ipv])
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
            ldp = [zeros((pvp.shape[0], npb, 2)) for npb in self.npb]
            for ids in range(self.data.size):
                for ipv in range(pvp.shape[0]):
                    ldp[ids][ipv, :, 0] = self._ip(self.wavelengths[ids], self.ld_knots, ldk[ipv, :, 0])
                    ldp[ids][ipv, :, 1] = self._ip(self.wavelengths[ids], self.ld_knots, ldk[ipv, :, 1])
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
        if isinstance(self.ldmodel, LDTkLD):
            ldp, istar = self.ldmodel(self.tms[0].mu, ldp)
            ldpi = dstack([ldp, istar])
            flux = []
            for i, tm in enumerate(self.tms):
                flux.append(tm.evaluate(k[i], ldpi[:, self.ldmodel.wlslices[i], :], t0, p, aor, inc, ecc, w, copy))
            return flux
        else:
            return [tm.evaluate(k[i], ldp[i], t0, p, aor, inc, ecc, w, copy) for i,tm in enumerate(self.tms)]

    def baseline_model(self, pv):
        pv = atleast_2d(pv)[:, self._sl_baseline]
        npv = pv.shape[0]
        if self._baseline_models is None or self._baseline_models[0].shape[0] != npv:
            self._baseline_models = [zeros((npv, d.nwl)) for d in self.data]
        j = 0
        for i, d in enumerate(self.data):
            nbl = d.n_baseline
            m = self._baseline_models[i]
            if nbl == 1:
                m[:, :] = pv[:, j][:, newaxis]
            else:
                for ipv in range(npv):
                    m[ipv, :] = splev(d.wavelength, splrep(self.baseline_knots[i], pv[ipv, j:j+nbl], k=min(nbl-1, 3)))
            j += nbl
        return self._baseline_models

    def flux_model(self, pv):
        transit_models = self.transit_model(pv)
        baseline_models = self.baseline_model(pv)
        for i in range(self.data.size):
            transit_models[i][:, :, :] *= baseline_models[i][:, :, newaxis]
        return transit_models

    def create_pv_population(self, npop: int = 50) -> ndarray:
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

    def lnlikelihood(self, pv) -> ndarray | float :
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
        pv = atleast_2d(pv)
        npv = pv.shape[0]
        fmod = self.flux_model(pv)
        wn_multipliers = pv[:, self._sl_wnm]
        lnl = zeros(npv)
        if self._nm == NM_WHITE:
            for i, d in enumerate(self.data):
                lnl += lnlike_normal(d.fluxes, fmod[i], d.errors, wn_multipliers[:, d.ngid])
        elif self._nm == NM_GP_FIXED:
            for j in range(npv):
                for i in range(self.data.size):
                    lnl[j] += self._gp[i].log_likelihood(self._gp_flux[i] - fmod[i][j].ravel())
        else:
            raise NotImplementedError("The free GP noise model hasn't been implemented yet.")
        return lnl if npv > 1 else lnl[0]

    def create_initial_population(self, n: int, source: str, add_noise: bool = True) -> ndarray:
        """Create an initial parameter vector population for DE.

        Parameters
        ----------
        n : int
            Number of parameter vectors in the population.
        source : str
            Source of the initial population. Must be either 'fit' or 'mcmc'.
        add_noise : bool, optional
            Flag indicating whether to add noise to the initial population. Default is True.

        Returns
        -------
        numpy.ndarray
            The initial population.

        Raises
        ------
        ValueError
            If the source is not 'fit' or 'mcmc'.
        """
        rng = default_rng()

        if source not in ('fit', 'mcmc'):
            raise ValueError("source must be either 'fit' or 'mcmc'")

        if source == 'fit':
            pvs = self._de_population
            if n == pvs.shape[0]:
                pvp = pvs.copy()
            else:
                pvp = rng.choice(pvs, size=n)
        else:
            pvs = self._mc_chains
            if n == pvs.shape[2]:
                pvp = pvs[:,-1,:].copy()
            else:
                pvp = rng.choice(pvs.reshape([-1, self.ndim]), size=n)

        if add_noise:
            pvp[:, 0] += rng.normal(0, 0.005, size=n)
            pvp[:, 1] += rng.normal(0, 0.001, size=n)
            pvp[:, 3] += rng.normal(0, 0.005, size=n)
            pvp[:, self._sl_rratios] += rng.normal(0, 1, pvp[:, self._sl_rratios].shape) * 0.002 * pvp[:, self._sl_rratios]
            pvp[:, self._sl_ld] += rng.normal(0, 1, pvp[:, self._sl_ld].shape) * 0.002 * pvp[:, self._sl_ld]
        return pvp

    def optimize_global(self, niter=200, npop=50, population=None, pool=None, lnpost=None, vectorize=True,
                        label='Global optimisation', leave=False, plot_convergence: bool = True, use_tqdm: bool = True,
                        plot_parameters: tuple = (0, 2, 3, 4), min_ptp: float = 5):

        if population is None:
            if self._de_population is None:
                population = self.create_pv_population()
            else:
                population = self._de_population

        super().optimize_global(niter=niter, npop=npop, population=population, pool=pool, lnpost=lnpost,
                                vectorize=vectorize, label=label, leave=leave, plot_convergence=plot_convergence,
                                use_tqdm=use_tqdm, plot_parameters=plot_parameters, min_ptp=min_ptp)
        self._de_population = self.de.population.copy()
        self._de_imin = self.de.minimum_index

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
