from ldtk import BoxcarFilter, LDPSetCreator
from numba import njit, prange
from numpy import zeros, log, pi, linspace, inf, atleast_2d, newaxis, clip, arctan2, ones, floor, sum, concatenate, sort

from pytransit import TSModel, RRModel, LDTkLD, TransitAnalysis
from pytransit.lpf.logposteriorfunction import LogPosteriorFunction

from pytransit.orbits import as_from_rhop, i_from_ba, fold, i_from_baew, d_from_pkaiews, epoch
from pytransit.param import ParameterSet, UniformPrior as UP, NormalPrior as NP, GParameter
from scipy.interpolate import splev, splrep


@njit(parallel=True, cache=False)
def lnlike_normal(o, m, e):
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
    def __init__(self, name: str, ldmodel, time, wavelength, fluxes, errors,
                 nk: int = None, nbl: int = None, nldc: int = None,
                 nthreads: int = 1, tmpars = None):
        super().__init__(name)
        self.ldmodel = ldmodel
        self.tm = TSModel(ldmodel, nthreads=nthreads, **(tmpars or {}))
        self.nthreads = nthreads
        self.time = time.copy()
        self.wavelength = wavelength.copy()
        self._original_flux = fluxes.copy()
        self.flux = fluxes.copy()
        self.ferr = errors.copy()
        self.npb = fluxes.shape[0]
        self.npt = fluxes.shape[1]
        self.tm.set_data(self.time)
        self.ootmask = None

        self.nk = self.npb if nk is None else min(nk, self.npb)
        self.kx_knots = linspace(wavelength[0], wavelength[-1], self.nk)

        self.nbl = self.npb if nbl is None else min(nbl, self.npb)
        self.bx_knots = linspace(wavelength[0], wavelength[-1], self.nbl)

        self.nldc = 10 if nldc is None else min(nldc, self.npb)
        self.ld_knots = linspace(wavelength[0], wavelength[-1], self.nldc)

        self._npv = 1
        self._bl_array = zeros((self._npv, self.npb, self.npt))

        self._init_parameters()
        self.white_model = None

    def _init_parameters(self):
        self.ps = ParameterSet([])
        self._init_p_star()
        self._init_p_orbit()
        self._init_limb_darkening()
        self._init_p_radius_ratios()
        self._init_p_baseline()
        self.ps.freeze()

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
        pp = [GParameter(f'k_{k:08.5f}', fr'radius ratio at {k:08.5f} $\mu$m', 'A_s', UP(0.02, 0.2), (0, inf)) for k in self.kx_knots]
        ps.add_global_block('radius_ratios', pp)
        self._start_rratios = ps.blocks[-1].start
        self._sl_rratios = ps.blocks[-1].slice

    def _init_p_baseline(self):
        ps = self.ps
        pp = [GParameter(f'c_{b:08.5f}', fr'Baseline constant at {b:08.5f} $\mu$m', 'A_s', NP(1.0, 0.001), (0, inf)) for b in self.bx_knots]
        ps.add_global_block('baseline', pp)
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
        self.set_k_knots(concatenate([self.kx_knots, knot_wavelengths]))

    def set_k_knots(self, knot_wavelengths):
        xo = self.kx_knots
        xn = self.kx_knots = sort(knot_wavelengths)
        self.nk = self.kx_knots.size

        pvpo = self.de.population.copy() if self.de is not None else None
        pso = self.ps
        slko = self._sl_rratios
        self._init_parameters()
        psn = self.ps
        slkn = self._sl_rratios
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
                pvpn[i, slkn] = resample(xn, xo, pvpo[i, slko])

            self.de = None
            self._pv_population = pvpn

    def _eval_k(self, pvp):
        if self.nk == self.npb:
            return pvp
        else:
            pvp = atleast_2d(pvp)
            ks = zeros((pvp.shape[0], self.npb))
            for ipv in range(pvp.shape[0]):
                ks[ipv,:] =  splev(self.wavelength, splrep(self.kx_knots, pvp[ipv], s=0.0))
            return ks

    def _eval_bl(self, pvp):
        if self.nbl == self.npb:
            return pvp
        else:
            pvp = atleast_2d(pvp)
            for ipv in range(pvp.shape[0]):
                self._bl_array[ipv, :, :] = splev(self.wavelength, splrep(self.bx_knots, pvp[ipv], s=0.0))[:, newaxis]
            return self._bl_array

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
        return self.baseline(pv) * self.transit_model(pv)

    def baseline(self, pvp):
        pvp = atleast_2d(pvp)
        if self._npv != pvp.shape[0]:
            self._npv = pvp.shape[0]
            self._bl_array = ones((pvp.shape[0], self.npb, self.npt))
        return self._eval_bl(pvp[:, self._sl_baseline])

    def create_pv_population(self, npop=50):
        return self.ps.sample_from_prior(npop)

    def set_radius_ratio_limits(self, kmin, kmax):
        for ipb in range(self.nk):
            self.set_prior(f'k_{ipb + 1:03d}', 'UP', kmin, kmax)

    def lnlikelihood(self, pv):
        fmod = self.flux_model(pv)
        return lnlike_normal(self.flux, fmod, self.ferr)

    def optimize_global(self, niter=200, npop=50, population=None, pool=None, lnpost=None, vectorize=True,
                        label='Global optimisation', leave=False, plot_convergence: bool = True, use_tqdm: bool = True,
                        plot_parameters: tuple = (0, 2, 3, 4)):

        if population is None:
            population = self._pv_population if self.de is None else None

        super().optimize_global(niter=niter, npop=npop, population=population, pool=pool, lnpost=lnpost,
                                vectorize=vectorize, label=label, leave=leave, plot_convergence=plot_convergence,
                                use_tqdm=use_tqdm, plot_parameters=plot_parameters)

