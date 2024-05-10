from numba import njit, prange
from numpy import zeros, log, pi, linspace, inf, atleast_2d, newaxis, clip, arctan2, ones, floor, sum

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


@njit
def cubic_coeffs(ys):
    n = ys.size
    cs = zeros((n-1, 4))
    for i in range(n-1):
        y0 = ys[max(0, i-1)]
        y1 = ys[i]
        y2 = ys[min(n-1, i+1)]
        y3 = ys[min(n-1, i+2)]

        cs[i,0] = y3 - y2 - y0 + y1
        cs[i,1] = y0 - y1 - cs[i,0]
        cs[i,2] = y2 - y0
        cs[i,3] = y1
    return cs


@njit
def cubic_interpolation(xs, ys, dx):
    npt = xs.size
    coeffs = cubic_coeffs(ys)
    nc = coeffs.shape[0]
    y = zeros(xs.size)
    for i in range(npt):
        x = min(xs[i], 0.9999)
        ix = int(floor(x / dx))
        a = (x - ix*dx) / dx
        a2 = a*a
        y[i] = coeffs[ix, 0]*a*a2 + coeffs[ix, 1]*a2 + coeffs[ix,2]*a + coeffs[ix,3]
    return y


@njit(parallel=True)
def cubic_interpolation_pvp(xs, ys, dx):
    y = zeros((ys.shape[0], xs.size))
    for ip in prange(ys.shape[0]):
        y[ip] = cubic_interpolation(xs, ys[ip], dx)
    return y


@njit(parallel=False)
def baseline(xs, ys, dx, npb, npt, bl):
    npv = ys.shape[0]
    for ipv in prange(npv):
        bl[ipv, :, 0] = cubic_interpolation(xs, ys[ipv], dx)
        for ipb in range(npb):
            bl[ipv, ipb, :] = bl[ipv, ipb, 0]
    return bl


class TSLPF(LogPosteriorFunction):
    def __init__(self, name: str, ldmodel, time, wavelength, fluxes, errors, nk: int = None, nbl: int = None, nthreads: int = 1, tmpars = None):
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
        self.kx_all = wavelength
        self.kx_knots = linspace(wavelength[0], wavelength[-1], self.nk)

        self.nbl = self.npb if nbl is None else min(nbl, self.npb)
        self.bx_all = linspace(0, 1, self.npb)
        self.bx_knots = linspace(0, 1, self.nbl)
        self.dbx = 1 / (self.nbl - 1)

        self._npv = 1
        self._bl_array = zeros((self._npv, self.npb, self.npt))

        self._init_parameters()
        self.white_model = None

    def _init_parameters(self):
        self.ps = ParameterSet([])
        self._init_p_star()
        self._init_p_orbit()
        self._init_p_radius_ratios()
        self._init_p_baseline()
        self.ps.freeze()

    def _init_p_star(self):
        pstar = [GParameter('rho', 'stellar density', 'g/cm^3', UP(0.1, 25.0), (0, inf)),
                 GParameter('teff', 'stellar TEff', 'K', NP(*self.ldmodel.sc.teff), (0, inf)),
                 GParameter('logg', 'stellar log g', '', NP(*self.ldmodel.sc.logg), (0, inf)),
                 GParameter('metal', 'stellar metallicity', '', NP(*self.ldmodel.sc.metal), (-inf, inf))]
        self.ps.add_global_block('star', pstar)

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
        pp = [GParameter(f'k_{i + 1:03d}', f'radius ratio {i + 1:03d}', 'A_s', UP(0.02, 0.2), (0, inf)) for i in
              range(self.nk)]
        ps.add_global_block('radius_ratios', pp)
        self._start_rratios = ps.blocks[-1].start
        self._sl_rratios = ps.blocks[-1].slice

    def _init_p_baseline(self):
        ps = self.ps
        pp = [GParameter(f'c_{i + 1:03d}', f'Baseline constant {i + 1:03d}', 'A_s', NP(1.0, 0.001), (0, inf)) for i in
              range(self.nbl)]
        ps.add_global_block('baseline', pp)
        self._start_baseline = ps.blocks[-1].start
        self._sl_baseline = ps.blocks[-1].slice

    def _eval_k(self, pvp):
        if self.nk == self.npb:
            return pvp
        else:
            pvp = atleast_2d(pvp)
            ks = zeros((pvp.shape[0], self.npb))
            for ipv in range(pvp.shape[0]):
                ks[ipv,:] =  splev(self.kx_all, splrep(self.kx_knots, pvp[ipv], s=0.0))
            return ks
            #return cubic_interpolation_pvp(self.kx_all, pvp, self.dkx)

    def _eval_bl(self, pvp):
        if self.nbl == self.npb:
            return pvp[:, :, newaxis] * self._bl_array
        else:
            return baseline(self.bx_all, pvp, self.dbx, self.npb, self.npt, self._bl_array)

    def transit_model(self, pv, copy=True):
        pv = atleast_2d(pv)
        ldp = pv[:, newaxis, 1:4]
        ldp[:, 0, 0] = clip(ldp[:, 0, 0], *self.ldmodel.sc.client.teffl)
        ldp[:, 0, 1] = clip(ldp[:, 0, 1], *self.ldmodel.sc.client.loggl)
        ldp[:, 0, 2] = clip(ldp[:, 0, 2], *self.ldmodel.sc.client.zl)
        t0 = pv[:, 4]
        p = pv[:, 5]
        k = self._eval_k(pv[:, self._sl_rratios])
        aor = as_from_rhop(pv[:, 0], p)
        inc = i_from_ba(pv[:, 6], aor)
        ecc = pv[:, 7] ** 2 + pv[:, 8] ** 2
        w = arctan2(pv[:, 8], pv[:, 7])
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
