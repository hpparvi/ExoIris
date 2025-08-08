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
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, setp
from numpy import (
    log10,
    diff,
    sqrt,
    floor,
    ceil,
    arange,
    newaxis,
    nanmean,
    isfinite,
    nan,
    where,
    nanstd,
    inf,
    atleast_2d,
    repeat,
    array,
    average,
    unique,
)
from scipy.optimize import minimize

from pytransit import BaseLPF, LinearModelBaseline
from pytransit.orbits import as_from_rhop, i_from_ba, fold, i_from_baew, d_from_pkaiews, epoch
from pytransit.param import GParameter, NormalPrior as NP, UniformPrior as UP
from pytransit.lpf.lpf import map_ldc

from .tslpf import TSLPF

class WhiteLPF(BaseLPF):
    def __init__(self, tsa: TSLPF):
        self.tsa = tsa
        fluxes, times, errors = [], [], []
        for t, f, e in zip(tsa.data.times, tsa.data.fluxes, tsa.data.errors):
            weights = where(isfinite(f) & isfinite(e), 1/e**2, 0.0)
            mf = average(where(isfinite(f), f, 0), axis=0, weights=weights)
            me = sqrt(1 / weights.sum(0))
            m = isfinite(mf)
            times.append(t[m])
            fluxes.append(mf[m])
            errors.append(me[m])
        covs = [(t-t.mean())[:, newaxis] for t in times]
        self.std_errors = errors
        self.neps = max(self.tsa.data.epoch_groups) + 1

        pbs = unique(tsa.data.noise_groups).astype('<U21')
        super().__init__('white', pbs, times, fluxes,
                         covariates=covs, wnids=tsa.data.noise_groups, pbids=tsa.data.noise_groups)

        self.tm.epids = array(self.tsa.data.epoch_groups)

        for i in range(self.neps):
            self.set_prior(f'tc_{i:02d}', tsa.ps[tsa.ps.find_pid(f'tc_{i:02d}')].prior)
        self.set_prior('p', tsa.ps[tsa.ps.find_pid('p')].prior)
        self.set_prior('rho', tsa.ps[tsa.ps.find_pid('rho')].prior)
        self.set_prior('b', tsa.ps[tsa.ps.find_pid('b')].prior)
        pr = tsa.ps[tsa._sl_rratios][0].prior
        if 'Uniform' in str(pr.__class__):
            self.set_prior('k2', 'UP', pr.a**2, pr.b**2)
        if 'Normal' in str(pr.__class__):
            self.set_prior('k2', 'NP', pr.mean**2, pr.std**2)
        ngids = tsa.data.noise_groups[self.lcids]
        for i in range(tsa.data.n_noise_groups):
            self.set_prior(f'wn_loge_{i}', 'NP', log10(diff(self.ofluxa[ngids==i]).std() / sqrt(2)), 0.1)

    def _init_baseline(self):
        self._add_baseline_model(LinearModelBaseline(self))

    def _init_p_orbit(self):
        """Orbit parameter initialisation.
        """
        porbit = [
            GParameter('p', 'period', 'd', NP(1.0, 1e-5), (0, inf)),
            GParameter('rho', 'stellar_density', 'g/cm^3', UP(0.1, 25.0), (0, inf)),
            GParameter('b', 'impact_parameter', 'R_s', UP(0.0, 1.0), (0, 1))]
        self.ps.add_global_block('orbit', porbit)

        ptc = [GParameter(f'tc_{i:02d}', f'transit_center_{i:02d}', '-', NP(0.0, 0.1), (-inf, inf)) for i in
               range(self.neps)]
        self.ps.add_global_block('tc', ptc)
        self._pid_tc = repeat(self.ps.blocks[-1].start, self.nlc)
        self._start_tc = self.ps.blocks[-1].start
        self._sl_tc = self.ps.blocks[-1].slice

    def transit_model(self, pv, copy=True):
        pv = atleast_2d(pv)
        ldc = map_ldc(pv[:, self._sl_ld])
        zero_epoch = pv[:, self._sl_tc] - self._tref
        period = pv[:, 0]
        smaxis = as_from_rhop(pv[:, 1], period)
        inclination = i_from_ba(pv[:, 2], smaxis)
        radius_ratio = sqrt(pv[:, self._sl_k2])
        return self.tm.evaluate(radius_ratio, ldc, zero_epoch, period, smaxis, inclination)

    def optimize(self, pv0=None, method='powell', maxfev: int = 5000):
        if pv0 is None:
            if self.de is not None:
                pv0 = self.de.minimum_location
            else:
                pv0 = self.ps.mean_pv
        res = minimize(lambda pv: -self.lnposterior(pv), pv0, method=method, options={'maxfev':maxfev})
        self._local_minimization = res

    @property
    def transit_center(self):
        pv = self._local_minimization.x
        return pv[3] + pv[0]*epoch(self.times[0].mean(), pv[3], pv[0])

    @property
    def transit_duration(self):
        pv = self._local_minimization.x
        a = as_from_rhop(pv[1], pv[0])
        i = i_from_ba(pv[2], a)
        t14 = d_from_pkaiews(pv[0], sqrt(pv[4]), a, i, 0., 0., 1, 14)
        return t14

    def plot(self, axs=None, figsize=None, ncols=2) -> Figure:
        if axs is None:
            nrows = int(ceil(self.nlc / ncols))
            fig, axs = subplots(nrows, ncols, figsize=figsize, sharey='all', squeeze=False, constrained_layout=True)
        else:
            fig = axs[0].get_figure()

        tref = floor(self.timea.min())
        fm = self.flux_model(self._local_minimization.x)
        t14 = self.transit_duration
        pv = self._local_minimization.x

        for i, sl in enumerate(self.lcslices):
            ax = axs.flat[i]
            tc = pv[0] + pv[1]*epoch(self.times[i].mean(), pv[0], pv[1])
            ax.plot(self.timea[sl] - tref, self.ofluxa[sl], '.k', alpha=0.25)
            ax.plot(self.timea[sl] - tref, fm[sl], 'k')
            ax.axvline(tc - tref, ls='--', c='0.5')
            ax.axvline(tc - tref - 0.5*t14, ls='--', c='0.5')
            ax.axvline(tc - tref + 0.5*t14, ls='--', c='0.5')
            setp(ax, xlabel=f'Time - {tref:.0f} [BJD]', xlim=(self.times[i].min()-tref, self.times[i].max()-tref))
        setp(axs[:,0], ylabel='Normalized flux')
        return fig
