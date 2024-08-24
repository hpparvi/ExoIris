#  EasyTS: fast, flexible, and easy exoplanet transmission spectroscopy in Python.
#  Copyright (C) 2024 Hannu Parviainen
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, setp
from numpy import array, log10, diff, sqrt, floor
from scipy.optimize import minimize

from pytransit import BaseLPF, LinearModelBaseline
from pytransit.orbits import as_from_rhop, i_from_ba, fold, i_from_baew, d_from_pkaiews, epoch

from .tslpf import TSLPF

class WhiteLPF(BaseLPF):
    def __init__(self, tsa: TSLPF):
        super().__init__('white', ['white'], tsa.time, tsa.flux.mean(0), covariates=[array([[]])])
        self.set_prior('tc', tsa.ps[tsa.ps.find_pid('tc')].prior)
        self.set_prior('p', tsa.ps[tsa.ps.find_pid('p')].prior)
        self.set_prior('rho', tsa.ps[tsa.ps.find_pid('rho')].prior)
        self.set_prior('b', tsa.ps[tsa.ps.find_pid('b')].prior)
        pr = tsa.ps[tsa._sl_rratios][0].prior
        if 'Uniform' in str(pr.__class__):
            self.set_prior('k2', 'UP', pr.a**2, pr.b**2)
        if 'Normal' in str(pr.__class__):
            self.set_prior('k2', 'NP', pr.mean**2, pr.std**2)
        self.set_prior('wn_loge_0', 'NP', log10(diff(self.ofluxa).std() / sqrt(2)), 1e-5)

    def _init_baseline(self):
        self._add_baseline_model(LinearModelBaseline(self))

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
        return pv[0] + pv[1]*epoch(self.timea.mean(), pv[0], pv[1])

    @property
    def transit_duration(self):
        pv = self._local_minimization.x
        a = as_from_rhop(pv[2], pv[1])
        i = i_from_ba(pv[3], a)
        t14 = d_from_pkaiews(pv[1], sqrt(pv[4]), a, i, 0., 0., 1, 14)
        return t14

    def plot(self, ax=None) -> Figure:
        if ax is None:
            fig, ax = subplots()
        else:
            fig = ax.get_figure()

        tref = floor(self.timea.min())
        fm = self.flux_model(self._local_minimization.x)
        ax.plot(self.timea - tref, self.ofluxa, '.k', alpha=0.25)
        ax.plot(self.timea - tref, fm, 'k')

        tc = self.transit_center
        t14 = self.transit_duration
        ax.axvline(tc - tref, ls='--', c='0.5')
        ax.axvline(tc - tref - 0.5*t14, ls='--', c='0.5')
        ax.axvline(tc - tref + 0.5*t14, ls='--', c='0.5')
        setp(ax, xlabel=f'Time - {tref:.0f} [BJD]', ylabel='Normalized flux', xlim=(self.timea.min()-tref, self.timea.max()-tref))
        return fig
