#  ExoIris: fast, flexible, and easy exoplanet transmission spectroscopy in Python.
#  Copyright (C) 2025 Hannu Parviainen
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

from numpy import exp, fabs, log, inf, array, vstack, atleast_2d, nan, unique
from scipy.interpolate import RegularGridInterpolator
from numba import njit

from pytransit.stars import create_bt_settl_interpolator
from pytransit.param import GParameter, UniformPrior as U

from exoiris.tsdata import TSData
from exoiris.util import bin2d

@njit
def spot_model(x, center, amplitude, fwhm, shape):
    c = fwhm / 2*(2*log(2))**(1/shape)
    return amplitude*exp(-(fabs(x-center) / c)**shape)


def unocculted_spot_contamination_factor(area_ratio, flux_ratio):
    return 1. / (1. + area_ratio*(flux_ratio-1.))


def bin_stellar_spectrum_model(sp: RegularGridInterpolator, data: TSData):
    lrange = array(data.bbox_wl) * 1e3
    ml = (sp.grid[1] >= lrange[0]) & (sp.grid[1] <= lrange[1])

    teff = sp.grid[0]
    wave = sp.grid[1][ml]
    flux = sp.values[:, ml]

    wl_l_edges = wave - 0.5
    wl_r_edges = wave + 0.5

    bflux = bin2d(flux.T, flux.T, wl_l_edges*1e-3, wl_r_edges*1e-3, vstack([data._wl_l_edges, data._wl_r_edges]).T)[0].T
    return RegularGridInterpolator((teff, data.wavelength), bflux, bounds_error=False, fill_value=nan)


class SpotModel:
    def __init__(self, tsa, teff: float, wlref: float, include_tlse: bool = True):
        self.tsa = tsa
        self.teff = teff
        self.wlref = wlref
        self.include_tlse = include_tlse
        self.model_spectrum = ms = create_bt_settl_interpolator()
        self.fratios = []
        self.nfratios = []
        for d in tsa.data:
            mb = bin_stellar_spectrum_model(self.model_spectrum, d)
            fratio = RegularGridInterpolator(mb.grid, mb((teff, mb.grid[1])) / mb.values, bounds_error=False, fill_value=nan)
            n = ms((teff, wlref*1e3)) / ms((fratio.grid[0], wlref*1e3))
            nfratio = RegularGridInterpolator(fratio.grid, fratio.values / n[:, None], bounds_error=False, fill_value=nan)
            self.fratios.append(fratio)
            self.nfratios.append(nfratio)

        if self.include_tlse:
            self._init_tlse_parameters()

        self.nspots = 0
        self.spot_epoch_groups = []
        self.spot_data_ids = []
        self.spot_pv_slices = []

    def include_tlse(self):
        if self.include_tlse is False:
            self.include_tlse = True
            self._init_tlse_parameters()

    def _init_tlse_parameters(self):
        ps = [GParameter('tlse_tspot', 'Effective temperature of unocculted spots', 'K', U(1200, 7000), (1200, 7000))]
        ps.extend([GParameter(f"tlse_afrac_e{e:02d}", "Area fraction covered by unocculted spots", "", U(0,1), (0,1)) for e in unique(self.tsa.data.epoch_groups)])
        self.tsa.ps.thaw()
        self.tsa.ps.add_global_block(f'tlse', ps)
        setattr(self.tsa, "_start_tlse", self.tsa.ps.blocks[-1].start)
        setattr(self.tsa, "_sl_tlse", self.tsa.ps.blocks[-1].slice)
        self.tlse_pv_slice = self.tsa.ps.blocks[-1].slice
        self.tsa.ps.freeze()

    def add_spot(self, epoch_group: int):
        self.nspots += 1
        self.spot_epoch_groups.append(epoch_group)
        self.spot_data_ids.append([i for i, d in enumerate(self.tsa.data) if d.epoch_group == epoch_group])

        i = self.nspots
        pspot = [GParameter(f"spc_{i:02d}", 'spot {i:02d} center', "d", U(0, 1), (0, inf)),
                 GParameter(f"spa_{i:02d}", 'spot {i:02d} amplitude', "", U(0, 1), (0, inf)),
                 GParameter(f"spw_{i:02d}", 'spot {i:02d} FWHM', "d", U(0, 1), (0, inf)),
                 GParameter(f"sps_{i:02d}", 'spot {i:02d} shape', "d", U(1, 5), (0, inf)),
                 GParameter(f"spt_{i:02d}", 'spot {i:02d} temperature', "K", U(3000, 6000), (0, inf))]
        ps = self.tsa.ps
        ps.thaw()
        ps.add_global_block(f'spot_{i:02d}', pspot)
        setattr(self.tsa, f"_start_spot_{i:02d}", ps.blocks[-1].start)
        setattr(self.tsa, f"_sl_spot_{i:02d}", ps.blocks[-1].slice)
        self.spot_pv_slices.append(ps.blocks[-1].slice)
        ps.freeze()

    def apply_tlse(self, pvp, models):
        pvp = atleast_2d(pvp)[:, self.tlse_pv_slice]
        npv = pvp.shape[0]
        for d, m, fr in zip(self.tsa.data, models, self.fratios):
            for i in range(npv):
                tspot = pvp[i, 0]
                farea = pvp[i, 1+d.epoch_group]
                m[i, :, :] = (m[i, :, :] - 1.0) * unocculted_spot_contamination_factor(farea, 1/fr((tspot, fr.grid[1])))[:, None] + 1

    def apply_spots(self, pvp, models):
        pvp = atleast_2d(pvp)
        npv = pvp.shape[0]
        if models[0].shape[0] != npv:
            raise ValueError('The _spot_models array has a wrong shape, it has not been initialized properly.')

        for isp in range(self.nspots):
            for ipv in range(npv):
                center, amplitude, fwhm, shape, tspot = pvp[ipv, self.spot_pv_slices[isp]]
                for idata in self.spot_data_ids[isp]:
                    fr = self.nfratios[idata]
                    models[idata][ipv, :, :] += spot_model(self.tsa.data[idata].time, center, amplitude, fwhm, shape) * fr((tspot, fr.grid[1]))[:, None]
