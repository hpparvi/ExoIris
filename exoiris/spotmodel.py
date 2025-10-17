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

from numpy import exp, fabs, log, inf, array, vstack, atleast_2d
from scipy.interpolate import RegularGridInterpolator
from numba import njit

from pytransit.stars import create_bt_settl_interpolator
from pytransit.param import GParameter, NormalPrior as N, UniformPrior as U

from exoiris.tsdata import TSData
from exoiris.util import bin2d

@njit
def spot_model(x, center, amplitude, fwhm, shape):
    c = fwhm / 2*(2*log(2))**(1/shape)
    return amplitude*exp(-(fabs(x-center) / c)**shape)


def create_interpolator(data: TSData, trange):
    ip = create_bt_settl_interpolator()
    lrange = array(data.bbox_wl) * 1e3
    mt = (ip.grid[0] >= trange[0]) & (ip.grid[0] <= trange[1])
    ml = (ip.grid[1] >= lrange[0]) & (ip.grid[1] <= lrange[1])

    teff = ip.grid[0][mt]
    wave = ip.grid[1][ml]
    flux = ip.values[mt][:, ml]

    wl_l_edges = wave - 0.5
    wl_r_edges = wave + 0.5

    bflux = bin2d(flux.T, flux.T, wl_l_edges*1e-3, wl_r_edges*1e-3, vstack([data._wl_l_edges, data._wl_r_edges]).T)[0].T
    return RegularGridInterpolator((teff, data.wavelength), bflux)


class SpotModel:
    def __init__(self, lpf: "TSLPF", epoch_group: int, tstar: float, ref_wavelength: float, teff_limits: tuple[float, float]):
        self.lpf = lpf
        self.epoch_group = epoch_group
        self.teff_limits = teff_limits
        self.tstar = tstar
        self.ref_wl = ref_wavelength
        self.pv_slice: None | slice = None

        self.sfluxes = []
        self.fratios = []
        self.times = []
        self.wavelengths = []
        self.data_ids = []

        lpf.nspots += 1

        self._init_data_and_interpolators()
        self._init_fratios(self.tstar, self.ref_wl)
        self._init_parameters()

    def _init_data_and_interpolators(self):
        for i, d in enumerate(self.lpf.data):
            if d.epoch_group == self.epoch_group:
                self.data_ids.append(i)
                self.times.append(d.time)
                self.wavelengths.append(d.wavelength)
                self.sfluxes.append(create_interpolator(d, self.teff_limits))

    def _init_fratios(self, tstar: float, ref_wavelength: float):
        self.fratios = []
        for sf in self.sfluxes:
            fr = deepcopy(sf)
            fr.values[:, :] = fr((tstar, fr.grid[1])) / fr.values
            fr.values[:, :] = fr.values / fr((fr.grid[0], ref_wavelength))[:, None]
            self.fratios.append(fr)

    def _init_parameters(self):
        i = self.lpf.nspots
        pspot = [GParameter(f"spc_{i:02d}", 'spot {i:02d} center', "d", U(0, 1), (0, inf)),
                 GParameter(f"spa_{i:02d}", 'spot {i:02d} amplitude', "", U(0, 1), (0, inf)),
                 GParameter(f"spw_{i:02d}", 'spot {i:02d} FWHM', "d", U(0, 1), (0, inf)),
                 GParameter(f"sps_{i:02d}", 'spot {i:02d} shape', "d", U(1, 5), (0, inf)),
                 GParameter(f"spt_{i:02d}", 'spot {i:02d} temperature', "K", U(3000, 6000), (0, inf))]
        ps = self.lpf.ps
        ps.thaw()
        ps.add_global_block(f'spot_{i:02d}', pspot)
        setattr(self.lpf, f"_start_spot_{i:02d}", ps.blocks[-1].start)
        setattr(self.lpf, f"_sl_spot_{i:02d}", ps.blocks[-1].slice)
        self.pv_slice = ps.blocks[-1].slice
        ps.freeze()

    def evaluate(self, pvs):
        pvs = atleast_2d(pvs)
        npv = pvs.shape[0]

        models = self.lpf.spot_model_fluxes
        if models[0].shape[0] != npv:
            raise ValueError('The _spot_models array has a wrong shape, it has not been initialized properly.')

        for ipv in range(npv):
            center, amplitude, fwhm, shape, tspot = pvs[ipv, self.pv_slice]
            for idata, t, fr in zip(self.data_ids, self.times, self.fratios):
                models[idata][ipv, :, :] += spot_model(t, center, amplitude, fwhm, shape) * fr((tspot, fr.grid[1]))[:, None]
