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

from numpy import (
    exp,
    fabs,
    log,
    inf,
    array,
    vstack,
    atleast_2d,
    nan,
    unique,
    linspace,
    floor,
)
from scipy.interpolate import RegularGridInterpolator
from numba import njit

from pytransit.stars import create_bt_settl_interpolator, create_husser2013_interpolator
from pytransit.param import GParameter, UniformPrior as U

from exoiris.tsdata import TSData
from exoiris.util import bin2d


@njit
def spot_model(x, center, amplitude, fwhm, shape):
    c = fwhm / 2*(2*log(2))**(1/shape)
    return amplitude*exp(-(fabs(x-center) / c)**shape)


@njit
def interpolate_spectrum(teff, values, tgrid):
    t0 = tgrid[0]
    dt = tgrid[1] - tgrid[0]
    k = (teff - t0) / dt
    i = int(floor(k))
    a = k - floor(k)
    return (1.0-a)*values[i] + a*values[i+1]


@njit
def tlse(tphot, tspot, tfac, aspot, afac, spectra, tgrid):
    fphot = interpolate_spectrum(tphot, spectra, tgrid)
    fspot = interpolate_spectrum(tspot, spectra, tgrid)
    ffac  = interpolate_spectrum(tfac, spectra, tgrid)
    return 1.0 / (1.0 - aspot*(1.0 - fspot/fphot) - afac*(1.0 - ffac/fphot))


def spot_contrast(tphot, tspot, spectra, spnorm):
    fphot = interpolate_spectrum(tphot, spectra.values, spectra.grid[0])
    fspot = interpolate_spectrum(tspot, spectra.values, spectra.grid[0])
    norm = interpolate_spectrum(tspot, spnorm, spectra.grid[0])
    return (fphot / fspot) / norm


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
    def __init__(self, tsa, tphot: float, wlref: float, include_tlse: bool = True):
        self.tsa = tsa
        self.tphot = tphot
        self.wlref = wlref
        self.include_tlse = include_tlse

        ms = create_bt_settl_interpolator()
        new_teff_grid = linspace(*ms.grid[0][[0, -1]], 117)
        new_spectrum = array([ms((t, ms.grid[1])) for t in new_teff_grid])
        self.full_spectrum = ms = RegularGridInterpolator((new_teff_grid, ms.grid[1]), new_spectrum, bounds_error=False, fill_value=nan)

        wave = ms.grid[1] / 1e3
        m = (wave > wlref - 0.025) & (wave < wlref + 0.025)
        spot_norm = ms.values[:, m].mean(1)
        self.spot_norm = interpolate_spectrum(tphot, spot_norm, ms.grid[0]) / spot_norm

        self.binned_spectra = []
        for d in tsa.data:
            self.binned_spectra.append(bin_stellar_spectrum_model(self.full_spectrum, d))

        if self.include_tlse:
            self._init_tlse_parameters()

        self.nspots = 0
        self.spot_epoch_groups = []
        self.spot_data_ids = []
        self.spot_pv_slices = []

    def use_tlse(self):
        if self.include_tlse is False:
            self.include_tlse = True
            self._init_tlse_parameters()

    def _init_tlse_parameters(self):
        ps = [GParameter('tlse_tspot', 'Effective temperature of unocculted spots', 'K', U(1200, 7000), (1200, 7000))]
        ps.append(GParameter('tlse_tfac', 'Effective temperature of unocculted faculae', 'K', U(1200, 7000), (1200, 7000)))
        for e in unique(self.tsa.data.epoch_groups):
            ps.append(GParameter(f"tlse_aspot_e{e:02d}", "Area fraction covered by unocculted spots", "", U(0,1), (0,1)))
            ps.append(GParameter(f"tlse_afac_e{e:02d}", "Area fraction covered by unocculted faculae", "", U(0,1), (0,1)))
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
        for d, m, sp in zip(self.tsa.data, models, self.binned_spectra):
            for i in range(npv):
                tspot = pvp[i, 0]
                tfac = pvp[i, 1]
                fspot = pvp[i, 2+d.epoch_group*2]
                ffac = pvp[i, 3+d.epoch_group*2]
                m[i, :, :] = (m[i, :, :] - 1.0) * tlse(self.tphot, tspot, tfac, fspot, ffac, sp.values, sp.grid[0])[:, None] + 1

    def apply_spots(self, pvp, models):
        pvp = atleast_2d(pvp)
        npv = pvp.shape[0]
        if models[0].shape[0] != npv:
            raise ValueError('The _spot_models array has a wrong shape, it has not been initialized properly.')

        for isp in range(self.nspots):
            for ipv in range(npv):
                center, amplitude, fwhm, shape, tspot = pvp[ipv, self.spot_pv_slices[isp]]
                for idata in self.spot_data_ids[isp]:
                    models[idata][ipv, :, :] += (spot_model(self.tsa.data[idata].time, center, amplitude, fwhm, shape) *
                                                 spot_contrast(self.tphot, tspot, self.binned_spectra[idata], self.spot_norm)[:, None])
