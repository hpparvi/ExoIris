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

import io
import warnings
from contextlib import redirect_stdout
import numpy as np

from petitRADTRANS import physical_constants as cst
from petitRADTRANS.spectral_model import SpectralModel

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from exoiris.exoiris import ExoIris
    from pytransit.lpf.logposteriorfunction import LogPosteriorFunction
    from pytransit.param import ParameterSet, GParameter, UniformPrior as UP, NormalPrior as NP

class PRTRetrieval(LogPosteriorFunction):
    def __init__(self, name: str, line_species, ei: ExoIris, rstar: float, mplanet: float, temperature_profile='isothermal',
                 cloud_model='deck', pres: int = 100, r: int | None = None, quiet: bool = False):
        super().__init__(name)
        if temperature_profile not in ('isothermal', 'guillot'):
            raise ValueError(f"temperature_profile must be 'isothermal' or 'guillot', not '{temperature_profile}'")
        if cloud_model not in ('deck', 'powerlaw'):
            raise ValueError(f"cloud_model must be 'deck' or 'powerlaw', not '{cloud_model}'")
        if r is None:
            self.line_species = line_species
        else:
            self.line_species = [ls + f'.R{r}' for ls in line_species]

        self.rstar = rstar
        self.mplanet = mplanet
        self.temperature_profile = temperature_profile
        self.cloud_model = cloud_model
        self.pres = pres
        self.ei = ei
        self.r = r

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if quiet:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    self._create_model()
            else:
                self._create_model()

        self._init_parameters()

        self.wavelengths = self.model()[0]
        self._loglike = ei.create_loglikelihood_function(self.wavelengths, 'radius_ratio', method='svd')

    def _init_parameters(self):
        self.ps = ParameterSet([])
        self._init_p_planet()
        self._init_p_temperature()
        self._init_p_clouds()
        self._init_p_species()

    def _init_p_planet(self):
        self.ps.thaw()
        ps = [GParameter('radius', 'Planet radius', 'R_Jup', NP(1.3, 0.05), (0.0, 2.5)),
              GParameter('logg', 'Planet log g', '', UP(2, 4), (2, 4))]
        self.ps.add_global_block('planet', ps)
        self.ps.freeze()
        self._sl_planet = self.ps.blocks[-1].slice
        self._st_planet = self.ps.blocks[-1].start

    def _init_p_clouds(self):
        self.ps.thaw()
        if self.cloud_model == 'deck':
            ps = [GParameter('cd_pc', 'log10 cloud top pressure', '', UP(-6, 2), (-6, 2)),
                  GParameter('cd_haze', 'log10 haze factor', '', UP(0, 3), (0, 3))]
        else:
            ps = [GParameter('cd_k350', 'log10 opacity at 350 nm', '', UP(-8, 4), (-8, 4)),
                  GParameter('cd_gamma', 'power-law opacity coefficient', '', UP(-13, 0), (-13, 0))]
        self.ps.add_global_block('clouds', ps)
        self.ps.freeze()
        self._sl_clouds = self.ps.blocks[-1].slice
        self._st_clouds = self.ps.blocks[-1].start

    def _init_p_temperature(self):
        self.ps.thaw()
        if self.temperature_profile == 'isothermal':
            ps = [GParameter('tp_teq', 'equilibrium temperature', 'K', UP(400, 1200), (300, 3500))]
        else:
            ps = [GParameter('tp_teq', 'equilibrium temperature', 'K', UP(400, 1200), (300, 3500)),
                  GParameter('tp_tin', 'intrinsic temperature', 'K', UP(400, 1200), (300, 3500)),
                  GParameter('tp_mo', 'guillot profile mean opacity', '', UP(0, 1), (0, 1)),
                  GParameter('tp_g', 'guillot profile gamma', '', UP(0, 1), (0, 1))]
        self.ps.add_global_block('temperature_profile', ps)
        self.ps.freeze()
        self._sl_tprof = self.ps.blocks[-1].slice
        self._st_tprof = self.ps.blocks[-1].start

    def _init_p_species(self):
        self.ps.thaw()
        ps = [GParameter(n, n, '', UP(-12, -0.2), (-12, -0.2)) for n in self.line_species]
        self.ps.add_global_block('line_species', ps)
        self.ps.freeze()
        self._sl_species = self.ps.blocks[-1].slice
        self._st_species = self.ps.blocks[-1].start

    def _c_temperature_profile(self, pv) -> dict:
        pv = pv[self._sl_tprof]
        pars = {}
        pars['temperature'] = pv[0]
        if self.temperature_profile == 'isothermal':
            pass
        elif self.temperature_profile == 'guillot':
            pars['intrinsic_temperature'] = pv[1]
            pars['guillot_temperature_profile_infrared_mean_opacity_solar_metallicity'] = pv[2]
            pars['guillot_temperature_profile_gamma'] = pv[3]
        return pars

    def _create_model(self):
        if self.temperature_profile == 'isothermal':
            temperature_parameters = {}
        else:
            temperature_parameters = {'intrinsic_temperature': 200.0,
                                      'guillot_temperature_profile_infrared_mean_opacity_solar_metallicity': 0.01,
                                      'guillot_temperature_profile_gamma': 0.4}
        if self.cloud_model == 'deck':
            cloud_parameters = {'opaque_cloud_top_pressure': 1e-2,
                                'haze_factor': 10.0}
        else:
            cloud_parameters = {'power_law_opacity_350nm': 0.01,
                                'power_law_opacity_coefficient': -4.0}
        self.sm = SpectralModel(
            pressures=np.logspace(-6, 2, self.pres),
            line_species=self.line_species,
            rayleigh_species=['H2', 'He'],
            gas_continuum_contributors=['H2--H2', 'H2--He'],
            wavelength_boundaries=[self.ei.data.wlmin, self.ei.data.wlmax],
            star_radius=self.rstar * cst.r_sun,
            planet_radius=1.0 * cst.r_jup_mean,
            planet_mass=self.mplanet * cst.m_jup,
            reference_gravity=391,
            reference_pressure=1e-2,
            temperature_profile_mode=self.temperature_profile,
            temperature=1000,
            use_equilibrium_chemistry=False,
            imposed_mass_fractions={s: 1e-4 for s in self.line_species},
            filling_species={'H2': 37, 'He': 12},
            **temperature_parameters,
            **cloud_parameters
        )

    def model(self, pv=None):
        if pv is not None:
            self.sm.model_parameters.update(self._c_temperature_profile(pv))
            self.sm.model_parameters['planet_radius'] = pv[self._st_planet] * cst.r_jup_mean
            self.sm.model_parameters['reference_gravity'] = 10 ** pv[self._st_planet + 1]
            if self.cloud_model == 'deck':
                self.sm.model_parameters['opaque_cloud_top_pressure'] = 10 ** pv[self._st_clouds]
                self.sm.model_parameters['haze_factor'] = 10 ** pv[self._st_clouds + 1]
            else:
                self.sm.model_parameters['power_law_opacity_350nm'] = 10 ** pv[self._st_clouds]
                self.sm.model_parameters['power_law_opacity_coefficient'] = pv[self._st_clouds + 1]
            for i, ls in enumerate(self.line_species):
                self.sm.model_parameters['imposed_mass_fractions'][ls] = 10 ** pv[self._st_species + i]
        wavelengths, radii = self.sm.calculate_spectrum(mode='transmission')
        return wavelengths[0] * 1e4, radii[0]

    def lnlikelihood(self, pv):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_spectrum = self.model(pv)[1]
        return self._loglike(model_spectrum / (self.rstar * cst.r_sun))
