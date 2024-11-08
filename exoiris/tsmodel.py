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

from typing import Union, List, Optional

from numpy import ndarray, atleast_2d, atleast_1d, array
from scipy.integrate import trapezoid

from pytransit.models.ldmodel import LDModel
from pytransit.models.numba.ldmodels import *
from pytransit import TSModel

from pytransit.models.roadrunner.model_trspec import tsmodel_serial, tsmodel_parallel

__all__ = ['TransmissionSpectroscopyModel']


class TransmissionSpectroscopyModel(TSModel):

    def evaluate(self, k: Union[float, ndarray], ldc: Union[ndarray, List],
                 t0: Union[float, ndarray], p: Union[float, ndarray], a: Union[float, ndarray],
                 i: Union[float, ndarray], e: Union[float, ndarray] = 0.0, w: Union[float, ndarray] = 0.0,
                 copy: bool = True) -> ndarray:
        """Evaluate the transit model for a set of scalar or vector parameters.

        Parameters
        ----------
        k
            Radius ratio(s) either as a single float, 1D vector, or 2D array.
        ldc
            Limb darkening coefficients as a 1D or 2D array.
        t0
            Transit center(s) as a float or a 1D vector.
        p
            Orbital period(s) as a float or a 1D vector.
        a
            Orbital semi-major axis (axes) divided by the stellar radius as a float or a 1D vector.
        i
            Orbital inclination(s) as a float or a 1D vector.
        e
            Orbital eccentricity as a float or a 1D vector.
        w
            Argument of periastron as a float or a 1D vector.

        Notes
        -----
        The model can be evaluated either for one set of parameters or for many sets of parameters simultaneously. In
        the first case, the orbital parameters should all be given as floats. In the second case, the orbital parameters
        should be given as a 1D array-like.

        Returns
        -------
        ndarray
        """
        k = atleast_2d(k)
        ldc = atleast_2d(ldc)
        t0, p, a, i, e, w = map(atleast_1d, (t0, p, a, i, e, w))
        npv = k.shape[0]

        # Limb darkening
        # --------------
        ldc = array(ldc)
        if npv > 1 and ldc.ndim != 3:
            raise ValueError("""The limb darkening parameters (ldp) should be given as a 3D array with shape [npv, npb, nldp] 
                    when evaluating the model for a set of parameters (npv > 1).""")
        if ldc.ndim == 1:
            ldc = ldc.reshape((1, 1, ldc.shape[1]))
        elif ldc.ndim == 2:
            ldc = ldc.reshape((1, ldc.shape[0], ldc.shape[1]))
        elif ldc.ndim == 3:
            pass
        else:
            raise ValueError()
        self.npb = npb = ldc.shape[1]

        if self.ldmodel is None:
            ldp, istar = ldc[:, :, :-1], ldc[:, :, -1]
        elif isinstance(self.ldmodel, LDModel):
            ldp, istar = self.ldmodel(self.mu, ldc)
        else:
            ldp = evaluate_ld(self.ldmodel, self.mu, ldc)

            if self.ldmmean is not None:
                istar = evaluate_ldi(self.ldmmean, ldc)
            else:
                istar = zeros((npv, npb))
                ldpi = evaluate_ld(self.ldmodel, self._ldmu, ldc)
                for ipv in range(npv):
                    for ipb in range(npb):
                        istar[ipv, ipb] = 2 * pi * trapezoid(self._ldz * ldpi[ipv, ipb], self._ldz)

        if self.interpolate:
            dk, dg, weights = self.dk, self.dg, self.weights
        else:
            dk, dg, weights = None, None, None

        if self.parallel:
            flux = tsmodel_parallel(self.time, k, t0, p, a, i, e, w, self.nsamples, self.exptimes,
                                    ldp, istar, weights, dk, self.klims[0], self.klims[1], self.ng, dg, self.ze,
                                    self.nthreads)
        else:
            flux = tsmodel_serial(self.time, k, t0, p, a, i, e, w, self.nsamples, self.exptimes,
                                  ldp, istar, weights, dk, self.klims[0], self.klims[1], self.ng, dg, self.ze)

        return flux

    def __call__(self, k: Union[float, ndarray], ldc: Union[ndarray, List],
                 t0: Union[float, ndarray], p: Union[float, ndarray], a: Union[float, ndarray],
                 i: Union[float, ndarray], e: Union[float, ndarray] = 0.0, w: Union[float, ndarray] = 0.0,
                 copy: bool = True) -> ndarray:
        return self.evaluate(k, ldc, t0, p, a, i, e, w, copy)