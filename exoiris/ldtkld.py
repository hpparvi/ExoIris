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

from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, setp
from numpy import sqrt, array, concatenate
from pytransit import LDTkLD as PTLDTkLD
from ldtk import BoxcarFilter

from .tsdata import TSData, TSDataSet


class LDTkLD(PTLDTkLD):
    def __init__(self, data: TSDataSet | TSData,
                 teff: tuple[float, float],
                 logg: tuple[float, float],
                 metal: tuple[float, float],
                 cache: str | Path | None = None,
                 dataset: str = 'vis-lowres') -> None:

        data = TSDataSet([data]) if isinstance(data, TSData) else data
        wl_edges = concatenate([array([d._wl_l_edges, d._wl_r_edges]) for d in data], axis=1).T
        filters = [BoxcarFilter(f"{0.5*(wla+wlb):08.5f}", wla*1e3, wlb*1e3) for wla, wlb in wl_edges]
        super().__init__(filters, teff, logg, metal, cache, dataset)
        self.wavelength = wl_edges.mean(1)
        self.dataset = dataset
        self.wlslices = []
        wl0 = 0
        for d in data:
            self.wlslices.append(slice(wl0, wl0+d.nwl))
            wl0 += d.nwl

    def plot_profiles(self, teff: float, logg: float, metal: float, x: str = 'mu', ax: Axes = None) -> Figure:
        """Plots the profiles of a star's surface brightness.

        Parameters
        ----------
        teff : float
            Effective temperature of the star.

        logg : float
            Logarithm of the surface gravity of the star.

        metal : float
            Metallicity of the star.

        x : str, optional
            The x-axis coordinate to plot against. Default is 'mu'.
            Must be either 'mu' or 'z'.

        ax : Axes, optional
            The matplotlib axes to plot the profiles on. If not provided,
            a new figure and axes will be created.

        Returns
        -------
        Figure
            The matplotlib figure containing the plot.

        Raises
        ------
        ValueError
            If the provided value of x is not 'mu' or 'z'.
        """
        if ax is None:
            fig, ax = subplots()
        else:
            fig = ax.get_figure()

        if x == 'z':
            xv = sqrt(1-self.mu**2)
            xl = 'z [R$_\star$]'
        elif x == 'mu':
            xv = self.mu
            xl = '$\mu$'
        else:
            raise ValueError("x must be either 'mu' or 'z'")

        ldi, _ = self(self.mu, array([teff, logg, metal]))
        l = ax.pcolormesh(xv, self.wavelength, ldi[0])
        fig.colorbar(l, label='Surface brightness')
        setp(ax, xlabel=xl, ylabel='Wavelength  [$\mu$m]' )
        fig.tight_layout()
        return fig
