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

from collections.abc import Sequence

from numpy import array, floor, linspace, vstack, nan, concatenate, ndarray


class Binning:
    """Class representing a homogeneous binning within a given range.

    The `Binning` is defined by its minimum and maximum values (`xmin` and `xmax`), and either the number
    of bins (`nb`), bin width (`bw`), or resolving power (`r`). If the `Binning` is
    initialized giving the number of bins, the range will be divided into `nb` equally
    wide bins. If the `Binning` is initialized giving the bin width, the range will be
    divided into bins that are as close as `bw` as possible. Finally, if the `Binning` is
    initialized with the resolving power, the bin widths aim to follow the x/xv relation.

    Parameters
    ----------
    xmin
        The minimum value of the range of values to be binned.
    xmax
        The maximum value of the range of values to be binned.
    nb
        The number of bins to be used for binning the range of values.
    bw
        The bin width to be used for binning the range of values.
    r
        The resolving power  to be used for binning the range of values.

    Raises
    ------
    ValueError
        When none or more than one of `nb`, `bw`, and `r` are provided.

    Attributes
    ----------
    xmin
        The minimum value of the range of values to be binned.
    xmax
        The maximum value of the range of values to be binned.
    nb
        The number of bins.
    bw
        The bin width.
    r
        The resolving power.
    bins
        An array of left and right bin edge values.

    """

    def __init__(self, xmin: float, xmax: float,
                 nb: float | None = None,
                 bw: float | None = None,
                 r: float | None = None) -> None:

        if (nb is not None) + (bw is not None) + (r is not None) != 1:
            raise ValueError(
                'A binning needs to be initialized either with the number of bins (nb), bin width (bw), or resolution (r)')

        self.xmin: float = xmin
        self.xmax: float = xmax
        self.nb: int | None = nb
        self.bw: float | None = bw
        self.r: float | None = r

        self.bins : ndarray | None = None
        """A (nb, 2) array of left and right bin edge values."""

        if r is not None:
            self._bin_r()
        elif bw is not None:
            self._bin_bw()
        else:
            self._bin_nb()

    def _bin_r(self) -> None:
        """Create bins for a given range using the resolution."""
        bins = []
        x0 = self.xmin
        while True:
            x1 = x0 * (2 * self.r + 1) / (2 * self.r - 1)
            bins.append([x0, min(x1, self.xmax)])
            x0 = x1
            if x1 >= self.xmax:
                break
        self.bins = array(bins)
        self.nb = self.bins.shape[0]

    def _bin_bw(self) -> None:
        """Create bins for a given range based on the bin width `bw` and the range limits (xmin, xmax).

        Raises
        ------
        ValueError
            If the bin width (bw) is greater than or equal to the binning span.
        """
        if self.xmax - self.xmin <= self.bw:
            raise ValueError("The bin width (bw) should be smaller than the binning span.")
        self.nb = int(floor((self.xmax - self.xmin) / self.bw))
        self._bin_nb()
        self.nb = self.bins.shape[0]

    def _bin_nb(self) -> None:
        """Create bins for a given range based on the number of bins `nb` and the range limits (xmin, xmax).

        Raises
        ------
        ValueError
            If the number of bins (nb) is less than or equal to zero.
        """
        if self.nb <= 0:
            raise ValueError("The number of bins (nb) should be larger than zero.")
        e = linspace(self.xmin, self.xmax, num=self.nb+1)
        self.bins = vstack([e[:-1], e[1:]]).T
        self.bw = (self.xmax - self.xmin) / self.nb

    def __repr__(self) -> str:
        return f"Binning {self.xmin:0.4f} - {self.xmax:.4f}: dx = {self.bw or nan:6.4f}, n = {self.nb:4d}, R = {self.r}"

    def __add__(self, other: 'Binning') -> 'CompoundBinning':
        if isinstance(other, Binning):
            return CompoundBinning([self, other])
        elif isinstance(other, CompoundBinning):
            return other + self
        else:
            raise TypeError(f"Can't concatenate Binning and {other.__class__.__name__}")


class CompoundBinning:
    """A class representing complex heterogeneous binning.

    This class allows for creating a compound binning by combining multiple binning objects.

    Parameters
    ----------
    binnings
        The list of Binning objects used for creating the bins.

    Attributes
    ----------
    binnings
        List of binning objects.
    bins
        Array of bin edges from all binning objects.
    """
    def __init__(self, binnings: Sequence[Binning]) -> None:
        self.binnings = binnings
        self.bins = concatenate([b.bins for b in self.binnings])

    def __repr__(self):
        return 'CompoundBinning:\n' + '\n'.join([b.__repr__() for b in self.binnings])

    def __add__(self, other) -> 'CompoundBinning':
        if isinstance(other, CompoundBinning):
            cb = CompoundBinning(self.binnings)
            cb.binnings.extend(other.binnings)
            cb.bins = concatenate([cb.bins, other.bins])
        elif isinstance(other, Binning):
            cb = CompoundBinning(self.binnings)
            cb.binnings.append(other)
            cb.bins = concatenate([cb.bins, other.bins])
        else:
            raise TypeError(f"Cannot concatenate CompoundBinning and {other.__class__.__name__}")
        return cb
