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

from typing import Optional

from numpy import array, floor, linspace, vstack, nan, concatenate


class Binning:
    """Class representing a binning of values within a given range.

    Parameters:
        xmin (float): The minimum value of the binning range.
        xmax (float): The maximum value of the binning range.
        nb (float, optional): The number of bins. Default is None, in which case the binning will be determined by bin width or resolution.
        bw (float, optional): The bin width. Default is None, in which case the binning will be determined by the number of bins or resolution.
        r (float, optional): The resolution. Default is None, in which case the binning will be determined by the number of bins or bin width.

    Raises:
        ValueError: If the binning is not initialized properly.

    Attributes:
        xmin (float): The minimum value of the binning range.
        xmax (float): The maximum value of the binning range.
        nb (int|Optional): The number of bins in the binning.
        bw (float|Optional): The width of each bin in the binning.
        r (float|Optional): The resolution of the binning.
        bins (numpy.ndarray|None): The array representing the bins in the binning.

    Methods:
        _bin_r(): Method to calculate the binning using the resolution.
        _bin_bw(): Method to calculate the binning using the bin width.
        _bin_nb(): Method to calculate the binning using the number of bins.

    Magic Methods:
        __repr__(): Returns a string representation of the Binning object.
        __add__(): Defines the concatenation behavior when adding a Binning object with another object.

    """
    def __init__(self, xmin: float, xmax: float, nb: Optional[float] = None, bw: Optional[float] = None, r: Optional[float] = None):
        if (nb is not None) + (bw is not None) + (r is not None) != 1:
            raise ValueError(
                'A binning needs to be initialized either with the number of bins (nb), bin width (bw), or resolution (r)')

        self.xmin: float = xmin
        self.xmax: float = xmax
        self.nb: Optional[int] = nb
        self.bw: Optional[float] = bw
        self.r: Optional[float] = r
        self.bins = None

        if r is not None:
            self._bin_r()
        elif bw is not None:
            self._bin_bw()
        else:
            self._bin_nb()

    def _bin_r(self) -> None:
        """Create bins for a given range using the resolution.

        Description:
            This method creates bins for a given range using the value of r. It calculates the start and end values of each bin
            based on the current xmin, xmax, and r values. The bins are then stored in the 'bins' attribute of the object instance.
            The number of bins is stored in the 'nb' attribute.
        """
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

        Raises:
            ValueError: If the bin width (bw) is greater than or equal to the binning span.
        """
        if self.xmax - self.xmin <= self.bw:
            raise ValueError("The bin width (bw) should be smaller than the binning span.")
        self.nb = int(floor((self.xmax - self.xmin) / self.bw))
        self._bin_nb()
        self.nb = self.bins.shape[0]

    def _bin_nb(self) -> None:
        """Create bins for a given range based on the number of bins `nb` and the range limits (xmin, xmax).

        Raises:
            ValueError: If the number of bins (nb) is less than or equal to zero.
        """
        if self.nb <= 0:
            raise ValueError("The number of bins (nb) should be larger than zero.")
        e = linspace(self.xmin, self.xmax, num=self.nb+1)
        self.bins = vstack([e[:-1], e[1:]]).T
        self.bw = (self.xmax - self.xmin) / self.nb

    def __repr__(self):
        return f"BinningRange {self.xmin:0.4f} - {self.xmax:.4f}: dx = {self.bw or nan:6.4f}, n = {self.nb:4d}, R = {self.r}"

    def __add__(self, other):
        if isinstance(other, Binning):
            return CompoundBinning([self, other])
        elif isinstance(other, CompoundBinning):
            return other + self
        else:
            raise TypeError(f"Can't concatenate Binning and {other.__class__.__name__}")


class CompoundBinning:
    """Class representing a compound binning.

    Attributes:
        binnings (list): List of binning objects.
        bins (ndarray): Concatenated bins from all the binning objects.
    """
    def __init__(self, binnings):
        self.binnings = binnings
        self.bins = concatenate([b.bins for b in self.binnings])

    def __repr__(self):
        return 'CompoundBinning:\n' + '\n'.join([b.__repr__() for b in self.binnings])

    def __add__(self, other):
        if isinstance(other, CompoundBinning):
            cb = CompoundBinning(self.binnings)
            cb.binnings.extend(other.binnings)
            cb.bins = concatenate([cb.bins, other.bins])
        elif isinstance(other, Binning):
            cb = CompoundBinning(self.binnings)
            cb.binnings.append(other)
            cb.bins = concatenate([cb.bins, other.bins])
        else:
            raise TypeError(f"Can't concatenate CompoundBinning and {other.__class__.__name__}")
        return cb
