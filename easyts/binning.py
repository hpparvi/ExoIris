from typing import Optional

from numpy import array, floor, linspace, vstack, nan, concatenate


class Binning:
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

    def _bin_r(self):
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

    def _bin_bw(self):
        if self.xmax - self.xmin <= self.bw:
            raise ValueError("The bin width (bw) should be smaller than the binning span.")
        self.nb = int(floor((self.xmax - self.xmin) / self.bw))
        self._bin_nb()
        self.nb = self.bins.shape[0]

    def _bin_nb(self):
        if self.nb <= 0:
            raise ValueError("The number of bins (nb) should be larger than zero.")
        e = linspace(self.xmin, self.xmax, num=self.nb)
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
            raise ValueError(f"Can't concatenate Binning and {other.__class__.__name__}")


class CompoundBinning:
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
            raise ValueError(f"Can't concatenate CompoundBinning and {other.__class__.__name__}")
        return cb
