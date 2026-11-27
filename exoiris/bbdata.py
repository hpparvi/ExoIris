#  ExoIris: fast, flexible, and easy exoplanet transmission spectroscopy in Python.
#  Copyright (C) 2026 Hannu Parviainen
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
from typing import Sequence

import astropy.io.fits as pf
from numpy import array, asarray, average, ndarray, interp, isfinite, linspace, tile, all as np_all

from .ephemeris import Ephemeris
from .tsdata import TSData, TSDataGroup


class BBData(TSData):
    """Broadband photometric transit light curve data.

    Wraps a 1D broadband light curve (observed through a filter with known
    transmission profile) into ExoIris's 2D framework. The filter bandpass is
    discretised into ``n_wl`` wavelength bins, the transit model is evaluated on
    this 2D grid, and the log-likelihood compares the transmission-weighted mean
    of the model to the observed 1D flux.

    The rows of the internal 2D flux, error, and mask arrays are identical copies of
    the 1D light curve, and the 1D arrays used by the likelihood are views of the
    first row. Methods that operate along the wavelength axis are not defined for
    broadband data and raise a `NotImplementedError`.
    """

    def __init__(self, time: Sequence, wavelength: Sequence, transmission: Sequence,
                 n_wl: int, flux: Sequence, errors: Sequence, name: str,
                 noise_group: int = 0, transit_mask: ndarray | None = None,
                 ephemeris: Ephemeris | None = None, n_baseline: int = 1,
                 mask: ndarray | None = None, epoch_group: int = 0,
                 offset_group: int = 0, covs: ndarray | None = None) -> None:
        """
        Parameters
        ----------
        time
            1D array of time values.
        wavelength
            1D array of wavelength values where the filter transmission is defined.
        transmission
            1D array of filter transmission values at the given wavelengths.
        n_wl
            Number of wavelength resolution elements for the internal 2D representation.
        flux
            1D array of observed broadband flux values with shape ``(npt,)``.
        errors
            1D array of flux error values with shape ``(npt,)``.
        name
            Name for the data set.
        noise_group
            Noise group the data belongs to.
        transit_mask
            1D boolean array marking out-of-transit times (True = out of transit).
        ephemeris
            Transit ephemeris for automatic transit masking.
        n_baseline
            Number of baseline polynomial coefficients.
        mask
            1D boolean array marking valid time points (True = valid).
        epoch_group
            Epoch group identifier.
        offset_group
            Offset/bias group identifier.
        covs
            Pre-computed baseline covariance matrix.
        """
        wavelength = asarray(wavelength, dtype=float)
        transmission = asarray(transmission, dtype=float)
        flux = asarray(flux, dtype=float)
        errors = asarray(errors, dtype=float)

        # Store the original filter profile for serialisation
        self._filter_wavelength: ndarray = wavelength.copy()
        self._filter_transmission: ndarray = transmission.copy()
        self._n_wl: int = n_wl

        # Identify the effective bandpass (transmission > 1% of peak)
        threshold = 0.01 * transmission.max()
        in_band = transmission >= threshold
        wl_min = wavelength[in_band].min()
        wl_max = wavelength[in_band].max()

        # Resample the filter onto n_wl uniform wavelength bins
        wl_resampled = linspace(wl_min, wl_max, n_wl)
        transmission_resampled = interp(wl_resampled, wavelength, transmission)

        # Normalise weights to sum to 1
        weights = transmission_resampled / transmission_resampled.sum()

        self._weights: ndarray = weights
        self._transmission: ndarray = transmission_resampled
        self.is_broadband: bool = True

        # Build 2D arrays by tiling the 1D data across wavelength bins
        npt = flux.size
        fluxes_2d = tile(flux, (n_wl, 1))
        errors_2d = tile(errors, (n_wl, 1))

        # A broadband exposure is either valid or not, so the mask is collapsed to 1D
        # and tiled. This keeps all the rows of the 2D arrays identical.
        mask_1d = isfinite(flux) & isfinite(errors)
        if mask is not None:
            mask = asarray(mask, dtype=bool)
            mask_1d &= mask if mask.ndim == 1 else np_all(mask, axis=0)
        mask_2d = tile(mask_1d, (n_wl, 1))

        super().__init__(
            time=time, wavelength=wl_resampled, fluxes=fluxes_2d, errors=errors_2d,
            name=name, noise_group=noise_group, transit_mask=transit_mask,
            ephemeris=ephemeris, n_baseline=n_baseline, mask=mask_2d,
            epoch_group=epoch_group, offset_group=offset_group,
            mask_nonfinite_errors=True, covs=covs,
        )

    # The 1D broadband arrays are views of the first row of the 2D arrays rather than
    # separate copies, so they stay consistent with the 2D arrays when the inherited
    # TSData methods modify the data in place.

    @property
    def _bb_flux(self) -> ndarray:
        return self.fluxes[0]

    @property
    def _bb_errors(self) -> ndarray:
        return self.errors[0]

    @property
    def _bb_mask(self) -> ndarray:
        return self.mask[0]

    def _from_tsdata(self, d: TSData) -> 'BBData':
        """Create a BBData with the passband of this data set from a TSData with identical rows."""
        bb = BBData(time=d.time, wavelength=self._filter_wavelength, transmission=self._filter_transmission,
                    n_wl=self._n_wl, flux=d.fluxes[0], errors=d.errors[0], name=d.name,
                    noise_group=d.noise_group, transit_mask=d.transit_mask, ephemeris=d._ephemeris,
                    n_baseline=d.n_baseline, mask=d.mask[0], epoch_group=d.epoch_group,
                    offset_group=d.offset_group, covs=d.covs)
        bb._tm_l_edges = d._tm_l_edges
        bb._tm_r_edges = d._tm_r_edges
        return bb

    def crop_time(self, tmin: float, tmax: float, inplace: bool = True) -> 'BBData':
        """Crop the data to include only the time range between tmin and tmax.

        Parameters
        ----------
        tmin
            The minimum time value to crop.
        tmax
            The maximum time value to crop.
        inplace
            If True, the data will be modified in place, otherwise a new BBData object will be returned.
        """
        d = super().crop_time(tmin, tmax, inplace=inplace)
        return self if inplace else self._from_tsdata(d)

    def remove_fully_masked(self, inplace: bool = True) -> 'BBData':
        """Drop the masked exposures.

        Parameters
        ----------
        inplace
            If True, modify the current object in place and return `self`.
            Otherwise return a new `BBData` object.
        """
        if not self.mask.any():
            raise ValueError("All the exposures are masked.")
        d = super().remove_fully_masked(inplace=inplace)
        return self if inplace else self._from_tsdata(d)

    def partition_time(self, tlims: tuple[tuple[float, float]]) -> TSDataGroup:
        """Partition the data into n segments defined by tlims.

        Parameters
        ----------
        tlims
            The lower and upper time limits for each segment.

        Returns
        -------
        TSDataGroup
            A group of `BBData` objects, one per segment.
        """
        return TSDataGroup([self._from_tsdata(d) for d in super().partition_time(tlims).data])

    def bin_time(self, binning=None, nb: int | None = None, bw: float | None = None,
                 estimate_errors: bool = False) -> 'BBData':
        """Bin the data along the time axis.

        Parameters
        ----------
        binning
            The binning method to use.
        nb
            Number of bins.
        bw
            Bin width in seconds.
        estimate_errors
            Should the uncertainties be estimated from the data.

        Returns
        -------
        BBData
        """
        return self._from_tsdata(super().bin_time(binning, nb, bw, estimate_errors=estimate_errors))

    def bin(self, wave_binning=None, time_binning=None,
            wave_nb: int | None = None, wave_bw: float | None = None, wave_r: float | None = None,
            time_nb: int | None = None, time_bw: float | None = None,
            estimate_errors: bool = False) -> 'BBData':
        """Bin the data along the time axis.

        Broadband data can be binned only in time. Giving any of the wavelength binning arguments
        raises a `NotImplementedError`.

        Parameters
        ----------
        time_binning
            The binning method to use.
        time_nb
            Number of bins.
        time_bw
            Bin width in seconds.
        estimate_errors
            Should the uncertainties be estimated from the data.

        Returns
        -------
        BBData
        """
        if not (wave_binning is None and wave_nb is None and wave_bw is None and wave_r is None):
            self._no_wavelength_axis('bin')
        return self.bin_time(time_binning, time_nb, time_bw, estimate_errors=estimate_errors)

    def _no_wavelength_axis(self, method: str):
        raise NotImplementedError(f"BBData.{method} is not defined for broadband data: the wavelength axis "
                                  f"is an internal discretisation of the filter passband.")

    def bin_wavelength(self, *args, **kwargs):
        """Not defined for broadband data."""
        self._no_wavelength_axis('bin_wavelength')

    def crop_wavelength(self, *args, **kwargs):
        """Not defined for broadband data."""
        self._no_wavelength_axis('crop_wavelength')

    def mask_outliers(self, *args, **kwargs):
        """Not defined for broadband data.

        The outliers are identified along the wavelength axis, which carries no information for a
        broadband light curve. Mask the outliers using the `mask` argument when creating the object.
        """
        self._no_wavelength_axis('mask_outliers')

    def create_white_light_curve(self, data=None) -> ndarray:
        """Create a white light curve.

        Averages a 2D array over the passband using the filter transmission as weights. For the fluxes this
        returns the observed broadband light curve, and for a model array the band-integrated model.

        Parameters
        ----------
        data
            A 2D array with the same shape as the internal flux array to average instead of the fluxes.

        Returns
        -------
        ndarray
            The light curve as a 1D array with one value per exposure.

        Raises
        ------
        ValueError
            If the shape of `data` does not match the shape of the internal flux array.
        """
        if data is not None and data.shape != self.fluxes.shape:
            raise ValueError("The data must have the same shape as the 2D flux array.")
        data = data if data is not None else self.fluxes
        return average(data, axis=0, weights=self._weights)

    def export_fits(self) -> pf.HDUList:
        """Generate a `~astropy.io.fits.HDUList` containing HDUs storing the broadband data.

        Returns
        -------
        ~astropy.io.fits.HDUList
        """
        time = pf.ImageHDU(self.time, name=f'time_{self.name}')
        flux = pf.ImageHDU(self._bb_flux, name=f'flux_{self.name}')
        errs = pf.ImageHDU(self._bb_errors, name=f'errs_{self.name}')
        fwav = pf.ImageHDU(self._filter_wavelength, name=f'fwav_{self.name}')
        ftrn = pf.ImageHDU(self._filter_transmission, name=f'ftrn_{self.name}')
        covs = pf.ImageHDU(self.covs, name=f'covs_{self.name}')
        ootm = pf.ImageHDU(self.transit_mask.astype(int), name=f'ootm_{self.name}')
        mask = pf.ImageHDU(self._bb_mask.astype(int), name=f'mask_{self.name}')
        # The wavelength edges follow from the filter profile, so only the time edges are stored.
        tmed = pf.ImageHDU(array([self._tm_l_edges, self._tm_r_edges]), name=f'tmed_{self.name}')

        flux.header['BBDATA'] = True
        flux.header['NWL'] = self._n_wl
        flux.header['NGROUP'] = self.noise_group
        flux.header['NBASEL'] = self.n_baseline
        flux.header['EPGROUP'] = self.epoch_group
        flux.header['OFFGROUP'] = self.offset_group

        return pf.HDUList([time, flux, errs, fwav, ftrn, covs, ootm, mask, tmed])

    @staticmethod
    def import_fits(name: str, hdul: pf.HDUList) -> 'BBData':
        """Import a broadband data set from a `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        name
            The name of the dataset to be imported.
        hdul
            The `~astropy.io.fits.HDUList` containing the data.

        Returns
        -------
        BBData
        """
        time = hdul[f'TIME_{name}'].data.astype('d')
        flux = hdul[f'FLUX_{name}'].data.astype('d')
        errors = hdul[f'ERRS_{name}'].data.astype('d')
        fwav = hdul[f'FWAV_{name}'].data.astype('d')
        ftrn = hdul[f'FTRN_{name}'].data.astype('d')
        ootm = hdul[f'OOTM_{name}'].data.astype(bool)
        mask = hdul[f'MASK_{name}'].data.astype(bool)

        try:
            covs = hdul[f'COVS_{name}'].data.astype('d')
        except KeyError:
            covs = None

        header = hdul[f'FLUX_{name}'].header
        n_wl = header['NWL']
        noise_group = header.get('NGROUP', 0)
        epoch_group = header.get('EPGROUP', 0)
        offset_group = header.get('OFFGROUP', 0)
        n_baseline = header.get('NBASEL', 1)

        bb = BBData(
            time=time, wavelength=fwav, transmission=ftrn, n_wl=n_wl,
            flux=flux, errors=errors, name=name, noise_group=noise_group,
            transit_mask=ootm, n_baseline=n_baseline, mask=mask,
            epoch_group=epoch_group, offset_group=offset_group, covs=covs,
        )
        # The time edges are missing from the files saved before they were stored.
        if f'TMED_{name}' in hdul:
            bb._tm_l_edges, bb._tm_r_edges = hdul[f'TMED_{name}'].data.astype('d')
        return bb

    def save(self, fname: Path, overwrite: bool = True):
        """Save the broadband data to a FITS file.

        Parameters
        ----------
        fname
            Output file path.
        overwrite
            Whether to overwrite an existing file.
        """
        hdul = pf.HDUList([pf.PrimaryHDU()] + self.export_fits())
        hdul[0].header['BBDATA'] = self.name
        hdul.writeto(fname, overwrite=overwrite)

    @staticmethod
    def load(fname: Path | str, noise_group: int | None = None) -> "BBData":
        """Load a BBData object from a FITS file.

        Parameters
        ----------
        fname
            Path to the FITS file.
        noise_group
            Noise group to assign to the loaded data set. The noise group stored in the file is used if None.

        Returns
        -------
        BBData
        """
        from .tsdata import _load
        d = _load(fname)
        if noise_group is not None:
            d.noise_group = noise_group
        return d
