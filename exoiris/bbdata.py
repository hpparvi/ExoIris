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
from numpy import asarray, average, ndarray, interp, linspace, tile, all as np_all, array

from .ephemeris import Ephemeris
from .tsdata import TSData


class BBData(TSData):
    """Broadband photometric transit light curve data.

    Wraps a 1D broadband light curve (observed through a filter with known
    transmission profile) into ExoIris's 2D framework. The filter bandpass is
    discretised into ``n_wl`` wavelength bins, the transit model is evaluated on
    this 2D grid, and the log-likelihood compares the transmission-weighted mean
    of the model to the observed 1D flux.
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

        # Store 1D broadband data
        self._bb_flux: ndarray = flux.copy()
        self._bb_errors: ndarray = errors.copy()
        self._weights: ndarray = weights
        self._transmission: ndarray = transmission_resampled
        self.is_broadband: bool = True

        # Build 2D arrays by tiling the 1D data across wavelength bins
        npt = flux.size
        fluxes_2d = tile(flux, (n_wl, 1))
        errors_2d = tile(errors, (n_wl, 1))

        # Handle mask: accept 1D and tile to 2D
        mask_2d = None
        if mask is not None:
            mask = asarray(mask)
            if mask.ndim == 1:
                mask_2d = tile(mask, (n_wl, 1))
            else:
                mask_2d = mask

        super().__init__(
            time=time, wavelength=wl_resampled, fluxes=fluxes_2d, errors=errors_2d,
            name=name, noise_group=noise_group, transit_mask=transit_mask,
            ephemeris=ephemeris, n_baseline=n_baseline, mask=mask_2d,
            epoch_group=epoch_group, offset_group=offset_group,
            mask_nonfinite_errors=True, covs=covs,
        )

        # 1D time mask: valid if all wavelength bins are unmasked at that time
        self._bb_mask: ndarray = np_all(self.mask, axis=0)

    def create_white_light_curve(self, data=None) -> ndarray:
        """Create a white light curve."""
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

        flux.header['BBDATA'] = True
        flux.header['NWL'] = self._n_wl
        flux.header['NGROUP'] = self.noise_group
        flux.header['NBASEL'] = self.n_baseline
        flux.header['EPGROUP'] = self.epoch_group
        flux.header['OFFGROUP'] = self.offset_group

        return pf.HDUList([time, flux, errs, fwav, ftrn, covs, ootm, mask])

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

        return BBData(
            time=time, wavelength=fwav, transmission=ftrn, n_wl=n_wl,
            flux=flux, errors=errors, name=name, noise_group=noise_group,
            transit_mask=ootm, n_baseline=n_baseline, mask=mask,
            epoch_group=epoch_group, offset_group=offset_group, covs=covs,
        )

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

        Returns
        -------
        BBData
        """
        from .tsdata import _load
        d = _load(fname)
        if noise_group is not None:
            d.noise_group = noise_group
        return d
