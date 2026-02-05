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

from numba import njit
from numpy import zeros, isfinite, where, nan, sqrt, ndarray, int32, float64

@njit
def bin2d(v: ndarray, e: ndarray,
             wl_l: ndarray, wl_r: ndarray,
             tm_l: ndarray, tm_r: ndarray,
             wl_bins: ndarray, tm_bins: ndarray,
             estimate_errors: bool = False) -> tuple[ndarray, ndarray, ndarray]:
    """Bin 2D spectrophotometry data in both wavelength and time dimensions.

    Parameters
    ----------
    v : ndarray
        A 2D spectrophotometry array with shape (n_wavelength, n_exposure).
    e : ndarray
        A 2D array of uncertainties matching the shape of `v`.
    wl_l : ndarray
        A 1D array of left wavelength edges for each spectral data point.
    wl_r : ndarray
        A 1D array of right wavelength edges for each spectral data point.
    tm_l : ndarray
        A 1D array of left time edges for each exposure.
    tm_r : ndarray
        A 1D array of right time edges for each exposure.
    wl_bins : ndarray
        A 2D array (n_wl_bins, 2) of wavelength bin edges [left, right].
    tm_bins : ndarray
        A 2D array (n_tm_bins, 2) of time bin edges [left, right].
    estimate_errors : bool, optional
        If True, estimate uncertainties from data scatter. Default is False.

    Returns
    -------
    tuple of ndarrays
        - bv: Binned values with shape (n_wl_bins, n_tm_bins).
        - be: Binned uncertainties with shape (n_wl_bins, n_tm_bins).
        - bn: Number of original finite pixels in each bin with shape (n_wl_bins, n_tm_bins).
    """
    n_wl_bins = len(wl_bins)
    n_tm_bins = len(tm_bins)
    n_wl = v.shape[0]
    n_tm = v.shape[1]

    bv = zeros((n_wl_bins, n_tm_bins))
    be = zeros((n_wl_bins, n_tm_bins))
    bn = zeros((n_wl_bins, n_tm_bins), dtype=int32)

    # Pre-compute masks and cleaned data
    nonfin_mask = isfinite(v)
    v_clean = where(nonfin_mask, v, 0.0)
    e2_clean = where(nonfin_mask, e**2, 0.0)
    nonfin_weights = nonfin_mask.astype(float64)

    # Pre-compute time bin indices and weights
    tm_il_arr = zeros(n_tm_bins, dtype=int32)
    tm_ir_arr = zeros(n_tm_bins, dtype=int32)
    tm_weights_all = zeros((n_tm_bins, n_tm))

    for itm_bin in range(n_tm_bins):
        tm_bel, tm_ber = tm_bins[itm_bin]

        # Find first time index
        tm_il = 0
        for j in range(n_tm - 1):
            if tm_l[j + 1] > tm_bel:
                tm_il = j
                break
        else:
            tm_il = n_tm - 1

        tm_il_arr[itm_bin] = tm_il

        # Calculate time weights with ROBUST overlap logic
        # Overlap = max(0, min(pixel_right, bin_right) - max(pixel_left, bin_left))
        # We perform the loop starting from the found index until pixels no longer overlap

        tm_idx = tm_il
        # Iterate until pixel starts after bin ends or we run out of pixels
        while tm_idx < n_tm:
            # Optimization: If pixel starts after bin ends, stop.
            if tm_l[tm_idx] >= tm_ber:
                # But careful: tm_il search might land us on a pixel that starts after bin
                # if the bin is in a gap. If so, we just want to record the index and break.
                break

            # Calculate overlap
            # min(pixel_r, bin_r)
            r_bound = tm_r[tm_idx] if tm_r[tm_idx] < tm_ber else tm_ber
            # max(pixel_l, bin_l)
            l_bound = tm_l[tm_idx] if tm_l[tm_idx] > tm_bel else tm_bel

            if r_bound > l_bound:
                tm_weights_all[itm_bin, tm_idx] = r_bound - l_bound

            # If this pixel goes past the bin, we are done with this bin
            if tm_r[tm_idx] >= tm_ber:
                break

            tm_idx += 1

        tm_ir_arr[itm_bin] = min(tm_idx, n_tm - 1)

    # Allocation moved outside loop
    wl_weights = zeros(n_wl)

    # Process wavelength bins
    wl_start = 0
    for iwl_bin in range(n_wl_bins):
        wl_bel, wl_ber = wl_bins[iwl_bin]

        # Reset weights for this iteration (faster than re-allocating)
        # We only need to zero out the range we are about to use or used previously?
        # Actually, since we calculate specific indices [wl_il, wl_ir],
        # we can just zero them out at the end of the loop, or re-zero the whole array.
        # Given n_wl is usually small (spectroscopy), zeroing whole array is fine.
        wl_weights[:] = 0.0

        # Find first wavelength index
        wl_il = wl_start
        for wl_il in range(wl_start, n_wl - 1):
            if wl_l[wl_il + 1] > wl_bel:
                break

        # Calculate wavelength weights (Same robust logic as time)
        wl_idx = wl_il
        while wl_idx < n_wl:
            if wl_l[wl_idx] >= wl_ber:
                break

            r_bound = wl_r[wl_idx] if wl_r[wl_idx] < wl_ber else wl_ber
            l_bound = wl_l[wl_idx] if wl_l[wl_idx] > wl_bel else wl_bel

            if r_bound > l_bound:
                wl_weights[wl_idx] = r_bound - l_bound

            if wl_r[wl_idx] >= wl_ber:
                break
            wl_idx += 1

        wl_ir = min(wl_idx, n_wl - 1)

        # Optimization for next bin search
        wl_start = wl_il

        # Process all time bins for this wavelength bin
        for itm_bin in range(n_tm_bins):
            tm_il = tm_il_arr[itm_bin]
            tm_ir = tm_ir_arr[itm_bin]

            total_weight = 0.0
            sum_w2 = 0.0
            weighted_sum = 0.0
            weighted_e2_sum = 0.0
            npt = 0

            for i in range(wl_il, wl_ir + 1):
                w_wl = wl_weights[i]
                if w_wl <= 0: continue # Skip if no overlap

                for j in range(tm_il, tm_ir + 1):
                    # Combine weights
                    w = w_wl * tm_weights_all[itm_bin, j] * nonfin_weights[i, j]

                    if w > 0:
                        total_weight += w
                        sum_w2 += w * w
                        weighted_sum += w * v_clean[i, j]
                        weighted_e2_sum += w * w * e2_clean[i, j]
                        npt += 1

            bn[iwl_bin, itm_bin] = npt

            if total_weight > 0:
                bv[iwl_bin, itm_bin] = vmean = weighted_sum / total_weight

                if estimate_errors:
                    # Need at least 2 points (or effective degrees of freedom) to estimate variance
                    if npt > 1 and total_weight**2 > sum_w2:
                        var_sum = 0.0
                        for i in range(wl_il, wl_ir + 1):
                            w_wl = wl_weights[i]
                            if w_wl <= 0: continue

                            for j in range(tm_il, tm_ir + 1):
                                w = w_wl * tm_weights_all[itm_bin, j] * nonfin_weights[i, j]
                                if w > 0:
                                    var_sum += w * (v_clean[i, j] - vmean) ** 2

                        # 1. Calculate Unbiased Weighted Sample Variance (S^2)
                        # S^2 = Sum(w*(x-mean)^2) / (V1 - V2/V1)
                        denominator = total_weight - (sum_w2 / total_weight)
                        sample_variance = var_sum / denominator

                        # 2. Calculate Standard Error of the Mean
                        # SEM = sqrt( S^2 * (V2 / V1^2) )
                        be[iwl_bin, itm_bin] = sqrt(sample_variance * (sum_w2 / (total_weight * total_weight)))
                    else:
                        be[iwl_bin, itm_bin] = nan
            else:
                bv[iwl_bin, itm_bin] = nan
                be[iwl_bin, itm_bin] = nan

    return bv, be, bn