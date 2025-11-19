# Changelog

All notable changes to ExoIris will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.20.0] - 2025-11-19

### Added
- Added the ability to model star spot crossings. The spots are modeled as generalized Gaussians parameterized by their 
  center time, amplitude in a given reference wavelength, full width at half maximum (FWHM), spot temperature in Kelvin,
  and spot shape. The spot amplitude along the wavelength is modeled using the theoretical BT-Settl spectra, the stellar
  effective temperature, the spot temperature, and the reference wavelength in which the spot amplitude is measured.
- Added the ability to model the contamination from unocculted spots and faculae (the transit light source effect, 
  or TLSE).
- Included two new radius ratio interpolation options: "nearest" and "linear".

## [0.19.2] - 2025-09-12

### Added

- Added a support for a free GP noise model. That is, the GP hyperparameters can be free parameters in the model.  
  However, this is somewhat slower than using the fixed GP model, since the GP needs to be computed for each model
  evaluation.
- Added a ``ExoIris.plot_white_gp_predictions`` method to visualize the GP predictions for the white light curves.

### Improved

- Improved TSData fits io to make sure all the properties (such as the noise and bias groups) are saved and read properly.
- Fixed a white light curve fitting but where the error array was not filtered correctly.
- Made `TSData` to raise a `ValueError` if either the time or wavelength array contains any non-finite values.

### Changed

- Renamed `TSDataSet` to `TSDataGroup` 

### Removed

## [0.18.0] - 2025-02-20

### Added

- Added `offset_group` variable to `TSData` that can be used in ExoIris to account for bias offsets between instruments
  and monochromatic transit-to-transit variations in transit depths caused by unocculted stellar heterogeneities, such
  as spots and faculae. 

## [0.17.0] - 2025-02-19

### Added

- Added support for masked data with GP noise model.
- Added an option to set the GP hyperparameters for an individual data set.
- Added  `white_times`, `white_fluxes`, `white_models`, and `white_errors` properties to `ExoIris` to access white ligth curve data.

### Fixed

- Fixed model loading with masked data.

## [0.16.0] - 2025-02-17

### Added
- **New Data Mask:** Introduced a general `mask` attribute in the `TSData` class to automatically flag valid data points (based on finite fluxes and errors).
- **Uncertainty Estimation:** Added the `estimate_average_uncertainties` method in `TSData` to compute per-wavelength uncertainties using first differences.
- **White Light Curve Processing:** Updated the `WhiteLPF` class to use `nanmean` and `isfinite` checks when computing the white light curve, ensuring only valid flux values are averaged.

### Changed
- **Transit Mask Renaming:** Renamed the old `ootmask` attribute to `transit_mask` throughout the codebase for clarity. This change affects plotting, normalization, binning, and file I/O.
- **Method Renaming:** Renamed `calculate_ootmask` in the `TSDataGroup` class to `mask_transit` to reflect the updated naming convention.
- **Normalization Enhancements:** Updated normalization methods (`normalize_to_poly` and `normalize_to_median`) to utilize the new `transit_mask` and `mask` attributes, improving the reliability of baseline fits.
- **Cropping Flexibility:** Added an `inplace` parameter to both `crop_wavelength` and `crop_time` methods, allowing users to choose between modifying the existing data or returning a new cropped instance.
- **Likelihood Function Update:** Modified the `lnlike_normal` function in `TSLPF` to accept an additional `mask` parameter and process only valid data points during likelihood calculations.
- **Outlier Handling:** Refined the `remove_outliers` method to flag outliers by setting affected fluxes and errors to NaN, rather than replacing them with median-filtered values.

### Removed
- **Removed Deprecated Method: `calculate_ootmask`**  
  This method (deprecated since v0.9) has been removed in favor of the new transit masking functionality.
- **Removed Deprecated Method: `normalize_baseline`**  
  The `normalize_baseline` method (deprecated since v0.9) has been removed; users should now use `normalize_to_poly`.
- **Removed Deprecated Method: `normalize_median`**  
  The deprecated `normalize_median` method has been removed. Its functionality is now available via `normalize_to_median`.
- **Removed Deprecated Method: `split_time`**  
  The `split_time` method, deprecated since v0.9, has been removed. Use `partition_time` instead.
