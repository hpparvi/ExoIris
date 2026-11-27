# Changelog

All notable changes to ExoIris will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- The save files now store the wavelength and time bin edges of the data sets and the hyperparameters of the
  `fixed_gp` noise model. Files saved with earlier versions can still be read.

### Changed
- Changing the radius ratio knots now removes the free radius ratio knot locations with a warning, because the
  locations refer to the knots by index.
- `ExoIris.save` now follows its `overwrite` argument and raises an `OSError` if the file exists and `overwrite` is
  False. The default is now True, so a plain `save()` call overwrites an existing file as before.
- `TSLPF.set_data` now raises a `ValueError` if the new data has a different number of data sets than the old data.

### Deprecated
- `ExoIris.set_baseline_prior` has no effect and gives a `DeprecationWarning`. The baseline coefficients are solved
  analytically and are not model parameters. The method raised an `AttributeError` before.

### Fixed
- Fixed the `fixed_gp` and `free_gp` likelihoods for `BBData`. The rows of the internal 2D representation were treated
  as independent light curves, so the broadband data were counted `n_wl` times.
- Fixed the `LDTkLD` limb darkening evaluation modifying the parameter array given to it. The stellar parameters were
  clipped in place, which changed the optimizer and sampler populations.
- Fixed the order of the MCMC chains read by `load_model`. The samples were assigned to wrong walkers and steps, so
  sampling continued from a loaded model did not start from the last step of the saved chains.
- Fixed `ExoIris.save` dropping the optimizer population and the MCMC chains from a model read with `load_model`, and
  after the radius ratio or limb darkening knots were changed.
- Fixed the GP noise models using outdated data. The GPs kept the fluxes, uncertainties, and masks from the time the
  noise model was set, so the baseline normalization was not seen and masking outliers raised an error.
- Fixed `ExoIris.create_initial_population` with the `mcmc` source, which drew the population from the whole chain
  instead of taking the last step.
- Fixed `LogLikelihood` with the `eigh` method keeping the smallest eigenvalues instead of the largest when the
  number of eigenvalues was limited.
- Fixed the spot and TLSE models reading arbitrary memory for temperatures outside the stellar spectrum grid. A
  temperature outside the grid now gives a NaN model that the likelihood rejects, and `ExoIris.initialize_spots`
  raises a `ValueError` for a photosphere temperature outside the grid.
- Fixed the free radius ratio knot locations being lost when the knots were changed. The model kept reading the
  locations from unrelated parameters. Changing the limb darkening knots now keeps the free knot locations.
- Fixed the knot order prior being added again each time the free radius ratio knot locations were set.
- Fixed `load_model` failing for models that use the `LDTkLD` limb darkening model.
- Fixed `ExoIris.free_radius_ratio_knot_locations` and `TSLPF.set_radius_ratio_limits`, which raised an error on
  every call.
- Fixed `ExoIris.set_ldtk_prior` rejecting the `quadratic-tri` limb darkening model.
- Fixed `ExoIris.set_data` failing when the new data has a different wavelength grid than the old data.
- Fixed `TSData.mask_transit` raising a `ValueError` for a zero epoch of 0.0, and creating a 2D transit mask from
  exposure index limits.
- Fixed `ExoIris.fit` failing when the radius ratios have a prior without limits, such as a normal prior.
- Fixed `LogLikelihood` failing when created from a mean spectrum and a covariance matrix with the default method.
  An unknown method now raises a `ValueError`.
- Fixed changing the radius ratio or limb darkening knots failing for a model with spots and an existing optimizer
  population or MCMC chains.
- Fixed the transit centre and duration lines in the white light curve plot, which were drawn using the orbital
  period and stellar density instead of the transit centre.
- Fixed `bin1d` for bins that start inside the last data point. The bin was given a wrong mix of the last two
  data points.
- Fixed `TSData.crop_wavelength` with `inplace=False` replacing the covariates with the default ones.
- Fixed a memory leak in the `white_profiled` noise model for non-finite covariates.
- Fixed the white light curve GP predictions shown by `ExoIris.plot_white_gp_predictions`. They were calculated
  for the last kernel the optimizer tried instead of the optimal one, and were kept after a new white light curve
  fit.
- Fixed adding to a `CompoundBinning` modifying the original object, and `Binning + CompoundBinning` putting the
  bins in the wrong order.
- Fixed `util.create_binning` creating one bin too few for each range.

## [1.2.0] - 2026-11-27

### Added
- Added `nsamples` and `likelihood` arguments to `ExoIris.set_ldtk_prior` for choosing the number of stellar parameter
  samples and the LDTk likelihood used to fit the limb darkening coefficients.
- Added a `cloud_model` option to `PRTRetrieval` with `deck` and `powerlaw` cloud descriptions, and a haze factor for
  the cloud deck model.
- Added a prior for the `quadratic-tri` limb darkening model that rejects coefficients outside [0, 1]. The prior is
  evaluated for the coefficients interpolated to the data wavelengths, because the interpolating spline can leave the
  valid range even when all the knot values are inside it.

### Changed
- ExoIris now requires PyTransit 2.9.2 or later and LDTk 1.9.3 or later.
- `ExoIris.set_ldtk_prior` now draws 1000 stellar parameter samples and uses the `diagonal` LDTk likelihood by
  default. The coefficients fitted with the `reduced-rank` likelihood can vary strongly between calls.
- The log posterior no longer evaluates the model for parameter vectors that the priors reject.
- Posteriors sampled with the `quadratic-tri` limb darkening model can change slightly near the edges of the
  coefficient range because of the new prior.
- Widened the equilibrium temperature prior and bounds in `PRTRetrieval`, and renamed the cloud top pressure
  parameter from `pc` to `cd_pc`.

### Fixed
- Fixed a memory leak that filled the memory when fitting with the `quadratic-tri` limb darkening model. The
  interpolated limb darkening coefficients could become negative, which made the transit model return NaNs, and each
  such model evaluation leaked memory.
- Fixed a memory leak in the `white_profiled` noise model for data with fully masked wavelengths.
- Fixed a memory leak in the `white_marginalized` noise model for non-finite models, zero uncertainties, and
  wavelengths with fewer valid points than baseline covariates.
- Fixed the `BBData` time cropping, time binning, time partitioning, masked exposure removal, and baseline
  normalization. The methods either returned a plain `TSData` object that was no longer modelled as broadband data,
  or left the light curve used by the likelihood unchanged. The methods that work along the wavelength axis
  (`bin_wavelength`, `crop_wavelength`, and `mask_outliers`) now raise a `NotImplementedError` for `BBData`.
- Fixed a `ZeroDivisionError` in `ExoIris.fit_white` and `TSData.create_white_light_curve` for data with fully masked
  exposures. `TSData.create_white_light_curve` now returns a NaN for a fully masked exposure.
- Fixed the white light curve uncertainties of `BBData`, which were too small by a factor of sqrt(`n_wl`).
- Fixed a crash when creating an `ExoIris` object with the `free_gp` noise model.
- Fixed the `free_gp` noise model losing its hyperparameters when the radius ratio or limb darkening knots were
  changed.
- Fixed the `free_gp` likelihood, which used the `gp_log_sigma` and `gp_log_rho` parameters as the kernel amplitude
  and length scale without exponentiating them. The parameters are base-10 logarithms.
- Fixed a `ZeroDivisionError` in the spot model for a zero spot width or shape. A zero-width spot now has no effect,
  and a negative width or a non-positive shape gives a NaN model that the likelihood rejects.
- Updated the transit model to work with the current PyTransit RoadRunner model.
- Fixed the bounds of the species abundance parameters and the units of the Guillot profile parameters in
  `PRTRetrieval`.

## [1.1.1] - 2026-05-29

### Added
- Added the `BBData` class for modelling broadband light curves jointly with spectroscopic time series, with
  `save`, `load`, and `create_white_light_curve` methods.
- Added `save` and `load` methods to `TSData` and `TSDataGroup`.
- Added `TSData.remove_fully_masked()` method for removing fully masked wavelengths and exposures.
- Added an experimental `PRTRetrieval` class for atmospheric retrieval with petitRADTRANS 3.
- Added feature overview pages and tutorials on spot crossings and the transit light source effect to the
  documentation.

### Changed
- The linear radius ratio interpolation now holds the radius ratio constant outside the knot wavelength range instead
  of extrapolating.

### Fixed
- Limited the number of eigenvalues in `LogLikelihood` to `nk` to improve numerical stability.

## [1.1.0] - 2026-02-05

### Added
- Added `TSData.create_white_light_curve()` method for creating white light curves with weighted averaging.
- Added `white_profiled` and `white_marginalized` noise model options in TSLPF:
  - `white_profiled` (default): computes likelihood with baseline coefficients estimated via least-squares profiling
  - `white_marginalized`: uses analytically marginalized log-likelihood that integrates out baseline coefficients
- Added `TSData.bin()` method for combined wavelength and time binning.

### Fixed
- Fixed the error propagation in 2D binning.
- Fixed the least-squares baseline fitting for wavelengths where the fit fails.
- Several minor bugs.

## [1.0.0] - 2026-01-28

### Added
- Added support for setting custom interpolation models for limb darkening via `set_limb_darkening_interpolator` method.
- Added support for setting custom interpolation models for radius ratios via `set_radius_ratio_interpolator` method.
- Added ability to set priors for offset parameters in the analysis workflow.
- Added interpolation type tracking in FITS header metadata (`INTERP_LD` for limb darkening interpolation).

### Changed
- Improved reinitialization logic for MCMC and differential evolution populations with updated limb darkening coefficients and knots.
- Refactored interpolation handling in `TSLPF` to separately manage radius ratio and limb darkening interpolators.
- Updated documentation for API methods and workflow.

### Fixed
- Fixed incorrect FWHM scaling in `spot_model` function. The scaling factor now correctly uses `log(4)^(1/shape)` for 
  proper full width at half maximum calculations in generalized Gaussian spot models.

## [0.23.1] - 2025-12-18

### Fixed
- Fixed least-squares baseline fitting for transit models with NaNs.
- Added "bspline-cubic" interpolation option as an alias for "bspline". The "bspline" option will be removed in a future 
  release.

## [0.23.0] - 2025-12-16

### Changed
- Switched baseline modeling to a least-squares approach.

### Fixed
- Corrected prior loading for parameter sets.
- Validated `samples` before posterior spectrum calculation to prevent runtime errors.

## [0.22.0] - 2025-12-13

### Added
- Added support for free-k knot configuration in spline models.
- Added SVD solver options in the `loglikelihood.LogLikelihood` class.
- Added `bspline-quadratic` interpolation option.

### Improved
- Improved transmission spectrum and log-likelihood methods for robustness and performance.
- Cleaned up and refactored the `LogLikelihood` class.
- Updated limb darkening parameter plotting.

### Fixed
- Added a safety check for uninitialized `white_gp_models` in white
  light curve processing.

## [0.21.0] - 2025-11-24

### Added
- Added a `loglikelihood.LogLikelihood` class for reduced-rank Gaussian log-likelihood calculation.
- Added `ExoIris.create_loglikelihood_function` method to generate log-likelihood instances for atmospheric retrieval.
- Added `ExoIris.radius_ratio_spectrum` and `ExoIris.area_ratio_spectrum` methods to compute posterior transmission 
  spectrum distributions for given wavelengths.
- Added `exoiris.util.create_mock_model` utility function for generating mock transmission spectra.

## [0.20.0] - 2025-11-19

### Added
- Added the ability to model star spot crossings. The spots are modeled as generalized Gaussians parameterized by their 
  center time, amplitude in a given reference wavelength, full width at half maximum (FWHM), spot temperature in Kelvin,
  and spot shape. The spot amplitude along the wavelength is modeled using the theoretical BT-Settl spectra, the stellar
  effective temperature, the spot temperature, and the reference wavelength in which the spot amplitude is measured.
- Added the ability to model the contamination from unocculted spots and faculae (**the transit light source effect**,
  or TLSE). This is modeled also using BT-Settl spectra following the formalism described by Rackham et al. (2018).
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
