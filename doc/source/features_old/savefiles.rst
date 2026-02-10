Reproducible Analyses
=====================

Reproducibility is a cornerstone of good science, and ExoIris makes it straightforward. Every analysis can be
saved to a single multi-extension FITS file that contains everything needed to reproduce the results, inspect
the fit, or continue the analysis later.

What Gets Saved
---------------

A call to ``iris.save()`` writes a FITS file with the following contents:

- **Primary header**: Analysis name, ephemeris, interpolation methods, noise model, limb darkening model
  configuration, and free knot indices.
- **Knot arrays**: Radius ratio knot wavelengths (``K_KNOTS``) and limb darkening knot wavelengths
  (``LD_KNOTS``).
- **Priors**: The full set of parameter priors, preserved exactly as configured.
- **Data**: Each dataset with its time array, wavelengths, fluxes, errors, covariate matrix, masks, and
  group assignments.
- **White light curve** (if fitted): Observed and model white light curves with the best-fit parameters.
- **Spot model** (if configured): Stellar temperature, reference wavelength, TLSE flag, and per-spot
  epoch associations.
- **Optimiser state**: The differential evolution population, so you can resume optimisation.
- **MCMC chains**: The full posterior samples from emcee, stored as a table with named columns.

Loading and Continuing
----------------------

A saved analysis can be fully reconstructed with a single call:

.. code-block:: python

   from exoiris import load_model
   iris = load_model('my_analysis.fits')

This restores the ``ExoIris`` object with all data, parameters, priors, and (if present) the optimiser
and MCMC states. You can then continue sampling, adjust knots, or extract results without repeating
any previous steps.

The FITS format was chosen because it is the standard data format in astronomy, is self-documenting,
and can be inspected with standard tools like ``astropy.io.fits`` or ``fitsinfo``.
