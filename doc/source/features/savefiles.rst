Reproducible Analyses
=====================

Reproducibility is a cornerstone of good science, and ExoIris makes it straightforward. Every analysis can
be saved to a single multi-extension FITS file that contains everything needed to reproduce the results,
inspect the fit, or continue the analysis later.

The saved file captures the complete state of the analysis: the ephemeris and model configuration, all
parameter priors, every input dataset with its wavelengths, fluxes, uncertainties, masks, and group
assignments, the white light curve fit, any spot model configuration, the optimiser population, and the
full MCMC posterior chains. Nothing is left out---a saved file is a self-contained record of the entire
analysis.

The FITS format was chosen because it is the standard data format in astronomy, is self-documenting, and
can be inspected with standard tools. A saved analysis can be fully reconstructed from the file alone,
allowing you to resume sampling, adjust settings, or extract results without repeating any previous steps.
