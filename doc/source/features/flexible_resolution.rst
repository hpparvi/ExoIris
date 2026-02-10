Flexible Spectral Resolution
============================

One of the central ideas in ExoIris is the separation of data resolution from the transmission spectrum
resolution. Rather than fitting an independent radius ratio to each wavelength bin---which can mean
hundreds or thousands of free parameters at native detector resolution---ExoIris parameterises the
transmission spectrum as an interpolating function defined by a set of radius-ratio knots.

Why This Matters
----------------

At native spectral resolution, fitting an independent parameter per wavelength bin leads to an explosion in
dimensionality. Each bin adds a free parameter, yet neighbouring bins carry highly correlated information.
The result is an over-parameterised model that is expensive to sample and prone to noise-driven features
masquerading as real spectral structure.

The spline-knot approach solves this by letting you control the information content of the output spectrum
directly. Place fewer knots for a quick, low-resolution overview of the broadband shape, or add more knots
in regions where you expect narrow spectral features. The data resolution stays untouched---every pixel
contributes to the fit---but the number of free parameters reflects the actual information content you want
to extract.

This also enables multi-resolution exploration. You can start with a coarse grid of knots to establish the
broad spectral slope, then progressively add resolution in wavelength regions of interest. Knots can be
repositioned, added, or removed at any point during the analysis without reprocessing the data.

ExoIris offers several interpolation methods---from simple nearest-neighbour and linear schemes to smooth
spline-based options---so you can match the interpolation behaviour to the expected spectral morphology.
The same knot-based approach is used for limb darkening coefficients, with an independently configurable
interpolator.
