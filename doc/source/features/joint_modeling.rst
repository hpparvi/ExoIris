Joint Modeling
==============

ExoIris is designed from the ground up for joint analysis of multiple spectrophotometric datasets. Whether
you have observations from different JWST instruments, HST gratings, ground-based spectrographs, multiple
visits of the same target, or any combination of these, ExoIris handles them all within a single,
self-consistent model.

Why Joint Modeling?
-------------------

Transmission spectra derived from a single transit observation with a single instrument can be affected by
instrumental systematics and astrophysical nuisances that are difficult to separate from the planetary
signal. Detector ramps, pointing jitter, stellar variability, and wavelength-dependent calibration offsets
can all imprint features on the measured spectrum that mimic or mask real atmospheric signatures. When only
one dataset is available, these contaminants are often degenerate with the quantities of interest.

Combining datasets from different instruments or epochs provides complementary constraints that help
resolve these ambiguities. Because different instruments have different systematics, spectral features that
appear consistently across datasets are far more likely to be real atmospheric signals. A feature that
shows up in both JWST NIRSpec and NIRISS data, for example, is much more credible than one seen in only a
single observation.

Joint modeling also reduces degeneracies between physical parameters. In a single-transit fit, limb
darkening, the planet-to-star radius ratio, and the impact parameter are correlated: changes in one can be
compensated by changes in the others without significantly affecting the fit quality. Transit photometry
taken in different passbands helps break these degeneracies. By sharing the orbital and stellar parameters
across all datasets while allowing each instrument its own systematics model, the joint fit places stronger
constraints on the physical parameters than any individual dataset could.

This is particularly important when combining data across wavelength ranges that do not overlap. If one
instrument covers the optical and another the near-infrared, neither dataset alone constrains the full
transmission spectrum. The shared transit model---with common orbital parameters, a continuous radius ratio
profile, and consistent limb darkening---ties everything together into a single coherent picture.

How ExoIris Organises Joint Models
----------------------------------

ExoIris uses three grouping concepts to give you fine-grained control over which parameters are shared and
which are independent across datasets. Noise groups control which datasets share the same noise model
parameters, so that each instrument's noise characteristics are handled independently. Epoch groups
associate datasets with specific transit events, allowing simultaneous observations to share a transit
center while observations from different nights get independent timing. Offset groups handle systematic
flux calibration differences between instruments through fitted multiplicative corrections.

These three axes---noise, epoch, and offset---cover the most common scenarios in multi-instrument transit
spectroscopy, from combining NIRISS orders within a single visit to stitching together data from entirely
different telescopes.
