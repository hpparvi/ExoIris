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
taken in different passbands helps break these degeneracies. By sharing the orbital and stellar parameters across all
datasets while allowing each instrument its own systematics model, the joint fit places stronger
constraints on the physical parameters than any individual dataset could.

This is particularly important when combining data across wavelength ranges that do not overlap. If one
instrument covers the optical and another the near-infrared, neither dataset alone constrains the full
transmission spectrum. The shared transit model---with common orbital parameters, a continuous radius ratio
profile, and consistent limb darkening---ties everything together into a single coherent picture.

Loading Data with TSData
------------------------

The primary data container in ExoIris is ``TSData``. Each ``TSData`` object holds one spectroscopic time
series: a 2D array of fluxes with dimensions wavelength by time, along with the corresponding wavelength
and time arrays, uncertainties, and metadata. When you create a ``TSData`` object, you assign it to groups
that tell ExoIris how to treat it in the joint model:

- **noise_group**: Datasets sharing a noise group use the same noise model parameters.
- **epoch_group**: Datasets sharing an epoch group were observed during the same transit and share the same transit center time.
- **offset_group**: Datasets in different offset groups get independent multiplicative flux offsets to account for calibration differences.

For example, consider a target observed with JWST NIRISS (orders 1 and 2) in one visit and NIRSpec G395H
in another:

.. code-block:: python

   from exoiris import TSData

   # NIRISS order 1 and 2 from the same visit:
   # same epoch (0), different noise groups (0, 1), different offset groups (0, 1)
   niriss_o1 = TSData(time_1, wl_1, flux_1, err_1, name='niriss-o1',
                       noise_group=0, epoch_group=0, offset_group=0)
   niriss_o2 = TSData(time_2, wl_2, flux_2, err_2, name='niriss-o2',
                       noise_group=1, epoch_group=0, offset_group=1)

   # NIRSpec from a different visit:
   # different epoch (1), its own noise group (2) and offset group (2)
   nirspec = TSData(time_3, wl_3, flux_3, err_3, name='nirspec-g395h',
                     noise_group=2, epoch_group=1, offset_group=2)

In this setup, the two NIRISS orders share a transit center because they were observed simultaneously, but
each has its own noise and offset parameters. The NIRSpec observation gets a separate epoch, noise group,
and offset group because it was taken at a different time with a different instrument.

Combining Datasets
------------------

Combining ``TSData`` objects into a ``TSDataGroup`` is as simple as adding them together with the ``+``
operator:

.. code-block:: python

   data = niriss_o1 + niriss_o2 + nirspec

This is transparent to the end user---you work with individual ``TSData`` objects and the addition creates
a ``TSDataGroup`` that keeps track of everything. The resulting group is then passed directly to
``ExoIris``:

.. code-block:: python

   from exoiris import ExoIris

   iris = ExoIris(data, ldmodel='power-2-pm')


Grouping Mechanisms
-------------------

The three grouping mechanisms give you fine-grained control over which parameters are shared and which are
independent across datasets.

**Noise groups** control which datasets share the same white noise model parameters. Datasets from the same
instrument and observing mode typically share a noise group, while datasets from different instruments get
separate noise parameters. This lets the model account for instrument-specific noise characteristics without
requiring separate fits.

**Epoch groups** associate datasets with specific transit epochs. Each epoch group gets its own transit center
parameter (``tc_00``, ``tc_01``, ...), enabling the model to account for transit timing variations
automatically. Multiple datasets from the same epoch---such as two spectral orders observed simultaneously---share
the same transit center.

**Offset groups** handle systematic flux offsets between datasets. Different instruments or observing modes
may have different absolute calibrations. The reference group (group 0) has no free offset parameter, and
all other groups get an additive offset fitted relative to it.
