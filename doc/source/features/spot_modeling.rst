Spot Crossing & TLSE
====================

Stellar surface inhomogeneities---spots and faculae---are a significant source of contamination in
transmission spectroscopy. Their effects are wavelength-dependent, which means they can introduce spurious
spectral features that mimic or mask real atmospheric signatures. ExoIris provides built-in support for
modelling both spot-crossing events during transit and the transit light source effect (TLSE) from
unocculted active regions.

Spot Crossing Events
--------------------

When a transiting planet crosses a spot or bright region on the stellar surface, it produces a
characteristic bump or dip in the light curve. The amplitude and shape of this event depend on the
contrast between the spot and the surrounding photosphere---a contrast that varies with wavelength. A spot
that is prominent in the optical may be nearly invisible in the infrared. Ignoring these events does not
just leave residuals in the light curve; it biases the measured transit depth as a function of wavelength,
directly contaminating the transmission spectrum.

ExoIris models spot crossings using a generalised Gaussian profile whose chromatic contrast is computed
from the BT-Settl stellar atmosphere model grid. This ensures that the wavelength dependence of the
crossing event is physically motivated rather than treated as an arbitrary free function. The model
captures the essential physics---center time, amplitude, width, profile shape, and spot temperature---while
remaining compact enough to fit alongside the transit and baseline parameters.

Transit Light Source Effect
---------------------------

Even when no spots are directly occulted, the presence of active regions on the visible stellar disk alters
the effective spectrum of the light blocked by the planet. The planet occults a local patch of the
photosphere, but the out-of-transit baseline is set by the disk-integrated flux, which includes
contributions from spots and faculae. This mismatch---the transit light source effect---introduces a
wavelength-dependent bias in the inferred transmission spectrum.

ExoIris models the TLSE through effective temperatures and covering fractions for both spots and faculae,
using the same BT-Settl model grid to ensure physical consistency with the spot crossing model. Because the
covering fractions can vary between epochs, the model naturally accommodates stellar variability across
multi-epoch datasets.
