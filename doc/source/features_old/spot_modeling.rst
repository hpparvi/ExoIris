Spot Crossing & TLSE
====================

Stellar surface inhomogeneities---spots and faculae---are a significant source of contamination in
transmission spectroscopy. ExoIris provides built-in support for modelling both spot-crossing events
during transit and the transit light source effect (TLSE) from unocculted active regions.

Spot Crossing Events
--------------------

When a transiting planet crosses a spot or bright region on the stellar surface, it produces a
characteristic bump (or dip) in the light curve. ExoIris models these events using a generalised
Gaussian profile with five parameters per spot:

- **Center** (``spc``): The time of the spot crossing.
- **Amplitude** (``spa``): The strength of the crossing event.
- **FWHM** (``spw``): The full width at half maximum of the event.
- **Shape** (``sps``): Controls the profile shape (values near 1 give a Gaussian, higher values produce flatter tops).
- **Temperature** (``spt``): The effective temperature of the spot.

The spot contrast is wavelength-dependent, computed from BT-Settl stellar atmosphere models. This means
the model correctly captures how spot crossings look different at different wavelengths---a critical detail
for transmission spectroscopy.

.. code-block:: python

   iris.initialize_spots(tstar=5500, wlref=1.5)
   iris.add_spot(epoch_group=0)  # Add a spot to the first epoch

Transit Light Source Effect
---------------------------

The TLSE arises because the region of the stellar disk occulted by the planet may have a different
spectrum than the integrated stellar disk. Unocculted spots and faculae modify the effective stellar
spectrum, biasing the measured transmission spectrum.

ExoIris models the TLSE using effective temperatures and covering fractions for both spots and faculae:

- **Spot temperature** (``tlse_tspot``): Effective temperature of unocculted spots.
- **Facula temperature** (``tlse_tfac``): Effective temperature of unocculted faculae.
- **Spot area fraction** (``tlse_aspot_eXX``): Fraction of the visible disk covered by spots (per epoch).
- **Facula area fraction** (``tlse_afac_eXX``): Fraction covered by faculae (per epoch).

The TLSE correction factor is wavelength-dependent and computed from the same BT-Settl model grid,
ensuring physical consistency with the spot crossing model.

.. code-block:: python

   iris.initialize_spots(tstar=5500, wlref=1.5, include_tlse=True)
