Flux Offsets
============

When combining data from different instruments or observing modes, the datasets may have different
absolute flux calibrations. A small mismatch in the flux scale between instruments---even at the fraction
of a percent level---can introduce systematic biases in the joint fit if left unaccounted for.

ExoIris handles this through a system of offset groups. One group serves as the reference, and all other
groups receive a fitted multiplicative correction. The offset is parameterised as:

.. math::

   F(\lambda, t) = b + (1 - b) \times F_\mathrm{transit}(\lambda, t)

where :math:`b` is the offset parameter, constrained by a tight prior centred at zero. This
parameterisation preserves the shape of the transit while allowing a small shift in the overall flux level,
which is the expected behaviour for well-calibrated data with slightly inconsistent absolute scales.

Offset groups are useful whenever you combine datasets from different JWST instruments, spectral orders, or
observing programmes. They let you fit all the data jointly without requiring perfectly matched flux scales
beforehand. If all your data come from the same instrument and mode, offsets are typically unnecessary.
