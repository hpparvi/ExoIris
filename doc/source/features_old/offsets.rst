Flux Offsets
============

When combining data from different instruments or observing modes, the datasets may have different
absolute flux calibrations. ExoIris accounts for this through multiplicative flux offsets controlled
by the offset group system.

How Offsets Work
----------------

Each dataset is assigned an offset group when it is created. Group 0 serves as the reference and has no
free offset parameter. All other groups get a ``bias_XX`` parameter that applies a multiplicative
correction:

.. math::

   F(\lambda, t) = b + (1 - b) \times F_\mathrm{transit}(\lambda, t)

where :math:`b` is the offset parameter. This parameterisation preserves the shape of the transit while
allowing a small shift in the overall flux level.

.. code-block:: python

   d1 = TSData(..., offset_group=0)  # Reference dataset
   d2 = TSData(..., offset_group=1)  # Gets a free offset parameter

The offset parameters have a tight normal prior centred at zero (``NP(0.0, 1e-6)``), reflecting the
expectation that the offsets are small. This is appropriate for well-calibrated space-based data where
the flux scales are nearly---but not perfectly---consistent between instruments.

When to Use Offsets
-------------------

Offset groups are useful when combining datasets from different JWST instruments (e.g., NIRISS and
NIRSpec), different spectral orders, or different observing programmes. They let you fit all the data
jointly without worrying about matching the absolute flux levels perfectly beforehand.

If all your data come from the same instrument and mode, you typically do not need offset groups and can
leave everything in group 0.
