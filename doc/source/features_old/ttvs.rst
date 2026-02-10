Transit Timing Variations
=========================

Transit timing variations (TTVs) occur when gravitational interactions between planets in a system cause
deviations from a strictly periodic transit schedule. Even small TTVs can bias a transmission spectroscopy
analysis if the transit center is held fixed, since a misplaced transit model will distort the measured
transit depths.

ExoIris handles TTVs naturally through its epoch group system. Each epoch group gets its own transit center
parameter (``tc_00``, ``tc_01``, ...), so the model can account for timing offsets between different
transit observations.

Setting Up Epoch Groups
-----------------------

When loading data, assign an epoch group to each dataset based on which transit it belongs to:

.. code-block:: python

   d1 = TSData(..., epoch_group=0)  # First transit
   d2 = TSData(..., epoch_group=0)  # Same transit, different instrument
   d3 = TSData(..., epoch_group=1)  # Second transit

Datasets from the same transit share the same epoch group and therefore the same transit center parameter.
Datasets from different transits get different epoch groups and independent timing parameters.

The transit center parameters are defined as offsets from the predicted ephemeris, with a normal prior
centred at zero. After fitting, you can inspect the posterior distributions of the ``tc_XX`` parameters
to measure the TTVs directly.

This approach has two advantages: it avoids biasing the transmission spectrum by forcing a single
ephemeris across multiple epochs, and it provides TTV measurements as a natural by-product of the
spectroscopic analysis.
