Transit Timing Variations
=========================

Transit timing variations (TTVs) occur when gravitational interactions between planets in a system cause
deviations from a strictly periodic transit schedule. Even small TTVs---on the order of minutes---can bias a
transmission spectroscopy analysis if multiple transits are modeled simultaneously assuming a linear ephemeris,
since a misplaced transit model will distort the measured transit depths across the spectrum.

ExoIris handles TTVs naturally through its epoch group system. Each epoch group receives its own transit
center parameter. Datasets observed during the same transit share an epoch group and therefore the same transit center,
while datasets from different transits get independent transit center parameters.

This approach has two advantages. First, it avoids biasing the transmission spectrum by forcing a single
ephemeris across multiple epochs. Second, it provides TTV measurements as a natural by-product of the
spectroscopic analysis---the posterior distributions of the timing parameters directly yield the transit
timing offsets, without requiring a separate TTV analysis.
