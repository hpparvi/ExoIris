Flux Offsets
============

When combining data from different instruments or observing modes, the datasets may have different
absolute flux calibrations. A small mismatch in the flux scale between instruments---even at the fraction
of a percent level---can introduce systematic biases in the joint fit if left unaccounted for.

ExoIris handles this through a system of offset groups. One group serves as the reference, and all other
groups receive a fitted additive correction. Offset groups are useful whenever you combine datasets from different
instruments, spectral orders, or observing programmes.
