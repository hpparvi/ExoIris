Flexible Spectral Resolution
============================

One of the central ideas in ExoIris is the separation of data resolution from the transmission spectrum
resolution. Rather than fitting an independent radius ratio to each wavelength bin, ExoIris parameterises
the transmission spectrum as an interpolating function along the wavelength defined by a set of radius-ratio knots.
This means you can model data at any spectral resolution while controlling the resolution of the output
transmission spectrum independently.

Spline Knots
------------

The transmission spectrum is defined by radius ratio values at a set of knot wavelengths. During model
evaluation, these knot values are interpolated to all data wavelengths. You can set the knots to any
wavelengths you like:

.. code-block:: python

   exoiris.set_radius_ratio_knots(np.linspace(0.6, 2.8, 50))

Knots can be added, removed, or repositioned at any point during the analysis. This makes it easy to
start with a low-resolution fit for quick exploration, then increase the resolution where you need it:

.. code-block:: python

   # Add more knots in a region of interest
   exoiris.add_radius_ratio_knots(np.linspace(1.1, 1.5, 20))

You can even set specific knot locations as free parameters, letting the optimiser find the optimal
placement for spectral features.

Interpolation Methods
---------------------

ExoIris supports six interpolation methods for the radius ratio knots:

- **nearest**: Nearest-neighbour interpolation. Fast, but produces a step function.
- **linear**: Linear interpolation between knots.
- **pchip**: Piecewise Cubic Hermite Interpolating Polynomial. Monotonicity-preserving and smooth.
- **makima**: Modified Akima interpolation. Smooth with reduced oscillation near flat regions.
- **bspline** (or **bspline-cubic**): Cubic B-spline. Very smooth, but can overshoot.
- **bspline-quadratic**: Quadratic B-spline. Smoother than linear, less oscillation than cubic.

The limb darkening coefficients use a separate interpolator (defaulting to ``bspline-quadratic``) that
can be configured independently:

.. code-block:: python

   iris.set_k_interpolator('pchip')
   iris.set_ld_interpolator('bspline-quadratic')

The choice of interpolation method affects the smoothness of the resulting transmission spectrum and can
have a noticeable impact on the posterior, especially at high spectral resolution. The ``pchip`` and
``makima`` methods are good general-purpose choices that avoid the ringing artefacts sometimes seen with
cubic splines.
