# ExoIris: Fast and Flexible Transmission Spectroscopy in Python

[![Docs](https://readthedocs.org/projects/exoiris/badge/)](https://exoiris.readthedocs.io)
![Python package](https://github.com/hpparvi/ExoIris/actions/workflows/python-package.yml/badge.svg)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Licence](http://img.shields.io/badge/license-GPLv3-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html)
[![PyPI version](https://badge.fury.io/py/exoiris.svg)](https://pypi.org/project/ExoIris/)

**ExoIris** is a Python package for modeling exoplanet transmission spectroscopy. ExoIris removes the typical 
limitations of the two-step workflow by modeling the full two-dimensional spectroscopic transit time series *directly*.
It supports combining transmission spectroscopy datasets from multiple instruments observed in different epochs, yielding 
self-consistent wavelength-independent and wavelength-dependent parameters, simplifying joint analyses, and delivering 
results quickly.

![](doc/source/examples/e01/example1.png)

## Why ExoIris?

Transmission spectroscopy is often done following a **two-step workflow**: (1) fit a white light curve to infer 
wavelength-independent parameters; (2) fit each spectroscopic light curve independently, constrained by the white-light 
solution. This split can introduce approximations and inconsistencies.

**ExoIris takes a different approach.** It models spectrophotometric time series *end-to-end*, enabling:

- Self-consistent inference of shared (wavelength-independent) and spectral (wavelength-dependent) parameters.
- **Joint** modeling of multiple datasets from different instruments and epochs.
- Accounting for **transit timing variations** and dataset-dependent offsets within a unified framework.

This design is a natural fit for **JWST-class** data, where correlated noise, multi-epoch observations, and 
cross-instrument combinations are the norm.

## Documentation

Full documentation and tutorials: <https://exoiris.readthedocs.io>

## Installation

Install from PyPI:

```bash
pip install exoiris
```

Latest development version:

```bash
git clone https://github.com/hpparvi/ExoIris.git
cd ExoIris
pip install -e .
```

ExoIris supports Python 3.9+. See the docs for dependency details and optional extras.

## Key Features

- **Direct modelling of spectroscopic transit time series**  
  Built on PyTransit’s `TSModel`, optimised for transmission spectroscopy; scales to hundreds–thousands of light curves simultaneously.

- **Flexible limb darkening**  
  Use standard analytical laws (quadratic, power-2, non-linear), numerical intensity profiles from stellar atmosphere models, or user-defined radially symmetric functions.

- **Robust noise treatment**  
  Choose white noise or **time-correlated** noise via a Gaussian Process likelihood, without changing the overall workflow.

- **Full control of spectral resolution**  
  The transmission spectrum is represented as a cubic spline with user-defined knots, allowing variable resolution across wavelength.

- **Reproducible, incremental workflows**  
  Save and reload models to refine a low-resolution run into a high-resolution analysis seamlessly.

- **Joint multi-dataset analyses**  
  Combine instruments and epochs in one fit, with support for transit timing variations and dataset-specific systematics and offsets.

## Performance

ExoIris is designed for speed and stability:

- A transmission spectroscopy analysis of a single JWST/NIRISS dataset at **R ≈ 100** typically runs in **3–5 minutes** 
  assuming white noise, or **5–15 minutes** with a GP noise model, on a standard desktop CPU.
- A high-resolution analysis of the JWST/NIRISS **WASP-39 b** dataset (~3800 spectroscopic light curves; see Feinstein 
  et al. 2023) can be optimised and sampled in about **1.5 hours** on an AMD Ryzen 7 5800X (8 cores, ~3-year-old desktop).

---

© 2025 Hannu Parviainen
