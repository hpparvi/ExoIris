# ExoIris: Transmission Spectroscopy Made Easy

[![Docs](https://readthedocs.org/projects/exoiris/badge/)](https://exoiris.readthedocs.io)
![Python package](https://github.com/hpparvi/EasyTS/actions/workflows/python-package.yml/badge.svg)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Licence](http://img.shields.io/badge/license-GPLv3-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html)
[![PyPI version](https://badge.fury.io/py/easyts.svg)](https://pypi.org/project/EasyTS/)

**ExoIris** is a user-friendly Python package designed to simplify and accelerate the analysis of transmission 
spectroscopy data for exoplanets. The package can estimate a self-consistent medium-resolution transmission spectrum 
with uncertainties from JWST NIRISS data in minutes, even when using a Gaussian Process-based noise model.

![](doc/source/examples/e01/example1.png)

## Documentation

Read the docs at [exoiris.readthedocs.io](https://exoiris.readthedocs.io).

## Key Features

- **Fast modelling of spectroscopic transit time series**: ExoIris uses PyTransit's advanced `TSModel` transit 
  model that is specially tailored for fast and efficient modelling of spectroscopic transit (or eclipse) time series.
- **Flexible handling of limb darkening**: The stellar limb darkening can be modelled freely either by any of the standard 
  limb darkening laws (quadratic, power-2, non-linear, etc.), by numerical stellar intensity profiles obtained
  directly from stellar atmosphere models, or by an arbitrary ser-defined radially symmetric function.
- **Handling of Correlated noise**: The noise model can be chosen between white or time-correlated noise, where the
  time-correlated noise is modelled as a Gaussian process.
- **Model saving and loading**: Seamless model saving and loading allows one to create a high-resolution analysis starting
  from a saved low-resolution analysis.
- **Full control of resolution**: ExoIris represents the transmission spectrum as a cubic spline, with complete 
  flexibility to set and modify the number and placement of spline knots, allowing variable resolution throughout the 
  analysis.

## Details

ExoIris uses PyTransit's `TSModel`, a transit model that is specially optimised for transmission spectroscopy and allows
for simultaneous modelling of hundreds to thousands of spectroscopic light curves 20-30 times faster than when using 
standard transit models not explicitly designed for transmission spectroscopy. 

A complete posterior solution for a low-resolution transmission spectrum with a data resolution of R=100 
takes 3-5 minutes to estimate assuming white noise, or 5-15 minutes if using a Gaussian process-based likelihood
model powered by the celerite2 package. A high-resolution spectrum of the JWST NIRISS WASP-39 b observations 
by [Feinstein et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023Natur.614..670F/abstract) with ~3800
spectroscopic light curves (as shown above) takes about 1.5 hours to optimise and sample on a three-year-old 
AMD Ryzen 7 5800X with eight cores.

---
&copy; 2024 Hannu Parviainen
