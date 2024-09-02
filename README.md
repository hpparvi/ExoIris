# Easy Transmission Spectroscopy (EasyTS)

[![Docs](https://readthedocs.org/projects/easyts/badge/)](https://easyts.readthedocs.io)
![Python package](https://github.com/hpparvi/EasyTS/actions/workflows/python-package.yml/badge.svg)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Licence](http://img.shields.io/badge/license-GPLv3-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html)
[![PyPI version](https://badge.fury.io/py/easyts.svg)](https://badge.fury.io/py/easyts)

Fast, flexible, and easy exoplanet transmission spectroscopy in Python. 

EasyTS uses a transit model that is specially optimised for transmission spectroscopy and allows for simultaneous 
modelling of hundreds to thousands of spectroscopic light curves 20-30 times faster than when using standard 
transit models not specifically designed for transmission spectroscopy. 

A full posterior solution for a low-resolution transmission spectrum with a data resolution of R=100 
takes 3-5 minutes to estimate assuming white noise, or 5-15 minutes if using a Gaussian process-based likelihood
model powered by the celerite2 package. A high-resolution spectrum of the JWST NIRISS WASP-39 b observations 
by [Feinstein et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023Natur.614..670F/abstract) with ~3800
spectroscopic light curves (as shown below) takes about 1.5 hours to optimise and sample on a three-year-old 
AMD Ryzen 7 5800X with 8 cores.

![](doc/source/examples/e01/example1.png)


## Documentation

Read the docs at [easyts.readthedocs.io](https://easyts.readthedocs.io).

## Installation

    pip install easyts

&copy; 2024 Hannu Parviainen
