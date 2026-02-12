# ExoIris: Fast and Flexible Transmission Spectroscopy in Python

[![Docs](https://readthedocs.org/projects/exoiris/badge/)](https://exoiris.readthedocs.io)
![Python package](https://github.com/hpparvi/ExoIris/actions/workflows/python-package.yml/badge.svg)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Licence](http://img.shields.io/badge/license-GPLv3-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html)
[![PyPI version](https://badge.fury.io/py/exoiris.svg)](https://pypi.org/project/ExoIris/)
[![DOI](https://zenodo.org/badge/805355873.svg)](https://doi.org/10.5281/zenodo.18598641)

**ExoIris** models exoplanet transmission spectroscopy end-to-end — fitting the full 2D spectroscopic transit time series 
directly instead of following the traditional two-step workflow. It delivers self-consistent results across wavelengths, 
instruments, and epochs, and is designed for JWST-class data.

See the [documentation & Tutorials](https://exoiris.readthedocs.io) for details, examples, and API reference.

![](doc/source/examples/e01/example1.png)

## Installation

```bash
pip install exoiris
```

Development version:

```bash
git clone https://github.com/hpparvi/ExoIris.git && cd ExoIris && pip install -e .
```
---

© 2026 Hannu Parviainen