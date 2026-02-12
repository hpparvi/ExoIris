# ExoIris: Fast and Flexible Transmission Spectroscopy in Python

[![Docs](https://readthedocs.org/projects/exoiris/badge/)](https://exoiris.readthedocs.io)
![Python package](https://github.com/hpparvi/ExoIris/actions/workflows/python-package.yml/badge.svg)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Licence](http://img.shields.io/badge/license-GPLv3-blue.svg?style=flat)](http://www.gnu.org/licenses/gpl-3.0.html)
[![PyPI version](https://badge.fury.io/py/exoiris.svg)](https://pypi.org/project/ExoIris/)
[![DOI](https://zenodo.org/badge/805355873.svg)](https://doi.org/10.5281/zenodo.18598641)

**ExoIris** is a Python package for exoplanet transmission spectroscopy that models the full 2D spectroscopic transit time
series directly, replacing the traditional two-step workflow. It jointly fits spectrophotometric datasets from different
instruments and epochs in a single self-consistent analysis — and does so fast, completing a typical JWST transmission 
spectrum in tens of minutes on a standard desktop.

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