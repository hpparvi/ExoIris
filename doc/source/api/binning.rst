.. _api.binning

Binning
=======
.. currentmodule:: easyts

While `EasyTS` aims to be fast enough that binning of data is not usually necessary, we still may want to do it every now
or then. For this, `EasyTS` provides the `binning.Binning` and `binning.CompoundBinning` classes that can be used to define complex,
non-regular binning geometries

Binning
-------

.. autosummary::
    :toctree: api/

    binning.Binning

CompoundBinning
---------------

.. autosummary::
    :toctree: api/

    binning.CompoundBinning
