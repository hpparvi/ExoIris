#  ExoIris: fast, flexible, and easy exoplanet transmission spectroscopy in Python.
#  Copyright (C) 2024 Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Import tests for ExoIris package.

These tests verify that ExoIris can be imported correctly after standard
installation and that all public API components are accessible.
"""

import pytest


def test_import_exoiris():
    """Test that the main exoiris package can be imported."""
    import exoiris
    assert exoiris is not None


def test_version():
    """Test that version string is accessible."""
    from exoiris import __version__
    assert isinstance(__version__, str)
    assert len(__version__) > 0


class TestPublicAPI:
    """Test that all public API classes and functions are importable."""

    def test_import_exoiris_class(self):
        """Test that ExoIris main class is importable."""
        from exoiris import ExoIris
        assert ExoIris is not None

    def test_import_load_model(self):
        """Test that load_model function is importable."""
        from exoiris import load_model
        assert callable(load_model)

    def test_import_tsdata(self):
        """Test that TSData class is importable."""
        from exoiris import TSData
        assert TSData is not None

    def test_import_tsdatagroup(self):
        """Test that TSDataGroup class is importable."""
        from exoiris import TSDataGroup
        assert TSDataGroup is not None

    def test_import_binning(self):
        """Test that Binning class is importable."""
        from exoiris import Binning
        assert Binning is not None

    def test_import_ldtkld(self):
        """Test that LDTkLD class is importable."""
        from exoiris import LDTkLD
        assert LDTkLD is not None

    def test_import_clean_knots(self):
        """Test that clean_knots function is importable."""
        from exoiris import clean_knots
        assert callable(clean_knots)


class TestInternalModules:
    """Test that key internal modules can be imported.

    These tests verify that the internal module structure is intact,
    which helps catch missing dependencies or syntax errors.
    """

    def test_import_ephemeris(self):
        """Test that ephemeris module is importable."""
        import exoiris.ephemeris
        assert exoiris.ephemeris is not None

    def test_import_spotmodel(self):
        """Test that spotmodel module is importable."""
        import exoiris.spotmodel
        assert exoiris.spotmodel is not None

    def test_import_tslpf(self):
        """Test that tslpf module is importable."""
        import exoiris.tslpf
        assert exoiris.tslpf is not None

    def test_import_wlpf(self):
        """Test that wlpf module is importable."""
        import exoiris.wlpf
        assert exoiris.wlpf is not None

    def test_import_tsmodel(self):
        """Test that tsmodel module is importable."""
        import exoiris.tsmodel
        assert exoiris.tsmodel is not None

    def test_import_loglikelihood(self):
        """Test that loglikelihood module is importable."""
        import exoiris.loglikelihood
        assert exoiris.loglikelihood is not None

    def test_import_tsdata_module(self):
        """Test that tsdata module is importable."""
        import exoiris.tsdata
        assert exoiris.tsdata is not None

    def test_import_binning_module(self):
        """Test that binning module is importable."""
        import exoiris.binning
        assert exoiris.binning is not None
