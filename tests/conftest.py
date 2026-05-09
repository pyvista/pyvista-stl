"""Shared fixtures for the pyvista-stl test suite."""

from pathlib import Path

import pytest


@pytest.fixture
def stl_dir(tmp_path: Path) -> Path:
    """Per-test scratch directory for synthesized STL fixtures."""
    return tmp_path
