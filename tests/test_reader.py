"""Test pyvista_stl."""

import os
from importlib.metadata import entry_points
from typing import Any
from typing import Callable

import numpy as np
import pytest

import pyvista_stl


try:
    import pyvista as pv
    from packaging.version import Version

    PYVISTA_INSTALLED = True
    _HAS_READER_REGISTRY = Version(pv.__version__) >= Version("0.48.dev0")
except ImportError:
    PYVISTA_INSTALLED = False
    _HAS_READER_REGISTRY = False

THIS_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_FILE_ASCII = os.path.join(THIS_PATH, "sphere_ascii.stl")
TEST_FILE_BINARY = os.path.join(THIS_PATH, "sphere_binary.stl")

EXPECTED_POINTS = np.array(
    [
        [0.0, 0.52573115, -0.8506509],
        [-0.52573115, 0.8506509, 0.0],
        [0.52573115, 0.8506509, 0.0],
        [0.0, 0.52573115, 0.8506509],
        [-0.8506509, 0.0, 0.52573115],
        [0.0, -0.52573115, 0.8506509],
        [0.8506509, 0.0, 0.52573115],
        [0.8506509, 0.0, -0.52573115],
        [0.0, -0.52573115, -0.8506509],
        [-0.8506509, 0.0, -0.52573115],
        [-0.52573115, -0.8506509, 0.0],
        [0.52573115, -0.8506509, 0.0],
    ],
    dtype=np.float32,
)


EXPECTED_FACES = np.array(
    [
        [0, 1, 2],
        [3, 2, 1],
        [3, 4, 5],
        [3, 5, 6],
        [0, 7, 8],
        [0, 8, 9],
        [5, 10, 11],
        [8, 11, 10],
        [1, 9, 4],
        [10, 4, 9],
        [2, 6, 7],
        [11, 7, 6],
        [3, 1, 4],
        [3, 6, 2],
        [0, 9, 1],
        [0, 2, 7],
        [8, 10, 9],
        [8, 7, 11],
        [5, 4, 10],
        [5, 11, 6],
    ],
    dtype=np.uint32,
)


def test_read_binary() -> None:
    points, ind = pyvista_stl.read(TEST_FILE_BINARY)
    assert np.allclose(EXPECTED_POINTS, points)
    assert np.allclose(EXPECTED_FACES, ind)


def test_read_ascii() -> None:
    points, ind = pyvista_stl.read(TEST_FILE_ASCII)
    assert np.allclose(EXPECTED_POINTS, points)
    assert np.allclose(EXPECTED_FACES, ind)


@pytest.mark.skipif(not PYVISTA_INSTALLED, reason="Requires PyVista")
def test_read_as_mesh() -> None:
    pv_mesh = pv.read(TEST_FILE_BINARY)

    stl_mesh = pyvista_stl.read_as_mesh(TEST_FILE_BINARY)
    assert pv_mesh == stl_mesh
    assert stl_mesh._connectivity_array.dtype == np.int32


def test_entry_point_registered() -> None:
    """``read_as_mesh`` is advertised on the ``pyvista.readers`` group."""
    matches = [ep for ep in entry_points(group="pyvista.readers") if ep.name == ".stl"]
    assert matches, "pyvista_stl did not publish a '.stl' entry point"
    assert matches[0].value == "pyvista_stl:read_as_mesh"
    assert matches[0].load() is pyvista_stl.read_as_mesh


@pytest.mark.skipif(
    not _HAS_READER_REGISTRY,
    reason="requires pyvista >= 0.48 entry-point hooks",
)
@pytest.mark.parametrize("func", [pyvista_stl.read, pyvista_stl.read_as_mesh])
def test_read_raises_for_remote_uri(func: Callable[[str], Any]) -> None:
    """Remote URIs raise :class:`pyvista.LocalFileRequiredError` so PyVista downloads first."""
    with pytest.raises(pv.LocalFileRequiredError):
        func("https://example.com/mesh.stl")


@pytest.mark.skipif(
    not _HAS_READER_REGISTRY,
    reason="requires pyvista >= 0.48 reader registry",
)
def test_pv_read_dispatches_to_entry_point() -> None:
    """``pv.read('*.stl')`` resolves to ``pyvista_stl.read_as_mesh`` via the registry."""
    pv.read(TEST_FILE_BINARY)
    from pyvista.core.utilities import reader_registry

    assert reader_registry._custom_ext_readers.get(".stl") is pyvista_stl.read_as_mesh
