"""Shared helpers for the ``benchmarks/`` scripts."""

import gc
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyvista as pv
from vtkmodules.vtkIOGeometry import vtkSTLReader

import pyvista_stl


def time_call(fn: Callable[[], object], *, n: int = 5, warmup: int = 1) -> tuple[float, float]:
    """Time ``fn`` ``n`` times and return ``(best, median)`` in seconds."""
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(n):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), statistics.median(times)


def read_pyvista_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    return pyvista_stl.read(path)


def read_vtk(path: Path) -> "pv.PolyData":
    """Read with VTK's STL reader, with vertex merging enabled."""
    reader = vtkSTLReader()
    reader.SetFileName(str(path))
    reader.Merging = True
    reader.Update()
    return reader.GetOutput()


def read_numpy_stl(path: Path) -> object:
    """Read with ``numpy-stl``. Note: does not deduplicate vertices."""
    from stl import mesh as npstl_mesh

    return npstl_mesh.Mesh.from_file(str(path))


def read_meshio(path: Path) -> object:
    import meshio

    return meshio.read(str(path))


READERS: dict[str, Callable[[Path], object]] = {
    "pyvista_stl": read_pyvista_stl,
    "vtk": read_vtk,
    "numpy_stl": read_numpy_stl,
    "meshio": read_meshio,
}


def make_test_mesh(resolution: int, *, subdivide: int = 2) -> "pv.PolyData":
    """Triangulated plane at the given resolution; ``2*resolution^2 * 4^subdivide`` triangles."""
    mesh = pv.Plane(i_resolution=resolution, j_resolution=resolution).triangulate()
    if subdivide:
        mesh = mesh.subdivide(subdivide)
    return mesh
