"""Helpers for synthesizing STL fixtures in the test suite."""

import struct
from pathlib import Path
from typing import Any

import numpy as np


def vtk_stl_reader() -> Any:
    """Return a fresh ``vtkSTLReader``, used across the suite as an oracle.

    The reader has to come from the same VTK build PyVista runs against.
    PyVista may sit on stock ``vtkmodules`` or on the ``cvista`` fork, and
    output from the wrong one is a foreign type PyVista refuses to wrap.
    Resolving through ``pyvista._vtk`` follows whichever it picked.

    Returns
    -------
    vtkSTLReader
        A new reader instance. Callers set ``Merging`` and the file name.

    """
    try:
        from pyvista._vtk import vtkSTLReader
    except ImportError:
        # PyVista older than 0.49 does not re-export the IO readers, but it
        # also predates backend selection and so always runs on stock
        # vtkmodules. Importing that directly cannot pull in a second VTK.
        from vtkmodules.vtkIOGeometry import vtkSTLReader

    return vtkSTLReader()


def write_binary_stl(
    path: Path,
    triangles: np.ndarray,
    *,
    header: bytes = b"",
    attribute_bytes: int = 0,
) -> None:
    """Write a minimal binary STL file.

    Parameters
    ----------
    path : Path
        Destination path.
    triangles : np.ndarray
        Float32 array of shape ``(n, 3, 3)``: ``n`` triangles, three
        vertices each, three coordinates each.
    header : bytes, default: ``b""``
        First 80 bytes of the file. Padded with zeros or truncated to 80.
    attribute_bytes : int, default: ``0``
        16-bit attribute byte count written after each triangle.

    """
    header_buf = header.ljust(80, b"\x00")[:80]
    n = triangles.shape[0]
    with path.open("wb") as f:
        f.write(header_buf)
        f.write(struct.pack("<I", n))
        for tri in triangles:
            # 3-component normal (zeroed) followed by 3 vertices.
            f.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for v in tri:
                f.write(struct.pack("<3f", *map(float, v)))
            f.write(struct.pack("<H", attribute_bytes))


def write_ascii_stl(
    path: Path,
    triangles: np.ndarray,
    *,
    name: str = "test",
    line_ending: str = "\n",
    indent: str = "  ",
    upper: bool = False,
) -> None:
    """Write a minimal ASCII STL file with configurable formatting."""
    if upper:
        kw_solid, kw_endsolid = "SOLID", "ENDSOLID"
        kw_facet, kw_endfacet = "FACET NORMAL", "ENDFACET"
        kw_outer, kw_endloop = "OUTER LOOP", "ENDLOOP"
        kw_vertex = "VERTEX"
    else:
        kw_solid, kw_endsolid = "solid", "endsolid"
        kw_facet, kw_endfacet = "facet normal", "endfacet"
        kw_outer, kw_endloop = "outer loop", "endloop"
        kw_vertex = "vertex"

    lines = [f"{kw_solid} {name}"]
    for tri in triangles:
        lines.append(f"{kw_facet} 0 0 0")
        lines.append(f"{indent}{kw_outer}")
        for v in tri:
            lines.append(f"{indent}{indent}{kw_vertex} {v[0]} {v[1]} {v[2]}")
        lines.append(f"{indent}{kw_endloop}")
        lines.append(kw_endfacet)
    lines.append(f"{kw_endsolid} {name}")
    text = line_ending.join(lines) + line_ending
    path.write_bytes(text.encode("ascii"))


def make_unit_triangle() -> np.ndarray:
    """Return a single right-angle triangle as ``(1, 3, 3)`` float32."""
    return np.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=np.float32,
    )


def make_two_triangles_shared_edge() -> np.ndarray:
    """Two triangles sharing one edge. Exercises vertex merging."""
    return np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )


def make_grid_triangles(n: int, *, rng: np.random.Generator | None = None) -> np.ndarray:
    """Construct ``2*n*n`` triangles tiling a regular ``[0,1]^2`` grid."""
    xs, ys = np.meshgrid(
        np.linspace(0, 1, n + 1, dtype=np.float32), np.linspace(0, 1, n + 1, dtype=np.float32)
    )
    if rng is not None:
        # add a small jitter so that no two grid points coincide
        xs = xs + rng.normal(scale=1e-6, size=xs.shape).astype(np.float32)
        ys = ys + rng.normal(scale=1e-6, size=ys.shape).astype(np.float32)
    pts = np.stack([xs, ys, np.zeros_like(xs)], axis=-1)
    tris = []
    for i in range(n):
        for j in range(n):
            tris.append([pts[i, j], pts[i, j + 1], pts[i + 1, j]])
            tris.append([pts[i, j + 1], pts[i + 1, j + 1], pts[i + 1, j]])
    return np.array(tris, dtype=np.float32)
