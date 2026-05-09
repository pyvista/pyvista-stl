"""Regenerate the committed STL test corpus in ``tests/data/``.

Run this script when you need to refresh or extend the corpus. Each
file targets a specific format variant or edge case, so the test
suite has fixed, version-controlled inputs to validate against. Files
are kept small (each < 100 KB) so the corpus stays well under 1 MB.

A scripted corpus avoids committing real-world models with unclear
licensing while still covering multiple writers, line endings,
indentations, sizes, geometries, and edge cases.
"""

import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    pass  # pv types referenced only via string annotations

DATA = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Geometry generators
# ---------------------------------------------------------------------------

UNIT_TRIANGLE = np.array(
    [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
    dtype=np.float32,
)


def shared_edge() -> np.ndarray:
    """Two triangles sharing one edge. Smallest mesh that exercises merging."""
    return np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )


def tetrahedron() -> np.ndarray:
    """4 triangles, 4 vertices. Every face shares two edges."""
    a, b, c, d = (
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    )
    tris = [[a, b, c], [a, c, d], [a, d, b], [b, d, c]]
    return np.array(tris, dtype=np.float32)


def cube() -> np.ndarray:
    """12 triangles forming the surface of the unit cube."""
    return _polydata_to_triangle_array(pv.Cube().triangulate())


def small_sphere() -> np.ndarray:
    """Triangulated icosphere with a small triangle count."""
    return _polydata_to_triangle_array(
        pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate()
    )


def _polydata_to_triangle_array(mesh: "pv.PolyData") -> np.ndarray:
    """Convert a triangulated PolyData to an ``(n, 3, 3)`` ``float32`` array."""
    pts = np.asarray(mesh.points, dtype=np.float32)
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]
    return np.ascontiguousarray(pts[faces], dtype=np.float32)


def grid(n: int) -> np.ndarray:
    """``2*n*n`` triangles tiling the unit square; well-merged manifold."""
    xs, ys = np.meshgrid(
        np.linspace(0, 1, n + 1, dtype=np.float32),
        np.linspace(0, 1, n + 1, dtype=np.float32),
    )
    pts = np.stack([xs, ys, np.zeros_like(xs)], axis=-1)
    tris = []
    for i in range(n):
        for j in range(n):
            tris.append([pts[i, j], pts[i, j + 1], pts[i + 1, j]])
            tris.append([pts[i, j + 1], pts[i + 1, j + 1], pts[i + 1, j]])
    return np.array(tris, dtype=np.float32)


def disconnected_triangles(n: int) -> np.ndarray:
    """``n`` triangles with no shared vertices. Exercises the no-merge path."""
    rng = np.random.default_rng(0)
    return rng.uniform(-1, 1, size=(n, 3, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_binary(
    path: Path,
    triangles: np.ndarray,
    *,
    header: bytes = b"",
    attribute_bytes: int = 0,
) -> None:
    """Write a binary STL file at ``path`` with the given triangles."""
    n = triangles.shape[0]
    buf = bytearray(header.ljust(80, b"\x00")[:80])
    buf += struct.pack("<I", n)
    for tri in triangles:
        buf += struct.pack("<3f", 0.0, 0.0, 0.0)  # zero normal
        for v in tri:
            buf += struct.pack("<3f", *map(float, v))
        buf += struct.pack("<H", attribute_bytes)
    path.write_bytes(buf)


def _format_value(value: float, fmt: str) -> str:
    return fmt.format(float(value))


def write_ascii(
    path: Path,
    triangles: np.ndarray,
    *,
    name: str = "test",
    line_ending: str = "\n",
    indent: str = "  ",
    upper: bool = False,
    float_format: str = "{:g}",
) -> None:
    """Write an ASCII STL file at ``path`` with configurable formatting."""
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
            x = _format_value(v[0], float_format)
            y = _format_value(v[1], float_format)
            z = _format_value(v[2], float_format)
            lines.append(f"{indent}{indent}{kw_vertex} {x} {y} {z}")
        lines.append(f"{indent}{kw_endloop}")
        lines.append(kw_endfacet)
    lines.append(f"{kw_endsolid} {name}")
    path.write_bytes((line_ending.join(lines) + line_ending).encode("ascii"))


# ---------------------------------------------------------------------------
# Corpus definition
# ---------------------------------------------------------------------------


def build_corpus(out: Path) -> list[Path]:
    """Write every fixture under ``out`` and return their paths."""
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _bin(name: str, **kw: object) -> None:
        path = out / name
        write_binary(path, **kw)  # type: ignore[arg-type]
        written.append(path)

    def _ascii(name: str, **kw: object) -> None:
        path = out / name
        write_ascii(path, **kw)  # type: ignore[arg-type]
        written.append(path)

    # Minimal cases.
    _bin("binary_zero_triangles.stl", triangles=np.zeros((0, 3, 3), dtype=np.float32))
    _bin("binary_single_triangle.stl", triangles=UNIT_TRIANGLE)
    _ascii("ascii_zero_triangles.stl", triangles=np.zeros((0, 3, 3), dtype=np.float32))
    _ascii("ascii_single_triangle.stl", triangles=UNIT_TRIANGLE)

    # Small geometry.
    _bin("binary_tetra.stl", triangles=tetrahedron())
    _bin("binary_cube.stl", triangles=cube())
    _bin("binary_sphere.stl", triangles=small_sphere())

    # Binary header variants. Some real-world writers prefix binary
    # STLs with "solid ..."; the parser must not mistake them for ASCII.
    _bin(
        "binary_solid_header.stl",
        triangles=shared_edge(),
        header=b"solid binary file written by buggy_writer v1.0",
    )
    # Non-zero per-triangle attribute byte count is also legal under the spec.
    _bin("binary_with_attribute_bytes.stl", triangles=cube(), attribute_bytes=0xABCD)
    _bin(
        "binary_geomagic_header.stl",
        triangles=small_sphere(),
        header=b"STL Output from geomagic Studio",
    )
    _bin(
        "binary_netfabb_header.stl",
        triangles=small_sphere(),
        header=b"STL File created by netfabb - http://www.netfabb.com UNITS=MM",
    )

    # ASCII line-ending variants.
    _ascii("ascii_lf.stl", triangles=shared_edge(), line_ending="\n")
    _ascii("ascii_crlf.stl", triangles=shared_edge(), line_ending="\r\n")
    _ascii("ascii_cr.stl", triangles=shared_edge(), line_ending="\r")

    # ASCII indentation and float-format variants.
    _ascii("ascii_tab_indent.stl", triangles=cube(), indent="\t")
    _ascii("ascii_no_indent.stl", triangles=cube(), indent="")
    _ascii("ascii_scientific_floats.stl", triangles=small_sphere(), float_format="{:.6e}")

    # Moderate sizes. The MT-path threshold itself is covered by
    # ``test_edge_cases.test_ascii_large_triggers_mt_path``; we keep
    # committed files small.
    _bin("binary_grid_50.stl", triangles=grid(50))
    _ascii("ascii_grid_30.stl", triangles=grid(30))

    # Stress: 200 disconnected triangles (600 unique verts), small
    # enough to commit while exercising the no-merge code path.
    _bin("binary_disconnected.stl", triangles=disconnected_triangles(200))

    return written


def main() -> int:
    written = build_corpus(DATA)
    total = 0
    for p in sorted(written):
        sz = p.stat().st_size
        total += sz
        print(f"{p.name:<40} {sz:>10} bytes")
    print(f"{'TOTAL':<40} {total:>10} bytes ({total / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
