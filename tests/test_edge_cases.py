"""Edge-case coverage for the pyvista-stl reader.

These tests synthesize STL files covering format variants and corner
cases, then verify that ``pyvista_stl.read`` recovers the expected
points and triangle indices.
"""

from pathlib import Path

import numpy as np
import pytest

import pyvista_stl

from _helpers import (
    make_grid_triangles,
    make_two_triangles_shared_edge,
    make_unit_triangle,
    write_ascii_stl,
    write_binary_stl,
)


def _canonicalize(triangles: np.ndarray) -> np.ndarray:
    """Sort vertices within each triangle, then sort triangles, for set-equality compares."""
    # Sort the 3 vertices of each triangle lexicographically.
    out = np.array([sorted(t.tolist()) for t in triangles], dtype=np.float64)
    # Sort the triangles themselves.
    flat = out.reshape(out.shape[0], -1)
    order = np.lexsort(flat.T[::-1])
    return out[order]


def _assert_mesh_equals(
    points: np.ndarray,
    indices: np.ndarray,
    expected_triangles: np.ndarray,
) -> None:
    """Assert that the ``(points, indices)`` mesh matches a per-triangle vertex array.

    Vertex ordering inside ``points`` is implementation-defined, so we
    compare canonicalized (sorted-within-and-across) triangle sets.
    """
    reconstructed = points[indices]  # shape (n_tris, 3, 3)
    assert reconstructed.shape == expected_triangles.shape

    np.testing.assert_array_equal(
        _canonicalize(reconstructed),
        _canonicalize(expected_triangles),
    )


# ---------------------------------------------------------------------------
# Binary STL
# ---------------------------------------------------------------------------


def test_binary_single_triangle(stl_dir: Path) -> None:
    tris = make_unit_triangle()
    fname = stl_dir / "single.stl"
    write_binary_stl(fname, tris)

    points, indices = pyvista_stl.read(fname)

    assert points.dtype == np.float32
    assert indices.dtype == np.int32
    assert points.shape == (3, 3)
    assert indices.shape == (1, 3)
    _assert_mesh_equals(points, indices, tris)


def test_binary_zero_triangles(stl_dir: Path) -> None:
    fname = stl_dir / "empty.stl"
    write_binary_stl(fname, np.zeros((0, 3, 3), dtype=np.float32))

    points, indices = pyvista_stl.read(fname)

    assert points.shape == (0, 3)
    assert indices.shape == (0, 3)


def test_binary_attribute_bytes_ignored(stl_dir: Path) -> None:
    """Per-triangle attribute byte count is allowed to be nonzero."""
    tris = make_two_triangles_shared_edge()
    fname = stl_dir / "with_attrs.stl"
    write_binary_stl(fname, tris, attribute_bytes=0xFFFF)

    points, indices = pyvista_stl.read(fname)

    assert points.shape == (4, 3)  # shared edge -> 4 unique verts
    assert indices.shape == (2, 3)
    _assert_mesh_equals(points, indices, tris)


def test_binary_solid_prefix_is_disambiguated(stl_dir: Path) -> None:
    """A binary STL whose 80-byte header begins with ``"solid "`` parses as binary.

    Several real-world writers (including older versions of PyVista
    and various CAD tools) prefix binary STLs with ``"solid"``. The
    parser disambiguates from ASCII by checking the size invariant
    ``size == 84 + 50 * ntris``.
    """
    tris = make_two_triangles_shared_edge()
    fname = stl_dir / "solid_header.stl"
    write_binary_stl(fname, tris, header=b"solid binary stl from buggy writer")

    points, indices = pyvista_stl.read(fname)

    assert points.shape == (4, 3)
    assert indices.shape == (2, 3)
    _assert_mesh_equals(points, indices, tris)


@pytest.mark.parametrize("n", [10, 50, 200])
def test_binary_grid_round_trip_triggers_mt_path(stl_dir: Path, n: int) -> None:
    """Larger grids exercise the multi-threaded binary path (>= 100k tris)."""
    tris = make_grid_triangles(n)
    fname = stl_dir / f"grid_{n}.stl"
    write_binary_stl(fname, tris)

    points, indices = pyvista_stl.read(fname)

    # The grid has shared vertices; the unique count is (n+1)^2.
    assert points.shape == ((n + 1) ** 2, 3)
    assert indices.shape == (tris.shape[0], 3)
    _assert_mesh_equals(points, indices, tris)


def test_binary_invalid_size_raises(stl_dir: Path) -> None:
    """A binary file whose size doesn't match its declared triangle count is rejected."""
    fname = stl_dir / "bad.stl"
    write_binary_stl(fname, make_unit_triangle())
    # Truncate the file mid-triangle.
    raw = fname.read_bytes()
    fname.write_bytes(raw[:-10])

    with pytest.raises(RuntimeError):
        pyvista_stl.read(fname)


def test_binary_too_short_raises(stl_dir: Path) -> None:
    fname = stl_dir / "short.stl"
    fname.write_bytes(b"too short")
    with pytest.raises(RuntimeError):
        pyvista_stl.read(fname)


# ---------------------------------------------------------------------------
# ASCII STL
# ---------------------------------------------------------------------------


def test_ascii_single_triangle(stl_dir: Path) -> None:
    tris = make_unit_triangle()
    fname = stl_dir / "single_ascii.stl"
    write_ascii_stl(fname, tris)

    points, indices = pyvista_stl.read(fname)

    assert points.shape == (3, 3)
    assert indices.shape == (1, 3)
    _assert_mesh_equals(points, indices, tris)


def test_ascii_shared_edge(stl_dir: Path) -> None:
    tris = make_two_triangles_shared_edge()
    fname = stl_dir / "shared.stl"
    write_ascii_stl(fname, tris)

    points, indices = pyvista_stl.read(fname)

    assert points.shape == (4, 3)
    assert indices.shape == (2, 3)
    _assert_mesh_equals(points, indices, tris)


@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"])
def test_ascii_line_endings(stl_dir: Path, line_ending: str) -> None:
    tris = make_two_triangles_shared_edge()
    fname = stl_dir / "le.stl"
    write_ascii_stl(fname, tris, line_ending=line_ending)

    points, indices = pyvista_stl.read(fname)

    assert points.shape == (4, 3)
    assert indices.shape == (2, 3)
    _assert_mesh_equals(points, indices, tris)


def test_ascii_tab_indented(stl_dir: Path) -> None:
    tris = make_two_triangles_shared_edge()
    fname = stl_dir / "tabs.stl"
    write_ascii_stl(fname, tris, indent="\t")

    points, indices = pyvista_stl.read(fname)

    assert points.shape == (4, 3)
    assert indices.shape == (2, 3)
    _assert_mesh_equals(points, indices, tris)


def test_ascii_trailing_whitespace(stl_dir: Path) -> None:
    tris = make_two_triangles_shared_edge()
    fname = stl_dir / "trailing.stl"
    write_ascii_stl(fname, tris)
    # Append blank lines and trailing spaces.
    fname.write_bytes(fname.read_bytes() + b"   \n\n\n")

    points, indices = pyvista_stl.read(fname)

    assert points.shape == (4, 3)
    assert indices.shape == (2, 3)
    _assert_mesh_equals(points, indices, tris)


def test_ascii_various_float_formats(stl_dir: Path) -> None:
    """Cover ``1``, ``1.0``, ``1.``, ``.5``, scientific notation, signed zeros."""
    fname = stl_dir / "floats.stl"
    body = (
        "solid floats\n"
        "facet normal 0 0 0\n"
        "  outer loop\n"
        "    vertex 0 0 -0.0\n"
        "    vertex 1. .5 +1.5e0\n"
        "    vertex -2.5E-1 1e1 -.25\n"
        "  endloop\n"
        "endfacet\n"
        "endsolid floats\n"
    )
    fname.write_bytes(body.encode("ascii"))

    points, indices = pyvista_stl.read(fname)
    assert points.shape == (3, 3)
    assert indices.shape == (1, 3)
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.5, 1.5],
            [-0.25, 10.0, -0.25],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.sort(points, axis=0), np.sort(expected, axis=0))


def test_ascii_large_triggers_mt_path(stl_dir: Path) -> None:
    """ASCII files above the 4 MB threshold use the multi-threaded path."""
    tris = make_grid_triangles(200)
    fname = stl_dir / "big_ascii.stl"
    write_ascii_stl(fname, tris)
    assert fname.stat().st_size > 4 * 1024 * 1024  # crosses MT threshold

    points, indices = pyvista_stl.read(fname)

    assert points.shape == ((200 + 1) ** 2, 3)
    assert indices.shape == (tris.shape[0], 3)


def test_ascii_blank_only_solid(stl_dir: Path) -> None:
    fname = stl_dir / "blank.stl"
    fname.write_bytes(b"solid empty\nendsolid empty\n")

    points, indices = pyvista_stl.read(fname)

    assert points.shape == (0, 3)
    assert indices.shape == (0, 3)


# ---------------------------------------------------------------------------
# Cross-format / general
# ---------------------------------------------------------------------------


def test_round_trip_binary_to_pyvista(stl_dir: Path) -> None:
    """Read a binary STL into PyVista and back, verifying parity with VTK."""
    pv = pytest.importorskip("pyvista")

    tris = make_grid_triangles(20)
    fname = stl_dir / "rt.stl"
    write_binary_stl(fname, tris)

    ours = pyvista_stl.read_as_mesh(fname)
    theirs = pv.read(fname)

    # Same number of points / cells (vertex order may differ).
    assert ours.n_points == theirs.n_points
    assert ours.n_cells == theirs.n_cells
    np.testing.assert_allclose(np.sort(ours.points, axis=0), np.sort(theirs.points, axis=0))


def test_file_not_found(stl_dir: Path) -> None:
    with pytest.raises(RuntimeError):
        pyvista_stl.read(stl_dir / "does_not_exist.stl")


def test_repeated_reads_are_independent(stl_dir: Path) -> None:
    """Reading the same file twice yields independent arrays."""
    tris = make_two_triangles_shared_edge()
    fname = stl_dir / "rep.stl"
    write_binary_stl(fname, tris)

    p1, i1 = pyvista_stl.read(fname)
    p2, i2 = pyvista_stl.read(fname)

    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(i1, i2)
    # Mutating one must not affect the other.
    p1[0, 0] += 1.0
    assert not np.array_equal(p1, p2)


def test_path_like_input(stl_dir: Path) -> None:
    """The reader accepts ``str``, ``Path``, and other os.PathLike inputs."""
    tris = make_unit_triangle()
    fname = stl_dir / "path.stl"
    write_binary_stl(fname, tris)

    p_str, _ = pyvista_stl.read(str(fname))
    p_path, _ = pyvista_stl.read(fname)

    np.testing.assert_array_equal(p_str, p_path)
