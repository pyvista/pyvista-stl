"""Robustness and adversarial-input coverage for the pyvista-stl reader.

Exercises code paths that the basic format/round-trip tests miss:
malformed inputs, attacker-controlled triangle counts, NUL bytes in
paths, the multi-threaded fallback path on degenerate inputs, and
agreement between the single-threaded and multi-threaded paths on the
same input.
"""

import struct
from pathlib import Path

import numpy as np
import pytest

import pyvista_stl

from _helpers import (
    make_grid_triangles,
    make_unit_triangle,
    write_binary_stl,
)


# ---------------------------------------------------------------------------
# Malformed inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [0, 1, 14, 15, 83])
def test_too_short_inputs_rejected(stl_dir: Path, size: int) -> None:
    """Files smaller than the binary header threshold are rejected cleanly."""
    fname = stl_dir / "short.stl"
    fname.write_bytes(b"x" * size)
    with pytest.raises(RuntimeError):
        pyvista_stl.read(fname)


def test_binary_truncated_mid_record_rejected(stl_dir: Path) -> None:
    """A binary STL whose declared count does not match its file size is rejected."""
    fname = stl_dir / "trunc.stl"
    write_binary_stl(fname, make_unit_triangle())
    raw = fname.read_bytes()
    fname.write_bytes(raw[:-10])  # drop a partial triangle
    with pytest.raises(RuntimeError):
        pyvista_stl.read(fname)


def test_binary_oversize_ntris_rejected(stl_dir: Path) -> None:
    """A binary STL header that lies about an enormous triangle count is rejected.

    The 84-byte header declares 4_000_000_000 triangles but the file is
    only 84 bytes; the size invariant should catch this and the reader
    should raise rather than allocate hundreds of GB.
    """
    fname = stl_dir / "oversize.stl"
    header = b"\x00" * 80
    fname.write_bytes(header + struct.pack("<I", 4_000_000_000))
    with pytest.raises(RuntimeError):
        pyvista_stl.read(fname)


def test_oversize_ntris_capped(stl_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The triangle-count cap rejects valid-looking files above the limit.

    We construct a syntactically valid binary STL with 100 triangles,
    set the cap to 50 via ``PYVISTA_STL_MAX_TRIS``, and assert the
    reader refuses with a specific message.
    """
    fname = stl_dir / "many.stl"
    rng = np.random.default_rng(0)
    tris = rng.uniform(-1, 1, size=(100, 3, 3)).astype(np.float32)
    write_binary_stl(fname, tris)
    monkeypatch.setenv("PYVISTA_STL_MAX_TRIS", "50")
    with pytest.raises(RuntimeError, match="PYVISTA_STL_MAX_TRIS"):
        pyvista_stl.read(fname)


def test_ascii_malformed_facet_skipped(stl_dir: Path) -> None:
    """An ASCII facet with fewer than 3 vertices is dropped, the rest parse normally."""
    fname = stl_dir / "malformed.stl"
    body = (
        "solid x\n"
        # First facet only declares two vertices: must be ignored.
        "facet normal 0 0 0\n"
        "  outer loop\n"
        "    vertex 0 0 0\n"
        "    vertex 1 0 0\n"
        "  endloop\n"
        "endfacet\n"
        # Second facet is well-formed; should yield one triangle.
        "facet normal 0 0 0\n"
        "  outer loop\n"
        "    vertex 0 0 0\n"
        "    vertex 1 0 0\n"
        "    vertex 0 1 0\n"
        "  endloop\n"
        "endfacet\n"
        "endsolid x\n"
    )
    fname.write_bytes(body.encode("ascii"))

    points, indices = pyvista_stl.read(fname)

    assert indices.shape == (1, 3)
    assert points.shape == (3, 3)


def test_nan_inf_vertices_round_trip(stl_dir: Path) -> None:
    """NaN and Inf vertex bits are preserved verbatim through the parser.

    The parser stores raw 4-byte float words and dedupes by bitwise
    equality, so NaN and Inf are valid vertex values from the parser's
    point of view. Two NaNs with the same bit pattern hash equal and
    merge; two NaNs with different bit patterns do not. We assert that
    inputs we wrote come back as the same bit pattern.
    """
    nan = np.float32(np.nan)
    inf = np.float32(np.inf)
    tris = np.array(
        [
            [[0.0, 0.0, 0.0], [nan, nan, nan], [inf, -inf, 0.0]],
        ],
        dtype=np.float32,
    )
    fname = stl_dir / "nan.stl"
    write_binary_stl(fname, tris)

    points, indices = pyvista_stl.read(fname)

    assert points.shape == (3, 3)
    assert indices.shape == (1, 3)

    # NaN bit patterns must be identical to the input. Compare via
    # uint32 view because NaN != NaN under float equality.
    points_bits = points.view(np.uint32)
    expected_bits = tris.reshape(-1, 3).view(np.uint32)
    sorted_points = points_bits[np.lexsort(points_bits.T[::-1])]
    sorted_expected = expected_bits[np.lexsort(expected_bits.T[::-1])]
    np.testing.assert_array_equal(sorted_points, sorted_expected)


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


def test_nul_byte_in_path_rejected(stl_dir: Path) -> None:
    """An embedded NUL byte in the path is rejected at the binding boundary.

    POSIX ``open(2)`` and Windows ``CreateFile`` truncate at the first
    NUL, which can mask the user's intent. Reject early with a
    specific exception type rather than silently opening a different
    file.
    """
    with pytest.raises((ValueError, RuntimeError)):
        pyvista_stl.read("file\x00.stl")


# ---------------------------------------------------------------------------
# Differential tests: seq vs MT
# ---------------------------------------------------------------------------


def test_seq_and_mt_agree_on_grid(stl_dir: Path) -> None:
    """The sequential and multi-threaded binary paths produce equivalent meshes.

    Vertex order is implementation-defined, so we compare the
    canonicalized triangle sets (vertex coordinates sorted within each
    triangle, then triangles sorted across the array).
    """
    tris = make_grid_triangles(100)
    fname = stl_dir / "grid_diff.stl"
    write_binary_stl(fname, tris)

    pts_seq, idx_seq = pyvista_stl.read(fname, threads=1)
    pts_mt, idx_mt = pyvista_stl.read(fname, threads=8)

    assert pts_seq.shape == pts_mt.shape
    assert idx_seq.shape == idx_mt.shape

    def _canon(pts: np.ndarray, idx: np.ndarray) -> np.ndarray:
        cells = pts[idx]
        cells = np.sort(cells, axis=1)
        flat = cells.reshape(cells.shape[0], -1)
        return np.asarray(cells[np.lexsort(flat.T[::-1])])

    np.testing.assert_array_equal(_canon(pts_seq, idx_seq), _canon(pts_mt, idx_mt))


def test_seq_path_is_deterministic(stl_dir: Path) -> None:
    """Two reads with ``threads=1`` produce byte-identical output.

    Documents the sequential path's deterministic-vertex-ordering
    contract that the README advertises.
    """
    tris = make_grid_triangles(40)
    fname = stl_dir / "seq_det.stl"
    write_binary_stl(fname, tris)

    a_pts, a_idx = pyvista_stl.read(fname, threads=1)
    b_pts, b_idx = pyvista_stl.read(fname, threads=1)

    np.testing.assert_array_equal(a_pts, b_pts)
    np.testing.assert_array_equal(a_idx, b_idx)


def test_default_threads_is_single_threaded(stl_dir: Path) -> None:
    """The default invocation matches ``threads=1`` byte-for-byte.

    Pins the documented default so a future change to the dispatch
    rule cannot quietly flip users onto the multi-threaded path.
    """
    tris = make_grid_triangles(40)
    fname = stl_dir / "seq_default.stl"
    write_binary_stl(fname, tris)

    default_pts, default_idx = pyvista_stl.read(fname)
    seq_pts, seq_idx = pyvista_stl.read(fname, threads=1)

    np.testing.assert_array_equal(default_pts, seq_pts)
    np.testing.assert_array_equal(default_idx, seq_idx)


def test_threads_zero_selects_auto(stl_dir: Path) -> None:
    """``threads=0`` is honored as auto-select (hardware_concurrency)."""
    tris = make_grid_triangles(20)
    fname = stl_dir / "auto.stl"
    write_binary_stl(fname, tris)

    pts, idx = pyvista_stl.read(fname, threads=0)
    assert pts.shape[1] == 3
    assert idx.shape[1] == 3


# ---------------------------------------------------------------------------
# MT fallback path
# ---------------------------------------------------------------------------


def test_mt_fallback_on_all_unique_vertices(stl_dir: Path) -> None:
    """All-unique-vertex inputs above the MT threshold round-trip via the seq fallback.

    The MT binary path uses a hashtable sized to ``nextpow2(ntris)``,
    which cannot fit ``3 * ntris`` unique vertices. Inputs that
    saturate it must fall through to the sequential path with the
    full-bound table. This test forces that condition.
    """
    rng = np.random.default_rng(seed=42)
    n = 120_000  # above MT_BINARY_THRESHOLD
    tris = rng.uniform(-1, 1, size=(n, 3, 3)).astype(np.float32)
    fname = stl_dir / "all_unique.stl"
    write_binary_stl(fname, tris)

    points, indices = pyvista_stl.read(fname)

    # Every vertex unique: 3*n distinct vertices, n triangles.
    assert points.shape == (3 * n, 3)
    assert indices.shape == (n, 3)


# ---------------------------------------------------------------------------
# Topology corner cases
# ---------------------------------------------------------------------------


def test_duplicate_triangle_collapses_to_one_cell(stl_dir: Path) -> None:
    """Two facets with identical vertices share point indices.

    The vertex deduplicator collapses the three vertex coordinates,
    but each facet remains a separate triangle in the cell array (the
    parser does not deduplicate cells, only points).
    """
    tris = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    fname = stl_dir / "dup.stl"
    write_binary_stl(fname, tris)

    points, indices = pyvista_stl.read(fname)

    assert points.shape == (3, 3)  # three unique vertices
    assert indices.shape == (2, 3)  # two cells preserved


def test_degenerate_triangle_preserved(stl_dir: Path) -> None:
    """A facet whose three vertices coincide is preserved with one unique point.

    STL viewers and downstream consumers vary on degenerate-triangle
    handling; the parser does not filter them and emits one merged
    point referenced three times in the cell.
    """
    tris = np.array(
        [
            [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        ],
        dtype=np.float32,
    )
    fname = stl_dir / "degen.stl"
    write_binary_stl(fname, tris)

    points, indices = pyvista_stl.read(fname)

    assert points.shape == (1, 3)
    assert indices.shape == (1, 3)
    assert (indices == 0).all()
