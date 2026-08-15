"""Property-based round-trip tests.

Generate random valid STL contents, write them in both binary and
ASCII forms, then confirm:

1. ``pyvista_stl`` accepts the file.
2. The output passes PyVista's parser-attributable validation fields.
3. The point set matches what ``vtkSTLReader`` produces (independent of
   vertex ordering).
4. The single-threaded (``threads=1``) and multi-threaded
   (``threads=0``) paths produce logically equivalent meshes.

The triangle generator avoids NaN/Inf, keeps coordinates in a finite
range, and uses values that round-trip exactly through ``%.7g``-style
ASCII formatting so we are testing the parser, not float-text drift.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import pyvista_stl  # noqa: E402

from _helpers import vtk_stl_reader  # noqa: E402


# Coordinates we know round-trip exactly through both float32 storage
# and the writer's ASCII formatter: small integers + a few simple
# halves are safe.
_SAFE_COORDS = st.sampled_from(
    [
        -8.0,
        -4.0,
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        -0.25,
        0.0,
        0.25,
        0.5,
        1.0,
        1.5,
        2.0,
        4.0,
        8.0,
    ]
)

_VERTEX = st.tuples(_SAFE_COORDS, _SAFE_COORDS, _SAFE_COORDS)
_TRIANGLE = st.tuples(_VERTEX, _VERTEX, _VERTEX)

_Vertex = tuple[float, float, float]
_Triangle = tuple[_Vertex, _Vertex, _Vertex]


def _write_binary_stl(path: Path, tris: np.ndarray, header: bytes = b"") -> None:
    n = tris.shape[0]
    with path.open("wb") as f:
        f.write(header.ljust(80, b"\x00")[:80])
        f.write(struct.pack("<I", n))
        for t in tris:
            f.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for v in t:
                f.write(struct.pack("<3f", *map(float, v)))
            f.write(struct.pack("<H", 0))


def _write_ascii_stl(path: Path, tris: np.ndarray, line_ending: str = "\n") -> None:
    parts = ["solid prop"]
    for t in tris:
        parts.append("facet normal 0 0 0")
        parts.append(" outer loop")
        for v in t:
            parts.append(f"  vertex {v[0]:.7g} {v[1]:.7g} {v[2]:.7g}")
        parts.append(" endloop")
        parts.append("endfacet")
    parts.append("endsolid prop")
    path.write_bytes(line_ending.join(parts).encode("ascii") + line_ending.encode())


def _vtk_point_set(path: Path) -> np.ndarray:
    r = vtk_stl_reader()
    r.SetFileName(str(path))
    r.Merging = True
    r.Update()
    out = r.GetOutput()
    return np.asarray(out.GetPoints().GetData()) if out.GetNumberOfPoints() else np.zeros((0, 3))


_PARSER_FIELDS = (
    "non_finite_points",
    "invalid_point_references",
    "wrong_number_of_points",
    "cell_data_wrong_length",
    "point_data_wrong_length",
    "unused_points",
)


def _check_parser_validity(path: Path) -> None:
    mesh = pyvista_stl.read_as_mesh(str(path))
    if mesh.n_points == 0:
        return
    rep = mesh.validate_mesh(validation_fields=list(_PARSER_FIELDS))
    failures = {f: getattr(rep, f) for f in _PARSER_FIELDS if getattr(rep, f)}
    assert not failures, f"validation failed: {failures}"


def _equivalent(p1: np.ndarray, i1: np.ndarray, p2: np.ndarray, i2: np.ndarray) -> bool:
    if p1.shape != p2.shape or i1.shape != i2.shape:
        return False
    if not np.array_equal(np.sort(p1.view(np.uint32).ravel()), np.sort(p2.view(np.uint32).ravel())):
        return False

    def expand(p: np.ndarray, i: np.ndarray) -> np.ndarray:
        c = p[i].view(np.uint32).reshape(-1, 3, 3)
        c = np.sort(c, axis=1)
        return np.sort(c.reshape(c.shape[0], -1), axis=0)

    return np.array_equal(expand(p1, i1), expand(p2, i2))


@given(tris=st.lists(_TRIANGLE, min_size=1, max_size=80))
@settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_property_binary_roundtrip(
    tmp_path_factory: pytest.TempPathFactory, tris: list[_Triangle]
) -> None:
    """Random binary STLs: pvstl accepts, validates, and matches VTK's point set."""
    tmp = tmp_path_factory.mktemp("prop_bin")
    arr = np.asarray(tris, dtype=np.float32)
    path = tmp / "rand.stl"
    _write_binary_stl(path, arr)

    pts, idx = pyvista_stl.read(str(path))
    assert idx.shape == (len(tris), 3)
    assert pts.shape[0] >= 1

    _check_parser_validity(path)

    vtk_pts = _vtk_point_set(path)
    assert vtk_pts.shape == pts.shape
    assert np.array_equal(np.sort(pts, axis=0), np.sort(vtk_pts, axis=0))


@given(
    tris=st.lists(_TRIANGLE, min_size=1, max_size=60),
    line_ending=st.sampled_from(["\n", "\r\n", "\r"]),
)
@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_property_ascii_roundtrip(
    tmp_path_factory: pytest.TempPathFactory, tris: list[_Triangle], line_ending: str
) -> None:
    """Random ASCII STLs (LF / CRLF / CR): pvstl accepts, validates, and
    matches VTK's point set."""
    tmp = tmp_path_factory.mktemp("prop_ascii")
    arr = np.asarray(tris, dtype=np.float32)
    path = tmp / "rand.stl"
    _write_ascii_stl(path, arr, line_ending=line_ending)

    pts, idx = pyvista_stl.read(str(path))
    assert idx.shape == (len(tris), 3)

    _check_parser_validity(path)

    vtk_pts = _vtk_point_set(path)
    assert vtk_pts.shape == pts.shape
    assert np.array_equal(np.sort(pts, axis=0), np.sort(vtk_pts, axis=0))


@given(tris=st.lists(_TRIANGLE, min_size=1, max_size=200))
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_property_st_mt_equivalence_binary(
    tmp_path_factory: pytest.TempPathFactory, tris: list[_Triangle]
) -> None:
    """Single-threaded and multi-threaded paths agree on logical mesh content."""
    tmp = tmp_path_factory.mktemp("prop_mt")
    arr = np.asarray(tris, dtype=np.float32)
    path = tmp / "rand.stl"
    _write_binary_stl(path, arr)

    p1, i1 = pyvista_stl.read(str(path), threads=1)
    p0, i0 = pyvista_stl.read(str(path), threads=0)
    assert _equivalent(p1, i1, p0, i0), "ST and MT paths produced inequivalent meshes"


@given(tris=st.lists(_TRIANGLE, min_size=1, max_size=200))
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_property_st_mt_equivalence_ascii(
    tmp_path_factory: pytest.TempPathFactory, tris: list[_Triangle]
) -> None:
    """ST/MT equivalence on the ASCII path."""
    tmp = tmp_path_factory.mktemp("prop_mt_a")
    arr = np.asarray(tris, dtype=np.float32)
    path = tmp / "rand.stl"
    _write_ascii_stl(path, arr)

    p1, i1 = pyvista_stl.read(str(path), threads=1)
    p0, i0 = pyvista_stl.read(str(path), threads=0)
    assert _equivalent(p1, i1, p0, i0), "ASCII ST/MT paths produced inequivalent meshes"
