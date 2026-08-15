"""Pinned regression tests for bugs found in the 2026-05 corpus audit.

Each bug from ``KNOWN_BUGS.md`` has a synthetic and (where available) a
real-fixture regression test. They were written as ``xfail(strict=True)``
during the audit; the markers were removed once the parser fix landed.
The assertions stay so that any future regression fails the suite.

VTK's ``vtkSTLReader`` is consulted as a sanity oracle on real fixtures
that VTK can read correctly. On the synthesized minimal files VTK's
binary-confusion heuristic (size against ``84 + 50*ntris``) misfires and
rejects the file, so those tests assert directly against the parser
output rather than against VTK.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _helpers import vtk_stl_reader

import pyvista_stl


def _vtk_counts(path: Path) -> tuple[int, int]:
    r = vtk_stl_reader()
    r.SetFileName(str(path))
    r.Merging = True
    r.Update()
    out = r.GetOutput()
    return int(out.GetNumberOfPoints()), int(out.GetNumberOfCells())


# ASCII body containing a single valid triangle, used by several tests.
_VALID_BODY = (
    "facet normal 0 0 1\n"
    " outer loop\n"
    "  vertex 0 0 0\n"
    "  vertex 1 0 0\n"
    "  vertex 0 1 0\n"
    " endloop\n"
    "endfacet\n"
)


def test_ascii_leading_whitespace_before_solid(tmp_path: Path) -> None:
    """Bug 1: ASCII STL with leading whitespace before ``solid`` parses.

    Real example: ``numpy-stl``'s PRO2STL ``Cube.stl`` fixture starts
    with a single space (``b" solid PRO2STL ..."``).
    """
    path = tmp_path / "leading_space.stl"
    path.write_text(f"   solid x\n{_VALID_BODY}endsolid x\n")

    pts, idx = pyvista_stl.read(str(path))
    assert idx.shape[0] == 1
    assert pts.shape[0] == 3


def test_ascii_utf8_bom_before_solid(tmp_path: Path) -> None:
    """Bug 1 companion: a UTF-8 BOM ahead of ``solid`` does not break detection."""
    path = tmp_path / "bom.stl"
    path.write_bytes(b"\xef\xbb\xbfsolid x\n" + _VALID_BODY.encode() + b"endsolid x\n")

    pts, idx = pyvista_stl.read(str(path))
    assert idx.shape[0] == 1
    assert pts.shape[0] == 3


def test_ascii_tab_after_solid_keyword(tmp_path: Path) -> None:
    """Bug 2: ``solid`` followed by a TAB delimiter is accepted.

    Real example: ``assimp/test/models/STL/sphereWithHole.stl``.
    """
    path = tmp_path / "tab_after_solid.stl"
    path.write_bytes(b"solid\tname\n" + _VALID_BODY.encode() + b"endsolid name\n")

    pts, idx = pyvista_stl.read(str(path))
    assert idx.shape[0] == 1
    assert pts.shape[0] == 3


def test_ascii_newline_after_solid_keyword(tmp_path: Path) -> None:
    """Bug 2 companion: ``solid`` directly followed by a newline (no name) parses."""
    path = tmp_path / "newline_after_solid.stl"
    path.write_bytes(b"solid\n" + _VALID_BODY.encode() + b"endsolid\n")

    pts, idx = pyvista_stl.read(str(path))
    assert idx.shape[0] == 1
    assert pts.shape[0] == 3


def test_ascii_uppercase_facet_keywords_in_body(tmp_path: Path) -> None:
    """Bug 3: a file mixing lowercase and uppercase facet keywords reads completely.

    The first facet uses the canonical lowercase form. The second uses
    uppercase for ``FACET NORMAL``, ``OUTER LOOP``, ``VERTEX``,
    ``ENDLOOP``, ``ENDFACET``. Before the fix the uppercase facet was
    silently dropped (no error) — a correctness-critical data-loss case.
    """
    path = tmp_path / "mixed_case.stl"
    upper_facet = (
        "FACET NORMAL 0 0 1\n"
        " OUTER LOOP\n"
        "  VERTEX 0 0 1\n"
        "  VERTEX 1 0 1\n"
        "  VERTEX 0 1 1\n"
        " ENDLOOP\n"
        "ENDFACET\n"
    )
    path.write_text(f"solid x\n{_VALID_BODY}{upper_facet}endsolid x\n")

    pts, idx = pyvista_stl.read(str(path))
    assert idx.shape[0] == 2
    assert pts.shape[0] == 6


def test_ascii_uppercase_facet_keywords_dont_silently_truncate(
    tmp_path: Path,
) -> None:
    """Bug 3b: an uppercase-only body parses (or, at minimum, is not silently dropped).

    Companion to Bug 3 — even pre-fix, returning zero triangles silently
    is unsafe. After the case-insensitive matcher fix it parses cleanly.
    """
    path = tmp_path / "uppercase_body_only.stl"
    upper_only = (
        "FACET NORMAL 0 0 1\n"
        " OUTER LOOP\n"
        "  VERTEX 0 0 0\n"
        "  VERTEX 1 0 0\n"
        "  VERTEX 0 1 0\n"
        " ENDLOOP\n"
        "ENDFACET\n"
    )
    path.write_text(f"solid x\n{upper_only}endsolid x\n")

    pts, idx = pyvista_stl.read(str(path))
    assert idx.shape[0] == 1, (
        "silent truncation: parser returned zero triangles for an uppercase body"
    )
    assert pts.shape[0] == 3


def test_ascii_uppercase_solid_header(tmp_path: Path) -> None:
    """KNOWN_BUGS Bug 4: header written as ``SOLID`` (admesh's ``block.stl``).

    The case-insensitive ``solid`` match in ``detect_format`` admits
    these files; the body parser already case-folds keywords.
    """
    path = tmp_path / "upper_solid.stl"
    path.write_text(f"SOLID X\n{_VALID_BODY}ENDSOLID X\n")

    pts, idx = pyvista_stl.read(str(path))
    assert idx.shape[0] == 1
    assert pts.shape[0] == 3


# ---------------------------------------------------------------------------
# Real fixtures from the corpus audit. Skipped when not cached locally.
# ---------------------------------------------------------------------------

_REAL_FIXTURE_DIR = Path("/tmp/stl_audit")


def _fixture(rel: str) -> Path | None:
    p = _REAL_FIXTURE_DIR / rel
    return p if p.exists() else None


@pytest.mark.skipif(
    _fixture("numpystl/tests/stl_ascii/Cube.stl") is None,
    reason="real fixture not cached locally; run download.py from the audit",
)
def test_real_fixture_cube_pro2stl() -> None:
    """Bug 1 on the real PRO2STL Cube fixture (12 triangles, 8 unique vertices).

    VTK's ASCII reader is itself buggy on this file (it appears to
    double-count facets when the line endings are CRLF), so the
    assertion is against the absolute correct count rather than VTK.
    """
    path = _fixture("numpystl/tests/stl_ascii/Cube.stl")
    assert path is not None
    pts, idx = pyvista_stl.read(str(path))
    assert idx.shape[0] == 12
    assert pts.shape[0] == 8


@pytest.mark.skipif(
    _fixture("assimp_src/test/models/STL/sphereWithHole.stl") is None,
    reason="real fixture not cached locally; run download.py from the audit",
)
def test_real_fixture_sphere_with_hole() -> None:
    """Bug 2 on the real assimp ``sphereWithHole`` fixture."""
    path = _fixture("assimp_src/test/models/STL/sphereWithHole.stl")
    assert path is not None
    pts, idx = pyvista_stl.read(str(path))
    vtk_pts, vtk_cells = _vtk_counts(path)
    assert (pts.shape[0], idx.shape[0]) == (vtk_pts, vtk_cells)


@pytest.mark.skipif(
    _fixture("numpystl/tests/stl_ascii/HalfDonut.stl") is None,
    reason="real fixture not cached locally; run download.py from the audit",
)
def test_real_fixture_halfdonut_uppercase_tail() -> None:
    """Bug 3 on the real ``HalfDonut`` fixture (last two facets are uppercase)."""
    path = _fixture("numpystl/tests/stl_ascii/HalfDonut.stl")
    assert path is not None
    pts, idx = pyvista_stl.read(str(path))
    vtk_pts, vtk_cells = _vtk_counts(path)
    assert idx.shape[0] == vtk_cells
    assert pts.shape[0] == vtk_pts


@pytest.mark.skipif(
    _fixture("admeshc/examples/block.stl") is None,
    reason="real fixture not cached locally; run download.py from the audit",
)
def test_real_fixture_admesh_block_uppercase() -> None:
    """KNOWN_BUGS Bug 4 on the real admesh ``block.stl`` (entirely uppercase keywords)."""
    path = _fixture("admeshc/examples/block.stl")
    assert path is not None
    pts, idx = pyvista_stl.read(str(path))
    assert idx.shape[0] > 0
    assert pts.shape[0] > 0
