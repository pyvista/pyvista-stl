"""Validate parser output with PyVista's mesh-validation API.

Every STL file the parser accepts must produce a mesh that passes
PyVista's validation for the **parser-attributable** fields below.
Fields that depend on the *source* mesh's geometric quality
(e.g. ``degenerate_faces``, ``intersecting_faces``, ``inverted_faces``,
``coincident_points``, ``non_convex``, ``non_planar_faces``) are
*not* checked here: those describe the input STL, not parser
correctness. A faithful parser that round-trips a degenerate input
would otherwise be flagged as broken.

Covered fields:

- ``non_finite_points`` — parser must not invent ``NaN``/``Inf``.
- ``invalid_point_references`` — every face must reference a valid id.
- ``wrong_number_of_points`` — every face is a triangle.
- ``cell_data_wrong_length`` / ``point_data_wrong_length`` — array
  lengths consistent with cell/point counts.
- ``unused_points`` — after dedup, every emitted point is referenced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import pyvista_stl

pv = pytest.importorskip("pyvista")

# Parser-attributable validation fields.
_PARSER_FIELDS = (
    "non_finite_points",
    "invalid_point_references",
    "wrong_number_of_points",
    "cell_data_wrong_length",
    "point_data_wrong_length",
    "unused_points",
)


def _bundled_test_stls() -> list[Path]:
    """Every ``.stl`` fixture under ``tests/data/`` plus the two top-level
    sphere examples."""
    here = Path(__file__).parent
    paths = list((here / "data").glob("*.stl"))
    paths += [here / "sphere_ascii.stl", here / "sphere_binary.stl"]
    return sorted(p for p in paths if p.is_file())


@pytest.fixture(params=_bundled_test_stls(), ids=lambda p: p.name)
def stl_path(request: pytest.FixtureRequest) -> Path:
    return request.param


def test_parsed_mesh_is_valid(stl_path: Path) -> None:
    """Every parsed mesh passes PyVista validation on parser-attributable fields."""
    points, indices = pyvista_stl.read(str(stl_path))

    # Empty meshes (zero-triangle fixtures) trivially satisfy validation.
    if points.shape[0] == 0 and indices.shape[0] == 0:
        return

    mesh = pyvista_stl.read_as_mesh(str(stl_path))
    report = mesh.validate_mesh(validation_fields=list(_PARSER_FIELDS))

    failures: dict[str, object] = {}
    for field in _PARSER_FIELDS:
        value = getattr(report, field)
        if value:
            failures[field] = value

    assert not failures, (
        f"Parser produced an invalid mesh for {stl_path.name}: {failures}"
    )


def test_parsed_indices_in_range(stl_path: Path) -> None:
    """Connectivity values are always within ``[0, n_points)``."""
    points, indices = pyvista_stl.read(str(stl_path))
    if indices.size == 0:
        return
    assert int(indices.min()) >= 0
    assert int(indices.max()) < points.shape[0]


def test_parsed_points_finite(stl_path: Path) -> None:
    """Parser never emits NaN or Inf coordinates for clean fixtures."""
    import numpy as np

    points, _ = pyvista_stl.read(str(stl_path))
    if points.size == 0:
        return
    assert np.isfinite(points).all(), f"non-finite coords in {stl_path.name}"
