"""Validate the parser against real-world STL files.

Downloads a small corpus of real STL files via PyVista's example
datasets (cached locally) and compares the parser's output against
``vtkSTLReader``. The corpus covers binary files from different
writers (netfabb, Artec, LYMB), an OpenSCAD ASCII file with CRLF line
endings, and sizes from 300 KB to about 15 MB.

Tests are skipped when a dataset cannot be fetched, so the suite still
passes on offline CI.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import pyvista_stl


pv = pytest.importorskip("pyvista")
examples = pv.examples


def _safe_download(loader: Callable[..., str]) -> str | None:
    """Fetch a PyVista example and return the local path, or ``None`` if unavailable.

    Network or filesystem failures return ``None`` so the caller can
    skip the test cleanly on offline CI.
    """
    try:
        return str(loader(load=False))
    except Exception:  # noqa: BLE001 - any network/IO failure means skip
        return None


# Each entry is (loader, format-tag) for a real-world STL produced by a
# distinct writer or formatting style:
# - netfabb / Artec / LYMB binary outputs
# - OpenSCAD ASCII output with CRLF line endings
# - File sizes from ~300 KB to ~15 MB
# - Headers that begin with "solid " on otherwise-binary files
_CORPUS: list[tuple[Callable[..., str], str]] = [
    (examples.download_great_white_shark, "binary-netfabb"),
    (examples.download_grey_nurse_shark, "binary-artec"),
    (examples.download_gears, "ascii-openscad-crlf"),
    (examples.download_urn, "binary-artec-large"),
    (examples.download_woman, "binary-artec-largest"),
]


def test_realworld_corpus_at_least_one_available() -> None:
    """Fail (do not silently skip) if every real-world example fetch fails.

    Without this guard a CI run with broken example downloads would
    show every parametrized real-world test as "skipped" and the suite
    would still pass green, hiding the regression.
    """
    available = [loader.__name__ for loader, _ in _CORPUS if _safe_download(loader)]
    assert available, "no real-world STL examples could be fetched; tests would all skip"


@pytest.fixture(params=_CORPUS, ids=[tag for _, tag in _CORPUS])
def stl_example(request: pytest.FixtureRequest) -> str:
    loader, _ = request.param
    path = _safe_download(loader)
    if path is None or not path.endswith(".stl"):
        pytest.skip(f"STL example {loader.__name__} not available")
        raise AssertionError("unreachable: pytest.skip exits the test")
    return path


def _vtk_polydata(path: str) -> Any:
    """Read with VTK's STL reader, used as the ground-truth oracle."""
    from vtkmodules.vtkIOGeometry import vtkSTLReader

    reader = vtkSTLReader()
    reader.SetFileName(path)
    reader.Merging = True  # match pyvista-stl's default vertex merging
    reader.Update()
    return pv.wrap(reader.GetOutput())


def test_realworld_matches_vtk(stl_example: str) -> None:
    """The parser produces a mesh equivalent to VTK's reader."""
    points, indices = pyvista_stl.read(stl_example)
    vtk_mesh = _vtk_polydata(stl_example)

    # Vertex counts after merging must match.
    assert points.shape == (vtk_mesh.n_points, 3)
    assert indices.shape == (vtk_mesh.n_cells, 3)

    # Bounds match exactly (we round-trip the same float32 bytes).
    np.testing.assert_array_equal(
        np.asarray(pv.PolyData(points).bounds),
        np.asarray(vtk_mesh.bounds),
    )

    # Point sets match (independent of vertex ordering).
    np.testing.assert_array_equal(
        np.sort(points, axis=0),
        np.sort(np.asarray(vtk_mesh.points), axis=0),
    )


def test_realworld_read_as_mesh(stl_example: str) -> None:
    """``read_as_mesh`` returns a PolyData equivalent to ``pv.read``."""
    ours = pyvista_stl.read_as_mesh(stl_example)
    theirs = _vtk_polydata(stl_example)

    assert ours.n_points == theirs.n_points
    assert ours.n_cells == theirs.n_cells
    np.testing.assert_array_equal(
        np.sort(np.asarray(ours.points), axis=0),
        np.sort(np.asarray(theirs.points), axis=0),
    )


_PARSER_VALIDATION_FIELDS = (
    "non_finite_points",
    "invalid_point_references",
    "wrong_number_of_points",
    "cell_data_wrong_length",
    "point_data_wrong_length",
    "unused_points",
)


def test_realworld_validates(stl_example: str) -> None:
    """Real-world parsed meshes pass PyVista's parser-attributable validation."""
    mesh = pyvista_stl.read_as_mesh(stl_example)
    report = mesh.validate_mesh(validation_fields=list(_PARSER_VALIDATION_FIELDS))
    failures = {f: getattr(report, f) for f in _PARSER_VALIDATION_FIELDS if getattr(report, f)}
    assert not failures, f"parser validation failed for {stl_example}: {failures}"
