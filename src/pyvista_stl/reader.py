"""High-level Python API for the pyvista-stl reader."""

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from pyvista_stl import _core as _stlfile_wrapper

if TYPE_CHECKING:
    from pyvista.core.pointset import PolyData

try:
    # On pyvista >= 0.48, raising ``LocalFileRequiredError`` from a
    # reader entry point makes ``pv.read("http://.../foo.stl")``
    # download the file first and retry against the local copy.
    from pyvista import LocalFileRequiredError as _LocalFileRequiredError
    from pyvista import has_scheme as _has_scheme
except ImportError:  # pragma: no cover - older pyvista or no pyvista installed
    _LocalFileRequiredError = None
    _has_scheme = None


def _polydata_from_faces(
    points: npt.NDArray[np.floating[Any]],
    faces: npt.NDArray[np.int32] | npt.NDArray[np.int64],
) -> "PolyData":
    """Build a :class:`pyvista.PolyData` from a triangle connectivity array.

    Bypasses :class:`pyvista.PolyData`'s padded-faces inflater for
    better throughput when the input is already triangles-only.

    Parameters
    ----------
    points : numpy.ndarray
        ``(n_points, 3)`` array of vertex coordinates.
    faces : numpy.ndarray
        ``(n_faces, 3)`` array of vertex indices. Must be ``int32`` or
        ``int64``.

    Returns
    -------
    pyvista.PolyData
        Triangulated polydata holding the supplied points and faces.

    """
    try:
        from pyvista import vtk_version_info
        from pyvista.core.pointset import PolyData
    except ModuleNotFoundError as exc:
        msg = "pyvista_stl.read_as_mesh requires PyVista. Install it with: pip install pyvista"
        raise ModuleNotFoundError(msg) from exc

    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import vtkTypeInt32Array, vtkTypeInt64Array
    from vtkmodules.vtkCommonDataModel import vtkCellArray

    if faces.ndim != 2:
        msg = f"Expected a 2-D face array, got shape {faces.shape!r}."
        raise ValueError(msg)

    if faces.dtype == np.int32:
        vtk_dtype = vtkTypeInt32Array().GetDataType()
    elif faces.dtype == np.int64:
        vtk_dtype = vtkTypeInt64Array().GetDataType()
    else:
        msg = f"Unsupported face dtype {faces.dtype!r}; expected int32 or int64."
        raise TypeError(msg)

    faces_vtk = numpy_to_vtk(faces.ravel(), deep=False, array_type=vtk_dtype)

    carr = vtkCellArray()
    if vtk_version_info >= (9, 6, 2):
        carr.SetData(faces.shape[1], faces_vtk)
    else:
        offset = np.arange(0, faces.size + 1, faces.shape[1], dtype=faces.dtype)
        offset_vtk = numpy_to_vtk(offset, deep=False, array_type=vtk_dtype)
        carr.SetData(offset_vtk, faces_vtk)
        carr._offset_np_ref = offset_vtk

    # Keep references on the cell array so the underlying numpy buffers
    # are not garbage-collected while VTK still holds raw pointers.
    carr._faces_np_ref = faces_vtk

    pdata = PolyData()
    pdata.points = points
    pdata.SetPolys(carr)
    return pdata


def read(
    filename: str | os.PathLike[str],
    *,
    threads: int = 1,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Read an STL file and return its merged vertex and triangle arrays.

    Both binary and ASCII STL files are supported; the format is
    detected automatically.

    Parameters
    ----------
    filename : str | os.PathLike
        Path to the STL file.
    threads : int, default: 1
        Number of worker threads. ``1`` (the default) uses the
        single-threaded path, which produces deterministic vertex
        ordering and is the safest choice for embedded/server use.
        Pass an integer ``>= 2`` to opt into the multi-threaded
        ASCII/binary parsers, or ``0`` to auto-select
        ``hardware_concurrency()``. Worker counts are capped at 32.

    Returns
    -------
    vertices : numpy.ndarray
        ``(n_points, 3)`` ``float32`` array of unique merged vertex
        coordinates.
    indices : numpy.ndarray
        ``(n_triangles, 3)`` ``int32`` array of vertex indices into
        ``vertices``.

    Raises
    ------
    RuntimeError
        If the file is missing, unreadable, or not a valid STL.
    pyvista.LocalFileRequiredError
        If a remote URI is passed and ``pyvista >= 0.48`` is installed.
        The PyVista reader registry uses this to download the file and
        retry against the local copy.

    Examples
    --------
    >>> import pyvista_stl
    >>> vertices, indices = pyvista_stl.read("example.stl")
    >>> vertices
    array([[-0.01671113,  0.5450843 , -0.8382146 ],
           [ 0.01671113,  0.5450843 , -0.8382146 ],
           [ 0.        ,  0.52573115, -0.8506509 ],
           ...,
           [ 0.5952229 , -0.57455426,  0.56178033],
           [ 0.56178033, -0.5952229 ,  0.57455426],
           [ 0.57455426, -0.56178033,  0.5952229 ]], dtype=float32)
    >>> indices
    array([[      0,       1,       2],
           [      1,       3,       4],
           [      4,       5,       2],
           ...,
           [9005998, 9005988, 9005999],
           [9005999, 9005996, 9005995],
           [9005998, 9005999, 9005995]], dtype=int32)

    """
    fname = os.fspath(filename)
    # When invoked via PyVista's readers registry with a remote URI,
    # signal PyVista to download the file and retry locally.
    if _has_scheme is not None and _has_scheme(fname):
        raise _LocalFileRequiredError
    return _stlfile_wrapper.get_stl_data(fname, threads)


def read_as_mesh(
    filename: str | os.PathLike[str],
    *,
    threads: int = 1,
) -> "PolyData":
    """Read an STL file and return it as a :class:`pyvista.PolyData`.

    Wraps :func:`read` and packs the merged vertex/triangle arrays
    into a :class:`pyvista.PolyData` without an extra copy.

    Parameters
    ----------
    filename : str | os.PathLike
        Path to the STL file.
    threads : int, default: 1
        Number of worker threads. ``1`` (the default) uses the
        single-threaded path. Pass ``>= 2`` to opt into the
        multi-threaded parsers, or ``0`` to auto-select
        ``hardware_concurrency()``. Worker counts are capped at 32.

    Returns
    -------
    pyvista.PolyData
        Triangulated polydata.

    Raises
    ------
    ModuleNotFoundError
        If PyVista is not installed.
    RuntimeError
        If the file is missing, unreadable, or not a valid STL.

    Notes
    -----
    Requires the ``pyvista`` package. The connectivity array is
    ``int32`` by default; it is promoted to ``int64`` when the
    connectivity offset exceeds the ``int32`` range.

    Examples
    --------
    >>> import pyvista_stl
    >>> mesh = pyvista_stl.read_as_mesh("example.stl")
    >>> mesh
    PolyData (0x7f43063ec700)
      N Cells:    1280000
      N Points:   641601
      N Strips:   0
      X Bounds:   -5.000e-01, 5.000e-01
      Y Bounds:   -5.000e-01, 5.000e-01
      Z Bounds:   -5.551e-17, 5.551e-17
      N Arrays:   0

    """
    vertices, indices = read(filename, threads=threads)

    # ``read`` already returns int32 indices. The vtkCellArray offset
    # array is what dictates int32 vs int64: it spans
    # ``range(0, indices.size + 1, 3)``, so promote both arrays to
    # int64 only when ``indices.size`` itself overflows int32.
    indices_int: npt.NDArray[np.int32] | npt.NDArray[np.int64]
    if indices.size >= np.iinfo(np.int32).max:
        indices_int = indices.astype(np.int64, copy=False)
    else:
        indices_int = indices
    return _polydata_from_faces(vertices, indices_int)
