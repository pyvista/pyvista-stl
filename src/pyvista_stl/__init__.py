"""pyvista-stl reader library."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from pyvista_stl.reader import read, read_as_mesh

try:
    __version__ = _version("pyvista-stl")
except PackageNotFoundError:
    __version__ = "0.0.0"


__all__ = ["__version__", "read", "read_as_mesh"]
