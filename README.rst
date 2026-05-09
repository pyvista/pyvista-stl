#############
 pyvista-stl
#############

|pypi| |MIT|

.. |pypi| image:: https://img.shields.io/pypi/v/pyvista-stl.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/pyvista-stl/

.. |MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

A fast STL reader for Python. Reads binary and ASCII files, merges
duplicate vertices on the way in, and returns NumPy arrays.

On the synthetic 1M-point binary benchmark below, the default
single-threaded path reads in about 100 ms on a Ryzen 9 8945HS, roughly
11x faster than VTK and 28x faster than ``meshio``. Opting into the
multi-threaded path with ``threads=0`` (auto) reads the same file in 50
ms — about 21x faster than VTK and 55x faster than ``meshio``. The
implementation is a memory-mapped parser, an optional multi-threaded
ASCII path, and a concurrent open-addressing hashtable for vertex
deduplication. See Benchmarks_ for the numbers and the reproduction
script.

The vertex hash function (``final96``) and the iterative table sizing
helper (``nextpow2``) are taken from `aki5/libstl
<https://github.com/aki5/libstl>`_; see ``src/hash96.h``. The rest of
the parser is independent.

**************
 Installation
**************

.. code:: sh

   pip install pyvista-stl

To build from source:

.. code:: sh

   git clone https://github.com/pyvista/pyvista-stl.git
   cd pyvista-stl
   pip install .

*******
 Usage
*******

Read an STL file as merged ``(vertices, indices)`` arrays:

.. code:: pycon

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

``vertices`` is the deduplicated ``(n_points, 3)`` ``float32`` array.
``indices`` is the ``(n_triangles, 3)`` ``int32`` array of vertex
indices into ``vertices``. Both binary and ASCII files are accepted; the
format is detected automatically.

By default the reader runs single-threaded, which produces a
deterministic vertex ordering. Pass ``threads=N`` (an integer ``>= 2``)
to opt into the multi-threaded parser, or ``threads=0`` to auto-select
``hardware_concurrency()``:

.. code:: python

   vertices, indices = pyvista_stl.read("example.stl", threads=0)
   mesh = pyvista_stl.read_as_mesh("example.stl", threads=8)

To get a ``pyvista.PolyData`` directly:

.. code:: pycon

   >>> import pyvista_stl
   >>> mesh = pyvista_stl.read_as_mesh('example.stl')
   >>> mesh
   PolyData (0x7f43063ec700)
     N Cells:    1280000
     N Points:   641601
     N Strips:   0
     X Bounds:   -5.000e-01, 5.000e-01
     Y Bounds:   -5.000e-01, 5.000e-01
     Z Bounds:   -5.551e-17, 5.551e-17
     N Arrays:   0

With ``pyvista >= 0.48`` installed, ``pyvista.read`` automatically
dispatches ``.stl`` files to ``pyvista_stl`` via the ``pyvista.readers``
entry point:

.. code:: pycon

   >>> import pyvista as pv
   >>> mesh = pv.read("example.stl")  # uses pyvista_stl

************
 Benchmarks
************

Reading a 1,002,001-point STL (``pyvista.Plane(i_resolution=250,
j_resolution=250).triangulate().subdivide(2)``, 2,000,000 triangles),
median of 5 runs on a 16-core Ryzen 9 8945HS. The two right-hand columns
show how much faster ``pyvista-stl`` is than the reader in that row, in
single-threaded (``threads=1``, the default) and multi-threaded
(``threads=0``, all cores) configurations.

Binary STL (~95 MB on disk):

+----------------------------+----------------+----------------------------+----------------------------+
| Reader                     | Time (seconds) | ``pyvista-stl`` ST speedup | ``pyvista-stl`` MT speedup |
+============================+================+============================+============================+
| ``pyvista-stl`` (1 thread) | 0.100          | (baseline)                 | 2.0x slower                |
+----------------------------+----------------+----------------------------+----------------------------+
| ``pyvista-stl`` (16 thr.)  | 0.051          | 2.0x faster                | (baseline)                 |
+----------------------------+----------------+----------------------------+----------------------------+
| ``numpy-stl``              | 0.206 [#nps]_  | 2.1x faster                | 4.0x faster                |
+----------------------------+----------------+----------------------------+----------------------------+
| ``pyvista`` (VTK)          | 1.080          | 10.8x faster               | 21.2x faster               |
+----------------------------+----------------+----------------------------+----------------------------+
| ``meshio``                 | 3.041          | 30.5x faster               | 59.7x faster               |
+----------------------------+----------------+----------------------------+----------------------------+

ASCII STL (~425 MB on disk):

+----------------------------+----------------+----------------------------+----------------------------+
| Reader                     | Time (seconds) | ``pyvista-stl`` ST speedup | ``pyvista-stl`` MT speedup |
+============================+================+============================+============================+
| ``pyvista-stl`` (1 thread) | 0.388          | (baseline)                 | 3.4x slower                |
+----------------------------+----------------+----------------------------+----------------------------+
| ``pyvista-stl`` (16 thr.)  | 0.114          | 3.4x faster                | (baseline)                 |
+----------------------------+----------------+----------------------------+----------------------------+
| ``pyvista`` (VTK)          | 2.761          | 7.1x faster                | 24.2x faster               |
+----------------------------+----------------+----------------------------+----------------------------+
| ``meshio``                 | 9.464          | 24.4x faster               | 83.0x faster               |
+----------------------------+----------------+----------------------------+----------------------------+

.. [#nps]

   ``numpy-stl`` does not merge duplicate vertices, so the time is for the
   larger non-deduplicated representation.

Across the fixture corpus in ``benchmarks/bench.py`` (binary and ASCII
files from a few KB to roughly 100 MB), single-threaded ``pyvista-stl``
is a median of **10.6x faster** than VTK, ranging from **4.0x to
134.3x**, and is never slower than VTK on any tested file. The
multi-threaded path widens the gap further on the larger ASCII files.

Reproduce these numbers with the script in ``benchmarks/``:

.. code:: sh

   python benchmarks/make_readme_figures.py

Comparison with VTK across mesh sizes
=====================================

The gap widens with file size. ``pyvista-stl`` scales near-linearly on
both the single-threaded and multi-threaded paths; VTK's reader scales
super-linearly. By the time the mesh reaches ~10 M points (~20 M
triangles, ~1 GB binary), ``pyvista-stl`` is roughly 65x faster
single-threaded and 160x faster multi-threaded:

.. image:: https://github.com/pyvista/pyvista-stl/raw/main/bench0.png

Same data on log-log axes:

.. image:: https://github.com/pyvista/pyvista-stl/raw/main/bench1.png

***************
 Configuration
***************

The ``threads`` keyword argument on ``read`` and ``read_as_mesh``
controls worker concurrency:

-  ``threads=1`` (default): single-threaded, deterministic vertex
   ordering. The safest choice for embedded/server use.
-  ``threads=N`` (``N >= 2``): use ``N`` workers. Worker counts are
   capped at 32.
-  ``threads=0``: auto-select using
   ``std::thread::hardware_concurrency()``.

``PYVISTA_STL_MAX_TRIS`` (environment variable, default:
``200_000_000``) caps the declared triangle count the reader will
accept. Files claiming more triangles than the cap raise
``RuntimeError`` before any large allocation, which prevents an
attacker-controlled header from forcing multi-GB allocations.

*****************************
 License and acknowledgments
*****************************

This project began as a wrapper around `aki5/libstl
<https://github.com/aki5/libstl>`_; the binary-format reader and the
hash-based vertex merge are derived from that library, used under its
`MIT License <https://github.com/aki5/libstl/blob/master/LICENSE>`_.

Significant changes since: mmap-backed input, ASCII reader, a
multi-threaded path with a concurrent hashtable, hugepage-backed scratch
buffers, and a nanobind interface.

This repository is also licensed under the MIT License.

*********
 Support
*********

Please open an issue at `pyvista/pyvista-stl
<https://github.com/pyvista/pyvista-stl/issues>`_ if you hit a problem.
