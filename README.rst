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

On the synthetic 1M-point benchmark below, reading takes about 50 ms on
a Ryzen 9 8945HS, roughly 20x faster than VTK's STL reader and 50x
faster than ``meshio``. The implementation is a memory-mapped parser, a
multi-threaded ASCII path, and a concurrent open-addressing hashtable
for vertex deduplication. See Benchmarks_ for the numbers and the
reproduction script.

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
          [9005998, 9005999, 9005995]], dtype=uint32)

``vertices`` is the deduplicated ``(n_points, 3)`` ``float32`` array.
``indices`` is the ``(n_triangles, 3)`` ``uint32`` array of vertex
indices into ``vertices``. Both binary and ASCII files are accepted; the
format is detected automatically.

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

Reading a 1,002,001-point binary STL (``pyvista.Plane(i_resolution=250,
j_resolution=250).triangulate().subdivide(2)``, 2,000,000 triangles),
measured on a Ryzen 9 8945HS:

+--------------------+-----------------+
| Library            | Time (seconds)  |
+====================+=================+
| ``pyvista-stl``    | 0.051           |
+--------------------+-----------------+
| ``numpy-stl``      | 0.225 [#nps]_   |
+--------------------+-----------------+
| ``pyvista`` (VTK)  | 1.094           |
+--------------------+-----------------+
| ``meshio``         | 2.801           |
+--------------------+-----------------+

.. [#nps]

   ``numpy-stl`` does not merge duplicate vertices, so the time is for the
   larger non-deduplicated representation.

Reproduce these numbers with the script in ``benchmarks/``:

.. code:: sh

   python benchmarks/make_readme_figures.py

Comparison with VTK across mesh sizes
=====================================

The gap widens with file size. ``pyvista-stl`` scales near-linearly;
VTK's reader scales super-linearly:

.. image:: https://github.com/pyvista/pyvista-stl/raw/main/bench0.png

Same data on log-log axes:

.. image:: https://github.com/pyvista/pyvista-stl/raw/main/bench1.png

ASCII files
===========

The ASCII parser is parallelized across CPU cores. On the synthetic
benchmark below, it runs roughly 30-60x faster than VTK on the multi-MB
inputs in the suite.

.. code:: python

   import time
   import pyvista_stl
   import pyvista as pv
   import numpy as np

   # Create and save an ASCII file
   n = 1000
   mesh = pv.Plane(i_resolution=n, j_resolution=n).triangulate()
   mesh.save("/tmp/tmp-ascii.stl", binary=False)

   tstart = time.perf_counter()
   mesh = pyvista_stl.read_as_mesh("/tmp/tmp-ascii.stl")
   print("pyvista-stl   ", time.perf_counter() - tstart)

   tstart = time.perf_counter()
   pv_mesh = pv.read("/tmp/tmp-ascii.stl")
   print("pyvista reader", time.perf_counter() - tstart)

   # Same point set (vertex order is implementation-defined)
   assert np.allclose(np.sort(mesh.points, axis=0),
                      np.sort(pv_mesh.points, axis=0))

   # Approximate timings for the 1M-point file:
   # pyvista-stl    0.022
   # pyvista reader 1.150

***************
 Configuration
***************

``PYVISTA_STL_THREADS`` (integer, default: logical core count, capped at
32) controls the number of worker threads. Set it to ``1`` for the
single-threaded path, which produces deterministic vertex ordering.

``PYVISTA_STL_MAX_TRIS`` (integer, default: 200,000,000) caps the
declared triangle count the reader will accept. Files claiming more
triangles than the cap raise ``RuntimeError`` before any large
allocation, which prevents an attacker-controlled header from forcing
multi-GB allocations.

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
