#################
 Test STL corpus
#################

A curated set of STL files used by ``tests/test_corpus.py``. The entire
corpus is well under 1 MB. Each file is small and targets a specific
format variant or edge case. Coverage:

-  Format detection: ASCII vs binary, including the trap where a binary
   file's 80-byte header begins with ``"solid "``.
-  Binary writer signatures: blank-padded headers and headers matching
   real-world writers (Geomagic Studio, netfabb).
-  Per-triangle attribute byte counts: zero (the common case) and
   non-zero (also legal under the spec).
-  ASCII line endings: ``\n``, ``\r\n``, and the rare ``\r``-only
   variant produced by some classic-Mac-era exporters.
-  ASCII indentation: spaces, tabs, none.
-  ASCII float formats: defaults plus scientific notation.
-  Sizes: 0-, 1-, and many-triangle meshes.
-  Topology: well-merged manifolds (cube, sphere, grid) and the
   fully-disconnected case where every vertex is unique.

To regenerate the corpus, run:

.. code::

   python tests/data/_generate.py

The validation tests in ``tests/test_corpus.py`` walk every ``.stl``
file in this directory and compare the parser's output against
``vtkSTLReader`` with merging enabled.
