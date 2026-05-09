"""Reproduce the benchmark figures and 1M-point table in the README.

Writes ``bench0.png`` (linear) and ``bench1.png`` (log-log) to the
repo root and prints the 1M-point comparison table to stdout.

Examples
--------
    python benchmarks/make_readme_figures.py
"""

import tempfile
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _common import READERS, make_test_mesh, time_call


def _bind(reader: Callable[[Path], object], path: Path) -> Callable[[], object]:
    """Return a zero-argument callable that invokes ``reader(path)``."""
    return lambda: reader(path)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sweep(stl_path: Path, resolutions: list[int], n: int) -> np.ndarray:
    """Return ``(len(resolutions), 4)`` array of [n_points, vtk_s, pvstl_st_s, pvstl_mt_s]."""
    rows = []
    for res in resolutions:
        mesh = make_test_mesh(res)
        mesh.save(stl_path)
        _, vtk_med = time_call(_bind(READERS["vtk"], stl_path), n=n)
        _, st_med = time_call(_bind(READERS["pyvista_stl"], stl_path), n=n)
        _, mt_med = time_call(_bind(READERS["pyvista_stl_mt"], stl_path), n=n)
        rows.append((mesh.n_points, vtk_med, st_med, mt_med))
        print(
            f"res={res:4d}  npts={mesh.n_points:>10,}  "
            f"vtk={vtk_med * 1000:9.1f}ms  pvstl(st)={st_med * 1000:9.1f}ms  "
            f"pvstl(mt)={mt_med * 1000:9.1f}ms  "
            f"speedup_st={vtk_med / st_med:5.1f}x  speedup_mt={vtk_med / mt_med:5.1f}x"
        )
    return np.asarray(rows)


def _plot(rows: np.ndarray, *, log_axes: bool, output: Path, title: str) -> None:
    npts, vtk_t, st_t, mt_t = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    plt.figure(figsize=(7, 5))
    plt.title(title)
    plot = plt.loglog if log_axes else plt.plot
    plot(npts, vtk_t, "o-", label="VTK", color="C1")
    plot(npts, st_t, "o-", label="pyvista-stl (single-threaded)", color="C0")
    plot(npts, mt_t, "o--", label="pyvista-stl (multi-threaded)", color="C2")
    plt.xlabel("Number of points")
    plt.ylabel("Time to load (seconds)")
    plt.grid(alpha=0.3, which="both" if log_axes else "major")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=110)
    plt.close()


def _table(stl_path: Path, n: int) -> None:
    target_mesh = make_test_mesh(250)  # ~ 1,002,001 points
    target_mesh.save(stl_path)
    print(
        f"\n1M-point table file: {target_mesh.n_points:,} points, {target_mesh.n_cells:,} triangles"
    )
    for name, fn in READERS.items():
        best, med = time_call(_bind(fn, stl_path), n=n)
        print(f"  {name:<14} best={best * 1000:7.1f} ms  median={med * 1000:7.1f} ms")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="pvstl_bench_") as tmpdir:
        stl = Path(tmpdir) / "bench.stl"
        rows = _sweep(stl, list(range(50, 801, 50)), n=3)
        _plot(rows, log_axes=False, output=REPO_ROOT / "bench0.png", title="STL load time")
        _plot(rows, log_axes=True, output=REPO_ROOT / "bench1.png", title="STL load time (log-log)")
        _table(stl, n=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
