"""Benchmark STL readers across a directory of ``.stl`` files.

Generates a small set of binary and ASCII STL fixtures under
``--dir`` (idempotent: existing files are reused) and prints
per-reader best/median read times.

Examples
--------
    python benchmarks/bench.py
    python benchmarks/bench.py --dir /tmp/stl_bench --readers pyvista_stl,vtk
    python benchmarks/bench.py --readers pyvista_stl,pyvista_stl_mt,vtk
"""

import argparse
from collections.abc import Callable
from pathlib import Path

from _common import READERS, make_test_mesh, time_call

# (label, plane_resolution, subdivide_levels, ascii_too)
DEFAULT_FIXTURES: tuple[tuple[str, int, int, bool], ...] = (
    ("tiny", 50, 0, True),
    ("small", 200, 1, True),
    ("medium", 400, 2, True),
    ("large", 700, 2, True),
    ("xl", 1000, 2, False),  # ASCII too slow to write at this size
)


def _ensure_fixtures(directory: Path) -> None:
    """Create benchmark STL files under ``directory`` if they don't yet exist."""
    directory.mkdir(parents=True, exist_ok=True)
    for label, res, sub, ascii_too in DEFAULT_FIXTURES:
        binary = directory / f"{label}_bin.stl"
        ascii_path = directory / f"{label}_ascii.stl"
        if binary.exists() and (not ascii_too or ascii_path.exists()):
            continue
        mesh = make_test_mesh(res, subdivide=sub)
        print(f"generating {label}: {mesh.n_points} points, {mesh.n_cells} triangles")
        if not binary.exists():
            mesh.save(binary)
        if ascii_too and not ascii_path.exists():
            mesh.save(ascii_path, binary=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=Path("/tmp/pvstl_bench"))
    parser.add_argument("-n", type=int, default=3, help="repetitions per measurement")
    parser.add_argument(
        "--readers",
        default="pyvista_stl,pyvista_stl_mt,vtk",
        help="comma-separated reader names: " + ",".join(READERS),
    )
    args = parser.parse_args()

    _ensure_fixtures(args.dir)
    selected = [name.strip() for name in args.readers.split(",") if name.strip()]
    unknown = [n for n in selected if n not in READERS]
    if unknown:
        parser.error(f"unknown reader(s): {unknown}")
    fns = {name: READERS[name] for name in selected}

    files = sorted(p for p in args.dir.iterdir() if p.suffix == ".stl")
    header = f"{'file':<25} {'size (MB)':>10}  " + "  ".join(f"{n:>16}" for n in fns)
    print(header)
    for path in files:
        size_mb = path.stat().st_size / 1e6
        cells = [f"{path.name:<25} {size_mb:>10.2f}"]
        for fn in fns.values():
            try:
                best, med = time_call(_invoke(fn, path), n=args.n)
                cells.append(f"{best * 1000:7.1f}/{med * 1000:6.1f}ms")
            except Exception as exc:  # noqa: BLE001 - we want any reader failure to surface in the table
                cells.append(f"ERR:{exc!r:>8}")
        print("  ".join(cells))
    return 0


def _invoke(reader: Callable[[Path], object], path: Path) -> Callable[[], object]:
    """Bind ``reader`` and ``path`` into a zero-argument callable for ``time_call``."""
    return lambda: reader(path)


if __name__ == "__main__":
    raise SystemExit(main())
