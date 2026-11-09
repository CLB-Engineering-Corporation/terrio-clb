"""Writes the synthetic breakline fixtures of crates/terrio/tests/data/hecras/ for CI.

They hold no model data: every coordinate and name is made up. The `Geometry/2D Flow Area Break
Lines` group has the dataset names, shapes and types HEC-RAS 6.x/7.0 writes (`Attributes` with
`Name` S32, `Cell Spacing Near`/`Far` f4, `Near Repeats` i4, `Protection Radius` u1; `Polyline
Info` (n, 4), `Polyline Parts` (n, 2) i4; `Polyline Points` (n, 2) f8), as in HEC's public
example geometries. A one-point breakline is what HEC-RAS 6.1 stored in a real model (Spring_BLE
g01, row 6010).

- `breaklines_good.h5`: a three-point line and a two-part line (2 + 2 points);
- `breaklines_empty.h5`: those two, then a breakline with no points;
- `breaklines_point.h5`: the two, a one-point breakline, a breakline with no points, then a
  two-point line;
- `breaklines_short_part.h5`: the two, then a three-point breakline whose parts have 2 and 1
  points (ras-commander raises on it).

Usage: python -I make_hecras_breaklines_fixture.py <out_dir>
"""
import sys
from pathlib import Path

import h5py
import numpy as np

GROUP = "Geometry/2D Flow Area Break Lines"

GOOD = [
    ("Upper Levee", [[(0.0, 0.0), (10.0, 0.5), (20.0, 1.0)]]),
    ("Road Embankment", [[(5.0, 5.0), (6.0, 6.5)], [(7.0, 7.0), (8.25, 9.0)]]),
]
EMPTY = ("Empty", [[]])
POINT = ("Single Point", [[(3.5, 4.5)]])
LAST = ("Lower Levee", [[(30.0, 0.0), (31.5, 2.0)]])
SHORT_PART = ("Short Part", [[(40.0, 0.0), (41.0, 1.0)], [(42.0, 2.0)]])


def write(path, rows):
    attrs = np.zeros(len(rows), dtype=[("Name", "S32"), ("Cell Spacing Near", "<f4"), ("Cell Spacing Far", "<f4"),
                                       ("Near Repeats", "<i4"), ("Protection Radius", "u1")])
    info, parts, points = [], [], []
    for i, (name, lines) in enumerate(rows):
        attrs[i] = (name.encode(), 10.0 + i, 50.0 + i if i % 2 == 0 else np.nan, i, i % 3)
        start = len(points)
        part_start = len(parts)
        offset = 0
        for line in lines:
            parts.append((offset, len(line)))
            offset += len(line)
            points.extend(line)
        info.append((start, len(points) - start, part_start, len(lines)))
    with h5py.File(path, "w") as f:
        f.attrs["File Type"] = np.bytes_(b"HEC-RAS Geometry")
        f.attrs["File Version"] = np.bytes_(b"HEC-RAS 6.6 September 2024")
        f.attrs["Units System"] = np.bytes_(b"US Customary")
        g = f.create_group(GROUP)
        g.create_dataset("Attributes", data=attrs)
        g.create_dataset("Polyline Info", data=np.asarray(info, dtype="<i4").reshape(-1, 4))
        g.create_dataset("Polyline Parts", data=np.asarray(parts, dtype="<i4").reshape(-1, 2))
        g.create_dataset("Polyline Points", data=np.asarray(points, dtype="<f8").reshape(-1, 2))


def main(out):
    out = Path(out)
    write(out / "breaklines_good.h5", GOOD)
    write(out / "breaklines_empty.h5", GOOD + [EMPTY])
    write(out / "breaklines_point.h5", GOOD + [POINT, EMPTY, LAST])
    write(out / "breaklines_short_part.h5", GOOD + [SHORT_PART])


if __name__ == "__main__":
    main(sys.argv[1])
