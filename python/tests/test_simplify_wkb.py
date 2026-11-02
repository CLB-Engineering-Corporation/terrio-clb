"""terrio.simplify_wkb: Douglas-Peucker on WKB, against GEOS's DouglasPeuckerSimplifier (shapely)."""

import math
import random

import pytest

import terrio

shapely = pytest.importorskip("shapely")


def _dp(geoms, tolerance, **kw):
    out = terrio.simplify_wkb([None if g is None else shapely.to_wkb(g) for g in geoms], tolerance, **kw)
    return [None if b is None else shapely.from_wkb(b) for b in out]


def _same_vertices(a, b):
    return a.geom_type == b.geom_type and shapely.equals_exact(a, b, tolerance=0) and _coords(a) == _coords(b)


def _coords(g):
    return shapely.get_coordinates(g).tolist()


def _wobbly_ring(rng, cx, cy, radius, n, noise):
    pts = []
    for k in range(n):
        a = 2 * math.pi * k / n
        r = radius * (1 + noise * (rng.random() - 0.5))
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def test_lines_and_rings_like_geos():
    line = shapely.LineString([(0, 0), (1, 0.05), (2, 0)])
    square = shapely.Polygon([(5, 0), (10, 0), (10, 10), (0, 10), (0, 0), (5, 0)])
    cw = shapely.Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (5, 0.01)])
    closed = shapely.LineString([(5, 0), (10, 0), (10, 10), (0, 10), (0, 0), (5, 0)])
    point = shapely.Point(1, 2)
    for g in [line, square, cw, closed, point]:
        (ours,) = _dp([g], 0.1)
        assert _same_vertices(ours, shapely.simplify(g, 0.1, preserve_topology=False)), g.wkt


def test_random_polygons_match_geos_vertex_for_vertex():
    rng = random.Random(15)
    geoms = []
    for _ in range(300):
        shell = _wobbly_ring(rng, 0, 0, 100, rng.randint(8, 400), rng.choice([0.01, 0.05, 0.2]))
        holes = [_wobbly_ring(rng, rng.uniform(-30, 30), rng.uniform(-30, 30), 10, rng.randint(6, 60), 0.05)[::-1]]
        geoms.append(shapely.Polygon(shell, holes if rng.random() < 0.5 else None))
        geoms.append(shapely.LineString(shell[: rng.randint(2, len(shell))]))
    far = shapely.Polygon([(x + 500, y) for x, y in geoms[2].exterior.coords])
    geoms.append(shapely.MultiPolygon([geoms[0], far]))
    geoms.append(None)
    compared = 0
    for tol in [0.0, 0.1, 1.0, 5.0]:
        ours = _dp(geoms, tol, collapse="drop")
        geos = shapely.simplify(geoms, tol, preserve_topology=False)
        for g, a, b in zip(geoms, ours, geos):
            if g is None:
                assert a is None
                continue
            if not b.is_valid or not shapely.is_valid(a):
                continue  # GEOS rebuilds an invalid result with buffer(0); terrio does not.
            assert _same_vertices(a, b), (tol, g.wkt[:80])
            compared += 1
    assert compared > 2000


def test_collapse_keep_and_drop():
    shell = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    thin = [(1, 1), (2, 1), (2, 1.05), (1, 1.05), (1, 1)]
    hole = [(5, 5), (6, 5), (6, 6), (5, 6), (5, 5)]
    p = shapely.Polygon(shell, [thin, hole])
    out, stats = terrio.simplify_wkb([shapely.to_wkb(p)], 0.1, stats=True)
    assert shapely.from_wkb(out[0]).equals_exact(p, 0)
    assert stats["rings_kept"] == 1 and stats["rings_dropped"] == 0
    out, stats = terrio.simplify_wkb([shapely.to_wkb(p)], 0.1, collapse="drop", stats=True)
    assert _same_vertices(shapely.from_wkb(out[0]), shapely.simplify(p, 0.1, preserve_topology=False))
    assert stats["rings_dropped"] == 1
    empty = shapely.Polygon(thin)
    (dropped,) = _dp([empty], 0.1, collapse="drop")
    assert dropped.is_empty and shapely.simplify(empty, 0.1, preserve_topology=False).is_empty
    (kept,) = _dp([empty], 0.1)
    assert kept.equals_exact(empty, 0)


def test_stats_and_z():
    g = shapely.LineString([(0, 0, 1), (1, 0.05, 2), (2, 0, 3)])
    out, stats = terrio.simplify_wkb([shapely.to_wkb(g), None], 0.1, stats=True)
    assert _coords(shapely.from_wkb(out[0])) == [[0, 0], [2, 0]]
    assert shapely.get_coordinates(shapely.from_wkb(out[0]), include_z=True)[:, 2].tolist() == [1, 3]
    assert out[1] is None
    assert {k: stats[k] for k in ("features", "nulls", "vertices_in", "vertices_out")} == {
        "features": 2,
        "nulls": 1,
        "vertices_in": 3,
        "vertices_out": 2,
    }
    assert stats["seconds"] >= 0


def test_preserve_topology_method_matches_geos():
    rng = random.Random(3)
    polys = [shapely.Polygon(_wobbly_ring(rng, 0, 0, 100, 200, 0.2)) for _ in range(20)]
    ours = _dp(polys, 5.0, method="preserve-topology")
    geos = shapely.simplify(polys, 5.0, preserve_topology=True)
    for a, b in zip(ours, geos):
        assert _same_vertices(a, b)


def test_errors():
    line = shapely.to_wkb(shapely.LineString([(0, 0), (1, 1)]))
    with pytest.raises(ValueError, match="tolerance"):
        terrio.simplify_wkb([line], -1)
    with pytest.raises(ValueError, match="method"):
        terrio.simplify_wkb([line], 1, method="visvalingam")
    with pytest.raises(ValueError, match="collapse"):
        terrio.simplify_wkb([line], 1, collapse="maybe")
    with pytest.raises(ValueError, match="feature 1"):
        terrio.simplify_wkb([line, line[:12]], 1)
    with pytest.raises(ValueError, match="bytes"):
        terrio.simplify_wkb(["LINESTRING (0 0, 1 1)"], 1)


def test_arrow_arrays_keep_their_type():
    pa = pytest.importorskip("pyarrow")
    wkb = [shapely.to_wkb(shapely.LineString([(0, 0), (1, 0.05), (2, 0)])), None]
    for typ in [pa.binary(), pa.large_binary()]:
        out = terrio.simplify_wkb(pa.array(wkb, typ), 0.1)
        assert isinstance(out, pa.Array) and out.type == typ
        assert out.null_count == 1
        assert _coords(shapely.from_wkb(out[0].as_py())) == [[0, 0], [2, 0]]
    chunked = pa.chunked_array([pa.array(wkb[:1], pa.binary()), pa.array(wkb[1:], pa.binary())])
    assert len(terrio.simplify_wkb(chunked, 0.1)) == 2
    with pytest.raises(ValueError, match="WKB"):
        terrio.simplify_wkb(pa.array([1, 2]), 0.1)


def test_layer_simplify_douglas_peucker():
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": [[0, 0], [5, 0.1], [10, 0]]}}
        ],
    }
    out = terrio.simplify(fc, 1.0, method="douglas-peucker").to_geojson()
    assert out["features"][0]["geometry"]["coordinates"] == [[0, 0], [10, 0]]
    with pytest.raises(ValueError):
        terrio.simplify(fc, 1.0, method="nope")
