"""terrio.write_geoparquet_layers: Arrow tables to GeoParquet with archive layout options."""

import json
import struct

import pytest

import terrio

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

CRS = {"id": {"authority": "EPSG", "code": 26918}}


def point(x, y):
    return struct.pack("<BIdd", 1, 1, x, y)


def square(x, y, size):
    ring = [(x, y), (x + size, y), (x + size, y + size), (x, y + size), (x, y)]
    return struct.pack("<BII", 1, 3, 1) + struct.pack("<I", len(ring)) + b"".join(struct.pack("<dd", *p) for p in ring)


def geometry_field():
    return pa.field("geometry", pa.binary(), metadata={"ARROW:extension:name": "geoarrow.wkb"})


def table():
    geometries = [square(100, 100, 8), square(0, 0, 4), square(50, 50, 2), point(7, 9), None]
    return pa.table(
        {
            "mesh_name": ["m", "m", "m", "p", "q"],
            "cell_id": pa.array([0, 1, 2, 0, 0], pa.int64()),
            "value": pa.array([1.5, None, 3.0, 4.0, 5.0], pa.float32()),
            "stations": pa.array([[1.0], [2.0, 3.0], None, [], [4.0]]),
            "geometry": pa.array(geometries, pa.binary()),
            "layer": ["cells", "cells", "cells", "points", "points"],
        }
    ).cast(
        pa.schema(
            [
                pa.field("mesh_name", pa.string()),
                pa.field("cell_id", pa.int64()),
                pa.field("value", pa.float32()),
                pa.field("stations", pa.list_(pa.float64())),
                geometry_field(),
                pa.field("layer", pa.string()),
            ],
            metadata={"geo": json.dumps({"primary_column": "geometry", "columns": {"geometry": {"encoding": "WKB", "crs": CRS}}})},
        )
    )


def test_archive_profile(tmp_path):
    out = tmp_path / "archive.parquet"
    report = terrio.write_geoparquet_layers(
        table(), out, profile="archive", row_group_rows=2, identity=["mesh_name", "cell_id"]
    )
    assert (report["rows"], report["layers"], report["row_groups"]) == (5, 2, 3)
    pf = pq.ParquetFile(out)
    assert [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)] == [2, 1, 2]
    t = pf.read()
    assert t.schema.names == ["mesh_name", "cell_id", "value", "stations", "geometry", "layer", "bbox", "hilbert_index"]
    assert [f.nullable for f in t.schema] == [False, False, True, True, True, False, True, True]
    assert t.schema.field("value").type == pa.float32()
    assert t.schema.field("geometry").metadata == {b"ARROW:extension:name": b"geoarrow.wkb"}
    # Hilbert order within each layer; the null geometry last.
    assert t["cell_id"].to_pylist() == [1, 2, 0, 0, 0]
    assert t["mesh_name"].to_pylist() == ["m", "m", "m", "p", "q"]
    assert t["bbox"].to_pylist()[0] == {"xmin": 0.0, "ymin": 0.0, "xmax": 4.0, "ymax": 4.0}
    assert t["bbox"][4].as_py() is None and t["hilbert_index"][4].as_py() is None
    geo = json.loads(pf.metadata.metadata[b"geo"])
    column = geo["columns"]["geometry"]
    assert geo["version"] == "1.1.0" and column["crs"] == CRS
    assert column["geometry_types"] == ["Point", "Polygon"]
    assert column["bbox"] == [0.0, 0.0, 108.0, 108.0]
    assert column["covering"]["bbox"]["xmin"] == ["bbox", "xmin"]
    # The Arrow schema stored in the file carries the geo metadata too.
    assert b"geo" in pq.read_schema(out).metadata
    rg = pf.metadata.row_group(0)
    columns = {rg.column(i).path_in_schema: rg.column(i) for i in range(rg.num_columns)}
    assert {name for name, c in columns.items() if c.has_dictionary_page} == {"mesh_name", "layer"}
    assert all(c.compression == "ZSTD" for c in columns.values())
    assert not any(c.has_offset_index for c in columns.values())


def test_pairs_and_overrides(tmp_path):
    t = table().drop_columns(["layer"])
    out = tmp_path / "pairs.parquet"
    terrio.write_geoparquet_layers(
        [("a", t.slice(0, 3)), ("b", t.slice(3))],
        out,
        layer_column="kind",
        compression="zstd",
        compression_level=5,
        bbox_covering="columns",
        crs="EPSG:26918",
    )
    back = pq.read_table(out)
    assert back["kind"].to_pylist() == ["a", "a", "a", "b", "b"]
    assert back["cell_id"].to_pylist() == [0, 1, 2, 0, 0]  # input order
    assert {"bbox_xmin", "bbox_ymax"} <= set(back.schema.names)
    geo = json.loads(pq.ParquetFile(out).metadata.metadata[b"geo"])
    assert geo["columns"]["geometry"]["crs"] == "EPSG:26918"


def test_refusals(tmp_path):
    with pytest.raises(terrio.InvalidArgumentError):
        terrio.write_geoparquet_layers(table(), tmp_path / "x.parquet", profile="nope")
    with pytest.raises(terrio.InvalidArgumentError):
        terrio.write_geoparquet_layers(table(), tmp_path / "x.parquet", compression="snappy", compression_level=3)
    with pytest.raises(terrio.InvalidArgumentError):
        # More than one layer needs a layer column.
        terrio.write_geoparquet_layers([("a", table()), ("b", table())], tmp_path / "x.parquet")
    with pytest.raises(terrio.InvalidArgumentError):
        # Nulls in an identity column.
        terrio.write_geoparquet_layers(table(), tmp_path / "x.parquet", profile="archive", identity=["value"])
    assert not (tmp_path / "x.parquet").exists()
