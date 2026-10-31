"""Tests of the direct HEC-RAS HDF -> archive GeoParquet path through Python
(`terrio.hecras.archive_layers`, `terrio.hecras.write_archive`), on the synthetic fixtures
`arrow_layers.h5` (a plan with summaries) and `results_mesh_xs.h5` (feature layers)."""

import json
from pathlib import Path

import pytest

import terrio

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

DATA = Path(__file__).resolve().parents[2] / "crates" / "terrio" / "tests" / "data" / "hecras"
PLAN = str(DATA / "arrow_layers.h5")
FEATURES = str(DATA / "results_mesh_xs.h5")

pytestmark = pytest.mark.skipif(
    not hasattr(terrio._terrio, "write_hecras_archive"), reason="built without the hecras feature"
)


def test_archive_layers_are_arrow_tables_with_ras2cng_names():
    import terrio.hecras

    found = terrio.hecras.archive_layers(FEATURES, results=False)
    by_name = {layer["name"]: layer for layer in found["layers"]}
    assert {"mesh_cells", "mesh_faces", "bc_lines", "cross_sections", "mesh_cell_points"} <= set(by_name)
    bc = by_name["bc_lines"]
    assert bc["kind"] == "geometry" and bc["canonical"] == "bc_lines"
    assert bc["identity"] == ["bc_line_id"]
    table = pa.table(bc["table"])
    assert table.num_rows == bc["rows"]
    assert table.column_names[0] == "bc_line_id"
    assert table.schema.field("geometry").metadata[b"ARROW:extension:name"] == b"geoarrow.wkb"
    points = pa.table(by_name["mesh_cell_points"]["table"])
    assert "is_ghost_cell" in points.column_names
    assert found["crs_wkt"].startswith("PROJCS")
    # The mesh stores no Perimeter: refused, listed, not fatal.
    assert any(s["layer"] == "mesh_areas" for s in found["skipped"])


def test_summaries_with_geometry_and_stored_times():
    import terrio.hecras

    found = terrio.hecras.archive_layers(PLAN, geometry=False)
    by_name = {layer["name"]: layer for layer in found["layers"]}
    wse = pa.table(by_name["maximum_water_surface"]["table"])
    assert by_name["maximum_water_surface"]["variable"] == "Maximum Water Surface"
    assert wse.column_names == [
        "mesh_name",
        "cell_id",
        "maximum_water_surface",
        "maximum_water_surface_time",
        "maximum_water_surface_time_days",
        "is_ghost_cell",
        "geometry",
    ]
    assert wse.column("maximum_water_surface_time_days").to_pylist() == [0.5, 1.25]
    assert wse.column("geometry").null_count == 0
    plain = terrio.hecras.archive_layers(
        PLAN, geometry=False, time_days=False, summary_geometry=False
    )
    plain_layers = {layer["name"]: layer for layer in plain["layers"]}
    wse = pa.table(plain_layers["maximum_water_surface"]["table"])
    assert "maximum_water_surface_time_days" not in wse.column_names and "geometry" not in wse.column_names
    only = terrio.hecras.archive_layers(PLAN, geometry=False, variables=["maximum_water_surface"])
    assert [layer["name"] for layer in only["layers"]] == ["maximum_water_surface"]


def test_write_archive_with_extra_layers(tmp_path):
    import terrio.hecras

    extra = pa.table(
        {
            "Name": pa.array(["SA 1"]),
            "geometry": pa.array([bytes.fromhex("0101000000000000000000e03f000000000000e03f")], pa.binary()),
        }
    )
    out = tmp_path / "plan.parquet"
    report = terrio.hecras.write_archive(
        PLAN,
        out,
        extra_layers=[("storage_areas_text", extra)],
        extra_identity={"storage_areas_text": ["Name"]},
    )
    table = pq.read_table(out)
    counts = dict(zip(*[c.to_pylist() for c in table.column("layer").value_counts().flatten()]))
    for name, rows in report["layer_rows"].items():
        assert counts[name] == rows
    assert counts["storage_areas_text"] == 1
    assert report["rows"] == table.num_rows
    geo = json.loads(pq.read_schema(out).metadata[b"geo"])
    # This fixture stores no projection, so none is guessed; a stored one is written.
    assert geo["columns"]["geometry"]["crs"] is None
    terrio.hecras.write_archive(FEATURES, tmp_path / "features.parquet", results=False)
    stored = json.loads(pq.read_schema(tmp_path / "features.parquet").metadata[b"geo"])
    assert stored["columns"]["geometry"]["crs"] is not None
    assert {"bbox", "hilbert_index", "layer"} <= set(table.column_names)
    # A caller CRS replaces the stored one; an unknown layer in `order` is refused.
    terrio.hecras.write_archive(PLAN, out, crs="EPSG:4326", geometry=False)
    assert json.loads(pq.read_schema(out).metadata[b"geo"])["columns"]["geometry"]["crs"] == "EPSG:4326"
    with pytest.raises(terrio.InvalidArgumentError):
        terrio.hecras.write_archive(PLAN, out, order=["nope"])
