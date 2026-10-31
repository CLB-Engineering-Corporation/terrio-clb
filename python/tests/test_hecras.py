"""Tests of the HECRAS vector driver (HEC-RAS geometry and plan HDF files) through the Python
API: detection by content, layers, the stored CRS, and joins by open options. Uses the synthetic
plan `crates/terrio/tests/data/hecras/results_mesh_xs.h5` (`scripts/make_hecras_fixture.py`)."""

from pathlib import Path

import pytest

import terrio

DATA = Path(__file__).resolve().parents[2] / "crates" / "terrio" / "tests" / "data" / "hecras"
PLAN = str(DATA / "results_mesh_xs.h5")

pytestmark = pytest.mark.skipif(
    "HECRAS" not in {d["name"] for d in terrio.drivers()}, reason="built without the hecras feature"
)


def test_plans_open_as_vector_layers():
    ds = terrio.open_vector(PLAN)
    assert ds.driver == "HECRAS"
    assert ds.layer_names[:3] == ["mesh_cells", "mesh_cell_points", "mesh_faces"]
    assert "cross_sections" in ds.layer_names
    cells = ds["mesh_cells"]
    assert cells.geometry_type == "Polygon"
    assert list(cells.fields)[:2] == ["mesh_name", "cell_id"]
    # The stored projection text, with no EPSG code guessed.
    assert cells.crs_wkt.startswith('PROJCS["Synthetic"')
    # `open` reads it as vector layers too.
    assert isinstance(terrio.open(PLAN), terrio.VectorDataset)


def test_open_options_join_stored_values():
    ds = terrio.open_vector(
        PLAN,
        layers=["mesh_cells"],
        options={
            "CELL_SUMMARIES": "Maximum Water Surface",
            "CELL_SERIES": "Water Surface",
            "TIME_STEP": "1",
        },
    )
    cells = ds["mesh_cells"]
    assert {"maximum_water_surface", "maximum_water_surface_time", "water_surface"} <= set(
        cells.fields
    )
    assert cells.columns()["water_surface"] == [12.75, 12.25]


def test_feature_layers_at_a_time_step():
    ds = terrio.open_vector(
        PLAN,
        layers=["cross_sections", "bc_lines"],
        options={"XS_RESULTS": "Water Surface", "BC_SERIES": "YES", "TIME_STEP": "2"},
    )
    assert ds["cross_sections"].columns()["water_surface"] == [21.0, 19.0, 15.5]
    assert ds["bc_lines"].columns()["stage"] == [12.25, 11.25]


def test_info_takes_open_options():
    summary = terrio.info(
        PLAN, layers=["mesh_cells"], options={"CELL_ATTRIBUTES": "MINIMUM_ELEVATION"}
    )
    assert summary["driver"] == "HECRAS"
    assert "minimum_elevation" in str(summary["layers"][0])


def test_misuse_is_an_error():
    # A series without TIME_STEP, and a value the file does not store: errors, never empty layers.
    with pytest.raises(terrio.TerrioError):
        terrio.open_vector(PLAN, layers=["mesh_cells"], options={"CELL_SERIES": "Water Surface"})
    with pytest.raises(terrio.TerrioError):
        terrio.open_vector(
            PLAN, layers=["mesh_cells"], options={"CELL_SERIES": "Depth", "TIME_STEP": "0"}
        )
    with pytest.raises(terrio.TerrioError):
        terrio.open_vector(PLAN, options={"NO_SUCH_OPTION": "1"})
    with pytest.raises(terrio.TerrioError):
        terrio.open_vector(PLAN, layers=["structures"])


def test_raster_opens_keep_reading_hdf5():
    # A raster open of a HEC-RAS file still goes to the HDF5 raster driver, as before: it reads
    # the file's arrays, or fails as that driver does, never through the HECRAS driver.
    try:
        terrio.open_raster(PLAN)
    except terrio.TerrioError as e:
        assert "HECRAS" not in str(e)
