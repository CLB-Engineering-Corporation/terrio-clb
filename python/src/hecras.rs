//! `hecras_archive_layers` and `write_hecras_archive`: HEC-RAS geometry and plan HDF files
//! straight to archive GeoParquet (`terrio::hecras::archive`).

use crate::invalid;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::time::Instant;
use terrio::geoparquet::{ArrowLayer, GeoParquetWriteOptions, MemoryLayer};
use terrio::hecras::HecRasHdf;
use terrio::hecras::archive::{ArchiveLayerKind, ArchiveOptions, ArchiveSkip, HecRasArchive, write_archive};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hecras_archive_layers, m)?)?;
    m.add_function(wrap_pyfunction!(write_hecras_archive, m)?)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn options(
    geometry: bool,
    results: bool,
    layers: Option<Vec<String>>,
    variables: Option<Vec<String>>,
    summary_geometry: bool,
    ghost_cell_points: bool,
    time_days: bool,
    batch_rows: usize,
) -> ArchiveOptions {
    let mut o = ArchiveOptions::default();
    o.geometry = geometry;
    o.results = results;
    o.layers = layers;
    o.variables = variables;
    o.summary_geometry = summary_geometry;
    o.ghost_cell_points = ghost_cell_points;
    o.time_days = time_days;
    o.batch_rows = batch_rows;
    o
}

fn read(path: std::path::PathBuf, options: ArchiveOptions) -> terrio::Result<HecRasArchive> {
    HecRasHdf::open(path)?.archive_layers(&options)
}

fn skipped<'py>(py: Python<'py>, skipped: &[ArchiveSkip]) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for s in skipped {
        let d = PyDict::new(py);
        d.set_item("layer", &s.layer)?;
        d.set_item("variable", &s.variable)?;
        d.set_item("kind", &s.kind)?;
        d.set_item("code", &s.code)?;
        d.set_item("reason", &s.reason)?;
        out.append(d)?;
    }
    Ok(out)
}

/// Reads the archive layers of a HEC-RAS geometry (`.g##.hdf`) or plan (`.p##.hdf`) HDF file
/// as Arrow tables, with no GeoDataFrame in between.
///
/// The layers are the HEC-RAS engine's catalog layers (ras-commander-fast-hdf) under ras2cng's
/// archive names and field names (`mesh_areas`, `bc_line_id`, ...): the geometry layers, then
/// one layer per stored 2D summary variable (named by its slug, in name order) with the cell
/// polygon or face line joined by `(mesh_name, cell_id|face_id)`. The geometry column is WKB
/// (`geoarrow.wkb`) without a CRS: the file's stored projection is `crs_wkt`.
///
/// A layer or variable the engine refuses (for example `Cell Last Iteration`, which stores no
/// element location) does not fail the call: it is listed in `skipped`, so that the caller can
/// read it another way and pass it to `write_hecras_archive(extra_layers=...)`.
///
/// Args:
///     path: The HDF file.
///     geometry: Read the geometry layers (default True).
///     results: Read the 2D summary results (default True).
///     layers: Only these geometry layers (archive or catalog names). Default: every stored one.
///     variables: Only these summary variables (stored names or slugs). Default: all.
///     summary_geometry: Join the element geometry to summary rows (default True).
///     ghost_cell_points: Keep ghost cells in `mesh_cell_points`, flagged `is_ghost_cell`
///         (default True).
///     time_days: Add `<slug>_time_days`, the stored time offset, to timed summaries (default
///         True).
///     batch_rows: Rows per record batch (default 65536).
///
/// Returns:
///     dict: `layers` (a list of dicts: `name`, `kind` (`geometry` or `summary`), `canonical`
///     or `variable`, `identity`, `rows`, `table`, an `ArrowTable`), `skipped` (a list of dicts:
///     `layer`, `variable`, `kind`, `code`, `reason`), `crs_wkt` and `seconds`.
#[pyfunction]
#[pyo3(signature = (
    path, *, geometry=true, results=true, layers=None, variables=None, summary_geometry=true,
    ghost_cell_points=true, time_days=true, batch_rows=65536,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn hecras_archive_layers<'py>(
    py: Python<'py>,
    path: std::path::PathBuf,
    geometry: bool,
    results: bool,
    layers: Option<Vec<String>>,
    variables: Option<Vec<String>>,
    summary_geometry: bool,
    ghost_cell_points: bool,
    time_days: bool,
    batch_rows: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let o = options(geometry, results, layers, variables, summary_geometry, ghost_cell_points, time_days, batch_rows);
    let started = Instant::now();
    let archive = crate::with_warnings(py, move || read(path, o))?;
    let out = PyDict::new(py);
    let list = PyList::empty(py);
    for layer in archive.layers {
        let d = PyDict::new(py);
        d.set_item("name", &layer.name)?;
        match &layer.kind {
            ArchiveLayerKind::Geometry { canonical } => {
                d.set_item("kind", "geometry")?;
                d.set_item("canonical", canonical)?;
            }
            ArchiveLayerKind::Summary { variable } => {
                d.set_item("kind", "summary")?;
                d.set_item("variable", variable)?;
            }
            _ => d.set_item("kind", "other")?,
        }
        d.set_item("identity", &layer.identity)?;
        d.set_item("rows", layer.num_rows())?;
        d.set_item("table", Py::new(py, crate::arrow::PyArrowTable::new(layer.schema, layer.batches))?)?;
        list.append(d)?;
    }
    out.set_item("layers", list)?;
    out.set_item("skipped", skipped(py, &archive.skipped)?)?;
    out.set_item("crs_wkt", archive.crs_wkt)?;
    out.set_item("seconds", started.elapsed().as_secs_f64())?;
    Ok(out)
}

/// Writes a HEC-RAS geometry or plan HDF file's archive layers (see `hecras_archive_layers`),
/// then `extra_layers`, as one GeoParquet file with the archive profile of
/// `write_geoparquet_layers` (ZSTD 3, `layer` column, layer row groups, `bbox` struct, Hilbert
/// order with `hilbert_index`, Utf8 dictionaries, chunk statistics).
///
/// Args:
///     path: The HDF file.
///     out: The GeoParquet file, replaced only once the new file is complete.
///     extra_layers: `(name, table)` pairs (any Arrow table with a WKB `geometry` column) written
///         after the HDF layers: text geometry layers, values the engine refused.
///     extra_identity: The identity columns of extra layers, a dict of lists by layer name.
///     order: Layer names in the order to write them (every layer once). Default: the HDF layers,
///         then the extra layers.
///     crs: The `crs` of the `geo` metadata (PROJJSON dict or string). Default: the stored
///         projection as PROJJSON (its text when terrio cannot convert it), `null` without one.
///     geometry, results, layers, variables, summary_geometry, ghost_cell_points, time_days,
///         batch_rows: as `hecras_archive_layers`.
///
/// Returns:
///     dict: `rows`, `layers`, `row_groups`, `bbox`, `layer_rows` (rows by layer name, HDF layers
///     only), `skipped`, `crs_wkt` and `seconds`.
#[pyfunction]
#[pyo3(signature = (
    path, out, *, extra_layers=None, extra_identity=None, order=None, crs=None, geometry=true, results=true,
    layers=None, variables=None, summary_geometry=true, ghost_cell_points=true, time_days=true, batch_rows=65536,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_hecras_archive<'py>(
    py: Python<'py>,
    path: std::path::PathBuf,
    out: std::path::PathBuf,
    extra_layers: Option<&Bound<'py, PyAny>>,
    extra_identity: Option<&Bound<'py, PyDict>>,
    order: Option<Vec<String>>,
    crs: Option<&Bound<'py, PyAny>>,
    geometry: bool,
    results: bool,
    layers: Option<Vec<String>>,
    variables: Option<Vec<String>>,
    summary_geometry: bool,
    ghost_cell_points: bool,
    time_days: bool,
    batch_rows: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let o = options(geometry, results, layers, variables, summary_geometry, ghost_cell_points, time_days, batch_rows);
    let mut extra: Vec<MemoryLayer> = Vec::new();
    if let Some(items) = extra_layers.filter(|e| !e.is_none()) {
        for item in items.try_iter()? {
            let item = item?;
            let pair = item.cast::<PyTuple>().map_err(|_| invalid("extra_layers must be (name, table) pairs"))?;
            if pair.len() != 2 {
                return Err(invalid("extra_layers must be (name, table) pairs"));
            }
            let name: String = pair.get_item(0)?.extract()?;
            let (schema, batches) = crate::geoparquet::arrow_of(&pair.get_item(1)?)?;
            let mut layer = ArrowLayer::new(Some(name.clone()), schema, batches.into_iter().map(Ok).collect());
            if let Some(identity) = extra_identity.and_then(|d| d.get_item(&name).ok().flatten()) {
                layer.identity = identity.extract()?;
            }
            extra.push(layer);
        }
    }
    let mut write = GeoParquetWriteOptions::archive();
    write.crs = crate::geoparquet::crs_value(py, crs)?;
    let out = out.to_str().ok_or_else(|| invalid("the output path is not valid Unicode"))?.to_string();
    let started = Instant::now();
    let (report, archive_rows, skipped_layers, crs_wkt) = crate::with_warnings(py, move || {
        let archive = read(path, o)?;
        let rows: Vec<(String, usize)> = archive.layers.iter().map(|l| (l.name.clone(), l.num_rows())).collect();
        let skipped = archive.skipped.clone();
        let wkt = archive.crs_wkt.clone();
        let report = write_archive(archive, extra, order.as_deref(), &out, &write)?;
        Ok((report, rows, skipped, wkt))
    })?;
    let d = PyDict::new(py);
    d.set_item("rows", report.rows)?;
    d.set_item("layers", report.layers)?;
    d.set_item("row_groups", report.row_groups)?;
    d.set_item("bbox", report.bbox.map(|b| b.to_vec()))?;
    let layer_rows = PyDict::new(py);
    for (name, rows) in archive_rows {
        layer_rows.set_item(name, rows)?;
    }
    d.set_item("layer_rows", layer_rows)?;
    d.set_item("skipped", skipped(py, &skipped_layers)?)?;
    d.set_item("crs_wkt", crs_wkt)?;
    d.set_item("seconds", started.elapsed().as_secs_f64())?;
    Ok(d)
}
