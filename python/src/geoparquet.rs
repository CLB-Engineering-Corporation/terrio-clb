//! `write_geoparquet_layers`: Arrow tables to one GeoParquet file with the layout options of
//! `terrio::geoparquet`.

use crate::{PyResultExt, invalid};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::time::Instant;
use terrio::geoparquet::{
    ArrowLayer, BboxCovering, ColumnStatistics, DictionaryColumns, GeoParquetWriteOptions, MemoryLayer, ParquetCompression, RowOrder,
    split_by_column, write_arrow_layers_to_path,
};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(write_geoparquet_layers, m)?)?;
    Ok(())
}

/// The record batches of an Arrow object (or a GeoDataFrame, through `to_arrow()`).
fn arrow_of(obj: &Bound<'_, PyAny>) -> PyResult<(terrio::arrow::arrow_schema::SchemaRef, Vec<terrio::arrow::arrow_array::RecordBatch>)> {
    if let Some(x) = crate::arrow::read_capsules(obj)? {
        return Ok(x);
    }
    if obj.hasattr("to_arrow")?
        && let Some(x) = crate::arrow::read_capsules(&obj.call_method0("to_arrow")?)?
    {
        return Ok(x);
    }
    Err(invalid("expected an object with __arrow_c_stream__ or __arrow_c_array__ (a pyarrow Table, ...)"))
}

/// `None` (keep the base option), `""` (no column) or a name.
fn optional_name(value: Option<&str>, base: Option<String>) -> Option<String> {
    match value {
        None => base,
        Some("") => None,
        Some(name) => Some(name.to_string()),
    }
}

/// Writes Arrow tables as one GeoParquet file, with layout options for analysis archives.
///
/// The columns keep their Arrow types (float32, lists, structs, timestamps...); the geometry
/// column holds WKB (`binary`). Several layers share the file, told apart by a layer column, with
/// the union of their fields as the schema.
///
/// Args:
///     layers: A pyarrow Table (or any object exporting Arrow data) or a list of
///         `(name, table)` pairs. A single table holding `layer_column` is split into layers by
///         its values, in first-appearance order.
///     path: The output file, replaced only once the new file is complete.
///     profile: `None` (the defaults below) or `"archive"`: ZSTD level 3, a `layer` column,
///         Hilbert order with a `hilbert_index` column, dictionaries on string columns only and
///         chunk statistics. Keyword arguments override the profile.
///     layer_column: The layer name column (default none; `""` for none).
///     compression: `"snappy"` (default), `"zstd"` or `"uncompressed"`.
///     compression_level: The ZSTD level (1 to 22; default 1, 3 in the archive profile).
///     row_group_rows: The maximum rows of a row group (default 65536).
///     layer_row_groups: Whether row groups also end at layer boundaries (default True).
///     bbox_covering: `"struct"` (default), `"columns"` or `"none"`.
///     bbox_column: The covering struct's name, or the flat columns' prefix (default `"bbox"`).
///     order: `"input"` (default) or `"hilbert"` (DuckDB `ST_Hilbert` of the bbox centres over
///         each layer's extent).
///     hilbert_column: With Hilbert order, a uint64 column for the keys (`""` for none).
///     dictionary: `"all"` (default), `"utf8"` or `"none"`.
///     statistics: `"page"` (default, with a page index) or `"chunk"` (as pyarrow writes).
///     geometry_statistics: Whether the WKB column has min/max statistics (default True).
///     geometry_column: The WKB column (default `"geometry"`).
///     crs: The `crs` of the `geo` metadata: a PROJJSON dict or string, or another CRS string;
///         by default the geometry field's GeoArrow metadata, or the table's `geo` metadata.
///     identity: Columns identifying rows, which must not hold nulls: a list for every layer,
///         or a dict of lists by layer name. They are non-nullable when every layer has them so.
///
/// Returns:
///     dict: `rows`, `layers`, `row_groups`, `bbox` (or None) and `seconds`.
#[pyfunction]
#[pyo3(signature = (
    layers, path, *, profile=None, layer_column=None, compression=None, compression_level=None, row_group_rows=None,
    layer_row_groups=None, bbox_covering=None, bbox_column=None, order=None, hilbert_column=None, dictionary=None,
    statistics=None, geometry_statistics=None, geometry_column=None, crs=None, identity=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_geoparquet_layers<'py>(
    py: Python<'py>,
    layers: &Bound<'py, PyAny>,
    path: std::path::PathBuf,
    profile: Option<&str>,
    layer_column: Option<&str>,
    compression: Option<&str>,
    compression_level: Option<i32>,
    row_group_rows: Option<usize>,
    layer_row_groups: Option<bool>,
    bbox_covering: Option<&str>,
    bbox_column: Option<&str>,
    order: Option<&str>,
    hilbert_column: Option<&str>,
    dictionary: Option<&str>,
    statistics: Option<&str>,
    geometry_statistics: Option<bool>,
    geometry_column: Option<&str>,
    crs: Option<&Bound<'py, PyAny>>,
    identity: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut options = match profile {
        None => GeoParquetWriteOptions::default(),
        Some("archive") => GeoParquetWriteOptions::archive(),
        Some(other) => return Err(invalid(format!("unknown profile {other:?} (archive)"))),
    };
    options.layer_column = optional_name(layer_column, options.layer_column);
    options.hilbert_column = optional_name(hilbert_column, options.hilbert_column);
    if compression.is_some() || compression_level.is_some() {
        let name = compression.unwrap_or(match options.compression {
            ParquetCompression::Zstd(_) => "zstd",
            ParquetCompression::Snappy => "snappy",
            ParquetCompression::Uncompressed => "uncompressed",
        });
        let level = match (compression_level, options.compression) {
            (Some(l), _) => Some(l),
            (None, ParquetCompression::Zstd(l)) if name.eq_ignore_ascii_case("zstd") => Some(l),
            _ => None,
        };
        options.compression = ParquetCompression::parse(name, level).py()?;
    }
    if let Some(n) = row_group_rows {
        options.row_group_rows = n;
    }
    if let Some(b) = layer_row_groups {
        options.layer_row_groups = b;
    }
    if let Some(s) = bbox_covering {
        options.bbox_covering = BboxCovering::parse(s).py()?;
    }
    if let Some(s) = bbox_column {
        options.bbox_column = s.to_string();
    }
    if let Some(s) = order {
        options.order = RowOrder::parse(s).py()?;
    }
    if let Some(s) = dictionary {
        options.dictionary = DictionaryColumns::parse(s).py()?;
    }
    if let Some(s) = statistics {
        options.statistics = ColumnStatistics::parse(s).py()?;
    }
    if let Some(b) = geometry_statistics {
        options.geometry_statistics = b;
    }
    if let Some(s) = geometry_column {
        options.geometry_column = s.to_string();
    }
    if let Some(c) = crs.filter(|c| !c.is_none()) {
        options.crs = Some(if let Ok(s) = c.extract::<String>() {
            match serde_json::from_str::<serde_json::Value>(&s) {
                Ok(v @ serde_json::Value::Object(_)) => v,
                _ => serde_json::Value::String(s),
            }
        } else {
            let text: String = py.import("json")?.call_method1("dumps", (c,))?.extract()?;
            serde_json::from_str(&text).map_err(|e| invalid(format!("crs: {e}")))?
        });
    }

    // The layers.
    let mut inputs: Vec<MemoryLayer> = Vec::new();
    if let Ok(list) = layers.cast::<PyList>() {
        for item in list.iter() {
            let pair = item.cast::<PyTuple>().map_err(|_| invalid("layers must be a table or a list of (name, table) pairs"))?;
            if pair.len() != 2 {
                return Err(invalid("layers must be a table or a list of (name, table) pairs"));
            }
            let name: Option<String> = pair.get_item(0)?.extract()?;
            let (schema, batches) = arrow_of(&pair.get_item(1)?)?;
            inputs.push(ArrowLayer::new(name, schema, batches.into_iter().map(Ok).collect()));
        }
    } else {
        let (schema, batches) = arrow_of(layers)?;
        match options.layer_column.as_deref().filter(|c| schema.index_of(c).is_ok()) {
            Some(column) => inputs = py.detach(|| split_by_column(&schema, &batches, column)).py()?,
            None => inputs.push(ArrowLayer::new(None, schema, batches.into_iter().map(Ok).collect())),
        }
    }
    if let Some(identity) = identity.filter(|i| !i.is_none()) {
        if let Ok(by_layer) = identity.cast::<PyDict>() {
            for (key, columns) in by_layer.iter() {
                let key: String = key.extract()?;
                let layer = inputs
                    .iter_mut()
                    .find(|l| l.name.as_deref() == Some(key.as_str()))
                    .ok_or_else(|| invalid(format!("identity names an unknown layer {key:?}")))?;
                layer.identity = columns.extract::<Vec<String>>()?;
            }
        } else {
            let columns: Vec<String> = identity.extract()?;
            for layer in &mut inputs {
                layer.identity = columns.clone();
            }
        }
    }

    let path = path.to_str().ok_or_else(|| invalid("the path is not valid Unicode"))?.to_string();
    let started = Instant::now();
    let report = crate::with_warnings(py, move || write_arrow_layers_to_path(inputs, &path, &options))?;
    let out = PyDict::new(py);
    out.set_item("rows", report.rows)?;
    out.set_item("layers", report.layers)?;
    out.set_item("row_groups", report.row_groups)?;
    out.set_item("bbox", report.bbox.map(|b| b.to_vec()))?;
    out.set_item("seconds", started.elapsed().as_secs_f64())?;
    Ok(out)
}
