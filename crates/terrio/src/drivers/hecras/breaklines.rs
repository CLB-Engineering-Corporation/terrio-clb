//! The `breaklines` layer with degenerate breaklines left out, as ras-commander reads it.
//!
//! HEC-RAS can store a 2D breakline with fewer than two points (Spring_BLE g01, HEC-RAS 6.1, has
//! one with a single point). ras-commander's `HdfBndry.get_breaklines` skips such a breakline,
//! logs it and keeps the others; the engine's catalog layer
//! (`ras_commander_fast_hdf::arrow_layers`) refuses the whole layer over one part with fewer
//! than two points, and draws a breakline with no points as a row without geometry. This module
//! reads the layer the way ras-commander does:
//!
//! - a breakline whose stored point count (`Polyline Info` column 1) is below two is left out
//!   and reported as an [`OmittedRow`] (`BREAKLINE_TOO_FEW_POINTS`, ras-commander's
//!   `reason_code`). The other rows keep their `source_row` (ras-commander's `bl_id`), so the
//!   omitted row leaves a gap;
//! - every other rule is the engine's: one part is a LineString, several a MultiLineString, a
//!   part with fewer than two points inside a breakline of two or more points still refuses the
//!   layer (ras-commander raises on it too), and the stored attribute columns keep their names
//!   and the engine's types (text, Int64, UInt64, Float32 or Float64 with NaN as null).
//!
//! When the engine reads the layer, its batches are used as they are, less any row without
//! geometry. Only when it refuses the layer over a short part are the rows built here from the
//! engine's typed family read, with the engine's column and WKB rules; on files the engine reads,
//! the two give identical batches (a test checks this).

use super::HecRasHdf;
use arrow_array::builder::{BinaryBuilder, StringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int64Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use ras_commander_fast_hdf::arrow_layers::{ArrowLayer, LayerOptions, consumer_field_name, geometry_field, wkb_line};
use ras_commander_fast_hdf::hdf::lines::GeometryFamily;
use ras_commander_fast_hdf::hdf::metadata::{CompoundTable, MetadataValue};
use ras_commander_fast_hdf::hdf::{AttributeType, NumericValues};
use ras_commander_fast_hdf::{Capability, Error, ErrorKind, Result, UnsupportedCode};
use std::sync::Arc;

/// The canonical catalog name of the layer.
pub(crate) const LAYER: &str = "breaklines";
/// ras-commander's `reason_code` for a breakline it skips.
pub const TOO_FEW_POINTS: &str = "BREAKLINE_TOO_FEW_POINTS";

/// A stored row left out of a layer, with the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct OmittedRow {
    /// The canonical layer name (`breaklines`).
    pub layer: String,
    /// The zero-based row of the stored `Attributes` table (ras-commander's `bl_id`).
    pub source_row: u64,
    /// The stored point count of the feature.
    pub point_count: u64,
    /// The reason code ([`TOO_FEW_POINTS`]).
    pub reason: String,
}

impl OmittedRow {
    /// A one-line description, for warnings.
    pub fn describe(&self) -> String {
        format!(
            "HEC-RAS {}: stored row {} has {} point(s) and is left out ({}), as ras-commander does",
            self.layer, self.source_row, self.point_count, self.reason
        )
    }
}

/// A layer as schema and batches, under a consumer's field names, with the rows left out.
pub(crate) struct LinesLayer {
    pub name: String,
    pub schema: SchemaRef,
    pub batches: Vec<RecordBatch>,
    pub identity: (String, Option<String>),
    pub omitted: Vec<OmittedRow>,
}

impl LinesLayer {
    fn from_engine(layer: ArrowLayer, omitted: Vec<OmittedRow>) -> Self {
        Self { name: layer.name, schema: layer.schema, batches: layer.batches, identity: layer.identity, omitted }
    }
}

fn refuse(code: UnsupportedCode, m: impl Into<String>) -> Error {
    Error::unsupported(code, m)
}

/// Whether the engine refused the layer because one of its line parts has fewer than two points.
fn short_part_refusal(e: &Error) -> bool {
    e.kind == ErrorKind::Unsupported && e.message.contains("a line part has fewer than two points")
}

impl HecRasHdf {
    /// The breaklines layer under `consumer`'s field names for its layer `consumer_layer`, with
    /// degenerate breaklines left out ([module](self) documentation). Refusals and damage are the
    /// engine's errors.
    pub(crate) fn breaklines_layer(&self, consumer: &str, consumer_layer: &str, options: &LayerOptions) -> Result<LinesLayer> {
        let layer = match self.session.read_layer(LAYER, None, options) {
            Ok(layer) => without_empty_rows(layer)?,
            Err(e) if short_part_refusal(&e) => self.breaklines_from_family(options).map_err(|own| {
                // A short part the rows here do not excuse: the engine's refusal stands.
                if short_part_refusal(&own) { e } else { own }
            })?,
            Err(e) => return Err(e),
        };
        rename(layer, consumer, consumer_layer)
    }

    /// The layer built from the engine's typed family read, degenerate breaklines left out.
    fn breaklines_from_family(&self, options: &LayerOptions) -> Result<LinesLayer> {
        let reported = self.session.geometry_family(GeometryFamily::Breaklines, options.max_bytes)?;
        let data = match reported.capability().clone() {
            Capability::Supported => reported.into_value().ok_or_else(|| Error::new(ErrorKind::Internal, "supported value missing"))?,
            Capability::Unavailable { reason } => {
                return Err(Error::new(ErrorKind::NotFound, format!("{LAYER}: not stored in this file ({reason})")));
            }
            Capability::Unsupported { code, reason } => return Err(refuse(code, format!("{LAYER}: {reason}"))),
            Capability::Corrupt { reason } => return Err(Error::new(ErrorKind::Corrupt, format!("{LAYER}: {reason}"))),
            other => return Err(refuse(UnsupportedCode::NotImplemented, format!("{LAYER}: {other:?}"))),
        };
        let points = xy(&data.points.values, &data.points.shape)?;
        let rows = data.features.len();
        let table = &data.attributes.table;
        if table.shape != [rows as u64] {
            return Err(Error::new(ErrorKind::Corrupt, format!("{LAYER}: Attributes and geometry row counts differ")));
        }
        let mut keep = Vec::with_capacity(rows);
        let mut omitted = Vec::new();
        let mut geoms = BinaryBuilder::new();
        let mut wkb = Vec::new();
        for (row, f) in data.features.iter().enumerate() {
            let whole = usize_range(&f.point_range)?;
            let parts = f.parts.iter().map(usize_range).collect::<Result<Vec<_>>>()?;
            if whole.end > points.len() || parts.iter().any(|p| p.end > points.len()) {
                return Err(Error::new(ErrorKind::Corrupt, format!("{LAYER}: row {row}: point range outside Points")));
            }
            if whole.len() < 2 {
                keep.push(false);
                omitted.push(OmittedRow {
                    layer: LAYER.into(),
                    source_row: row as u64,
                    point_count: whole.len() as u64,
                    reason: TOO_FEW_POINTS.into(),
                });
                continue;
            }
            keep.push(true);
            line_wkb(row, &points, &whole, &parts, &mut wkb)?;
            geoms.append_value(&wkb);
        }
        let mut fields = vec![Field::new("source_row", DataType::Int64, false)];
        let mut columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from((0..rows as i64).collect::<Vec<_>>()))];
        let (f, c) = table_columns(table, rows)?;
        fields.extend(f);
        columns.extend(c);
        let mask = BooleanArray::from(keep);
        let mut columns =
            columns.iter().map(|c| arrow_select::filter::filter(c.as_ref(), &mask).map_err(arrow_error)).collect::<Result<Vec<_>>>()?;
        fields.push(geometry_field());
        columns.push(Arc::new(geoms.finish()));
        let schema = Arc::new(Schema::new(fields));
        let all = RecordBatch::try_new(schema.clone(), columns).map_err(arrow_error)?;
        Ok(LinesLayer {
            name: LAYER.into(),
            schema,
            batches: slices(&all, options.batch_rows)?,
            identity: ("source_row".into(), None),
            omitted,
        })
    }
}

/// The engine's layer less its rows without geometry (breaklines stored with no points).
fn without_empty_rows(layer: ArrowLayer) -> Result<LinesLayer> {
    let g = layer.schema.index_of("geometry").map_err(arrow_error)?;
    if layer.batches.iter().all(|b| b.column(g).null_count() == 0) {
        return Ok(LinesLayer::from_engine(layer, Vec::new()));
    }
    let s = layer.schema.index_of("source_row").map_err(arrow_error)?;
    let mut omitted = Vec::new();
    let mut kept = Vec::new();
    for batch in &layer.batches {
        let geometry = batch.column(g);
        let source = batch.column(s).as_primitive::<Int64Type>();
        for r in 0..batch.num_rows() {
            if geometry.is_null(r) {
                omitted.push(OmittedRow {
                    layer: LAYER.into(),
                    source_row: u64::try_from(source.value(r)).map_err(|_| Error::new(ErrorKind::Internal, "negative source_row"))?,
                    point_count: 0,
                    reason: TOO_FEW_POINTS.into(),
                });
            }
        }
        let mask = BooleanArray::from((0..batch.num_rows()).map(|r| !geometry.is_null(r)).collect::<Vec<_>>());
        let filtered = arrow_select::filter::filter_record_batch(batch, &mask).map_err(arrow_error)?;
        if filtered.num_rows() > 0 {
            kept.push(filtered);
        }
    }
    let mut out = LinesLayer::from_engine(layer, omitted);
    out.batches = kept;
    Ok(out)
}

/// The layer under `consumer`'s field names, as the engine's `ArrowLayer::with_consumer_names`.
fn rename(layer: LinesLayer, consumer: &str, consumer_layer: &str) -> Result<LinesLayer> {
    let mut seen = std::collections::HashSet::new();
    let mut fields = Vec::with_capacity(layer.schema.fields().len());
    for f in layer.schema.fields() {
        let name = consumer_field_name(consumer, consumer_layer, f.name());
        if name.is_empty() || !seen.insert(name.clone()) {
            return Err(refuse(
                UnsupportedCode::Schema,
                format!("{}: field {:?} gives the {consumer} name {name:?} twice, or an empty one", layer.name, f.name()),
            ));
        }
        fields.push(f.as_ref().clone().with_name(name));
    }
    let schema = Arc::new(Schema::new_with_metadata(fields, layer.schema.metadata().clone()));
    let batches = layer
        .batches
        .iter()
        .map(|b| RecordBatch::try_new(schema.clone(), b.columns().to_vec()).map_err(arrow_error))
        .collect::<Result<Vec<_>>>()?;
    let identity = (
        consumer_field_name(consumer, consumer_layer, &layer.identity.0),
        layer.identity.1.as_ref().map(|n| consumer_field_name(consumer, consumer_layer, n)),
    );
    Ok(LinesLayer { name: layer.name, schema, batches, identity, omitted: layer.omitted })
}

fn slices(all: &RecordBatch, batch_rows: usize) -> Result<Vec<RecordBatch>> {
    if batch_rows == 0 {
        return Err(Error::new(ErrorKind::InvalidRequest, "batch_rows must be positive"));
    }
    let mut out = Vec::new();
    let mut start = 0;
    while start < all.num_rows() {
        let len = batch_rows.min(all.num_rows() - start);
        out.push(all.slice(start, len));
        start += len;
    }
    Ok(out)
}

fn arrow_error(e: arrow_schema::ArrowError) -> Error {
    Error::new(ErrorKind::Internal, format!("arrow: {e}"))
}

fn usize_range(r: &std::ops::Range<u64>) -> Result<std::ops::Range<usize>> {
    let c = |v: u64| usize::try_from(v).map_err(|_| refuse(UnsupportedCode::ResourceLimit, "point index"));
    Ok(c(r.start)?..c(r.end)?)
}

/// The stored `(n, 2)` points as f64 (f32 widened exactly), as the engine's layers read them.
fn xy(values: &NumericValues, shape: &[u64]) -> Result<Vec<[f64; 2]>> {
    if shape.len() != 2 || shape[1] != 2 {
        return Err(refuse(UnsupportedCode::Schema, format!("{LAYER}: points stored with shape {shape:?}; only (n, 2) is read")));
    }
    let flat: Vec<f64> = match values {
        NumericValues::F64(v) => v.clone(),
        NumericValues::F32(v) => v.iter().map(|&x| f64::from(x)).collect(),
        _ => return Err(refuse(UnsupportedCode::Encoding, format!("{LAYER}: coordinates are not f32/f64"))),
    };
    Ok(flat.as_chunks::<2>().0.to_vec())
}

/// The engine's line WKB for a feature of at least two points: one part a LineString, several a
/// MultiLineString; a part with fewer than two points, no parts, or a single part not spanning
/// the feature's points refuse the layer with the engine's message.
fn line_wkb(
    row: usize,
    points: &[[f64; 2]],
    whole: &std::ops::Range<usize>,
    parts: &[std::ops::Range<usize>],
    out: &mut Vec<u8>,
) -> Result<()> {
    out.clear();
    let bad = |m: &str| refuse(UnsupportedCode::NotImplemented, format!("{LAYER}: row {row}: {m}"));
    if parts.is_empty() {
        return Err(bad("points without parts"));
    }
    if parts.iter().any(|p| p.len() < 2) {
        return Err(bad("a line part has fewer than two points"));
    }
    if let [part] = parts {
        if part != whole {
            return Err(bad("the single part does not span the feature's points"));
        }
        return wkb_line(out, &points[part.clone()]);
    }
    out.push(1);
    out.extend_from_slice(&5u32.to_le_bytes());
    out.extend_from_slice(&u32::try_from(parts.len()).map_err(|_| refuse(UnsupportedCode::ResourceLimit, "parts"))?.to_le_bytes());
    let mut one = Vec::new();
    for p in parts {
        one.clear();
        wkb_line(&mut one, &points[p.clone()])?;
        out.extend_from_slice(&one);
    }
    Ok(())
}

/// Arrow fields and arrays of the stored compound table, under the stored names, as the engine's
/// feature layers make them: text as Utf8, signed integers as Int64, unsigned as UInt64, floats
/// in their stored width (NaN as null). Bitfields, unproven or damaged columns are refused.
fn table_columns(table: &CompoundTable, rows: usize) -> Result<(Vec<Field>, Vec<ArrayRef>)> {
    let mut fields = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    for c in &table.columns {
        if matches!(c.name.as_str(), "source_row" | "geometry") || c.name.is_empty() {
            return Err(refuse(UnsupportedCode::Schema, format!("{LAYER}: stored column name {:?} is reserved", c.name)));
        }
        let (field, array): (Field, ArrayRef) = match &c.value {
            MetadataValue::Strings(v) if v.len() == rows => {
                let mut b = StringBuilder::new();
                v.iter().for_each(|s| b.append_value(s));
                (Field::new(&c.name, DataType::Utf8, false), Arc::new(b.finish()))
            }
            MetadataValue::Signed(v) if v.len() == rows => {
                (Field::new(&c.name, DataType::Int64, false), Arc::new(Int64Array::from(v.clone())))
            }
            MetadataValue::Unsigned(v) if v.len() == rows => {
                (Field::new(&c.name, DataType::UInt64, false), Arc::new(UInt64Array::from(v.clone())))
            }
            MetadataValue::Float(v) if v.len() == rows => {
                if matches!(c.attribute_type, AttributeType::Float { bytes: 4, .. }) {
                    let a = Float32Array::from(v.iter().map(|&x| (!x.is_nan()).then_some(x as f32)).collect::<Vec<_>>());
                    (Field::new(&c.name, DataType::Float32, true), Arc::new(a))
                } else {
                    let a = Float64Array::from(v.iter().map(|&x| (!x.is_nan()).then_some(x)).collect::<Vec<_>>());
                    (Field::new(&c.name, DataType::Float64, true), Arc::new(a))
                }
            }
            MetadataValue::Unsupported { code, reason } => return Err(refuse(*code, format!("{LAYER}/{}: {reason}", c.name))),
            MetadataValue::Corrupt { reason } => {
                return Err(Error::new(ErrorKind::Corrupt, format!("{LAYER}/{}: {reason}", c.name)));
            }
            MetadataValue::Bitfield(_) => {
                return Err(refuse(UnsupportedCode::Encoding, format!("{LAYER}/{}: bitfield columns are not read as Arrow", c.name)));
            }
            _ => {
                return Err(Error::new(ErrorKind::Corrupt, format!("{LAYER}/{}: column length differs from the table", c.name)));
            }
        };
        fields.push(field);
        arrays.push(array);
    }
    Ok((fields, arrays))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ras_commander_fast_hdf::arrow_layers::LayerOptions;

    fn open(name: &str) -> HecRasHdf {
        HecRasHdf::open(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/hecras").join(name)).unwrap()
    }

    fn source_rows(l: &LinesLayer) -> Vec<i64> {
        l.batches.iter().flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec()).collect()
    }

    /// On a file the engine reads, the family path gives the engine's batches exactly.
    #[test]
    fn the_family_path_equals_the_engine_layer() {
        for batch_rows in [1, 65_536] {
            let options = LayerOptions::default().with_batch_rows(batch_rows);
            let hdf = open("breaklines_good.h5");
            let engine = hdf.session.read_layer(LAYER, None, &options).unwrap();
            let own = hdf.breaklines_from_family(&options).unwrap();
            assert_eq!(own.schema, engine.schema);
            assert_eq!(own.batches, engine.batches);
            assert_eq!(own.identity, engine.identity);
            assert!(own.omitted.is_empty());
        }
    }

    #[test]
    fn one_point_and_empty_breaklines_are_left_out_and_reported() {
        let hdf = open("breaklines_point.h5");
        let options = LayerOptions::default();
        assert!(short_part_refusal(&hdf.session.read_layer(LAYER, None, &options).unwrap_err()));
        let l = hdf.breaklines_layer("terrio", LAYER, &options).unwrap();
        assert_eq!(source_rows(&l), vec![0, 1, 4]);
        let omitted: Vec<(u64, u64)> = l.omitted.iter().map(|o| (o.source_row, o.point_count)).collect();
        assert_eq!(omitted, vec![(2, 1), (3, 0)]);
        assert!(l.omitted.iter().all(|o| o.reason == TOO_FEW_POINTS && o.layer == LAYER));
        // The kept rows equal the engine's reading of the same rows in a file without the
        // degenerate ones.
        let good = hdf_rows(&open("breaklines_good.h5"));
        let kept = hdf_rows_of(&l);
        assert_eq!(&kept[..2], &good[..]);
    }

    #[test]
    fn an_empty_breakline_is_left_out_of_the_engine_layer() {
        let hdf = open("breaklines_empty.h5");
        let l = hdf.breaklines_layer("ras2cng", LAYER, &LayerOptions::default()).unwrap();
        assert_eq!(l.identity.0, "bl_id");
        assert_eq!(source_rows(&l), vec![0, 1]);
        assert_eq!(l.omitted.len(), 1);
        assert_eq!((l.omitted[0].source_row, l.omitted[0].point_count), (2, 0));
        assert_eq!(hdf_rows_of(&l), hdf_rows(&open("breaklines_good.h5")));
    }

    #[test]
    fn a_short_part_in_a_longer_breakline_still_refuses_the_layer() {
        let hdf = open("breaklines_short_part.h5");
        let e = hdf.breaklines_layer("terrio", LAYER, &LayerOptions::default()).err().unwrap();
        assert!(short_part_refusal(&e), "{e:?}");
    }

    /// Each row's columns as debug text, for comparing rows across files.
    fn hdf_rows(hdf: &HecRasHdf) -> Vec<String> {
        hdf_rows_of(&hdf.breaklines_layer("ras2cng", LAYER, &LayerOptions::default()).unwrap())
    }

    fn hdf_rows_of(l: &LinesLayer) -> Vec<String> {
        let mut out = Vec::new();
        for b in &l.batches {
            for r in 0..b.num_rows() {
                out.push(b.columns().iter().map(|c| format!("{:?}", c.slice(r, 1))).collect::<Vec<_>>().join("|"));
            }
        }
        out
    }
}
