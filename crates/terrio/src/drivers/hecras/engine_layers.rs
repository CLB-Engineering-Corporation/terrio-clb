//! The engine's Arrow layers (`ras_commander_fast_hdf::arrow_layers`) as terrio features.
//!
//! The engine's layer catalog is the one source of HEC-RAS layer rules (which rows, which
//! order, which geometry, which refusals) and of field names: [`HecRasHdf::engine_layer`] asks
//! the catalog for terrio's names ([`ArrowLayer::with_consumer_names`] with consumer `terrio`:
//! the stored names lower-snake-cased, plus the catalog's explicit aliases such as `is_ghost`).
//! This module only turns its Arrow values into terrio values:
//!
//! - `Utf8` is text, `Int64` an integer, `UInt64` an integer (a value above `i64::MAX` is
//!   refused), `Float32` and `Float64` floats (`f32` widened exactly), `Boolean` a boolean;
//!   nulls stay nulls. Any other type is refused.
//! - The WKB `geometry` is decoded exactly: rings lose their closing vertex (terrio rings store
//!   none), and a null geometry stays a feature without geometry.
//! - The catalog's identity fields are non-nullable.

use super::{HecRasHdf, error};
use crate::vector::{Coord, Ring};
use crate::{FieldDef, FieldType, FieldValue, Geometry};
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, Int64Type, UInt64Type};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use ras_commander_fast_hdf::arrow_layers::{ArrowLayer, LayerOptions};

/// The consumer name of terrio in the engine's catalog.
pub(crate) const CONSUMER: &str = "terrio";

/// A converted engine layer: its fields (the geometry left out) and, per row, the field values
/// and the geometry.
pub(crate) struct EngineRows {
    pub fields: Vec<FieldDef>,
    pub rows: Vec<(Vec<FieldValue>, Option<Geometry>)>,
}

impl EngineRows {
    /// The index of a field.
    pub fn index(&self, name: &str) -> crate::Result<usize> {
        self.fields
            .iter()
            .position(|f| f.name == name)
            .ok_or_else(|| crate::Error::InvalidData(format!("HEC-RAS engine layer without the field {name:?}")))
    }
}

impl HecRasHdf {
    /// A catalog layer (canonical name) through the engine's Arrow API, with terrio's field names.
    pub(crate) fn engine_layer(&self, canonical: &str, options: &LayerOptions) -> crate::Result<ArrowLayer> {
        self.session
            .read_layer(canonical, None, options)
            .and_then(|layer| layer.with_consumer_names(CONSUMER, canonical))
            .map_err(error::from_engine)
    }
}

/// The values and geometries of an engine layer.
pub(crate) fn rows(layer: &ArrowLayer) -> crate::Result<EngineRows> {
    let identity: Vec<&str> = std::iter::once(layer.identity.0.as_str()).chain(layer.identity.1.as_deref()).collect();
    let geometry = layer.schema.index_of("geometry").ok();
    let mut fields = Vec::new();
    let mut columns = Vec::new();
    for (i, f) in layer.schema.fields().iter().enumerate() {
        if Some(i) == geometry {
            continue;
        }
        let kind = match f.data_type() {
            DataType::Utf8 => FieldType::Text,
            DataType::Int64 | DataType::UInt64 => FieldType::Integer,
            DataType::Float32 | DataType::Float64 => FieldType::Float,
            DataType::Boolean => FieldType::Boolean,
            other => {
                return Err(crate::Error::Unsupported(format!("HEC-RAS {} field {:?}: unmapped Arrow type {other}", layer.name, f.name())));
            }
        };
        let mut def = FieldDef::new(f.name().clone(), kind);
        def.nullable = !identity.contains(&f.name().as_str());
        fields.push(def);
        columns.push(i);
    }
    let mut rows = Vec::with_capacity(layer.num_rows());
    for batch in &layer.batches {
        let arrays: Vec<&ArrayRef> = columns.iter().map(|&i| batch.column(i)).collect();
        let wkb = match geometry {
            Some(g) => Some(
                batch
                    .column(g)
                    .as_binary_opt::<i32>()
                    .ok_or_else(|| crate::Error::InvalidData(format!("HEC-RAS {}: geometry is not WKB Binary", layer.name)))?,
            ),
            None => None,
        };
        for row in 0..batch.num_rows() {
            let values = arrays.iter().map(|a| value(a.as_ref(), row, &layer.name)).collect::<crate::Result<Vec<_>>>()?;
            let geometry =
                match wkb {
                    Some(w) if !w.is_null(row) => Some(decode_wkb(w.value(row)).ok_or_else(|| {
                        crate::Error::InvalidData(format!("HEC-RAS {} row {row}: WKB the driver does not map", layer.name))
                    })?),
                    _ => None,
                };
            rows.push((values, geometry));
        }
    }
    Ok(EngineRows { fields, rows })
}

fn value(a: &dyn Array, row: usize, layer: &str) -> crate::Result<FieldValue> {
    if a.is_null(row) {
        return Ok(FieldValue::Null);
    }
    Ok(match a.data_type() {
        DataType::Utf8 => FieldValue::Text(a.as_string::<i32>().value(row).to_string()),
        DataType::Int64 => FieldValue::Integer(a.as_primitive::<Int64Type>().value(row)),
        DataType::UInt64 => FieldValue::Integer(
            i64::try_from(a.as_primitive::<UInt64Type>().value(row))
                .map_err(|_| crate::Error::Unsupported(format!("HEC-RAS {layer}: value above i64")))?,
        ),
        DataType::Float32 => FieldValue::Float(f64::from(a.as_primitive::<Float32Type>().value(row))),
        DataType::Float64 => FieldValue::Float(a.as_primitive::<Float64Type>().value(row)),
        DataType::Boolean => FieldValue::Boolean(a.as_boolean().value(row)),
        other => return Err(crate::Error::Unsupported(format!("HEC-RAS {layer}: unmapped Arrow type {other}"))),
    })
}

/// Little-endian 2D WKB as the engine writes it: Point, LineString, one-ring Polygon or
/// MultiLineString. Anything else (or trailing bytes) is `None`.
fn decode_wkb(b: &[u8]) -> Option<Geometry> {
    let mut r = Reader { b, pos: 0 };
    let g = r.geometry()?;
    (r.pos == b.len()).then_some(g)
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn u32(&mut self) -> Option<u32> {
        let v = u32::from_le_bytes(self.b.get(self.pos..self.pos + 4)?.try_into().ok()?);
        self.pos += 4;
        Some(v)
    }
    fn coord(&mut self) -> Option<Coord> {
        let x = f64::from_le_bytes(self.b.get(self.pos..self.pos + 8)?.try_into().ok()?);
        let y = f64::from_le_bytes(self.b.get(self.pos + 8..self.pos + 16)?.try_into().ok()?);
        self.pos += 16;
        Some(Coord::xy(x, y))
    }
    fn coords(&mut self) -> Option<Vec<Coord>> {
        let n = self.u32()? as usize;
        // Every coordinate takes 16 bytes: refuse counts the input cannot hold before allocating.
        if n > (self.b.len() - self.pos) / 16 {
            return None;
        }
        (0..n).map(|_| self.coord()).collect()
    }
    fn header(&mut self) -> Option<u32> {
        (*self.b.get(self.pos)? == 1).then_some(())?;
        self.pos += 1;
        self.u32()
    }
    fn line(&mut self) -> Option<Vec<Coord>> {
        (self.header()? == 2).then_some(())?;
        self.coords()
    }
    fn geometry(&mut self) -> Option<Geometry> {
        match self.header()? {
            1 => Some(Geometry::Point(self.coord()?)),
            2 => Some(Geometry::LineString(self.coords()?)),
            3 => {
                if self.u32()? != 1 {
                    return None;
                }
                let mut ring = self.coords()?;
                if ring.len() < 4 || ring.first() != ring.last() {
                    return None;
                }
                ring.pop();
                Some(Geometry::Polygon { exterior: Ring::new(ring), interiors: Vec::new() })
            }
            5 => {
                let n = self.u32()? as usize;
                if n > self.b.len() - self.pos {
                    return None;
                }
                Some(Geometry::MultiLineString((0..n).map(|_| self.line()).collect::<Option<_>>()?))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wkb(kind: u32, body: &[u8]) -> Vec<u8> {
        let mut out = vec![1];
        out.extend_from_slice(&kind.to_le_bytes());
        out.extend_from_slice(body);
        out
    }

    fn pts(p: &[[f64; 2]]) -> Vec<u8> {
        let mut out = (p.len() as u32).to_le_bytes().to_vec();
        for [x, y] in p {
            out.extend_from_slice(&x.to_le_bytes());
            out.extend_from_slice(&y.to_le_bytes());
        }
        out
    }

    #[test]
    fn polygons_lose_the_closing_vertex() {
        let mut body = 1u32.to_le_bytes().to_vec();
        body.extend(pts(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]));
        match decode_wkb(&wkb(3, &body)) {
            Some(Geometry::Polygon { exterior, interiors }) => {
                assert_eq!(exterior.coords().len(), 3);
                assert!(interiors.is_empty());
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn multilines_points_and_refusals() {
        let mut body = 2u32.to_le_bytes().to_vec();
        body.extend(wkb(2, &pts(&[[0.0, 0.0], [1.0, 1.0]])));
        body.extend(wkb(2, &pts(&[[2.0, 2.0], [3.0, 3.0]])));
        assert!(matches!(decode_wkb(&wkb(5, &body)), Some(Geometry::MultiLineString(l)) if l.len() == 2));
        let mut point = 0.5f64.to_le_bytes().to_vec();
        point.extend_from_slice(&2.0f64.to_le_bytes());
        assert_eq!(decode_wkb(&wkb(1, &point)), Some(Geometry::Point(Coord::xy(0.5, 2.0))));
        // An open ring, a huge count, trailing bytes and big-endian WKB are refused.
        let mut open = 1u32.to_le_bytes().to_vec();
        open.extend(pts(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]));
        assert_eq!(decode_wkb(&wkb(3, &open)), None);
        assert_eq!(decode_wkb(&wkb(2, &u32::MAX.to_le_bytes())), None);
        let mut trailing = wkb(1, &point);
        trailing.push(0);
        assert_eq!(decode_wkb(&trailing), None);
        let mut big = wkb(1, &point);
        big[0] = 0;
        assert_eq!(decode_wkb(&big), None);
    }
}
