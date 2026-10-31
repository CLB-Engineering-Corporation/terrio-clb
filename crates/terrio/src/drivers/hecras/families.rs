//! Line, polygon and point families stored under `Geometry/`: 1D cross-section cut lines, river
//! lines, storage areas, structure centerlines, river junctions and pump stations, the 2D
//! boundary-condition lines, breaklines, refinement regions, reference lines and points, and
//! pipe-network conduits and nodes.
//!
//! Each layer is the engine's Arrow catalog layer of the same name
//! (`ras_commander_fast_hdf::arrow_layers`), which owns every rule:
//!
//! - one feature per stored feature, in stored order, with `source_row` (the row of the family's
//!   attribute table, or of its Info table when it has none) as the identity field;
//! - every attribute-table column as a field, named by the catalog for terrio (the stored name in
//!   snake case), with its stored values (text, integer, float). Bitfield columns are refused;
//! - a line with one part is a LineString, with several a MultiLineString; a polygon family's
//!   single part is a Polygon. Multi-part polygons are refused: which parts are holes is not a
//!   proven layout. A feature without drawn geometry has none.
//!
//! The typed family reads remain for listing (`family_present`) only; `declare_columns` and
//! `column_value` translate the 2D flow area table joined to `mesh_perimeters`.

use super::{HecRasHdf, engine_layers, error, field_name};
use crate::{Feature, FieldDef, FieldType, FieldValue, GeometryType, Layer};
use ras_commander_fast_hdf::arrow_layers::LayerOptions;
use ras_commander_fast_hdf::hdf::AttributeType;
use ras_commander_fast_hdf::hdf::DEFAULT_MAX_BYTES;
use ras_commander_fast_hdf::hdf::lines::GeometryFamily;
use ras_commander_fast_hdf::hdf::metadata::{CompoundColumn, MetadataValue};
use ras_commander_fast_hdf::hdf::xs_geometry::{OneDLineFamily, OneDPointData};
use ras_commander_fast_hdf::{Capability, Reported};

/// The family layers, by layer name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Family {
    OneD(OneDLineFamily, Shape),
    TwoD(GeometryFamily, Shape),
    ReferencePoints,
    /// A one-point-per-row family (`Attributes` plus `Points`).
    Points(PointFamily),
}

/// The point families the engine reads with one stored point per attribute row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PointFamily {
    Junctions,
    PumpStations,
    PipeNodes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Shape {
    Line,
    Polygon,
}

/// Layer names of the family layers, with their engine families, in listing order.
pub(crate) const FAMILY_LAYERS: &[(&str, Family)] = &[
    ("cross_sections", Family::OneD(OneDLineFamily::CrossSectionCutLines, Shape::Line)),
    ("river_centerlines", Family::OneD(OneDLineFamily::RiverCenterlines, Shape::Line)),
    ("river_bank_lines", Family::OneD(OneDLineFamily::RiverBankLines, Shape::Line)),
    ("river_edge_lines", Family::OneD(OneDLineFamily::RiverEdgeLines, Shape::Line)),
    ("river_flow_paths", Family::OneD(OneDLineFamily::RiverFlowPaths, Shape::Line)),
    ("storage_areas", Family::OneD(OneDLineFamily::StorageAreaPolygons, Shape::Polygon)),
    ("structures", Family::OneD(OneDLineFamily::StructureCenterlines, Shape::Line)),
    ("bc_lines", Family::TwoD(GeometryFamily::BoundaryConditions, Shape::Line)),
    ("breaklines", Family::TwoD(GeometryFamily::Breaklines, Shape::Line)),
    ("refinement_regions", Family::TwoD(GeometryFamily::RefinementRegions, Shape::Polygon)),
    ("reference_lines", Family::TwoD(GeometryFamily::ReferenceLines, Shape::Line)),
    ("reference_points", Family::ReferencePoints),
    ("river_junctions", Family::Points(PointFamily::Junctions)),
    ("pump_stations", Family::Points(PointFamily::PumpStations)),
    ("pipe_conduits", Family::OneD(OneDLineFamily::PipeConduits, Shape::Line)),
    ("pipe_nodes", Family::Points(PointFamily::PipeNodes)),
];

impl HecRasHdf {
    /// Whether the file stores the family: the engine reads it and reports anything but
    /// canonical absence. Unsupported or corrupt families are listed, so that reading them shows
    /// the refusal; host I/O failures propagate.
    pub(crate) fn family_present(&self, family: Family) -> crate::Result<bool> {
        use ras_commander_fast_hdf::ErrorKind;
        let s = &self.session;
        let status = match family {
            Family::OneD(f, _) => s.one_d_line_family(f, DEFAULT_MAX_BYTES).map(|r| r.is_unavailable()),
            Family::TwoD(f, _) => s.geometry_family(f, DEFAULT_MAX_BYTES).map(|r| r.is_unavailable()),
            Family::ReferencePoints => s.reference_point_geometry(DEFAULT_MAX_BYTES).map(|r| r.is_unavailable()),
            Family::Points(f) => self.point_data(f).map(|r| r.is_unavailable()),
        };
        match status {
            Ok(unavailable) => Ok(!unavailable),
            Err(e) if e.kind == ErrorKind::NotFound => Ok(false),
            Err(e) if matches!(e.kind, ErrorKind::Io | ErrorKind::Internal | ErrorKind::FileNotFound) => Err(error::from_engine(e)),
            Err(_) => Ok(true),
        }
    }

    /// Reads one family layer: the engine's catalog layer of the same name
    /// ([`super::engine_layers`]), in its stored order, with `source_row` and the stored
    /// attribute columns under terrio's names. A feature HEC-RAS stores without drawn geometry
    /// (some structures) is a feature without geometry.
    pub(crate) fn family_layer(&self, name: &str, family: Family) -> crate::Result<Layer> {
        let engine = self.engine_layer(name, &LayerOptions::default())?;
        let rows = engine_layers::rows(&engine)?;
        let mut layer = Layer::new(name);
        if let Some(wkt) = self.crs()? {
            layer = layer.with_crs_wkt(wkt);
        }
        layer.geom_type = Some(family.geometry_type());
        for field in rows.fields {
            layer.add_field(field);
        }
        for (fid, (attributes, geometry)) in rows.rows.into_iter().enumerate() {
            layer.push(Feature { fid: fid as u64, geometry, attributes });
        }
        Ok(layer)
    }

    fn point_data(&self, family: PointFamily) -> ras_commander_fast_hdf::Result<Reported<OneDPointData>> {
        match family {
            PointFamily::Junctions => self.session.river_junctions(DEFAULT_MAX_BYTES),
            PointFamily::PumpStations => self.session.pump_stations(DEFAULT_MAX_BYTES),
            PointFamily::PipeNodes => self.session.pipe_nodes(DEFAULT_MAX_BYTES),
        }
    }
}

impl Family {
    /// The declared geometry type of the layer (a line layer can hold MultiLineStrings).
    fn geometry_type(self) -> GeometryType {
        match self {
            Family::OneD(_, Shape::Polygon) | Family::TwoD(_, Shape::Polygon) => GeometryType::Polygon,
            Family::OneD(_, Shape::Line) | Family::TwoD(_, Shape::Line) => GeometryType::LineString,
            Family::ReferencePoints | Family::Points(_) => GeometryType::Point,
        }
    }
}

fn column_capability(value: &MetadataValue) -> Capability {
    value.capability()
}

/// Adds one field per stored attribute-table column, named after the column (snake case), with
/// its stored value type (text, integer, float; bitfields as bytes). Unsupported or corrupt
/// columns, and names that collide, are errors.
pub(crate) fn declare_columns(layer: &mut Layer, columns: &[CompoundColumn]) -> crate::Result<()> {
    for column in columns {
        let kind = match (&column.value, &column.attribute_type) {
            (MetadataValue::Strings(_), _) => FieldType::Text,
            (MetadataValue::Signed(_) | MetadataValue::Unsigned(_), _) => FieldType::Integer,
            (MetadataValue::Float(_), _) => FieldType::Float,
            (MetadataValue::Bitfield(_), AttributeType::Bitfield { .. }) => FieldType::Blob,
            (MetadataValue::Unsupported { .. } | MetadataValue::Corrupt { .. }, _) => {
                error::require(&column_capability(&column.value), &format!("{} column {:?}", layer.name, column.name))?;
                unreachable!("require refuses unsupported and corrupt columns")
            }
            _ => return Err(crate::Error::Unsupported(format!("{} column {:?}: unmapped value type", layer.name, column.name))),
        };
        let name = field_name(&column.name);
        if name.is_empty() || layer.schema.fields().iter().any(|f| f.name == name) {
            return Err(crate::Error::Unsupported(format!(
                "{} column {:?} gives an empty or repeated field name {name:?}",
                layer.name, column.name
            )));
        }
        layer.add_field(FieldDef::new(name, kind));
    }
    Ok(())
}

/// The stored value of `column` at `row`, as [`declare_columns`] typed it.
pub(crate) fn column_value(column: &CompoundColumn, row: usize, layer: &str) -> crate::Result<FieldValue> {
    let missing = || crate::Error::InvalidData(format!("{layer} column {:?}: no row {row}", column.name));
    Ok(match (&column.value, &column.attribute_type) {
        (MetadataValue::Strings(v), _) => FieldValue::Text(v.get(row).ok_or_else(missing)?.clone()),
        (MetadataValue::Signed(v), _) => FieldValue::Integer(*v.get(row).ok_or_else(missing)?),
        (MetadataValue::Unsigned(v), _) => FieldValue::Integer(
            i64::try_from(*v.get(row).ok_or_else(missing)?)
                .map_err(|_| crate::Error::Unsupported(format!("{layer} column {:?}: value above i64", column.name)))?,
        ),
        (MetadataValue::Float(v), _) => FieldValue::Float(*v.get(row).ok_or_else(missing)?),
        (MetadataValue::Bitfield(v), AttributeType::Bitfield { size, .. }) => {
            let size = *size;
            FieldValue::Blob(v.get(row * size..(row + 1) * size).ok_or_else(missing)?.to_vec())
        }
        _ => return Err(crate::Error::Unsupported(format!("{layer} column {:?}: unmapped value type", column.name))),
    })
}
