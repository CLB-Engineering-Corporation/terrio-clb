//! HEC-RAS geometry and plan HDF files straight to archive GeoParquet: the engine's Arrow
//! layers, named for ras2cng's archive, written with [`crate::geoparquet`]'s archive profile.
//!
//! [`HecRasHdf::archive_layers`] returns the layers as Arrow record batches, so that a caller
//! (ras2cng) can add its own (text geometry layers, values the engine refuses) before
//! [`write_archive`] writes them as one file. Nothing passes through terrio's [`crate::Layer`]
//! model or a GeoDataFrame.
//!
//! - **Layers** are the engine's catalog layers (`ras_commander_fast_hdf::arrow_layers`), named
//!   by the catalog's `ras2cng` aliases (`mesh_perimeters` is `mesh_areas`, `river_centerlines`
//!   is `centerlines`, ...), with the catalog's `ras2cng` field names (`bc_line_id`, `bl_id`,
//!   ...; every other field keeps its stored name). Geometry layers come in ras2cng's order
//!   ([`GEOMETRY_LAYERS`]), then summary results, one layer per stored 2D summary variable in
//!   name order (ras2cng's), named by its slug.
//! - **Summary results** carry the element geometry, read once per file and joined by identity:
//!   the cell polygon (`mesh_cells`) of a cell variable, the face line (`mesh_faces`) of a face
//!   variable; null where the element has no polygon.
//! - **Lossless** where ras2cng's Python path is not, by default ([`ArchiveOptions`]):
//!   undrawn features (structures without a centerline) are rows with a null geometry; a timed
//!   summary keeps the stored offset in days next to the 100 ms timestamp
//!   (`<slug>_time_days`); `mesh_cell_points` keeps the ghost cells, flagged by
//!   `is_ghost_cell`, because face cell indexes refer to them.
//! - **Refusals** do not abort the file: a layer or variable the engine refuses (an unproven
//!   layout, `Cell Last Iteration`, which stores no element `Location`) is listed in
//!   [`HecRasArchive::skipped`] with the engine's reason, so that the caller can fill it from
//!   another reader. Absent families are not listed. Damaged data and I/O errors fail.

use super::{HecRasHdf, error};
use crate::Result;
use crate::geoparquet::{
    ArrowLayer as ParquetLayer, GeoParquetWriteOptions, GeoParquetWriteReport, MemoryLayer, write_arrow_layers_to_path,
};
use arrow_array::builder::BinaryBuilder;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{Field, Schema, SchemaRef};
use ras_commander_fast_hdf::ErrorKind;
use ras_commander_fast_hdf::arrow_layers::{ArrowLayer as EngineLayer, LayerOptions, geometry_field, lookup};
use std::collections::HashMap;
use std::sync::Arc;

/// The consumer whose names the archive uses in the engine's catalog.
pub const ARCHIVE_CONSUMER: &str = "ras2cng";

/// The geometry layers of an archive, by archive (ras2cng) name, in ras2cng's order, then the
/// catalog layers ras2cng's Python path does not write (`mesh_cell_points`, `river_flow_paths`,
/// `river_junctions`).
pub const GEOMETRY_LAYERS: &[&str] = &[
    "mesh_cells",
    "mesh_faces",
    "mesh_areas",
    "cross_sections",
    "centerlines",
    "edge_lines",
    "bank_lines",
    "bc_lines",
    "breaklines",
    "refinement_regions",
    "reference_lines",
    "reference_points",
    "structures",
    "storage_areas",
    "pipe_conduits",
    "pipe_nodes",
    "pump_stations",
    "mannings_n_regions",
    "mesh_cell_points",
    "river_flow_paths",
    "river_junctions",
];

/// What [`HecRasHdf::archive_layers`] reads.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ArchiveOptions {
    /// Read the geometry layers.
    pub geometry: bool,
    /// Read the 2D summary results.
    pub results: bool,
    /// Only these geometry layers (archive or canonical names), in [`GEOMETRY_LAYERS`] order.
    /// `None` reads every layer the file stores.
    pub layers: Option<Vec<String>>,
    /// Only these summary variables (exact stored names or their slugs). `None` reads every
    /// stored variable.
    pub variables: Option<Vec<String>>,
    /// Join the cell polygon or face line to each summary row.
    pub summary_geometry: bool,
    /// Keep ghost cells in `mesh_cell_points`, with `is_ghost_cell`.
    pub ghost_cell_points: bool,
    /// Add `<slug>_time_days`, the stored time offset, to timed summaries.
    pub time_days: bool,
    /// Rows per record batch.
    pub batch_rows: usize,
}

impl Default for ArchiveOptions {
    fn default() -> Self {
        Self {
            geometry: true,
            results: true,
            layers: None,
            variables: None,
            summary_geometry: true,
            ghost_cell_points: true,
            time_days: true,
            batch_rows: 65_536,
        }
    }
}

impl ArchiveOptions {
    /// Options with or without the geometry layers.
    pub fn with_geometry(mut self, on: bool) -> Self {
        self.geometry = on;
        self
    }
    /// Options with or without the summary results.
    pub fn with_results(mut self, on: bool) -> Self {
        self.results = on;
        self
    }
    /// Options reading only these geometry layers.
    pub fn with_layers<S: Into<String>>(mut self, layers: impl IntoIterator<Item = S>) -> Self {
        self.layers = Some(layers.into_iter().map(Into::into).collect());
        self
    }
    /// Options reading only these summary variables.
    pub fn with_variables<S: Into<String>>(mut self, variables: impl IntoIterator<Item = S>) -> Self {
        self.variables = Some(variables.into_iter().map(Into::into).collect());
        self
    }
    /// Options with `rows` rows per record batch.
    pub fn with_batch_rows(mut self, rows: usize) -> Self {
        self.batch_rows = rows;
        self
    }
}

/// What kind of archive layer.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArchiveLayerKind {
    /// A geometry layer: its canonical catalog name.
    Geometry {
        /// The canonical catalog name (`mesh_perimeters` for `mesh_areas`).
        canonical: String,
    },
    /// A 2D summary variable: its stored name.
    Summary {
        /// The stored variable name (`Maximum Water Surface`).
        variable: String,
    },
}

/// One archive layer: its archive name, Arrow schema and batches, and identity columns.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ArchiveLayer {
    /// The archive (ras2cng) name, written in the `layer` column.
    pub name: String,
    /// Geometry or summary.
    pub kind: ArchiveLayerKind,
    /// The schema: the engine's columns under ras2cng's names, then `geometry` (WKB).
    pub schema: SchemaRef,
    /// The record batches, at most `batch_rows` rows each.
    pub batches: Vec<RecordBatch>,
    /// The columns identifying each row (never null): `mesh_name` and `cell_id`/`face_id` for
    /// mesh and summary layers, the renamed `source_row` for feature layers.
    pub identity: Vec<String>,
}

impl ArchiveLayer {
    /// The number of rows.
    pub fn num_rows(&self) -> usize {
        self.batches.iter().map(RecordBatch::num_rows).sum()
    }
}

/// A layer or variable the engine refused, left out of the archive.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ArchiveSkip {
    /// The archive name the layer would have had.
    pub layer: String,
    /// The stored summary variable, for a summary.
    pub variable: Option<String>,
    /// The engine's error kind (`unsupported`, `invalid_request`, ...).
    pub kind: String,
    /// The engine's unsupported code (`not_implemented`, `schema`, ...), when it gave one.
    pub code: Option<String>,
    /// The engine's message.
    pub reason: String,
}

/// The archive layers of one HEC-RAS HDF file.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct HecRasArchive {
    /// The layers, geometry first, then summaries.
    pub layers: Vec<ArchiveLayer>,
    /// What the engine refused, in reading order.
    pub skipped: Vec<ArchiveSkip>,
    /// The stored `Projection` text (WKT), when the file stores one.
    pub crs_wkt: Option<String>,
}

impl HecRasArchive {
    /// The layers as input of [`crate::geoparquet::write_arrow_layers`], with their identities.
    pub fn into_parquet_layers(self) -> Vec<MemoryLayer> {
        self.layers
            .into_iter()
            .map(|l| ParquetLayer::new(Some(l.name), l.schema, l.batches.into_iter().map(Ok).collect()).with_identity(l.identity))
            .collect()
    }
}

/// The archive name of a canonical catalog layer: its first `ras2cng` alias, or itself.
pub fn archive_name(canonical: &str) -> &str {
    lookup(canonical).and_then(|d| d.aliases.iter().find(|(c, _)| *c == ARCHIVE_CONSUMER)).map_or(canonical, |(_, a)| a)
}

/// The canonical catalog name of an archive (or canonical) layer name.
fn canonical_name(name: &str) -> Option<&'static str> {
    lookup(name).map(|d| d.name)
}

/// Whether an engine error is a refusal the caller can fill from another reader (and so a
/// skipped layer), rather than damage or an I/O failure.
fn refusal(e: &ras_commander_fast_hdf::Error) -> bool {
    matches!(e.kind, ErrorKind::Unsupported | ErrorKind::InvalidRequest)
}

fn skip(layer: &str, variable: Option<&str>, e: &ras_commander_fast_hdf::Error) -> ArchiveSkip {
    ArchiveSkip {
        layer: layer.to_string(),
        variable: variable.map(str::to_string),
        kind: e.kind.as_str().to_string(),
        code: e.code.map(|c| c.as_str().to_string()),
        reason: e.message.clone(),
    }
}

impl HecRasHdf {
    /// The archive layers of the file ([module](self) documentation).
    pub fn archive_layers(&self, options: &ArchiveOptions) -> Result<HecRasArchive> {
        if options.batch_rows == 0 {
            return Err(crate::Error::InvalidArgument("batch_rows must be positive".into()));
        }
        let base = LayerOptions::default().with_batch_rows(options.batch_rows);
        let mut layers = Vec::new();
        let mut skipped = Vec::new();
        if options.geometry {
            let wanted: Option<Vec<&str>> = match &options.layers {
                None => None,
                Some(names) => Some(
                    names
                        .iter()
                        .map(|n| {
                            canonical_name(n)
                                .filter(|c| GEOMETRY_LAYERS.contains(&archive_name(c)))
                                .ok_or_else(|| crate::Error::InvalidArgument(format!("no HEC-RAS archive layer named {n:?}")))
                        })
                        .collect::<Result<_>>()?,
                ),
            };
            for name in GEOMETRY_LAYERS {
                let canonical = canonical_name(name).expect("archive layers are catalog layers");
                if wanted.as_ref().is_some_and(|w| !w.contains(&canonical)) {
                    continue;
                }
                let layer_options = base.with_ghost_cells(options.ghost_cell_points && canonical == "mesh_cell_points");
                match self.session.read_layer(canonical, None, &layer_options) {
                    Ok(layer) if layer.num_rows() == 0 => {}
                    Ok(layer) => layers.push(archive_layer(layer, name, ArchiveLayerKind::Geometry { canonical: canonical.into() })?),
                    // An absent family, or a geometry file without 2D flow areas.
                    Err(e) if e.kind == ErrorKind::NotFound => {}
                    Err(e) if refusal(&e) => skipped.push(skip(name, None, &e)),
                    Err(e) => return Err(error::from_engine(e)),
                }
            }
        }
        if options.results {
            self.archive_summaries(options, base, &mut layers, &mut skipped)?;
        }
        Ok(HecRasArchive { layers, skipped, crs_wkt: self.crs()? })
    }

    /// The stored 2D summary variables over every mesh, in name order; none for a file without
    /// 2D flow areas or summary output.
    pub fn summary_variables(&self) -> Result<Vec<String>> {
        let meshes = match self.session.mesh_area_names(ras_commander_fast_hdf::hdf::DEFAULT_MAX_BYTES) {
            Ok(m) => m,
            Err(e) if e.kind == ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(error::from_engine(e)),
        };
        let mut names = std::collections::BTreeSet::new();
        for mesh in meshes {
            match self.session.mesh_summary_inventory(&mesh, ras_commander_fast_hdf::hdf::DEFAULT_MAX_BYTES) {
                Ok(inventory) => names.extend(inventory.variables.into_iter().map(|v| v.name)),
                Err(e) if e.kind == ErrorKind::NotFound => {}
                Err(e) => return Err(error::from_engine(e)),
            }
        }
        Ok(names.into_iter().collect())
    }

    fn archive_summaries(
        &self,
        options: &ArchiveOptions,
        base: LayerOptions,
        layers: &mut Vec<ArchiveLayer>,
        skipped: &mut Vec<ArchiveSkip>,
    ) -> Result<()> {
        use ras_commander_fast_hdf::arrow_layers::variable_slug;
        let slug = |v: &str| variable_slug(v).unwrap_or_else(|_| v.to_string());
        let variables: Vec<String> = self
            .summary_variables()?
            .into_iter()
            .filter(|v| options.variables.as_ref().is_none_or(|w| w.iter().any(|x| x == v || x.eq_ignore_ascii_case(v) || *x == slug(v))))
            .collect();
        if variables.is_empty() {
            return Ok(());
        }
        let summary_options = base.with_summary_time_days(options.time_days);
        // The element geometry, read once per element kind and joined by identity.
        let mut geometry: HashMap<bool, Option<GeometryIndex>> = HashMap::new();
        for variable in variables {
            let name = slug(&variable);
            let layer = match self.session.read_layer("mesh_summary", Some(&variable), &summary_options) {
                Ok(layer) => layer,
                // A stored variable is never dropped silently: absent metadata it needs (such as
                // the plan's simulation start of a timed summary) is listed like a refusal.
                Err(e) if refusal(&e) || e.kind == ErrorKind::NotFound => {
                    skipped.push(skip(&name, Some(&variable), &e));
                    continue;
                }
                Err(e) => return Err(error::from_engine(e)),
            };
            let face = layer.identity.1.as_deref() == Some("face_id");
            let mut layer = layer;
            if options.summary_geometry {
                let index = match geometry.entry(face) {
                    std::collections::hash_map::Entry::Occupied(o) => o.into_mut(),
                    std::collections::hash_map::Entry::Vacant(v) => {
                        let element = if face { "mesh_faces" } else { "mesh_cells" };
                        let whole = base.with_batch_rows(usize::MAX);
                        v.insert(match self.session.read_layer(element, None, &whole) {
                            Ok(l) => Some(GeometryIndex::new(&l)?),
                            Err(e) if refusal(&e) => {
                                skipped.push(skip(element, None, &e));
                                None
                            }
                            Err(e) => return Err(error::from_engine(e)),
                        })
                    }
                };
                match index {
                    Some(index) => layer = index.join(layer)?,
                    // The element geometry was refused: the variable is skipped, not written
                    // without geometry.
                    None => {
                        skipped.push(ArchiveSkip {
                            layer: name.clone(),
                            variable: Some(variable.clone()),
                            kind: "unsupported".into(),
                            code: None,
                            reason: format!("the {} geometry was refused", if face { "face" } else { "cell" }),
                        });
                        continue;
                    }
                }
            }
            if layer.num_rows() > 0 {
                layers.push(archive_layer(layer, &name, ArchiveLayerKind::Summary { variable })?);
            }
        }
        Ok(())
    }
}

/// An engine layer under its archive name and the catalog's `ras2cng` field names.
fn archive_layer(layer: EngineLayer, name: &str, kind: ArchiveLayerKind) -> Result<ArchiveLayer> {
    let alias_layer = match kind {
        ArchiveLayerKind::Summary { .. } => "mesh_summary",
        ArchiveLayerKind::Geometry { .. } => name,
    };
    let layer = layer.with_consumer_names(ARCHIVE_CONSUMER, alias_layer).map_err(error::from_engine)?;
    let identity = std::iter::once(layer.identity.0.clone()).chain(layer.identity.1.clone()).collect();
    Ok(ArchiveLayer { name: name.to_string(), kind, schema: layer.schema, batches: layer.batches, identity })
}

/// The element geometry of one file by `(mesh_name, id)`: per mesh, the row of each element id.
struct GeometryIndex {
    wkb: ArrayRef,
    rows: HashMap<String, Vec<Option<usize>>>,
}

impl GeometryIndex {
    fn new(layer: &EngineLayer) -> Result<Self> {
        let batch = arrow_select_concat(layer)?;
        let names = batch.column(0).as_string::<i32>();
        let ids = batch.column(1).as_primitive::<Int64Type>();
        let mut rows: HashMap<String, Vec<Option<usize>>> = HashMap::new();
        for r in 0..batch.num_rows() {
            let id = usize::try_from(ids.value(r)).map_err(|_| crate::Error::InvalidData("HEC-RAS: negative element id".into()))?;
            let mesh = match rows.get_mut(names.value(r)) {
                Some(m) => m,
                None => rows.entry(names.value(r).to_string()).or_default(),
            };
            if mesh.len() <= id {
                mesh.resize(id + 1, None);
            }
            mesh[id] = Some(r);
        }
        Ok(Self { wkb: batch.column(2).clone(), rows })
    }

    /// `layer` with a last `geometry` field: each row's element geometry, null where it has none.
    fn join(&self, layer: EngineLayer) -> Result<EngineLayer> {
        let wkb = self.wkb.as_binary::<i32>();
        let mut fields: Vec<Field> = layer.schema.fields().iter().map(|f| f.as_ref().clone()).collect();
        fields.push(geometry_field());
        let schema = Arc::new(Schema::new_with_metadata(fields, layer.schema.metadata().clone()));
        let mut batches = Vec::with_capacity(layer.batches.len());
        for batch in &layer.batches {
            let names = batch.column(0).as_string::<i32>();
            let ids = batch.column(1).as_primitive::<Int64Type>();
            let mut out = BinaryBuilder::with_capacity(batch.num_rows(), 0);
            for r in 0..batch.num_rows() {
                let row = self
                    .rows
                    .get(names.value(r))
                    .and_then(|m| usize::try_from(ids.value(r)).ok().and_then(|id| m.get(id).copied().flatten()));
                match row {
                    Some(i) => out.append_value(wkb.value(i)),
                    None => out.append_null(),
                }
            }
            let mut columns = batch.columns().to_vec();
            columns.push(Arc::new(out.finish()));
            batches.push(RecordBatch::try_new(schema.clone(), columns).map_err(|e| crate::Error::InvalidData(format!("arrow: {e}")))?);
        }
        let mut joined = layer;
        joined.schema = schema;
        joined.batches = batches;
        Ok(joined)
    }
}

/// The layer's batches as one (an element layer is read with one batch already).
fn arrow_select_concat(layer: &EngineLayer) -> Result<RecordBatch> {
    match layer.batches.as_slice() {
        [one] => Ok(one.clone()),
        batches => {
            arrow_select::concat::concat_batches(&layer.schema, batches).map_err(|e| crate::Error::InvalidData(format!("arrow: {e}")))
        }
    }
}

/// Writes HEC-RAS archive layers, then `extra` layers, as one GeoParquet file with `write`
/// options (the archive profile: [`GeoParquetWriteOptions::archive`]). Without a `crs` in
/// `write`, the file's stored projection is written as PROJJSON when terrio can convert it, and
/// as `null` (with no CRS guessed) when the file stores none. `order`, when given, lists the
/// layer names in the order to write them (every layer, once).
pub fn write_archive(
    archive: HecRasArchive,
    extra: Vec<MemoryLayer>,
    order: Option<&[String]>,
    path: &str,
    write: &GeoParquetWriteOptions,
) -> Result<GeoParquetWriteReport> {
    let mut options = write.clone();
    if options.crs.is_none()
        && let Some(wkt) = &archive.crs_wkt
    {
        options.crs = Some(projjson(wkt)?);
    }
    let mut layers = archive.into_parquet_layers();
    layers.extend(extra);
    if let Some(order) = order {
        let mut by_name: HashMap<String, MemoryLayer> = HashMap::new();
        for layer in layers {
            let name = layer.name.clone().unwrap_or_default();
            if by_name.insert(name.clone(), layer).is_some() {
                return Err(crate::Error::InvalidArgument(format!("two layers named {name:?}")));
            }
        }
        layers = order
            .iter()
            .map(|n| by_name.remove(n).ok_or_else(|| crate::Error::InvalidArgument(format!("order names {n:?}, not a layer (or twice)"))))
            .collect::<Result<_>>()?;
        if let Some(left) = by_name.keys().next() {
            return Err(crate::Error::InvalidArgument(format!("order leaves out the layer {left:?}")));
        }
    }
    if layers.is_empty() {
        return Err(crate::Error::InvalidArgument("no layers to write".into()));
    }
    write_arrow_layers_to_path(layers, path, &options)
}

/// The stored projection as the `crs` of GeoParquet `geo` metadata: PROJJSON from terrio's WKT
/// reader, or the WKT text when terrio cannot convert it (GeoParquet readers accept neither
/// guess nor loss; the text is the stored evidence).
fn projjson(wkt: &str) -> Result<serde_json::Value> {
    match crate::drivers::vector::projjson::from_wkt1(wkt, None) {
        Some(v) => Ok(v),
        None => Ok(serde_json::Value::String(wkt.to_string())),
    }
}

/// The archive layers of a geometry or plan HDF file at `path`.
pub fn archive_layers(path: impl AsRef<std::path::Path>, options: &ArchiveOptions) -> Result<HecRasArchive> {
    HecRasHdf::open(path)?.archive_layers(options)
}

/// Reads a geometry or plan HDF file's archive layers and writes them, then `extra`, as one
/// GeoParquet file with the archive profile ([`GeoParquetWriteOptions::archive`]).
pub fn write_hecras_archive(
    hdf_path: impl AsRef<std::path::Path>,
    out_path: &str,
    options: &ArchiveOptions,
    extra: Vec<MemoryLayer>,
) -> Result<(GeoParquetWriteReport, Vec<ArchiveSkip>)> {
    let archive = archive_layers(hdf_path, options)?;
    let skipped = archive.skipped.clone();
    let report = write_archive(archive, extra, None, out_path, &GeoParquetWriteOptions::archive())?;
    Ok((report, skipped))
}
