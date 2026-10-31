//! The HECRAS vector driver: detection of HEC-RAS files and the layers of an open, with the
//! joins [`HecRasOpenOptions`](crate::HecRasOpenOptions) asks for.

use super::{
    CrossSectionOptions, CrossSectionTable, Extreme, FeatureSeriesOptions, GeometryPolicy, HecRasHdf, MESH_CELL_POINTS, MESH_CELLS,
    MESH_FACE_POINTS, MESH_FACES, MESH_PERIMETERS, MeshAttribute, MeshOptions, ResultMode,
};
use crate::{Error, HecRasOpenOptions, Layer, Result, VectorOpenOptions};
use ras_commander_fast_hdf::hdf::metadata::HEC_RAS_FILE_TYPES;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// The HDF5 superblock signature.
const SIGNATURE: &[u8; 8] = b"\x89HDF\r\n\x1a\n";

/// Whether `path` is a HEC-RAS geometry or plan HDF file: an HDF5 signature (at offset 0, 512,
/// 1024, ...) and a root `File Type` the engine classifies as HEC-RAS's
/// ([`HEC_RAS_FILE_TYPES`]). Any other HDF5 file, including NetCDF-4 files and HEC terrain and
/// land-cover files, is left to the HDF5 and NetCDF drivers.
///
/// `File Type` says only that the file is HEC-RAS's: which kind it is does not follow from it
/// (plans before 6.4 can say `HEC-RAS Geometry`, and 6.4+ geometry files say `HEC-RAS Results`),
/// so the layers come from what the file stores, never from `File Type`.
pub(crate) fn is_hecras(path: &str) -> bool {
    has_signature(path)
        && ras_commander_fast_hdf::hdf::ReadOnlyHdfSession::with_options(Path::new(path), 1, false)
            .and_then(|s| s.hec_ras_file_type())
            .is_ok_and(|t| t.is_hec_ras())
}

fn has_signature(path: &str) -> bool {
    let Ok(mut f) = std::fs::File::open(path) else { return false };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let mut offset = 0u64;
    let mut buf = [0u8; 8];
    while offset + 8 <= len {
        if f.seek(SeekFrom::Start(offset)).is_err() || f.read_exact(&mut buf).is_err() {
            return false;
        }
        if &buf == SIGNATURE {
            return true;
        }
        offset = if offset == 0 { 512 } else { offset * 2 };
    }
    false
}

/// Whether bytes held in memory are a HEC-RAS geometry or plan file: an HDF5 signature and a
/// root `File Type` in [`HEC_RAS_FILE_TYPES`], as [`is_hecras`] decides for files. The bytes are
/// sniffed in place (an engine session would take ownership of a copy).
pub(crate) fn is_hecras_bytes(bytes: &[u8]) -> bool {
    crate::drivers::hdf5::root_file_type_in(bytes).is_some_and(|t| HEC_RAS_FILE_TYPES.contains(&t.trim_end_matches('\0')))
}

/// Reads the layers `options` selects with the joins `options.hecras` asks for.
///
/// Named layers must all read: any refusal is the open's error. With no layer named, every
/// stored layer is read, and a layer the engine refuses (an unsupported layout, or a table it
/// needs that the file does not store, such as the ghost-cell count of HEC-RAS 5.0 files) is left
/// out with a warning ([`crate::warnings`]); damaged data and I/O errors still fail the open.
pub(crate) fn read(path: &str, options: &VectorOpenOptions) -> Result<Vec<Layer>> {
    read_hdf(&HecRasHdf::open(path)?, path, options)
}

/// Reads a HEC-RAS file held in memory as [`read`] reads files. `name` names the file in
/// messages and identities.
pub(crate) fn read_bytes(bytes: Vec<u8>, name: &str, options: &VectorOpenOptions) -> Result<Vec<Layer>> {
    read_hdf(&HecRasHdf::from_bytes(bytes, name)?, name, options)
}

fn read_hdf(hdf: &HecRasHdf, path: &str, options: &VectorOpenOptions) -> Result<Vec<Layer>> {
    let available = hdf.layer_names()?;
    if !options.layers.is_empty() {
        if let Some(missing) = options.layers.iter().find(|l| !available.contains(l)) {
            return Err(Error::InvalidArgument(format!("no layer named {missing} (layers: {})", available.join(", "))));
        }
        return options.layers.iter().map(|name| read_layer(hdf, name, &options.hecras)).collect();
    }
    let mut layers = Vec::with_capacity(available.len());
    for name in &available {
        match read_layer(hdf, name, &options.hecras) {
            Ok(layer) => layers.push(layer),
            Err(e @ (Error::Unsupported(_) | Error::InvalidArgument(_))) => {
                crate::warnings::warn(format!("HECRAS: layer {name} left out: {e} (name it with a layer option to see it fail)"))
            }
            Err(e) => return Err(e),
        }
    }
    if layers.is_empty() {
        return Err(Error::Unsupported(format!("'{path}': the engine refused every HEC-RAS layer (see the warnings)")));
    }
    Ok(layers)
}

fn read_layer(hdf: &HecRasHdf, name: &str, o: &HecRasOpenOptions) -> Result<Layer> {
    let series_row =
        || o.time_step.ok_or_else(|| Error::InvalidArgument(format!("{name}: the series options need TIME_STEP (a stored output row)")));
    let feature_series = |variables: &[String]| -> Result<FeatureSeriesOptions> {
        let mut f = FeatureSeriesOptions::at_row(series_row()?);
        f.variables = variables.to_vec();
        budgets(o, &mut f.metadata_max_bytes, &mut f.max_bytes);
        Ok(f)
    };
    match name {
        MESH_CELLS => Ok(hdf.mesh_cells(&mesh_options(o, true)?)?.layer),
        MESH_CELL_POINTS => Ok(hdf.mesh_cell_points(&mesh_options(o, true)?)?.layer),
        MESH_FACES => Ok(hdf.mesh_faces(&mesh_options(o, false)?)?.layer),
        MESH_FACE_POINTS => Ok(hdf.mesh_face_points(&base_mesh_options(o))?.layer),
        MESH_PERIMETERS => Ok(hdf.mesh_perimeters(&base_mesh_options(o))?.layer),
        "cross_sections" if !(o.xs_summaries.is_empty() && o.xs_results.is_empty() && o.xs_geometry.is_empty()) => {
            let mut x = CrossSectionOptions { summaries: o.xs_summaries.clone(), ..Default::default() };
            for g in &o.xs_geometry {
                x.geometry.push(match g.as_str() {
                    "STATION_ELEVATION" => CrossSectionTable::StationElevation,
                    "MANNINGS_N" => CrossSectionTable::ManningsN,
                    "INEFFECTIVE_AREAS" => CrossSectionTable::IneffectiveAreas,
                    other => return Err(Error::InvalidArgument(format!("unknown XS_GEOMETRY value '{other}'"))),
                });
            }
            if !o.xs_results.is_empty() {
                x.results = o.xs_results.clone();
                (x.mode, x.row) = match (o.xs_profile, o.time_step) {
                    (Some(p), _) => (ResultMode::Steady, Some(p)),
                    (None, Some(t)) => (ResultMode::Unsteady, Some(t)),
                    (None, None) => {
                        return Err(Error::InvalidArgument("XS_RESULTS need TIME_STEP (unsteady) or XS_PROFILE (steady)".into()));
                    }
                };
            }
            budgets(o, &mut x.metadata_max_bytes, &mut x.max_bytes);
            Ok(hdf.cross_sections(&x)?.layer)
        }
        "structures" if !o.structure_series.is_empty() => Ok(hdf.structures_at(&feature_series(&o.structure_series)?)?.layer),
        "bc_lines" if o.bc_series => Ok(hdf.bc_lines_at(&feature_series(&[])?)?.layer),
        "reference_lines" if !o.reference_line_series.is_empty() => {
            Ok(hdf.reference_lines_at(&feature_series(&o.reference_line_series)?)?.layer)
        }
        "reference_points" if !o.reference_point_series.is_empty() => {
            Ok(hdf.reference_points_at(&feature_series(&o.reference_point_series)?)?.layer)
        }
        other => hdf.read_layer(other),
    }
}

fn budgets(o: &HecRasOpenOptions, metadata: &mut usize, arrays: &mut usize) {
    if let Some(m) = o.metadata_max_bytes {
        *metadata = m;
    }
    if let Some(m) = o.max_bytes {
        *arrays = m;
    }
}

fn base_mesh_options(o: &HecRasOpenOptions) -> MeshOptions {
    let mut m = MeshOptions::default();
    if !o.meshes.is_empty() {
        m.meshes = Some(o.meshes.clone());
    }
    if o.quarantine {
        m.policy = GeometryPolicy::QuarantineInvalid;
    }
    budgets(o, &mut m.metadata_max_bytes, &mut m.max_bytes);
    m
}

/// The mesh options of the cell layers (`cells`) or the face layer.
fn mesh_options(o: &HecRasOpenOptions, cells: bool) -> Result<MeshOptions> {
    let mut m = base_mesh_options(o);
    let (summaries, attributes, series, maximum, minimum) = if cells {
        (&o.cell_summaries, &o.cell_attributes, &o.cell_series, &o.cell_maximum, &o.cell_minimum)
    } else {
        (&o.face_summaries, &o.face_attributes, &o.face_series, &o.face_maximum, &o.face_minimum)
    };
    m.summaries = summaries.clone();
    for a in attributes {
        m.attributes.push(match a.as_str() {
            "MINIMUM_ELEVATION" => MeshAttribute::MinimumElevation,
            "MANNINGS_N" => MeshAttribute::ManningsN,
            "NORMAL_AND_LENGTH" => MeshAttribute::FaceNormalAndLength,
            "CELLS" => MeshAttribute::FaceCells,
            other => return Err(Error::InvalidArgument(format!("unknown mesh attribute '{other}'"))),
        });
    }
    if !series.is_empty() {
        m.series = series.clone();
        m.time_step = Some(
            o.time_step.ok_or_else(|| Error::InvalidArgument("CELL_SERIES and FACE_SERIES need TIME_STEP (a stored output row)".into()))?,
        );
    }
    m.extremes =
        maximum.iter().map(|v| (v.clone(), Extreme::Maximum)).chain(minimum.iter().map(|v| (v.clone(), Extreme::Minimum))).collect();
    Ok(m)
}
