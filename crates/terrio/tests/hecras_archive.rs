//! The direct HEC-RAS HDF to archive GeoParquet path (`terrio::hecras::archive`) on the synthetic
//! plan fixture `results_mesh_xs.h5`: layer names and fields from the engine catalog's ras2cng
//! aliases, summary geometry joined by identity, the lossless additions, refusals listed instead
//! of failing the file, and the written file read back.
#![cfg(all(feature = "hecras", feature = "parquet"))]

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Int64Type};
use std::path::PathBuf;
use terrio::hecras::HecRasHdf;
use terrio::hecras::archive::{ArchiveLayerKind, ArchiveOptions, GEOMETRY_LAYERS, archive_name, write_archive};

fn path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/hecras/results_mesh_xs.h5")
}

fn open() -> HecRasHdf {
    HecRasHdf::open_with_threads(path(), 1).unwrap()
}

fn names(schema: &arrow_schema::Schema) -> Vec<&str> {
    schema.fields().iter().map(|f| f.name().as_str()).collect()
}

fn one(batches: &[RecordBatch]) -> RecordBatch {
    arrow_select::concat::concat_batches(&batches[0].schema(), batches).unwrap()
}

#[test]
fn archive_names_come_from_the_catalog() {
    assert_eq!(archive_name("mesh_perimeters"), "mesh_areas");
    assert_eq!(archive_name("river_centerlines"), "centerlines");
    assert_eq!(archive_name("bc_lines"), "bc_lines");
    assert!(GEOMETRY_LAYERS.iter().all(|n| terrio::hecras::archive::archive_name(n) == *n || n == &"mesh_areas"));
}

#[test]
fn geometry_layers_have_ras2cng_names_and_lossless_cell_points() {
    let archive = open().archive_layers(&ArchiveOptions::default().with_results(false)).unwrap();
    let layer_names: Vec<&str> = archive.layers.iter().map(|l| l.name.as_str()).collect();
    // The fixture's mesh stores no Perimeter: the engine refuses `mesh_areas`, which is listed.
    assert!(archive.skipped.iter().any(|s| s.layer == "mesh_areas"), "{:?}", archive.skipped);
    for expected in ["mesh_cells", "mesh_faces", "cross_sections", "bc_lines", "reference_lines", "reference_points"] {
        assert!(layer_names.contains(&expected), "{expected} in {layer_names:?}");
    }
    // ras2cng's order: mesh layers, then 1D, then 2D features, then the catalog-only layers.
    let at = |n: &str| layer_names.iter().position(|l| *l == n).unwrap();
    assert!(at("mesh_cells") < at("mesh_faces") && at("mesh_faces") < at("bc_lines") && at("bc_lines") < at("mesh_cell_points"));
    let bc = archive.layers.iter().find(|l| l.name == "bc_lines").unwrap();
    assert_eq!(names(&bc.schema)[0], "bc_line_id", "source_row under ras2cng's name");
    assert_eq!(bc.identity, ["bc_line_id"]);
    assert!(matches!(&bc.kind, ArchiveLayerKind::Geometry { canonical } if canonical == "bc_lines"));
    // Stored attribute names are kept for ras2cng (no snake_case).
    assert!(names(&bc.schema).contains(&"Name"), "{:?}", names(&bc.schema));

    let points = archive.layers.iter().find(|l| l.name == "mesh_cell_points").unwrap();
    let b = one(&points.batches);
    assert_eq!(names(&points.schema), ["mesh_name", "cell_id", "is_ghost_cell", "geometry"]);
    let ghosts = b.column(2).as_boolean();
    assert!(ghosts.iter().any(|g| g == Some(true)), "ghost cells kept and flagged");
    let cells = archive.layers.iter().find(|l| l.name == "mesh_cells").unwrap();
    assert_eq!(b.num_rows(), cells.num_rows() + ghosts.iter().filter(|g| *g == Some(true)).count());
    assert!(archive.crs_wkt.is_some());
}

fn plan() -> HecRasHdf {
    HecRasHdf::open_with_threads(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/hecras/arrow_layers.h5"), 1).unwrap()
}

#[test]
fn summaries_join_the_element_geometry_once_and_keep_stored_times() {
    let hdf = plan();
    let archive = hdf.archive_layers(&ArchiveOptions::default().with_geometry(false)).unwrap();
    let variables = hdf.summary_variables().unwrap();
    let mut sorted = variables.clone();
    sorted.sort();
    assert_eq!(variables, sorted, "name order, as ras2cng");
    // Every stored variable is a layer or listed as skipped.
    for v in &variables {
        assert!(
            archive.layers.iter().any(|l| matches!(&l.kind, ArchiveLayerKind::Summary { variable } if variable == v))
                || archive.skipped.iter().any(|s| s.variable.as_deref() == Some(v.as_str())),
            "{v}"
        );
    }
    let layer = |n: &str| archive.layers.iter().find(|l| l.name == n).unwrap_or_else(|| panic!("{n}: {:?}", archive.skipped));
    let wse = layer("maximum_water_surface");
    assert!(matches!(&wse.kind, ArchiveLayerKind::Summary { variable } if variable == "Maximum Water Surface"));
    assert_eq!(wse.identity, ["mesh_name", "cell_id"]);
    assert_eq!(
        names(&wse.schema),
        [
            "mesh_name",
            "cell_id",
            "maximum_water_surface",
            "maximum_water_surface_time",
            "maximum_water_surface_time_days",
            "is_ghost_cell",
            "geometry"
        ]
    );
    let b = one(&wse.batches);
    assert_eq!(b.num_rows(), 2, "ghost cell values are left out, as ras-commander does");
    assert_eq!(b.column(4).as_primitive::<Float32Type>().values(), &[0.5, 1.25], "the stored days");
    // Each row's geometry is the polygon of the same (mesh_name, cell_id) in mesh_cells.
    let cells = hdf.archive_layers(&ArchiveOptions::default().with_results(false).with_layers(["mesh_cells"])).unwrap();
    let c = one(&cells.layers[0].batches);
    assert_eq!(c.column(1).as_primitive::<Int64Type>().values(), b.column(1).as_primitive::<Int64Type>().values());
    assert_eq!(b.column(6).as_binary::<i32>(), c.column(2).as_binary::<i32>());
    // Face variables join the face lines.
    let faces = hdf.archive_layers(&ArchiveOptions::default().with_results(false).with_layers(["mesh_faces"])).unwrap();
    let f = one(&faces.layers[0].batches);
    let velocity = layer("maximum_face_velocity");
    let v = one(&velocity.batches);
    assert_eq!(velocity.identity, ["mesh_name", "face_id"]);
    let geometry = v.schema().index_of("geometry").unwrap();
    assert_eq!(v.column(geometry).as_binary::<i32>(), f.column(2).as_binary::<i32>());
}

#[test]
fn a_stored_variable_is_never_dropped_silently() {
    // `results_mesh_xs.h5` stores a timed summary but no `Plan Data` simulation start: the
    // engine cannot date it, and the variable is listed as skipped, not left out silently.
    let hdf = open();
    let archive = hdf.archive_layers(&ArchiveOptions::default().with_geometry(false)).unwrap();
    assert!(archive.layers.is_empty());
    assert_eq!(archive.skipped.len(), 1, "{:?}", archive.skipped);
    let s = &archive.skipped[0];
    assert_eq!(
        (s.layer.as_str(), s.variable.as_deref(), s.kind.as_str()),
        ("maximum_water_surface", Some("Maximum Water Surface"), "not_found")
    );
    assert!(s.reason.contains("Plan Data"), "{}", s.reason);
}

#[test]
fn unknown_layers_and_zero_batches_are_refused() {
    let hdf = open();
    let e = hdf.archive_layers(&ArchiveOptions::default().with_layers(["no_such_layer"])).unwrap_err();
    assert!(matches!(e, terrio::Error::InvalidArgument(_)), "{e}");
    assert!(hdf.archive_layers(&ArchiveOptions::default().with_batch_rows(0)).is_err());
}

#[test]
fn the_file_has_every_layer_and_extra_layers() {
    let hdf = open();
    let archive = hdf.archive_layers(&ArchiveOptions::default()).unwrap();
    let expected: Vec<(String, usize)> = archive.layers.iter().map(|l| (l.name.clone(), l.num_rows())).collect();
    // An extra layer, as ras2cng appends its text geometry layers.
    let point = {
        let mut w = vec![1u8];
        w.extend_from_slice(&1u32.to_le_bytes());
        w.extend_from_slice(&0.5f64.to_le_bytes());
        w.extend_from_slice(&0.5f64.to_le_bytes());
        w
    };
    let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("Name", arrow_schema::DataType::Utf8, false),
        arrow_schema::Field::new("geometry", arrow_schema::DataType::Binary, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            std::sync::Arc::new(arrow_array::StringArray::from(vec!["SA 1"])),
            std::sync::Arc::new(arrow_array::BinaryArray::from(vec![point.as_slice()])),
        ],
    )
    .unwrap();
    let extra = terrio::geoparquet::ArrowLayer::new(Some("storage_areas_text".into()), schema, vec![Ok(batch)]);
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("plan.parquet");
    let report =
        write_archive(archive, vec![extra], None, out.to_str().unwrap(), &terrio::geoparquet::GeoParquetWriteOptions::archive()).unwrap();
    assert_eq!(report.layers, expected.len() + 1);
    assert_eq!(report.rows, expected.iter().map(|(_, n)| n).sum::<usize>() + 1);

    let file = std::fs::File::open(&out).unwrap();
    let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let geo = reader.metadata().file_metadata().key_value_metadata().unwrap().iter().find(|kv| kv.key == "geo").unwrap().value.clone();
    let geo: serde_json::Value = serde_json::from_str(geo.as_deref().unwrap()).unwrap();
    assert!(!geo["columns"]["geometry"]["crs"].is_null(), "the stored projection is the file's CRS");
    let batches: Vec<RecordBatch> = reader.build().unwrap().collect::<Result<_, _>>().unwrap();
    let table = one(&batches);
    let layer = table.column_by_name("layer").unwrap();
    let layer = arrow_array::cast::AsArray::as_string_opt::<i32>(layer.as_ref())
        .map(|a| a.iter().map(|v| v.unwrap().to_string()).collect::<Vec<_>>())
        .unwrap_or_else(|| {
            let d = layer.as_dictionary::<arrow_array::types::Int32Type>();
            let values = d.values().as_string::<i32>();
            d.keys().iter().map(|k| values.value(k.unwrap() as usize).to_string()).collect()
        });
    for (name, rows) in &expected {
        assert_eq!(layer.iter().filter(|l| l == &name).count(), *rows, "{name}");
    }
    assert_eq!(layer.iter().filter(|l| *l == "storage_areas_text").count(), 1);
    assert!(table.column_by_name("hilbert_index").is_some() && table.column_by_name("bbox").is_some());

    // `order` puts the extra layer first; a missing or unknown name is refused.
    let archive = hdf.archive_layers(&ArchiveOptions::default()).unwrap();
    let mut order: Vec<String> = archive.layers.iter().map(|l| l.name.clone()).collect();
    order.pop();
    let e = write_archive(archive, vec![], Some(&order), out.to_str().unwrap(), &terrio::geoparquet::GeoParquetWriteOptions::archive())
        .unwrap_err();
    assert!(matches!(e, terrio::Error::InvalidArgument(_)), "{e}");
}
