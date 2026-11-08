use super::*;
use arrow_array::{BooleanArray, Float32Array, Int64Array, ListArray, types::Float64Type};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::reader::{FileReader, SerializedFileReader};

fn point(x: f64, y: f64) -> Vec<u8> {
    [&[1u8, 1, 0, 0, 0][..], &x.to_le_bytes(), &y.to_le_bytes()].concat()
}

/// A big-endian square polygon with a hole, `size` wide at `(x, y)`.
fn square(x: f64, y: f64, size: f64) -> Vec<u8> {
    let ring = |pts: &[(f64, f64)]| {
        let mut out = (pts.len() as u32).to_be_bytes().to_vec();
        for (a, b) in pts {
            out.extend(a.to_be_bytes());
            out.extend(b.to_be_bytes());
        }
        out
    };
    let outer = [(x, y), (x + size, y), (x + size, y + size), (x, y + size), (x, y)];
    let c = (x + size / 2.0, y + size / 2.0);
    let hole = [c, (c.0 + 0.1, c.1), (c.0, c.1 + 0.1), c];
    [&[0u8][..], &3u32.to_be_bytes(), &2u32.to_be_bytes(), &ring(&outer), &ring(&hole)].concat()
}

fn empty_polygon() -> Vec<u8> {
    [&[1u8][..], &3u32.to_le_bytes(), &0u32.to_le_bytes()].concat()
}

fn crs_field() -> Field {
    Field::new("geometry", DataType::Binary, true).with_metadata(HashMap::from([
        ("ARROW:extension:name".to_string(), "geoarrow.wkb".to_string()),
        (
            "ARROW:extension:metadata".to_string(),
            r#"{"crs": {"id": {"authority": "EPSG", "code": 26918}}, "crs_type": "projjson"}"#.to_string(),
        ),
    ]))
}

/// A layer of `mesh_name`, `cell_id`, an extra column and geometries.
fn layer(name: &str, extra: (&str, ArrayRef), geometries: Vec<Option<Vec<u8>>>) -> MemoryLayer {
    let n = geometries.len();
    let schema = Arc::new(Schema::new(vec![
        Field::new("mesh_name", DataType::Utf8, true),
        Field::new("cell_id", DataType::Int64, true),
        Field::new(extra.0, extra.1.data_type().clone(), true),
        crs_field(),
    ]));
    let wkb: BinaryArray = geometries.iter().map(|g| g.as_deref()).collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from_iter_values(std::iter::repeat_n("m", n))),
            Arc::new(Int64Array::from_iter_values(0..n as i64)),
            extra.1,
            Arc::new(wkb),
        ],
    )
    .unwrap();
    ArrowLayer::new(Some(name.into()), schema, vec![Ok(batch)]).with_identity(["mesh_name", "cell_id"])
}

fn read(bytes: &[u8]) -> (SchemaRef, RecordBatch, Value, Vec<usize>) {
    let bytes = bytes::Bytes::copy_from_slice(bytes);
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes.clone()).unwrap();
    let schema = builder.schema().clone();
    let batches: Vec<RecordBatch> = builder.build().unwrap().map(|b| b.unwrap()).collect();
    let batch = arrow_select::concat::concat_batches(&schema, &batches).unwrap();
    let reader = SerializedFileReader::new(bytes).unwrap();
    let meta = reader.metadata();
    let geo = meta.file_metadata().key_value_metadata().unwrap().iter().find(|kv| kv.key == "geo").unwrap().value.clone().unwrap();
    let groups = meta.row_groups().iter().map(|g| g.num_rows() as usize).collect();
    (schema, batch, serde_json::from_str(&geo).unwrap(), groups)
}

fn ids(batch: &RecordBatch) -> Vec<i64> {
    batch.column_by_name("cell_id").unwrap().as_primitive::<arrow_array::types::Int64Type>().values().to_vec()
}

fn strings(batch: &RecordBatch, name: &str) -> Vec<String> {
    batch.column_by_name(name).unwrap().as_string::<i32>().iter().map(|v| v.unwrap().to_string()).collect()
}

#[test]
fn hilbert_matches_duckdb_spatial() {
    // DuckDB spatial: ST_Hilbert(x, y, {'min_x':0,'min_y':0,'max_x':10,'max_y':10}::BOX_2D)
    let extent = [0., 0., 10., 10.];
    for (x, y, expected) in [
        (0., 0., 0u32),
        (10., 10., 2863311530),
        (5., 5., 715827882),
        (1.5, 2.5, 157903209),
        (9.99, 0.01, 4294959096),
        (3.3, 7.7, 1629721655),
        (0., 10., 1431655765),
        (10., 0., 4294967295),
        (2.5, 2.5, 178956970),
    ] {
        assert_eq!(hilbert_index(x, y, extent), expected, "({x}, {y})");
    }
    // A degenerate axis is cell 0; out-of-extent values saturate.
    assert_eq!(hilbert_index(3., 3., [3., 3., 3., 3.]), 0);
    // Outside the extent, cells saturate to u32 and keep their low 16 bits, as in DuckDB.
    assert_eq!(hilbert_index(-5., 20., extent), hilbert_encode_16(0, 131070 & 0xFFFF));
}

#[test]
fn hilbert_permutation_is_stable_with_empty_rows_last() {
    let b = |x: f64, y: f64| Some([x, y, x, y]);
    let keys = hilbert_keys(&[None, b(10., 0.), b(0., 0.), b(0., 0.), None, b(5., 5.)]);
    assert_eq!(hilbert_permutation(&keys), vec![2, 3, 5, 1, 0, 4]);
}

#[test]
fn wkb_bounds_cover_every_coordinate_and_skip_empties() {
    assert_eq!(wkb_bounds(&square(10., 20., 4.)).unwrap(), Some([10., 20., 14., 24.]));
    assert_eq!(wkb_bounds(&point(1., 2.)).unwrap(), Some([1., 2., 1., 2.]));
    assert_eq!(wkb_bounds(&empty_polygon()).unwrap(), None);
    assert_eq!(wkb_bounds(&point(f64::NAN, f64::NAN)).unwrap(), None);
    // ISO Z point: Z is not part of the bounds; the type name says Z.
    let z = [&[1u8][..], &1001u32.to_le_bytes(), &3f64.to_le_bytes(), &4f64.to_le_bytes(), &99f64.to_le_bytes()].concat();
    assert_eq!(wkb_bounds(&z).unwrap(), Some([3., 4., 3., 4.]));
    assert_eq!(wkb_type_name(&z).unwrap(), "Point Z");
    // A multipolygon of two squares.
    let multi = [&[1u8][..], &6u32.to_le_bytes(), &2u32.to_le_bytes(), &square(0., 0., 1.), &square(5., 5., 1.)].concat();
    assert_eq!(wkb_bounds(&multi).unwrap(), Some([0., 0., 6., 6.]));
    // Truncated and unknown input is refused, not guessed.
    assert!(wkb_bounds(&square(0., 0., 1.)[..30]).is_err());
    assert!(wkb_bounds(&[1u8, 9, 0, 0, 0]).is_err());
    assert!(wkb_bounds(&[1u8, 2, 0, 0, 0, 255, 255, 255, 255]).is_err());
}

#[test]
fn archive_profile_unions_layers_with_covering_hilbert_and_layer_groups() {
    let cells = layer(
        "cells",
        ("min_elevation", Arc::new(Float32Array::from(vec![Some(1.5), None, Some(3.0)]))),
        vec![Some(square(100., 100., 8.)), Some(square(0., 0., 4.)), Some(square(50., 50., 2.))],
    );
    let points = layer("points", ("active", Arc::new(BooleanArray::from(vec![true, false]))), vec![Some(point(7., 9.)), None]);
    let mut options = GeoParquetWriteOptions::archive();
    options.row_group_rows = 2;
    let mut out = Vec::new();
    let report = write_arrow_layers(vec![cells, points], &mut out, &options).unwrap();
    assert_eq!((report.rows, report.layers, report.row_groups), (5, 2, 3));
    assert_eq!(report.bbox, Some([0., 0., 108., 108.]));
    let (schema, batch, geo, groups) = read(&out);
    // Row groups of 2 rows that also end at the layer boundary.
    assert_eq!(groups, [2, 1, 2]);
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, ["mesh_name", "cell_id", "min_elevation", "geometry", "active", "layer", "bbox", "hilbert_index"]);
    // Identity columns of every layer are non-nullable; a column missing from a layer is null.
    let nullable: Vec<bool> = schema.fields().iter().map(|f| f.is_nullable()).collect();
    assert_eq!(nullable, [false, false, true, true, true, false, true, true]);
    assert_eq!(schema.field_with_name("min_elevation").unwrap().data_type(), &DataType::Float32);
    assert_eq!(
        schema.field_with_name("geometry").unwrap().metadata().get("ARROW:extension:name").map(String::as_str),
        Some("geoarrow.wkb")
    );
    // Hilbert order within each layer; the empty geometry last.
    assert_eq!(ids(&batch), [1, 2, 0, 0, 1]);
    assert_eq!(strings(&batch, "layer"), ["cells", "cells", "cells", "points", "points"]);
    let bbox = batch.column_by_name("bbox").unwrap().as_struct();
    assert!(bbox.is_null(4) && bbox.is_valid(0));
    assert_eq!(bbox.column(2).as_primitive::<Float64Type>().value(0), 4.0);
    let h = batch.column_by_name("hilbert_index").unwrap().as_primitive::<arrow_array::types::UInt64Type>();
    let keys = hilbert_keys(&[Some([100., 100., 108., 108.]), Some([0., 0., 4., 4.]), Some([50., 50., 52., 52.])]);
    assert_eq!(h.value(0), u64::from(keys[1].unwrap()));
    assert!(h.is_null(4));
    let column = &geo["columns"]["geometry"];
    assert_eq!(geo["version"], "1.1.0");
    assert_eq!(column["geometry_types"], json!(["Point", "Polygon"]));
    assert_eq!(column["bbox"], json!([0.0, 0.0, 108.0, 108.0]));
    assert_eq!(column["covering"]["bbox"]["xmin"], json!(["bbox", "xmin"]));
    assert_eq!(column["crs"], json!({"id": {"authority": "EPSG", "code": 26918}}));
}

#[test]
fn layout_encoding_options() {
    let cells = || {
        layer(
            "cells",
            ("note", Arc::new(StringArray::from(vec!["a", "b", "c"]))),
            vec![Some(square(0., 0., 1.)), Some(square(2., 2., 1.)), Some(empty_polygon())],
        )
    };
    let column_chunks = |options: &GeoParquetWriteOptions| {
        let mut out = Vec::new();
        write_arrow_layers(vec![cells()], &mut out, options).unwrap();
        let reader = SerializedFileReader::new(bytes::Bytes::from(out)).unwrap();
        let meta = reader.metadata().clone();
        let group = meta.row_group(0);
        let chunks: Vec<(String, bool, bool, Compression)> = group
            .columns()
            .iter()
            .map(|c| (c.column_path().string(), c.dictionary_page_offset().is_some(), c.statistics().is_some(), c.compression()))
            .collect();
        (chunks, meta.offset_index().is_some(), meta.file_metadata().schema_descr().root_schema().name().to_string())
    };
    let (chunks, page_index, root) = column_chunks(&GeoParquetWriteOptions::archive());
    assert_eq!(root, "schema");
    assert!(!page_index);
    let dictionary: Vec<&str> = chunks.iter().filter(|c| c.1).map(|c| c.0.as_str()).collect();
    assert_eq!(dictionary, ["mesh_name", "note", "layer"]);
    // The level is not stored in the file.
    assert!(chunks.iter().all(|c| matches!(c.3, Compression::ZSTD(_))));
    // Flat covering columns, no layer column, no geometry statistics, input order.
    let options = GeoParquetWriteOptions {
        bbox_covering: BboxCovering::Columns,
        bbox_column: "bbox".into(),
        geometry_statistics: false,
        ..GeoParquetWriteOptions::default()
    };
    let (chunks, _, _) = column_chunks(&options);
    let names: Vec<&str> = chunks.iter().map(|c| c.0.as_str()).collect();
    assert_eq!(names, ["mesh_name", "cell_id", "note", "geometry", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]);
    assert!(!chunks[3].2, "geometry statistics are off");
    assert!(chunks.iter().all(|c| c.3 == Compression::SNAPPY));
    let mut out = Vec::new();
    write_arrow_layers(vec![cells()], &mut out, &options).unwrap();
    let (_, batch, geo, _) = read(&out);
    assert_eq!(ids(&batch), [0, 1, 2]);
    assert!(batch.column_by_name("bbox_xmin").unwrap().is_null(2));
    assert_eq!(geo["columns"]["geometry"]["covering"]["bbox"]["ymax"], json!(["bbox_ymax"]));
}

#[test]
fn lists_and_null_columns_round_trip() {
    let values = ListArray::from_iter_primitive::<Float64Type, _, _>(vec![Some(vec![Some(1.0), Some(2.0)]), None]);
    let typed = layer("a", ("stations", Arc::new(values)), vec![Some(point(0., 0.)), Some(point(1., 1.))]);
    let untyped = layer("b", ("stations", Arc::new(arrow_array::NullArray::new(1))), vec![Some(point(2., 2.))]);
    let mut out = Vec::new();
    write_arrow_layers(vec![untyped, typed], &mut out, &GeoParquetWriteOptions::archive()).unwrap();
    let (schema, batch, _, groups) = read(&out);
    assert_eq!(groups, [1, 2]);
    let DataType::List(item) = schema.field_with_name("stations").unwrap().data_type() else { panic!("not a list") };
    assert_eq!(item.name(), "element");
    let stations = batch.column_by_name("stations").unwrap().as_list::<i32>();
    assert!(stations.is_null(0));
    assert_eq!(stations.value(1).as_primitive::<Float64Type>().values().to_vec(), [1.0, 2.0]);
}

#[test]
fn split_by_column_keeps_first_appearance_order() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("geometry", DataType::Binary, true),
        Field::new("layer", DataType::Utf8, false),
    ]));
    let batch = |ids: Vec<i64>, layers: Vec<&str>| {
        let wkb: BinaryArray = ids.iter().map(|&i| Some(point(i as f64, 0.0))).collect();
        RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(ids)), Arc::new(wkb), Arc::new(StringArray::from(layers))])
            .unwrap()
    };
    let batches = [batch(vec![0, 1, 2], vec!["b", "b", "a"]), batch(vec![3, 4], vec!["b", "a"])];
    let layers = split_by_column(&schema, &batches, "layer").unwrap();
    let summary: Vec<(String, Vec<i64>)> = layers
        .iter()
        .map(|l| {
            let ids = l
                .batches
                .iter()
                .flat_map(|b| b.as_ref().unwrap().column(0).as_primitive::<arrow_array::types::Int64Type>().values().to_vec())
                .collect();
            (l.name.clone().unwrap(), ids)
        })
        .collect();
    assert_eq!(summary, [("b".to_string(), vec![0, 1, 3]), ("a".to_string(), vec![2, 4])]);
    assert_eq!(layers[0].schema.fields().len(), 2);
    let mut options = GeoParquetWriteOptions::archive();
    options.crs = Some(json!("EPSG:4326"));
    let mut out = Vec::new();
    write_arrow_layers(layers, &mut out, &options).unwrap();
    let (schema, batch, geo, groups) = read(&out);
    assert_eq!(groups, [3, 2]);
    assert_eq!(strings(&batch, "layer"), ["b", "b", "b", "a", "a"]);
    assert!(!schema.field_with_name("id").unwrap().is_nullable());
    assert_eq!(geo["columns"]["geometry"]["crs"], json!("EPSG:4326"));
}

#[test]
fn refusals() {
    let one = || layer("a", ("x", Arc::new(Int64Array::from(vec![1]))), vec![Some(point(0., 0.))]);
    let write = |layers: Vec<MemoryLayer>, options: &GeoParquetWriteOptions| write_arrow_layers(layers, Vec::new(), options).map(|_| ());
    // Several layers need a layer column, and unique names.
    assert!(
        write(vec![one(), layer("b", ("x", Arc::new(Int64Array::from(vec![1]))), vec![None])], &GeoParquetWriteOptions::default()).is_err()
    );
    assert!(write(vec![one(), one()], &GeoParquetWriteOptions::archive()).is_err());
    // Different types for one field.
    let other = layer("b", ("x", Arc::new(StringArray::from(vec!["1"]))), vec![None]);
    assert!(write(vec![one(), other], &GeoParquetWriteOptions::archive()).is_err());
    // A generated column colliding with an input column.
    let clash = layer("a", ("BBOX", Arc::new(Int64Array::from(vec![1]))), vec![None]);
    assert!(write(vec![clash], &GeoParquetWriteOptions::archive()).is_err());
    // A Hilbert column without Hilbert order.
    let options = GeoParquetWriteOptions { hilbert_column: Some("h".into()), ..GeoParquetWriteOptions::default() };
    assert!(write(vec![one()], &options).is_err());
    // Nulls in an identity column.
    let nulls = layer("a", ("x", Arc::new(Int64Array::from(vec![1]))), vec![None]).with_identity(["x", "mesh_name"]);
    let mut nulls = nulls;
    nulls.batches = vec![Ok(RecordBatch::try_new(
        nulls.schema.clone(),
        vec![
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(Int64Array::from(vec![0])),
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(BinaryArray::from(vec![None::<&[u8]>])),
        ],
    )
    .unwrap())];
    assert!(write(vec![nulls], &GeoParquetWriteOptions::default()).is_err());
    // Bad WKB, and a bad ZSTD level.
    let bad = layer("a", ("x", Arc::new(Int64Array::from(vec![1]))), vec![Some(vec![1, 2, 3])]);
    assert!(write(vec![bad], &GeoParquetWriteOptions::default()).is_err());
    let options = GeoParquetWriteOptions { compression: ParquetCompression::Zstd(99), ..GeoParquetWriteOptions::default() };
    assert!(write(vec![one()], &options).is_err());
    assert!(ParquetCompression::parse("snappy", Some(3)).is_err());
    assert_eq!(ParquetCompression::parse("zstd", Some(3)).unwrap(), ParquetCompression::Zstd(3));
}

#[test]
fn path_writer_replaces_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out.parquet");
    std::fs::write(&path, b"old").unwrap();
    let cells = layer("a", ("x", Arc::new(Int64Array::from(vec![1]))), vec![Some(point(0., 0.))]);
    write_arrow_layers_to_path(vec![cells], path.to_str().unwrap(), &GeoParquetWriteOptions::archive()).unwrap();
    let (_, batch, _, _) = read(&std::fs::read(&path).unwrap());
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1, "no staged file is left behind");
}

#[test]
fn open_sniffs_geoparquet_without_its_extension() {
    // terrio's own driver reads it (wbvector's GeoParquet is not built).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cells.data");
    let cells = layer("a", ("x", Arc::new(Int64Array::from(vec![7]))), vec![Some(point(1., 2.))]);
    let options =
        GeoParquetWriteOptions { crs: Some(json!({"id": {"authority": "EPSG", "code": 26918}})), ..GeoParquetWriteOptions::default() };
    write_arrow_layers_to_path(vec![cells], path.to_str().unwrap(), &options).unwrap();
    let crate::Dataset::Vector(ds) = crate::open(path.to_str().unwrap()).unwrap() else { panic!("not a vector dataset") };
    assert_eq!(ds.layers[0].len(), 1);
    assert_eq!(ds.layers[0].features[0].geometry, Some(wbvector::Geometry::point(1., 2.)));
}

/// A varied table for the Zstandard tests: dictionary text, integers, floats and lists with nulls,
/// and polygons, in two layers, large enough for several pages per chunk and several row groups.
fn zstd_layers() -> Vec<MemoryLayer> {
    let make = |name: &str, n: usize, seed: u64| {
        let note: ArrayRef = Arc::new(StringArray::from_iter((0..n).map(|i| (i % 13 != 0).then(|| format!("note-{}", (i as u64 * seed) % 97)))));
        let mut l = layer(name, ("note", note), (0..n).map(|i| (i % 101 != 0).then(|| square((i % 400) as f64, (i / 400) as f64, 0.75))).collect());
        // A float column with nulls, and a list column, appended to the layer's batch.
        let batch = l.batches.remove(0).unwrap();
        let values: ArrayRef = Arc::new(Float64Array::from_iter((0..n).map(|i| (i % 7 != 0).then(|| (i as f64 * 0.37).sin() * 1e3))));
        let lists: ArrayRef = Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>(
            (0..n).map(|i| (i % 5 != 0).then(|| (0..i % 4).map(|k| Some(k as f64 + i as f64)).collect::<Vec<_>>())),
        ));
        let mut fields: Vec<Field> = batch.schema().fields().iter().map(|f| f.as_ref().clone()).collect();
        fields.insert(3, Field::new("value", DataType::Float64, true));
        fields.insert(4, Field::new_list("items", Field::new_list_field(DataType::Float64, true), true));
        let schema = Arc::new(Schema::new(fields));
        let mut columns = batch.columns().to_vec();
        columns.insert(3, values);
        columns.insert(4, lists);
        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
        ArrowLayer::new(Some(name.into()), schema, vec![Ok(batch)]).with_identity(["mesh_name", "cell_id"])
    };
    vec![make("cells", 90_000, 31), make("faces", 70_000, 7)]
}

/// Every page of every column chunk, read (and decompressed) by the `parquet` crate.
fn pages(bytes: &bytes::Bytes) -> Vec<(String, parquet::basic::PageType, u32, parquet::basic::Encoding, Vec<u8>)> {
    use parquet::file::serialized_reader::SerializedPageReader;
    let reader = SerializedFileReader::new(bytes.clone()).unwrap();
    let mut out = Vec::new();
    for g in reader.metadata().row_groups() {
        for c in g.columns() {
            for page in SerializedPageReader::new(Arc::new(bytes.clone()), c, g.num_rows() as usize, None).unwrap() {
                let page = page.unwrap();
                out.push((c.column_path().string(), page.page_type(), page.num_values(), page.encoding(), page.buffer().to_vec()));
            }
        }
    }
    out
}

/// The record batches of a file, read by terrio's pure-Rust Zstandard pages.
fn read_pure(bytes: &bytes::Bytes) -> RecordBatch {
    use crate::drivers::vector::geoparquet_zstd;
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes.clone()).unwrap();
    let (meta, schema) = (builder.metadata().clone(), builder.schema().clone());
    let groups: Vec<usize> = (0..meta.num_row_groups()).collect();
    let mask = parquet::arrow::ProjectionMask::all();
    assert!(geoparquet_zstd::has_zstd(&meta, &groups, &mask));
    let reader = geoparquet_zstd::reader(bytes.clone(), &meta, &schema, mask, groups, None, 8192).unwrap();
    let batches: Vec<RecordBatch> = reader.map(|b| b.unwrap()).collect();
    arrow_select::concat::concat_batches(&schema, &batches).unwrap()
}

/// GeoParquet Zstandard pages compressed in pure Rust (`ruzstd`) and by the `parquet` crate (the
/// zstd C library) make the same file but for the compressed bytes: the same metadata, row groups,
/// chunk statistics and encodings, and page for page the same decompressed bytes. Each reader
/// decodes both files to the same table.
#[test]
fn pure_rust_zstd_pages_match_the_zstd_c_library_page_for_page() {
    let options = GeoParquetWriteOptions { row_group_rows: 40_000, ..GeoParquetWriteOptions::archive() };
    let write = |pure: bool| {
        let (report, out) = write_layers_with(zstd_layers(), Vec::new(), &options, pure).unwrap();
        (report, bytes::Bytes::from(out))
    };
    let ((pure_report, pure), (c_report, c)) = (write(true), write(false));
    assert_eq!(pure_report, c_report);
    assert_ne!(pure, c, "the compressed bytes differ");
    let (pr, cr) = (SerializedFileReader::new(pure.clone()).unwrap(), SerializedFileReader::new(c.clone()).unwrap());
    let (pm, cm) = (pr.metadata(), cr.metadata());
    assert_eq!(pm.file_metadata().key_value_metadata(), cm.file_metadata().key_value_metadata());
    assert_eq!(pm.file_metadata().schema_descr().root_schema(), cm.file_metadata().schema_descr().root_schema());
    assert_eq!(pm.num_row_groups(), cm.num_row_groups());
    assert!(pm.num_row_groups() > 4);
    for (pg, cg) in pm.row_groups().iter().zip(cm.row_groups()) {
        assert_eq!(pg.num_rows(), cg.num_rows());
        for (p, c) in pg.columns().iter().zip(cg.columns()) {
            let path = p.column_path().string();
            assert_eq!(p.compression(), c.compression(), "{path}");
            assert!(matches!(p.compression(), Compression::ZSTD(_)), "{path}");
            assert_eq!(p.num_values(), c.num_values(), "{path}");
            assert_eq!(p.encodings_mask(), c.encodings_mask(), "{path}");
            assert_eq!(p.statistics(), c.statistics(), "{path}");
            assert_eq!(p.dictionary_page_offset().is_some(), c.dictionary_page_offset().is_some(), "{path}");
            assert_eq!(p.page_encoding_stats(), c.page_encoding_stats(), "{path}");
        }
    }
    let (pp, cp) = (pages(&pure), pages(&c));
    assert!(pp.len() > 2 * pm.row_groups().iter().map(|g| g.num_columns()).sum::<usize>(), "several pages per chunk");
    assert_eq!(pp, cp);
    // The crate (with the C library) and terrio's pure-Rust pages read both to the same table.
    let (_, want, _, _) = read(&c);
    assert_eq!(want.num_rows(), 160_000);
    assert_eq!(read(&pure).1, want);
    assert_eq!(read_pure(&pure), want);
    assert_eq!(read_pure(&c), want);
}

/// Each level, as terrio maps it (see `zstandard::pure::ruzstd_level`), writes a file the
/// `parquet` crate reads back.
#[test]
fn pure_rust_zstd_levels_round_trip() {
    let mut sizes = Vec::new();
    for level in [1, 3, 9, 19] {
        let options = GeoParquetWriteOptions { compression: ParquetCompression::Zstd(level), ..GeoParquetWriteOptions::archive() };
        let (_, out) = write_layers_with(vec![zstd_layers().remove(1)], Vec::new(), &options, true).unwrap();
        let out = bytes::Bytes::from(out);
        assert_eq!(read(&out).1.num_rows(), 70_000, "level {level}");
        sizes.push(out.len());
    }
    // Higher levels compress no worse.
    assert!(sizes.windows(2).all(|w| w[1] <= w[0]), "{sizes:?}");
    // The crate's level range is kept.
    let options = GeoParquetWriteOptions { compression: ParquetCompression::Zstd(23), ..GeoParquetWriteOptions::archive() };
    assert!(write_layers_with(zstd_layers(), Vec::new(), &options, true).is_err());
}
