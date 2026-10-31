# HEC-RAS

terrio reads HEC-RAS geometry (`.g##.hdf`) and plan (`.p##.hdf`) HDF files as vector layers: the
2D mesh cells, cell centres, faces and perimeters, the 1D cross sections, rivers, structures and
storage areas, boundary-condition lines, reference lines and points, with the results HEC-RAS
stored joined to them. The driver is called `HECRAS`, is part of the default build (the `hecras` Cargo feature), and only
reads: it does not run HEC-RAS, write HEC-RAS files, or read project, plan or `.prj` text.

```sh
terrio info BaldEagleDamBrk.p03.hdf
terrio convert BaldEagleDamBrk.p03.hdf cells.fgb -l mesh_cells --oo "CELL_SUMMARIES=Maximum Water Surface"
```

```python
import terrio

ds = terrio.open_vector(
    "BaldEagleDamBrk.p03.hdf",
    layers=["mesh_cells"],
    options={"CELL_SUMMARIES": "Maximum Water Surface", "CELL_SERIES": "Water Surface", "TIME_STEP": "12"},
)
cells = ds["mesh_cells"].to_geopandas()
```

```rust,no_run
let mut options = terrio::VectorOpenOptions { layers: vec!["mesh_cells".into()], ..Default::default() };
options.set("CELL_SUMMARIES", "Maximum Water Surface")?;
let ds = terrio::open_vector("BaldEagleDamBrk.p03.hdf", &options)?;
terrio::write_vector(&ds.layers, "cells.fgb", &Default::default())?;
# Ok::<(), terrio::Error>(())
```

```js
const ds = terrio.openVector(bytes, "BaldEagleDamBrk.p03.hdf", {
  layers: ["mesh_cells"],
  openOptions: { CELL_SUMMARIES: "Maximum Water Surface" },
});
```

## Supported versions

The driver is **tested on files written by HEC-RAS 7.0.1**: the conformance comparison and the
real-data tests use 7.0.1 files only. Files from older versions are read where the engine
supports their layout, best effort and not guaranteed. A layout the engine does not support is
refused with an error; it is never read as empty or approximate data.

## How it works

The driver is a thin adapter over
[`ras-commander-fast-hdf`](https://crates.io/crates/ras-commander-fast-hdf), the read-only
HEC-RAS engine of [ras-commander](https://github.com/gpt-cmdr/ras-commander). The engine owns
every HEC-RAS rule: the HDF layouts, mesh, cell and cross-section identities, polygon ring order,
CRS evidence, units and refusals. The driver translates its values into terrio layers, fields
and errors.

**One source of layer rules and names.** The mesh layers (`mesh_cells`, `mesh_cell_points`,
`mesh_faces`, `mesh_perimeters`) and every family layer are the engine's Arrow catalog layers
(`ras_commander_fast_hdf::arrow_layers`, `ReadOnlyHdfSession::read_layer`): the same rows, order,
geometry and refusals ras2cng archives. Field names come from the catalog too
(`ArrowLayer::with_consumer_names("terrio", layer)`: the stored name lower-snake-cased, and the
catalog's explicit aliases such as `is_ghost`), so the driver renames nothing itself. The
driver keeps the engine's typed API only for what the catalog does not provide:

- `mesh_face_points`, and the 2D flow area table joined to `mesh_perimeters`;
- every join (summaries, static mesh arrays, time-step values, series extremes; cross-section,
  structure, boundary-condition and reference-line/point results and tables);
- mesh cells under `GeometryPolicy::QuarantineInvalid` (the catalog refuses invalid cells);
- files without a 2D flow area table (`Geometry/2D Flow Areas/Attributes`, HEC-RAS 5.0
  layouts): the catalog lists meshes from that table, as ras-commander does;
- listing (`layer_names`), which only asks whether each family is stored.

**Detection.** A file is a HEC-RAS file when it has an HDF5 signature and a root `File Type`
attribute of `HEC-RAS Geometry` or `HEC-RAS Results`, as the engine classifies it
(`hec_ras_file_type`). The extension does not decide: HEC-RAS files share `.hdf` and `.h5` with
other HDF5 files, which stay with the HDF5 and NetCDF drivers. That includes NetCDF-4 files,
HDF5 files without a `File Type`, and HEC's terrain and land-cover files (`HEC Terrain`,
`HEC Land Cover`).
`File Type` only says that the file is HEC-RAS's, not which kind it is (plans before 6.4 can
say `HEC-RAS Geometry`, and 6.4 and later geometry files say `HEC-RAS Results`), so the layers
come from what the file stores. `terrio::open`, `open_vector`, `terrio info` and
`terrio convert` read HEC-RAS files as vector layers; `open_raster` (and `open_raster_bytes`)
still reads them as HDF5 rasters, as before.

## Layers

A default open reads every layer the file stores. A layer the engine refuses there (an
unsupported layout, or a table the file lacks, such as the ghost-cell count of HEC-RAS 5.0
files) is left out with a warning; damaged data and I/O errors fail the open. A layer named with
`-l` (`layers`) must read, or the open fails with the engine's refusal.

| Layer | Geometry | Fields |
|---|---|---|
| `mesh_cells` | Polygon | `mesh_name`, `cell_id`, joined cell values |
| `mesh_cell_points` | Point | `mesh_name`, `cell_id`, `is_ghost`, joined cell values |
| `mesh_faces` | LineString | `mesh_name`, `face_id`, joined face values |
| `mesh_face_points` | Point | `mesh_name`, `face_point_id`, `is_perimeter` (the stored `FacePoints Is Perimeter` integer: -1 on the perimeter, 0 inside), `elevation` (the stored `FacePoints Elevation`; null when not stored) |
| `mesh_perimeters` | Polygon | `mesh_name`, then the stored 2D flow area table (`name`, `locked`, `mann`, `cell_count`, spacing and tolerances, ...); listed when every 2D flow area stores a `Perimeter` |
| `cross_sections` | LineString | `source_row`, stored columns (`river`, `reach`, `rs`, `name`, bank stations, ...), joined 1D results |
| `river_centerlines`, `river_bank_lines`, `river_edge_lines`, `river_flow_paths` | LineString | `source_row`, stored columns |
| `storage_areas` | Polygon | `source_row`, stored columns |
| `structures` | LineString | `source_row`, stored columns (`type`, `river`, `reach`, `rs`, `connection`, ...), joined structure series |
| `bc_lines`, `breaklines`, `reference_lines` | LineString | `source_row`, stored columns (`name`, `sa_2d`, `type`, ...), joined series |
| `refinement_regions` | Polygon | `source_row`, stored columns |
| `reference_points` | Point | `source_row`, stored columns, joined series |
| `river_junctions`, `pump_stations` | Point | `source_row`, stored columns (`name`, ...) |
| `pipe_conduits` | LineString | `source_row`, stored columns (`name`, `us_node`, `ds_node`, `shape`, `rise`, `span`, `manning_s_n`, ...) |
| `pipe_nodes` | Point | `source_row`, stored columns (`name`, `node_type`, `invert_elevation`, ...) |

- Mesh features are identified by `mesh_name` with `cell_id` or `face_id`, the stored indices,
  which repeat across meshes. Other features are identified by `source_row`, the stored
  attribute-table row, and keep every stored attribute column as a field named after it (lower
  case, other characters as `_`).
- Cell polygons are the engine's: rings run clockwise from the first listed face. Cells the
  engine does not draw (fewer than three faces, such as ghost cells) are left out and reported
  in `MeshLayer::omitted`. A family line with several parts is a MultiLineString; a polygon
  family feature with several parts is refused, because which parts are holes is not a proven
  layout. A structure stored without a centerline is a feature without geometry.
- A family attribute stored as NaN is null: the engine's Arrow layers hold a NaN as null, so
  the stored NaN bit pattern is not kept. Every other value is identical.
- `mesh_cell_points` has every stored cell centre, ghost cells included (`is_ghost` true);
  `is_ghost` is null for a mesh whose physical cell count is not recorded.
- Attributes are read without h5py, by the engine: compound `Attributes` tables column by
  column with their stored types (text, integers, floats; `f32` widened exactly), fixed-length
  strings decoded as h5py decodes them (padding removed, nothing else trimmed), and the root
  attributes (`File Type`, `File Version`, `Units System`, `Projection`, ...) through
  `HecRasHdf::file_attributes`.
- The CRS is the stored `Projection` text, attached as WKT. No EPSG code is guessed and none is
  identified from the text. A file without a projection gives layers without a CRS (assign one
  with `--a_srs`). Malformed, conflicting, unsupported or corrupt projection evidence is an
  error.

### Not read

These HEC-RAS datasets are left out on purpose:

- A family with a line part of fewer than two points (one Spring_BLE breakline, HEC-RAS 6.1): the
  engine's catalog refuses the whole layer. ras-commander skips that row instead.
- Bitfield attribute columns: the engine's catalog does not map them to Arrow, so a family with
  one is refused (no tested file has one).
- 2D flow areas stored as outlines only, with no computed mesh (`Polygon Info`/`Parts`/`Points`
  under `Geometry/2D Flow Areas`): the engine refuses them, so the mesh layers are listed but
  refused.
- Calibration regions (`Geometry/2D Flow Area Calibration Regions`): no tested file stores one.
- Pipe network cells (`Geometry/Pipe Networks`), cross-section interpolation surfaces, bridge
  and culvert profile and coefficient tables, and hydraulic property tables: they are model
  tables, not feature layers.
- Values HEC-RAS does not store (depth where only water surface is stored, maxima of structure
  series): the driver joins stored values only and computes none.

## Archive GeoParquet

`terrio::hecras::archive` (features `hecras` and `parquet`) writes a geometry or plan HDF file
straight to one archive GeoParquet file, as ras2cng archives a project, without terrio's
`Layer` model or a GeoDataFrame in between:

```rust,no_run
use terrio::hecras::HecRasHdf;
use terrio::hecras::archive::{ArchiveOptions, write_archive};
use terrio::geoparquet::GeoParquetWriteOptions;

let archive = HecRasHdf::open("Model.p01.hdf")?.archive_layers(&ArchiveOptions::default().with_geometry(false))?;
for skip in &archive.skipped {
    eprintln!("{} refused: {}", skip.layer, skip.reason); // fill these another way
}
write_archive(archive, Vec::new(), None, "Model.p01.parquet", &GeoParquetWriteOptions::archive())?;
# Ok::<(), terrio::Error>(())
```

Python: `terrio.hecras.archive_layers(path, ...)` returns the layers as Arrow tables, and
`terrio.hecras.write_archive(path, out, extra_layers=[(name, table), ...])` writes them, then
the caller's layers (ras2cng's text geometry layers, values the engine refused), as one file.

- Layer and field names are the catalog's `ras2cng` aliases (`mesh_areas`, `centerlines`,
  `bc_line_id`, `bl_id`, ...); other fields keep their stored names. Geometry layers come in
  ras2cng's order, then the catalog layers ras2cng's Python path does not write
  (`mesh_cell_points`, `river_flow_paths`, `river_junctions`).
- A plan's results are one layer per stored 2D summary variable, in name order, named by its
  slug, each with the cell polygon or face line read once per file and joined by
  `(mesh_name, cell_id|face_id)`.
- Lossless by default: a structure without a centerline is a row with a null geometry, a timed
  summary also keeps its stored offset (`<slug>_time_days`, days from the simulation start, in
  the stored float type; `<slug>_time` is rounded to 100 ms), and `mesh_cell_points` keeps the
  ghost cells (`is_ghost_cell`), which face cell indexes refer to.
- A layer or variable the engine refuses does not abort the file: it is listed in
  `HecRasArchive::skipped` (Python `skipped`) with the engine's reason. `Cell Last Iteration`
  is one: it stores no element `Location`, so the engine does not prove its element domain.
- The layout is `GeoParquetWriteOptions::archive()`: ZSTD 3, a `layer` column, row groups ending
  at layer boundaries, a `bbox` struct covering, Hilbert order with `hilbert_index`, Utf8
  dictionaries, chunk statistics. The CRS is the stored projection as PROJJSON (its text when
  terrio cannot convert it); none is guessed.

## Open options

Every joined value is a stored number, joined by the engine's element or feature identity.
Asking for a value the file does not store is an error, never an empty field; options for
layers that are not read are ignored. Lists are comma-separated stored names.

| Option | Joins | Fields |
|---|---|---|
| `MESHES` | only these 2D flow areas | |
| `GEOMETRY_POLICY=STRICT\|QUARANTINE` | `QUARANTINE` leaves invalid cell polygons out instead of failing | |
| `CELL_SUMMARIES`, `FACE_SUMMARIES` | stored `Summary Output` variables (`Maximum Water Surface`, `Maximum Face Velocity`, ...) to the cell layers or `mesh_faces` | `<variable>`, `<variable>_time` (stored days, when the engine proves a Time row) |
| `CELL_ATTRIBUTES` | `MINIMUM_ELEVATION`, `MANNINGS_N` | `minimum_elevation`, `mannings_n` |
| `FACE_ATTRIBUTES` | `MINIMUM_ELEVATION`, `NORMAL_AND_LENGTH`, `CELLS` | `minimum_elevation`, `normal_x`, `normal_y`, `length`, `cell_a`, `cell_b` |
| `CELL_SERIES`, `FACE_SERIES` (with `TIME_STEP`) | stored time series at one output row (`Water Surface`, `Face Velocity`, `Depth` where stored) | `<variable>` |
| `CELL_MAXIMUM`, `CELL_MINIMUM`, `FACE_MAXIMUM`, `FACE_MINIMUM` | the engine's reduction of a stored series over every row (first occurrence; NaN propagates) | `<variable>_max`, `<variable>_max_row` (or `_min`) |
| `TIME_STEP` | the stored unsteady output row (zero-based) of the series options | |
| `XS_SUMMARIES` | stored `Summary Output/Cross Sections` variables to `cross_sections` | `<variable>`, `<variable>_time`, `<variable>_<row label>` |
| `XS_RESULTS` (with `TIME_STEP` or `XS_PROFILE`) | stored cross-section results at an unsteady row or a steady profile | `<variable>` |
| `XS_PROFILE` | the steady profile (zero-based) of `XS_RESULTS` | |
| `XS_GEOMETRY` | `STATION_ELEVATION`, `MANNINGS_N`, `INEFFECTIVE_AREAS`: the stored per-section tables of `Geometry/Cross Sections` to `cross_sections` | `station_elevation` (`[[station, elevation], ...]`), `mannings_n` (`[[station, n], ...]`), `ineffective_areas` (`[{"Left Sta": .., "Right Sta": .., "Elevation": .., "Permanent": ..}, ...]`), JSON fields; a NaN, which JSON cannot hold, is `null` |
| `STRUCTURE_SERIES` (with `TIME_STEP`) | stored structure series (`Structure Variables`, ...) to `structures` | `<series>_<column label>` |
| `BC_SERIES=YES` (with `TIME_STEP`) | each 2D boundary-condition line's stored series to `bc_lines` | one field per stored column label (`stage`, `flow`, ...) |
| `REFERENCE_LINE_SERIES`, `REFERENCE_POINT_SERIES` (with `TIME_STEP`) | stored reference-line and reference-point variables | `<variable>` |
| `METADATA_MAX_BYTES`, `MAX_BYTES` | the engine's byte budgets per metadata object and per array | |

Values are the stored numbers (`f32` widened to `f64` exactly). Depth is joined only where
HEC-RAS stored it (HEC-RAS 5.0 plans store a cell `Depth` series); the driver never computes
one. Units, time axes and other report details are on the `terrio::hecras` API
(`MeshLayer::time_step`, `SummaryField::units`, ...), since vector layers have no metadata slot.

`HecRasOpenOptions` holds the same options in Rust, and `terrio::hecras::HecRasHdf` gives the
whole engine-backed API (`mesh_cells`, `cross_sections`, `structures_at`, ...) with typed
options and the reports of each read.

## Errors

| Engine | terrio |
|---|---|
| `file_not_found` | `Error::Io` (`NotFound`) |
| `io`, `internal` | `Error::Io` |
| `unsupported`, or a byte budget (`resource_limit`) | `Error::Unsupported` |
| `not_found`, `invalid_request`, an unavailable capability | `Error::InvalidArgument` |
| `corrupt`, `structural_topology`, `invalid_geometry` | `Error::InvalidData` |
| CRS evidence that is not usable | `Error::Crs` |

Messages start with the engine's stable spellings, such as
`HEC-RAS unsupported(identity_layout): ...`.

## In memory and in WebAssembly

`terrio::memory::open_bytes`, `open_vector_bytes`, Python's `open_bytes` and the JavaScript
package's `open` and `openVector` read HEC-RAS files from bytes (natively the bytes are spilled
to a temporary file; in WebAssembly the engine reads them in memory). Limits:

- The whole file is held in memory while it is open, and opening needs about twice its size for a
  moment (the detection reads the root attributes from a copy).
- The wasm32 module's memory is 4 GiB in all, so plan files approaching that size cannot be read
  there; native builds have no such limit.
- The engine's byte budgets (`METADATA_MAX_BYTES`, `MAX_BYTES`) apply as for files: a dataset
  above them is refused, never allocated blindly.

Range reads (HTTP, S3) are not supported yet. They would plug in at the engine's byte source,
as terrio's `sparse::SparseFile` reads remote files: plan the byte ranges a read needs from the
HDF5 metadata, fetch them, then decode. That needs an engine byte-source interface and is on the
engine's roadmap.

## Build and size

The `hecras` feature is a default feature; build with `--no-default-features` (and the features you
want) to leave it out. It implies `netcdf-hdf5`. Over the default features it adds one crate,
the engine; its other dependencies, including the HDF5 reader, are already terrio's.

| | Without `hecras` | With `hecras` | Difference |
|---|---:|---:|---:|
| `terrio` command-line binary (Windows x86-64, release, 2026-10-28) | 69.4 MB | 72.0 MB | +2.6 MB (+3.8%) |
| JavaScript module (`wasm-pack --release`, `wasm-opt`; 2026-10-24, when the engine still embedded a second HDF5 reader, so an upper bound) | 22.3 MB | 24.6 MB | +2.3 MB (+10.4%) |
| JavaScript module, gzipped (same) | 7.28 MB | 7.82 MB | +0.54 MB (+7.5%) |

The Python package and the JavaScript package's full module include it.

### One HDF5 reader

terrio and the engine use the same `hdf5-reader` 0.10 from crates.io (roteiro-gis/netcdf-rust),
so a build with `hecras` compiles one HDF5 reader. The engine adds its own read policy on top
(refusing filters without an encoded-size bound as intermediate decoding stages) instead of
patching the reader. `cargo tree -d -e normal -p terrio --features hecras` lists the same
duplicate crates as a build without `hecras`, and none of them is an HDF5 or NetCDF crate.

## Testing

```sh
cargo test -p terrio --test hecras_driver --test hecras_results --test hecras_synthetic
python scripts/compare_hecras_with_h5py.py <corpus>       # the 7.0.1 conformance comparison
```

The CI tests run on synthetic fixtures (`crates/terrio/tests/data/hecras`, written by
`scripts/make_hecras_fixture.py` or taken from the engine's committed fixtures). The comparison
with h5py reads a corpus directory whose location is a parameter, and compares the HEC-RAS 7.0.1
files in it; see [Conformance](conformance.md). GDAL cannot read HEC-RAS files, so the reference
is h5py: ras-commander, the open-source Python library for HEC-RAS, reads these datasets with
h5py, and `scripts/hecras_oracle.py` reads them the same way, independently of terrio and its
engine. Every compared value must be identical.
