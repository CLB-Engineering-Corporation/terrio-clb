# Changelog

## Unreleased

### Formats

- NetCDF and HDF5: hdf5-reader, hdf5-core, netcdf-reader and netcdf-core 0.10.1 from
  crates.io, the release ras-commander-fast-hdf builds with, so one HDF5 reader serves both.
  The git patch on 0.9.1 is gone: its fix (v1/v2 compound member names, old fill value
  messages) is in the release. netCDF-4 variables are now listed in creation order, as
  netCDF-C lists them (they were in link-storage order), and deflated one-element chunks
  that 0.9.1 refused ("DEFLATE stream did not finish") read, as h5py reads them.
- HEC-RAS geometry (`.g##.hdf`) and plan (`.p##.hdf`) HDF files are read as
  vector layers (the `HECRAS` driver, `hecras` feature), through the
  [`ras-commander-fast-hdf`](https://crates.io/crates/ras-commander-fast-hdf)
  engine: 2D mesh cells, cell centres, faces, face points and perimeters (with
  the stored 2D flow area table); cross sections (with station-elevation,
  Manning's n and ineffective areas as JSON fields), river lines, storage
  areas, structures, boundary-condition lines, breaklines, refinement regions,
  reference lines and points, junctions, pump stations, and pipe conduits and
  nodes, each with its stored attribute columns. Stored results join through
  open options (summary maxima and minima with their times, time-step values,
  series extremes, cross-section, structure, boundary and reference results).
  The CRS is the stored `Projection` WKT. A file is claimed only when its root
  `File Type` is `HEC-RAS Geometry` or `HEC-RAS Results`; other HDF5 and
  NetCDF files are not affected. Compared with h5py on the HEC-RAS 7.0.1 files
  of a public corpus: every value identical. See `docs/hecras.md`.
- The HECRAS driver reads its mesh and family layers through the engine's Arrow layer catalog
  (`ras_commander_fast_hdf::arrow_layers`), which also names their fields: one source of
  HEC-RAS layer rules and names for terrio and ras2cng. `mesh_cell_points` keeps a null
  `is_ghost` for a mesh without a recorded cell count, a structure without a centerline is a
  feature without geometry, a family attribute stored as NaN is null, and `MeshLayer::omitted` lists the cells left out of
  `mesh_cells`. The typed engine API remains for joins, face points, quarantined cells and
  HEC-RAS 5.0 layouts without a 2D flow area table.
- `terrio::hecras::archive` (Python `terrio.hecras.archive_layers` and
  `terrio.hecras.write_archive`): a HEC-RAS geometry or plan HDF file straight to an archive
  GeoParquet file (ras2cng's layout), with the layers as Arrow tables so that a caller can add
  its own; refused layers and variables are listed, not fatal. See `docs/hecras.md`.
- GeoParquet from Arrow record batches (`terrio::geoparquet::write_arrow_layers`, Python
  `terrio.write_geoparquet_layers`), keeping the Arrow column types, with archive layout options:
  several layers in one file with a layer column and a union schema (identity columns
  non-nullable), row groups ending at layer boundaries, a GeoParquet 1.1 `bbox` struct (or flat)
  covering, Hilbert order within each layer bit-identical to DuckDB spatial's `ST_Hilbert` with an
  optional `hilbert_index` column, the ZSTD level, dictionaries on Utf8 columns only and chunk
  statistics without a page index. The `archive` profile reproduces ras2cng's archive files
  (rows, order, schema, row groups and `geo` metadata).
- Arrow and Parquet move to major 59 (arrow 59.0, parquet 59.0, the versions
  ras-commander-fast-hdf builds with), so the HEC-RAS engine's record batches pass to
  `terrio::geoparquet` without conversion. wbvector's own GeoParquet support (parquet
  58) is no longer built: terrio's GeoParquet driver already read and wrote `.parquet`,
  and a GeoParquet file without that extension is now recognized by its `PAR1`
  signature with terrio's driver.
- SRTM HGT tiles are read and written (the `SRTMHGT` driver), as GDAL's SRTMHGT
  driver does: 3", 1" and 0.5" tiles georeferenced from their name
  (`N37W120.hgt`), `.hgt.zip` archives, and the NASADEM and water body variants.
  Tiles are written as Int16 with -32768 for nodata; a grid of another size is
  refused.
- GeoTIFF and COG: LERC masks the padding of partial edge tiles of
  floating-point bands (one band, or `INTERLEAVE=BAND`), as GDAL does, whatever
  the nodata value. The padding was stored as valid zeros: on a 6705 x 4852
  flood-depth COG, LERC with `MAX_Z_ERROR=0.01` shrinks by 3.5% (NaN nodata)
  to 13% (-9999 nodata). Integer and pixel-interleaved LERC tiles, and the
  other codecs, keep GDAL's zeros. Pixels inside the raster are unchanged.
- GeoTIFF and COG area-matched overviews: `overview_report_metadata`
  (Python `report_metadata={"area_ratios": ..., "base_valid_pixels": ...}`)
  also stores the overview report in the default metadata domain under the
  caller's item names, written into the header with the overviews: the area
  ratios as Python's `"%.6f"` (correctly rounded, ties to even),
  comma-separated, bands separated by `;`. A reader of the default domain
  (ras2cng's `overview_area_ratios`, `overview_base_valid_pixels`) no longer
  needs a pass of its own over the raster. Names of the caller's `metadata`
  items are refused.
- `area_matched_nan_valid` counts NaN under a numeric nodata as valid in
  area-matched overviews, as rasterio's mask (and ras2cng's NumPy builder)
  does: NaN adds to the coverage and the mean of a kept cell holding one is
  NaN, bit for bit as NumPy computes it. The default (NaN is never valid) is
  unchanged. LERC stores such NaN as masked pixels, which read back as NaN.
- The area-matched `OverviewReport` (Python: the dict `Raster.write` returns)
  also counts, per band, the NaN samples under a numeric nodata
  (`nan_pixels`), from the same pass, whether or not `area_matched_nan_valid`
  makes them valid.
- SRTM HGT tiles and the other raw-layout rasters (EHdr, ENVI, SAGA, ERS,
  Idrisi, Surfer 6 binary grids) are read in place (#333): `open_raster_lazy`
  reads only the header, a window reads only the rows it covers, and `info`
  decodes no pixels. Remote SRTM tiles are read with HTTP range requests, and
  `convert` to a GeoTIFF or COG writes these inputs a strip at a time.
- PMTiles and MBTiles written from a raster opened lazily (`terrio convert`, `Raster.write`)
  warp a raster that is not on its zoom level's grid a chunk at a time, from windows of the
  source, instead of reading it whole: the colour and alpha warps of GDAL's MBTiles driver run in
  their own chunks, in step. The archive is byte-identical. A geographic raster reaching beyond
  Web Mercator's latitudes is still read whole.

### Remote I/O

- Signed S3 uploads of local files over 8 MiB sent over HTTPS read the file once
  per attempt instead of twice (#335), as a retried, redirected or
  region-rerouted request reads it again: they are signed with
  `UNSIGNED-PAYLOAD`, as the AWS SDKs sign S3 uploads over HTTPS, and the body
  is streamed. The signature no longer covers the body of these uploads; TLS
  protects it in transit. Over plain HTTP, and for smaller or staged uploads,
  the payload hash is signed as before.

### Breaking changes

- Signed S3 uploads of local files over 8 MiB over HTTPS use `UNSIGNED-PAYLOAD`
  (#335). An S3-compatible service, or a bucket policy, that refuses it now
  rejects those uploads; they can be sent over plain HTTP to keep a signed
  payload, with `AWS_HTTPS=NO`, or an `http://` `AWS_S3_ENDPOINT` when the
  endpoint names its scheme (an `https://` endpoint ignores `AWS_HTTPS`). Plain
  HTTP does not encrypt the request body or headers, including
  `x-amz-security-token` for temporary credentials; use this workaround only
  over a trusted connection.
- `mosaic`, `warp`, `concat` and `filter merge` now treat an input beginning
  with `@` as a file list, so a file named `@name.tif` must be written
  `./@name.tif`.
- WKT2 `Goode Homolosine` (PROJ's uninterrupted `goode`) is refused; it was read
  as `igh`.

### Command line

- `mosaic`, `warp`, `concat` and `filter merge` read inputs from a file list:
  `@FILE` stands for the paths listed in FILE, one per line, as GDAL's
  `--input @filelist.txt` does (#323). MCP tool calls refuse `@FILE`, so that a
  client cannot read a file back through the errors naming its lines.
- `terrio calc` reads its inputs by window and writes a strip of rows at a time, with the values
  of the calculation in memory (`calc_to_path`; `--max-memory MIB`, 256 by default, sets the
  strips' height and the evaluation threads). Inputs of other extents (`--extent union|intersect`,
  `--projwin`) are aligned with the rounding of the whole read. Python's `terrio.calc(expr, dst,
  A=path, ...)` does the same when every input is a path (`str` or `os.PathLike`), takes
  `max_memory_mib`, and returns the result opened lazily from `dst`.

### Point clouds

- PDAL's `filters.hexbin`: the boundary of points on a grid of hexagons (the hexagons with at
  least a threshold of points, as polygons with holes), with its area, density and point
  spacing, and the hexagons with their counts. `terrio filter hexbin` (and MCP) writes them as
  vector datasets, reading the points in chunks; Python's `PointCloud.hexbin` returns PDAL's
  metadata (`boundary` as WKT, `boundary_json`...) with both as layers; JavaScript's
  `PointCloud.hexbin` returns both as a vector dataset; Rust has `pointcloud::filters::hexbin`
  and `HexBinner`. PDAL's options (`edge_length`/`edge_size` or its estimate from
  `sample_size` points, `threshold`, `smooth`, `hole_cull_area_tolerance`, `precision`) and a
  grid `origin`; H3 grids are not implemented. Hexagons, counts and boundaries (smoothed or not)
  are identical to PDAL 2.10's on autzen; points are assigned to hexagons as hexer assigns them
  (a port, attributed in `NOTICE`).
- Reprojection (`terrio convert --t-srs`, `convert_points(t_srs=...)`,
  `PointCloud.reproject`, `append_points`) now transforms `z` when the points'
  CRS and the target CRS both have heights, as PDAL's `filters.reprojection`
  does: vertical units, geoid models and vertical datum transformations apply.
  It previously transformed only `x` and `y`, so 300 ftUS in
  `EPSG:6588+6360` stayed 300 in `EPSG:6588+5703` (now 91.44018 m). The
  vertical CRS a LAS file declares is now part of the CRS reprojected from
  (it was dropped) unless a CRS is assigned with `--a-srs`, and LAS files
  written with GeoKeys record a compound CRS's vertical CRS in
  `VerticalCSTypeGeoKey`. When only the target has heights, `z` is kept and a
  warning says so.

### CRS compatibility

- Transformer construction now checks celestial-body compatibility for all CRS
  pairs, including custom PROJ definitions without IAU identifiers. Different
  known bodies are rejected. When either body is unidentified, semi-major axes
  must agree within 0.7% of the target radius. For example, `+R=1` to `+R=2`, or a
  lunar/custom small sphere to EPSG:4326, now returns an error; matching custom
  figures remain supported.
- Earth datum operations (`+datum`, `+towgs84`, or datum grids) require an Earth
  figure, including for otherwise matching unidentified custom radii. They are
  rejected on planetary/custom small figures rather than interpreted as valid
  Earth shifts. This intentionally also applies without IAU identifiers. There
  is no `PROJ_IGNORE_CELESTIAL_BODY` override in Rust, CLI/MCP, Python or
  JavaScript. Use a CRS on the correct body; changes of arbitrary coordinate
  units should use the applicable CRS unit parameters.
- Geographic WKT2 definitions with west- or south-positive axes now preserve
  those directions and their declared order, including Earth CRSs. They were
  previously treated as ordinary east/north coordinates. This matches GDAL's
  traditional GIS axis strategy; ordinary north/east Earth axes still use
  longitude, latitude order.
- Vertical CRSs without geoid models of their own now use those of the vertical
  CRSs of their datum, as PROJ does. NAVD88 height in US survey feet
  (EPSG:6360) or feet (EPSG:8228) and NAVD88 depth (EPSG:6357, EPSG:6358) go
  through GEOID18 like NAVD88 height (EPSG:5703), in both directions, with the
  unit and direction converted. They previously only converted the unit,
  an error of the geoid undulation (about 21 to 28 m across Texas):
  `EPSG:6318+6360` to `EPSG:6319` at 30 N, 97 W turns 328.0833 ftUS
  (100 m) into 73.4074 m, as PROJ 9.9.0, instead of 100 m.

### Projections

- `+proj=igh` (interrupted Goode homolosine) is computed as PROJ does; it was
  refused. GDAL's WKT1 `Interrupted_Goode_Homolosine`, ESRI's
  `Goode_Homolosine` with `Option` 1 and WKT2 `Interrupted Goode Homolosine`
  are read, and the first is written.
- `+proj=ob_tran` with an `+o_proj` other than `longlat` (a projection on a
  rotated sphere or ellipsoid) is computed as PROJ does; it was refused. The
  pole may be given by `+o_lat_p`/`+o_lon_p`, by `+o_lon_c`/`+o_lat_c`/`+o_alpha`
  or by `+o_lon_1`/`+o_lat_1`/`+o_lon_2`/`+o_lat_2`, for rotated-pole grids too.
  PROJ's WKT2 `PROJ ob_tran o_proj=...` method is read.

### Warp

- With bilinear, cubic, cubic spline and Lanczos, an output pixel whose source
  pixel under the sample point is nodata (or cut out) is now left unwritten
  (at the output's nodata value, when it has one) for a single-band source too,
  as GDAL does: it took a value renormalised over the valid neighbours, which
  moved the edge of the data about a pixel into nodata areas. Several bands
  already behaved so, with unified nodata.
- `terrio warp` and `warp_to_path` warp several inputs a chunk at a time too, each chunk reading
  a window of its input, and write the output a strip of rows at a time: each pixel still takes
  the inputs in order, so the output is the in-memory warp's, bit for bit. Python's
  `terrio.warp(paths, dst=path)` does the same for several paths.
- `terrio mosaic` and `mosaic_to_path` write every mosaic a strip at a time from windows of the
  inputs, bit for bit as the mosaic in memory: inputs resampled by any method, at fractional
  offsets or non-integer resolution ratios, now too (each strip computed with the rounding of the
  whole mosaic, not a VRT's per-window rounding), and inputs in other CRSs, warped a chunk at a
  time. Only an output that is one of the inputs reads them whole.

### Shapefile

- In polygon records of several parts, rings that touch themselves at a vertex
  to enclose a hole, which Esri allows, are now split into an exterior and
  holes when read, wherever the stored ring starts, as GDAL 3.13 does (GDAL
  3.12 keeps them as stored). A ring whose loops are not holes (a figure-eight,
  a loop outside the ring, a spike), and a record of a single ring, are still
  read as stored, also as GDAL does. The split rings start at the vertex the
  rings GEOS rebuilds for GDAL start at. The `shapefile-rings` conformance
  suite compares 18 such cases with GDAL.
- A hole whose first vertex lies on the exterior now belongs to that exterior.
  It was read as a polygon of its own.
