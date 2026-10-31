# HEC-RAS fixtures

Synthetic HEC-RAS-shaped HDF5 files for the HECRAS driver's tests. They contain no model data.
The `coverage_*`, `pipes.h5` and `terrain.h5` files are copied unchanged from the engine repository,
[gpt-cmdr/ras-commander-fast-hdf](https://github.com/gpt-cmdr/ras-commander-fast-hdf) (MIT,
Copyright (c) 2024 CLB Engineering Corporation; see NOTICE), `tests/fixtures/coverage/` at
`d6d52f41cb0ef0d4170f60db25d28e01dd543ecd` (release 0.2.0; the same bytes as at the earlier
`a7243bfa36b2fb7d2d54c1a49ea2906e68d6ef3d`). Their generator and full manifest are listed in that
directory's `coverage_manifest.json`.

| File | SHA-256 | Contents |
|---|---|---|
| `coverage_mesh.h5` | `ff8172ccd3ff76e643416faa474110e6ba8bf8ff218333e8423dbe8591491117` | 19 meshes: `Good` (two drawable cells and a one-face cell), precision and width variants, and broken topologies |
| `coverage_mesh_nogroups.h5` | `4ed5ad7c89e14ff367a4a0ac34d2ea85c5b78913d651d62a49fe1eddedef84fd` | Legacy names-only layout (the engine refuses it) |
| `coverage_mesh_nogeometry.h5` | `3117dbec3e3c323ffbec8ea95728a55b3e592d5f792ab6fa15151902e0220485` | No `Geometry/2D Flow Areas` group |
| `results_mesh_xs.h5` | `797d88f9cccaa655ff243b19319bdb7a1cf9e584cce5b3658500066aa96ad429` | Written by `scripts/make_hecras_fixture.py` from `coverage_mesh.h5`: the `Good` mesh with made-up static arrays, summary, time series and three cross sections with summary and unsteady results, two boundary-condition lines with series, two reference points and one reference line with results, one river junction, the cross sections' station-elevation, Manning's n and ineffective-area tables and the mesh's face-point perimeter flags and elevations, in the stored HEC-RAS 6.x/7.0 layout. No model data |
| `pipes.h5` | `0df382abd29f03f88ea5423349318c80adf347167be660175ac2fb3610a10f3f` | Copied unchanged from the engine's `tests/fixtures/terrio_io/` (generator `make_fixtures.py` there): two pipe conduits and three pipe nodes in the HEC-RAS 7.0.1 layout, root `File Type` `HEC-RAS Geometry` |
| `arrow_layers.h5` | `70c1bad4f2866b9ff7ddcaa127eee18c33ba7160c0469f8939e332911c15f037` | Copied unchanged from the engine's `tests/fixtures/arrow_layers/` (generator `tests/generate_arrow_layers_fixture.py`, released in v0.3.0): one 2D area `Area A` with two cells and a ghost cell, a curved face, a stored `Perimeter`, `Plan Data/Plan Information` `Simulation Start Time` and timed `Maximum Water Surface` / `Maximum Face Velocity` summaries. Used by `tests/hecras_archive.rs` |
| `terrain.h5` | `47f92cb1eefe16c0818a3f0b9da1cde9de4dee7299367ce5a91a518fdbaeeeac` | Copied unchanged from the same directory: an HDF5 file whose root `File Type` is `HEC Terrain`, which the driver must not claim |

Real HEC-RAS models are not committed. `tests/hecras_corpus.rs` reads the files of a corpus
directory (`TERRIO_HECRAS_TEST_DATA`) and compares them with the JSON that
`scripts/hecras_oracle.py` writes with h5py; `scripts/compare_hecras_with_h5py.py` runs both on
the HEC-RAS 7.0.1 files of a directory.
