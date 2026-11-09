//! `terrio warp --wm`, the working memory of a warp chunk, read as `gdalwarp -wm` reads it.

use std::path::Path;
use std::process::{Command, Output};

fn run(args: &[&str], dir: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_terrio")).args(args).current_dir(dir).output().unwrap()
}

fn values(path: &Path) -> Vec<f64> {
    terrio::open_raster(path.to_str().unwrap()).unwrap().raster.data.to_f64_vec()
}

#[test]
fn chunk_memory_is_read_as_gdalwarp_reads_it() {
    let tmp = tempfile::tempdir().unwrap();
    let d = tmp.path();
    let (cols, rows) = (300, 200);
    let cfg = terrio::RasterConfig {
        cols,
        rows,
        x_min: 500000.0,
        y_min: 4000000.0,
        cell_size: 30.0,
        data_type: terrio::DataType::F32,
        ..Default::default()
    };
    let v = (0..cols * rows).map(|i| ((i % cols) + 3 * (i / cols)) as f64).collect();
    let mut r = terrio::RasterDataset::new(terrio::Raster::from_data(cfg, v).unwrap(), false);
    r.raster.crs.epsg = Some(32611);
    terrio::write_raster_dataset(&r, d.join("utm.tif").to_str().unwrap(), &Default::default()).unwrap();

    let warp = |out: &str, extra: &[&str]| {
        let mut args = vec!["warp", "utm.tif", out, "--t-srs", "EPSG:4326", "-r", "bilinear"];
        args.extend_from_slice(extra);
        run(&args, d)
    };
    for (out, extra) in [
        ("default.tif", &[][..]),
        ("64.tif", &["--wm", "64"][..]),
        ("64mb.tif", &["--wm", "64MB"][..]),
        ("bytes.tif", &["--wm", "67108864"][..]),
        ("zero.tif", &["--wm", "0"][..]),
        ("small.tif", &["--wm", "20000"][..]),
    ] {
        let r = warp(out, extra);
        assert!(r.status.success(), "{extra:?}: {}", String::from_utf8_lossy(&r.stderr));
    }
    // 64 (megabytes below 10000), 64MB and 67108864 bytes are the default, and 0 keeps it.
    let default = values(&d.join("default.tif"));
    for out in ["64.tif", "64mb.tif", "bytes.tif", "zero.tif"] {
        assert_eq!(values(&d.join(out)), default, "{out}");
    }
    // 20000 bytes warps in many chunks, of the same grid.
    assert_eq!(values(&d.join("small.tif")).len(), default.len());
    for bad in ["-1", "nan", "64X", "101%"] {
        let r = warp("bad.tif", &["--wm", bad]);
        assert!(!r.status.success(), "{bad}");
        assert!(String::from_utf8_lossy(&r.stderr).contains("warp memory"), "{bad}: {}", String::from_utf8_lossy(&r.stderr));
    }
}
