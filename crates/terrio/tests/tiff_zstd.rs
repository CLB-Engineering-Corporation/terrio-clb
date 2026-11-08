//! Zstandard GeoTIFFs of the layouts the `tiff` crate decodes with the zstd C library (the
//! `libzstd-geotiff` feature, on by default) and terrio decodes itself without it: Float16 samples (with each predictor),
//! unsigned samples of 1 to 7 bits, and WhiteIsZero images. Each file of `tests/data/tiff_zstd`
//! (`scripts/make_tiff_zstd_fixtures.py`, written by GDAL) has a DEFLATE twin, which the crate
//! decodes in every build: both read to the same values, or both are refused.

use std::path::{Path, PathBuf};

/// The data type, bands and value bits of a raster, or the error reading it.
fn read(path: &Path) -> Result<(String, usize, Vec<u64>), String> {
    let ds = terrio::open_raster(path.to_str().unwrap()).map_err(|e| e.to_string())?;
    let r = &ds.raster;
    let bits = (0..r.bands).flat_map(|b| r.band_view(b).as_slice().iter().map(|v| v.to_bits()).collect::<Vec<_>>()).collect();
    Ok((format!("{:?}", r.data_type), r.bands, bits))
}

#[test]
fn zstd_chunks_read_like_their_deflate_twins() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/tiff_zstd");
    let mut zstd: Vec<PathBuf> =
        std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.to_string_lossy().contains("_zstd_")).collect();
    zstd.sort();
    assert_eq!(zstd.len(), 30);
    let (mut read_ok, mut refused) = (Vec::new(), Vec::new());
    for z in &zstd {
        let name = z.file_name().unwrap().to_string_lossy().to_string();
        let deflate = dir.join(name.replace("_zstd_", "_deflate_"));
        match (read(z), read(&deflate)) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a.0, b.0, "{name}: data type");
                assert_eq!(a.1, b.1, "{name}: bands");
                assert_eq!(a.2.len(), b.2.len(), "{name}: samples");
                let differ = a.2.iter().zip(&b.2).filter(|(x, y)| x != y).count();
                assert_eq!(differ, 0, "{name}: {differ} samples differ");
                read_ok.push(name);
            }
            (Err(_), Err(_)) => refused.push(name),
            (a, b) => panic!("{name}: ZSTD {:?}, DEFLATE {:?}", a.map(|_| ()), b.map(|_| ())),
        }
    }
    // The crate refuses to invert a two-sample WhiteIsZero image, and decodes no RGB image of
    // fewer than 8 bits: the same files are refused with either compression. With the `gdal`
    // feature, GDAL reads what terrio refuses, both files of each pair.
    let want: &[&str] = if cfg!(feature = "gdal") {
        &[]
    } else {
        &["miniswhite_u8x2_zstd_strips.tif", "miniswhite_u8x2_zstd_tiled.tif", "u4x3_zstd_strips.tif", "u4x3_zstd_tiled.tif"]
    };
    assert_eq!(refused, want, "{read_ok:?}");
    assert_eq!(read_ok.len(), 30 - want.len());
}
