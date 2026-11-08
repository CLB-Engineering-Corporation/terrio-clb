"""Writes the GeoTIFFs of `crates/terrio/tests/data/tiff_zstd`: the TIFF layouts whose Zstandard
chunks terrio decodes itself without the zstd C library, but which the `tiff` crate decodes with
it: Float16 samples, unsigned samples of 1 to 7 bits, and WhiteIsZero images. Each is written twice
by GDAL, with ZSTD and with DEFLATE (which the crate decodes in every build), tiled with partial
edge tiles and in strips, so that a test can read both and compare.

Usage:
    python scripts/make_tiff_zstd_fixtures.py
"""

from pathlib import Path

import numpy as np
import rasterio

OUT = Path(__file__).parent.parent / "crates/terrio/tests/data/tiff_zstd"
COLS, ROWS = 37, 29


def values(dtype, bands=1):
    """Deterministic values over the full range of `dtype` (floats in [-3, 3])."""
    i = np.arange(COLS * ROWS * bands).reshape(bands, ROWS, COLS)
    if np.dtype(dtype).kind == "f":
        return (np.sin(i * 0.37) * 3).astype(dtype)
    info = np.iinfo(dtype)
    return ((i * 2654435761) % (int(info.max) + 1)).astype(dtype)


def write(name, data, *, nbits=None, photometric=None, predictor=1):
    """Writes `data` as `<name>_zstd_*.tif` and `<name>_deflate_*.tif`, tiled and in strips."""
    for compress in ("ZSTD", "DEFLATE"):
        for layout in ("tiled", "strips"):
            options = {"compress": compress, "predictor": predictor}
            if layout == "tiled":
                options.update(tiled=True, blockxsize=16, blockysize=16)
            else:
                options.update(blockysize=5)
            if nbits:
                options["nbits"] = nbits
            if photometric:
                options["photometric"] = photometric
            path = OUT / f"{name}_{compress.lower()}_{layout}.tif"
            profile = dict(driver="GTiff", width=COLS, height=ROWS, count=data.shape[0], dtype=data.dtype, **options)
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(data)


def main():
    """Writes the files."""
    OUT.mkdir(parents=True, exist_ok=True)
    f32 = values("float32")
    # Float16 (NBITS=16 of Float32), with each predictor.
    for predictor in (1, 2, 3):
        write(f"f16_p{predictor}", f32, nbits=16, predictor=predictor)
    # Unsigned samples of 1 to 7 bits, one and three samples per pixel.
    for bits in (1, 2, 3, 4, 7):
        data = (values("uint8") % (1 << bits)).astype("uint8")
        write(f"u{bits}", data, nbits=bits)
    write("u4x3", (values("uint8", 3) % 16).astype("uint8"), nbits=4)
    # WhiteIsZero.
    write("miniswhite_u1", (values("uint8") % 2).astype("uint8"), nbits=1, photometric="MINISWHITE")
    write("miniswhite_u8", values("uint8"), photometric="MINISWHITE", predictor=2)
    write("miniswhite_u16", values("uint16"), photometric="MINISWHITE", predictor=2)
    write("miniswhite_f32", f32, photometric="MINISWHITE", predictor=3)
    write("miniswhite_f64", values("float64"), photometric="MINISWHITE")
    # Two samples: the crate refuses to invert them, with either compression.
    write("miniswhite_u8x2", values("uint8", 2), photometric="MINISWHITE")


if __name__ == "__main__":
    main()
