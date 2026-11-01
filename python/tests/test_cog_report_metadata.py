"""Area-matched overviews: the report in the default metadata domain, and NaN under a numeric
nodata counted as valid (rasterio's mask, as ras2cng's NumPy builder reads a band)."""

import numpy as np
import pytest

import terrio

TRANSFORM = (500000.0, 10.0, 0.0, 4000000.0, 0.0, -10.0)
REPORT_NAMES = {
    "area_ratios": "overview_area_ratios",
    "base_valid_pixels": "overview_base_valid_pixels",
}


def river(width=1300, height=1100, nodata=-9999.0):
    y, x = np.mgrid[0:height, 0:width]
    a = (1.0 + x % 10 + y / 7.0).astype("float32")
    a[np.abs(x - y) > 3] = nodata
    return a


def raster(a, nodata=-9999.0):
    return terrio.Raster.from_array(a, TRANSFORM, "EPSG:32610", nodata)


def overviews(path):
    r = terrio.open_raster(str(path), lazy=True)
    return [np.asarray(r.read(overview=i))[0] for i in range(len(r.overviews))]


def test_report_metadata_in_the_default_domain(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    rng = np.random.default_rng(7)
    seen = set()
    # Random wet masks in 16-pixel tiles: four levels each.
    for i in range(40):
        a = rng.uniform(0.0, 9.0, (150 + i, 200 - i)).astype("float32")
        a[rng.uniform(size=a.shape) < rng.uniform(0.05, 0.95)] = -9999.0
        out = tmp_path / f"r{i}.tif"
        report = raster(a).write(
            str(out),
            driver="COG",
            compress="DEFLATE",
            options={"BLOCKSIZE": "16"},
            overview_resampling="area_matched",
            report_metadata=REPORT_NAMES,
            metadata={"overview_method": "area_matched_mean"},
        )
        with rasterio.open(out) as src:
            tags = src.tags()
        # Exactly ras2cng's OverviewReport.as_tags() of the returned (exact) ratios.
        expected = ",".join(f"{r:.6f}" for r in report["area_ratios"][0])
        assert tags["overview_area_ratios"] == expected
        assert tags["overview_base_valid_pixels"] == str(report["base_valid_pixels"][0])
        assert tags["overview_method"] == "area_matched_mean"
        seen.update(expected.split(","))
    assert len(seen) > 100


def test_report_metadata_of_several_bands(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    a = np.stack([river(), river()[:, ::-1].copy()])
    a[1, :400] = -9999.0
    out = tmp_path / "two.tif"
    report = terrio.Raster.from_array(a, TRANSFORM, "EPSG:32610", -9999.0).write(
        str(out), driver="COG", overview_resampling="area_matched", report_metadata=REPORT_NAMES
    )
    with rasterio.open(out) as src:
        tags = src.tags()
    bands = [",".join(f"{r:.6f}" for r in b) for b in report["area_ratios"]]
    assert tags["overview_area_ratios"] == ";".join(bands)
    assert tags["overview_base_valid_pixels"] == ";".join(map(str, report["base_valid_pixels"]))


def test_report_metadata_refusals(tmp_path):
    a = river()
    out = str(tmp_path / "x.tif")
    bad = [
        # A name the caller's metadata also sets.
        dict(report_metadata=REPORT_NAMES, metadata={"overview_area_ratios": "1.0"}),
        dict(report_metadata={"area_ratio": "x"}),
        dict(report_metadata={"area_ratios": "x", "base_valid_pixels": "x"}),
        dict(report_metadata={"area_ratios": ""}),
    ]
    for kwargs in bad:
        with pytest.raises(terrio.TerrioError):
            raster(a).write(out, driver="COG", overview_resampling="area_matched", **kwargs)
    # Only with area-matched overviews.
    for kwargs in (dict(report_metadata=REPORT_NAMES), dict(area_matched_nan_valid=True)):
        with pytest.raises(terrio.TerrioError):
            raster(a).write(out, driver="COG", overview_resampling="average", **kwargs)
    # Nothing to store without levels.
    small = raster(river(100, 80)).write(
        out, driver="COG", overview_resampling="area_matched", report_metadata=REPORT_NAMES
    )
    assert small is None
    assert not any(k.startswith("overview_") for k, _ in terrio.open_raster(out).metadata)


# ras2cng's `build_area_matched_overviews` (CLB Engineering, MIT) on a whole band, reading it as
# rasterio's mask does: a numeric nodata leaves NaN valid.


def _dither(indices, salt):
    x = indices.astype(np.uint64) ^ np.uint64(salt & 0xFFFFFFFFFFFFFFFF)
    x = (x * np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= x >> np.uint64(29)
    x = (x * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= x >> np.uint64(32)
    return x


def _select(weights, factor, base_valid):
    capacity = factor * factor
    target = int(round(base_valid / capacity))
    wet = weights > 0
    if target <= 0:
        return np.zeros_like(wet)
    if not wet.any() or target >= int(wet.sum()):
        return wet
    counts = np.bincount(weights.ravel(), minlength=capacity + 1)
    accumulated = threshold = 0
    for level in range(capacity, 0, -1):
        if accumulated + int(counts[level]) >= target:
            threshold = level
            break
        accumulated += int(counts[level])
    keep = weights > threshold
    remaining = target - accumulated
    if remaining <= 0:
        return keep
    tie = weights == threshold
    tie_indices = np.flatnonzero(tie.ravel())
    if remaining >= tie_indices.size:
        return keep | tie
    coherence = np.zeros(weights.shape, dtype=np.int16)
    coherence[1:, :] += keep[:-1, :]
    coherence[:-1, :] += keep[1:, :]
    coherence[:, 1:] += keep[:, :-1]
    coherence[:, :-1] += keep[:, 1:]
    order = np.lexsort((_dither(tie_indices, factor * 0x2545F491), coherence.ravel()[tie_indices]))[
        ::-1
    ]
    flat = keep.ravel()
    flat[tie_indices[order[:remaining]]] = True
    return flat.reshape(weights.shape)


def _reduce_sum(a):
    h, w = a.shape
    if (h % 2, w % 2) != (0, 0):
        p = np.zeros((h + h % 2, w + w % 2), dtype=a.dtype)
        p[:h, :w] = a
        a = p
    return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).sum(axis=(1, 3))


def ras2cng_mean_levels(a, nodata, count):
    valid = ~np.isnan(a) if np.isnan(nodata) else a != nodata
    sums = _reduce_sum(np.where(valid, a.astype("float64"), 0.0))
    weights = _reduce_sum(valid.astype("int64"))
    base_valid = int(valid.sum())
    levels = []
    for index in range(count):
        factor = 2 ** (index + 1)
        if index:
            sums, weights = _reduce_sum(sums), _reduce_sum(weights)
        keep = _select(weights, factor, base_valid)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(weights > 0, sums / np.maximum(weights, 1), nodata)
        levels.append(np.where(keep, means, nodata).astype(a.dtype))
    return levels, base_valid


def nan_river(nodata=-9999.0):
    a = river(nodata=nodata)
    rng = np.random.default_rng(11)
    a[rng.uniform(size=a.shape) < 0.01] = np.nan
    # A NaN of another sign and payload, and a block of NaN only.
    a[5, 7] = np.array(0xFFC00123, "uint32").view("float32")
    a[200:204, 198:206] = np.nan
    return a


def test_nan_valid_area_matched_matches_ras2cng_bit_for_bit(tmp_path):
    a = nan_river()
    out = tmp_path / "nan.tif"
    report = raster(a).write(
        str(out),
        driver="COG",
        compress="DEFLATE",
        overview_resampling="area_matched",
        area_matched_nan_valid=True,
    )
    want, base_valid = ras2cng_mean_levels(a, -9999.0, len(report["factors"]))
    assert report["base_valid_pixels"] == [base_valid]
    # The NaN under the numeric nodata, counted in the same pass (ras2cng's NaN check).
    assert report["nan_pixels"] == [int(np.isnan(a).sum())]
    got = overviews(out)
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        assert np.isnan(g).any()
        np.testing.assert_array_equal(g.view("uint32"), w.view("uint32"))
    # The full resolution is untouched.
    full = np.asarray(terrio.open_raster(str(out)).read())[0]
    np.testing.assert_array_equal(full.view("uint32"), a.view("uint32"))
    # By default NaN is invalid: fewer valid pixels, and no NaN reaches the levels.
    report = raster(a).write(
        str(out), driver="COG", compress="DEFLATE", overview_resampling="area_matched"
    )
    assert report["base_valid_pixels"] == [int(((a != -9999.0) & ~np.isnan(a)).sum())]
    assert report["nan_pixels"] == [int(np.isnan(a).sum())]
    assert not any(np.isnan(o).any() for o in overviews(out))
    # Under a NaN nodata, NaN stays nodata: the option changes nothing.
    b = np.where(a == -9999.0, np.float32(np.nan), a)
    plain, valid = tmp_path / "p.tif", tmp_path / "v.tif"
    raster(b, np.nan).write(
        str(plain), driver="COG", compress="DEFLATE", overview_resampling="area_matched"
    )
    raster(b, np.nan).write(
        str(valid),
        driver="COG",
        compress="DEFLATE",
        overview_resampling="area_matched",
        area_matched_nan_valid=True,
    )
    for p, v in zip(overviews(plain), overviews(valid), strict=True):
        np.testing.assert_array_equal(p.view("uint32"), v.view("uint32"))


def test_nan_valid_with_lerc_reads_back_nan(tmp_path):
    a = nan_river()
    out = tmp_path / "lerc.tif"
    raster(a).write(
        str(out),
        driver="COG",
        compress="LERC_DEFLATE",
        options={"MAX_Z_ERROR": "0.01"},
        overview_resampling="area_matched",
        area_matched_nan_valid=True,
    )
    want, _ = ras2cng_mean_levels(a, -9999.0, 2)
    readers = [overviews(out)]
    try:
        import rasterio
    except ImportError:
        rasterio = None
    if rasterio is not None:
        with rasterio.open(out) as src:
            shapes = [(-(-src.height // f), -(-src.width // f)) for f in src.overviews(1)]
            readers.append([src.read(1, out_shape=s) for s in shapes])
    for got in readers:
        assert len(got) == len(want)
        for g, w in zip(got, want, strict=True):
            # NaN where the lossless levels hold NaN (masked LERC pixels read back as NaN, without
            # their sign or payload), nodata where they hold nodata, values within MAX_Z_ERROR.
            np.testing.assert_array_equal(np.isnan(g), np.isnan(w))
            np.testing.assert_array_equal(g == -9999.0, w == -9999.0)
            finite = ~np.isnan(w) & (w != -9999.0)
            assert np.abs(g[finite] - w[finite]).max() <= 0.01 + 1e-6
