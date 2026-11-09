"""Tests on small data made in the tests, that always run."""

import json

import numpy as np
import pytest

import terrio

TRANSFORM = (500000.0, 30.0, 0.0, 4100000.0, 0.0, -30.0)

POLYGONS = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "west", "value": 3, "area": 1.5},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
            },
        },
        {
            "type": "Feature",
            "properties": {"name": "east", "value": 7, "area": None},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[20, 0], [30, 0], [30, 10], [20, 10], [20, 0]]],
            },
        },
    ],
}


def make_raster(dtype="int16", bands=2, nodata=-1):
    data = np.arange(bands * 6 * 8, dtype=dtype).reshape(bands, 6, 8)
    return data, terrio.Raster.from_array(data, TRANSFORM, crs="EPSG:32610", nodata=nodata)


def test_version():
    assert isinstance(terrio.__version__, str)


@pytest.mark.parametrize(
    "dtype", ["uint8", "int8", "uint16", "int16", "uint32", "int32", "int64", "float32", "float64"]
)
def test_from_array_properties(dtype):
    data, r = make_raster(dtype)
    assert (r.count, r.height, r.width) == (2, 6, 8)
    assert r.shape == data.shape
    assert r.dtype == dtype
    assert r.nodata == -1
    assert r.crs == "EPSG:32610"
    assert "UTM zone 10N" in r.crs_wkt
    assert r.transform == TRANSFORM
    assert r.res == (30.0, 30.0)
    assert r.bounds == (500000.0, 4100000.0 - 6 * 30, 500000.0 + 8 * 30, 4100000.0)
    assert not r.is_lazy
    out = r.read()
    assert out.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(out, data)
    np.testing.assert_array_equal(np.asarray(r), data)


def test_read_bands_and_windows():
    data, r = make_raster()
    np.testing.assert_array_equal(r.read(2), data[1])
    np.testing.assert_array_equal(r.read([2, 1]), data[[1, 0]])
    np.testing.assert_array_equal(r.read(1, window=(2, 1, 3, 4)), data[0, 1:5, 2:5])
    with pytest.raises(terrio.InvalidArgumentError):
        r.read(3)
    with pytest.raises(ValueError):
        r.read(window=(6, 0, 4, 1))
    with pytest.raises(terrio.InvalidArgumentError):
        r.read(overview=0)


def test_from_array_2d_and_bool():
    r = terrio.Raster.from_array(np.ones((3, 4), dtype=bool), TRANSFORM)
    assert r.count == 1
    assert r.dtype == "uint8"
    assert r.crs is None
    assert r.nodata is None
    assert r.read(1).sum() == 12


def test_from_array_rejects_bad_input():
    with pytest.raises(terrio.InvalidArgumentError):
        terrio.Raster.from_array(np.zeros(5), TRANSFORM)
    with pytest.raises(terrio.InvalidArgumentError):
        terrio.Raster.from_array(np.zeros((2, 2)), (0, 1, 2, 0, 0.5, 1))
    with pytest.raises(terrio.InvalidArgumentError):
        terrio.Raster.from_array(np.zeros((2, 2), dtype=np.complex64), TRANSFORM)
    with pytest.raises(terrio.CRSError):
        terrio.Raster.from_array(np.zeros((2, 2)), TRANSFORM, crs="not a crs")


@pytest.mark.parametrize("ext", ["tif", "asc"])
def test_write_and_reopen(tmp_path, ext):
    data, r = make_raster("float32", bands=1 if ext == "asc" else 2)
    path = tmp_path / f"out.{ext}"
    r.write(str(path))
    back = terrio.open(str(path))
    assert isinstance(back, terrio.Raster)
    np.testing.assert_allclose(back.read(), data)
    assert back.transform == pytest.approx(TRANSFORM)


def test_write_compressed_tiled_and_lazy(tmp_path):
    data, r = make_raster("uint16")
    path = tmp_path / "tiled.tif"
    terrio.write_raster(r, str(path), compress="DEFLATE", tile_size=16)
    lazy = terrio.open_raster(str(path), lazy=True)
    assert lazy.is_lazy
    assert lazy.nodata == -1 or lazy.nodata is not None
    np.testing.assert_array_equal(lazy.read(2, window=(1, 1, 3, 2)), data[1, 1:3, 1:4])
    assert "lazy" in repr(lazy)


def test_bytes_round_trip():
    data, r = make_raster("int32")
    tif = r.to_bytes("x.tif")
    assert tif[:4] in (b"II*\x00", b"MM\x00*")
    back = terrio.open_bytes(tif, "x.tif")
    np.testing.assert_array_equal(back.read(), data)
    lazy = terrio.open_bytes(tif, "x.tif", lazy=True)
    assert lazy.is_lazy
    np.testing.assert_array_equal(lazy.read(1, window=(0, 0, 2, 2)), data[0, :2, :2])


def test_vector_from_geojson_and_back():
    layer = terrio.Layer.from_geojson(POLYGONS, name="squares")
    assert layer.name == "squares"
    assert len(layer) == 2
    assert layer.crs == "EPSG:4326"
    assert layer.geometry_type == "Polygon"
    assert layer.fields == {"name": "Text", "value": "Integer", "area": "Float"}
    assert layer.bounds == (0, 0, 30, 10)
    cols = layer.columns()
    assert cols["name"] == ["west", "east"]
    assert cols["area"] == [1.5, None]
    fc = layer.to_geojson()
    assert fc["type"] == "FeatureCollection"
    assert [f["properties"]["value"] for f in fc["features"]] == [3, 7]
    assert layer.__geo_interface__["type"] == "FeatureCollection"
    wkb = layer.to_wkb()
    assert len(wkb) == 2 and all(isinstance(w, bytes) for w in wkb)
    # From a JSON string too.
    assert len(terrio.Layer.from_geojson(json.dumps(POLYGONS))) == 2
    layer.name = "renamed"
    assert layer.name == "renamed"


@pytest.mark.parametrize("ext", ["geojson", "fgb", "gpkg", "kml"])
def test_write_vector_formats(tmp_path, ext):
    path = tmp_path / f"squares.{ext}"
    terrio.write_vector(POLYGONS, str(path))
    ds = terrio.open_vector(str(path))
    assert isinstance(ds, terrio.VectorDataset)
    layer = ds[0]
    assert len(layer) == 2
    assert layer.name == "squares"
    assert terrio.list_layers(str(path)) == ["squares"]


def test_vector_dataset_access(tmp_path):
    a = terrio.Layer.from_geojson(POLYGONS, name="a")
    b = terrio.Layer.from_geojson(POLYGONS, name="b")
    path = tmp_path / "two.gpkg"
    terrio.write_vector([a, b], str(path))
    ds = terrio.open_vector(str(path))
    assert ds.layer_names == ["a", "b"]
    assert ds.driver == "GPKG"
    assert len(ds) == 2
    assert ds["b"].name == "b"
    assert ds[-1].name == "b"
    assert [layer.name for layer in ds] == ["a", "b"]
    assert [layer.name for layer in ds.layers] == ["a", "b"]
    with pytest.raises(KeyError):
        ds["c"]
    with pytest.raises(IndexError):
        ds[2]
    only_b = terrio.open_vector(str(path), layers=["b"])
    assert only_b.layer_names == ["b"]
    with pytest.raises(terrio.InvalidArgumentError):
        terrio.write_vector(ds, str(tmp_path / "two.geojson"))


def test_layer_bytes_and_to_crs():
    layer = terrio.Layer.from_geojson(POLYGONS, name="squares")
    fgb = layer.to_bytes("squares.fgb")
    back = terrio.open_bytes(fgb, "squares.fgb")
    assert len(back[0]) == 2
    projected = layer.to_crs("EPSG:3857")
    assert projected.crs == "EPSG:3857"
    assert projected.bounds[2] == pytest.approx(30 * 20037508.342789244 / 180)


def test_convert_with_filters(tmp_path):
    src = tmp_path / "squares.geojson"
    terrio.write_vector(POLYGONS, str(src))
    dst = tmp_path / "east.gpkg"
    terrio.convert(str(src), str(dst), where="value > 5", select=["name"], t_srs="EPSG:32631")
    layer = terrio.open_vector(str(dst))[0]
    assert len(layer) == 1
    assert layer.fields == {"name": "Text"}
    assert layer.crs == "EPSG:32631"
    with pytest.raises(terrio.InvalidArgumentError):
        terrio.convert(str(src), str(tmp_path / "x.fgb"), nlt="NOT_A_TYPE")


def test_convert_raster_window(tmp_path):
    data, r = make_raster("uint8")
    src = tmp_path / "src.tif"
    r.write(str(src))
    dst = tmp_path / "win.tif"
    terrio.convert(str(src), str(dst), srcwin=(2, 1, 4, 3))
    np.testing.assert_array_equal(terrio.open_raster(str(dst)).read(), data[:, 1:4, 2:6])
    with pytest.raises(terrio.InvalidArgumentError):
        terrio.convert(str(src), str(tmp_path / "x.tif"), where="a = 1")


def test_rasterize_and_polygonize(tmp_path):
    layer = terrio.Layer.from_geojson(POLYGONS, name="squares", crs="EPSG:32610")
    r = terrio.rasterize(layer, attribute="value", tr=(1, 1), ot="Byte")
    # As gdal_rasterize -tr 1 1 makes it.
    assert (r.width, r.height) == (31, 11)
    a = r.read(1)
    assert a.dtype == np.uint8
    assert (a == 3).sum() == (a == 7).sum() > 0
    out = tmp_path / "burned.tif"
    terrio.rasterize(layer, str(out), burn=1, ts=(15, 5), nodata=255)
    assert 0 < terrio.open_raster(str(out)).read(1).sum() < 75
    polys = terrio.polygonize(r, field="v")
    assert polys.geometry_type == "Polygon"
    assert sorted(polys.columns()["v"]) == [3, 7]
    path = tmp_path / "polys.gpkg"
    terrio.polygonize(r, str(path), connectedness=8)
    assert len(terrio.open_vector(str(path))[0]) == 2
    with pytest.raises(terrio.InvalidArgumentError):
        terrio.rasterize(layer, tr=(1, 1))


def test_info_and_drivers(tmp_path):
    _, r = make_raster()
    path = tmp_path / "i.tif"
    r.write(str(path))
    summary = terrio.info(str(path), stats=True)
    assert summary["size"] == [8, 6]
    assert summary["band_count"] == 2
    assert summary["bands"][0]["statistics"]["min"] == 0
    names = {d["name"] for d in terrio.drivers()}
    assert {"GTiff", "COG", "GPKG", "GeoJSON"} <= names


def test_transform_points():
    xs, ys = terrio.transform_points([0, 180], [0, 0], "EPSG:4326", "EPSG:3857")
    assert xs == pytest.approx([0, 20037508.342789244])
    assert ys == pytest.approx([0, 0], abs=1e-6)
    x, y = terrio.transform_points(np.array([699960.0]), [3900000], "EPSG:32610", "EPSG:4326")
    assert x[0] == pytest.approx(-120.8029, abs=1e-4)
    assert y[0] == pytest.approx(35.2231, abs=1e-4)
    with pytest.raises(terrio.InvalidArgumentError):
        terrio.transform_points([1, 2], [1], "EPSG:4326", "EPSG:3857")


def test_transform_points_with_heights_and_epochs():
    """Heights come back with `zs`; coordinate epochs drive time-dependent transformations."""
    # Heights are kept between 2D CRSs, as GDAL keeps them.
    xs, ys, zs = terrio.transform_points([0], [0], "EPSG:4326", "EPSG:3857", [100.0])
    assert zs[0] == 100.0
    # ITRF2014 coordinates at epoch 2010.0 to GDA2020, a time-dependent transformation: GDAL's
    # result with the source CRS's coordinate epoch set.
    xs, ys, zs = terrio.transform_points(
        [135.0], [-25.0], "EPSG:9000", "EPSG:7844", [0.0], s_coord_epoch=2010.0
    )
    assert xs[0] == pytest.approx(135.0000030617912, abs=1e-11)
    assert ys[0] == pytest.approx(-24.999994698869124, abs=1e-11)
    assert zs[0] == pytest.approx(-0.0015074610710144043, abs=1e-6)
    xs, ys = terrio.transform_points([135.0], [-25.0], "EPSG:9000", "EPSG:7844")
    assert xs[0] == pytest.approx(135.0, abs=1e-11)
    with pytest.raises(terrio.InvalidArgumentError):
        terrio.transform_points([1, 2], [1, 2], "EPSG:4326", "EPSG:3857", [1])


def test_errors():
    with pytest.raises(terrio.NotFoundError) as e:
        terrio.open("does/not/exist.tif")
    assert isinstance(e.value, FileNotFoundError)
    assert isinstance(e.value, OSError)
    assert isinstance(e.value, terrio.TerrioError)
    with pytest.raises(terrio.CRSError) as e:
        terrio.transform_points([0], [0], "EPSG:4326", "nonsense")
    assert isinstance(e.value, ValueError)
    with pytest.raises(terrio.UnsupportedError):
        terrio.open_bytes(b"not a dataset", "x.unknown")


def test_to_geopandas():
    gpd = pytest.importorskip("geopandas")
    layer = terrio.Layer.from_geojson(POLYGONS, name="squares")
    gdf = layer.to_geopandas()
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert list(gdf["name"]) == ["west", "east"]
    assert gdf.crs.to_epsg() == 4326
    assert gdf.geometry.iloc[0].area == 100
    # And back, keeping the CRS.
    back = terrio.Layer.from_geojson(gdf.to_crs(3857))
    assert back.crs == "EPSG:3857"


def test_translate():
    data, r = make_raster("int16")
    out = terrio.translate(r, srcwin=(2, 1, 4, 3), bands=[2], ot="Float32")
    assert out.dtype == "float32"
    np.testing.assert_array_equal(out.read(1), data[1, 1:4, 2:6].astype("float32"))
    assert out.transform == pytest.approx((500060.0, 30.0, 0.0, 4099970.0, 0.0, -30.0))
    half = terrio.translate(r, outsize=("50%", "50%"), resampling="average")
    assert (half.width, half.height) == (4, 3)
    scaled = terrio.translate(r, scale=(0, 95, 0, 255), ot="Byte", a_nodata="none")
    assert scaled.nodata is None
    assert scaled.read(2).max() == 255


def test_warp_and_mosaic(tmp_path):
    data, r = make_raster("float32", bands=1)
    geo = terrio.warp(r, t_srs="EPSG:4326")
    assert geo.crs == "EPSG:4326"
    west, south, east, north = geo.bounds
    assert -123.01 < west < east < -122.99 and 37.0 < south < north < 37.1
    out = tmp_path / "w.tif"
    terrio.warp([r], str(out), tr=(15, 15), resampling="bilinear")
    assert terrio.open_raster(str(out)).res == (15, 15)
    # Split in two and mosaic back.
    left = terrio.translate(r, srcwin=(0, 0, 4, 6))
    right = terrio.translate(r, srcwin=(4, 0, 4, 6))
    whole = terrio.mosaic([left, right])
    np.testing.assert_array_equal(whole.read(1), data[0])
    assert whole.transform == pytest.approx(TRANSFORM)
    paths = []
    for name, part in [("l.tif", left), ("r.tif", right)]:
        part.write(str(tmp_path / name))
        paths.append(str(tmp_path / name))
    assert terrio.mosaic(paths, str(tmp_path / "m.vrt")) is None
    np.testing.assert_array_equal(terrio.open_raster(str(tmp_path / "m.vrt")).read(1), data[0])
    with pytest.raises(terrio.InvalidArgumentError):
        terrio.mosaic([left, right], str(tmp_path / "x.vrt"))


def test_warp_and_mosaic_paths_are_written_as_the_cli_writes_them(tmp_path):
    """A source path warped (or source paths mosaicked) to a path is written by the command
    line's path-to-path writer, with the values of the warp in memory, and returned lazily."""
    rows, cols = np.mgrid[0:90, 0:70]
    data = ((rows * 7 + cols * 13) % 251).astype("float32")
    src = tmp_path / "src.tif"
    terrio.Raster.from_array(data, TRANSFORM, crs="EPSG:32610", nodata=0).write(str(src))
    kw = {"t_srs": "EPSG:4326", "resampling": "bilinear"}
    for wm in [None, 64, "64MB", 20000, "0.02MB"]:
        extra = {} if wm is None else {"wm": wm}
        whole = terrio.warp(terrio.open_raster(str(src)), **kw, **extra)
        out = terrio.warp(str(src), dst=str(tmp_path / "w.tif"), **kw, **extra)
        assert out.is_lazy
        np.testing.assert_array_equal(out.read(), whole.read())
        assert out.transform == pytest.approx(whole.transform) and out.nodata == whole.nodata
    # 64 is megabytes, 67108864 bytes: both are the default.
    default = terrio.warp(str(src), **kw).read()
    np.testing.assert_array_equal(terrio.warp(str(src), wm=67108864, **kw).read(), default)
    for bad in [-1, "64X", True, "101%"]:
        with pytest.raises((terrio.InvalidArgumentError, ValueError)):
            terrio.warp(str(src), wm=bad, **kw)
    # The CLB build's max_memory_mib (MiB) is the same working memory; the two exclude each other.
    for mib in [16, 0.0625, 0.015625]:
        np.testing.assert_array_equal(
            terrio.warp(str(src), dst=str(tmp_path / "mib.tif"), max_memory_mib=mib, **kw).read(),
            terrio.warp(str(src), dst=str(tmp_path / "wm.tif"), wm=f"{mib * 1024}K", **kw).read(),
        )
    with pytest.raises((terrio.InvalidArgumentError, ValueError)):
        terrio.warp(str(src), wm=16, max_memory_mib=16, **kw)
    # Several sources and a source path given with a Raster.
    two = terrio.warp([str(src), str(src)], dst=str(tmp_path / "w2.tif"), **kw)
    assert two.is_lazy
    np.testing.assert_array_equal(two.read(), default)
    mixed = terrio.warp(
        [str(src), terrio.open_raster(str(src))], dst=str(tmp_path / "w3.tif"), **kw
    )
    assert not mixed.is_lazy
    # Mosaic: sources on the output grid, and one at twice its resolution.
    left = terrio.translate(str(src), srcwin=(0, 0, 40, 90))
    left.write(str(tmp_path / "l.tif"))
    srcs = [str(tmp_path / "l.tif"), str(src)]
    ref = terrio.mosaic([terrio.open_raster(p) for p in srcs])
    m = terrio.mosaic(srcs, str(tmp_path / "m.tif"))
    assert m.is_lazy
    np.testing.assert_array_equal(m.read(), ref.read())
    coarse = terrio.Raster.from_array(
        data[::2, ::2].copy(),
        (TRANSFORM[0], TRANSFORM[1] * 2, 0.0, TRANSFORM[3], 0.0, TRANSFORM[5] * 2),
        crs="EPSG:32610",
        nodata=0,
    )
    coarse.write(str(tmp_path / "c.tif"))
    srcs = [str(tmp_path / "c.tif"), str(src)]
    ref = terrio.mosaic([terrio.open_raster(p) for p in srcs], resolution="highest")
    m = terrio.mosaic(srcs, str(tmp_path / "m2.tif"), resolution="highest")
    assert m.is_lazy
    np.testing.assert_array_equal(m.read(), ref.read())


def test_dem_products(tmp_path):
    """Checks hillshade, slope, aspect and color relief on a tilted plane."""
    cols = np.arange(8, dtype="float32")
    plane = np.tile(cols * 30.0, (6, 1))  # rises 1 m per metre eastward
    r = terrio.Raster.from_array(plane, TRANSFORM, crs="EPSG:32610", nodata=-9999)
    slope = terrio.dem(r, "slope").read(1)
    assert slope.dtype == np.float32
    assert slope[2, 3] == pytest.approx(45.0, abs=1e-4)
    assert slope[0, 0] == -9999
    aspect = terrio.dem(r, "aspect", trigonometric=True).read(1)
    assert aspect[2, 3] == pytest.approx(180.0)
    shade = terrio.dem(r, "hillshade", compute_edges=True, dst=str(tmp_path / "h.tif")).read(1)
    assert shade.dtype == np.uint8 and shade.min() > 0
    colors = tmp_path / "colors.txt"
    colors.write_text("0 0 0 255\n210 255 0 0\n")
    rgb = terrio.dem(r, "color-relief", color_file=str(colors), alpha=True)
    assert rgb.count == 4
    assert tuple(rgb.read()[:, 0, 7]) == (255, 0, 0, 255)
    with pytest.raises(terrio.InvalidArgumentError):
        terrio.dem(r, "slope", multidirectional=True)


def test_warp_and_mosaic_paths_are_streamed(tmp_path):
    """A source path warped (or source paths mosaicked) to a path is read by window and written
    a chunk at a time, with the values of the computation in memory, and returned lazily."""
    rows, cols = np.mgrid[0:90, 0:70]
    data = ((rows * 7 + cols * 13) % 251).astype("float32")
    src = tmp_path / "src.tif"
    terrio.Raster.from_array(data, TRANSFORM, crs="EPSG:32610", nodata=0).write(str(src))
    kw = {"t_srs": "EPSG:4326", "resampling": "bilinear"}
    whole = terrio.warp(terrio.open_raster(str(src)), **kw)
    out = terrio.warp(str(src), dst=str(tmp_path / "w.tif"), **kw)
    assert out.is_lazy
    np.testing.assert_array_equal(out.read(), whole.read())
    assert out.transform == pytest.approx(whole.transform) and out.nodata == whole.nodata
    # A budget only changes the chunks.
    small = terrio.warp(str(src), dst=str(tmp_path / "w2.tif"), max_memory_mib=0.01, **kw)
    assert small.read().shape == whole.read().shape
    with pytest.raises(terrio.InvalidArgumentError):
        terrio.warp(str(src), max_memory_mib=0, **kw)
    # Mosaic: a source at twice the resolution of the other.
    coarse = terrio.Raster.from_array(
        data[::2, ::2].copy(),
        (TRANSFORM[0], TRANSFORM[1] * 2, 0.0, TRANSFORM[3], 0.0, TRANSFORM[5] * 2),
        crs="EPSG:32610",
        nodata=0,
    )
    coarse.write(str(tmp_path / "c.tif"))
    srcs = [str(tmp_path / "c.tif"), str(src)]
    ref = terrio.mosaic([terrio.open_raster(p) for p in srcs], resolution="highest")
    m = terrio.mosaic(srcs, str(tmp_path / "m.tif"), resolution="highest")
    assert m.is_lazy
    np.testing.assert_array_equal(m.read(), ref.read())


@pytest.mark.parametrize("resampling", ["near", "bilinear", "average"])
def test_several_sources_warped_path_to_path_are_streamed(tmp_path, resampling):
    """Overlapping source paths (one in another CRS) warped or mosaicked to a path are warped a
    chunk at a time, each pixel taking its sources in order, bit for bit as in memory."""
    rows, cols = np.mgrid[0:90, 0:70]
    data = ((rows * 7 + cols * 13) % 251).astype("float32") + 0.5
    data[40:44, 5:60] = -9999
    a = tmp_path / "a.tif"
    terrio.Raster.from_array(data, TRANSFORM, crs="EPSG:32610", nodata=-9999).write(str(a))
    shifted = (TRANSFORM[0] + 611.0, 41.0, 0.0, TRANSFORM[3] - 377.0, 0.0, -41.0)
    seed = terrio.Raster.from_array(data[:50, :40] + 1000, shifted, crs="EPSG:32610", nodata=-9999)
    b = tmp_path / "b.tif"
    terrio.warp(seed, t_srs="EPSG:4326").write(str(b))
    srcs = [str(a), str(b)]
    for kw in [{}, {"t_srs": "EPSG:32611"}]:
        whole = terrio.warp([terrio.open_raster(p) for p in srcs], resampling=resampling, **kw)
        for budget in [None, 0.002]:
            out = terrio.warp(
                srcs,
                dst=str(tmp_path / "w.tif"),
                resampling=resampling,
                max_memory_mib=budget,
                **kw,
            )
            assert out.is_lazy
            if budget is None:
                assert np.asarray(out.read()).tobytes() == np.asarray(whole.read()).tobytes()
                assert out.transform == whole.transform and out.nodata == whole.nodata
            else:
                ref = terrio.warp(
                    [terrio.open_raster(p) for p in srcs],
                    resampling=resampling,
                    max_memory_mib=budget,
                    **kw,
                )
                assert np.asarray(out.read()).tobytes() == np.asarray(ref.read()).tobytes()
    ref = terrio.mosaic([terrio.open_raster(p) for p in srcs], resampling=resampling)
    m = terrio.mosaic(srcs, str(tmp_path / "m.tif"), resampling=resampling)
    assert m.is_lazy
    assert np.asarray(m.read()).tobytes() == np.asarray(ref.read()).tobytes()


@pytest.mark.parametrize("resampling", ["near", "bilinear", "cubic", "average", "mode"])
def test_resampled_mosaic_path_equals_the_whole_mosaic(tmp_path, resampling):
    """Sources at fractional offsets and resolutions are mosaicked to a path a strip at a time,
    bit for bit as the whole mosaic in memory."""
    rows, cols = np.mgrid[0:90, 0:70]
    data = ((rows * 7 + cols * 13) % 251).astype("float32") + 0.25
    data[30:33, 10:50] = -9999
    srcs = []
    for i, (dx, dy, res) in enumerate([(0.0, 0.0, 10.0), (37.3, -21.9, 14.6), (-12.7, 8.1, 6.3)]):
        t = (TRANSFORM[0] + dx, res, 0.0, TRANSFORM[3] + dy, 0.0, -res)
        path = str(tmp_path / f"s{i}.tif")
        terrio.Raster.from_array(
            data[i:, : 70 - i].copy(), t, crs="EPSG:32610", nodata=-9999
        ).write(path)
        srcs.append(path)
    ref = terrio.mosaic(
        [terrio.open_raster(p) for p in srcs], resolution="highest", resampling=resampling
    )
    out = terrio.mosaic(srcs, str(tmp_path / "m.tif"), resolution="highest", resampling=resampling)
    assert out.is_lazy
    a, b = np.asarray(out.read()), np.asarray(ref.read())
    assert a.shape == b.shape and a.tobytes() == b.tobytes()
    assert out.transform == ref.transform and out.nodata == ref.nodata


def test_dem_path_to_path_is_streamed(tmp_path):
    """A DEM path to an output path is processed a strip of rows at a time, with the values of
    the computation in memory, and returned opened lazily from the output."""
    rows, cols = np.mgrid[0:60, 0:45]
    dem = ((rows * 7 + cols * 13) % 37 + rows * 2.5).astype("float32")
    dem[0, 0] = dem[20, 30] = -9999
    src = tmp_path / "dem.tif"
    terrio.Raster.from_array(dem, TRANSFORM, crs="EPSG:32610", nodata=-9999).write(str(src))
    colors = tmp_path / "colors.txt"
    colors.write_text("0% 0 0 255\n50 0 255 0 200\n100% 255 0 0\nnv 1 2 3 0\n")
    cases = [
        ("hillshade", {"multidirectional": True, "compute_edges": True}),
        ("slope", {"compute_edges": True}),
        ("color-relief", {"color_file": str(colors), "alpha": True}),
    ]
    for processing, kwargs in cases:
        whole = terrio.dem(terrio.open_raster(str(src)), processing, **kwargs)
        for budget in [None, 0.001]:
            out = tmp_path / f"{processing}.tif"
            r = terrio.dem(str(src), processing, dst=str(out), max_memory_mib=budget, **kwargs)
            assert r.is_lazy
            np.testing.assert_array_equal(r.read(), whole.read())
            assert r.nodata == whole.nodata and r.count == whole.count
    with pytest.raises(terrio.InvalidArgumentError):
        terrio.dem(str(src), "slope", max_memory_mib=10)


def test_shaded_relief_matches_the_numpy_blend(tmp_path):
    """A shaded relief is the color relief darkened by the hillshade as the NumPy step
    `rgb * (1 - s * (1 - shade / 255))` (float32, truncated, where alpha and the hillshade are not
    0) computes it, in memory and path to path; transparent pixels keep the nv color."""
    rows, cols = np.mgrid[0:50, 0:40]
    dem = (np.sin(rows / 5.0) * 40 + cols * 3.0 + 100).astype("float32")
    dem[:3, :4] = -9999
    src = tmp_path / "dem.tif"
    terrio.Raster.from_array(dem, TRANSFORM, crs="EPSG:32610", nodata=-9999).write(str(src))
    colors = tmp_path / "colors.txt"
    colors.write_text("60 20 120 40\n160 230 200 90\n260 255 255 255\nnv 20 120 40 0\n")
    shade_kw = {"multidirectional": True, "compute_edges": True}
    relief = terrio.dem(str(src), "color-relief", color_file=str(colors), alpha=True)
    shade = terrio.dem(str(src), "hillshade", **shade_kw)
    rgba = np.asarray(relief.read()).copy()
    values = np.asarray(shade.read()).astype(np.float32)[0]
    factor = (1.0 - 0.6 * (1.0 - values / 255.0)).astype(np.float32)
    factor = np.where((rgba[3] > 0) & (values > 0), factor, np.float32(1.0))
    rgba[:3] = np.clip(rgba[:3].astype(np.float32) * factor, 0, 255).astype(np.uint8)
    kwargs = {"color_file": str(colors), "alpha": True, **shade_kw}
    whole = terrio.dem(str(src), "shaded-relief", **kwargs)
    streamed = terrio.dem(
        str(src), "shaded-relief", dst=str(tmp_path / "s.tif"), max_memory_mib=0.001, **kwargs
    )
    for r in (whole, streamed):
        np.testing.assert_array_equal(r.read(), rgba)
    assert tuple(rgba[:, 0, 0]) == (20, 120, 40, 0)
    lighter = terrio.dem(str(src), "shaded-relief", hillshade_strength=0.2, **kwargs).read()
    assert (lighter[:3].astype(int) >= rgba[:3]).all()
    with pytest.raises(terrio.InvalidArgumentError):
        terrio.dem(str(src), "color-relief", color_file=str(colors), hillshade_strength=0.5)


@pytest.mark.parametrize("extent", ["ignore", "union", "intersect"])
def test_calc_paths_to_path_are_streamed(tmp_path, extent):
    """Input paths computed to a path are read and written a strip at a time, lazily returned,
    with the values of the calculation in memory, for each extent rule and mixed types."""
    rows, cols = np.mgrid[0:700, 0:500]
    a = ((rows * 7 + cols * 13) % 251).astype("float32") - 20.5
    a[100:103, 50:400] = -9999
    b = ((rows * 3 + cols) % 200).astype("int16")
    b[300, :] = -1
    shift = 0.0 if extent == "ignore" else 37.0
    ta = TRANSFORM
    tb = (TRANSFORM[0] + shift, TRANSFORM[1], 0.0, TRANSFORM[3] - shift, 0.0, TRANSFORM[5])
    pa, pb = str(tmp_path / "a.tif"), str(tmp_path / "b.tif")
    terrio.Raster.from_array(a, ta, crs="EPSG:32610", nodata=-9999).write(pa)
    terrio.Raster.from_array(b, tb, crs="EPSG:32610", nodata=-1).write(pb)
    expr = "where(A > B, A * 2, B) + (B % 3)"
    ref = terrio.calc(expr, A=terrio.open_raster(pa), B=terrio.open_raster(pb), extent=extent)
    out = terrio.calc(
        expr, str(tmp_path / f"c_{extent}.tif"), A=pa, B=pb, extent=extent, max_memory_mib=0.5
    )
    assert out.is_lazy
    x, y = np.asarray(out.read()), np.asarray(ref.read())
    assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
    assert out.transform == ref.transform and out.nodata == ref.nodata
    # `os.PathLike` inputs are paths too.
    out = terrio.calc(
        expr, str(tmp_path / "p.tif"), A=tmp_path / "a.tif", B=tmp_path / "b.tif", extent=extent
    )
    assert out.is_lazy and np.asarray(out.read()).tobytes() == y.tobytes()
    with pytest.raises(terrio.InvalidArgumentError):
        terrio.calc(expr, A=terrio.open_raster(pa), B=pb, max_memory_mib=1)


def test_calc_numpy_typing():
    """Checks that calc follows NumPy's typing rules, as gdal_calc.py does."""
    a = np.array([[200, 3], [0, 100]], dtype="uint8")
    b = np.array([[100, 7], [5, 0]], dtype="uint8")
    ra = terrio.Raster.from_array(a, TRANSFORM, crs="EPSG:32610", nodata=0)
    rb = terrio.Raster.from_array(b, TRANSFORM, crs="EPSG:32610", nodata=0)
    out = terrio.calc("A + B", A=ra, B=rb)
    assert out.read(1).dtype == np.uint8
    assert out.nodata == 255
    np.testing.assert_array_equal(out.read(1), [[44, 10], [255, 255]])
    ndvi = terrio.calc(
        "(B.astype(float32) - A) / (B.astype(float32) + A)", A=ra, B=rb, type="Float32", nodata=-1
    )
    np.testing.assert_allclose(ndvi.read(1), [[-1 / 3, 0.4], [-1, -1]], rtol=1e-6)
    two = terrio.calc(["A", "where(B > 4, 1, 0)"], A=ra, B=rb, hide_nodata=True)
    assert two.count == 2 and two.nodata is None
    np.testing.assert_array_equal(two.read(2), [[1, 1], [1, 0]])
    with pytest.raises(terrio.InvalidArgumentError):
        terrio.calc("A + C", A=ra)


def test_proximity_and_sieve():
    """Checks distances to target pixels and the merging of small areas."""
    data = np.zeros((5, 5), dtype="uint8")
    data[2, 2] = 1
    r = terrio.Raster.from_array(data, TRANSFORM, crs="EPSG:32610")
    dist = terrio.proximity(r).read(1)
    assert dist[2, 4] == 2.0 and dist[2, 2] == 0.0
    geo = terrio.proximity(r, distunits="GEO", maxdist=45, nodata=-1).read(1)
    assert geo[2, 3] == 30.0 and geo[0, 0] == -1
    sieved = terrio.sieve(r, threshold=2).read(1)
    assert sieved[2, 2] == 0
    assert terrio.sieve(r, threshold=1).read(1)[2, 2] == 1


def test_sql_expressions_distinct_summary():
    src = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0, 0]},
                "properties": {"v": value},
            }
            for value in [2, 2, None]
        ],
    }
    result = terrio.sql(src, "SELECT v*2 AS doubled FROM layer")
    assert result.to_geojson()["features"][0]["properties"]["doubled"] == 4
    summary = terrio.sql(src, "SELECT COUNT(*) AS n, SUM(v) AS total FROM layer")
    feature = summary.to_geojson()["features"][0]
    assert feature["properties"] == {"n": 3, "total": 4}
    assert feature["geometry"] is None
    assert len(terrio.sql(src, "SELECT DISTINCT v FROM layer")) == 2


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("step", [1.0, 2.0])
def test_geolocation_arrays_match_gdal(gdal, tmp_path, center, step):
    """Center offsets are measured in array steps, including subsampled arrays."""
    values = np.arange(48, dtype="float64").reshape(6, 8)
    source = terrio.Raster.from_array(values, (0, 1, 0, 6, 0, -1))
    lon = np.array([[10, 12, 14], [10.1, 12.1, 14.1], [10.2, 12.2, 14.2]], dtype="float64")
    lat = np.array([[50, 50.1, 50.2], [48, 48.1, 48.2], [46, 46.1, 46.2]], dtype="float64")
    geo = {"pixel_step": step, "line_step": step, "pixel_center": center, "x_band": 2, "y_band": 2}
    paths = []
    for name, data in [
        ("source", values),
        ("lon", np.stack([np.zeros_like(lon), lon])),
        ("lat", np.stack([np.zeros_like(lat), lat])),
    ]:
        path = str(tmp_path / f"{name}.tif")
        terrio.Raster.from_array(data, (0, 1, 0, data.shape[-2], 0, -1)).write(path)
        paths.append(path)
    ref_source = gdal.Open(paths[0])
    from osgeo import osr

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ref_source.SetMetadata(
        {
            "SRS": srs.ExportToWkt(),
            "X_DATASET": paths[1],
            "Y_DATASET": paths[2],
            "X_BAND": "2",
            "Y_BAND": "2",
            "PIXEL_OFFSET": "0",
            "LINE_OFFSET": "0",
            "PIXEL_STEP": str(step),
            "LINE_STEP": str(step),
            "GEOREFERENCING_CONVENTION": "PIXEL_CENTER" if center else "TOP_LEFT_CORNER",
        },
        "GEOLOCATION",
    )
    ref = gdal.Warp(
        "",
        ref_source,
        format="MEM",
        geoloc=True,
        dstSRS="EPSG:4326",
        outputBounds=(10.2, 46.4, 13.8, 49.7),
        width=31,
        height=29,
        dstNodata=-9999,
        errorThreshold=0,
    )
    actual = terrio.warp(
        source,
        geolocation={"x": paths[1], "y": paths[2], **geo},
        te=(10.2, 46.4, 13.8, 49.7),
        ts=(31, 29),
        dstnodata=-9999,
        t_srs="EPSG:4326",
        et=0,
    )
    np.testing.assert_array_equal(actual.read(1), ref.ReadAsArray())


@pytest.mark.parametrize(
    "geometry",
    [
        {"type": "Point", "coordinates": [8.5, 47.4]},
        {"type": "Point", "coordinates": [8.5, 47.4, 123.5]},
        {"type": "LineString", "coordinates": [[0, 0], [2, 3], [5, 1]]},
        {"type": "Polygon", "coordinates": [[[0, 0], [0, 5], [5, 5], [5, 0], [0, 0]]]},
    ],
)
def test_shapefile_zip_bytes(geometry):
    import io
    import zipfile

    source = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {"name": "Zürich", "id": 12345678901, "real": 2.5, "flag": True},
            },
            {
                "type": "Feature",
                "geometry": None,
                "properties": {"name": None, "id": None, "real": None, "flag": None},
            },
        ],
    }
    layer = terrio.Layer.from_geojson(source, name="source")
    archive = layer.to_bytes("places.shp.zip")
    with zipfile.ZipFile(io.BytesIO(archive)) as members:
        assert sorted(members.namelist()) == [
            "places.cpg",
            "places.dbf",
            "places.prj",
            "places.shp",
            "places.shx",
        ]
        assert members.read("places.cpg") == b"UTF-8"
    back = terrio.open_bytes(archive, "places.shp.zip")[0]
    assert back.crs == "EPSG:4326"
    assert back.columns() == {
        "name": ["Zürich", None],
        "id": [12345678901, None],
        "real": [2.5, None],
        "flag": [True, None],
    }
    with pytest.raises(terrio.InvalidArgumentError, match="shp.zip"):
        layer.to_bytes("places.shp")
    source["features"][0]["properties"]["name"] = "https://example.org/" + "x" * 254
    with pytest.raises(terrio.UnsupportedError, match="254 UTF-8 bytes"):
        terrio.Layer.from_geojson(source).to_bytes("long.shp.zip")


def test_geojson_utf8_text_uses_shared_parser():
    source = (
        '{"type":"Feature","geometry":{"type":"Point","coordinates":[1,2]},'
        '"properties":{"name":"Zürich 東京 🌍"}}'
    )
    layer = terrio.Layer.from_geojson(source)
    assert layer.columns()["name"] == ["Zürich 東京 🌍"]
