//! Area-matched overviews: coarse levels whose valid (wet) area matches the full-resolution
//! image's, instead of growing at every level as GDAL's `average` (a coarse pixel is valid when any
//! of its source pixels is) or speckling the margin as `nearest`.
//!
//! The algorithm is ras2cng's `cog.build_area_matched_overviews` (CLB Engineering, MIT), ported
//! so that its outputs are identical. Each level `k` (factor `f = 2^k`) carries, per coarse cell,
//! the number of valid full-resolution pixels it covers (its weight) and their sum. The number of
//! cells to keep is `round(valid / f^2)` (ties to even), and a threshold on the weight is solved
//! for from the weight histogram so that the heaviest cells are kept. The cells exactly at the
//! threshold are broken by coherence (how many of their four neighbours are already kept, most
//! first), then by a deterministic hash of their index, so that the margin stays connected and two
//! builds of the same raster give the same level. Kept cells hold the mean of their valid source
//! pixels (rounded half to even for integer types), or, for categorical rasters, the mode of the
//! 2x2 modes of the level below (the lowest value breaking ties). Others are nodata.
//!
//! Level 1 is reduced from the full-resolution pixels, each later level from the one before (odd
//! sizes padded with empty cells), so a level is `ceil(cols / f) x ceil(rows / f)`, as the COG
//! writer sizes its overviews.

use crate::pixel::Pixel;

/// The levels of one band and the valid area each reproduces.
pub(crate) struct AreaMatchedBand<T> {
    /// The levels, from the largest (factor 2), each `ceil(cols / f) * ceil(rows / f)` values.
    pub(crate) levels: Vec<Vec<T>>,
    /// Per level, the kept cells times `f^2` over the full-resolution valid pixels.
    pub(crate) area_ratios: Vec<f64>,
    /// The valid full-resolution pixels.
    pub(crate) base_valid: u64,
    /// The full-resolution NaN samples under a numeric nodata, counted whether or not `nan_valid`
    /// makes them valid (0 under a NaN nodata and on integer bands).
    pub(crate) nan_pixels: u64,
}

/// One level's accumulators: per cell, the sum and count of valid full-resolution pixels, and the
/// modal value (categorical rasters).
struct Accumulators {
    cols: usize,
    rows: usize,
    sums: Vec<f64>,
    weights: Vec<u64>,
    modes: Vec<f64>,
    mode_valid: Vec<bool>,
}

/// Whether a sample is valid: not nodata, and not NaN unless `nan_valid` and the nodata value is
/// not NaN itself (rasterio's mask, which ras2cng reads: a numeric nodata leaves NaN valid).
#[inline]
fn valid(v: f64, nodata: f64, nan_valid: bool) -> bool {
    if v.is_nan() { nan_valid && !nodata.is_nan() } else { v != nodata }
}

/// The mode of the valid values of a 2x2 block (in row order: top left, top right, bottom left,
/// bottom right) as ras2cng's `_reduce_mode` computes it: the value matched by the most valid
/// values, the lowest on ties. Not valid when no value is.
#[inline]
fn block_mode(values: [f64; 4], valid: [bool; 4]) -> (f64, bool) {
    let mut counts = [-1i32; 4];
    for i in 0..4 {
        if valid[i] {
            counts[i] = (0..4).filter(|&j| valid[j] && values[j] == values[i]).count() as i32;
        }
    }
    let best = *counts.iter().max().unwrap_or(&-1);
    let mode = (0..4).filter(|&i| counts[i] == best).map(|i| values[i]).fold(f64::INFINITY, f64::min);
    (mode, best > 0)
}

impl Accumulators {
    /// Level 1, from a band of `cols` x `rows` samples.
    /// Returns the accumulators, the valid pixels and the NaN samples under a numeric nodata.
    fn first<T: Pixel>(band: &[T], cols: usize, rows: usize, nodata: f64, nan_valid: bool, categorical: bool) -> (Self, u64, u64) {
        let (oc, or) = (cols.div_ceil(2), rows.div_ceil(2));
        let n = oc * or;
        let mut acc = Self {
            cols: oc,
            rows: or,
            sums: vec![0.0; n],
            weights: vec![0; n],
            modes: if categorical { vec![0.0; n] } else { Vec::new() },
            mode_valid: if categorical { vec![false; n] } else { Vec::new() },
        };
        let mut base_valid = 0u64;
        let mut nan_pixels = 0u64;
        let numeric_nodata = !nodata.is_nan();
        for oy in 0..or {
            for ox in 0..oc {
                let mut values = [0.0f64; 4];
                let mut ok = [false; 4];
                for (k, (dy, dx)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
                    let (y, x) = (2 * oy + dy, 2 * ox + dx);
                    if y < rows && x < cols {
                        let v = band[y * cols + x].to_f64();
                        if numeric_nodata && v.is_nan() {
                            nan_pixels += 1;
                        }
                        if valid(v, nodata, nan_valid) {
                            values[k] = v;
                            ok[k] = true;
                        }
                    }
                }
                let o = oy * oc + ox;
                acc.sums[o] = sum4(values);
                acc.weights[o] = ok.iter().filter(|v| **v).count() as u64;
                base_valid += acc.weights[o];
                if categorical {
                    (acc.modes[o], acc.mode_valid[o]) = block_mode(values, ok);
                }
            }
        }
        (acc, base_valid, nan_pixels)
    }

    /// The next (half-size) level.
    fn reduce(&self, categorical: bool) -> Self {
        let (oc, or) = (self.cols.div_ceil(2), self.rows.div_ceil(2));
        let n = oc * or;
        let mut next = Self {
            cols: oc,
            rows: or,
            sums: vec![0.0; n],
            weights: vec![0; n],
            modes: if categorical { vec![0.0; n] } else { Vec::new() },
            mode_valid: if categorical { vec![false; n] } else { Vec::new() },
        };
        for oy in 0..or {
            for ox in 0..oc {
                let mut sums = [0.0f64; 4];
                let mut weight = 0u64;
                let mut values = [0.0f64; 4];
                let mut ok = [false; 4];
                for (k, (dy, dx)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
                    let (y, x) = (2 * oy + dy, 2 * ox + dx);
                    if y < self.rows && x < self.cols {
                        let i = y * self.cols + x;
                        sums[k] = self.sums[i];
                        weight += self.weights[i];
                        if categorical {
                            values[k] = self.modes[i];
                            ok[k] = self.mode_valid[i];
                        }
                    }
                }
                let o = oy * oc + ox;
                next.sums[o] = sum4(sums);
                next.weights[o] = weight;
                if categorical {
                    (next.modes[o], next.mode_valid[o]) = block_mode(values, ok);
                }
            }
        }
        next
    }
}

/// The sum of a 2x2 block in NumPy's order for `reshape(h, 2, w, 2).sum(axis=(1, 3))`: the
/// columns of each row, then the rows.
#[inline]
fn sum4(v: [f64; 4]) -> f64 {
    (v[0] + v[1]) + (v[2] + v[3])
}

/// ras2cng's `_dither`: a deterministic hash of a cell index (SplitMix64's finalizer).
#[inline]
fn dither(index: u64, salt: u64) -> u64 {
    let mut x = index ^ salt;
    x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    x
}

/// Which cells of a level of factor `factor` stay valid so that the level reproduces the
/// full-resolution valid area (ras2cng's `_select_area_matched`).
fn select(weights: &[u64], cols: usize, rows: usize, factor: u64, base_valid: u64) -> Vec<bool> {
    let capacity = factor * factor;
    let target = (base_valid as f64 / capacity as f64).round_ties_even() as u64;
    let wet: Vec<bool> = weights.iter().map(|w| *w > 0).collect();
    if target == 0 {
        return vec![false; weights.len()];
    }
    let wet_count = wet.iter().filter(|w| **w).count() as u64;
    if wet_count == 0 || target >= wet_count {
        return wet;
    }
    let mut counts = vec![0u64; capacity as usize + 1];
    for w in weights {
        counts[(*w).min(capacity) as usize] += 1;
    }
    let mut accumulated = 0u64;
    let mut threshold = 0u64;
    for level in (1..=capacity).rev() {
        if accumulated + counts[level as usize] >= target {
            threshold = level;
            break;
        }
        accumulated += counts[level as usize];
    }
    if threshold == 0 {
        return wet;
    }
    let mut keep: Vec<bool> = weights.iter().map(|w| *w > threshold).collect();
    let remaining = (target - accumulated) as usize;
    let ties: Vec<usize> = (0..weights.len()).filter(|&i| weights[i] == threshold).collect();
    if remaining >= ties.len() {
        for i in ties {
            keep[i] = true;
        }
        return keep;
    }
    // Ties by the number of kept 4-neighbours (most first), then the dither (highest first), then
    // the index (highest first), as NumPy's reversed stable `lexsort((dither, coherence))`.
    let salt = factor.wrapping_mul(0x2545_F491);
    let mut ranked: Vec<(u8, u64, usize)> = ties
        .iter()
        .map(|&i| {
            let (y, x) = (i / cols, i % cols);
            let mut c = 0u8;
            if y > 0 && keep[i - cols] {
                c += 1;
            }
            if y + 1 < rows && keep[i + cols] {
                c += 1;
            }
            if x > 0 && keep[i - 1] {
                c += 1;
            }
            if x + 1 < cols && keep[i + 1] {
                c += 1;
            }
            (c, dither(i as u64, salt), i)
        })
        .collect();
    ranked.sort_unstable_by(|a, b| b.cmp(a));
    for &(_, _, i) in &ranked[..remaining] {
        keep[i] = true;
    }
    keep
}

/// The area-matched levels of one band of `cols` x `rows` samples: `count` levels of factors 2,
/// 4, 8... `integral` rounds the means of integer types half to even, as NumPy's `rint`.
/// `nan_valid` counts NaN samples as valid under a numeric nodata, as rasterio's mask does: they
/// add to the coverage, and the mean of a cell holding one is NaN (NumPy's sum propagates it).
/// They never make a mode: NaN equals no value, so a block of NaNs only has no mode.
#[allow(clippy::too_many_arguments)]
pub(crate) fn area_matched_band<T: Pixel>(
    band: &[T],
    cols: usize,
    rows: usize,
    nodata: f64,
    nan_valid: bool,
    count: usize,
    categorical: bool,
    integral: bool,
) -> AreaMatchedBand<T> {
    let mut out =
        AreaMatchedBand { levels: Vec::with_capacity(count), area_ratios: Vec::with_capacity(count), base_valid: 0, nan_pixels: 0 };
    if count == 0 {
        return out;
    }
    let (mut acc, base_valid, nan_pixels) = Accumulators::first(band, cols, rows, nodata, nan_valid, categorical);
    out.base_valid = base_valid;
    out.nan_pixels = nan_pixels;
    for k in 0..count {
        if k > 0 {
            acc = acc.reduce(categorical);
        }
        let factor = 1u64 << (k + 1);
        let keep = select(&acc.weights, acc.cols, acc.rows, factor, base_valid);
        let kept = keep.iter().filter(|v| **v).count() as u64;
        out.area_ratios.push(if base_valid == 0 { 0.0 } else { (kept * factor * factor) as f64 / base_valid as f64 });
        let level: Vec<T> = (0..keep.len())
            .map(|i| {
                let v = if categorical {
                    if keep[i] && acc.mode_valid[i] { acc.modes[i] } else { nodata }
                } else if keep[i] && acc.weights[i] > 0 {
                    let mean = acc.sums[i] / acc.weights[i] as f64;
                    if integral { mean.round_ties_even() } else { mean }
                } else {
                    nodata
                };
                T::from_f64(v)
            })
            .collect();
        out.levels.push(level);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_takes_the_most_frequent_then_lowest_valid_value() {
        assert_eq!(block_mode([3.0, 1.0, 3.0, 1.0], [true; 4]), (1.0, true));
        assert_eq!(block_mode([3.0, 1.0, 3.0, 2.0], [true; 4]), (3.0, true));
        assert_eq!(block_mode([3.0, 1.0, 3.0, 2.0], [false, true, false, true]), (1.0, true));
        assert!(!block_mode([0.0; 4], [false; 4]).1);
    }

    #[test]
    fn dither_matches_ras2cng() {
        // _dither(np.array([0, 1, 12345]), salt=2 * 0x2545F491) in ras2cng.
        let salt = 2u64 * 0x2545_F491;
        let got: Vec<u64> = [0, 1, 12345].iter().map(|i| dither(*i, salt)).collect();
        assert_eq!(got, [12459186666850337160, 6905069863801935470, 9666813718830214867]);
    }

    #[test]
    fn levels_keep_the_valid_area() {
        // A diagonal band of valid pixels: `average` would mark every touched coarse cell valid.
        let (cols, rows) = (64, 48);
        let nd = -9999.0f32;
        let band: Vec<f32> =
            (0..rows).flat_map(|y| (0..cols).map(move |x| if (x as i32 - y as i32).abs() <= 2 { (x + y) as f32 } else { nd })).collect();
        let out = area_matched_band(&band, cols, rows, nd as f64, false, 3, false, false);
        assert_eq!(out.levels.len(), 3);
        assert_eq!(out.levels[0].len(), 32 * 24);
        assert_eq!(out.levels[2].len(), 8 * 6);
        let valid = band.iter().filter(|v| **v != nd).count() as u64;
        assert_eq!(out.base_valid, valid);
        for (k, level) in out.levels.iter().enumerate() {
            let f = 1u64 << (k + 1);
            let kept = level.iter().filter(|v| **v != nd).count() as u64;
            assert_eq!(kept, (valid as f64 / (f * f) as f64).round_ties_even() as u64, "level {k}");
            assert!((out.area_ratios[k] - (kept * f * f) as f64 / valid as f64).abs() < 1e-12);
        }
        // Deterministic.
        let again = area_matched_band(&band, cols, rows, nd as f64, false, 3, false, false);
        assert_eq!(out.levels, again.levels);
    }

    #[test]
    fn means_of_kept_cells_and_integer_rounding() {
        // One 2x2 block of 1, 2, 2, 2 (mean 1.75) beside an empty block.
        let band: Vec<i16> = vec![1, 2, 0, 0, 2, 2, 0, 0];
        let out = area_matched_band(&band, 4, 2, 0.0, false, 1, false, true);
        assert_eq!(out.levels[0], vec![2, 0]);
        // Half-way means round to even.
        let band: Vec<i16> = vec![1, 2, 3, 4, 3, 2, 5, 6];
        let out = area_matched_band(&band, 4, 2, -1.0, false, 1, false, true);
        assert_eq!(out.levels[0], vec![2, 4]);
        let out = area_matched_band(&band, 4, 2, -1.0, false, 1, true, true);
        assert_eq!(out.levels[0], vec![2, 3]);
    }

    #[test]
    fn nan_under_a_numeric_nodata_is_valid_when_asked() {
        let nd = -9999.0f32;
        // A 4x2 band: the left block holds a NaN, the right block one valid pixel.
        let band = vec![2.0f32, 1.0, nd, nd, f32::NAN, 3.0, 5.0, nd];
        let out = area_matched_band(&band, 4, 2, nd as f64, true, 1, false, false);
        // Five valid pixels (the NaN counts): round(5 / 4) = 1 cell, the fuller one, whose mean is NaN.
        assert_eq!(out.base_valid, 5);
        assert_eq!(out.nan_pixels, 1);
        assert!(out.levels[0][0].is_nan());
        assert_eq!(out.levels[0][1], nd);
        assert_eq!(out.area_ratios, vec![4.0 / 5.0]);
        // By default the NaN is invalid: three valid pixels on the left, mean 2.
        let out = area_matched_band(&band, 4, 2, nd as f64, false, 1, false, false);
        assert_eq!(out.base_valid, 4);
        // Counted the same way when NaN is not valid.
        assert_eq!(out.nan_pixels, 1);
        assert_eq!(out.levels[0], vec![2.0, nd]);
        // The NaN reaches every coarser level holding the cell.
        let big: Vec<f32> = (0..64).map(|i| if i == 9 { f32::NAN } else { 1.0 }).collect();
        let out = area_matched_band(&big, 8, 8, nd as f64, true, 3, false, false);
        assert_eq!(out.base_valid, 64);
        assert!(out.levels[0][0].is_nan() && out.levels[0].iter().filter(|v| v.is_nan()).count() == 1);
        assert!(out.levels[1][0].is_nan() && out.levels[2][0].is_nan());
        // A NaN nodata keeps NaN invalid.
        let out = area_matched_band(&[f32::NAN, 1.0, f32::NAN, f32::NAN], 2, 2, f64::NAN, true, 1, false, false);
        assert_eq!(out.base_valid, 1);
        // NaN under a NaN nodata is nodata, not counted.
        assert_eq!(out.nan_pixels, 0);
        // Integer bands have no NaN: unchanged.
        let ints: Vec<i16> = vec![1, 2, 0, 0, 2, 2, 0, 0];
        assert_eq!(
            area_matched_band(&ints, 4, 2, 0.0, true, 1, false, true).levels,
            area_matched_band(&ints, 4, 2, 0.0, false, 1, false, true).levels
        );
    }

    #[test]
    fn valid_nans_cover_but_never_make_a_mode() {
        let nd = -9999.0f32;
        // Left block: all NaN (covered, no mode). Right block: 1, NaN, 1, nodata (mode 1).
        let band = vec![f32::NAN, f32::NAN, 1.0, f32::NAN, f32::NAN, f32::NAN, 1.0, nd];
        let out = area_matched_band(&band, 4, 2, nd as f64, true, 1, true, false);
        assert_eq!(out.base_valid, 7);
        // round(7 / 4) = 2 cells kept; the left one has no mode, so it stays nodata.
        assert_eq!(out.levels[0], vec![nd, 1.0]);
        assert_eq!(out.area_ratios, vec![8.0 / 7.0]);
    }

    #[test]
    fn nan_is_never_valid() {
        let band = vec![f32::NAN, 1.0, f32::NAN, f32::NAN];
        let out = area_matched_band(&band, 2, 2, f64::NAN, false, 1, false, false);
        // One valid pixel of four: round(1 / 4) = 0 cells kept.
        assert!(out.levels[0][0].is_nan());
        let band = vec![2.0f32, 1.0, f32::NAN, 3.0];
        let out = area_matched_band(&band, 2, 2, -9999.0, false, 1, false, false);
        assert_eq!(out.levels[0], vec![2.0]);
    }
}
