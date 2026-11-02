//! Douglas-Peucker simplification of lines and polygons, directly on Well-Known Binary.
//!
//! [`simplify_wkb`] and [`simplify_wkb_many`] (in parallel over features) run GEOS 3.14's
//! `DouglasPeuckerSimplifier`, which shapely's `simplify(..., preserve_topology=False)` and
//! PostGIS's `ST_Simplify` call, on WKB without decoding it into geometry objects:
//!
//! - Every linear component is simplified on its own (`DouglasPeuckerLineSimplifier`): a section
//!   is replaced by its chord when no vertex is farther than the tolerance from the chord segment
//!   (a distance equal to the tolerance flattens it), otherwise it is split at the farthest vertex
//!   (the first one on ties). Line endpoints stay. Distances use GEOS's formulas, operation for
//!   operation, so vertices at exactly the tolerance are decided as GEOS decides them.
//! - A polygon ring may also lose its start point: when the simplified ring has at least five
//!   points and its start lies within the tolerance of the segment joining its neighbours, the
//!   ring is closed on its second point instead (GEOS's `simplifyRingEndpoint`). Closed
//!   linestrings keep their endpoints, as in GEOS.
//! - A ring left with fewer than four points has collapsed. With [`RingCollapse::Keep`] (the
//!   default) it keeps its input vertices, so that no polygon or hole disappears and the
//!   collapsed rings are counted ([`SimplifyStats::rings_kept`]). [`RingCollapse::Drop`] does as
//!   GEOS: a hole is removed, a polygon whose shell collapses becomes empty, or leaves its
//!   multipolygon.
//! - There is no repair step. GEOS rebuilds an invalid polygonal result with a zero-width buffer;
//!   here it is returned as simplified. Douglas-Peucker can make a ring self-intersect or a hole
//!   cross its shell when the tolerance is large against the feature; use
//!   [`SimplifyMethod::PreserveTopology`] ([`crate::ops::simplify::simplify_preserve_topology`])
//!   where that matters.
//!
//! So wherever GEOS's result is valid and no ring collapses (or with [`RingCollapse::Drop`]),
//! the output has GEOS's vertices, in GEOS's order. Every output vertex is an input vertex,
//! copied with its Z and M. Each (sub)geometry keeps its byte order and type code (ISO or
//! extended WKB, with an SRID). Points pass through. A feature with a non-finite X or Y passes
//! through unchanged and is counted ([`SimplifyStats::non_finite`]).

use super::simplify::{point_segment_distance, simplify_preserve_topology};
use crate::{Error, Geometry, Result};
use std::str::FromStr;

type P = (f64, f64);

/// The simplification algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimplifyMethod {
    /// Douglas-Peucker on each component (GEOS's `DouglasPeuckerSimplifier`, see the module
    /// documentation): fast, but rings may self-intersect at large tolerances.
    #[default]
    DouglasPeucker,
    /// GEOS's `TopologyPreservingSimplifier` ([`crate::ops::simplify::simplify_preserve_topology`]):
    /// no flattening may cross another segment of the feature. Z is kept, M dropped, and the
    /// output is little-endian ISO WKB.
    PreserveTopology,
}

impl FromStr for SimplifyMethod {
    type Err = Error;

    /// Parses `douglas-peucker` (also `dp`) or `preserve-topology` (also `topology`), ignoring
    /// case, with `_` for `-`.
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().replace('_', "-").as_str() {
            "douglas-peucker" | "dp" => Ok(Self::DouglasPeucker),
            "preserve-topology" | "topology" => Ok(Self::PreserveTopology),
            _ => Err(Error::InvalidArgument(format!("unknown simplification method '{s}' (douglas-peucker, preserve-topology)"))),
        }
    }
}

/// What happens to a polygon ring that Douglas-Peucker leaves with fewer than four points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RingCollapse {
    /// The ring keeps its input vertices: no polygon or hole disappears.
    #[default]
    Keep,
    /// As GEOS: the hole is removed; a polygon whose shell collapses becomes empty, and leaves
    /// its multipolygon.
    Drop,
}

impl FromStr for RingCollapse {
    type Err = Error;

    /// Parses `keep` or `drop`, ignoring case.
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "keep" => Ok(Self::Keep),
            "drop" => Ok(Self::Drop),
            _ => Err(Error::InvalidArgument(format!("unknown ring collapse policy '{s}' (keep, drop)"))),
        }
    }
}

/// Simplification options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimplifyOptions {
    /// The distance tolerance, in CRS units (non-negative and finite).
    pub tolerance: f64,
    /// The algorithm.
    pub method: SimplifyMethod,
    /// Collapsed rings, with [`SimplifyMethod::DouglasPeucker`].
    pub collapse: RingCollapse,
}

impl SimplifyOptions {
    /// Douglas-Peucker with `tolerance`, keeping collapsed rings.
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance, method: SimplifyMethod::default(), collapse: RingCollapse::default() }
    }

    /// Checks the tolerance.
    pub fn validate(&self) -> Result<()> {
        if !(self.tolerance >= 0.0 && self.tolerance.is_finite()) {
            return Err(Error::InvalidArgument("the simplification tolerance must be a non-negative number".into()));
        }
        Ok(())
    }
}

/// What a simplification did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SimplifyStats {
    /// Features (null ones included).
    pub features: u64,
    /// Null features, passed through.
    pub nulls: u64,
    /// Vertices read (points included, closing points of rings included, as GEOS counts them).
    pub vertices_in: u64,
    /// Vertices written.
    pub vertices_out: u64,
    /// Rings that collapsed and kept their input vertices ([`RingCollapse::Keep`]).
    pub rings_kept: u64,
    /// Rings removed ([`RingCollapse::Drop`]): collapsed holes, and every ring of an emptied
    /// polygon.
    pub rings_dropped: u64,
    /// Polygons emptied, or removed from their multipolygon, by a collapsed shell
    /// ([`RingCollapse::Drop`]).
    pub polygons_emptied: u64,
    /// Features passed through unchanged because of a non-finite X or Y.
    pub non_finite: u64,
}

impl std::ops::AddAssign for SimplifyStats {
    fn add_assign(&mut self, o: Self) {
        self.features += o.features;
        self.nulls += o.nulls;
        self.vertices_in += o.vertices_in;
        self.vertices_out += o.vertices_out;
        self.rings_kept += o.rings_kept;
        self.rings_dropped += o.rings_dropped;
        self.polygons_emptied += o.polygons_emptied;
        self.non_finite += o.non_finite;
    }
}

/// Buffers reused from feature to feature.
#[derive(Default)]
struct Scratch {
    pts: Vec<P>,
    keep: Vec<usize>,
    drop: Vec<bool>,
}

/// GEOS's `Coordinate::distance`.
#[inline]
fn dist(p: P, a: P) -> f64 {
    let (dx, dy) = (p.0 - a.0, p.1 - a.1);
    (dx * dx + dy * dy).sqrt()
}

/// Sections longer than this search for their farthest vertex in parallel, and split into
/// two such sections that are simplified in parallel. The results do not depend on it.
const PARALLEL_SECTION: usize = 1 << 15;

/// Vertices per piece of a parallel farthest-vertex search.
const SEARCH_CHUNK: usize = 1 << 13;

/// The farthest of `pts[from..to]` from the segment `a`-`b`, as `(distance, index)`: the first
/// one on ties, -1 when there is none (`Distance::pointToSegment`, with the segment's terms
/// computed once: the same operations, so the same values as `point_segment_distance`).
fn farthest(pts: &[P], from: usize, to: usize, a: P, b: P) -> (f64, usize) {
    let (mut max, mut far) = (-1.0, from);
    if a == b {
        for (k, &p) in pts.iter().enumerate().take(to).skip(from) {
            let d = dist(p, a);
            if d > max {
                max = d;
                far = k;
            }
        }
        return (max, far);
    }
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    let len = len2.sqrt();
    for (k, &p) in pts.iter().enumerate().take(to).skip(from) {
        let r = ((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2;
        let d = if r <= 0.0 {
            dist(p, a)
        } else if r >= 1.0 {
            dist(p, b)
        } else {
            (((a.1 - p.1) * dx - (a.0 - p.0) * dy) / len2).abs() * len
        };
        if d > max {
            max = d;
            far = k;
        }
    }
    (max, far)
}

/// [`farthest`] over the interior of the section `i`-`j`, in parallel pieces for long sections.
/// Pieces are combined in order with the same strict comparison, so ties go to the first vertex.
fn section_farthest(pts: &[P], i: usize, j: usize) -> (f64, usize) {
    let (a, b) = (pts[i], pts[j]);
    if j - i <= PARALLEL_SECTION {
        return farthest(pts, i + 1, j, a, b);
    }
    let pieces = (j - i - 1).div_ceil(SEARCH_CHUNK);
    let found = crate::par::map_range_coarse(pieces, |c| {
        let from = i + 1 + c * SEARCH_CHUNK;
        farthest(pts, from, (from + SEARCH_CHUNK).min(j), a, b)
    });
    found.into_iter().fold((-1.0, i), |best, x| if x.0 > best.0 { x } else { best })
}

/// GEOS's `DouglasPeuckerLineSimplifier::simplifySection` on the section `i`-`j`: `drop` holds
/// one flag per interior vertex (`i + 1..j`), set to true for the vertices removed. The
/// smaller side of a split is recursed into and the larger one looped on, so the depth stays
/// logarithmic; two long sides are simplified in parallel.
fn simplify_section(pts: &[P], mut i: usize, mut j: usize, tolerance: f64, mut drop: &mut [bool]) {
    while i + 1 < j {
        let (max, far) = section_farthest(pts, i, j);
        if max <= tolerance {
            drop.fill(true);
            return;
        }
        let (left, rest) = drop.split_at_mut(far - i - 1);
        let right = &mut rest[1..];
        if far - i > PARALLEL_SECTION && j - far > PARALLEL_SECTION {
            crate::par::join(|| simplify_section(pts, i, far, tolerance, left), || simplify_section(pts, far, j, tolerance, right));
            return;
        }
        if far - i <= j - far {
            simplify_section(pts, i, far, tolerance, left);
            (i, drop) = (far, right);
        } else {
            simplify_section(pts, far, j, tolerance, right);
            (j, drop) = (far, left);
        }
    }
}

/// The indices of the vertices Douglas-Peucker keeps, in order.
fn douglas_peucker(pts: &[P], tolerance: f64, drop: &mut Vec<bool>, keep: &mut Vec<usize>) {
    keep.clear();
    let n = pts.len();
    if n < 3 {
        keep.extend(0..n);
        return;
    }
    drop.clear();
    drop.resize(n - 2, false);
    simplify_section(pts, 0, n - 1, tolerance, drop);
    keep.push(0);
    keep.extend((1..n - 1).filter(|&k| !drop[k - 1]));
    keep.push(n - 1);
}

/// GEOS's `DouglasPeuckerLineSimplifier::simplifyRingEndpoint`: drops the start of a closed
/// ring of at least five points when it lies within the tolerance of its neighbours' segment.
fn simplify_ring_endpoint(pts: &[P], keep: &mut Vec<usize>, tolerance: f64) {
    let n = keep.len();
    if n < 5 || pts[keep[0]] != pts[keep[n - 1]] {
        return;
    }
    if point_segment_distance(pts[keep[0]], pts[keep[1]], pts[keep[n - 2]]) <= tolerance {
        keep.pop();
        keep.remove(0);
        keep.push(keep[0]);
    }
}

fn wkb_error(pos: usize, msg: &str) -> Error {
    Error::InvalidData(format!("WKB at byte {pos}: {msg}"))
}

/// A (sub)geometry header: byte order, base type (1 to 7) and coordinate dimensions, and the
/// span of its bytes (order, type and SRID), copied to the output as they are.
struct Header {
    le: bool,
    kind: u32,
    dims: usize,
    start: usize,
    end: usize,
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.b.len()).ok_or_else(|| wkb_error(self.pos, "truncated"))?;
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self, le: bool) -> Result<u32> {
        let b: [u8; 4] = self.take(4)?.try_into().expect("4 bytes");
        Ok(if le { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) })
    }

    /// A count of items of `size` bytes each, checked against the bytes left.
    fn count(&mut self, le: bool, size: usize) -> Result<usize> {
        let at = self.pos;
        let n = self.u32(le)? as usize;
        if n.checked_mul(size).is_none_or(|bytes| bytes > self.b.len() - self.pos) {
            return Err(wkb_error(at, "count larger than the data"));
        }
        Ok(n)
    }

    fn header(&mut self) -> Result<Header> {
        let start = self.pos;
        let le = match self.take(1)?[0] {
            0 => false,
            1 => true,
            _ => return Err(wkb_error(start, "invalid byte order")),
        };
        let t = self.u32(le)?;
        let (mut z, mut m) = (t & 0x8000_0000 != 0, t & 0x4000_0000 != 0);
        if t & 0x2000_0000 != 0 {
            self.take(4)?;
        }
        let base = t & 0x0FFF_FFFF;
        match base / 1000 {
            0 => {}
            1 => z = true,
            2 => m = true,
            3 => (z, m) = (true, true),
            _ => return Err(wkb_error(start, &format!("unsupported geometry type {t}"))),
        }
        let kind = base % 1000;
        if !(1..=7).contains(&kind) {
            return Err(wkb_error(start, &format!("unsupported geometry type {t}")));
        }
        Ok(Header { le, kind, dims: 2 + usize::from(z) + usize::from(m), start, end: self.pos })
    }
}

fn put_u32(out: &mut Vec<u8>, v: u32, le: bool) {
    out.extend_from_slice(&if le { v.to_le_bytes() } else { v.to_be_bytes() });
}

fn patch_u32(out: &mut [u8], at: usize, v: u32, le: bool) {
    out[at..at + 4].copy_from_slice(&if le { v.to_le_bytes() } else { v.to_be_bytes() });
}

fn f64_at(b: &[u8], le: bool) -> f64 {
    let a: [u8; 8] = b[..8].try_into().expect("8 bytes");
    if le { f64::from_le_bytes(a) } else { f64::from_be_bytes(a) }
}

/// Douglas-Peucker on one WKB feature.
struct Run<'s> {
    tolerance: f64,
    collapse: RingCollapse,
    s: &'s mut Scratch,
    stats: SimplifyStats,
    non_finite: bool,
}

impl Run<'_> {
    /// Reads `n` coordinates into the scratch points.
    fn points(&mut self, coords: &[u8], n: usize, stride: usize, le: bool) {
        self.s.pts.clear();
        for c in coords.chunks_exact(stride).take(n) {
            let (x, y) = (f64_at(c, le), f64_at(&c[8..], le));
            if !(x.is_finite() && y.is_finite()) {
                self.non_finite = true;
            }
            self.s.pts.push((x, y));
        }
        self.stats.vertices_in += n as u64;
    }

    fn write_kept(&mut self, out: &mut Vec<u8>, coords: &[u8], stride: usize, le: bool) {
        put_u32(out, self.s.keep.len() as u32, le);
        for &k in &self.s.keep {
            out.extend_from_slice(&coords[k * stride..(k + 1) * stride]);
        }
        self.stats.vertices_out += self.s.keep.len() as u64;
    }

    /// Simplifies a ring into `keep`; false when it collapsed and is to be dropped.
    fn ring(&mut self, n: usize) -> bool {
        let s = &mut *self.s;
        if n < 4 {
            s.keep.clear();
            s.keep.extend(0..n);
            return true;
        }
        douglas_peucker(&s.pts, self.tolerance, &mut s.drop, &mut s.keep);
        simplify_ring_endpoint(&s.pts, &mut s.keep, self.tolerance);
        if s.keep.len() >= 4 {
            return true;
        }
        match self.collapse {
            RingCollapse::Keep => {
                s.keep.clear();
                s.keep.extend(0..n);
                self.stats.rings_kept += 1;
                true
            }
            RingCollapse::Drop => false,
        }
    }

    /// Copies a geometry to `out`, simplified. Returns true for a polygon emptied by a dropped
    /// shell (which a multipolygon leaves out).
    fn geometry(&mut self, r: &mut Reader, out: &mut Vec<u8>) -> Result<bool> {
        let h = r.header()?;
        out.extend_from_slice(&r.b[h.start..h.end]);
        let stride = 8 * h.dims;
        match h.kind {
            1 => {
                out.extend_from_slice(r.take(stride)?);
                self.stats.vertices_in += 1;
                self.stats.vertices_out += 1;
                Ok(false)
            }
            2 => {
                let n = r.count(h.le, stride)?;
                let coords = r.take(n * stride)?;
                self.points(coords, n, stride, h.le);
                let s = &mut *self.s;
                douglas_peucker(&s.pts, self.tolerance, &mut s.drop, &mut s.keep);
                self.write_kept(out, coords, stride, h.le);
                Ok(false)
            }
            3 => {
                let rings = r.count(h.le, 4)?;
                let count_at = out.len();
                put_u32(out, 0, h.le);
                let mut written = 0u32;
                for i in 0..rings {
                    let n = r.count(h.le, stride)?;
                    let coords = r.take(n * stride)?;
                    self.points(coords, n, stride, h.le);
                    if self.ring(n) {
                        self.write_kept(out, coords, stride, h.le);
                        written += 1;
                    } else if i > 0 {
                        self.stats.rings_dropped += 1;
                    } else {
                        // The shell collapsed: the polygon is empty. Read the holes past.
                        for _ in 1..rings {
                            let n = r.count(h.le, stride)?;
                            let coords = r.take(n * stride)?;
                            self.points(coords, n, stride, h.le);
                        }
                        out.truncate(count_at);
                        put_u32(out, 0, h.le);
                        self.stats.rings_dropped += rings as u64;
                        self.stats.polygons_emptied += 1;
                        return Ok(true);
                    }
                }
                patch_u32(out, count_at, written, h.le);
                Ok(false)
            }
            kind => {
                // Multi geometries and collections: each part is a WKB geometry.
                let parts = r.count(h.le, 9)?;
                let count_at = out.len();
                put_u32(out, 0, h.le);
                let mut written = 0u32;
                for _ in 0..parts {
                    let at = out.len();
                    if self.geometry(r, out)? && kind == 6 {
                        out.truncate(at);
                    } else {
                        written += 1;
                    }
                }
                patch_u32(out, count_at, written, h.le);
                Ok(false)
            }
        }
    }
}

fn count_coords(g: &Geometry) -> u64 {
    match g {
        Geometry::Polygon { exterior, interiors } => {
            std::iter::once(exterior).chain(interiors).filter(|r| !r.0.is_empty()).map(|r| r.0.len() as u64 + 1).sum()
        }
        Geometry::MultiPolygon(ps) => {
            ps.iter().flat_map(|(e, h)| std::iter::once(e).chain(h)).filter(|r| !r.0.is_empty()).map(|r| r.0.len() as u64 + 1).sum()
        }
        Geometry::GeometryCollection(gs) => gs.iter().map(count_coords).sum(),
        g => g.all_coords().len() as u64,
    }
}

fn simplify_one(wkb: &[u8], options: &SimplifyOptions, s: &mut Scratch) -> Result<(Vec<u8>, SimplifyStats)> {
    if options.method == SimplifyMethod::PreserveTopology {
        let g = Geometry::from_wkb(wkb).map_err(|e| Error::InvalidData(format!("WKB: {e}")))?;
        let n = count_coords(&g);
        let mut stats = SimplifyStats { features: 1, vertices_in: n, ..Default::default() };
        if g.all_coords().iter().any(|c| !(c.x.is_finite() && c.y.is_finite())) {
            stats.vertices_out = n;
            stats.non_finite = 1;
            return Ok((wkb.to_vec(), stats));
        }
        let out = simplify_preserve_topology(&g, options.tolerance)?;
        stats.vertices_out = count_coords(&out);
        return Ok((out.to_wkb(), stats));
    }
    let mut run = Run { tolerance: options.tolerance, collapse: options.collapse, s, stats: SimplifyStats::default(), non_finite: false };
    let mut r = Reader { b: wkb, pos: 0 };
    let mut out = Vec::with_capacity(wkb.len());
    run.geometry(&mut r, &mut out)?;
    if r.pos != wkb.len() {
        return Err(wkb_error(r.pos, "trailing bytes after the geometry"));
    }
    let mut stats = run.stats;
    stats.features = 1;
    if run.non_finite {
        let n = stats.vertices_in;
        return Ok((wkb.to_vec(), SimplifyStats { features: 1, vertices_in: n, vertices_out: n, non_finite: 1, ..Default::default() }));
    }
    Ok((out, stats))
}

/// Simplifies one WKB geometry (see the module documentation).
///
/// # Errors
/// An invalid tolerance, or WKB that is malformed, has trailing bytes or holds a type other
/// than the seven simple feature types.
pub fn simplify_wkb(wkb: &[u8], options: &SimplifyOptions) -> Result<(Vec<u8>, SimplifyStats)> {
    options.validate()?;
    simplify_one(wkb, options, &mut Scratch::default())
}

/// Simplifies WKB geometries in parallel, keeping their order; `None` stays `None`.
///
/// # Errors
/// As [`simplify_wkb`], naming the first failing feature's index.
pub fn simplify_wkb_many<B: AsRef<[u8]> + Sync>(
    geometries: &[Option<B>],
    options: &SimplifyOptions,
) -> Result<(Vec<Option<Vec<u8>>>, SimplifyStats)> {
    options.validate()?;
    let results = crate::par::map_init(geometries, Scratch::default, |s, g| match g {
        None => Ok((None, SimplifyStats { features: 1, nulls: 1, ..Default::default() })),
        Some(b) => simplify_one(b.as_ref(), options, s).map(|(w, st)| (Some(w), st)),
    });
    let mut stats = SimplifyStats::default();
    let mut out = Vec::with_capacity(results.len());
    for (i, r) in results.into_iter().enumerate() {
        let (w, st) = r.map_err(|e| Error::InvalidData(format!("feature {i}: {e}")))?;
        stats += st;
        out.push(w);
    }
    Ok((out, stats))
}

/// Douglas-Peucker simplification of a geometry (see the module documentation), through WKB.
/// Z is kept; M is dropped (by wbvector's WKB writer).
pub fn simplify_douglas_peucker(g: &Geometry, tolerance: f64, collapse: RingCollapse) -> Result<Geometry> {
    let options = SimplifyOptions { tolerance, method: SimplifyMethod::DouglasPeucker, collapse };
    let (wkb, _) = simplify_wkb(&g.to_wkb(), &options)?;
    Geometry::from_wkb(&wkb).map_err(|e| Error::InvalidData(format!("WKB: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn le_line(kind: u32, pts: &[(f64, f64)]) -> Vec<u8> {
        let mut b = vec![1];
        b.extend_from_slice(&kind.to_le_bytes());
        b.extend_from_slice(&(pts.len() as u32).to_le_bytes());
        for p in pts {
            b.extend_from_slice(&p.0.to_le_bytes());
            b.extend_from_slice(&p.1.to_le_bytes());
        }
        b
    }

    fn polygon(rings: &[&[(f64, f64)]], le: bool) -> Vec<u8> {
        let u = |v: u32| if le { v.to_le_bytes() } else { v.to_be_bytes() };
        let f = |v: f64| if le { v.to_le_bytes() } else { v.to_be_bytes() };
        let mut b = vec![u8::from(le)];
        b.extend_from_slice(&u(3));
        b.extend_from_slice(&u(rings.len() as u32));
        for r in rings {
            b.extend_from_slice(&u(r.len() as u32));
            for p in *r {
                b.extend_from_slice(&f(p.0));
                b.extend_from_slice(&f(p.1));
            }
        }
        b
    }

    fn rings(wkb: &[u8]) -> Vec<Vec<(f64, f64)>> {
        let g = Geometry::from_wkb(wkb).unwrap();
        let ring = |r: &wbvector::Ring| {
            let mut v: Vec<(f64, f64)> = r.0.iter().map(|c| (c.x, c.y)).collect();
            if let Some(&first) = v.first() {
                v.push(first);
            }
            v
        };
        match g {
            Geometry::Polygon { exterior, interiors } => {
                if exterior.0.is_empty() {
                    Vec::new()
                } else {
                    std::iter::once(&exterior).chain(&interiors).map(ring).collect()
                }
            }
            Geometry::LineString(cs) => vec![cs.iter().map(|c| (c.x, c.y)).collect()],
            g => panic!("unexpected {g:?}"),
        }
    }

    fn opts(tolerance: f64) -> SimplifyOptions {
        SimplifyOptions::new(tolerance)
    }

    #[test]
    fn simplifies_lines_like_geos() {
        // shapely 2.2 / GEOS 3.14: LINESTRING (0 0, 2 0); the endpoints of a closed line stay.
        let (w, st) = simplify_wkb(&le_line(2, &[(0., 0.), (1., 0.05), (2., 0.)]), &opts(0.1)).unwrap();
        assert_eq!(rings(&w), vec![vec![(0., 0.), (2., 0.)]]);
        assert_eq!((st.vertices_in, st.vertices_out), (3, 2));
        let closed = [(5., 0.), (10., 0.), (10., 10.), (0., 10.), (0., 0.), (5., 0.)];
        let (w, _) = simplify_wkb(&le_line(2, &closed), &opts(0.1)).unwrap();
        assert_eq!(rings(&w), vec![closed.to_vec()]);
        let (w, _) = simplify_wkb(&le_line(2, &[(0., 0.), (1., 0.01), (0., 0.02), (0., 0.)]), &opts(0.1)).unwrap();
        assert_eq!(rings(&w), vec![vec![(0., 0.), (1., 0.01), (0., 0.)]]);
    }

    #[test]
    fn ring_start_moves_like_geos() {
        // GEOS: POLYGON ((10 0, 10 10, 0 10, 0 0, 10 0)).
        let p = polygon(&[&[(5., 0.), (10., 0.), (10., 10.), (0., 10.), (0., 0.), (5., 0.)]], true);
        let (w, _) = simplify_wkb(&p, &opts(0.1)).unwrap();
        assert_eq!(rings(&w), vec![vec![(10., 0.), (10., 10.), (0., 10.), (0., 0.), (10., 0.)]]);
        // At exactly the tolerance the start goes; just beyond it stays. A four-point ring keeps it.
        let p = polygon(&[&[(5., 0.1), (10., 0.), (5., 10.), (0., 0.), (5., 0.1)]], true);
        assert_eq!(rings(&simplify_wkb(&p, &opts(0.1)).unwrap().0), vec![vec![(10., 0.), (5., 10.), (0., 0.), (10., 0.)]]);
        let p = polygon(&[&[(5., 0.1000001), (10., 0.), (5., 10.), (0., 0.), (5., 0.1000001)]], true);
        assert_eq!(rings(&simplify_wkb(&p, &opts(0.1)).unwrap().0)[0].len(), 5);
        // Orientation and start are kept (a clockwise shell stays clockwise).
        let p = polygon(&[&[(0., 0.), (0., 10.), (10., 10.), (10., 0.), (5., 0.01), (0., 0.)]], true);
        assert_eq!(rings(&simplify_wkb(&p, &opts(0.1)).unwrap().0), vec![vec![(0., 0.), (0., 10.), (10., 10.), (10., 0.), (0., 0.)]]);
    }

    #[test]
    fn collapsed_rings_are_kept_or_dropped() {
        let shell: &[(f64, f64)] = &[(0., 0.), (10., 0.), (10., 10.), (0., 10.), (0., 0.)];
        let thin: &[(f64, f64)] = &[(1., 1.), (2., 1.), (2., 1.05), (1., 1.05), (1., 1.)];
        let hole: &[(f64, f64)] = &[(5., 5.), (6., 5.), (6., 6.), (5., 6.), (5., 5.)];
        let p = polygon(&[shell, thin, hole], true);
        let (w, st) = simplify_wkb(&p, &opts(0.1)).unwrap();
        assert_eq!(rings(&w), vec![shell.to_vec(), thin.to_vec(), hole.to_vec()]);
        assert_eq!((st.rings_kept, st.rings_dropped), (1, 0));
        let drop = SimplifyOptions { collapse: RingCollapse::Drop, ..opts(0.1) };
        let (w, st) = simplify_wkb(&p, &drop).unwrap();
        assert_eq!(rings(&w), vec![shell.to_vec(), hole.to_vec()]);
        assert_eq!((st.rings_kept, st.rings_dropped, st.polygons_emptied), (0, 1, 0));
        // A collapsed shell empties the polygon, holes and all.
        let p = polygon(&[thin, hole], true);
        let (w, st) = simplify_wkb(&p, &drop).unwrap();
        assert!(rings(&w).is_empty());
        assert_eq!((st.rings_dropped, st.polygons_emptied, st.vertices_in, st.vertices_out), (2, 1, 10, 0));
        // And leaves its multipolygon.
        let mut mp = vec![1u8];
        mp.extend_from_slice(&6u32.to_le_bytes());
        mp.extend_from_slice(&2u32.to_le_bytes());
        mp.extend(polygon(&[shell], true));
        mp.extend(polygon(&[thin], true));
        let (w, st) = simplify_wkb(&mp, &drop).unwrap();
        let Geometry::MultiPolygon(ps) = Geometry::from_wkb(&w).unwrap() else { panic!() };
        assert_eq!((ps.len(), st.polygons_emptied), (1, 1));
        let (w, _) = simplify_wkb(&mp, &opts(0.1)).unwrap();
        let Geometry::MultiPolygon(ps) = Geometry::from_wkb(&w).unwrap() else { panic!() };
        assert_eq!(ps.len(), 2);
    }

    #[test]
    fn keeps_byte_order_dimensions_and_srid() {
        let ring: &[(f64, f64)] = &[(5., 0.), (10., 0.), (10., 10.), (0., 10.), (0., 0.), (5., 0.)];
        let be = polygon(&[ring], false);
        let (w, _) = simplify_wkb(&be, &opts(0.1)).unwrap();
        assert_eq!(w[0], 0);
        assert_eq!(rings(&w), vec![vec![(10., 0.), (10., 10.), (0., 10.), (0., 0.), (10., 0.)]]);
        // EWKB LineString Z with an SRID: Z values are carried with their vertices.
        let mut b = vec![1u8];
        b.extend_from_slice(&(2u32 | 0x8000_0000 | 0x2000_0000).to_le_bytes());
        b.extend_from_slice(&2278u32.to_le_bytes());
        b.extend_from_slice(&3u32.to_le_bytes());
        for (x, y, z) in [(0., 0., 1.), (1., 0.05, 2.), (2., 0., 3.)] {
            for v in [x, y, z] {
                b.extend_from_slice(&f64::to_le_bytes(v));
            }
        }
        let (w, _) = simplify_wkb(&b, &opts(0.1)).unwrap();
        assert_eq!(&w[..13], &[&b[..9], &2u32.to_le_bytes()[..]].concat()[..]);
        let zs: Vec<f64> = w[13..].chunks(24).map(|c| f64::from_le_bytes(c[16..24].try_into().unwrap())).collect();
        assert_eq!(zs, vec![1., 3.]);
        // ISO Polygon ZM (3003) through the Geometry entry point.
        let g = Geometry::from_wkb(&{
            let mut b = vec![1u8];
            b.extend_from_slice(&3003u32.to_le_bytes());
            b.extend_from_slice(&1u32.to_le_bytes());
            b.extend_from_slice(&6u32.to_le_bytes());
            for (i, p) in ring.iter().enumerate() {
                for v in [p.0, p.1, i as f64, 10. * i as f64] {
                    b.extend_from_slice(&f64::to_le_bytes(v));
                }
            }
            b
        })
        .unwrap();
        let s = simplify_douglas_peucker(&g, 0.1, RingCollapse::Keep).unwrap();
        let Geometry::Polygon { exterior, .. } = s else { panic!() };
        assert_eq!(exterior.0.len(), 4);
        // Z is kept; wbvector's WKB writer drops M.
        assert_eq!(exterior.0[0].z, Some(1.));
    }

    #[test]
    fn many_keeps_order_nulls_and_reports_errors() {
        let line = le_line(2, &[(0., 0.), (1., 0.05), (2., 0.)]);
        let point = {
            let mut b = vec![1u8];
            b.extend_from_slice(&1u32.to_le_bytes());
            b.extend_from_slice(&f64::NAN.to_le_bytes());
            b.extend_from_slice(&f64::NAN.to_le_bytes());
            b
        };
        let items = vec![Some(line.clone()), None, Some(point.clone()), Some(line.clone())];
        let (out, st) = simplify_wkb_many(&items, &opts(0.1)).unwrap();
        assert_eq!(out[1], None);
        assert_eq!(out[2].as_deref(), Some(&point[..]));
        assert_eq!(out[0], out[3]);
        assert_eq!((st.features, st.nulls, st.vertices_in, st.vertices_out), (4, 1, 7, 5));
        let bad = vec![Some(line.clone()), Some(line[..20].to_vec())];
        let e = simplify_wkb_many(&bad, &opts(0.1)).unwrap_err().to_string();
        assert!(e.contains("feature 1") && e.contains("WKB at byte"), "{e}");
        assert!(simplify_wkb(&line[..3], &opts(0.1)).unwrap_err().to_string().contains("truncated"));
        let mut trailing = line.clone();
        trailing.push(0);
        assert!(simplify_wkb(&trailing, &opts(0.1)).is_err());
        assert!(simplify_wkb(&line, &opts(-1.0)).is_err());
        assert!(simplify_wkb(&line, &opts(f64::NAN)).is_err());
        // A huge count is refused before any allocation.
        let mut huge = line[..5].to_vec();
        huge.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(simplify_wkb(&huge, &opts(0.1)).is_err());
    }

    #[test]
    fn non_finite_features_pass_through() {
        let line = le_line(2, &[(0., 0.), (1., f64::NAN), (2., 0.), (3., 0.)]);
        let (w, st) = simplify_wkb(&line, &opts(0.1)).unwrap();
        assert_eq!(w, line);
        assert_eq!((st.non_finite, st.vertices_out), (1, 4));
    }

    #[test]
    fn matches_the_reference_section_distance() {
        // The inlined distance must equal `point_segment_distance` bit for bit.
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2000.0 - 1000.0
        };
        for _ in 0..200 {
            let pts: Vec<P> = (0..40).map(|_| (next(), next())).collect();
            for tol in [0.0, 1.0, 50.0, 400.0] {
                let (mut drop, mut keep) = (Vec::new(), Vec::new());
                douglas_peucker(&pts, tol, &mut drop, &mut keep);
                assert_eq!(keep, reference(&pts, tol), "tolerance {tol}");
            }
        }
        // A long noisy closed line takes the parallel path.
        let n = 5 * PARALLEL_SECTION;
        let mut pts: Vec<P> = (0..n)
            .map(|k| {
                let a = std::f64::consts::TAU * k as f64 / n as f64;
                let r = 10_000.0 + next() * 0.002;
                (r * a.cos(), r * a.sin())
            })
            .collect();
        pts.push(pts[0]);
        for tol in [0.1, 0.5, 2.0] {
            let (mut drop, mut keep) = (Vec::new(), Vec::new());
            douglas_peucker(&pts, tol, &mut drop, &mut keep);
            assert_eq!(keep, reference(&pts, tol), "long line, tolerance {tol}");
        }

        fn reference(pts: &[P], tol: f64) -> Vec<usize> {
            fn section(pts: &[P], i: usize, j: usize, tol: f64, use_pt: &mut [bool]) {
                if i + 1 == j {
                    return;
                }
                let (mut max, mut idx) = (-1.0, i);
                for (k, &p) in pts.iter().enumerate().take(j).skip(i + 1) {
                    let d = point_segment_distance(p, pts[i], pts[j]);
                    if d > max {
                        max = d;
                        idx = k;
                    }
                }
                if max <= tol {
                    use_pt[i + 1..j].fill(false);
                } else {
                    section(pts, i, idx, tol, use_pt);
                    section(pts, idx, j, tol, use_pt);
                }
            }
            let mut use_pt = vec![true; pts.len()];
            section(pts, 0, pts.len() - 1, tol, &mut use_pt);
            (0..pts.len()).filter(|&k| use_pt[k]).collect()
        }
    }

    #[test]
    fn preserve_topology_method() {
        let p = polygon(&[&[(5., 0.), (10., 0.), (10., 10.), (0., 10.), (0., 0.), (5., 0.)]], false);
        let o = SimplifyOptions { method: SimplifyMethod::PreserveTopology, ..opts(0.1) };
        let (w, st) = simplify_wkb(&p, &o).unwrap();
        assert_eq!(w[0], 1);
        // GEOS's TopologyPreservingSimplifier: POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0)).
        assert_eq!(rings(&w), vec![vec![(0., 0.), (10., 0.), (10., 10.), (0., 10.), (0., 0.)]]);
        assert_eq!((st.vertices_in, st.vertices_out), (6, 5));
        assert_eq!("dp".parse::<SimplifyMethod>().unwrap(), SimplifyMethod::DouglasPeucker);
        assert_eq!("Preserve_Topology".parse::<SimplifyMethod>().unwrap(), SimplifyMethod::PreserveTopology);
        assert!("x".parse::<SimplifyMethod>().is_err() && "x".parse::<RingCollapse>().is_err());
    }
}
