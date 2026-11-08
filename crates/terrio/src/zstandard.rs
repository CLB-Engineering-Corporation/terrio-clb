//! Zstandard: with the zstd C library with the `libzstd` feature, in native builds; otherwise
//! with the pure-Rust `ruzstd` (the `zstd-rust` crate, a maintained fork of `ruzstd` with the
//! `Default`, `Better` and `Best` levels). Both read every frame of a stream (skippable frames
//! skipped), and each reads what the other writes.

use std::io::{Error, ErrorKind, Read, Result};

// `decompress` reads EPT tiles (the `pointcloud` feature).
#[cfg(all(feature = "libzstd", not(target_family = "wasm")))]
#[cfg_attr(not(feature = "pointcloud"), allow(unused_imports))]
pub(crate) use libzstd::{compress, decompress};
#[cfg(not(all(feature = "libzstd", not(target_family = "wasm"))))]
#[cfg_attr(not(feature = "pointcloud"), allow(unused_imports))]
pub(crate) use pure::{compress, decompress};

/// The error of a stream decompressing to more than `limit` bytes.
fn too_long(limit: usize) -> Error {
    Error::new(ErrorKind::InvalidData, format!("Zstandard: decompresses to more than {limit} bytes"))
}

#[cfg(all(feature = "libzstd", not(target_family = "wasm")))]
mod libzstd {
    use super::*;

    /// Compresses `data` at `level`.
    pub(crate) fn compress(data: &[u8], level: i32) -> Result<Vec<u8>> {
        zstd::bulk::compress(data, level)
    }

    /// Decompresses every frame of `data`, to at most `limit` bytes (more is an error, rather
    /// than an allocation of whatever a small stream expands to).
    #[cfg_attr(not(feature = "pointcloud"), allow(dead_code))]
    pub(crate) fn decompress(data: &[u8], limit: usize) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        zstd::stream::read::Decoder::with_buffer(data)?.take((limit as u64).saturating_add(1)).read_to_end(&mut out)?;
        if out.len() > limit {
            return Err(too_long(limit));
        }
        Ok(out)
    }
}

/// `ruzstd`, used without `libzstd`, and in WebAssembly.
#[cfg_attr(all(feature = "libzstd", not(target_family = "wasm")), allow(dead_code))]
pub(crate) mod pure {
    use super::*;
    use ruzstd::decoding::errors::{FrameDecoderError, ReadFrameHeaderError};

    /// Compresses `data` at the `ruzstd` level closest to zstd's `level` (see [`ruzstd_level`]).
    pub(crate) fn compress(data: &[u8], level: i32) -> Result<Vec<u8>> {
        Ok(ruzstd::encoding::compress_to_vec(data, ruzstd_level(level)))
    }

    /// The `ruzstd` level for a zstd level: up to 2 `Fastest` (about zstd 1), 0 (zstd's default)
    /// and 3 to 5 `Default` (about zstd 3), 6 to 9 `Better`, from 10 on `Best`.
    pub(crate) fn ruzstd_level(level: i32) -> ruzstd::encoding::CompressionLevel {
        use ruzstd::encoding::CompressionLevel as L;
        match level {
            0 | 3..=5 => L::Default,
            ..=2 => L::Fastest,
            6..=9 => L::Better,
            _ => L::Best,
        }
    }

    /// Decompresses every frame of `data`, to at most `limit` bytes (more is an error).
    #[cfg_attr(not(feature = "pointcloud"), allow(dead_code))]
    pub(crate) fn decompress(data: &[u8], limit: usize) -> Result<Vec<u8>> {
        let out = frames(data, limit.saturating_add(1))?;
        if out.len() > limit {
            return Err(too_long(limit));
        }
        Ok(out)
    }

    /// The first `max` bytes `data` decompresses to, frame after frame (`ruzstd`'s decoder reads
    /// one), skippable frames skipped.
    pub(crate) fn frames(mut data: &[u8], max: usize) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        while !data.is_empty() && out.len() < max {
            match ruzstd::decoding::StreamingDecoder::new(&mut data) {
                Ok(frame) => {
                    frame.take((max - out.len()) as u64).read_to_end(&mut out)?;
                }
                Err(FrameDecoderError::ReadFrameHeaderError(ReadFrameHeaderError::SkipFrame { length, .. })) => {
                    data = data
                        .get(length as usize..)
                        .ok_or_else(|| Error::new(ErrorKind::UnexpectedEof, "Zstandard: truncated skippable frame"))?;
                }
                Err(e) => return Err(Error::new(ErrorKind::InvalidData, format!("Zstandard: {e}"))),
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A skippable frame of `n` bytes.
    fn skippable(n: usize) -> Vec<u8> {
        let mut out = 0x184D_2A50u32.to_le_bytes().to_vec();
        out.extend_from_slice(&(n as u32).to_le_bytes());
        out.extend(std::iter::repeat_n(7, n));
        out
    }

    fn data(n: usize, seed: u32) -> Vec<u8> {
        (0..n as u32).map(|i| (i.wrapping_mul(seed) >> 3) as u8).collect()
    }

    #[test]
    fn frames_are_read_one_after_another() {
        let (a, b) = (data(100_000, 31), data(70_000, 97));
        let stream = [pure::compress(&a, 3).unwrap(), skippable(9), pure::compress(&b, 3).unwrap()].concat();
        let both = [a.clone(), b.clone()].concat();
        assert_eq!(pure::decompress(&stream, both.len()).unwrap(), both);
        assert_eq!(decompress(&stream, both.len()).unwrap(), both);
        // At most `max` bytes, the rest not decoded.
        assert_eq!(pure::frames(&stream, 120_000).unwrap(), both[..120_000]);
        // A truncated skippable frame is an error.
        let cut = [pure::compress(&a, 3).unwrap(), skippable(9)[..12].to_vec()].concat();
        assert!(pure::decompress(&cut, usize::MAX).is_err());
    }

    #[test]
    fn unlimited_output_limit_does_not_overflow() {
        let input = b"a finite frame with an unlimited caller policy";
        let compressed = pure::compress(input, 1).unwrap();
        assert_eq!(decompress(&compressed, usize::MAX).unwrap(), input);
        assert_eq!(pure::decompress(&compressed, usize::MAX).unwrap(), input);
    }

    #[test]
    fn small_streams_expanding_past_the_limit_are_errors() {
        // 64 MiB of zeros in a few kilobytes.
        let bomb = pure::compress(&vec![0; 64 << 20], 3).unwrap();
        assert!(bomb.len() < 1 << 20);
        for e in [pure::decompress(&bomb, 1 << 20).unwrap_err(), decompress(&bomb, 1 << 20).unwrap_err()] {
            assert!(e.to_string().contains("more than 1048576 bytes"), "{e}");
        }
        assert_eq!(decompress(&bomb, 64 << 20).unwrap().len(), 64 << 20);
    }

    #[test]
    fn zstd_levels_map_to_the_nearest_ruzstd_level() {
        use ruzstd::encoding::CompressionLevel as L;
        let level = |l| format!("{:?}", pure::ruzstd_level(l));
        for (zstd, want) in [
            (i32::MIN, L::Fastest),
            (-5, L::Fastest),
            (0, L::Default),
            (1, L::Fastest),
            (2, L::Fastest),
            (3, L::Default),
            (5, L::Default),
            (6, L::Better),
            (9, L::Better),
            (10, L::Best),
            (22, L::Best),
            (i32::MAX, L::Best),
        ] {
            assert_eq!(level(zstd), format!("{want:?}"), "zstd level {zstd}");
        }
        // Every level round-trips, and compresses.
        let a = data(300_000, 13);
        for l in [1, 3, 9, 19] {
            let c = pure::compress(&a, l).unwrap();
            assert!(c.len() < a.len() / 4, "level {l}: {} bytes", c.len());
            assert_eq!(pure::decompress(&c, a.len()).unwrap(), a, "level {l}");
        }
    }

    /// Each implementation reads what the other writes, frames concatenated across them too.
    #[cfg(all(feature = "libzstd", not(target_family = "wasm")))]
    #[test]
    fn ruzstd_and_libzstd_read_each_other() {
        let (a, b) = (data(200_000, 13), data(50_000, 7));
        for level in [1, 3, 9, 19] {
            let c = libzstd::compress(&a, level).unwrap();
            assert_eq!(pure::decompress(&c, a.len()).unwrap(), a, "level {level}");
        }
        // Every `ruzstd` level (Fastest, Default, Better, Best).
        for level in [1, 3, 9, 19] {
            let p = pure::compress(&a, level).unwrap();
            assert_eq!(libzstd::decompress(&p, a.len()).unwrap(), a, "level {level}");
        }
        let mixed = [libzstd::compress(&a, 9).unwrap(), skippable(3), pure::compress(&b, 9).unwrap()].concat();
        let both = [a, b].concat();
        assert_eq!(pure::decompress(&mixed, both.len()).unwrap(), both);
        assert_eq!(libzstd::decompress(&mixed, both.len()).unwrap(), both);
    }
}
