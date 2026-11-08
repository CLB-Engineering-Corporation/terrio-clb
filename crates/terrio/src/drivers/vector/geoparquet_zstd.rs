//! Zstandard column chunks of Parquet files in pure Rust, without the `parquet` crate's Zstandard,
//! which is the zstd C library (enabled by the `libzstd-parquet` feature only, which then reads and writes
//! them instead). A chunk's pages are read as if they were not compressed, and then decompressed
//! here with `ruzstd` (`crate::zstandard::pure`). Other compressions stay with the crate.
//!
//! Writing is the mirror image ([`append_zstd`]): a row group's chunks are encoded uncompressed by
//! the crate, and each page is then compressed here and written again with its header, the chunk's
//! metadata (codec, sizes, offsets and page locations) following. Page boundaries, encodings,
//! dictionaries and statistics are the crate's, as they would be with its own Zstandard: only the
//! compressed bytes differ.
//!
//! Ported from opengeos/terrio (the Parquet readers of #444, and the writer of the GeoParquet
//! layer creation options), to parquet 59.

use std::sync::Arc;

use arrow_schema::{Fields, Schema};
use bytes::Bytes;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, RowGroups, RowSelection};
use parquet::arrow::parquet_to_arrow_field_levels;
use parquet::basic::Compression;
use parquet::column::page::{Page, PageIterator, PageMetadata, PageReader};
use parquet::errors::{ParquetError, Result};
use parquet::file::metadata::{ColumnChunkMetaData, ParquetMetaData, RowGroupMetaData};
use parquet::file::reader::{ChunkReader, Length};
use parquet::file::serialized_reader::SerializedPageReader;

/// One opened input shared by a metadata reader and a page reader, so that a file is not opened
/// again for its Zstandard chunks.
pub(super) struct SharedInput<R>(pub(super) Arc<R>);

impl<R> Clone for SharedInput<R> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<R: Length> Length for SharedInput<R> {
    fn len(&self) -> u64 {
        self.0.len()
    }
}

impl<R: ChunkReader> ChunkReader for SharedInput<R> {
    type T = R::T;

    fn get_read(&self, start: u64) -> Result<Self::T> {
        self.0.get_read(start)
    }

    fn get_bytes(&self, start: u64, length: usize) -> Result<Bytes> {
        self.0.get_bytes(start, length)
    }
}

/// Whether a chunk of the projected columns of `groups` is compressed with Zstandard.
pub(crate) fn has_zstd(meta: &ParquetMetaData, groups: &[usize], mask: &ProjectionMask) -> bool {
    groups.iter().any(|&g| {
        meta.row_group(g)
            .columns()
            .iter()
            .enumerate()
            .any(|(i, c)| mask.leaf_included(i) && matches!(c.compression(), Compression::ZSTD(_)))
    })
}

/// A reader of the rows of `groups`, like `ParquetRecordBatchReaderBuilder`'s, that also reads
/// Zstandard chunks.
pub(crate) fn reader<R: ChunkReader + 'static>(
    input: R,
    meta: &Arc<ParquetMetaData>,
    schema: &Schema,
    mask: ProjectionMask,
    groups: Vec<usize>,
    selection: Option<RowSelection>,
    batch_size: usize,
) -> Result<ParquetRecordBatchReader> {
    let fields: &Fields = schema.fields();
    let levels = parquet_to_arrow_field_levels(meta.file_metadata().schema_descr(), mask, Some(fields))?;
    // As `ParquetRecordBatchReaderBuilder` does, to avoid allocating a large buffer.
    let batch_size = batch_size.min(meta.file_metadata().num_rows() as usize).max(1);
    let row_groups = ZstdRowGroups { input: Arc::new(input), meta: meta.clone(), groups };
    ParquetRecordBatchReader::try_new_with_row_groups(&levels, &row_groups, batch_size, selection)
}

struct ZstdRowGroups<R> {
    input: Arc<R>,
    meta: Arc<ParquetMetaData>,
    groups: Vec<usize>,
}

impl<R: ChunkReader + 'static> RowGroups for ZstdRowGroups<R> {
    fn num_rows(&self) -> usize {
        self.groups.iter().map(|&g| self.meta.row_group(g).num_rows() as usize).sum()
    }

    fn column_chunks(&self, i: usize) -> Result<Box<dyn PageIterator>> {
        Ok(Box::new(Chunks { input: self.input.clone(), meta: self.meta.clone(), column: i, groups: self.groups.clone().into_iter() }))
    }

    fn row_groups(&self) -> Box<dyn Iterator<Item = &RowGroupMetaData> + '_> {
        Box::new(self.groups.iter().map(|&g| self.meta.row_group(g)))
    }

    fn metadata(&self) -> &ParquetMetaData {
        &self.meta
    }
}

/// The page readers of a column's chunks. Metadata is read without its page index, so pages are
/// found from their headers.
struct Chunks<R> {
    input: Arc<R>,
    meta: Arc<ParquetMetaData>,
    column: usize,
    groups: std::vec::IntoIter<usize>,
}

impl<R: ChunkReader + 'static> Chunks<R> {
    fn pages(&self, group: usize) -> Result<Box<dyn PageReader>> {
        let rg = self.meta.row_group(group);
        let chunk = rg.column(self.column);
        let rows = rg.num_rows() as usize;
        if !matches!(chunk.compression(), Compression::ZSTD(_)) {
            return Ok(Box::new(SerializedPageReader::new(self.input.clone(), chunk, rows, None)?));
        }
        let raw: ColumnChunkMetaData = chunk.clone().into_builder().set_compression(Compression::UNCOMPRESSED).build()?;
        let limit = usize::try_from(chunk.uncompressed_size()).map_err(|e| ParquetError::External(Box::new(e)))?;
        Ok(Box::new(ZstdPages { pages: SerializedPageReader::new(self.input.clone(), &raw, rows, None)?, limit }))
    }
}

impl<R: ChunkReader + 'static> Iterator for Chunks<R> {
    type Item = Result<Box<dyn PageReader>>;

    fn next(&mut self) -> Option<Self::Item> {
        let group = self.groups.next()?;
        Some(self.pages(group))
    }
}

impl<R: ChunkReader + 'static> PageIterator for Chunks<R> {}

/// The pages of a Zstandard chunk, decompressed.
struct ZstdPages<R: ChunkReader> {
    pages: SerializedPageReader<R>,
    /// The chunk's uncompressed size: no page decompresses to more.
    limit: usize,
}

impl<R: ChunkReader> ZstdPages<R> {
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        // A page with no non-null values may have no data at all.
        if data.is_empty() {
            return Ok(Vec::new());
        }
        crate::zstandard::pure::decompress(data, self.limit).map_err(|e| ParquetError::External(Box::new(e)))
    }

    fn page(&self, page: Page) -> Result<Page> {
        Ok(match page {
            Page::DataPage { buf, num_values, encoding, def_level_encoding, rep_level_encoding, statistics } => {
                let buf = Bytes::from(self.decompress(&buf)?);
                Page::DataPage { buf, num_values, encoding, def_level_encoding, rep_level_encoding, statistics }
            }
            // The levels of a v2 page come first, never compressed.
            Page::DataPageV2 {
                buf,
                num_values,
                encoding,
                num_nulls,
                num_rows,
                def_levels_byte_len,
                rep_levels_byte_len,
                is_compressed,
                statistics,
            } => {
                let levels = def_levels_byte_len as usize + rep_levels_byte_len as usize;
                let buf = match is_compressed {
                    true if levels > buf.len() => return Err(ParquetError::General("Invalid page header".to_string())),
                    true => {
                        let values = self.decompress(&buf[levels..])?;
                        Bytes::from([&buf[..levels], &values].concat())
                    }
                    false => buf,
                };
                Page::DataPageV2 {
                    buf,
                    num_values,
                    encoding,
                    num_nulls,
                    num_rows,
                    def_levels_byte_len,
                    rep_levels_byte_len,
                    is_compressed: false,
                    statistics,
                }
            }
            Page::DictionaryPage { buf, num_values, encoding, is_sorted } => {
                Page::DictionaryPage { buf: Bytes::from(self.decompress(&buf)?), num_values, encoding, is_sorted }
            }
        })
    }
}

impl<R: ChunkReader> Iterator for ZstdPages<R> {
    type Item = Result<Page>;

    fn next(&mut self) -> Option<Self::Item> {
        self.get_next_page().transpose()
    }
}

impl<R: ChunkReader> PageReader for ZstdPages<R> {
    fn get_next_page(&mut self) -> Result<Option<Page>> {
        self.pages.get_next_page()?.map(|page| self.page(page)).transpose()
    }

    fn peek_next_page(&mut self) -> Result<Option<PageMetadata>> {
        self.pages.peek_next_page()
    }

    fn skip_next_page(&mut self) -> Result<()> {
        self.pages.skip_next_page()
    }

    fn at_record_boundary(&mut self) -> Result<bool> {
        self.pages.at_record_boundary()
    }
}

/// Appends a row group's column chunks, encoded without compression, to `group`, every page
/// compressed with Zstandard at `level` by `ruzstd` (see [`crate::zstandard::pure::compress`]). `schema` and `props`
/// are the file's.
pub(crate) fn append_zstd<W: std::io::Write + Send>(
    chunks: Vec<parquet::arrow::arrow_writer::ArrowColumnChunk>,
    group: &mut parquet::file::writer::SerializedRowGroupWriter<'_, W>,
    schema: parquet::schema::types::TypePtr,
    props: parquet::file::properties::WriterPropertiesPtr,
    level: i32,
) -> Result<()> {
    use parquet::file::writer::SerializedFileWriter;
    // The chunks' bytes are reachable only through a row group: lay them out in a scratch file
    // first, keeping what their metadata does not carry.
    let kept: Vec<_> = chunks
        .iter()
        .map(|c| {
            let close = c.close();
            (close.rows_written, close.column_index.clone(), close.offset_index.clone())
        })
        .collect();
    let mut scratch = SerializedFileWriter::new(Vec::new(), schema, props)?;
    let mut raw = scratch.next_row_group()?;
    for chunk in chunks {
        chunk.append_to_row_group(&mut raw)?;
    }
    let meta = raw.close()?;
    let bytes = Arc::new(Bytes::from(scratch.into_inner()?));
    let rows = usize::try_from(meta.num_rows()).map_err(|e| ParquetError::External(Box::new(e)))?;
    let compressed = crate::par::map_range(meta.num_columns(), |i| compress_chunk(&bytes, meta.column(i), rows, level));
    for (chunk, (rows_written, column_index, offset_index)) in compressed.into_iter().zip(kept) {
        let (data, metadata, locations) = chunk?;
        let offset_index = offset_index.map(|mut index| {
            for (location, (offset, size)) in index.page_locations.iter_mut().zip(&locations) {
                location.offset = *offset;
                location.compressed_page_size = *size;
            }
            index
        });
        let close = parquet::column::writer::ColumnCloseResult {
            bytes_written: data.len() as u64,
            rows_written,
            metadata,
            bloom_filter: None,
            column_index,
            offset_index,
        };
        group.append_column(&data, close)?;
    }
    Ok(())
}

/// A compressed chunk: its bytes (from offset 0), metadata, and each data page's offset and size
/// (header included), for the page index.
type CompressedChunk = (Bytes, ColumnChunkMetaData, Vec<(i64, i32)>);

/// The pages of the uncompressed chunk `chunk` of `input`, compressed with Zstandard.
fn compress_chunk(input: &Arc<Bytes>, chunk: &ColumnChunkMetaData, rows: usize, level: i32) -> Result<CompressedChunk> {
    use parquet::column::page::{CompressedPage, PageWriter};
    use parquet::file::writer::{SerializedPageWriter, TrackedWrite};
    let compress = |data: &[u8]| -> Result<Bytes> {
        crate::zstandard::pure::compress(data, level).map(Bytes::from).map_err(|e| ParquetError::External(Box::new(e)))
    };
    let mut sink = TrackedWrite::new(Vec::with_capacity(chunk.compressed_size().max(0) as usize / 2));
    let mut writer = SerializedPageWriter::new(&mut sink);
    let (mut dictionary, mut first_data, mut locations) = (None, None, Vec::new());
    let (mut compressed, mut uncompressed) = (0i64, 0i64);
    for page in SerializedPageReader::new(input.clone(), chunk, rows, None)? {
        let page = page?;
        let size = page.buffer().len();
        let page = match page {
            Page::DataPage { buf, num_values, encoding, def_level_encoding, rep_level_encoding, statistics } => {
                Page::DataPage { buf: compress(&buf)?, num_values, encoding, def_level_encoding, rep_level_encoding, statistics }
            }
            // The levels of a v2 page stay uncompressed, in front.
            Page::DataPageV2 {
                buf,
                num_values,
                encoding,
                num_nulls,
                num_rows,
                def_levels_byte_len,
                rep_levels_byte_len,
                statistics,
                ..
            } => {
                let levels = def_levels_byte_len as usize + rep_levels_byte_len as usize;
                let (head, values) = buf.split_at_checked(levels).ok_or_else(|| ParquetError::General("Invalid page header".into()))?;
                Page::DataPageV2 {
                    buf: Bytes::from([head, &compress(values)?].concat()),
                    num_values,
                    encoding,
                    num_nulls,
                    num_rows,
                    def_levels_byte_len,
                    rep_levels_byte_len,
                    is_compressed: true,
                    statistics,
                }
            }
            Page::DictionaryPage { buf, num_values, encoding, is_sorted } => {
                Page::DictionaryPage { buf: compress(&buf)?, num_values, encoding, is_sorted }
            }
        };
        let is_dictionary = page.is_dictionary_page();
        let spec = writer.write_page(CompressedPage::new(page, size))?;
        let offset = i64::try_from(spec.offset).map_err(|e| ParquetError::External(Box::new(e)))?;
        let written = i32::try_from(spec.compressed_size).map_err(|e| ParquetError::External(Box::new(e)))?;
        if is_dictionary {
            dictionary = Some(offset);
        } else {
            first_data.get_or_insert(offset);
            locations.push((offset, written));
        }
        compressed += i64::from(written);
        uncompressed += spec.uncompressed_size as i64;
    }
    let data = Bytes::from(sink.into_inner()?);
    let metadata = chunk
        .clone()
        .into_builder()
        .set_compression(Compression::ZSTD(parquet::basic::ZstdLevel::try_new(level.clamp(1, 22))?))
        .set_total_compressed_size(compressed)
        .set_total_uncompressed_size(uncompressed)
        .set_dictionary_page_offset(dictionary)
        .set_data_page_offset(first_data.unwrap_or(0))
        .build()?;
    Ok((data, metadata, locations))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::fs::File;
    use std::path::PathBuf;

    use super::super::geoparquet::ParquetStream;

    /// How many times reading a fixture opens it.
    fn opens(file: &str, bbox: Option<(f64, f64, f64, f64)>) -> u32 {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/parquet_zstd").join(file);
        let count = Cell::new(0);
        let open = || {
            count.set(count.get() + 1);
            Ok(File::open(&path)?)
        };
        assert!(ParquetStream::open(open, "t", None, bbox, 64, None).is_ok());
        count.get()
    }

    /// Zstandard chunks open a file no more often than others, so a remote file's footer is not
    /// fetched again.
    #[test]
    fn zstd_chunks_add_no_opens() {
        let bbox = Some((-5.5, -2.5, 5.5, 2.5));
        for file in ["snappy.parquet", "zstd_v1.parquet", "zstd_mixed.parquet"] {
            assert_eq!(opens(file, None), 1, "{file}");
            assert_eq!(opens(file, bbox), 2, "{file}");
        }
    }
}
