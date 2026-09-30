// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Local file reads with `O_DIRECT` (Linux only).
//!
//! Enabled by opening a dataset or file with the `file+direct://` URI scheme
//! instead of `file://`. Reads bypass the OS page cache: data goes straight
//! from the device into an aligned buffer owned by the returned [`Bytes`],
//! with no copy out of the page cache and no page-cache insertion.
//!
//! Buffered reads of large files at high concurrency spend most of their
//! kernel time spinning on page-cache locks (the memcg LRU lock and each
//! file's page-cache lock, taken for every folio readahead inserts). Direct
//! reads avoid both, and a single large `pread` is submitted to the block
//! layer all at once rather than one readahead window at a time.
//!
//! Trade-offs: nothing is cached, so re-reading a file always goes to the
//! device, and a warm page cache does not help. Reads of a file that has
//! dirty cached pages force those pages to be written back first, so use
//! this scheme for data that is not being written concurrently. Writes
//! through a `file+direct://` store are ordinary buffered writes.
//!
//! `O_DIRECT` requires the file offset, length and buffer address to be
//! aligned to the device's logical block size. Each read is widened to
//! [`DIRECT_IO_ALIGN`] boundaries, and the requested range is returned as a
//! slice of the aligned buffer. If the file system rejects `O_DIRECT` (e.g.
//! tmpfs), the reader falls back to buffered reads with a one-time warning.

use std::alloc::{Layout, alloc, dealloc};
use std::fs::{File, OpenOptions};
use std::io::ErrorKind;
use std::ops::Range;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytes::Bytes;
use futures::future::BoxFuture;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use object_store::path::Path;
use tokio::sync::OnceCell;
use tracing::instrument;

use crate::local::{LocalObjectReader, join_local_io, to_local_path};
use crate::object_store::DEFAULT_LOCAL_IO_PARALLELISM;
use crate::traits::Reader;
use crate::utils::tracking_store::IOTracker;

/// Alignment for direct reads: file offsets, lengths and buffer addresses are
/// multiples of this. 4 KiB satisfies devices with 512-byte and 4 KiB logical
/// blocks.
pub(crate) const DIRECT_IO_ALIGN: usize = 4096;

static FALLBACK_WARNED: AtomicBool = AtomicBool::new(false);

/// `range` widened outward to multiples of `align` (a power of two).
fn aligned_range(range: &Range<usize>, align: usize) -> Range<usize> {
    let start = range.start & !(align - 1);
    let end = range.end.div_ceil(align) * align;
    start..end
}

/// A heap buffer aligned to [`DIRECT_IO_ALIGN`], owned by the [`Bytes`] it is
/// wrapped in.
struct AlignedBuf {
    ptr: NonNull<u8>,
    layout: Layout,
    /// Bytes written so far; only this prefix is exposed through `AsRef`.
    filled: usize,
}

// SAFETY: `AlignedBuf` exclusively owns its allocation; nothing else aliases
// it, and it is only read (through `AsRef`) once filled.
unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    fn new(len: usize) -> std::io::Result<Self> {
        let layout = Layout::from_size_align(len, DIRECT_IO_ALIGN)
            .map_err(|err| std::io::Error::new(ErrorKind::InvalidInput, err))?;
        // `len` is non-zero: callers return early for empty ranges.
        debug_assert!(len > 0);
        // SAFETY: `layout` has a non-zero size.
        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            std::io::Error::new(
                ErrorKind::OutOfMemory,
                format!("failed to allocate {len} bytes for a direct read"),
            )
        })?;
        Ok(Self {
            ptr,
            layout,
            filled: 0,
        })
    }

    /// The not-yet-filled tail of the buffer, as a `pread` destination.
    fn unfilled_mut(&mut self) -> &mut [u8] {
        // SAFETY: in bounds of the `layout.size()`-byte allocation. The tail
        // may be uninitialized; it is only written (by `pread`) through this
        // slice, never read, until `filled` covers it. Same pattern as the
        // buffered reader's `BytesMut::set_len` before `read_exact_at`.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.ptr.as_ptr().add(self.filled),
                self.layout.size() - self.filled,
            )
        }
    }
}

impl AsRef<[u8]> for AlignedBuf {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: the first `filled` bytes were written by `pread`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.filled) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

/// Reads `range` from a file opened with `O_DIRECT`. Reads the enclosing
/// aligned range into an aligned buffer and returns `range` as a zero-copy
/// slice of it.
fn read_direct(file: &File, range: Range<usize>) -> std::io::Result<Bytes> {
    if range.is_empty() {
        return Ok(Bytes::new());
    }
    let aligned = aligned_range(&range, DIRECT_IO_ALIGN);
    let mut buf = AlignedBuf::new(aligned.len())?;
    while buf.filled < aligned.len() {
        let offset = (aligned.start + buf.filled) as u64;
        match file.read_at(buf.unfilled_mut(), offset) {
            // End of file: the aligned range may extend past it.
            Ok(0) => break,
            Ok(n) => {
                buf.filled += n;
                // A read that stops short of an alignment boundary ended at
                // end of file. Stop here: another read would start at an
                // unaligned offset, which O_DIRECT rejects with EINVAL.
                if buf.filled % DIRECT_IO_ALIGN != 0 {
                    break;
                }
            }
            Err(err) if err.kind() == ErrorKind::Interrupted => {}
            Err(err) => return Err(err),
        }
    }
    let filled = buf.filled;
    let needed = range.end - aligned.start;
    if filled < needed {
        return Err(std::io::Error::new(
            ErrorKind::UnexpectedEof,
            format!(
                "direct read of {range:?} reached end of file after {} bytes",
                (aligned.start + filled).saturating_sub(range.start)
            ),
        ));
    }
    let offset = range.start - aligned.start;
    Ok(Bytes::from_owner(buf).slice(offset..offset + range.len()))
}

/// Object reader for local files opened with `O_DIRECT`. See the module docs.
#[derive(Debug)]
pub(crate) struct DirectObjectReader {
    file: Arc<File>,
    path: Path,
    size: OnceCell<usize>,
    block_size: usize,
    io_tracker: Arc<IOTracker>,
}

impl DeepSizeOf for DirectObjectReader {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        // Skipping `file` as it should just be a file handle
        self.path.as_ref().deep_size_of_children(context)
    }
}

impl DirectObjectReader {
    /// Opens `path` with `O_DIRECT`, or falls back to a buffered
    /// [`LocalObjectReader`] if the file system does not support it.
    #[instrument(level = "debug")]
    pub(crate) async fn open_with_tracker(
        path: &Path,
        block_size: usize,
        known_size: Option<usize>,
        io_tracker: Arc<IOTracker>,
    ) -> Result<Box<dyn Reader>> {
        let local_path = to_local_path(path);
        let not_found_path = path.clone();
        let opened = tokio::task::spawn_blocking(move || {
            match OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(&local_path)
            {
                Ok(file) => Ok(Some(file)),
                // The file system does not support O_DIRECT.
                Err(err) if err.raw_os_error() == Some(libc::EINVAL) => Ok(None),
                Err(err) if err.kind() == ErrorKind::NotFound => {
                    Err(Error::not_found(not_found_path.to_string()))
                }
                Err(err) => Err(err.into()),
            }
        })
        .await??;

        let Some(file) = opened else {
            if !FALLBACK_WARNED.swap(true, Ordering::Relaxed) {
                log::warn!(
                    "file+direct: the file system does not support O_DIRECT for {path}; \
                     falling back to buffered reads"
                );
            }
            return LocalObjectReader::open_with_tracker(path, block_size, known_size, io_tracker)
                .await;
        };
        Ok(Box::new(Self {
            file: Arc::new(file),
            path: path.clone(),
            size: OnceCell::new_with(known_size),
            block_size,
            io_tracker,
        }))
    }
}

impl Reader for DirectObjectReader {
    fn path(&self) -> &Path {
        &self.path
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn io_parallelism(&self) -> usize {
        DEFAULT_LOCAL_IO_PARALLELISM
    }

    fn size(&self) -> BoxFuture<'_, object_store::Result<usize>> {
        Box::pin(async move {
            let file = self.file.clone();
            self.size
                .get_or_try_init(|| async move {
                    let metrics = self.io_tracker.begin_io("head");
                    let result =
                        join_local_io(tokio::task::spawn_blocking(move || file.metadata())).await;
                    metrics.record(&result, 0);
                    Ok(result?.len() as usize)
                })
                .await
                .cloned()
        })
    }

    #[instrument(level = "debug", skip(self))]
    fn get_range(&self, range: Range<usize>) -> BoxFuture<'static, object_store::Result<Bytes>> {
        let file = self.file.clone();
        let io_tracker = self.io_tracker.clone();
        let path = self.path.clone();
        let num_bytes = range.len() as u64;
        let range_u64 = (range.start as u64)..(range.end as u64);

        Box::pin(async move {
            let metrics = io_tracker.begin_io("get");
            let result =
                join_local_io(tokio::task::spawn_blocking(move || read_direct(&file, range)))
                    .await;
            metrics.record(&result, num_bytes);
            if result.is_ok() {
                io_tracker.record_read("get_range", path, num_bytes, Some(range_u64));
            }
            result
        })
    }

    #[instrument(level = "debug", skip(self))]
    fn get_all(&self) -> BoxFuture<'_, object_store::Result<Bytes>> {
        Box::pin(async move {
            let size = self.size().await?;
            let file = self.file.clone();
            let metrics = self.io_tracker.begin_io("get");
            let result =
                join_local_io(tokio::task::spawn_blocking(move || read_direct(&file, 0..size)))
                    .await;
            let num_bytes = result.as_ref().map_or(0, |bytes| bytes.len() as u64);
            metrics.record(&result, num_bytes);
            if let Ok(bytes) = &result {
                self.io_tracker
                    .record_read("get_all", self.path.clone(), bytes.len() as u64, None);
            }
            result
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object_store::ObjectStore;
    use rstest::rstest;
    use std::io::Write;

    #[rstest]
    #[case::first_byte(0..1, 0..4096)]
    #[case::one_block(0..4096, 0..4096)]
    #[case::straddles_boundary(1..4097, 0..8192)]
    #[case::aligned_second_block(4096..8192, 4096..8192)]
    #[case::inside_second_block(5000..5001, 4096..8192)]
    fn test_aligned_range(#[case] range: Range<usize>, #[case] expected: Range<usize>) {
        assert_eq!(aligned_range(&range, 4096), expected);
    }

    const TEST_FILE_SIZE: usize = 3 * DIRECT_IO_ALIGN + 123;

    /// A file of `size` bytes with a position-dependent pattern, in a temp dir
    /// under the current directory (often a disk file system that supports
    /// `O_DIRECT`, unlike a tmpfs `/tmp`).
    fn create_test_file(size: usize) -> (tempfile::TempDir, std::path::PathBuf, Vec<u8>) {
        let dir = tempfile::tempdir_in(".").unwrap();
        let path = dir.path().join("data.bin");
        let data: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
        let mut file = File::create(&path).unwrap();
        file.write_all(&data).unwrap();
        file.sync_all().unwrap();
        (dir, path, data)
    }

    fn open_direct(path: &std::path::Path) -> Option<File> {
        match OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
        {
            Ok(file) => Some(file),
            Err(err) if err.raw_os_error() == Some(libc::EINVAL) => None,
            Err(err) => panic!("open failed: {err}"),
        }
    }

    // Skipped (returns early) where the file system rejects O_DIRECT; the
    // reader falls back to buffered reads there, covered by the scheme test.
    #[rstest]
    #[case::whole_file(0..TEST_FILE_SIZE)]
    #[case::first_byte(0..1)]
    #[case::straddles_boundary(1..4097)]
    #[case::aligned_block(4096..8192)]
    #[case::to_end_of_file(100..TEST_FILE_SIZE)]
    #[case::last_byte(TEST_FILE_SIZE - 1..TEST_FILE_SIZE)]
    #[case::inside_one_block(4000..4200)]
    #[case::empty(5..5)]
    fn test_read_direct_range(#[case] range: Range<usize>) {
        let (_dir, path, data) = create_test_file(TEST_FILE_SIZE);
        let Some(file) = open_direct(&path) else {
            return;
        };
        let bytes = read_direct(&file, range.clone()).unwrap();
        assert_eq!(bytes.as_ref(), &data[range]);
    }

    #[test]
    fn test_read_direct_past_eof_fails() {
        let size = DIRECT_IO_ALIGN + 10;
        let (_dir, path, _data) = create_test_file(size);
        let Some(file) = open_direct(&path) else {
            return;
        };
        let err = read_direct(&file, 0..size + 1).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UnexpectedEof);
    }

    #[tokio::test]
    async fn test_file_direct_scheme_matches_data() {
        let size = 5 * DIRECT_IO_ALIGN + 777;
        let (_dir, path, data) = create_test_file(size);
        let uri = format!("file+direct://{}", path.canonicalize().unwrap().display());
        let (store, object_path) = ObjectStore::from_uri(&uri).await.unwrap();
        assert!(store.is_local());

        let reader = store.open(&object_path).await.unwrap();
        assert_eq!(reader.size().await.unwrap(), size);
        assert_eq!(reader.get_all().await.unwrap().as_ref(), data.as_slice());
        for range in [0..10, 4095..4097, 1000..size, size - 5..size] {
            let bytes = reader.get_range(range.clone()).await.unwrap();
            assert_eq!(bytes.as_ref(), &data[range.clone()], "range {range:?}");
        }

        let reader = store.open_with_size(&object_path, size).await.unwrap();
        let bytes = reader.get_range(123..size - 1).await.unwrap();
        assert_eq!(bytes.as_ref(), &data[123..size - 1]);
    }
}
