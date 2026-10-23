// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Pluggable allocator for the large aligned buffers
//! used by `O_DIRECT` reads (see `direct.rs`). A host (e.g. lance-cuvs) can
//! register a fixed pool (for instance of pinned memory) so these ~32 MiB page
//! buffers do not go through the global allocator.

use std::ptr::NonNull;
use std::sync::{Arc, RwLock};

/// A buffer on loan from a pool. `release` is called exactly once when the
/// buffer is dropped.
pub struct PoolBuf {
    /// Start of the buffer; must be aligned to 4096 bytes.
    pub ptr: NonNull<u8>,
    pub release: Box<dyn FnOnce() + Send + Sync>,
}

pub trait DirectBufAllocator: Send + Sync {
    /// A 4096-aligned buffer of at least `len` bytes, or `None` to fall back
    /// to the heap.
    fn alloc(&self, len: usize) -> Option<PoolBuf>;
}

static ALLOCATOR: RwLock<Option<Arc<dyn DirectBufAllocator>>> = RwLock::new(None);

/// Installs (or, with `None`, removes) the allocator used for large direct reads.
pub fn set_direct_buf_allocator(allocator: Option<Arc<dyn DirectBufAllocator>>) {
    *ALLOCATOR.write().unwrap_or_else(|e| e.into_inner()) = allocator;
}

pub(crate) fn try_alloc(len: usize) -> Option<PoolBuf> {
    // Small reads (metadata, footers) stay on the heap.
    if len < (1 << 20) {
        return None;
    }
    let guard = ALLOCATOR.read().unwrap_or_else(|e| e.into_inner());
    guard.as_ref().and_then(|a| a.alloc(len))
}
