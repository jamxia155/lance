// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Pluggable allocator for the contiguous output buffer of a decoded batch.
//!
//! A batch that spans several pages is assembled by copying the per-page arrays into one
//! contiguous array. A host (e.g. lance-cuvs) can install an allocator so that, for batches of
//! `FixedSizeList<Float32>` without nulls, this buffer comes from a fixed pool (for instance of
//! pinned memory) instead of the global allocator. Anything the allocator declines, and every
//! other data type, takes the normal `concat` path.

use std::ptr::NonNull;
use std::sync::{Arc, Mutex, RwLock};

use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array};
use arrow_buffer::{Buffer, ScalarBuffer};
use arrow_schema::DataType;

/// A buffer on loan from a pool. `release` is called exactly once when the last Arrow
/// reference to the buffer is dropped.
pub struct OutputBuf {
    /// Start of the buffer; must be aligned to at least 64 bytes.
    pub ptr: NonNull<u8>,
    pub release: Box<dyn FnOnce() + Send + Sync>,
}

pub trait DecodeOutputAllocator: Send + Sync {
    /// A buffer of at least `len` bytes, or `None` to fall back to the heap.
    fn alloc(&self, len: usize) -> Option<OutputBuf>;
}

static ALLOCATOR: RwLock<Option<Arc<dyn DecodeOutputAllocator>>> = RwLock::new(None);

/// Installs (or, with `None`, removes) the allocator used for decoded batch buffers.
pub fn set_decode_output_allocator(allocator: Option<Arc<dyn DecodeOutputAllocator>>) {
    *ALLOCATOR.write().unwrap_or_else(|e| e.into_inner()) = allocator;
}

/// Owner handed to Arrow: returns the buffer to its pool on drop.
struct PooledOwner {
    release: Mutex<Option<Box<dyn FnOnce() + Send + Sync>>>,
}

impl Drop for PooledOwner {
    fn drop(&mut self) {
        let release = self
            .release
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(release) = release {
            release();
        }
    }
}

/// Concatenates `arrays` into a buffer from the installed allocator, or returns `None` if no
/// allocator is installed, the type or nulls are unsupported, or the allocator declines.
pub(crate) fn try_pooled_concat(arrays: &[ArrayRef], data_type: &DataType) -> Option<ArrayRef> {
    // Small batches are not worth pooling.
    const MIN_BYTES: usize = 1 << 20;
    let DataType::FixedSizeList(child_field, dimension) = data_type else {
        return None;
    };
    if child_field.data_type() != &DataType::Float32 || *dimension <= 0 {
        return None;
    }
    let allocator = ALLOCATOR
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()?;

    let mut lists = Vec::with_capacity(arrays.len());
    let mut rows = 0usize;
    for array in arrays {
        let list = array.as_fixed_size_list_opt()?;
        if list.null_count() != 0
            || list.values().null_count() != 0
            || list.value_length() != *dimension
        {
            return None;
        }
        list.values().as_primitive_opt::<Float32Type>()?;
        rows += list.len();
        lists.push(list);
    }
    let len_bytes = rows * (*dimension as usize) * std::mem::size_of::<f32>();
    if len_bytes < MIN_BYTES {
        return None;
    }
    let out = allocator.alloc(len_bytes)?;
    debug_assert_eq!(out.ptr.as_ptr() as usize % std::mem::align_of::<f32>(), 0);

    let mut offset = 0usize;
    for list in &lists {
        let values: &[f32] = list.values().as_primitive::<Float32Type>().values();
        // `values` is already restricted to this array's window.
        let n = list.len() * (*dimension as usize);
        // SAFETY: the destination holds `len_bytes` bytes, the sum of all `n * 4`, and the
        // source slice is a distinct allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                (out.ptr.as_ptr() as *mut f32).add(offset),
                n,
            );
        }
        offset += n;
    }

    let owner = Arc::new(PooledOwner {
        release: Mutex::new(Some(out.release)),
    });
    // SAFETY: `ptr` is valid for `len_bytes`, aligned for f32, and stays alive (and unaliased by
    // writers) until `owner` is dropped.
    let buffer = unsafe { Buffer::from_custom_allocation(out.ptr, len_bytes, owner) };
    let values = Float32Array::new(ScalarBuffer::from(buffer), None);
    Some(Arc::new(FixedSizeListArray::new(
        child_field.clone(),
        *dimension,
        Arc::new(values),
        None,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct HeapAllocator {
        live: Arc<AtomicUsize>,
    }

    impl DecodeOutputAllocator for HeapAllocator {
        fn alloc(&self, len: usize) -> Option<OutputBuf> {
            let layout = std::alloc::Layout::from_size_align(len, 4096).unwrap();
            // SAFETY: non-zero size (callers pass at least 1 MiB).
            let ptr = NonNull::new(unsafe { std::alloc::alloc(layout) })?;
            self.live.fetch_add(1, Ordering::SeqCst);
            let live = self.live.clone();
            let addr = ptr.as_ptr() as usize;
            Some(OutputBuf {
                ptr,
                release: Box::new(move || {
                    // SAFETY: allocated above with the same layout.
                    unsafe { std::alloc::dealloc(addr as *mut u8, layout) };
                    live.fetch_sub(1, Ordering::SeqCst);
                }),
            })
        }
    }

    fn fsl(start: f32, rows: usize, dim: i32) -> ArrayRef {
        let values =
            Float32Array::from_iter_values((0..rows * dim as usize).map(|i| start + i as f32));
        Arc::new(FixedSizeListArray::new(
            Arc::new(arrow_schema::Field::new("item", DataType::Float32, false)),
            dim,
            Arc::new(values),
            None,
        ))
    }

    #[test]
    fn pooled_concat_matches_concat_and_releases() {
        let live = Arc::new(AtomicUsize::new(0));
        set_decode_output_allocator(Some(Arc::new(HeapAllocator { live: live.clone() })));
        let dim = 128;
        let a = fsl(0.0, 2000, dim);
        // A sliced array exercises the window handling.
        let b = fsl(1.0e6, 2500, dim).slice(100, 2000);
        let arrays = vec![a, b];
        let data_type = arrays[0].data_type().clone();
        let pooled = try_pooled_concat(&arrays, &data_type).expect("pooled");
        set_decode_output_allocator(None);

        let refs = arrays.iter().map(|a| a.as_ref()).collect::<Vec<_>>();
        let expected = arrow_select::concat::concat(&refs).unwrap();
        assert_eq!(pooled.as_ref(), expected.as_ref());
        assert_eq!(live.load(Ordering::SeqCst), 1);
        drop(pooled);
        assert_eq!(live.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn declines_other_types() {
        let ints: ArrayRef = Arc::new(arrow_array::Int32Array::from(vec![1, 2, 3]));
        assert!(try_pooled_concat(&[ints.clone(), ints], &DataType::Int32).is_none());
    }
}
