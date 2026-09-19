// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::buffer::NullBuffer;
use arrow::compute::cast;
use arrow_array::ArrowPrimitiveType;
use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use arrow_array::{Array, ArrayRef, BooleanArray, FixedSizeListArray, cast::AsArray};
use arrow_schema::{DataType, Field};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use lance_io::encodings::plain::bytes_to_array;
use lance_linalg::distance::DistanceType;
use prost::bytes;
use std::sync::LazyLock;
use std::{ops::Range, sync::Arc};

use super::pb;
use crate::pb::Tensor;
use crate::vector::flat::storage::FlatBinStorage;
use crate::vector::flat::storage::FlatFloatStorage;
use crate::vector::hnsw::HNSW;
use crate::vector::hnsw::builder::{HnswBuildParams, HnswQueryParams};
use crate::vector::v3::subindex::IvfSubIndex;

enum SimpleIndexStatus {
    Auto,
    Enabled,
    Disabled,
}

static USE_HNSW_SPEEDUP_INDEXING: LazyLock<SimpleIndexStatus> = LazyLock::new(|| {
    if let Ok(v) = std::env::var("LANCE_USE_HNSW_SPEEDUP_INDEXING") {
        if v == "enabled" {
            SimpleIndexStatus::Enabled
        } else if v == "disabled" {
            SimpleIndexStatus::Disabled
        } else {
            SimpleIndexStatus::Auto
        }
    } else {
        SimpleIndexStatus::Auto
    }
});

#[derive(Debug)]
pub struct SimpleIndex {
    store: SimpleStore,
    index: HNSW,
}

#[derive(Debug)]
enum SimpleStore {
    Float(FlatFloatStorage),
    Binary(FlatBinStorage),
}

impl SimpleIndex {
    fn try_new(store: SimpleStore) -> Result<Self> {
        let hnsw = match &store {
            SimpleStore::Float(store) => HNSW::index_vectors(
                store,
                HnswBuildParams::default().ef_construction(15).num_edges(12),
            )?,
            SimpleStore::Binary(store) => HNSW::index_vectors(
                store,
                HnswBuildParams::default().ef_construction(15).num_edges(12),
            )?,
        };
        Ok(Self { store, index: hnsw })
    }

    // train HNSW over the centroids to speed up finding the nearest clusters,
    // only train if all conditions are met:
    //  - the centroids are float16/float32 or uint8 with hamming distance
    //  - `num_centroids * dimension >= 1_000_000`
    //      we benchmarked that it's 2x faster in the case of 1024 centroids and 1024 dimensions,
    //      so set the threshold to 1_000_000.
    pub fn may_train_index(
        centroids: ArrayRef,
        dimension: usize,
        distance_type: DistanceType,
    ) -> Result<Option<Self>> {
        match *USE_HNSW_SPEEDUP_INDEXING {
            SimpleIndexStatus::Auto => {
                if centroids.len() < 1_000_000 {
                    return Ok(None);
                }
            }
            SimpleIndexStatus::Disabled => return Ok(None),
            _ => {}
        }

        let store = match (centroids.data_type(), distance_type) {
            (DataType::Float16 | DataType::Float32 | DataType::Float64, _) => {
                let fsl = FixedSizeListArray::try_new_from_values(centroids, dimension as i32)?;
                SimpleStore::Float(FlatFloatStorage::new(fsl, distance_type))
            }
            (DataType::UInt8, DistanceType::Hamming) => {
                let fsl = FixedSizeListArray::try_new_from_values(centroids, dimension as i32)?;
                SimpleStore::Binary(FlatBinStorage::new(fsl, distance_type))
            }
            _ => return Ok(None),
        };
        Self::try_new(store).map(Some)
    }

    pub(crate) fn search(&self, query: ArrayRef) -> Result<(u32, f32)> {
        let params = HnswQueryParams {
            ef: 15,
            lower_bound: None,
            upper_bound: None,
            dist_q_c: 0.0,
        };
        let res = match &self.store {
            SimpleStore::Float(store) => self.index.search_basic(query, 1, &params, None, store)?,
            SimpleStore::Binary(store) => {
                let query = if query.data_type() == &DataType::UInt8 {
                    query
                } else {
                    cast(&query, &DataType::UInt8).map_err(|e| Error::index(e.to_string()))?
                };
                self.index.search_basic(query, 1, &params, None, store)?
            }
        };
        Ok((res[0].id, res[0].dist.0))
    }
}

#[inline]
pub(crate) fn do_prefetch<T>(ptrs: Range<*const T>) {
    // TODO use rust intrinsics instead of x86 intrinsics
    // TODO finish this
    unsafe {
        let (ptr, end_ptr) = (ptrs.start as *const i8, ptrs.end as *const i8);
        let mut current_ptr = ptr;
        while current_ptr < end_ptr {
            const CACHE_LINE_SIZE: usize = 64;
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
                _mm_prefetch(current_ptr, _MM_HINT_T0);
            }
            current_ptr = current_ptr.add(CACHE_LINE_SIZE);
        }
    }
}

impl From<pb::tensor::DataType> for DataType {
    fn from(dt: pb::tensor::DataType) -> Self {
        match dt {
            pb::tensor::DataType::Uint8 => Self::UInt8,
            pb::tensor::DataType::Uint16 => Self::UInt16,
            pb::tensor::DataType::Uint32 => Self::UInt32,
            pb::tensor::DataType::Uint64 => Self::UInt64,
            pb::tensor::DataType::Float16 => Self::Float16,
            pb::tensor::DataType::Float32 => Self::Float32,
            pb::tensor::DataType::Float64 => Self::Float64,
            pb::tensor::DataType::Bfloat16 => unimplemented!(),
        }
    }
}

impl TryFrom<&DataType> for pb::tensor::DataType {
    type Error = Error;

    fn try_from(dt: &DataType) -> Result<Self> {
        match dt {
            DataType::UInt8 => Ok(Self::Uint8),
            DataType::UInt16 => Ok(Self::Uint16),
            DataType::UInt32 => Ok(Self::Uint32),
            DataType::UInt64 => Ok(Self::Uint64),
            DataType::Float16 => Ok(Self::Float16),
            DataType::Float32 => Ok(Self::Float32),
            DataType::Float64 => Ok(Self::Float64),
            _ => Err(Error::index(format!(
                "pb tensor type not supported: {:?}",
                dt
            ))),
        }
    }
}

impl TryFrom<DataType> for pb::tensor::DataType {
    type Error = Error;

    fn try_from(dt: DataType) -> Result<Self> {
        (&dt).try_into()
    }
}

impl TryFrom<&FixedSizeListArray> for pb::Tensor {
    type Error = Error;

    fn try_from(array: &FixedSizeListArray) -> Result<Self> {
        let mut tensor = Self::default();
        tensor.data_type = pb::tensor::DataType::try_from(array.value_type())? as i32;
        tensor.shape = vec![Array::len(array) as u32, array.value_length() as u32];
        let flat_array = array.values();
        tensor.data = flat_array.into_data().buffers()[0].to_vec();
        Ok(tensor)
    }
}

impl TryFrom<&pb::Tensor> for FixedSizeListArray {
    type Error = Error;

    fn try_from(tensor: &Tensor) -> Result<Self> {
        if tensor.shape.len() != 2 {
            return Err(Error::index(format!(
                "only accept 2-D tensor shape, got: {:?}",
                tensor.shape
            )));
        }
        let dim = tensor.shape[1] as usize;
        let num_rows = tensor.shape[0] as usize;

        let data = bytes::Bytes::from(tensor.data.clone());
        let flat_array = bytes_to_array(
            &DataType::from(pb::tensor::DataType::try_from(tensor.data_type).unwrap()),
            data,
            dim * num_rows,
            0,
        )?;

        if flat_array.len() != dim * num_rows {
            return Err(Error::index(format!(
                "Tensor shape {:?} does not match to data len: {}",
                tensor.shape,
                flat_array.len()
            )));
        }

        let field = Field::new("item", flat_array.data_type().clone(), true);
        Ok(Self::try_new(
            Arc::new(field),
            dim as i32,
            flat_array,
            None,
        )?)
    }
}

const DEFAULT_IS_FINITE_THREADS: usize = 8;
// Below this row count, threading overhead isn't worth it -- run single-threaded.
const IS_FINITE_MIN_ROWS_FOR_THREADING: usize = 4096;

fn is_finite_threads_from_env() -> usize {
    std::env::var("LANCE_IS_FINITE_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|threads| *threads > 0)
        .unwrap_or(DEFAULT_IS_FINITE_THREADS)
}

/// Check if all vectors in the FixedSizeListArray are finite
/// null values are considered as not finite
/// returns a BooleanArray
/// with the same length as the FixedSizeListArray
/// with true for finite values and false for non-finite values
pub fn is_finite(fsl: &FixedSizeListArray) -> BooleanArray {
    match fsl.values().data_type() {
        DataType::Float16 => is_finite_typed::<Float16Type>(fsl),
        DataType::Float32 => is_finite_typed::<Float32Type>(fsl),
        DataType::Float64 => is_finite_typed::<Float64Type>(fsl),
        _ => is_finite_no_float(fsl),
    }
}

/// `is_finite` for a float child type: every row must have no nulls (row-level
/// or within the row's `dim` child values) and every value must be finite.
fn is_finite_typed<T: ArrowPrimitiveType>(fsl: &FixedSizeListArray) -> BooleanArray
where
    T::Native: num_traits::Float,
{
    let dim = fsl.value_length() as usize;
    let values = fsl.values().as_primitive::<T>();
    let data = values.values();
    let child_nulls = values.nulls().cloned();
    let row_nulls = fsl.nulls().cloned();

    compute_row_parallel(fsl.len(), move |row| {
        if row_nulls.as_ref().is_some_and(|nulls| !nulls.is_valid(row)) {
            return false;
        }
        let start = row * dim;
        if let Some(nulls) = &child_nulls
            && (start..start + dim).any(|i| !nulls.is_valid(i))
        {
            return false;
        }
        data[start..start + dim].iter().all(|v| v.is_finite())
    })
}

/// `is_finite` for a non-float child type: finiteness isn't meaningful, so
/// this only checks for nulls, matching the original per-row
/// `Array::null_count(&v) == 0` check.
fn is_finite_no_float(fsl: &FixedSizeListArray) -> BooleanArray {
    let dim = fsl.value_length() as usize;
    let child_nulls = fsl.values().nulls().cloned();
    let row_nulls = fsl.nulls().cloned();

    compute_row_parallel(fsl.len(), move |row| {
        if row_nulls.as_ref().is_some_and(|nulls| !nulls.is_valid(row)) {
            return false;
        }
        let start = row * dim;
        child_nulls
            .as_ref()
            .is_none_or(|nulls| (start..start + dim).all(|i| nulls.is_valid(i)))
    })
}

/// Runs `row_is_ok(row)` for every row in `[0, num_rows)`, in parallel across
/// `LANCE_IS_FINITE_THREADS` (default 8) threads for large enough inputs,
/// via disjoint output slices -- no locking needed since each thread only
/// ever writes its own chunk.
fn compute_row_parallel(
    num_rows: usize,
    row_is_ok: impl Fn(usize) -> bool + Sync,
) -> BooleanArray {
    let mut result = vec![false; num_rows];
    if num_rows == 0 {
        return BooleanArray::from(result);
    }

    let num_threads = if num_rows < IS_FINITE_MIN_ROWS_FOR_THREADING {
        1
    } else {
        is_finite_threads_from_env().min(num_rows)
    };

    if num_threads <= 1 {
        for (row, out) in result.iter_mut().enumerate() {
            *out = row_is_ok(row);
        }
    } else {
        let chunk_len = num_rows.div_ceil(num_threads);
        let row_is_ok = &row_is_ok;
        std::thread::scope(|scope| {
            for (chunk_idx, out_chunk) in result.chunks_mut(chunk_len).enumerate() {
                let start = chunk_idx * chunk_len;
                scope.spawn(move || {
                    for (i, out) in out_chunk.iter_mut().enumerate() {
                        *out = row_is_ok(start + i);
                    }
                });
            }
        });
    }

    BooleanArray::from(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{Float16Array, Float32Array, Float64Array, UInt8Array};
    use half::f16;
    use lance_arrow::FixedSizeListArrayExt;
    use num_traits::identities::Zero;

    use arrow::compute::cast;
    use rstest::rstest;

    fn build_index(centroids: ArrayRef, dim: usize) -> SimpleIndex {
        let f32_centroids = cast(&centroids, &DataType::Float32).unwrap();
        let fsl = FixedSizeListArray::try_new_from_values(f32_centroids, dim as i32).unwrap();
        let store = SimpleStore::Float(FlatFloatStorage::new(fsl, DistanceType::L2));
        SimpleIndex::try_new(store).unwrap()
    }

    fn build_binary_index(centroids: ArrayRef, dim: usize) -> SimpleIndex {
        let u8_centroids = if centroids.data_type() == &DataType::UInt8 {
            centroids
        } else {
            cast(&centroids, &DataType::UInt8).unwrap()
        };
        let fsl = FixedSizeListArray::try_new_from_values(u8_centroids, dim as i32).unwrap();
        let store = SimpleStore::Binary(FlatBinStorage::new(fsl, DistanceType::Hamming));
        SimpleIndex::try_new(store).unwrap()
    }

    #[rstest]
    #[case::f16(Arc::new(Float16Array::from(
        (0..100).flat_map(|i| std::iter::repeat_n(f16::from_f32(i as f32), 16)).collect::<Vec<_>>(),
    )) as ArrayRef)]
    #[case::f32(Arc::new(Float32Array::from(
        (0..100).flat_map(|i| std::iter::repeat_n(i as f32, 16)).collect::<Vec<_>>(),
    )) as ArrayRef)]
    fn test_simple_index_nearest_centroid(#[case] centroids: ArrayRef) {
        let index = build_index(centroids, 16);
        let query: ArrayRef = Arc::new(Float32Array::from(vec![42.1f32; 16]));
        let (id, _) = index.search(query).unwrap();
        assert_eq!(id, 42);
    }

    #[test]
    fn test_simple_index_nearest_centroid_binary() {
        let centroids: ArrayRef = Arc::new(UInt8Array::from(
            (0..100)
                .flat_map(|i| std::iter::repeat_n(i as u8, 16))
                .collect::<Vec<_>>(),
        ));
        let index = build_binary_index(centroids, 16);
        let query: ArrayRef = Arc::new(UInt8Array::from(vec![42u8; 16]));
        let (id, dist) = index.search(query).unwrap();
        assert_eq!(id, 42);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_simple_index_rejects_f64() {
        let centroids: ArrayRef = Arc::new(Float64Array::from(vec![0.0; 1600]));
        let result = SimpleIndex::may_train_index(centroids, 16, DistanceType::L2).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_simple_index_rejects_uint8_non_hamming() {
        let centroids: ArrayRef = Arc::new(UInt8Array::from(vec![0u8; 1600]));
        let result = SimpleIndex::may_train_index(centroids, 16, DistanceType::L2).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_fsl_to_tensor() {
        let fsl =
            FixedSizeListArray::try_new_from_values(Float16Array::from(vec![f16::zero(); 20]), 5)
                .unwrap();
        let tensor = pb::Tensor::try_from(&fsl).unwrap();
        assert_eq!(tensor.data_type, pb::tensor::DataType::Float16 as i32);
        assert_eq!(tensor.shape, vec![4, 5]);
        assert_eq!(tensor.data.len(), 20 * 2);

        let fsl =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0; 20]), 5).unwrap();
        let tensor = pb::Tensor::try_from(&fsl).unwrap();
        assert_eq!(tensor.data_type, pb::tensor::DataType::Float32 as i32);
        assert_eq!(tensor.shape, vec![4, 5]);
        assert_eq!(tensor.data.len(), 20 * 4);

        let fsl =
            FixedSizeListArray::try_new_from_values(Float64Array::from(vec![0.0; 20]), 5).unwrap();
        let tensor = pb::Tensor::try_from(&fsl).unwrap();
        assert_eq!(tensor.data_type, pb::tensor::DataType::Float64 as i32);
        assert_eq!(tensor.shape, vec![4, 5]);
        assert_eq!(tensor.data.len(), 20 * 8);
    }

    #[test]
    fn test_is_finite_f32_basic() {
        // 3 rows of dim 2: all-finite, contains NaN, contains +inf.
        let values = Float32Array::from(vec![1.0, 2.0, 3.0, f32::NAN, f32::INFINITY, 4.0]);
        let fsl = FixedSizeListArray::try_new_from_values(values, 2).unwrap();
        let result = is_finite(&fsl);
        assert_eq!(
            result.values().iter().collect::<Vec<_>>(),
            vec![true, false, false]
        );
    }

    #[test]
    fn test_is_finite_f16() {
        let values = Float16Array::from(vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::NAN,
            f16::from_f32(2.0),
        ]);
        let fsl = FixedSizeListArray::try_new_from_values(values, 2).unwrap();
        let result = is_finite(&fsl);
        assert_eq!(result.values().iter().collect::<Vec<_>>(), vec![true, false]);
    }

    #[test]
    fn test_is_finite_row_null() {
        let values = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let row_nulls = NullBuffer::from(vec![true, false]);
        let fsl =
            FixedSizeListArray::try_new(field, 2, Arc::new(values), Some(row_nulls)).unwrap();
        let result = is_finite(&fsl);
        // second row is null at the FixedSizeListArray level -- must be false
        // regardless of the (finite) values it points at.
        assert_eq!(result.values().iter().collect::<Vec<_>>(), vec![true, false]);
    }

    #[test]
    fn test_is_finite_child_null() {
        // Row 1's second element is null -- row must be false even though
        // every present value is finite.
        let values = Float32Array::from(vec![Some(1.0), Some(2.0), Some(3.0), None]);
        let fsl = FixedSizeListArray::try_new_from_values(values, 2).unwrap();
        let result = is_finite(&fsl);
        assert_eq!(result.values().iter().collect::<Vec<_>>(), vec![true, false]);
    }

    #[test]
    fn test_is_finite_large_matches_expected_across_chunks() {
        // Large enough to exercise the multi-threaded path (>
        // IS_FINITE_MIN_ROWS_FOR_THREADING), with non-finite rows scattered
        // across the row range so every thread's chunk contains at least one,
        // to catch chunk-boundary bugs.
        const NUM_ROWS: usize = 10_000;
        const DIM: usize = 4;
        let mut data = vec![0.0f32; NUM_ROWS * DIM];
        let mut expected = vec![true; NUM_ROWS];
        for row in (0..NUM_ROWS).step_by(97) {
            data[row * DIM] = f32::NAN;
            expected[row] = false;
        }
        let fsl =
            FixedSizeListArray::try_new_from_values(Float32Array::from(data), DIM as i32).unwrap();
        let result = is_finite(&fsl);
        assert_eq!(result.values().iter().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn test_is_finite_empty() {
        let values = Float32Array::from(Vec::<f32>::new());
        let fsl = FixedSizeListArray::try_new_from_values(values, 4).unwrap();
        let result = is_finite(&fsl);
        assert_eq!(result.len(), 0);
    }
}
