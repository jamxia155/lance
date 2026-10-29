// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::mem;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arrow_array::cast::AsArray;
use arrow_array::{FixedSizeListArray, RecordBatch, UInt8Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::cache::LanceCache;
use lance_core::datatypes::Schema;
use lance_core::{Error, ROW_ID, Result};
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_file::version::{ConcreteFileVersion, LanceFileVersion};
use lance_file::versions as file_versions;
use lance_file::writer::{FileWriter, FileWriterOptions};
use lance_index::vector::v3::shuffler::{ShufflePartitionWindowPlan, ShuffleReader};
use lance_index::vector::{PART_ID_COLUMN, PQ_CODE_COLUMN};
use lance_io::ReadBatchParams;
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::stream::{RecordBatchStream, RecordBatchStreamAdapter};
use lance_io::traits::Writer;
use lance_io::utils::CachedFileSize;
use object_store::path::Path;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

const PARTITION_ARTIFACT_MANIFEST_VERSION: u32 = 1;
const PARTITION_ARTIFACT_MANIFEST_FILE_NAME: &str = "manifest.json";
const PARTITION_ARTIFACT_PARTITIONS_DIR: &str = "partitions";
const PARTITION_ARTIFACT_DEFAULT_BUCKETS: usize = 256;
const PARTITION_ARTIFACT_BUCKET_PREFIX: &str = "bucket-";
const PARTITION_ARTIFACT_FILE_VERSION: &str = "2.2";
const PARTITION_ARTIFACT_BUCKET_BUFFER_ROWS: usize = 32 * 1024;

/// Top-level manifest for a precomputed partition artifact.
///
/// The manifest is intentionally small and JSON-encoded so an external backend
/// can materialize partition data once and Lance can reopen it later without
/// understanding any backend-specific details.
#[derive(Debug, Serialize, Deserialize)]
struct PartitionArtifactManifest {
    version: u32,
    num_partitions: usize,
    #[serde(default)]
    metadata_file: Option<String>,
    #[serde(default)]
    total_loss: Option<f64>,
    partitions: Vec<PartitionArtifactPartition>,
}

/// Describes where one logical IVF partition lives inside the artifact.
///
/// Multiple logical partitions can share the same physical file when they hash
/// to the same bucket. `ranges` records the row spans within that file that
/// belong to this partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PartitionArtifactPartition {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    num_rows: usize,
    #[serde(default)]
    ranges: Vec<PartitionArtifactRange>,
}

/// A contiguous row range for a partition inside one bucket file.
///
/// The builder sorts each finalized bucket by partition id, so a partition is
/// usually represented by a single range. The type still allows multiple runs
/// so the reader does not depend on that implementation detail.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PartitionArtifactRange {
    offset: u64,
    num_rows: u64,
}

/// In-memory staging buffer for one bucket before it is flushed to disk.
///
/// Batches arrive grouped arbitrarily by the backend. The builder first
/// appends rows into per-bucket buffers so it can write larger sequential runs
/// to temporary files instead of issuing tiny file writes.
#[derive(Default, Debug)]
struct BucketBuffer {
    row_ids: Vec<u64>,
    partition_ids: Vec<u32>,
    pq_values: Vec<u8>,
}

impl BucketBuffer {
    /// Number of staged rows currently buffered for this bucket.
    fn len(&self) -> usize {
        self.row_ids.len()
    }

    /// Whether the bucket currently has any staged rows.
    fn is_empty(&self) -> bool {
        self.row_ids.is_empty()
    }
}

/// Number of flush jobs that can queue in front of each flush worker. Each job holds one bucket
/// buffer (32K rows, about 9 MB for 256-byte codes), which bounds the memory in flight.
const PARTITION_ARTIFACT_FLUSH_QUEUE_PER_WORKER: usize = 4;
const PARTITION_ARTIFACT_FLUSH_WORKERS_DEFAULT: usize = 8;

/// Number of tasks that sort, encode and write full bucket buffers, from
/// `LANCE_PARTITION_ARTIFACT_FLUSH_WORKERS`. `0` flushes inline on the appending task.
fn partition_artifact_flush_workers() -> usize {
    std::env::var("LANCE_PARTITION_ARTIFACT_FLUSH_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(PARTITION_ARTIFACT_FLUSH_WORKERS_DEFAULT)
}

/// Reads a positive integer from the environment variable `name`, falling back to `default` when
/// it is unset, unparsable or zero.
fn positive_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

/// Number of bucket files the artifact is split into, from `LANCE_PARTITION_ARTIFACT_BUCKETS`
/// (default 256). Partition `p` goes to bucket `p % buckets`.
fn partition_artifact_buckets() -> usize {
    positive_env_usize(
        "LANCE_PARTITION_ARTIFACT_BUCKETS",
        PARTITION_ARTIFACT_DEFAULT_BUCKETS,
    )
}

/// Rows a bucket buffers before it is flushed, from `LANCE_PARTITION_ARTIFACT_BUCKET_BUFFER_ROWS`
/// (default 32K). Each flush adds one contiguous range per partition in the bucket, so larger
/// buffers mean fewer, larger ranges per partition at merge time, for more memory in flight.
fn partition_artifact_bucket_buffer_rows() -> usize {
    positive_env_usize(
        "LANCE_PARTITION_ARTIFACT_BUCKET_BUFFER_ROWS",
        PARTITION_ARTIFACT_BUCKET_BUFFER_ROWS,
    )
}

#[derive(Default, Debug)]
struct PartitionArtifactBuilderStats {
    append_batches: usize,
    append_rows: usize,
    append_time: Duration,
    validate_time: Duration,
    extract_time: Duration,
    scatter_time: Duration,
    /// Time the appending task spent handing full buffers to the flush (including waiting for a
    /// worker's queue to drain); with inline flushing this contains the whole flush.
    dispatch_time: Duration,
    finish_flush_time: Duration,
    manifest_time: Duration,
}

/// Counters of the flush itself. With flush workers, times are summed over the workers.
#[derive(Default, Debug)]
struct FlushStats {
    flushes: usize,
    flushed_rows: usize,
    single_partition_flushes: usize,
    multi_partition_flushes: usize,
    flush_time: Duration,
    final_batch_time: Duration,
    writer_time: Duration,
    writer_finish_time: Duration,
}

impl FlushStats {
    fn add(&mut self, other: &FlushStats) {
        self.flushes += other.flushes;
        self.flushed_rows += other.flushed_rows;
        self.single_partition_flushes += other.single_partition_flushes;
        self.multi_partition_flushes += other.multi_partition_flushes;
        self.flush_time += other.flush_time;
        self.final_batch_time += other.final_batch_time;
        self.writer_time += other.writer_time;
        self.writer_finish_time += other.writer_finish_time;
    }
}

/// What a flush needs to know about the artifact.
#[derive(Debug)]
struct FlushContext {
    object_store: Arc<ObjectStore>,
    root_dir: Path,
    num_partitions: usize,
    num_buckets: usize,
    pq_code_width: usize,
    final_schema: Arc<ArrowSchema>,
}

impl FlushContext {
    /// Path of the finalized file for one bucket.
    fn final_bucket_path(&self, bucket_id: usize) -> Path {
        self.root_dir
            .clone()
            .join(PARTITION_ARTIFACT_PARTITIONS_DIR)
            .join(format!(
                "{PARTITION_ARTIFACT_BUCKET_PREFIX}{bucket_id:05}.lance"
            ))
    }

    /// Relative path recorded in the manifest for one finalized bucket.
    fn final_bucket_relative_path(&self, bucket_id: usize) -> String {
        format!(
            "{PARTITION_ARTIFACT_PARTITIONS_DIR}/{PARTITION_ARTIFACT_BUCKET_PREFIX}{bucket_id:05}.lance"
        )
    }
}

/// One full bucket buffer on its way to the bucket file.
struct FlushJob {
    bucket_id: usize,
    /// Row index in the bucket file where this job's rows start. Assigned in dispatch order, so
    /// it does not depend on when the job is written.
    file_offset: u64,
    row_ids: Vec<u64>,
    part_ids: Vec<u32>,
    pq_values: Vec<u8>,
}

/// Everything a flush produces for the buckets it owns.
#[derive(Debug, Default)]
struct FlushOutput {
    /// Manifest entries of the partitions that received rows.
    partitions: Vec<(usize, PartitionArtifactPartition)>,
    /// Buckets that have a file.
    buckets: Vec<usize>,
    stats: FlushStats,
}

/// Sorts, encodes and writes full bucket buffers. It owns the writers and manifest entries of
/// its buckets: every partition lives in exactly one bucket, so a bucket's flushes only touch
/// state that is private to the state that owns it. A bucket's jobs must reach it in order.
struct FlushState {
    ctx: Arc<FlushContext>,
    writers: HashMap<usize, FileWriter>,
    partitions: HashMap<usize, PartitionArtifactPartition>,
    stats: FlushStats,
}

impl FlushState {
    fn new(ctx: Arc<FlushContext>) -> Self {
        Self {
            ctx,
            writers: HashMap::new(),
            partitions: HashMap::new(),
            stats: FlushStats::default(),
        }
    }

    fn partition_entry(&mut self, partition_id: usize) -> &mut PartitionArtifactPartition {
        self.partitions
            .entry(partition_id)
            .or_insert_with(|| PartitionArtifactPartition {
                path: None,
                num_rows: 0,
                ranges: Vec::new(),
            })
    }

    /// Flush one bucket buffer into its finalized bucket file.
    ///
    /// Each flush sorts only the buffered rows for this bucket and appends them
    /// to the final file while recording new manifest ranges for the affected
    /// partitions.
    async fn process(&mut self, job: FlushJob) -> Result<()> {
        let flush_start = Instant::now();
        let FlushJob {
            bucket_id,
            file_offset,
            row_ids,
            part_ids,
            pq_values,
        } = job;
        let total_rows = row_ids.len();
        self.stats.flushes += 1;
        self.stats.flushed_rows += total_rows;

        let result = if self.ctx.num_buckets == self.ctx.num_partitions {
            debug_assert!(part_ids.is_empty());
            self.stats.single_partition_flushes += 1;
            self.flush_single_partition_bucket(
                bucket_id,
                file_offset,
                row_ids,
                pq_values,
                total_rows,
            )
            .await
        } else {
            self.stats.multi_partition_flushes += 1;

            let row_ids = UInt64Array::from(row_ids);
            let pq_values = UInt8Array::from(pq_values);
            let pq_code_width = self.ctx.pq_code_width;
            let mut permutation = (0..total_rows).collect::<Vec<_>>();
            permutation.sort_unstable_by_key(|&idx| part_ids[idx]);

            let mut sorted_row_ids = Vec::with_capacity(total_rows);
            let mut sorted_partition_ids = Vec::with_capacity(total_rows);
            let mut sorted_pq_values = Vec::with_capacity(total_rows * pq_code_width);
            for idx in permutation {
                sorted_row_ids.push(row_ids.value(idx));
                sorted_partition_ids.push(part_ids[idx]);
                let start = idx * pq_code_width;
                let end = start + pq_code_width;
                sorted_pq_values.extend_from_slice(&pq_values.values()[start..end]);
            }

            let final_relative_path = self.ctx.final_bucket_relative_path(bucket_id);
            let mut offset = 0usize;
            while offset < sorted_partition_ids.len() {
                let partition_id = sorted_partition_ids[offset] as usize;
                let mut end = offset + 1;
                while end < sorted_partition_ids.len()
                    && sorted_partition_ids[end] == sorted_partition_ids[offset]
                {
                    end += 1;
                }
                let partition = self.partition_entry(partition_id);
                match &partition.path {
                    Some(existing) if existing != &final_relative_path => {
                        return Err(Error::io(format!(
                            "partition {} is split across multiple bucket files: '{}' vs '{}'",
                            partition_id, existing, final_relative_path
                        )));
                    }
                    None => partition.path = Some(final_relative_path.clone()),
                    _ => {}
                }
                partition.num_rows += end - offset;
                append_partition_range(
                    partition,
                    file_offset + offset as u64,
                    (end - offset) as u64,
                );
                offset = end;
            }

            let pq_codes = FixedSizeListArray::try_new_from_values(
                UInt8Array::from(sorted_pq_values),
                pq_code_width as i32,
            )?;
            self.write_bucket_batch(
                bucket_id,
                UInt64Array::from(sorted_row_ids),
                pq_codes,
                total_rows,
            )
            .await
        };
        self.stats.flush_time += flush_start.elapsed();
        result
    }

    /// Flush a bucket that maps exactly to one partition.
    ///
    /// When the number of buckets equals the number of partitions, all rows in
    /// one bucket already belong to the same partition. Sorting by partition id
    /// would only add CPU work and another PQ-code copy.
    async fn flush_single_partition_bucket(
        &mut self,
        bucket_id: usize,
        file_offset: u64,
        row_ids: Vec<u64>,
        pq_values: Vec<u8>,
        total_rows: usize,
    ) -> Result<()> {
        let final_relative_path = self.ctx.final_bucket_relative_path(bucket_id);
        let partition = self.partition_entry(bucket_id);
        match &partition.path {
            Some(existing) if existing != &final_relative_path => {
                return Err(Error::io(format!(
                    "partition {} is split across multiple bucket files: '{}' vs '{}'",
                    bucket_id, existing, final_relative_path
                )));
            }
            None => partition.path = Some(final_relative_path),
            _ => {}
        }
        partition.num_rows += total_rows;
        append_partition_range(partition, file_offset, total_rows as u64);

        let pq_codes = FixedSizeListArray::try_new_from_values(
            UInt8Array::from(pq_values),
            self.ctx.pq_code_width as i32,
        )?;
        self.write_bucket_batch(bucket_id, UInt64Array::from(row_ids), pq_codes, total_rows)
            .await
    }

    /// Write a finalized bucket batch.
    async fn write_bucket_batch(
        &mut self,
        bucket_id: usize,
        row_ids: UInt64Array,
        pq_codes: FixedSizeListArray,
        _total_rows: usize,
    ) -> Result<()> {
        let final_batch_start = Instant::now();
        let final_batch = RecordBatch::try_new(
            self.ctx.final_schema.clone(),
            vec![Arc::new(row_ids), Arc::new(pq_codes)],
        )?;
        self.stats.final_batch_time += final_batch_start.elapsed();

        let writer_start = Instant::now();
        {
            let writer = self.ensure_final_writer(bucket_id).await?;
            writer.write_batch(&final_batch).await?;
        }
        self.stats.writer_time += writer_start.elapsed();
        Ok(())
    }

    /// Lazily create the finalized writer for a bucket.
    ///
    /// Buckets that never receive rows never create a file, which keeps sparse
    /// artifacts compact.
    async fn ensure_final_writer(&mut self, bucket_id: usize) -> Result<&mut FileWriter> {
        if !self.writers.contains_key(&bucket_id) {
            let path = self.ctx.final_bucket_path(bucket_id);
            let writer = file_versions::create_writer(
                partition_artifact_file_version()?,
                self.ctx.object_store.create(&path).await?,
                Schema::try_from(self.ctx.final_schema.as_ref())?,
                FileWriterOptions::default(),
            )?;
            self.writers.insert(bucket_id, writer);
        }
        Ok(self
            .writers
            .get_mut(&bucket_id)
            .expect("final writer initialized"))
    }

    /// Finish the bucket files and hand back what this state wrote.
    async fn finish(mut self) -> Result<FlushOutput> {
        let writer_finish_start = Instant::now();
        let mut buckets = Vec::with_capacity(self.writers.len());
        for (bucket_id, writer) in self.writers.iter_mut() {
            writer.finish().await?;
            buckets.push(*bucket_id);
        }
        self.stats.writer_finish_time += writer_finish_start.elapsed();
        Ok(FlushOutput {
            partitions: self.partitions.into_iter().collect(),
            buckets,
            stats: self.stats,
        })
    }
}

/// Where full bucket buffers go.
enum FlushSink {
    /// Not started yet; the first flush starts it (spawning needs a runtime).
    Unstarted,
    /// Flushed on the appending task.
    Inline(Box<FlushState>),
    /// Flushed by worker tasks; bucket `b` is owned by worker `b % workers`.
    Workers {
        senders: Vec<tokio::sync::mpsc::Sender<FlushJob>>,
        handles: Vec<tokio::task::JoinHandle<Result<FlushOutput>>>,
    },
    Finished,
}

/// Waits for the flush workers and returns their outputs, or the first error.
async fn join_flush_workers(
    handles: Vec<tokio::task::JoinHandle<Result<FlushOutput>>>,
) -> Result<Vec<FlushOutput>> {
    let mut outputs = Vec::with_capacity(handles.len());
    let mut first_error = None;
    for handle in handles {
        match handle.await {
            Ok(Ok(output)) => outputs.push(output),
            Ok(Err(error)) => {
                first_error.get_or_insert(error);
            }
            Err(join_error) => {
                first_error.get_or_insert(Error::io(format!(
                    "partition artifact flush worker failed: {join_error}"
                )));
            }
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(outputs),
    }
}

/// Writes partition-addressable encoded rows for a later Lance finalization.
///
/// The builder uses bucket-local buffering to keep append-time memory bounded.
/// Each flush sorts only the current in-memory bucket and appends it directly to
/// the finalized bucket file, while the manifest accumulates per-partition row
/// ranges. This keeps the writer streaming and avoids a full read/sort/rewrite
/// pass at `finish()` time.
///
/// The flush (sort, encode, write) runs on `LANCE_PARTITION_ARTIFACT_FLUSH_WORKERS` tasks
/// (default 8; 0 runs it inline on the task that appends), so the appending task only scatters
/// rows into buffers. Buckets are independent files, and each is owned by one worker, so the
/// files and the manifest are the same as with inline flushing.
pub struct PartitionArtifactBuilder {
    ctx: Arc<FlushContext>,
    buffers: Vec<BucketBuffer>,
    /// Rows handed to the flush so far per bucket, which is where the next flush starts.
    bucket_row_counts: Vec<u64>,
    flush_workers: usize,
    bucket_buffer_rows: usize,
    sink: FlushSink,
    stats: PartitionArtifactBuilderStats,
    flush_stats: FlushStats,
}

impl PartitionArtifactBuilder {
    /// Create a builder from a URI and optional storage options.
    ///
    /// This is the external entry point used by backends that only know an
    /// artifact URI. It resolves the object store and then delegates to the
    /// store-aware constructor.
    pub async fn try_new(
        uri: &str,
        num_partitions: usize,
        pq_code_width: usize,
        storage_options: Option<&HashMap<String, String>>,
    ) -> Result<Self> {
        let registry = Arc::new(ObjectStoreRegistry::default());
        let params = if let Some(storage_options) = storage_options {
            ObjectStoreParams {
                storage_options_accessor: Some(Arc::new(
                    lance_io::object_store::StorageOptionsAccessor::with_static_options(
                        storage_options.clone(),
                    ),
                )),
                ..Default::default()
            }
        } else {
            ObjectStoreParams::default()
        };
        let (object_store, root_dir) =
            ObjectStore::from_uri_and_params(registry, uri, &params).await?;
        Self::try_new_with_store(object_store, root_dir, num_partitions, pq_code_width)
    }

    /// Create a builder against an already-resolved object store.
    ///
    /// The builder precomputes the final schema and allocates one staging
    /// buffer per bucket. Buckets are a write-time sharding scheme: they are
    /// not visible to readers, but they keep memory usage bounded and avoid one
    /// file per partition.
    pub fn try_new_with_store(
        object_store: Arc<ObjectStore>,
        root_dir: Path,
        num_partitions: usize,
        pq_code_width: usize,
    ) -> Result<Self> {
        if num_partitions == 0 {
            return Err(Error::invalid_input(
                "partition artifact builder requires num_partitions > 0".to_string(),
            ));
        }
        if pq_code_width == 0 {
            return Err(Error::invalid_input(
                "partition artifact builder requires pq_code_width > 0".to_string(),
            ));
        }

        let num_buckets = num_partitions.min(partition_artifact_buckets()).max(1);
        let final_schema = Arc::new(ArrowSchema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new(
                PQ_CODE_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::UInt8, true)),
                    pq_code_width as i32,
                ),
                true,
            ),
        ]));

        Ok(Self {
            ctx: Arc::new(FlushContext {
                object_store,
                root_dir,
                num_partitions,
                num_buckets,
                pq_code_width,
                final_schema,
            }),
            buffers: (0..num_buckets).map(|_| BucketBuffer::default()).collect(),
            bucket_row_counts: vec![0; num_buckets],
            flush_workers: partition_artifact_flush_workers().min(num_buckets),
            bucket_buffer_rows: partition_artifact_bucket_buffer_rows(),
            sink: FlushSink::Unstarted,
            stats: PartitionArtifactBuilderStats::default(),
            flush_stats: FlushStats::default(),
        })
    }

    /// Append one encoded batch into the artifact staging area.
    ///
    /// Input batches must already contain row ids, partition ids, and PQ codes.
    /// Rows are redistributed into bucket-local in-memory buffers and flushed to
    /// temporary files once they become large enough.
    pub async fn append_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        let append_start = Instant::now();
        let validate_start = Instant::now();
        validate_input_batch(batch, self.ctx.pq_code_width)?;
        self.stats.validate_time += validate_start.elapsed();

        let extract_start = Instant::now();
        let row_ids = batch[ROW_ID].as_primitive::<arrow::datatypes::UInt64Type>();
        let part_ids = batch[PART_ID_COLUMN].as_primitive::<arrow::datatypes::UInt32Type>();
        let pq_codes = batch[PQ_CODE_COLUMN].as_fixed_size_list();
        let pq_values = pq_codes
            .values()
            .as_primitive::<arrow::datatypes::UInt8Type>();
        let pq_values = pq_values.values().as_ref();
        self.stats.extract_time += extract_start.elapsed();

        let num_partitions = self.ctx.num_partitions;
        let num_buckets = self.ctx.num_buckets;
        let pq_code_width = self.ctx.pq_code_width;
        let scatter_start = Instant::now();
        for row_idx in 0..batch.num_rows() {
            let partition_id = part_ids.value(row_idx) as usize;
            if partition_id >= num_partitions {
                return Err(Error::invalid_input(format!(
                    "partition artifact batch contains partition id {} but num_partitions is {}",
                    partition_id, num_partitions
                )));
            }
            let bucket_id = partition_id % num_buckets;
            let buffer = &mut self.buffers[bucket_id];
            buffer.row_ids.push(row_ids.value(row_idx));
            if num_buckets != num_partitions {
                buffer.partition_ids.push(partition_id as u32);
            }
            let start = row_idx * pq_code_width;
            let end = start + pq_code_width;
            buffer.pq_values.extend_from_slice(&pq_values[start..end]);
            if buffer.len() >= self.bucket_buffer_rows {
                let dispatch_start = Instant::now();
                self.flush_bucket(bucket_id).await?;
                self.stats.dispatch_time += dispatch_start.elapsed();
            }
        }
        self.stats.scatter_time += scatter_start.elapsed();
        self.stats.append_batches += 1;
        self.stats.append_rows += batch.num_rows();
        self.stats.append_time += append_start.elapsed();
        Ok(())
    }

    /// Finalize the artifact and return the relative files that were created.
    ///
    /// bucket file.
    /// Finalization only needs to flush the remaining in-memory buffers and
    /// persist the manifest because bucket files are already in their final
    /// layout.
    pub async fn finish(
        &mut self,
        metadata_file: &str,
        total_loss: Option<f64>,
    ) -> Result<Vec<String>> {
        let finish_flush_start = Instant::now();
        for bucket_id in 0..self.ctx.num_buckets {
            self.flush_bucket(bucket_id).await?;
        }
        let outputs = match mem::replace(&mut self.sink, FlushSink::Finished) {
            FlushSink::Unstarted => vec![FlushState::new(self.ctx.clone()).finish().await?],
            FlushSink::Inline(state) => vec![state.finish().await?],
            FlushSink::Workers { senders, handles } => {
                // Closing the queues lets each worker drain its jobs and finish its files.
                drop(senders);
                join_flush_workers(handles).await?
            }
            FlushSink::Finished => {
                return Err(Error::invalid_input(
                    "partition artifact builder is already finished".to_string(),
                ));
            }
        };
        self.stats.finish_flush_time += finish_flush_start.elapsed();

        let mut partitions = vec![
            PartitionArtifactPartition {
                path: None,
                num_rows: 0,
                ranges: Vec::new(),
            };
            self.ctx.num_partitions
        ];
        let mut buckets = Vec::new();
        for output in outputs {
            for (partition_id, partition) in output.partitions {
                partitions[partition_id] = partition;
            }
            buckets.extend(output.buckets);
            self.flush_stats.add(&output.stats);
        }
        buckets.sort_unstable();
        let artifact_files = buckets
            .iter()
            .map(|&bucket_id| self.ctx.final_bucket_relative_path(bucket_id))
            .collect::<Vec<_>>();

        let manifest = PartitionArtifactManifest {
            version: PARTITION_ARTIFACT_MANIFEST_VERSION,
            num_partitions: self.ctx.num_partitions,
            metadata_file: Some(metadata_file.to_string()),
            total_loss,
            partitions,
        };
        let manifest_start = Instant::now();
        write_json(
            self.ctx.object_store.as_ref(),
            &self
                .ctx
                .root_dir
                .clone()
                .join(PARTITION_ARTIFACT_MANIFEST_FILE_NAME),
            &manifest,
        )
        .await?;
        self.stats.manifest_time += manifest_start.elapsed();
        self.log_stats();

        let mut files = vec![PARTITION_ARTIFACT_MANIFEST_FILE_NAME.to_string()];
        files.extend(artifact_files);
        Ok(files)
    }

    #[cfg(test)]
    fn with_flush_workers(mut self, workers: usize) -> Self {
        self.flush_workers = workers.min(self.ctx.num_buckets);
        self
    }

    /// Start the flush on first use.
    fn start_sink(&mut self) {
        if !matches!(self.sink, FlushSink::Unstarted) {
            return;
        }
        if self.flush_workers == 0 {
            self.sink = FlushSink::Inline(Box::new(FlushState::new(self.ctx.clone())));
            return;
        }
        let mut senders = Vec::with_capacity(self.flush_workers);
        let mut handles = Vec::with_capacity(self.flush_workers);
        for _ in 0..self.flush_workers {
            let (tx, mut rx) =
                tokio::sync::mpsc::channel::<FlushJob>(PARTITION_ARTIFACT_FLUSH_QUEUE_PER_WORKER);
            let mut state = FlushState::new(self.ctx.clone());
            handles.push(tokio::spawn(async move {
                while let Some(job) = rx.recv().await {
                    state.process(job).await?;
                }
                state.finish().await
            }));
            senders.push(tx);
        }
        self.sink = FlushSink::Workers { senders, handles };
    }

    /// Hand the current in-memory buffer of one bucket to the flush.
    async fn flush_bucket(&mut self, bucket_id: usize) -> Result<()> {
        if self.buffers[bucket_id].is_empty() {
            return Ok(());
        }
        self.start_sink();

        let buffer = &mut self.buffers[bucket_id];
        let row_ids = mem::take(&mut buffer.row_ids);
        let job = FlushJob {
            bucket_id,
            file_offset: self.bucket_row_counts[bucket_id],
            part_ids: mem::take(&mut buffer.partition_ids),
            pq_values: mem::take(&mut buffer.pq_values),
            row_ids,
        };
        self.bucket_row_counts[bucket_id] += job.row_ids.len() as u64;

        match &mut self.sink {
            FlushSink::Inline(state) => state.process(job).await,
            FlushSink::Workers { senders, .. } => {
                let worker = bucket_id % senders.len();
                if senders[worker].send(job).await.is_ok() {
                    return Ok(());
                }
                // The worker stopped; its error says why.
                let FlushSink::Workers { senders, handles } =
                    mem::replace(&mut self.sink, FlushSink::Finished)
                else {
                    unreachable!("sink was just matched as workers");
                };
                drop(senders);
                join_flush_workers(handles).await?;
                Err(Error::io(
                    "partition artifact flush worker stopped unexpectedly".to_string(),
                ))
            }
            FlushSink::Unstarted | FlushSink::Finished => Err(Error::invalid_input(
                "partition artifact builder is already finished".to_string(),
            )),
        }
    }

    fn log_stats(&self) {
        let flush = &self.flush_stats;
        eprintln!(
            "partition artifact builder stages: append_batches={} append_rows={} append_s={:.3} validate_s={:.3} extract_s={:.3} scatter_s={:.3} flushes={} flushed_rows={} single_flushes={} multi_flushes={} flush_s={:.3} final_batch_s={:.3} writer_s={:.3} finish_flush_s={:.3} writer_finish_s={:.3} manifest_s={:.3} flush_workers={} dispatch_s={:.3}",
            self.stats.append_batches,
            self.stats.append_rows,
            self.stats.append_time.as_secs_f64(),
            self.stats.validate_time.as_secs_f64(),
            self.stats.extract_time.as_secs_f64(),
            self.stats.scatter_time.as_secs_f64(),
            flush.flushes,
            flush.flushed_rows,
            flush.single_partition_flushes,
            flush.multi_partition_flushes,
            flush.flush_time.as_secs_f64(),
            flush.final_batch_time.as_secs_f64(),
            flush.writer_time.as_secs_f64(),
            self.stats.finish_flush_time.as_secs_f64(),
            flush.writer_finish_time.as_secs_f64(),
            self.stats.manifest_time.as_secs_f64(),
            self.flush_workers,
            self.stats.dispatch_time.as_secs_f64(),
        );
    }
}

/// Reopens a partition artifact as a `ShuffleReader`.
///
/// The final Lance builder consumes artifacts through the generic
/// [`ShuffleReader`] interface, so this adapter hides the manifest parsing and
/// file caching needed to expose partition-local record batch streams.
#[derive(Debug)]
pub(crate) struct PartitionArtifactShuffleReader {
    scheduler: Arc<ScanScheduler>,
    root_dir: Path,
    partitions: Vec<PartitionArtifactPartition>,
    total_loss: Option<f64>,
    file_readers: Mutex<HashMap<String, Arc<FileReader>>>,
    /// Decoded bytes per row (all artifact columns are fixed width), used to
    /// size partition-build admission. `None` if the artifact has no rows.
    estimated_row_bytes: Option<usize>,
}

/// Admission headroom per partition window, matching Lance's own shuffle
/// readers: Arrow buffers and batch allocations add a little beyond the
/// fixed-width values.
const WINDOW_ADMISSION_FIXED_HEADROOM_BYTES: usize = 1024 * 1024;

/// Bytes per row of a schema whose columns are all fixed width, else `None`.
fn fixed_row_bytes(schema: &ArrowSchema) -> Option<usize> {
    fn fixed_width(data_type: &DataType) -> Option<usize> {
        match data_type {
            DataType::FixedSizeList(child, size) => {
                fixed_width(child.data_type()).map(|width| width * *size as usize)
            }
            other => other.primitive_width(),
        }
    }
    schema
        .fields()
        .iter()
        .map(|field| fixed_width(field.data_type()))
        .sum()
}

/// File version for all files stored inside a partition artifact.
///
/// The artifact uses a fixed file version so external backends and Lance
/// finalization agree on the on-disk layout.
fn partition_artifact_file_version() -> Result<ConcreteFileVersion> {
    Ok(PARTITION_ARTIFACT_FILE_VERSION
        .parse::<LanceFileVersion>()
        .map_err(|error| {
            Error::invalid_input(format!(
                "invalid partition artifact file version '{}': {}",
                PARTITION_ARTIFACT_FILE_VERSION, error
            ))
        })?
        .resolve())
}

/// Validate that a backend-produced batch matches the artifact contract.
///
/// The builder is intentionally strict here because any schema drift would only
/// surface much later during finalization.
fn validate_input_batch(batch: &RecordBatch, pq_code_width: usize) -> Result<()> {
    let Some(row_ids) = batch.column_by_name(ROW_ID) else {
        return Err(Error::invalid_input(format!(
            "partition artifact batch must contain {ROW_ID}"
        )));
    };
    if row_ids.data_type() != &DataType::UInt64 {
        return Err(Error::invalid_input(format!(
            "partition artifact batch column {ROW_ID} must be uint64, got {}",
            row_ids.data_type()
        )));
    }
    let Some(part_ids) = batch.column_by_name(PART_ID_COLUMN) else {
        return Err(Error::invalid_input(format!(
            "partition artifact batch must contain {PART_ID_COLUMN}"
        )));
    };
    if part_ids.data_type() != &DataType::UInt32 {
        return Err(Error::invalid_input(format!(
            "partition artifact batch column {PART_ID_COLUMN} must be uint32, got {}",
            part_ids.data_type()
        )));
    }
    let Some(pq_codes) = batch.column_by_name(PQ_CODE_COLUMN) else {
        return Err(Error::invalid_input(format!(
            "partition artifact batch must contain {PQ_CODE_COLUMN}"
        )));
    };
    match pq_codes.data_type() {
        DataType::FixedSizeList(_, width) if *width as usize == pq_code_width => Ok(()),
        other => Err(Error::invalid_input(format!(
            "partition artifact batch column {PQ_CODE_COLUMN} must be fixed_size_list<uint8>[{}], got {}",
            pq_code_width, other
        ))),
    }
}

/// Serialize a small JSON sidecar directly into the object store.
async fn write_json<T: Serialize>(
    object_store: &ObjectStore,
    path: &Path,
    value: &T,
) -> Result<()> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        Error::invalid_input(format!(
            "failed to serialize partition artifact manifest '{}': {}",
            path, error
        ))
    })?;
    let mut writer = object_store.create(path).await?;
    writer.write_all(&bytes).await?;
    Writer::shutdown(writer.as_mut()).await?;
    Ok(())
}

impl PartitionArtifactShuffleReader {
    /// Open an artifact reader from a URI and optional storage options.
    pub(crate) async fn try_open(
        uri: &str,
        storage_options: Option<&HashMap<String, String>>,
    ) -> Result<Self> {
        let registry = Arc::new(ObjectStoreRegistry::default());
        let params = if let Some(storage_options) = storage_options {
            ObjectStoreParams {
                storage_options_accessor: Some(Arc::new(
                    lance_io::object_store::StorageOptionsAccessor::with_static_options(
                        storage_options.clone(),
                    ),
                )),
                ..Default::default()
            }
        } else {
            ObjectStoreParams::default()
        };
        let (object_store, root_dir) =
            ObjectStore::from_uri_and_params(registry, uri, &params).await?;
        Self::try_open_with_store(object_store, root_dir).await
    }

    /// Open an artifact reader once the object store has already been resolved.
    ///
    /// This reads the manifest once, validates it, and initializes the shared
    /// scheduler and reader cache used by partition reads.
    async fn try_open_with_store(object_store: Arc<ObjectStore>, root_dir: Path) -> Result<Self> {
        let manifest_path = root_dir.clone().join("manifest.json");
        let manifest_bytes = object_store.read_one_all(&manifest_path).await?;
        let manifest: PartitionArtifactManifest =
            serde_json::from_slice(&manifest_bytes).map_err(|error| {
                Error::invalid_input(format!(
                    "failed to parse partition artifact manifest '{}': {}",
                    manifest_path, error
                ))
            })?;
        if manifest.version != 1 {
            return Err(Error::invalid_input(format!(
                "unsupported partition artifact manifest version {}",
                manifest.version
            )));
        }
        if manifest.partitions.len() != manifest.num_partitions {
            return Err(Error::invalid_input(format!(
                "partition artifact manifest has {} partitions but num_partitions is {}",
                manifest.partitions.len(),
                manifest.num_partitions
            )));
        }

        // Use the object store's own I/O parallelism default here (e.g. 8 for
        // local disk) instead of `ObjectStore::io_parallelism()`/
        // `LANCE_IO_THREADS`: that env var is tuned for the much larger,
        // single-scheduler dataset scan, and reusing it here means every one
        // of the many small per-partition reads gets an oversized queue,
        // adding overhead without benefit (empirically: raising
        // LANCE_IO_THREADS from 8 to 32 roughly doubled this stage's take/
        // build time and peak RSS while helping the dataset scan). This is
        // deliberately the store's pre-existing default, not a value we
        // invented -- it's what this stage always used before `LANCE_IO_
        // THREADS` started leaking into it.
        // `LANCE_PARTITION_ARTIFACT_IO_PARALLELISM` overrides it for this stage only.
        let io_capacity = positive_env_usize(
            "LANCE_PARTITION_ARTIFACT_IO_PARALLELISM",
            object_store.default_io_parallelism(),
        );
        let scheduler = ScanScheduler::new(
            object_store.clone(),
            SchedulerConfig::new(32 * 1024 * 1024 * io_capacity as u64)
                .with_io_capacity(io_capacity),
        );
        let mut reader = Self {
            scheduler,
            root_dir,
            partitions: manifest.partitions,
            total_loss: manifest.total_loss,
            file_readers: Mutex::new(HashMap::new()),
            estimated_row_bytes: None,
        };
        // Every bucket file shares one schema; read it from the first one.
        let first_path = reader
            .partitions
            .iter()
            .find_map(|partition| partition.path.clone());
        if let Some(path) = first_path {
            let file_reader = reader.open_file_reader(&path).await?;
            let schema: ArrowSchema = file_reader.schema().as_ref().into();
            reader.estimated_row_bytes = fixed_row_bytes(&schema);
        }
        Ok(reader)
    }

    /// Open and cache a file reader for a finalized bucket file.
    ///
    /// Multiple logical partitions can point at the same bucket file, so the
    /// reader cache prevents redundant file opens during finalization.
    async fn open_file_reader(&self, relative_path: &str) -> Result<Arc<FileReader>> {
        if let Some(reader) = self
            .file_readers
            .lock()
            .expect("partition artifact reader mutex poisoned")
            .get(relative_path)
            .cloned()
        {
            return Ok(reader);
        }

        let path = join_relative_path(&self.root_dir, relative_path);
        let reader = Arc::new(
            FileReader::try_open(
                self.scheduler
                    .open_file(&path, &CachedFileSize::unknown())
                    .await?,
                None,
                Arc::<DecoderPlugins>::default(),
                &LanceCache::no_cache(),
                FileReaderOptions::default(),
            )
            .await?,
        );
        self.file_readers
            .lock()
            .expect("partition artifact reader mutex poisoned")
            .insert(relative_path.to_string(), reader.clone());
        Ok(reader)
    }
}

/// Join a manifest-relative path onto the artifact root.
fn join_relative_path(root_dir: &Path, relative_path: &str) -> Path {
    relative_path
        .split('/')
        .filter(|segment| !segment.is_empty())
        .fold(root_dir.clone(), |path, segment| path.join(segment))
}

#[async_trait::async_trait]
impl ShuffleReader for PartitionArtifactShuffleReader {
    /// Return a stream over all rows belonging to one logical partition.
    ///
    /// The manifest already records the precise row ranges for each partition,
    /// so the reader can issue targeted range reads without scanning unrelated
    /// partitions.
    async fn read_partition(
        &self,
        partition_id: usize,
    ) -> Result<Option<Box<dyn RecordBatchStream + Unpin + 'static>>> {
        let Some(partition) = self.partitions.get(partition_id) else {
            return Ok(None);
        };
        if partition.num_rows == 0 {
            return Ok(None);
        }
        let path = partition.path.as_ref().ok_or_else(|| {
            Error::invalid_input(format!(
                "partition artifact partition {} has {} rows but no path",
                partition_id, partition.num_rows
            ))
        })?;
        if partition.ranges.is_empty() {
            return Err(Error::invalid_input(format!(
                "partition artifact partition {} has {} rows but no ranges",
                partition_id, partition.num_rows
            )));
        }

        let reader = self.open_file_reader(path).await?;
        let ranges = merge_adjacent_ranges(
            partition
                .ranges
                .iter()
                .map(|range| Range {
                    start: range.offset,
                    end: range.offset + range.num_rows,
                })
                .collect::<Vec<_>>(),
        );
        let schema = Arc::new(reader.schema().as_ref().into());
        Ok(Some(Box::new(RecordBatchStreamAdapter::new(
            schema,
            reader
                .read_stream(
                    ReadBatchParams::Ranges(ranges.into()),
                    u32::MAX,
                    16,
                    FilterExpression::no_filter(),
                )
                .await?,
        ))))
    }

    /// Number of encoded rows available for one logical partition.
    fn partition_size(&self, partition_id: usize) -> Result<usize> {
        Ok(self
            .partitions
            .get(partition_id)
            .map(|partition| partition.num_rows)
            .unwrap_or(0))
    }

    /// Plan a one-partition window with an admission charge based on the
    /// partition's row count. Without it the trait default charges
    /// `usize::MAX`, which makes the windowed partition build read artifact
    /// partitions one at a time.
    fn plan_partition_window(
        &self,
        start_partition_id: usize,
        max_decoded_bytes: usize,
    ) -> Result<ShufflePartitionWindowPlan> {
        if max_decoded_bytes == 0 {
            return Err(Error::invalid_input(
                "max_decoded_bytes must be greater than 0",
            ));
        }
        let end_partition_id = start_partition_id.checked_add(1).ok_or_else(|| {
            Error::invalid_input(format!(
                "start_partition_id={} cannot be advanced",
                start_partition_id
            ))
        })?;
        let partition_rows = self.partition_size(start_partition_id)?;
        let estimated_decoded_bytes = match (partition_rows, self.estimated_row_bytes) {
            (0, _) => 0,
            (rows, Some(row_bytes)) => {
                let value_bytes = rows.checked_mul(row_bytes).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "decoded byte estimate overflows for {} rows at {} bytes per row",
                        rows, row_bytes
                    ))
                })?;
                value_bytes
                    .saturating_add(value_bytes / 4)
                    .saturating_add(WINDOW_ADMISSION_FIXED_HEADROOM_BYTES)
            }
            (_, None) => usize::MAX,
        };
        Ok(ShufflePartitionWindowPlan {
            partition_range: start_partition_id..end_partition_id,
            estimated_decoded_bytes,
        })
    }

    /// Optional training loss propagated from the backend into the artifact.
    fn total_loss(&self) -> Option<f64> {
        self.total_loss
    }
}

fn append_partition_range(partition: &mut PartitionArtifactPartition, offset: u64, num_rows: u64) {
    if num_rows == 0 {
        return;
    }
    if let Some(last) = partition.ranges.last_mut()
        && last.offset + last.num_rows == offset
    {
        last.num_rows += num_rows;
        return;
    }
    partition
        .ranges
        .push(PartitionArtifactRange { offset, num_rows });
}

fn merge_adjacent_ranges(mut ranges: Vec<Range<u64>>) -> Vec<Range<u64>> {
    if ranges.len() < 2 {
        return ranges;
    }

    let mut merged: Vec<Range<u64>> = Vec::with_capacity(ranges.len());
    for range in ranges.drain(..) {
        if let Some(last) = merged.last_mut()
            && last.end == range.start
        {
            last.end = range.end;
            continue;
        }
        merged.push(range);
    }
    merged
}

#[cfg(test)]
mod tests {
    use std::fs;

    use arrow_array::cast::AsArray;
    use arrow_array::{FixedSizeListArray, RecordBatch, UInt8Array, UInt32Array, UInt64Array};
    use futures::TryStreamExt;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::ROW_ID;
    use lance_core::datatypes::Schema;
    use lance_file::writer::FileWriterOptions;
    use lance_io::object_store::ObjectStore;

    use crate::Error;

    use super::*;

    #[test]
    fn partition_artifact_ranges_are_coalesced() {
        let mut partition = PartitionArtifactPartition {
            path: None,
            num_rows: 0,
            ranges: Vec::new(),
        };

        append_partition_range(&mut partition, 0, 10);
        append_partition_range(&mut partition, 10, 5);
        append_partition_range(&mut partition, 20, 3);
        append_partition_range(&mut partition, 23, 7);

        assert_eq!(partition.ranges.len(), 2);
        assert_eq!(partition.ranges[0].offset, 0);
        assert_eq!(partition.ranges[0].num_rows, 15);
        assert_eq!(partition.ranges[1].offset, 20);
        assert_eq!(partition.ranges[1].num_rows, 10);

        let merged = merge_adjacent_ranges(vec![0..10, 10..15, 20..23, 23..30]);
        assert_eq!(merged, vec![0..15, 20..30]);
    }

    #[tokio::test]
    async fn partition_artifact_builder_compacts_runs_into_single_partition_range() {
        let tempdir = tempfile::tempdir().unwrap();
        let root_dir = tempdir.path().join("artifact");
        fs::create_dir_all(&root_dir).unwrap();
        let object_store = Arc::new(ObjectStore::local());
        let root_path = Path::from_filesystem_path(&root_dir).unwrap();

        let mut builder = PartitionArtifactBuilder::try_new_with_store(
            object_store.clone(),
            root_path.clone(),
            300,
            2,
        )
        .unwrap();
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new(PART_ID_COLUMN, DataType::UInt32, false),
            Field::new(
                PQ_CODE_COLUMN,
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::UInt8, true)), 2),
                true,
            ),
        ]));

        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![10_u64, 11, 12, 13])),
                Arc::new(UInt32Array::from(vec![0_u32, 256, 0, 256])),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        UInt8Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]),
                        2,
                    )
                    .unwrap(),
                ),
            ],
        )
        .unwrap();
        let batch2 = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![14_u64, 15])),
                Arc::new(UInt32Array::from(vec![1_u32, 256])),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        UInt8Array::from(vec![9, 10, 11, 12]),
                        2,
                    )
                    .unwrap(),
                ),
            ],
        )
        .unwrap();
        builder.append_batch(&batch1).await.unwrap();
        builder.append_batch(&batch2).await.unwrap();
        let artifact_files = builder.finish("metadata.lance", Some(2.5)).await.unwrap();
        assert_eq!(artifact_files[0], "manifest.json");
        assert!(
            artifact_files
                .iter()
                .any(|path| path.ends_with("bucket-00000.lance"))
        );

        let manifest: PartitionArtifactManifest =
            serde_json::from_slice(&fs::read(root_dir.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest.version, 1);
        assert_eq!(manifest.metadata_file.as_deref(), Some("metadata.lance"));
        assert_eq!(manifest.total_loss, Some(2.5));
        assert_eq!(manifest.partitions[0].num_rows, 2);
        assert_eq!(manifest.partitions[0].ranges.len(), 1);
        assert_eq!(manifest.partitions[1].num_rows, 1);
        assert_eq!(manifest.partitions[1].ranges.len(), 1);
        assert_eq!(manifest.partitions[256].num_rows, 3);
        assert_eq!(manifest.partitions[256].ranges.len(), 1);
        assert_eq!(
            manifest.partitions[0].path, manifest.partitions[256].path,
            "partitions sharing a bucket should share one final file"
        );

        let reader = PartitionArtifactShuffleReader::try_open_with_store(object_store, root_path)
            .await
            .unwrap();
        let partition_0 = reader
            .read_partition(0)
            .await
            .unwrap()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let partition_0_row_ids = partition_0
            .iter()
            .flat_map(|batch| {
                batch[ROW_ID]
                    .as_primitive::<arrow::datatypes::UInt64Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(partition_0_row_ids, vec![10, 12]);

        let partition_256 = reader
            .read_partition(256)
            .await
            .unwrap()
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let partition_256_row_ids = partition_256
            .iter()
            .flat_map(|batch| {
                batch[ROW_ID]
                    .as_primitive::<arrow::datatypes::UInt64Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(partition_256_row_ids, vec![11, 13, 15]);
    }

    #[tokio::test]
    async fn partition_artifact_reader_reads_partition_ranges() {
        let tempdir = tempfile::tempdir().unwrap();
        let root_dir = tempdir.path().join("artifact");
        fs::create_dir_all(root_dir.join("partitions")).unwrap();

        let object_store = Arc::new(ObjectStore::local());
        let root_path = Path::from_filesystem_path(&root_dir).unwrap();
        let partition_path = root_path
            .clone()
            .join("partitions")
            .join("bucket-00000.lance");
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new(ROW_ID, arrow_schema::DataType::UInt64, false),
            arrow_schema::Field::new(
                lance_index::vector::PQ_CODE_COLUMN,
                arrow_schema::DataType::FixedSizeList(
                    Arc::new(arrow_schema::Field::new(
                        "item",
                        arrow_schema::DataType::UInt8,
                        true,
                    )),
                    2,
                ),
                true,
            ),
        ]));
        let mut writer = file_versions::create_writer(
            partition_artifact_file_version().unwrap(),
            object_store.create(&partition_path).await.unwrap(),
            Schema::try_from(schema.as_ref()).unwrap(),
            FileWriterOptions::default(),
        )
        .unwrap();
        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![10_u64, 11, 12])),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(
                        UInt8Array::from(vec![1, 2, 3, 4, 5, 6]),
                        2,
                    )
                    .unwrap(),
                ),
            ],
        )
        .unwrap();
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(vec![13_u64, 14])),
                Arc::new(
                    FixedSizeListArray::try_new_from_values(UInt8Array::from(vec![7, 8, 9, 10]), 2)
                        .unwrap(),
                ),
            ],
        )
        .unwrap();
        writer.write_batch(&batch1).await.unwrap();
        writer.write_batch(&batch2).await.unwrap();
        writer.finish().await.unwrap();

        let manifest = serde_json::json!({
            "version": 1,
            "num_partitions": 3,
            "total_loss": 1.5,
            "partitions": [
                {
                    "path": "partitions/bucket-00000.lance",
                    "num_rows": 2,
                    "ranges": [
                        {"offset": 0, "num_rows": 1},
                        {"offset": 3, "num_rows": 1},
                    ],
                },
                {
                    "path": "partitions/bucket-00000.lance",
                    "num_rows": 2,
                    "ranges": [
                        {"offset": 1, "num_rows": 2},
                    ],
                },
                {
                    "num_rows": 0,
                    "ranges": [],
                },
            ],
        });
        fs::write(
            root_dir.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let reader = PartitionArtifactShuffleReader::try_open_with_store(object_store, root_path)
            .await
            .unwrap();
        assert_eq!(reader.partition_size(0).unwrap(), 2);
        assert_eq!(reader.partition_size(1).unwrap(), 2);
        assert_eq!(reader.partition_size(2).unwrap(), 0);
        assert_eq!(reader.total_loss(), Some(1.5));

        let stream = reader.read_partition(0).await.unwrap().unwrap();
        let batches = stream.try_collect::<Vec<_>>().await.unwrap();
        let row_ids = batches
            .iter()
            .flat_map(|batch| {
                batch[ROW_ID]
                    .as_primitive::<arrow::datatypes::UInt64Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(row_ids, vec![10, 13]);
        assert!(reader.read_partition(2).await.unwrap().is_none());

        // Rows are a u64 row id plus a 2-byte PQ code: 10 bytes each. The
        // windowed partition build admits on this estimate; without it the
        // trait default charges usize::MAX and reads partitions one at a time.
        assert_eq!(reader.estimated_row_bytes, Some(10));
        let plan = reader.plan_partition_window(0, 1024).unwrap();
        assert_eq!(plan.partition_range, 0..1);
        assert_eq!(
            plan.estimated_decoded_bytes,
            20 + 20 / 4 + WINDOW_ADMISSION_FIXED_HEADROOM_BYTES
        );
        let plan = reader.plan_partition_window(2, 1024).unwrap();
        assert_eq!(plan.partition_range, 2..3);
        assert_eq!(plan.estimated_decoded_bytes, 0);
        assert!(reader.plan_partition_window(0, 0).is_err());
    }

    #[tokio::test]
    async fn partition_artifact_reader_rejects_missing_partition_entry() {
        let tempdir = tempfile::tempdir().unwrap();
        let root_dir = tempdir.path().join("artifact");
        fs::create_dir_all(&root_dir).unwrap();
        let manifest = serde_json::json!({
            "version": 1,
            "num_partitions": 2,
            "partitions": [{"num_rows": 0, "ranges": []}],
        });
        fs::write(
            root_dir.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let error = PartitionArtifactShuffleReader::try_open_with_store(
            Arc::new(ObjectStore::local()),
            Path::from_filesystem_path(&root_dir).unwrap(),
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
    }

    #[tokio::test]
    async fn partition_artifact_builder_coalesces_repeated_flush_ranges() {
        let tempdir = tempfile::tempdir().unwrap();
        let root_dir = tempdir.path().join("artifact");
        fs::create_dir_all(&root_dir).unwrap();
        let object_store = Arc::new(ObjectStore::local());
        let root_path = Path::from_filesystem_path(&root_dir).unwrap();

        let mut builder =
            PartitionArtifactBuilder::try_new_with_store(object_store, root_path, 4, 2).unwrap();
        let num_rows = PARTITION_ARTIFACT_BUCKET_BUFFER_ROWS + 1024;
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new(ROW_ID, DataType::UInt64, false),
            Field::new(PART_ID_COLUMN, DataType::UInt32, false),
            Field::new(
                PQ_CODE_COLUMN,
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::UInt8, true)), 2),
                true,
            ),
        ]));
        let row_ids = UInt64Array::from_iter_values((0..num_rows as u64).into_iter());
        let part_ids = UInt32Array::from_iter_values((0..num_rows).map(|_| 0_u32));
        let pq_values = UInt8Array::from_iter_values((0..num_rows * 2).map(|v| (v % 251) as u8));
        let pq_codes = FixedSizeListArray::try_new_from_values(pq_values, 2).unwrap();
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(row_ids), Arc::new(part_ids), Arc::new(pq_codes)],
        )
        .unwrap();

        builder.append_batch(&batch).await.unwrap();
        builder.finish("metadata.lance", None).await.unwrap();

        let manifest: PartitionArtifactManifest =
            serde_json::from_slice(&fs::read(root_dir.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest.partitions[0].num_rows, num_rows);
        assert_eq!(manifest.partitions[0].ranges.len(), 1);
        assert_eq!(manifest.partitions[0].ranges[0].offset, 0);
        assert_eq!(manifest.partitions[0].ranges[0].num_rows, num_rows as u64);
    }

    /// Builds an artifact from the same input with a given number of flush workers.
    async fn build_artifact_with_workers(
        root_dir: &std::path::Path,
        num_partitions: usize,
        workers: usize,
    ) -> Vec<String> {
        fs::create_dir_all(root_dir).unwrap();
        let object_store = Arc::new(ObjectStore::local());
        let root_path = Path::from_filesystem_path(root_dir).unwrap();
        let mut builder = PartitionArtifactBuilder::try_new_with_store(
            object_store,
            root_path,
            num_partitions,
            2,
        )
        .unwrap()
        .with_flush_workers(workers);
        // Rows concentrate on a few buckets so each is flushed several times.
        let hot = [0u32, 1, 2, 3, 256, 257, 300 % num_partitions as u32];
        let mut next_row = 0u64;
        for batch_idx in 0..8u64 {
            let rows = 50_000usize;
            let row_ids = UInt64Array::from_iter_values(next_row..next_row + rows as u64);
            next_row += rows as u64;
            let part_ids =
                UInt32Array::from_iter_values((0..rows).map(|i| {
                    hot[(i * 7 + batch_idx as usize) % hot.len()] % num_partitions as u32
                }));
            let codes = FixedSizeListArray::try_new_from_values(
                UInt8Array::from_iter_values((0..rows * 2).map(|i| (i % 251) as u8)),
                2,
            )
            .unwrap();
            let batch = RecordBatch::try_new(
                Arc::new(ArrowSchema::new(vec![
                    Field::new(ROW_ID, DataType::UInt64, false),
                    Field::new(PART_ID_COLUMN, DataType::UInt32, false),
                    Field::new(
                        PQ_CODE_COLUMN,
                        DataType::FixedSizeList(
                            Arc::new(Field::new("item", DataType::UInt8, true)),
                            2,
                        ),
                        true,
                    ),
                ])),
                vec![Arc::new(row_ids), Arc::new(part_ids), Arc::new(codes)],
            )
            .unwrap();
            builder.append_batch(&batch).await.unwrap();
        }
        builder.finish("metadata.lance", None).await.unwrap()
    }

    #[tokio::test]
    async fn partition_artifact_flush_workers_match_inline_flush() {
        for num_partitions in [300usize, 4] {
            let tempdir = tempfile::tempdir().unwrap();
            let inline_dir = tempdir.path().join("inline");
            let inline_files = build_artifact_with_workers(&inline_dir, num_partitions, 0).await;
            for workers in [1usize, 3] {
                let dir = tempdir.path().join(format!("workers{workers}"));
                let files = build_artifact_with_workers(&dir, num_partitions, workers).await;
                assert_eq!(files, inline_files, "files with {workers} workers");
                for file in &files {
                    assert_eq!(
                        fs::read(dir.join(file)).unwrap(),
                        fs::read(inline_dir.join(file)).unwrap(),
                        "{file} differs with {workers} workers ({num_partitions} partitions)"
                    );
                }
            }
        }
    }
}
