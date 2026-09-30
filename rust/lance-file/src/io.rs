// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use futures::{FutureExt, future::BoxFuture};
use lance_encoding::EncodingsIo;
use lance_io::scheduler::FileScheduler;

use super::reader::DEFAULT_READ_CHUNK_SIZE;

#[derive(Debug)]
pub struct LanceEncodingsIo {
    scheduler: FileScheduler,
    /// Size of chunks when reading large pages
    read_chunk_size: u64,
}

impl LanceEncodingsIo {
    pub fn new(scheduler: FileScheduler) -> Self {
        Self {
            scheduler,
            read_chunk_size: DEFAULT_READ_CHUNK_SIZE,
        }
    }

    pub fn with_read_chunk_size(mut self, read_chunk_size: u64) -> Self {
        self.read_chunk_size = read_chunk_size;
        self
    }
}

impl EncodingsIo for LanceEncodingsIo {
    fn submit_request(
        &self,
        ranges: Vec<std::ops::Range<u64>>,
        priority: u64,
    ) -> BoxFuture<'static, lance_core::Result<Vec<bytes::Bytes>>> {
        let mut split_ranges = Vec::new();
        let mut split_indices = Vec::new(); // Track which original range each split came from

        // A direct-I/O read is returned as one zero-copy buffer, so splitting
        // it below the store's IOP size gains nothing and costs a copy to
        // reassemble the chunks below. That copy runs on the async task that
        // awaits the read, and can delay the scheduler's I/O loop enough to
        // leave most I/O capacity idle.
        let read_chunk_size = if self.scheduler.is_direct_io() {
            self.read_chunk_size.max(self.scheduler.max_iop_size())
        } else {
            self.read_chunk_size
        };

        // Split large ranges into smaller chunks
        //
        // TODO: consider read_chunk_size before submitting requests.
        for (idx, range) in ranges.iter().enumerate() {
            let range_size = range.end - range.start;

            if range_size > read_chunk_size {
                let num_chunks = range_size.div_ceil(read_chunk_size);
                let chunk_size = range_size / num_chunks;

                for i in 0..num_chunks {
                    let start = range.start + i * chunk_size;
                    let end = if i == num_chunks - 1 {
                        range.end // Last chunk gets any remaining bytes
                    } else {
                        start + chunk_size
                    };
                    split_ranges.push(start..end);
                    split_indices.push(idx);
                }
            } else {
                split_ranges.push(range.clone());
                split_indices.push(idx);
            }
        }

        let fut = self.scheduler.submit_request(split_ranges, priority);

        async move {
            let split_results = fut.await?;

            // Fast path: if no splitting occurred, return results directly
            if split_results.len() == ranges.len() {
                return Ok(split_results);
            }

            // Slow path: reassemble split results
            let mut results = vec![Vec::new(); ranges.len()];

            for (split_result, &orig_idx) in split_results.iter().zip(split_indices.iter()) {
                results[orig_idx].push(split_result.clone());
            }

            Ok(results
                .into_iter()
                .map(|chunks| {
                    if chunks.len() == 1 {
                        chunks.into_iter().next().unwrap()
                    } else {
                        // Concatenate multiple chunks
                        let total_size: usize = chunks.iter().map(|c| c.len()).sum();
                        let mut combined = Vec::with_capacity(total_size);
                        for chunk in chunks {
                            combined.extend_from_slice(&chunk);
                        }
                        bytes::Bytes::from(combined)
                    }
                })
                .collect())
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_core::utils::tempfile::TempStdDir;
    use lance_io::object_store::ObjectStore;
    use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
    use lance_io::utils::CachedFileSize;
    use rstest::rstest;

    // A 32 MiB read (a Lance 2.0 page) is split into 8 MiB chunks, then 16 MiB
    // IOPs, on a buffered store, and read as one IOP on a direct-I/O store.
    #[rstest]
    #[case::buffered("file", 2)]
    #[cfg_attr(target_os = "linux", case::direct_io("file+direct", 1))]
    #[tokio::test]
    async fn test_large_read_iops(#[case] scheme: &str, #[case] expected_iops: u64) {
        const PAGE: usize = 32 * 1024 * 1024;
        let dir = TempStdDir::default();
        let file_path = dir.join("data.bin");
        let data: Vec<u8> = (0..PAGE + 4096).map(|i| (i % 251) as u8).collect();
        std::fs::write(&file_path, &data).unwrap();

        let uri = format!("{scheme}://{}", file_path.display());
        let (store, path) = ObjectStore::from_uri(&uri).await.unwrap();
        let scheduler = ScanScheduler::new(store, SchedulerConfig::default_for_testing());
        let file_scheduler = scheduler
            .open_file(&path, &CachedFileSize::unknown())
            .await
            .unwrap();
        let io = LanceEncodingsIo::new(file_scheduler);

        let range = 100..(100 + PAGE as u64);
        let bytes = io.submit_request(vec![range.clone()], 0).await.unwrap();
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0].as_ref(), &data[range.start as usize..range.end as usize]);
        assert_eq!(scheduler.stats().iops, expected_iops);
    }
}
