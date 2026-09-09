//! Bounded, read-only replay for regenerating encrypted checkpoints.

use std::io;
use std::io::Seek;
use std::io::SeekFrom;

use codex_protocol::protocol::ThreadHistoryMode;
use codex_rollout::ReverseJsonlScanner;
use codex_rollout::RolloutItem;
use codex_rollout::ScanOutcome;

use super::LocalThreadStore;
use super::read_thread;
use crate::LoadThreadHistoryParams;
use crate::ReadThreadParams;
use crate::StoredModelContext;
use crate::ThreadStoreError;
use crate::ThreadStoreResult;

const MAX_SOURCE_BYTES: u64 = 128 * 1024 * 1024;
const MAX_SOURCE_ITEMS: usize = 100_000;

#[cfg(test)]
#[path = "compaction_history_tests.rs"]
mod tests;

pub(super) async fn load(
    store: &LocalThreadStore,
    params: LoadThreadHistoryParams,
) -> ThreadStoreResult<StoredModelContext> {
    let thread = read_thread::read_thread(
        store,
        ReadThreadParams {
            thread_id: params.thread_id,
            include_archived: params.include_archived,
            include_history: false,
        },
    )
    .await?;
    if thread.history_mode == ThreadHistoryMode::Legacy {
        let history = store.load_history(params).await?;
        return Ok(StoredModelContext {
            thread_id: history.thread_id,
            items: history.items,
        });
    }

    let lineage = store.resolve_rollout_lineage(params.thread_id).await?;
    let source = lineage
        .segments()
        .last()
        .ok_or_else(|| ThreadStoreError::Internal {
            message: "checkpoint recovery has no source rollout".to_string(),
        })?;
    let session_meta = codex_rollout::read_session_meta_line(source.rollout_path.as_path())
        .await
        .map_err(|error| ThreadStoreError::Internal {
            message: format!("cannot read checkpoint recovery metadata: {error}"),
        })?;
    let items = tokio::task::spawn_blocking(move || -> io::Result<Vec<RolloutItem>> {
        let mut bytes = 0u64;
        let mut items = Vec::new();
        for segment in lineage.segments().iter().rev() {
            let mut file =
                codex_rollout::open_rollout_seekable_reader(segment.rollout_path.as_path())?;
            let end = match segment.end {
                Some(end) => end.end_byte_offset,
                None => file.seek(SeekFrom::End(0))?,
            };
            bytes = bytes.saturating_add(end);
            if bytes > MAX_SOURCE_BYTES {
                return Err(io::Error::other(
                    "checkpoint source history exceeds the byte limit",
                ));
            }
            let mut scanner = ReverseJsonlScanner::new_at(file, end)?;
            while let Some(outcome) = scanner.scan_next_rollout_line()? {
                let line = match outcome {
                    ScanOutcome::Parsed(line) => line,
                    ScanOutcome::Rejected(error) => return Err(io::Error::other(error)),
                };
                // The shared lineage resolver supplies frozen fork cutoffs. Each segment
                // contributes its own delta; use only the requested thread's session metadata.
                if matches!(line.item, RolloutItem::SessionMeta(_)) {
                    break;
                }
                if items.len() >= MAX_SOURCE_ITEMS {
                    return Err(io::Error::other(
                        "checkpoint source history exceeds the item limit",
                    ));
                }
                items.push(line.item);
            }
        }
        items.reverse();
        items.insert(0, RolloutItem::SessionMeta(session_meta));
        Ok(items)
    })
    .await
    .map_err(|error| ThreadStoreError::Internal {
        message: format!("checkpoint source reader failed: {error}"),
    })?
    .map_err(|error| ThreadStoreError::Internal {
        message: format!("cannot read checkpoint source history: {error}"),
    })?;
    Ok(StoredModelContext {
        thread_id: params.thread_id,
        items,
    })
}
