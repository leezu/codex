use std::fs::OpenOptions;
use std::io::Write;

use codex_protocol::ThreadId;
use codex_protocol::protocol::ThreadHistoryMode;

use super::MAX_SOURCE_BYTES;
use crate::LoadThreadHistoryParams;
use crate::LocalThreadStore;
use crate::ThreadStore;
use crate::local::test_support::test_config;
use crate::local::test_support::write_session_file_with_history_mode;

#[tokio::test]
async fn refuses_oversized_or_corrupt_source_instead_of_silently_losing_history() {
    enum Source {
        Oversized,
        Corrupt,
    }
    for source in [Source::Oversized, Source::Corrupt] {
        let home = tempfile::tempdir().expect("home");
        let uuid = uuid::Uuid::new_v4();
        let thread_id = ThreadId::from_string(&uuid.to_string()).expect("thread id");
        let path = write_session_file_with_history_mode(
            home.path(),
            "2025-01-03T12-00-00",
            uuid,
            ThreadHistoryMode::Paginated,
        )
        .expect("rollout");
        let mut file = OpenOptions::new().append(true).open(path).expect("open");
        let expected = match source {
            Source::Oversized => {
                file.set_len(MAX_SOURCE_BYTES + 1).expect("resize");
                "byte limit"
            }
            Source::Corrupt => {
                file.write_all(b"not valid json\n").expect("append");
                "cannot read checkpoint source history"
            }
        };
        let store = LocalThreadStore::new(test_config(home.path()), /*state_db*/ None);
        let error = store
            .load_compaction_history(LoadThreadHistoryParams {
                thread_id,
                include_archived: false,
            })
            .await
            .expect_err("unsafe recovery source must fail");
        assert!(error.to_string().contains(expected), "{error}");
    }
}
