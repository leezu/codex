//! Recreate region-bound checkpoints from their durable source windows.
//!
//! Work on a private rollout copy and install replacements only after every required
//! checkpoint has been regenerated. Never omit a checkpoint or replay tool execution.

use super::Session;
use super::step_context::StepContext;
use crate::client_common::Prompt;
use crate::compact_remote_v2::run_remote_compaction_request_v2;
use crate::context_manager::ContextManager;
use crate::responses_metadata::CodexResponsesRequestKind;
use crate::responses_metadata::CompactionTurnMetadata;
use codex_analytics::CompactionImplementation;
use codex_analytics::CompactionPhase;
use codex_analytics::CompactionReason;
use codex_analytics::CompactionTrigger;
use codex_history::RolloutItem;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result;
use codex_protocol::models::ResponseItem;
use std::collections::HashMap;
use std::collections::HashSet;

const MAX_CHECKPOINTS: usize = 16;
const MAX_REPLAY_ITEMS: usize = 20_000;
const MAX_REPLAY_BYTES: usize = 32 * 1024 * 1024;

impl Session {
    pub(crate) async fn recover_region_scoped_compaction(
        &self,
        step_context: &StepContext,
        error: &CodexErr,
    ) -> Result<bool> {
        let turn = &step_context.turn;
        if !self
            .services
            .model_client
            .provider()
            .is_compaction_recovery_candidate(error)
        {
            return Ok(false);
        }
        let current_history = self.clone_history().await;
        let mut pending: Vec<String> = current_history
            .raw_items()
            .filter_map(checkpoint_content)
            .map(str::to_owned)
            .collect();
        if pending.is_empty() {
            return Ok(false);
        }
        let live_thread = self.live_thread().ok_or_else(|| {
            recovery_error("the original transcript is not available for this session")
        })?;
        live_thread
            .flush()
            .await
            .map_err(|error| recovery_error(&format!("cannot flush the transcript: {error}")))?;
        let mut rollout = live_thread
            .load_compaction_history()
            .await
            .map_err(|error| recovery_error(&format!("cannot load the transcript: {error}")))?
            .items;
        let mut replacements = HashMap::new();
        let mut regenerated = HashSet::new();
        let mut client_session = self.services.model_client.new_session();
        let metadata = self
            .responses_metadata(
                turn,
                CodexResponsesRequestKind::Compaction(CompactionTurnMetadata::new(
                    CompactionTrigger::Auto,
                    CompactionReason::CompHashChanged,
                    CompactionImplementation::ResponsesCompactionV2,
                    CompactionPhase::MidTurn,
                )),
            )
            .await;

        while let Some(content) = pending.last().cloned() {
            if replacements.contains_key(&content) {
                pending.pop();
                continue;
            }
            if pending.len() + replacements.len() > MAX_CHECKPOINTS {
                return Err(recovery_error(
                    "too many dependent checkpoints to rebuild safely",
                ));
            }
            let index = rollout
                .iter()
                .rposition(|item| {
                    matches!(item, RolloutItem::Compacted(checkpoint)
                        if checkpoint.replacement_history.as_ref().is_some_and(|items|
                            items.iter().any(|item| checkpoint_content(&item.item) == Some(content.as_str()))))
                })
                .ok_or_else(|| recovery_error("the source checkpoint is missing from the transcript"))?;
            let reconstructed = self
                .reconstruct_history_from_rollout(turn, &rollout[..index])
                .await;
            if reconstructed.history.is_empty() {
                return Err(recovery_error("the checkpoint's source window is missing"));
            }
            // Rebuild ancestors first. Replaying the original reconstruction code preserves
            // rollback/fork boundaries and does not resurrect discarded turns.
            if let Some(ancestor) = reconstructed
                .history
                .iter()
                .filter_map(|item| checkpoint_content(&item.item))
                .find(|ancestor| !regenerated.contains(*ancestor))
            {
                if pending.iter().any(|item| item == ancestor) {
                    return Err(recovery_error(
                        "the checkpoint's source window is incomplete",
                    ));
                }
                pending.push(ancestor.to_owned());
                continue;
            }
            if reconstructed.history.len() > MAX_REPLAY_ITEMS {
                return Err(recovery_error(
                    "the source window exceeds the replay item limit",
                ));
            }
            let mut history = ContextManager::default();
            history.replace_annotated(reconstructed.history);
            let mut input = history.for_prompt(&step_context.settings.model_info.input_modalities);
            input.retain(|item| !matches!(item, ResponseItem::Reasoning { .. }));
            input.push(ResponseItem::CompactionTrigger {});
            let prompt = Prompt {
                input,
                tools: step_context.tool_router.model_visible_specs(),
                parallel_tool_calls: true,
                base_instructions: self.get_prompt_base_instructions().await,
                output_schema: None,
                output_schema_strict: true,
                cyber_access_program: turn.cyber_access_program,
            };
            let bytes = serde_json::to_vec(&prompt.input)?.len();
            if bytes > MAX_REPLAY_BYTES {
                return Err(recovery_error(
                    "the source window exceeds the replay byte limit",
                ));
            }
            let mut estimate = ContextManager::default();
            estimate.record_items(
                &prompt.input,
                step_context.settings.model_info.truncation_policy.into(),
            );
            let tokens = estimate
                .estimate_token_count_with_base_instructions(&prompt.base_instructions)
                .unwrap_or(i64::MAX);
            // A historical compaction window can exceed the normal conversation's
            // compaction threshold. Use the advertised capacity for this one-shot
            // reconstruction, retaining model headroom and room for the checkpoint.
            let model_info = &step_context.settings.model_info;
            let replay_limit = model_info
                .max_context_window
                .or(model_info.context_window)
                .map(|limit| {
                    limit.saturating_mul(model_info.effective_context_window_percent) / 100
                });
            if replay_limit.is_none_or(|limit| tokens > limit.saturating_mul(9) / 10) {
                return Err(recovery_error(
                    "the source window exceeds the active model's context budget",
                ));
            }
            tracing::info!(
                checkpoint_index = index,
                "regenerating region-bound compaction"
            );
            let output = run_remote_compaction_request_v2(
                self,
                step_context,
                &mut client_session,
                &prompt,
                &metadata,
            )
            .await?;
            if let Some(usage) = output.token_usage {
                self.record_rollout_budget_usage(&usage)?;
            }
            let new_content = checkpoint_content(&output.compaction_output)
                .ok_or_else(|| recovery_error("regeneration did not return a checkpoint"))?;
            if new_content == content {
                return Err(recovery_error(
                    "regeneration returned the rejected checkpoint",
                ));
            }
            regenerated.insert(new_content.to_owned());
            // Only this private copy is modified while dependent windows are regenerated.
            if let RolloutItem::Compacted(checkpoint) = &mut rollout[index]
                && let Some(items) = &mut checkpoint.replacement_history
            {
                for item in items {
                    if checkpoint_content(&item.item) == Some(content.as_str()) {
                        item.item = output.compaction_output.clone();
                    }
                }
            }
            replacements.insert(content, output.compaction_output);
            pending.pop();
        }

        let replacements = current_history
            .raw_items()
            .filter_map(|item| {
                checkpoint_content(item)
                    .and_then(|content| replacements.get(content))
                    .map(|replacement| (item.clone(), replacement.clone()))
            })
            .collect();
        if !self
            .services
            .model_client
            .provider()
            .install_compaction_replacements(replacements)
        {
            return Err(recovery_error(
                "regenerated checkpoints exceed the provider's cache limit",
            ));
        }
        Ok(true)
    }
}

fn checkpoint_content(item: &ResponseItem) -> Option<&str> {
    match item {
        ResponseItem::Compaction {
            encrypted_content, ..
        } => Some(encrypted_content),
        _ => None,
    }
}

fn recovery_error(reason: &str) -> CodexErr {
    CodexErr::InvalidRequest(format!(
        "Cannot migrate the encrypted compaction checkpoint to this region: {reason}. \
         The saved conversation has been preserved. Resume in the checkpoint's original \
         region and create a handoff summary for a new session."
    ))
}
