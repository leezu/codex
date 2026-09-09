//! Recover incompatible encrypted state without rewriting the saved conversation.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::PoisonError;

use codex_api::ApiError;
use codex_api::TransportError;
use codex_protocol::models::ResponseItem;
use http::StatusCode;
use serde::Deserialize;
use sha2::Digest;
use sha2::Sha256;

// Bound session state even if the endpoint repeatedly changes regions.
const MAX_REJECTED_ITEMS: usize = 16_384;
const MAX_COMPACTION_REPLACEMENTS: usize = 64;
const MAX_COMPACTION_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Default)]
pub(super) struct ReasoningRecovery {
    rejected: Mutex<HashSet<[u8; 32]>>,
    compactions: Mutex<HashMap<[u8; 32], ResponseItem>>,
}

#[derive(Deserialize)]
struct ErrorResponse {
    error: ResponseError,
}

#[derive(Deserialize)]
struct ResponseError {
    code: String,
    message: String,
    r#type: String,
}

impl ReasoningRecovery {
    pub(super) fn prepare(&self, input: &mut Vec<ResponseItem>) {
        let rejected = self.rejected.lock().unwrap_or_else(PoisonError::into_inner);
        if !rejected.is_empty() {
            input.retain(|item| reasoning_digest(item).is_none_or(|key| !rejected.contains(&key)));
        }
        let compactions = self
            .compactions
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        for item in input {
            if let ResponseItem::Compaction {
                encrypted_content, ..
            } = item
                && let Some(replacement) = compactions.get(&<[u8; 32]>::from(Sha256::digest(
                    encrypted_content.as_bytes(),
                )))
            {
                *item = replacement.clone();
            }
        }
    }

    pub(super) fn install_compactions(&self, items: Vec<(ResponseItem, ResponseItem)>) -> bool {
        let mut compactions = self
            .compactions
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mut updated = compactions.clone();
        for (original, replacement) in items {
            let ResponseItem::Compaction {
                encrypted_content, ..
            } = original
            else {
                return false;
            };
            if !matches!(replacement, ResponseItem::Compaction { .. }) {
                return false;
            }
            updated.insert(
                Sha256::digest(encrypted_content.as_bytes()).into(),
                replacement,
            );
        }
        let bytes = updated.values().try_fold(0usize, |total, item| {
            serde_json::to_vec(item)
                .ok()
                .and_then(|bytes| total.checked_add(bytes.len()))
        });
        if updated.len() > MAX_COMPACTION_REPLACEMENTS
            || bytes.is_none_or(|bytes| bytes > MAX_COMPACTION_BYTES)
        {
            return false;
        }
        *compactions = updated;
        true
    }

    pub(super) fn recover(&self, error: &ApiError, input: &[ResponseItem]) -> bool {
        let ApiError::Transport(TransportError::Http {
            status,
            body: Some(body),
            ..
        }) = error
        else {
            return false;
        };
        // Some Bedrock models report incompatible encrypted state as a generic 500.
        // Treat this as a bounded fallback, not proof that the region changed.
        let recoverable = match *status {
            StatusCode::BAD_REQUEST => is_region_error(body),
            StatusCode::INTERNAL_SERVER_ERROR => serde_json::from_str::<ErrorResponse>(body)
                .is_ok_and(|response| {
                    response.error.code == "internal_server_error"
                        && response.error.r#type == "server_error"
                }),
            _ => false,
        };
        if !recoverable {
            return false;
        }

        let mut rejected = self.rejected.lock().unwrap_or_else(PoisonError::into_inner);
        let keys: HashSet<_> = input.iter().filter_map(reasoning_digest).collect();
        let new_count = keys.difference(&rejected).count();
        if new_count == 0 || rejected.len() + new_count > MAX_REJECTED_ITEMS {
            return false;
        }
        rejected.extend(keys);
        true
    }
}

pub(super) fn is_region_error(body: &str) -> bool {
    let Ok(response) = serde_json::from_str::<ErrorResponse>(body) else {
        return false;
    };
    response.error.code == "validation_error"
        && response.error.r#type == "invalid_request_error"
        && response.error.message.contains(
            "encrypted reasoning is scoped to the region that produced it and cannot be replayed in a different region",
        )
}

fn reasoning_digest(item: &ResponseItem) -> Option<[u8; 32]> {
    match item {
        ResponseItem::Reasoning {
            encrypted_content: Some(content),
            ..
        } => Some(Sha256::digest(content.as_bytes()).into()),
        _ => None,
    }
}

#[cfg(test)]
#[path = "reasoning_recovery_tests.rs"]
mod tests;
