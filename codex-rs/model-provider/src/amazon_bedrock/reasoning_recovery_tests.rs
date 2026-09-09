use super::ReasoningRecovery;
use codex_api::ApiError;
use codex_api::TransportError;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use http::StatusCode;
use pretty_assertions::assert_eq;
use serde_json::json;

const REGION_ERROR: &str = "encrypted reasoning is scoped to the region that produced it and cannot be replayed in a different region. Start a new response in this region instead of threading reasoning from another region.";

fn http_error(status: StatusCode, code: &str, message: &str) -> ApiError {
    ApiError::Transport(TransportError::Http {
        status,
        url: None,
        headers: None,
        body: Some(
            json!({"error": {
                "code": code, "type": "invalid_request_error", "message": message
            }})
            .to_string(),
        ),
    })
}

fn reasoning(content: &str) -> ResponseItem {
    ResponseItem::Reasoning {
        id: None,
        summary: vec![],
        content: None,
        encrypted_content: Some(content.to_string()),
        internal_chat_message_metadata_passthrough: None,
    }
}

#[test]
fn recovery_filters_only_rejected_reasoning_and_preserves_new_reasoning() {
    let recovery = ReasoningRecovery::default();
    let old = reasoning("old-region");
    let fresh = reasoning("new-region");
    let message = ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: "continue".to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    };
    let compaction = ResponseItem::Compaction {
        id: None,
        encrypted_content: "compacted history".to_string(),
        internal_chat_message_metadata_passthrough: None,
    };
    let input = vec![compaction.clone(), old.clone(), message.clone()];
    let mut outgoing = input.clone();
    recovery.prepare(&mut outgoing);
    assert_eq!(outgoing, input);

    let error = http_error(StatusCode::BAD_REQUEST, "validation_error", REGION_ERROR);
    assert!(recovery.recover(&error, &input));
    outgoing.push(fresh.clone());
    recovery.prepare(&mut outgoing);
    assert_eq!(outgoing, vec![compaction, message.clone(), fresh.clone()]);
    assert!(!recovery.recover(&error, &input));

    // Another switch can reject the new region, while earlier rejected items stay excluded.
    assert!(recovery.recover(&error, std::slice::from_ref(&fresh)));
    let mut outgoing = vec![old, message.clone(), fresh];
    recovery.prepare(&mut outgoing);
    assert_eq!(outgoing, vec![message]);
}

#[test]
fn unrelated_errors_and_inputs_without_encrypted_reasoning_do_not_recover() {
    let recovery = ReasoningRecovery::default();
    let input = vec![reasoning("old-region")];
    for error in [
        http_error(StatusCode::BAD_REQUEST, "validation_error", "invalid input"),
        http_error(StatusCode::BAD_REQUEST, "other_error", REGION_ERROR),
        http_error(StatusCode::UNAUTHORIZED, "validation_error", REGION_ERROR),
        ApiError::Stream(REGION_ERROR.to_string()),
    ] {
        assert!(!recovery.recover(&error, &input));
    }
    let malformed = ApiError::Transport(TransportError::Http {
        status: StatusCode::BAD_REQUEST,
        url: None,
        headers: None,
        body: Some(REGION_ERROR.to_string()),
    });
    assert!(!recovery.recover(&malformed, &input));
    assert!(!recovery.recover(
        &http_error(StatusCode::BAD_REQUEST, "validation_error", REGION_ERROR),
        &[],
    ));
    let mut outgoing = input.clone();
    recovery.prepare(&mut outgoing);
    assert_eq!(outgoing, input);
}

#[test]
fn regenerated_compactions_are_replaced_atomically_and_can_switch_regions_again() {
    let recovery = ReasoningRecovery::default();
    let checkpoint = |content: &str| ResponseItem::Compaction {
        id: None,
        encrypted_content: content.to_owned(),
        internal_chat_message_metadata_passthrough: None,
    };
    let original = checkpoint("original");
    let east = checkpoint("east");
    let west = checkpoint("west");
    let fresh_reasoning = reasoning("fresh");
    for replacement in [&east, &west] {
        assert!(recovery.install_compactions(vec![(original.clone(), replacement.clone())]));
        let mut outgoing = vec![original.clone(), fresh_reasoning.clone()];
        recovery.prepare(&mut outgoing);
        assert_eq!(outgoing, vec![replacement.clone(), fresh_reasoning.clone()]);
    }
    assert!(!recovery.install_compactions(vec![
        (original.clone(), east),
        (checkpoint("other"), reasoning("not a checkpoint")),
    ]));
    let mut outgoing = vec![original];
    recovery.prepare(&mut outgoing);
    assert_eq!(outgoing, vec![west]);
}

#[test]
fn oversized_regenerated_compaction_is_not_installed() {
    let recovery = ReasoningRecovery::default();
    let original = ResponseItem::Compaction {
        id: None,
        encrypted_content: "original".into(),
        internal_chat_message_metadata_passthrough: None,
    };
    let oversized = ResponseItem::Compaction {
        id: None,
        encrypted_content: "x".repeat(super::MAX_COMPACTION_BYTES + 1),
        internal_chat_message_metadata_passthrough: None,
    };
    assert!(!recovery.install_compactions(vec![(original.clone(), oversized)]));
    let mut outgoing = vec![original.clone()];
    recovery.prepare(&mut outgoing);
    assert_eq!(outgoing, vec![original]);
}

#[test]
fn internal_server_error_recovery_is_bounded_and_keeps_fresh_reasoning() {
    let recovery = ReasoningRecovery::default();
    let error = ApiError::Transport(TransportError::Http {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        url: None,
        headers: None,
        body: Some(
            json!({"error": {
                "code": "internal_server_error",
                "type": "server_error",
                "message": "The server had an error while processing your request. Sorry about that!"
            }})
            .to_string(),
        ),
    });
    let old = reasoning("old-model");
    let fresh = reasoning("current-model");
    assert!(!recovery.recover(&error, &[]));
    assert!(recovery.recover(&error, std::slice::from_ref(&old)));
    assert!(!recovery.recover(&error, std::slice::from_ref(&old)));
    let mut input = vec![old, fresh.clone()];
    recovery.prepare(&mut input);
    assert_eq!(input, vec![fresh]);
}

#[test]
fn unrelated_internal_errors_do_not_discard_reasoning() {
    let recovery = ReasoningRecovery::default();
    let input = vec![reasoning("current-model")];
    for body in [
        "internal proxy error".to_string(),
        json!({"error": {
            "code": "other_error", "type": "server_error", "message": "other failure"
        }})
        .to_string(),
    ] {
        assert!(!recovery.recover(
            &ApiError::Transport(TransportError::Http {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                url: None,
                headers: None,
                body: Some(body),
            }),
            &input,
        ));
    }
    let mut outgoing = input.clone();
    recovery.prepare(&mut outgoing);
    assert_eq!(outgoing, input);
}
