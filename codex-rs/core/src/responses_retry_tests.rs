use super::ResponsesStreamRequest;
use super::ResponsesStreamRetryState;
use super::handle_response_stream_error;
use super::log_retry;
use crate::realtime_history::RealtimeHistoryState;
use crate::session::step_context::StepContext;
use crate::session::tests::make_session_and_context;
use crate::session::tests::make_session_and_context_with_auth_and_config_and_rx;
use codex_http_client::RetryAfter;
use codex_login::CodexAuth;
use codex_login::auth::BedrockApiKeyAuth;
use codex_model_provider_info::AMAZON_BEDROCK_RUNTIME_PROVIDER_ID;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::error::CodexErr;
use codex_protocol::error::CodexErrorDetails;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::time::Instant;
use tracing_test::internal::MockWriter;

#[tokio::test]
async fn sampling_retry_logs_stream_error_context() {
    let (_session, turn_context) = make_session_and_context().await;
    let buffer: &'static std::sync::Mutex<Vec<u8>> =
        Box::leak(Box::new(std::sync::Mutex::new(Vec::new())));
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .with_max_level(tracing::Level::WARN)
        .with_writer(MockWriter::new(buffer))
        .finish();
    let _subscriber_guard = tracing::subscriber::set_default(subscriber);

    log_retry(
        ResponsesStreamRequest::Sampling,
        &turn_context,
        &CodexErr::Stream("websocket closed by server before response.completed".to_string()),
        /*retries*/ 2,
        /*max_retries*/ 5,
        Duration::from_secs(1),
    );

    let logs = String::from_utf8(
        buffer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone(),
    )
    .expect("retry log should be valid utf-8");
    assert!(logs.contains("stream disconnected - retrying sampling request"));
    assert!(logs.contains(&format!("turn_id={}", turn_context.sub_id)));
    assert!(logs.contains("retries=2"));
    assert!(logs.contains("max_retries=5"));
    assert!(logs.contains(
        "sampling_error=stream disconnected before completion: websocket closed by server before response.completed"
    ));
}

/// Time spent reporting a retry must count toward the server's original deadline.
#[tokio::test]
#[expect(
    clippy::await_holding_invalid_type,
    reason = "test holds the event-delivery lock to delay notification while virtual time advances"
)]
async fn stream_retry_preserves_deadline_across_delayed_notification() {
    let (mut session, turn_context) = make_session_and_context().await;
    let step_context = StepContext::for_test(Arc::new(turn_context));
    session.realtime_history = Some(Mutex::new(RealtimeHistoryState::default()));
    let mut client_session = session.services.model_client.new_session();
    // The second retry emits a notification even when release builds hide the first.
    let mut retry_state = ResponsesStreamRetryState {
        retries: 1,
        ..Default::default()
    };

    tokio::time::pause();
    let advice = RetryAfter::from_delay(Duration::from_secs(10)).expect("retry deadline");
    tokio::time::advance(Duration::from_secs(4)).await;

    // Hold the event-delivery lock so reporting the error consumes part of the deadline.
    let history_guard = session.realtime_history.as_ref().unwrap().lock().await;
    let retry = handle_response_stream_error(
        &mut retry_state,
        /*max_retries*/ 2,
        CodexErr::InternalServerError.with_retry_after(advice),
        &mut client_session,
        &session,
        &step_context,
        ResponsesStreamRequest::Sampling,
    );
    tokio::pin!(retry);
    assert!(futures::poll!(&mut retry).is_pending());
    tokio::time::advance(Duration::from_secs(4)).await;
    drop(history_guard);

    assert!(futures::poll!(&mut retry).is_pending());
    tokio::time::advance(Duration::from_secs(1)).await;
    assert!(futures::poll!(&mut retry).is_pending());
    retry.await.expect("retry should be allowed");
    // Tokio rounds timer deadlines up to the next millisecond.
    let resumed_at = Instant::now();
    assert!(
        (advice.deadline()..=advice.deadline() + Duration::from_millis(1)).contains(&resumed_at),
        "retry resumed at {resumed_at:?}, expected {advice:?}"
    );
}

#[test_case::test_case(None; "local backoff capped")]
#[test_case::test_case(Some(90); "server deadline preserved")]
#[tokio::test]
async fn bedrock_compaction_retry_bounds_backoff_without_shortening_server_advice(
    server_delay_secs: Option<u64>,
) {
    let (session, turn_context, _events) = make_session_and_context_with_auth_and_config_and_rx(
        CodexAuth::BedrockApiKey(BedrockApiKeyAuth {
            api_key: "dummy".to_string(),
            region: "us-west-2".to_string(),
        }),
        Vec::new(),
        |config| {
            config.model_provider_id = AMAZON_BEDROCK_RUNTIME_PROVIDER_ID.to_string();
            config.model_provider = built_in_model_providers(/*openai_base_url*/ None)
                .remove(AMAZON_BEDROCK_RUNTIME_PROVIDER_ID)
                .expect("built-in Bedrock provider");
            config.model = Some("global.openai.gpt-6.1-sol".to_string());
        },
    )
    .await;
    let step_context = StepContext::for_test(turn_context);
    let mut client_session = session.services.model_client.new_session();
    let mut retry_state = ResponsesStreamRetryState {
        retries: u64::MAX,
        ..Default::default()
    };

    tokio::time::pause();
    let started_at = Instant::now();
    let mut err = CodexErr::Stream("stream closed before response.completed".to_string());
    if let Some(seconds) = server_delay_secs {
        err = err.with_retry_after(
            RetryAfter::from_delay(Duration::from_secs(seconds)).expect("retry deadline"),
        );
    }
    let expected_delay = Duration::from_secs(server_delay_secs.unwrap_or(60));
    let retry = handle_response_stream_error(
        &mut retry_state,
        /*max_retries*/ 2,
        err,
        &mut client_session,
        &session,
        &step_context,
        ResponsesStreamRequest::RemoteCompactionV2,
    );
    tokio::pin!(retry);
    assert!(futures::poll!(&mut retry).is_pending());
    tokio::time::advance(expected_delay - Duration::from_secs(1)).await;
    assert!(futures::poll!(&mut retry).is_pending());
    retry.await.expect("compaction disconnect should retry");
    let elapsed = Instant::now() - started_at;
    assert!(
        (expected_delay..=expected_delay + Duration::from_millis(1)).contains(&elapsed),
        "retry waited {elapsed:?}, expected {expected_delay:?}"
    );
}

#[tokio::test]
async fn bedrock_sampling_disconnects_retain_the_retry_budget() {
    let (session, turn_context, _events) = make_session_and_context_with_auth_and_config_and_rx(
        CodexAuth::BedrockApiKey(BedrockApiKeyAuth {
            api_key: "dummy".to_string(),
            region: "us-west-2".to_string(),
        }),
        Vec::new(),
        |config| {
            config.model_provider_id = AMAZON_BEDROCK_RUNTIME_PROVIDER_ID.to_string();
            config.model_provider = built_in_model_providers(/*openai_base_url*/ None)
                .remove(AMAZON_BEDROCK_RUNTIME_PROVIDER_ID)
                .expect("built-in Bedrock provider");
            config.model = Some("global.openai.gpt-6.1-sol".to_string());
        },
    )
    .await;
    let step_context = StepContext::for_test(turn_context);
    let mut client_session = session.services.model_client.new_session();
    let mut retry_state = ResponsesStreamRetryState {
        retries: 2,
        ..Default::default()
    };
    let error = handle_response_stream_error(
        &mut retry_state,
        /*max_retries*/ 2,
        CodexErr::Stream("stream closed before response.completed".to_string()),
        &mut client_session,
        &session,
        &step_context,
        ResponsesStreamRequest::Sampling,
    )
    .await
    .expect_err("sampling should still exhaust its retry budget");
    assert!(matches!(error.details(), CodexErrorDetails::Stream(_)));
}
