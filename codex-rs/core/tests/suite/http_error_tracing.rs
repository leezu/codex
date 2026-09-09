use codex_core::ModelClient;
use codex_core::Prompt;
use codex_login::AuthManager;
use codex_login::CodexAuth;
use codex_login::auth::AgentIdentityAuthPolicy;
use codex_otel::SessionTelemetry;
use codex_protocol::ThreadId;
use codex_protocol::config_types::ReasoningSummary;
use codex_protocol::error::CodexErrorDetails;
use codex_protocol::protocol::SessionSource;
use codex_rollout_trace::InferenceTraceContext;
use codex_rollout_trace::TraceWriter;
use core_test_support::TestCodexResponsesRequestKind;
use core_test_support::load_default_config_for_test;
use core_test_support::responses_metadata;
use pretty_assertions::assert_eq;
use serde_json::Value;
use std::sync::Arc;
use test_case::test_case;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

#[test_case(r#"{"error":{"code":"internal_error","message":"checkpoint could not be decoded"}}"#; "json")]
#[test_case("upstream failure: région unavailable"; "plain_text")]
#[tokio::test]
async fn http_500_trace_preserves_details_and_retry_classification(
    body: &str,
) -> anyhow::Result<()> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(
            ResponseTemplate::new(/*s*/ 500)
                .insert_header("x-request-id", "req-diagnostic")
                .insert_header("set-cookie", "private-response-cookie")
                .set_body_string(body),
        )
        .expect(1)
        .mount(&server)
        .await;
    let home = tempfile::tempdir()?;
    let config = load_default_config_for_test(&home).await;
    let model = codex_core::test_support::get_model_offline(config.model.as_deref());
    let model_info = codex_core::test_support::construct_model_info_offline(&model, &config);
    let mut provider = config.model_provider.clone();
    provider.base_url = Some(format!("{}/v1", server.uri()));
    provider.request_max_retries = Some(0);
    provider.supports_websockets = false;
    let thread_id = ThreadId::new();
    let telemetry = SessionTelemetry::new(
        thread_id,
        &model,
        &model_info.slug,
        /*account_id*/ None,
        /*account_email*/ None,
        /*auth_mode*/ None,
        "test".to_string(),
        /*log_user_prompts*/ false,
        "test".to_string(),
        SessionSource::Exec,
    );
    let client = ModelClient::new(
        Some(AuthManager::from_auth_for_testing(CodexAuth::from_api_key(
            "test-api-key",
        ))),
        AgentIdentityAuthPolicy::JwtOnly,
        thread_id,
        provider,
        SessionSource::Exec,
        "test".to_string(),
        config.model_verbosity,
        /*content_item_kinds_enabled*/ false,
        /*enable_request_compression*/ false,
        /*include_timing_metrics*/ false,
        /*beta_features_header*/ None,
        /*concurrent_reasoning_summaries_enabled*/ false,
        /*attestation_provider*/ None,
        config.http_client_factory(),
    );
    let trace_dir = tempfile::tempdir()?;
    let writer = Arc::new(TraceWriter::create(
        trace_dir.path(),
        "trace-diagnostic".to_string(),
        "rollout-diagnostic".to_string(),
        thread_id.to_string(),
    )?);
    let trace = InferenceTraceContext::enabled(
        writer,
        thread_id.to_string(),
        "turn-diagnostic".to_string(),
        model,
        "test".to_string(),
    );
    let metadata = responses_metadata(
        "test-installation",
        &thread_id.to_string(),
        &thread_id.to_string(),
        /*turn_id*/ None,
        "test-window".to_string(),
        &SessionSource::Exec,
        /*parent_thread_id*/ None,
        TestCodexResponsesRequestKind::Turn,
    );
    let result = client
        .new_session()
        .stream(
            &Prompt::default(),
            &model_info,
            &telemetry,
            /*effort*/ None,
            ReasoningSummary::None,
            /*service_tier*/ None,
            &metadata,
            &trace,
        )
        .await;
    let error = result.err().expect("HTTP 500 must fail");
    assert!(matches!(
        error.details(),
        CodexErrorDetails::InternalServerError
    ));
    assert!(error.is_retryable());
    let events = std::fs::read_to_string(trace_dir.path().join("trace.jsonl"))?;
    let failures = events
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|event| event["payload"].clone())
        .filter(|event| event["type"] == "inference_failed")
        .map(|event| {
            serde_json::json!({
                "error": event["error"],
                "upstream_request_id": event["upstream_request_id"],
                "partial_response_payload": event["partial_response_payload"],
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        failures,
        vec![serde_json::json!({
            "error": format!("http 500 Internal Server Error: {:?}", Some(body)),
            "upstream_request_id": "req-diagnostic",
            "partial_response_payload": null,
        })],
    );
    server.verify().await;
    Ok(())
}
