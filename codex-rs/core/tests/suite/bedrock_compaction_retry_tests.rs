use std::time::Duration;

use codex_login::CodexAuth;
use codex_login::auth::BedrockApiKeyAuth;
use codex_model_provider_info::AMAZON_BEDROCK_PROVIDER_ID;
use codex_model_provider_info::AMAZON_BEDROCK_RUNTIME_PROVIDER_ID;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::protocol::ThreadHistoryMode;
use codex_protocol::turn_input::TurnInputRequest;
use codex_protocol::user_input::UserInput;
use core_test_support::responses;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::TestCodexBuilder;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use wiremock::ResponseTemplate;

fn bedrock_test(provider_id: &'static str, model: &'static str) -> TestCodexBuilder {
    test_codex()
        .with_auth(CodexAuth::BedrockApiKey(BedrockApiKeyAuth {
            api_key: "dummy".to_string(),
            region: "us-west-2".to_string(),
        }))
        .with_model(model)
        .with_history_mode(ThreadHistoryMode::Paginated)
        .with_config(move |config| {
            let base_url = config.model_provider.base_url.clone();
            config.model_provider_id = provider_id.to_string();
            config.model_provider = built_in_model_providers(/*openai_base_url*/ None)
                .remove(provider_id)
                .expect("built-in Bedrock provider");
            config.model_provider.base_url = base_url;
            config.model_provider.request_max_retries = Some(0);
            config.model_provider.stream_max_retries = Some(2);
            config.service_tier = Some("ultrafast".to_string());
        })
}

fn compaction_response(encrypted_content: &str, completed: bool) -> String {
    let mut events = vec![serde_json::json!({
        "type": "response.output_item.done",
        "item": {
            "type": "compaction",
            "encrypted_content": encrypted_content,
        }
    })];
    if completed {
        events.push(responses::ev_completed("compacted"));
    }
    responses::sse(events)
}

#[test_case::test_case(AMAZON_BEDROCK_PROVIDER_ID, "openai.gpt-6.1-sol", false; "mantle manual")]
#[test_case::test_case(AMAZON_BEDROCK_PROVIDER_ID, "openai.gpt-6.1-sol", true; "mantle automatic")]
#[test_case::test_case(AMAZON_BEDROCK_RUNTIME_PROVIDER_ID, "global.openai.gpt-6.1-sol", false; "runtime manual")]
#[test_case::test_case(AMAZON_BEDROCK_RUNTIME_PROVIDER_ID, "global.openai.gpt-6.1-sol", true; "runtime automatic")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bedrock_compaction_recovers_after_repeated_stream_disconnects(
    provider_id: &'static str,
    model: &'static str,
    automatic: bool,
) -> anyhow::Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let mut bodies = vec![responses::sse(vec![
        responses::ev_assistant_message("initial", "ORIGINAL_CONTEXT"),
        responses::ev_completed_with_tokens("initial", /*total_tokens*/ 500),
    ])];
    // Exceed both the compaction cap (two) and the default stream retry budget (five).
    bodies.extend((0..6).map(|_| compaction_response("DISCARDED_COMPACTION", false)));
    bodies.push(compaction_response("RECOVERED_COMPACTION", true));
    bodies.push(responses::sse(vec![responses::ev_completed("follow-up")]));
    let mock = responses::mount_sse_sequence(&server, bodies).await;
    let test = bedrock_test(provider_id, model)
        .with_config(move |config| {
            config.model_auto_compact_token_limit = automatic.then_some(200);
        })
        .build_with_auto_env(&server)
        .await?;
    test.submit_turn("original user message").await?;
    if automatic {
        test.codex
            .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
                text: "after compaction".to_string(),
                text_elements: Vec::new(),
            }]))
            .await?;
    } else {
        test.codex.submit(Op::Compact).await?;
    }

    let retry_count = tokio::time::timeout(Duration::from_secs(30), async {
        let mut retries = 0;
        loop {
            match test.codex.next_event().await?.msg {
                EventMsg::StreamError(error) => {
                    assert!(
                        error
                            .additional_details
                            .as_deref()
                            .unwrap_or_default()
                            .contains("stream closed before response.completed")
                    );
                    retries += 1;
                }
                EventMsg::Error(error) => anyhow::bail!("compaction failed: {}", error.message),
                EventMsg::TurnComplete(completed) => {
                    assert!(completed.error.is_none());
                    return Ok::<_, anyhow::Error>(retries);
                }
                _ => {}
            }
        }
    })
    .await??;
    assert_eq!(retry_count, 6);
    if !automatic {
        test.submit_turn("after compaction").await?;
    }

    let requests = mock.requests();
    assert_eq!(requests.len(), 9);
    let original_input = requests[1].input();
    for request in &requests[1..8] {
        assert_eq!(request.body_json()["model"], model);
        assert_eq!(request.body_json()["service_tier"], "ultrafast");
        assert_eq!(request.input(), original_input);
        assert_eq!(request.inputs_of_type("compaction_trigger").len(), 1);
    }
    let follow_up = requests[8].body_json().to_string();
    assert!(follow_up.contains("RECOVERED_COMPACTION"));
    assert!(follow_up.contains("original user message"));
    assert!(!follow_up.contains("DISCARDED_COMPACTION"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bedrock_compaction_retry_can_be_interrupted() -> anyhow::Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let mut bodies = vec![responses::sse(vec![responses::ev_completed("initial")])];
    bodies.extend((0..3).map(|_| compaction_response("DISCARDED_COMPACTION", false)));
    bodies.push(responses::sse(vec![responses::ev_completed("follow-up")]));
    let mock = responses::mount_sse_sequence(&server, bodies).await;
    let test = bedrock_test(
        AMAZON_BEDROCK_RUNTIME_PROVIDER_ID,
        "global.openai.gpt-6.1-sol",
    )
    .build_with_auto_env(&server)
    .await?;
    test.submit_turn("before interrupted compaction").await?;
    test.codex.submit(Op::Compact).await?;

    // Interrupt during backoff after crossing the old two-retry limit.
    for _ in 0..3 {
        let event = wait_for_event(&test.codex, |event| {
            matches!(
                event,
                EventMsg::StreamError(_) | EventMsg::Error(_) | EventMsg::TurnComplete(_)
            )
        })
        .await;
        assert!(matches!(event, EventMsg::StreamError(_)), "{event:?}");
    }
    test.codex.submit(Op::Interrupt).await?;
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnAborted(_))
    })
    .await;
    // A subsequent turn still uses the uncompacted history, and no retry leaks into it.
    test.submit_turn("after interruption").await?;
    let requests = mock.requests();
    assert_eq!(requests.len(), 5);
    let follow_up = requests[4].body_json().to_string();
    assert!(follow_up.contains("before interrupted compaction"));
    assert!(!follow_up.contains("DISCARDED_COMPACTION"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bedrock_compaction_invalid_request_is_terminal() -> anyhow::Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let mock = responses::mount_response_sequence(
        &server,
        vec![
            responses::sse_response(responses::sse(vec![responses::ev_completed("initial")])),
            ResponseTemplate::new(400).set_body_json(serde_json::json!({
                "error": {
                    "message": "invalid compaction request",
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                }
            })),
            responses::sse_response(responses::sse(vec![responses::ev_completed("follow-up")])),
        ],
    )
    .await;
    let test = bedrock_test(
        AMAZON_BEDROCK_RUNTIME_PROVIDER_ID,
        "global.openai.gpt-6.1-sol",
    )
    .build_with_auto_env(&server)
    .await?;
    test.submit_turn("before failed compaction").await?;
    test.codex.submit(Op::Compact).await?;
    let error = wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::Error(_) | EventMsg::StreamError(_))
    })
    .await;
    assert!(matches!(error, EventMsg::Error(_)), "{error:?}");
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    test.submit_turn("after failed compaction").await?;
    assert_eq!(mock.requests().len(), 3);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_bedrock_compaction_still_exhausts_its_retry_budget() -> anyhow::Result<()> {
    skip_if_no_network!(Ok(()));
    let server = responses::start_mock_server().await;
    let mut bodies = vec![responses::sse(vec![responses::ev_completed("initial")])];
    bodies.extend((0..3).map(|_| compaction_response("DISCARDED_COMPACTION", false)));
    bodies.push(responses::sse(vec![responses::ev_completed("follow-up")]));
    let mock = responses::mount_sse_sequence(&server, bodies).await;
    let test = test_codex()
        .with_auth(CodexAuth::create_dummy_chatgpt_auth_for_testing())
        .with_history_mode(ThreadHistoryMode::Paginated)
        .with_config(|config| {
            config.model_provider.request_max_retries = Some(0);
            config.model_provider.stream_max_retries = Some(2);
        })
        .build_with_auto_env(&server)
        .await?;
    test.submit_turn("before failed compaction").await?;
    test.codex.submit(Op::Compact).await?;
    let mut retries = 0;
    wait_for_event(&test.codex, |event| match event {
        EventMsg::StreamError(_) => {
            retries += 1;
            false
        }
        EventMsg::Error(_) => true,
        _ => false,
    })
    .await;
    assert_eq!(retries, 2);
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    test.submit_turn("after failed compaction").await?;
    assert_eq!(mock.requests().len(), 5);
    Ok(())
}
