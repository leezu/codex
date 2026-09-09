use codex_login::CodexAuth;
use codex_login::auth::BedrockApiKeyAuth;
use codex_model_provider_info::AMAZON_BEDROCK_GPT_5_5_MODEL_ID;
use codex_model_provider_info::AMAZON_BEDROCK_PROVIDER_ID;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::turn_input::TurnInputRequest;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_function_call;
use core_test_support::responses::ev_reasoning_item;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::mount_sse_once;
use core_test_support::responses::sse;
use core_test_support::test_codex::TestCodexBuilder;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use serde_json::json;
use test_case::test_case;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;
use wiremock::matchers::body_string_contains;
use wiremock::matchers::method;
use wiremock::matchers::path;

const INTERNAL_ERROR: &str =
    "The server had an error while processing your request. Sorry about that!";

#[test_case("encrypted reasoning is scoped to the region that produced it and cannot be replayed in a different region. Start a new response in this region instead of threading reasoning from another region.", 1, 400; "recovers")]
#[test_case("encrypted reasoning is scoped to the region that produced it and cannot be replayed in a different region.", 2, 400; "stops_after_one_retry")]
#[test_case("invalid tool schema", 1, 400; "unrelated_validation")]
#[test_case(INTERNAL_ERROR, 1, 500; "internal_error_recovers")]
#[test_case(INTERNAL_ERROR, 2, 500; "internal_error_stops_after_one_retry")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bedrock_reasoning_recovery(
    error_message: &str,
    error_count: u64,
    status: u16,
) -> anyhow::Result<()> {
    let server = MockServer::start().await;
    let test = bedrock_builder(&server)
        .build_with_auto_env(&server)
        .await?;

    // Seed reasoning and a tool call/output before moving to the rejecting endpoint.
    mount_sse_once(
        &server,
        sse(vec![
            ev_response_created("resp-old"),
            ev_reasoning_item("rs_old", &["old summary"], &["old region"]),
            ev_function_call("call-old", "unsupported_tool", "{}"),
            ev_completed("resp-old"),
        ]),
    )
    .await;
    mount_sse_once(
        &server,
        sse(vec![
            ev_response_created("resp-answer"),
            ev_assistant_message("msg-answer", "old answer"),
            ev_completed("resp-answer"),
        ]),
    )
    .await;
    test.submit_turn("first turn").await?;

    let success = mount_sse_once(
        &server,
        sse(vec![
            ev_response_created("resp-new"),
            ev_reasoning_item("rs_new", &["new summary"], &["new region"]),
            ev_assistant_message("msg-new", "recovered"),
            ev_completed("resp-new"),
        ]),
    )
    .await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(error_response(status, error_message))
        .up_to_n_times(error_count)
        .expect(error_count)
        .with_priority(/*p*/ 1)
        .mount(&server)
        .await;

    let recovers =
        error_count == 1 && (error_message.starts_with("encrypted reasoning") || status == 500);
    if !recovers {
        test.codex
            .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
                text: "second turn".into(),
                text_elements: vec![],
            }]))
            .await?;
        wait_for_event(&test.codex, |event| matches!(event, EventMsg::Error(_))).await;
        assert_eq!(success.requests().len(), 0);
        server.verify().await;
        return Ok(());
    }

    test.submit_turn("second turn").await?;
    let retry = success.single_request().body_json();
    let requests = server.received_requests().await.expect("recorded requests");
    let failed: serde_json::Value = requests
        .iter()
        .filter(|request| request.url.path() == "/v1/responses")
        .nth(/*n*/ 2)
        .expect("rejected request after the initial turn")
        .body_json()?;
    let mut expected = failed.clone();
    expected["input"]
        .as_array_mut()
        .expect("request input array")
        .retain(|item| item["type"] != "reasoning");
    assert_eq!(retry["input"], expected["input"]);
    assert!(
        failed["input"]
            .as_array()
            .expect("request input array")
            .iter()
            .any(|item| { item["type"] == "reasoning" })
    );
    assert!(
        retry["input"]
            .as_array()
            .expect("request input array")
            .iter()
            .any(|item| { item["type"] == "function_call_output" })
    );

    let next_turn = mount_sse_once(
        &server,
        sse(vec![
            ev_response_created("resp-next"),
            ev_completed("resp-next"),
        ]),
    )
    .await;
    test.submit_turn("third turn").await?;
    let body = next_turn.single_request().body_json();
    let reasoning: Vec<_> = body["input"]
        .as_array()
        .expect("request input array")
        .iter()
        .filter(|item| item["type"] == "reasoning")
        .cloned()
        .collect();
    assert_eq!(
        reasoning,
        vec![ev_reasoning_item("rs_new", &["new summary"], &["new region"])["item"].clone()]
    );
    server.verify().await;
    Ok(())
}

fn bedrock_builder(server: &MockServer) -> TestCodexBuilder {
    let mut provider = built_in_model_providers(/*openai_base_url*/ None)
        .remove(AMAZON_BEDROCK_PROVIDER_ID)
        .expect("built-in Bedrock provider");
    provider.base_url = Some(format!("{}/v1", server.uri()));
    provider.aws = None;
    provider.request_max_retries = Some(0);
    provider.stream_max_retries = Some(0);
    test_codex()
        .with_model(AMAZON_BEDROCK_GPT_5_5_MODEL_ID)
        .with_auth(CodexAuth::BedrockApiKey(BedrockApiKeyAuth {
            api_key: "test-bedrock-token".into(),
            region: "us-east-1".into(),
        }))
        .with_config(move |config| {
            let _ = config
                .features
                .enable(codex_features::Feature::RemoteCompactionV2);
            config.model_provider_id = AMAZON_BEDROCK_PROVIDER_ID.into();
            config.model_provider = provider;
        })
}

#[derive(Clone, Copy)]
enum CompactionRecoveryCase {
    Live,
    Resumed,
    Nested,
    MissingSource,
    DuringCompaction,
    RegenerationFails,
    Forked,
    Mixed,
    StillFails,
    PaginatedLive,
    PaginatedForked,
    PaginatedNested,
    ExtendedWindow,
    SourceExceedsCapacity,
}

#[test_case(CompactionRecoveryCase::Live, 400; "live")]
#[test_case(CompactionRecoveryCase::Resumed, 400; "resumed")]
#[test_case(CompactionRecoveryCase::Nested, 400; "nested")]
#[test_case(CompactionRecoveryCase::MissingSource, 400; "missing_source")]
#[test_case(CompactionRecoveryCase::DuringCompaction, 400; "during_compaction")]
#[test_case(CompactionRecoveryCase::RegenerationFails, 400; "regeneration_fails")]
#[test_case(CompactionRecoveryCase::Live, 500; "internal_live")]
#[test_case(CompactionRecoveryCase::Resumed, 500; "internal_resumed")]
#[test_case(CompactionRecoveryCase::Nested, 500; "internal_nested")]
#[test_case(CompactionRecoveryCase::MissingSource, 500; "internal_missing_source")]
#[test_case(CompactionRecoveryCase::DuringCompaction, 500; "internal_during_compaction")]
#[test_case(CompactionRecoveryCase::RegenerationFails, 500; "internal_regeneration_fails")]
#[test_case(CompactionRecoveryCase::Forked, 500; "internal_forked")]
#[test_case(CompactionRecoveryCase::Mixed, 500; "internal_mixed")]
#[test_case(CompactionRecoveryCase::StillFails, 500; "internal_stops_after_one_regeneration")]
#[test_case(CompactionRecoveryCase::PaginatedLive, 500; "internal_paginated_live")]
#[test_case(CompactionRecoveryCase::PaginatedForked, 500; "internal_paginated_forked")]
#[test_case(CompactionRecoveryCase::PaginatedNested, 500; "internal_paginated_nested")]
#[test_case(CompactionRecoveryCase::ExtendedWindow, 500; "internal_extended_window")]
#[test_case(CompactionRecoveryCase::SourceExceedsCapacity, 500; "internal_source_exceeds_capacity")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bedrock_compaction_recovery(
    case: CompactionRecoveryCase,
    status: u16,
) -> anyhow::Result<()> {
    let server = MockServer::start().await;
    let mut builder = bedrock_builder(&server);
    if matches!(
        case,
        CompactionRecoveryCase::PaginatedLive
            | CompactionRecoveryCase::PaginatedForked
            | CompactionRecoveryCase::PaginatedNested
    ) {
        builder = builder.with_history_mode(codex_protocol::protocol::ThreadHistoryMode::Paginated);
    }
    let mut test = builder.build_with_auto_env(&server).await?;
    mount_sse_once(
        &server,
        sse(vec![
            ev_reasoning_item("rs-source", &["source summary"], &["source reasoning"]),
            ev_function_call("call-source", "unsupported_tool", "{}"),
            ev_completed("resp-source"),
        ]),
    )
    .await;
    let source_answer = if matches!(
        case,
        CompactionRecoveryCase::ExtendedWindow | CompactionRecoveryCase::SourceExceedsCapacity
    ) {
        "Preserve this earlier decision. ".repeat(5_000)
    } else {
        "Preserve this earlier decision.".into()
    };
    mount_sse_once(
        &server,
        sse(vec![
            ev_assistant_message("msg-source", &source_answer),
            ev_completed("resp-source-answer"),
        ]),
    )
    .await;
    test.submit_turn("remember this task").await?;
    let original_compaction = mount_sse_once(&server, compaction_response("old-checkpoint")).await;
    test.codex.submit(Op::Compact).await?;
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    let original_input = original_compaction.single_request().input();
    if matches!(case, CompactionRecoveryCase::Mixed) {
        mount_sse_once(
            &server,
            sse(vec![
                ev_reasoning_item("rs-after", &["later summary"], &["later reasoning"]),
                ev_assistant_message("msg-after", "Keep this post-compaction decision."),
                ev_completed("resp-after"),
            ]),
        )
        .await;
        test.submit_turn("continue before switching").await?;
    }
    let mut second_input = None;
    if matches!(
        case,
        CompactionRecoveryCase::Nested | CompactionRecoveryCase::PaginatedNested
    ) {
        mount_sse_once(
            &server,
            sse(vec![
                ev_assistant_message("msg-later", "Preserve a later decision too."),
                ev_completed("resp-later"),
            ]),
        )
        .await;
        test.submit_turn("another task").await?;
        let second = mount_sse_once(&server, compaction_response("second-checkpoint")).await;
        test.codex.submit(Op::Compact).await?;
        wait_for_event(&test.codex, |event| {
            matches!(event, EventMsg::TurnComplete(_))
        })
        .await;
        second_input = Some(second.single_request().input());
    }
    if matches!(case, CompactionRecoveryCase::MissingSource) {
        test.codex.shutdown_and_wait().await?;
        let path = test
            .session_configured
            .rollout_path
            .as_ref()
            .expect("rollout path");
        let mut lines = Vec::new();
        for line in std::fs::read_to_string(path)?.lines() {
            let item: serde_json::Value = serde_json::from_str(line)?;
            if item["type"] != "response_item" {
                lines.push(line.to_owned());
            }
        }
        std::fs::write(path, format!("{}\n", lines.join("\n")))?;
        test = bedrock_builder(&server)
            .resume(&server, std::sync::Arc::clone(&test.home), path.clone())
            .await?;
    } else if matches!(
        case,
        CompactionRecoveryCase::ExtendedWindow | CompactionRecoveryCase::SourceExceedsCapacity
    ) {
        test = bedrock_builder(&server)
            .with_model_info_override("gpt-5.5", move |model| {
                model.slug = AMAZON_BEDROCK_GPT_5_5_MODEL_ID.into();
                model.max_context_window = Some(
                    if matches!(case, CompactionRecoveryCase::SourceExceedsCapacity) {
                        10_000
                    } else {
                        100_000
                    },
                );
            })
            .with_model(AMAZON_BEDROCK_GPT_5_5_MODEL_ID)
            .with_config(|config| config.model_context_window = Some(10_000))
            .restart(&server, &test)
            .await?;
    } else if matches!(
        case,
        CompactionRecoveryCase::Resumed | CompactionRecoveryCase::Nested
    ) {
        test = bedrock_builder(&server).restart(&server, &test).await?;
    } else if matches!(
        case,
        CompactionRecoveryCase::Forked | CompactionRecoveryCase::PaginatedForked
    ) {
        let options = codex_core::StartThreadOptions::new(test.config.clone());
        let fork = if matches!(case, CompactionRecoveryCase::PaginatedForked) {
            let prepared = test
                .thread_store
                .prepare_fork(codex_thread_store::PrepareForkParams {
                    thread_id: test.session_configured.thread_id,
                    boundary: codex_thread_store::ForkBoundary::Latest,
                })
                .await?;
            test.thread_manager
                .fork_prepared_thread(options, prepared)
                .await?
        } else {
            test.thread_manager
                .fork_thread(
                    codex_core::ForkSnapshot::Interrupted,
                    options,
                    test.codex.rollout_path().expect("parent rollout"),
                )
                .await?
        };
        test.codex = fork.thread;
        test.session_configured = fork.session_configured;
    }

    for replacement in ["east-checkpoint", "west-checkpoint"] {
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(error_response(
                status,
                if status == 500 {
                    INTERNAL_ERROR
                } else {
                    "encrypted reasoning is scoped to the region that produced it and cannot be replayed in a different region."
                },
            ))
            .up_to_n_times(if matches!(case, CompactionRecoveryCase::Mixed)
                && replacement == "east-checkpoint"
            {
                2
            } else {
                1
            })
            .with_priority(/*p*/ 1)
            .mount(&server)
            .await;

        if matches!(
            case,
            CompactionRecoveryCase::RegenerationFails
                | CompactionRecoveryCase::MissingSource
                | CompactionRecoveryCase::SourceExceedsCapacity
        ) {
            if matches!(case, CompactionRecoveryCase::RegenerationFails) {
                Mock::given(method("POST"))
                    .and(path("/v1/responses"))
                    .respond_with(
                        ResponseTemplate::new(/*s*/ 400).set_body_string("regeneration failed"),
                    )
                    .up_to_n_times(/*n*/ 1)
                    .with_priority(/*p*/ 2)
                    .mount(&server)
                    .await;
            }
            test.codex
                .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
                    text: "continue in another region".into(),
                    text_elements: vec![],
                }]))
                .await?;
            let error =
                wait_for_event(&test.codex, |event| matches!(event, EventMsg::Error(_))).await;
            if matches!(
                case,
                CompactionRecoveryCase::MissingSource
                    | CompactionRecoveryCase::SourceExceedsCapacity
            ) {
                let EventMsg::Error(error) = error else {
                    unreachable!()
                };
                let expected = if matches!(case, CompactionRecoveryCase::SourceExceedsCapacity) {
                    "exceeds the active model's context budget"
                } else {
                    "source window is missing"
                };
                assert!(error.message.contains(expected), "{}", error.message);
            }
            wait_for_event(&test.codex, |event| {
                matches!(event, EventMsg::TurnComplete(_))
            })
            .await;
            let unchanged =
                mount_sse_once(&server, sse(vec![ev_completed("resp-unchanged")])).await;
            test.submit_turn("retry later").await?;
            assert_eq!(
                checkpoint_items(&unchanged.single_request().input()),
                vec![json!({
                    "type": "compaction", "encrypted_content": "old-checkpoint"
                })],
            );
            return Ok(());
        }

        let regeneration = mount_sse_once(&server, compaction_response(replacement)).await;
        if matches!(case, CompactionRecoveryCase::StillFails) {
            Mock::given(method("POST"))
                .and(path("/v1/responses"))
                .and(body_string_contains(replacement))
                .respond_with(error_response(/*status*/ 500, INTERNAL_ERROR))
                .expect(1)
                .with_priority(/*p*/ 1)
                .mount(&server)
                .await;
            test.codex
                .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
                    text: "try recovering once".into(),
                    text_elements: vec![],
                }]))
                .await?;
            wait_for_event(&test.codex, |event| matches!(event, EventMsg::Error(_))).await;
            wait_for_event(&test.codex, |event| {
                matches!(event, EventMsg::TurnComplete(_))
            })
            .await;
            let mut expected = original_input.clone();
            expected.retain(|item| item["type"] != "reasoning");
            assert_eq!(regeneration.single_request().input(), expected);
            server.verify().await;
            return Ok(());
        }
        let second_regeneration = if second_input.is_some() {
            Some(
                mount_sse_once(
                    &server,
                    compaction_response(&format!("{replacement}-second")),
                )
                .await,
            )
        } else {
            None
        };
        let response = if matches!(case, CompactionRecoveryCase::DuringCompaction) {
            compaction_response("manual-region")
        } else {
            sse(vec![ev_completed("resp-recovered")])
        };
        let resumed = mount_sse_once(&server, response).await;
        if matches!(case, CompactionRecoveryCase::DuringCompaction) {
            test.codex.submit(Op::Compact).await?;
            wait_for_event(&test.codex, |event| {
                matches!(event, EventMsg::TurnComplete(_))
            })
            .await;
        } else {
            test.codex
                .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
                    text: "continue in another region".into(),
                    text_elements: vec![],
                }]))
                .await?;
            let event = wait_for_event(&test.codex, |event| {
                matches!(event, EventMsg::Error(_) | EventMsg::TurnComplete(_))
            })
            .await;
            if let EventMsg::Error(error) = event {
                panic!("region recovery failed: {}", error.message);
            }
        }
        // The regeneration request contains the original tool calls/results and assistant
        // decision, with only region-bound reasoning removed. No tools are re-executed.
        let mut expected = original_input.clone();
        expected.retain(|item| item["type"] != "reasoning");
        assert_eq!(regeneration.single_request().input(), expected);
        let installed = if let (Some(second_input), Some(second_regeneration)) =
            (&second_input, &second_regeneration)
        {
            let mut expected = second_input.clone();
            for item in &mut expected {
                if item["type"] == "compaction" {
                    *item = json!({"type": "compaction", "encrypted_content": replacement});
                }
            }
            assert_eq!(second_regeneration.single_request().input(), expected);
            format!("{replacement}-second")
        } else {
            replacement.to_owned()
        };
        assert_eq!(
            checkpoint_items(&resumed.single_request().input()),
            vec![json!({"type": "compaction", "encrypted_content": installed})],
        );
        let cached = mount_sse_once(&server, sse(vec![ev_completed("resp-cached")])).await;
        test.submit_turn("continue in the same region").await?;
        if matches!(case, CompactionRecoveryCase::DuringCompaction) {
            assert_eq!(
                checkpoint_items(&cached.single_request().input()),
                vec![json!({"type": "compaction", "encrypted_content": "manual-region"})],
            );
            return Ok(());
        }
        assert_eq!(
            checkpoint_items(&cached.single_request().input()),
            checkpoint_items(&resumed.single_request().input()),
        );
    }
    Ok(())
}

fn error_response(status: u16, message: &str) -> ResponseTemplate {
    let (code, error_type) = if status == 500 {
        ("internal_server_error", "server_error")
    } else {
        ("validation_error", "invalid_request_error")
    };
    ResponseTemplate::new(status).set_body_json(json!({
        "error": {"code": code, "type": error_type, "message": message, "param": null}
    }))
}

#[tokio::test]
async fn bedrock_internal_error_without_encrypted_state_does_not_recover() -> anyhow::Result<()> {
    let server = MockServer::start().await;
    let test = bedrock_builder(&server)
        .build_with_auto_env(&server)
        .await?;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(error_response(/*status*/ 500, INTERNAL_ERROR))
        .expect(1)
        .mount(&server)
        .await;
    test.codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "first turn without any encrypted state".into(),
            text_elements: vec![],
        }]))
        .await?;
    wait_for_event(&test.codex, |event| matches!(event, EventMsg::Error(_))).await;
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    server.verify().await;
    Ok(())
}

fn compaction_response(content: &str) -> String {
    sse(vec![
        json!({
            "type": "response.output_item.done",
            "item": {"type": "compaction", "encrypted_content": content}
        }),
        ev_completed("resp-compact"),
    ])
}

fn checkpoint_items(input: &[serde_json::Value]) -> Vec<serde_json::Value> {
    input
        .iter()
        .filter(|item| item["type"] == "compaction")
        .map(|item| json!({"type": "compaction", "encrypted_content": item["encrypted_content"]}))
        .collect()
}
