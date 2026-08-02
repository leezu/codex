use codex_login::CodexAuth;
use codex_login::auth::BedrockApiKeyAuth;
use codex_model_provider_info::AMAZON_BEDROCK_GPT_5_6_SOL_MODEL_ID;
use codex_model_provider_info::AMAZON_BEDROCK_PROVIDER_ID;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::error::CodexErr;
use codex_protocol::protocol::CodexErrorInfo;
use codex_protocol::protocol::ErrorEvent;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::protocol::TokenUsageInfo;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::mount_response_sequence;
use core_test_support::responses::mount_sse_once;
use core_test_support::responses::sse;
use core_test_support::responses::sse_failed;
use core_test_support::responses::sse_response;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::TestCodex;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use wiremock::MockServer;
use wiremock::ResponseTemplate;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn amazon_bedrock_internal_server_error_retries_the_turn() {
    skip_if_no_network!();

    let server = MockServer::start().await;
    let responses = mount_response_sequence(
        &server,
        vec![
            ResponseTemplate::new(400).set_body_string("Internal server error"),
            sse_response(sse(vec![
                ev_response_created("resp-ok"),
                ev_assistant_message("msg-ok", "done"),
                ev_completed("resp-ok"),
            ])),
        ],
    )
    .await;

    let mut provider = built_in_model_providers(/*openai_base_url*/ None)
        .remove(AMAZON_BEDROCK_PROVIDER_ID)
        .expect("Amazon Bedrock provider should be built in");
    provider.base_url = Some(format!("{}/v1", server.uri()));
    provider.aws = None;
    provider.request_max_retries = Some(0);
    provider.stream_max_retries = Some(1);

    let TestCodex { codex, .. } = test_codex()
        .with_auth(CodexAuth::BedrockApiKey(BedrockApiKeyAuth {
            api_key: "test-bedrock-api-key".to_string(),
            region: "us-east-2".to_string(),
        }))
        .with_config(move |config| {
            config.model_provider_id = AMAZON_BEDROCK_PROVIDER_ID.to_string();
            config.model_provider = provider;
            config.model = Some(AMAZON_BEDROCK_GPT_5_6_SOL_MODEL_ID.to_string());
        })
        .build(&server)
        .await
        .expect("test Codex should build");

    codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "retry a Bedrock server error".to_string(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await
        .expect("user input should submit");

    let EventMsg::TurnComplete(completed) =
        wait_for_event(&codex, |event| matches!(event, EventMsg::TurnComplete(_))).await
    else {
        unreachable!("predicate guarantees a turn complete event");
    };

    assert_eq!(completed.error, None);
    assert_eq!(responses.requests().len(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn amazon_bedrock_prompt_token_limit_marks_the_context_window_full() {
    skip_if_no_network!();

    const EFFECTIVE_CONTEXT_WINDOW: i64 = (272_000 * 95) / 100;
    const PROMPT_TOKEN_LIMIT_ERROR: &str =
        "prompt tokens (282061) exceed model maximum (278528) for openai.gpt-5.6-sol";

    let server = MockServer::start().await;
    let response = mount_sse_once(
        &server,
        sse_failed(
            "resp-context-window",
            "invalid_prompt",
            PROMPT_TOKEN_LIMIT_ERROR,
        ),
    )
    .await;

    let mut provider = built_in_model_providers(/*openai_base_url*/ None)
        .remove(AMAZON_BEDROCK_PROVIDER_ID)
        .expect("Amazon Bedrock provider should be built in");
    provider.base_url = Some(format!("{}/v1", server.uri()));
    provider.aws = None;
    provider.request_max_retries = Some(0);
    provider.stream_max_retries = Some(0);

    let TestCodex { codex, .. } = test_codex()
        .with_auth(CodexAuth::BedrockApiKey(BedrockApiKeyAuth {
            api_key: "test-bedrock-api-key".to_string(),
            region: "us-east-2".to_string(),
        }))
        .with_config(move |config| {
            config.model_provider_id = AMAZON_BEDROCK_PROVIDER_ID.to_string();
            config.model_provider = provider;
            config.model = Some(AMAZON_BEDROCK_GPT_5_6_SOL_MODEL_ID.to_string());
        })
        .build(&server)
        .await
        .expect("test Codex should build");

    codex
        .submit(Op::UserInput {
            items: vec![UserInput::Text {
                text: "overflow the Bedrock context window".to_string(),
                text_elements: Vec::new(),
            }],
            final_output_json_schema: None,
            responsesapi_client_metadata: None,
            additional_context: Default::default(),
            thread_settings: Default::default(),
        })
        .await
        .expect("user input should submit");

    let EventMsg::TokenCount(token_count) = wait_for_event(&codex, |event| {
        matches!(
            event,
            EventMsg::TokenCount(token_count)
                if token_count.info.as_ref().is_some_and(|info| {
                    info.model_context_window == Some(EFFECTIVE_CONTEXT_WINDOW)
                        && info.total_token_usage.total_tokens == EFFECTIVE_CONTEXT_WINDOW
                })
        )
    })
    .await
    else {
        unreachable!("predicate guarantees a token count event");
    };
    let token_info = token_count
        .info
        .expect("context window error should record token usage");
    assert_eq!(
        token_info,
        TokenUsageInfo::full_context_window(EFFECTIVE_CONTEXT_WINDOW)
    );

    let EventMsg::TurnComplete(completed) =
        wait_for_event(&codex, |event| matches!(event, EventMsg::TurnComplete(_))).await
    else {
        unreachable!("predicate guarantees a turn complete event");
    };
    assert_eq!(
        completed.error,
        Some(ErrorEvent {
            message: CodexErr::ContextWindowExceeded.to_string(),
            codex_error_info: Some(CodexErrorInfo::ContextWindowExceeded),
        })
    );
    assert_eq!(response.requests().len(), 1);
}
