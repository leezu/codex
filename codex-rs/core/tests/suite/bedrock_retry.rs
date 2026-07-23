use codex_login::CodexAuth;
use codex_login::auth::BedrockApiKeyAuth;
use codex_model_provider_info::AMAZON_BEDROCK_GPT_5_6_SOL_MODEL_ID;
use codex_model_provider_info::AMAZON_BEDROCK_PROVIDER_ID;
use codex_model_provider_info::built_in_model_providers;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::Op;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::mount_response_sequence;
use core_test_support::responses::sse;
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
