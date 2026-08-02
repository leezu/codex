use codex_api::ApiError;
use codex_protocol::error::CodexErr;
use codex_protocol::error::CodexErrorDetails;
use http::StatusCode;

pub(super) const BEDROCK_EXPIRED_SIGNATURE_MESSAGE: &str = concat!(
    "Amazon Bedrock rejected the request because its AWS signature has expired. ",
    "Refresh your AWS credentials and retry. If `AWS_BEARER_TOKEN_BEDROCK` is set, ",
    "update or unset it, then restart Codex",
);

pub(super) fn map_api_error(error: ApiError) -> CodexErr {
    let error = codex_api::map_api_error(error);
    if let CodexErrorDetails::InvalidRequest(message) = error.details() {
        let message = message.trim();
        let is_context_window_exceeded = message
            .strip_prefix("prompt tokens (")
            .and_then(|message| {
                [
                    ") exceed model maximum (",
                    ") exceed customer model maximum (",
                ]
                .into_iter()
                .find_map(|separator| message.split_once(separator))
            })
            .and_then(|(prompt_tokens, message)| {
                let (model_maximum, model) = message.split_once(") for ")?;
                Some((prompt_tokens, model_maximum, model))
            })
            .is_some_and(|(prompt_tokens, model_maximum, model)| {
                prompt_tokens.parse::<u64>().is_ok()
                    && model_maximum.parse::<u64>().is_ok()
                    && !model.is_empty()
            });
        if is_context_window_exceeded {
            return CodexErr::ContextWindowExceeded;
        }
        if message.eq_ignore_ascii_case("Internal server error") {
            return CodexErr::InternalServerError;
        }
    }
    if let CodexErrorDetails::UnexpectedStatus(response) = error.details()
        && response.status == StatusCode::UNAUTHORIZED
        && response.body.contains("Signature expired:")
    {
        let mut response = response.clone();
        response.user_message = Some(BEDROCK_EXPIRED_SIGNATURE_MESSAGE.to_string());
        let mapped_error = CodexErr::new(CodexErrorDetails::UnexpectedStatus(response));
        return match error.retry_delay() {
            Some(retry_delay) => mapped_error.with_retry_delay(retry_delay),
            None => mapped_error,
        };
    }
    error
}
