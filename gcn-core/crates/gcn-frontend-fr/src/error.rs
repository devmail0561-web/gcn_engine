use thiserror::Error;

#[derive(Debug, Error)]
pub enum FrParseError {
    #[error("empty input")]
    EmptyInput,
    #[error("no parseable clause found in: {0}")]
    NoParseable(String),
}
