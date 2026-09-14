use thiserror::Error;

#[derive(Debug, Error)]
pub enum EnParseError {
    #[error("empty input")]
    EmptyInput,
    #[error("no parseable causal structure in: {0}")]
    NoParseable(String),
}
