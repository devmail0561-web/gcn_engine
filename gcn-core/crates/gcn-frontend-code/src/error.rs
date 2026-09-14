use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodeParserError {
    #[error("tree-sitter language error: {0}")]
    Language(String),
    #[error("parse failed: source could not be parsed")]
    ParseFailed,
}
