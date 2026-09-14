use thiserror::Error;

#[derive(Debug, Error)]
pub enum GcnError {
    #[error("parse error: {0}")]
    Parse(String),
    #[error("node not found: {0}")]
    NodeNotFound(u32),
    #[error("invalid taxonomy: {0}")]
    InvalidTaxonomy(String),
    #[error("unsupported language: {0}")]
    UnsupportedLanguage(String),
    #[error("serialization: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("causal constraint violation: {0}")]
    CausalConstraintViolation(String),
    #[error("invalid cycle: {0}")]
    InvalidCycle(String),
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
}

pub type GcnResult<T> = Result<T, GcnError>;
