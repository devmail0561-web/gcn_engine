#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("node not found matching '{0}'")]
    NodeNotFound(String),
    #[error("query parse error: {0}")]
    QueryParseError(String),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("no path from '{0}' to '{1}'")]
    NoPath(String, String),
}
