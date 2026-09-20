// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("node not found matching '{0}'")]
    NodeNotFound(String),
    #[error("query parse error: {0}")]
    QueryParseError(String),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    /// Réservé à l'API stricte `chain_strict` — `execute(CHAIN)` retourne
    /// `Path { found: false }` au lieu d'erreur pour compatibilité.
    #[error("no path from '{0}' to '{1}'")]
    NoPath(String, String),
}
