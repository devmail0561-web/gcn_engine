// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodeParserError {
    #[error("tree-sitter language error: {0}")]
    Language(String),
    #[error("parse failed: source could not be parsed")]
    ParseFailed,
}
