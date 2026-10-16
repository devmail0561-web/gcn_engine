// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TableParserError {
    #[error("empty input (no header row)")]
    EmptyInput,
    #[error("missing required column: {0}")]
    MissingColumn(String),
    #[error("duplicate column header: {0} (rename to disambiguate)")]
    DuplicateColumn(String),
    #[error("row {0}: expected {1} fields, found {2}")]
    FieldCount(usize, usize, usize),
    #[error("CSV quote error at row {0}: unterminated quoted field")]
    UnterminatedQuote(usize),
}
