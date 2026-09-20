// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use thiserror::Error;

#[derive(Debug, Error)]
pub enum FrParseError {
    #[error("empty input")]
    EmptyInput,
    #[error("no parseable clause found in: {0}")]
    NoParseable(String),
}
