// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EnParseError {
    #[error("empty input")]
    EmptyInput,
    #[error("no parseable causal structure in: {0}")]
    NoParseable(String),
}
