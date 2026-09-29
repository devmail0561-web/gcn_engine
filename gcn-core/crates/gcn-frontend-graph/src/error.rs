// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use thiserror::Error;

#[derive(Debug, Error)]
pub enum GraphParserError {
    #[error("invalid JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),
    #[error("not a STIX bundle (expected {{\"type\":\"bundle\"}}, found type={0:?})")]
    NotABundle(Option<String>),
    #[error("bundle has no \"objects\" array")]
    MissingObjects,
}
