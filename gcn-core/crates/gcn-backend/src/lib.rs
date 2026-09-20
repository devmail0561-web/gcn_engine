// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

// Phase 4: Pearl reasoning (level 1), GCN-QL query engine, export

pub mod error;
pub mod export;
pub mod pearl;
pub mod query;

pub use error::BackendError;
pub use export::{to_dot, to_json};
pub use query::{execute, Query, QueryResult};
