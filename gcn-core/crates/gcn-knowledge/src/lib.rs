// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! GCN Knowledge: support aux outils de bootstrap d'annotation (gcn-frontend-fr, gcn-frontend-code).
//! Charge les taxonomies de gcn-references/ pour les annotateurs — pas utilisé par le pipeline ML d'inférence.

pub mod inference;
pub mod lexicon;
pub mod loader;
pub mod taxonomy;

use std::path::PathBuf;

pub use inference::{InferenceEngine, InferenceNote};
pub use lexicon::{AliasTable, Lexicon};
pub use loader::{load_all_taxonomies, load_taxonomy};
pub use taxonomy::{CompositionalRule, LexicalEntry, SubType, Taxonomy, TaxonomyClass};

#[derive(Debug, thiserror::Error)]
pub enum KnowledgeError {
    #[error("I/O error reading {1}: {0}")]
    Io(std::io::Error, PathBuf),
    #[error("YAML parse error in {1}: {0}")]
    Yaml(yaml_serde::Error, PathBuf),
}
