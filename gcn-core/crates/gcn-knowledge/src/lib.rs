//! GCN Knowledge: support aux outils de bootstrap d'annotation (gcn-frontend-fr, gcn-frontend-code).
//! Charge les taxonomies de gcn-references/ pour les annotateurs — pas utilisé par le pipeline ML d'inférence.

pub mod taxonomy;
pub mod loader;
pub mod lexicon;
pub mod inference;

use std::path::PathBuf;

pub use taxonomy::{Taxonomy, TaxonomyClass, LexicalEntry, SubType, CompositionalRule};
pub use lexicon::Lexicon;
pub use inference::InferenceEngine;
pub use loader::{load_taxonomy, load_all_taxonomies};

#[derive(Debug, thiserror::Error)]
pub enum KnowledgeError {
    #[error("I/O error reading {1}: {0}")]
    Io(std::io::Error, PathBuf),
    #[error("YAML parse error in {1}: {0}")]
    Yaml(yaml_serde::Error, PathBuf),
}
