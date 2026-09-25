// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

pub mod annotator;
pub mod emitter;
pub mod error;
pub mod resources;
pub mod rules;
pub mod tagger;
pub mod tokenizer;

pub use error::FrParseError;
pub use resources::LexicalResources;

use gcn_ir::CausalIR;
use gcn_knowledge::KnowledgeError;
use std::path::Path;

#[derive(Debug)]
pub enum ParserInitError {
    Knowledge(KnowledgeError),
}

impl std::fmt::Display for ParserInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParserInitError::Knowledge(e) => write!(f, "taxonomy load error: {}", e),
        }
    }
}

impl std::error::Error for ParserInitError {}

impl From<KnowledgeError> for ParserInitError {
    fn from(e: KnowledgeError) -> Self {
        ParserInitError::Knowledge(e)
    }
}

pub struct FrenchParser {
    resources: LexicalResources,
}

impl FrenchParser {
    /// Bootstrap annotation tool: auto-annotates French text → CausalIR to help build gcn-datasets/.
    /// Not the inference pipeline — replaced by trained ML layers once the model is trained.
    /// Loads linguistic resources from `taxonomies_root/fr/` (gcn-references/taxonomies/).
    pub fn new(taxonomies_root: &Path) -> Result<Self, ParserInitError> {
        let resources = LexicalResources::load(&taxonomies_root.join("fr"))?;
        Ok(FrenchParser { resources })
    }

    pub fn parse(&self, text: &str) -> Result<CausalIR, FrParseError> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(FrParseError::EmptyInput);
        }

        let tokens = tokenizer::tokenize(trimmed);
        if tokens.is_empty() {
            return Err(FrParseError::NoParseable(trimmed.to_string()));
        }

        let tagged = tagger::tag(&tokens, &self.resources);
        let annotation = annotator::annotate(&tagged, &self.resources);

        if annotation.clauses.is_empty() {
            return Err(FrParseError::NoParseable(trimmed.to_string()));
        }

        Ok(emitter::emit(annotation, trimmed.to_string()))
    }
}
