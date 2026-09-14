pub mod annotator;
pub mod emitter;
pub mod error;
pub mod resources;
pub mod rules;
pub mod tagger;
pub mod tokenizer;

pub use error::FrParseError;
pub use resources::LexicalResources;

use std::path::Path;
use gcn_ir::CausalIR;
use gcn_knowledge::KnowledgeError;

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
    /// Create a parser by loading linguistic resources from `data_dir`
    /// (path to the `data/taxonomies/` directory).
    pub fn new(data_dir: &Path) -> Result<Self, ParserInitError> {
        let resources = LexicalResources::load(data_dir)?;
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
