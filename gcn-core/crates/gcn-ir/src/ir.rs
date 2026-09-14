use serde::{Deserialize, Serialize};

use crate::edge::{CausalEdge, CycleId};
use crate::node::{CausalNode, NodeId};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalIR {
    pub source_lang: SourceLanguage,
    pub source_text: String,
    pub nodes: Vec<CausalNode>,
    pub edges: Vec<(NodeId, NodeId, CausalEdge)>,
    pub cycles: Vec<CausalCycle>,
    pub unresolved: Vec<Ambiguity>,
    pub metadata: IrMetadata,
}

impl CausalIR {
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn has_cycles(&self) -> bool {
        !self.cycles.is_empty()
    }

    pub fn has_gaps(&self) -> bool {
        self.edges.iter().any(|(_, _, e)| e.temporal_gap.is_some())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceLanguage {
    Natural { lang: NaturalLanguage },
    Programming { lang: ProgrammingLanguage },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NaturalLanguage {
    French,
    Wolof,
    Arabic,
    English,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgrammingLanguage {
    Python,
    Rust,
    JavaScript,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalCycle {
    pub id: CycleId,
    pub path: Vec<NodeId>,
    pub cycle_type: CycleType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CycleType {
    FeedbackPositive,
    FeedbackNegative,
    Oscillation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ambiguity {
    pub node_id: NodeId,
    pub field: AmbiguousField,
    pub candidates: Vec<AmbiguityCandidate>,
    pub context_hint: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AmbiguousField {
    NodeType,
    Scope,
    AgentType,
    CorefTarget,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbiguityCandidate {
    pub value: String,
    pub confidence: f32,
    pub rule_source: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IrMetadata {
    pub schema_version: String,
    pub pipeline: Vec<String>,
    pub created_at: Option<String>,
}
