// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::modifier::Modifier;
use crate::scope::Scope;
use crate::temporal::TemporalRef;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeType {
    Etat,
    Action,
    Transition,
    Processus,
    Condition,
    Entite,
    EtatSystemique,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalDirection {
    Forward,
    Backward,
    Both,
    Accumulative,
    Suspended,
    None,
}

impl NodeType {
    pub fn causal_direction(&self) -> CausalDirection {
        match self {
            NodeType::Etat => CausalDirection::Backward,
            NodeType::Action => CausalDirection::Forward,
            NodeType::Transition => CausalDirection::Forward,
            NodeType::Processus => CausalDirection::Both,
            NodeType::Condition => CausalDirection::Suspended,
            NodeType::Entite => CausalDirection::None,
            NodeType::EtatSystemique => CausalDirection::Accumulative,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentType {
    Human,
    Collective,
    Institutional,
    Natural,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NodeOrigin {
    #[default]
    Explicit,
    Inferred,
    Hypothetical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceSpan {
    TokenSpan {
        start: u32,
        end: u32,
    },
    CodeSpan {
        start_line: u32,
        start_col: u32,
        end_line: u32,
        end_col: u32,
    },
    Synthetic,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeAttributes {
    pub entity: Option<String>,
    pub quality: Option<String>,
    pub agent: Option<String>,
    pub patient: Option<String>,
    pub agent_type: Option<AgentType>,
    pub reversible: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalNode {
    pub id: NodeId,
    pub node_type: NodeType,
    pub label: String,
    pub source_span: SourceSpan,
    pub scope: Scope,
    pub modifiers: SmallVec<[Modifier; 4]>,
    pub temporal_ref: TemporalRef,
    pub temporal_index: Option<i32>,
    pub origin: NodeOrigin,
    pub attributes: NodeAttributes,
    /// Nœud parent dans la hiérarchie multi-échelle (CIR v2).
    /// None = nœud de premier niveau. Optionnel en lecture pour compat CIR v1.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub parent: Option<NodeId>,
}
