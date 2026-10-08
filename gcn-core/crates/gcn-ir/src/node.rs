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
    /// D5 — absorbe action + transition (v2)
    #[serde(alias = "action", alias = "transition")]
    Processus,
    /// D5 — était "etat" (v2)
    #[serde(rename = "etat_local", alias = "etat")]
    EtatLocal,
    /// D5 — était "etat_systemique" (v2)
    #[serde(rename = "etat_global", alias = "etat_systemique")]
    EtatGlobal,
    Entite,
    Condition,
    /// D5 — nouveau
    Concept,
    /// D5 — nouveau
    Evenement,
    Contrainte,
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
    /// Nom canonique snake_case (identique au serde) → `NodeType`.
    /// Source unique des conversions dans le moteur (D1).
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "processus" => Some(NodeType::Processus),
            "etat_local" => Some(NodeType::EtatLocal),
            "etat_global" => Some(NodeType::EtatGlobal),
            "entite" => Some(NodeType::Entite),
            "condition" => Some(NodeType::Condition),
            "concept" => Some(NodeType::Concept),
            "evenement" => Some(NodeType::Evenement),
            "contrainte" => Some(NodeType::Contrainte),
            _ => None,
        }
    }

    pub fn causal_direction(&self) -> CausalDirection {
        match self {
            NodeType::Processus => CausalDirection::Both,
            NodeType::EtatLocal => CausalDirection::Backward,
            NodeType::EtatGlobal => CausalDirection::Accumulative,
            NodeType::Entite => CausalDirection::None,
            NodeType::Condition => CausalDirection::Suspended,
            NodeType::Concept => CausalDirection::None,
            NodeType::Evenement => CausalDirection::Forward,
            NodeType::Contrainte => CausalDirection::Suspended,
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

/// Span source : snake_case canonique (`token_span`, `synthetic`) ;
/// aliases PascalCase = sorties Rust antérieures (tolérance D-S03, socle 0b).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceSpan {
    #[serde(alias = "TokenSpan")]
    TokenSpan { start: u32, end: u32 },
    #[serde(alias = "CodeSpan")]
    CodeSpan {
        start_line: u32,
        start_col: u32,
        end_line: u32,
        end_col: u32,
    },
    #[serde(alias = "Synthetic")]
    Synthetic,
}

/// Attributs nœud : tous optionnels en lecture (tolérance D-S03, socle 0b).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeAttributes {
    #[serde(default)]
    pub entity: Option<String>,
    #[serde(default)]
    pub quality: Option<String>,
    #[serde(default)]
    pub agent: Option<String>,
    #[serde(default)]
    pub patient: Option<String>,
    #[serde(default)]
    pub agent_type: Option<AgentType>,
    #[serde(default)]
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
    /// Optionnel en lecture (tolérance D-S03, socle 0b).
    #[serde(default)]
    pub temporal_index: Option<i32>,
    pub origin: NodeOrigin,
    pub attributes: NodeAttributes,
    /// Nœud parent dans la hiérarchie multi-échelle (CIR v2).
    /// None = nœud de premier niveau. Optionnel en lecture pour compat CIR v1.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub parent: Option<NodeId>,
    /// Symbole grammatical source (tree-sitter `kind`, ex. "if_statement").
    /// Peuplé par gcn-frontend-code uniquement ; None ailleurs et en lecture
    /// des CIR antérieurs. Vocabulaire fini et stable (cf. kinds.rs).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub kind: Option<String>,
}
