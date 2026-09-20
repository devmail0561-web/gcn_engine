// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use gcn_ir::{NodeType, RelationType};

// ---------------------------------------------------------------------------
// Type mappings: YAML string values → Rust enum variants
// These are the only things that may be hardcoded (per architecture rules).
// All language-specific data comes from YAML via CodeResources.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LabelStrategy {
    /// Truncate full source text to 64 chars
    #[default]
    FullText,
    /// Use the tree-sitter "condition" named field (if/while constructs)
    ConditionField,
    /// Use the tree-sitter "name" named field (function/class definitions)
    NameField,
}

pub fn label_strategy_str_to_enum(s: &str) -> Option<LabelStrategy> {
    match s {
        "full_text"       => Some(LabelStrategy::FullText),
        "condition_field" => Some(LabelStrategy::ConditionField),
        "name_field"      => Some(LabelStrategy::NameField),
        _ => None,
    }
}

pub fn node_type_str_to_enum(s: &str) -> Option<NodeType> {
    match s {
        "etat"            => Some(NodeType::Etat),
        "action"          => Some(NodeType::Action),
        "transition"      => Some(NodeType::Transition),
        "processus"       => Some(NodeType::Processus),
        "condition"       => Some(NodeType::Condition),
        "entite"          => Some(NodeType::Entite),
        "etat_systemique" => Some(NodeType::EtatSystemique),
        _ => None,
    }
}

pub fn relation_type_str_to_enum(s: &str) -> Option<RelationType> {
    match s {
        "cause"              => Some(RelationType::Cause),
        "enable"             => Some(RelationType::Enable),
        "prevent"            => Some(RelationType::Prevent),
        "condition"          => Some(RelationType::Condition),
        "concession"         => Some(RelationType::Concession),
        "sequence"           => Some(RelationType::Sequence),
        "motivation"         => Some(RelationType::Motivation),
        "filter"             => Some(RelationType::Filter),
        "opposition"         => Some(RelationType::Opposition),
        "data_dependency"    => Some(RelationType::DataDependency),
        "control_dependency" => Some(RelationType::ControlDependency),
        _ => None,
    }
}
