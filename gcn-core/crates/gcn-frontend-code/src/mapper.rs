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
        "full_text" => Some(LabelStrategy::FullText),
        "condition_field" => Some(LabelStrategy::ConditionField),
        "name_field" => Some(LabelStrategy::NameField),
        _ => None,
    }
}

pub fn node_type_str_to_enum(s: &str) -> Option<NodeType> {
    match s {
        "etat" => Some(NodeType::EtatLocal),
        "action" => Some(NodeType::Processus),
        "transition" => Some(NodeType::Processus),
        "processus" => Some(NodeType::Processus),
        "condition" => Some(NodeType::Condition),
        "entite" => Some(NodeType::Entite),
        "etat_systemique" => Some(NodeType::EtatGlobal),
        "contrainte" => Some(NodeType::Contrainte),
        _ => None,
    }
}

/// Délègue aux noms canoniques `gcn_ir` (source unique, B3 audit).
/// Accepte en plus les 8 relations v3.0 (jamais présentes dans les YAML AST
/// actuels — sans effet sur les données existantes, future-proof).
/// `node_type_str_to_enum` ci-dessus est VOLONTAIREMENT inchangé : les YAML
/// utilisent les aliases v2 (`action`, `transition`, `etat`) que les noms
/// canoniques rejettent à raison.
pub fn relation_type_str_to_enum(s: &str) -> Option<RelationType> {
    RelationType::from_name(s)
}
