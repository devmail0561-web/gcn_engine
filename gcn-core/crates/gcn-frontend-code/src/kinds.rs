// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! Mapping grammaire tree-sitter → types causaux, en dur.
//!
//! NOTE DE DOCTRINE — ceci n'est PAS du word matching linguistique :
//! les `kind` ("if_statement", "call_expression"...) sont des symboles
//! formels de la grammaire tree-sitter (API du parseur, ensemble fini et
//! stable — comme matcher des variantes d'AST dans n'importe quel
//! compilateur). Aucun mot de langue naturelle n'est comparé ici.
//! Remplace les anciens `*_ast.yaml` (supprimés : zéro chargement YAML).

use gcn_ir::{NodeType, RelationType};

/// Stratégie d'extraction du label depuis l'AST (champs tree-sitter).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LabelStrategy {
    /// Texte source tronqué à 256 chars
    #[default]
    FullText,
    /// Champ AST "condition" (if/while)
    ConditionField,
    /// Champ AST "name" (définitions)
    NameField,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    Python,
    Rust,
    Js,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KindInfo {
    pub node_type: NodeType,
    pub edge_type: RelationType,
    pub label: LabelStrategy,
}

/// Nœuds grammaticaux sans sémantique propre — on descend dans leur
/// premier enfant nommé.
pub fn is_transparent(lang: Lang, kind: &str) -> bool {
    match lang {
        Lang::Python => matches!(kind, "expression_statement" | "decorated_definition"),
        Lang::Rust => matches!(kind, "expression_statement"),
        Lang::Js => matches!(kind, "expression_statement" | "export_statement"),
    }
}

/// Symboles tree-sitter → (NodeType, RelationType, LabelStrategy).
/// Contenu repris des anciens `*_ast.yaml` (supprimés 2026-10-01, parité
/// 52/52 vérifiée par script) avec fusion D5 : `action`/`transition` →
/// `Processus`, `etat` → `EtatLocal`.
pub fn kind_info(lang: Lang, kind: &str) -> Option<KindInfo> {
    let (node_type, edge_type, label) = match lang {
        Lang::Python => match kind {
            "if_statement" => (
                NodeType::Condition,
                RelationType::Condition,
                LabelStrategy::ConditionField,
            ),
            "elif_clause" => (
                NodeType::Condition,
                RelationType::Condition,
                LabelStrategy::ConditionField,
            ),
            "assert_statement" => (
                NodeType::Condition,
                RelationType::Condition,
                LabelStrategy::FullText,
            ),
            "for_statement" => (
                NodeType::Processus,
                RelationType::Sequence,
                LabelStrategy::FullText,
            ),
            "while_statement" => (
                NodeType::Processus,
                RelationType::Sequence,
                LabelStrategy::FullText,
            ),
            "with_statement" => (
                NodeType::Processus,
                RelationType::Enable,
                LabelStrategy::FullText,
            ),
            "function_definition" => (
                NodeType::Processus,
                RelationType::Cause,
                LabelStrategy::NameField,
            ),
            "call" => (
                NodeType::Processus,
                RelationType::DataDependency,
                LabelStrategy::FullText,
            ),
            "assignment" => (
                NodeType::Processus,
                RelationType::DataDependency,
                LabelStrategy::FullText,
            ),
            "augmented_assignment" => (
                NodeType::Processus,
                RelationType::DataDependency,
                LabelStrategy::FullText,
            ),
            "named_expression" => (
                NodeType::Processus,
                RelationType::DataDependency,
                LabelStrategy::FullText,
            ),
            "return_statement" => (
                NodeType::Processus,
                RelationType::DataDependency,
                LabelStrategy::FullText,
            ),
            "raise_statement" => (
                NodeType::Processus,
                RelationType::Concession,
                LabelStrategy::FullText,
            ),
            "import_statement" => (
                NodeType::Entite,
                RelationType::DataDependency,
                LabelStrategy::FullText,
            ),
            "import_from_statement" => (
                NodeType::Entite,
                RelationType::DataDependency,
                LabelStrategy::FullText,
            ),
            "class_definition" => (
                NodeType::Entite,
                RelationType::Cause,
                LabelStrategy::FullText,
            ),
            "try_statement" => (
                NodeType::EtatLocal,
                RelationType::Concession,
                LabelStrategy::FullText,
            ),
            "except_clause" => (
                NodeType::EtatLocal,
                RelationType::Concession,
                LabelStrategy::FullText,
            ),
            _ => return None,
        },
        Lang::Rust => match kind {
            "if_expression" => (
                NodeType::Condition,
                RelationType::Condition,
                LabelStrategy::ConditionField,
            ),
            "while_expression" => (
                NodeType::Processus,
                RelationType::Sequence,
                LabelStrategy::ConditionField,
            ),
            "for_expression" => (
                NodeType::Processus,
                RelationType::Sequence,
                LabelStrategy::FullText,
            ),
            "loop_expression" => (
                NodeType::Processus,
                RelationType::Sequence,
                LabelStrategy::FullText,
            ),
            "match_expression" => (
                NodeType::Condition,
                RelationType::Condition,
                LabelStrategy::FullText,
            ),
            "function_item" => (
                NodeType::Processus,
                RelationType::Cause,
                LabelStrategy::NameField,
            ),
            "let_declaration" => (
                NodeType::Processus,
                RelationType::DataDependency,
                LabelStrategy::FullText,
            ),
            "assignment_expression" => (
                NodeType::Processus,
                RelationType::DataDependency,
                LabelStrategy::FullText,
            ),
            "compound_assignment_expr" => (
                NodeType::Processus,
                RelationType::DataDependency,
                LabelStrategy::FullText,
            ),
            "call_expression" => (
                NodeType::Processus,
                RelationType::DataDependency,
                LabelStrategy::FullText,
            ),
            "macro_invocation" => (
                NodeType::Processus,
                RelationType::DataDependency,
                LabelStrategy::FullText,
            ),
            "return_expression" => (
                NodeType::Processus,
                RelationType::DataDependency,
                LabelStrategy::FullText,
            ),
            "use_declaration" => (
                NodeType::Entite,
                RelationType::DataDependency,
                LabelStrategy::FullText,
            ),
            "struct_item" => (
                NodeType::Entite,
                RelationType::Cause,
                LabelStrategy::NameField,
            ),
            "enum_item" => (
                NodeType::Entite,
                RelationType::Cause,
                LabelStrategy::NameField,
            ),
            "impl_item" => (
                NodeType::Entite,
                RelationType::Cause,
                LabelStrategy::FullText,
            ),
            "trait_item" => (
                NodeType::Entite,
                RelationType::Cause,
                LabelStrategy::NameField,
            ),
            _ => return None,
        },
        Lang::Js => match kind {
            "if_statement" => (
                NodeType::Condition,
                RelationType::Condition,
                LabelStrategy::ConditionField,
            ),
            "while_statement" => (
                NodeType::Processus,
                RelationType::Sequence,
                LabelStrategy::ConditionField,
            ),
            "for_statement" => (
                NodeType::Processus,
                RelationType::Sequence,
                LabelStrategy::FullText,
            ),
            "for_in_statement" => (
                NodeType::Processus,
                RelationType::Sequence,
                LabelStrategy::FullText,
            ),
            "function_declaration" => (
                NodeType::Processus,
                RelationType::Cause,
                LabelStrategy::NameField,
            ),
            "function" => (
                NodeType::Processus,
                RelationType::Cause,
                LabelStrategy::NameField,
            ),
            "arrow_function" => (
                NodeType::Processus,
                RelationType::Cause,
                LabelStrategy::FullText,
            ),
            "variable_declaration" => (
                NodeType::Processus,
                RelationType::DataDependency,
                LabelStrategy::FullText,
            ),
            "lexical_declaration" => (
                NodeType::Processus,
                RelationType::DataDependency,
                LabelStrategy::FullText,
            ),
            "assignment_expression" => (
                NodeType::Processus,
                RelationType::DataDependency,
                LabelStrategy::FullText,
            ),
            "call_expression" => (
                NodeType::Processus,
                RelationType::DataDependency,
                LabelStrategy::FullText,
            ),
            "return_statement" => (
                NodeType::Processus,
                RelationType::DataDependency,
                LabelStrategy::FullText,
            ),
            "throw_statement" => (
                NodeType::Processus,
                RelationType::Concession,
                LabelStrategy::FullText,
            ),
            "try_statement" => (
                NodeType::EtatLocal,
                RelationType::Concession,
                LabelStrategy::FullText,
            ),
            "catch_clause" => (
                NodeType::EtatLocal,
                RelationType::Concession,
                LabelStrategy::FullText,
            ),
            "import_statement" => (
                NodeType::Entite,
                RelationType::DataDependency,
                LabelStrategy::FullText,
            ),
            "class_declaration" => (
                NodeType::Entite,
                RelationType::Cause,
                LabelStrategy::NameField,
            ),
            _ => return None,
        },
    };
    Some(KindInfo {
        node_type,
        edge_type,
        label,
    })
}
