// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use gcn_ir::{
    CausalEdge, CausalNode, NodeAttributes, NodeId, NodeOrigin, NodeType, Provenance, RelationType,
    Scope, SourceSpan, TemporalRef,
};
use smallvec::SmallVec;

use crate::kinds::LabelStrategy;

/// Compteurs de ce qui n'a PAS produit de nœud/arête — transparence anti-silence
/// (même patron que Graph/TableParseReport). P0-3 : les kinds AST non mappés
/// (`continue`) et les troncatures de labels sont désormais visibles.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CodeParseReport {
    /// Nœuds AST nommés ignorés car kind absent de la table de mapping.
    pub skipped_unmapped: usize,
    /// Labels tronqués à `MAX_LABEL_CHARS`.
    pub truncated_labels: usize,
}

impl CodeParseReport {
    pub fn total_skipped(&self) -> usize {
        self.skipped_unmapped + self.truncated_labels
    }
}

/// Longueur max d'un label de nœud (P3 : 64 → 256, troncatures comptées).
pub const MAX_LABEL_CHARS: usize = 256;

pub fn emit_node(
    node: tree_sitter::Node<'_>,
    src: &[u8],
    node_type: NodeType,
    nodes: &mut Vec<CausalNode>,
    next_id: &mut u32,
    strategy: LabelStrategy,
    report: &mut CodeParseReport,
) -> NodeId {
    let id = NodeId(*next_id);
    *next_id += 1;
    let start = node.start_position();
    let end = node.end_position();
    let (label, truncated) = node_label(node, src, strategy);
    if truncated {
        report.truncated_labels += 1;
    }
    nodes.push(CausalNode {
        id,
        node_type,
        label,
        source_span: SourceSpan::CodeSpan {
            start_line: start.row as u32,
            start_col: start.column as u32,
            end_line: end.row as u32,
            end_col: end.column as u32,
        },
        scope: Scope::Unknown,
        modifiers: SmallVec::new(),
        temporal_ref: TemporalRef::Unresolved,
        temporal_index: Some(*next_id as i32 - 1),
        origin: NodeOrigin::Explicit,
        // P3 : seul attribut peuplé — le nom défini (fonction/méthode/classe),
        // langage-agnostique via le champ AST `name`. Le reste (data flow,
        // side effects, pureté : CodeCausalType/DataFlow/ControlFlow) reste
        // hors périmètre tant que le mapping YAML ne les porte pas.
        attributes: NodeAttributes {
            agent: defined_name(node, src),
            ..Default::default()
        },
        parent: None,
    });
    id
}

/// Nom défini par le nœud (fonction, méthode, classe…) via le champ AST `name`.
/// `None` si absent — aucun fallback, pas de devinette.
fn defined_name(node: tree_sitter::Node<'_>, src: &[u8]) -> Option<String> {
    node.child_by_field_name("name")
        .and_then(|c| c.utf8_text(src).ok())
        .map(|s| s.trim().chars().take(MAX_LABEL_CHARS).collect())
        .filter(|s: &String| !s.is_empty())
}

/// Retourne `(label, truncated)`. `truncated` = le texte source dépassait
/// `MAX_LABEL_CHARS` (P3 : compté dans le report au lieu d'être silencieux).
pub fn node_label(
    node: tree_sitter::Node<'_>,
    src: &[u8],
    strategy: LabelStrategy,
) -> (String, bool) {
    match strategy {
        LabelStrategy::ConditionField => node
            .child_by_field_name("condition")
            .and_then(|c| c.utf8_text(src).ok())
            .map(|s| truncate(s.trim()))
            .unwrap_or_else(|| full_text_label(node, src)),
        LabelStrategy::NameField => node
            .child_by_field_name("name")
            .and_then(|c| c.utf8_text(src).ok())
            .map(|s| truncate(s.trim()))
            .unwrap_or_else(|| full_text_label(node, src)),
        LabelStrategy::FullText => full_text_label(node, src),
    }
}

fn truncate(s: &str) -> (String, bool) {
    let truncated = s.chars().count() > MAX_LABEL_CHARS;
    (s.chars().take(MAX_LABEL_CHARS).collect(), truncated)
}

pub fn full_text_label(node: tree_sitter::Node<'_>, src: &[u8]) -> (String, bool) {
    match node.utf8_text(src) {
        Ok(t) => truncate(t.trim()),
        Err(_) => ("?".to_string(), false),
    }
}

pub fn control_edge(relation: RelationType) -> CausalEdge {
    control_edge_with_ref(relation, None, SourceSpan::Synthetic)
}

pub fn control_edge_with_ref(
    relation: RelationType,
    doc_ref: Option<String>,
    span: SourceSpan,
) -> CausalEdge {
    CausalEdge {
        relation,
        confidence: 1.0,
        temporal_gap: None,
        explicit: true,
        negated: false,
        marker_token: None,
        in_cycle: None,
        provenance: Some(Provenance::with_ref(doc_ref, span)),
        derivation: None,
        joint_group_id: None,
        third: None,
    }
}
