// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use gcn_ir::{
    CausalEdge, CausalNode, NodeId, NodeOrigin, NodeType, RelationType, Scope, SourceSpan,
    TemporalRef,
};
use smallvec::SmallVec;

use crate::mapper::LabelStrategy;
use crate::resources::CodeResources;

pub fn emit_node(
    node: tree_sitter::Node<'_>,
    src: &[u8],
    node_type: NodeType,
    nodes: &mut Vec<CausalNode>,
    next_id: &mut u32,
    res: &CodeResources,
) -> NodeId {
    let id = NodeId(*next_id);
    *next_id += 1;
    let start = node.start_position();
    let end = node.end_position();
    nodes.push(CausalNode {
        id,
        node_type,
        label: node_label(node, src, res),
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
        attributes: Default::default(),
    });
    id
}

pub fn node_label(node: tree_sitter::Node<'_>, src: &[u8], res: &CodeResources) -> String {
    let strategy = res.kind_to_label_strategy.get(node.kind()).copied().unwrap_or_default();
    match strategy {
        LabelStrategy::ConditionField => node
            .child_by_field_name("condition")
            .and_then(|c| c.utf8_text(src).ok())
            .map(|s| s.trim().chars().take(64).collect())
            .unwrap_or_else(|| full_text_label(node, src)),
        LabelStrategy::NameField => node
            .child_by_field_name("name")
            .and_then(|c| c.utf8_text(src).ok())
            .map(|s| s.trim().chars().take(64).collect())
            .unwrap_or_else(|| full_text_label(node, src)),
        LabelStrategy::FullText => full_text_label(node, src),
    }
}

pub fn full_text_label(node: tree_sitter::Node<'_>, src: &[u8]) -> String {
    node.utf8_text(src).unwrap_or("?").trim().chars().take(64).collect()
}

pub fn control_edge(relation: RelationType) -> CausalEdge {
    CausalEdge {
        relation,
        confidence: 1.0,
        temporal_gap: None,
        explicit: true,
        negated: false,
        marker_token: None,
        in_cycle: None,
    }
}
