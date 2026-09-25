// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use gcn_ir::{
    CausalEdge, CausalIR, CausalNode, CycleType, IrMetadata, NaturalLanguage, NodeAttributes,
    NodeId, NodeOrigin, NodeType, RelationType, Scope, SourceLanguage, SourceSpan, TemporalRef,
};
use gcn_middleend::{DiagnosticKind, DiagnosticSeverity, process};
use smallvec::SmallVec;

fn node(id: u32, node_type: NodeType) -> CausalNode {
    CausalNode {
        id: NodeId(id),
        node_type,
        label: format!("node_{id}"),
        source_span: SourceSpan::Synthetic,
        scope: Scope::Unknown,
        modifiers: SmallVec::new(),
        temporal_ref: TemporalRef::Unresolved,
        temporal_index: None,
        origin: NodeOrigin::Explicit,
        attributes: NodeAttributes::default(),
    }
}

fn edge(src: u32, dst: u32, relation: RelationType) -> (NodeId, NodeId, CausalEdge) {
    (
        NodeId(src),
        NodeId(dst),
        CausalEdge {
            relation,
            confidence: 0.9,
            temporal_gap: None,
            explicit: true,
            negated: false,
            marker_token: None,
            in_cycle: None,
        },
    )
}

fn edge_neg(src: u32, dst: u32, relation: RelationType) -> (NodeId, NodeId, CausalEdge) {
    let (s, d, mut e) = edge(src, dst, relation);
    e.negated = true;
    (s, d, e)
}

fn edge_low_conf(src: u32, dst: u32) -> (NodeId, NodeId, CausalEdge) {
    let (s, d, mut e) = edge(src, dst, RelationType::Cause);
    e.confidence = 0.1;
    (s, d, e)
}

fn make_ir(nodes: Vec<CausalNode>, edges: Vec<(NodeId, NodeId, CausalEdge)>) -> CausalIR {
    CausalIR {
        source_lang: SourceLanguage::Natural {
            lang: NaturalLanguage::French,
        },
        source_text: String::new(),
        nodes,
        edges,
        cycles: vec![],
        unresolved: vec![],
        metadata: IrMetadata {
            schema_version: "1.0".to_string(),
            pipeline: vec!["gcn-frontend-fr".to_string()],
            created_at: None,
        },
    }
}

// ─── cycle detection ────────────────────────────────────────────────────────

#[test]
fn no_cycle_chain() {
    // A → B → C : no cycle
    let ir = make_ir(
        vec![
            node(0, NodeType::Action),
            node(1, NodeType::Etat),
            node(2, NodeType::Processus),
        ],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 2, RelationType::Cause),
        ],
    );
    let result = process(ir).unwrap();
    assert!(result.ir.cycles.is_empty());
    assert!(result.ir.edges.iter().all(|(_, _, e)| e.in_cycle.is_none()));
}

#[test]
fn positive_feedback_loop() {
    // A → B → A : FeedbackPositive
    let ir = make_ir(
        vec![node(0, NodeType::Action), node(1, NodeType::Etat)],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 0, RelationType::Cause),
        ],
    );
    let result = process(ir).unwrap();
    assert_eq!(result.ir.cycles.len(), 1);
    assert_eq!(result.ir.cycles[0].cycle_type, CycleType::FeedbackPositive);
    assert!(result.ir.edges.iter().all(|(_, _, e)| e.in_cycle.is_some()));
}

#[test]
fn negative_feedback_loop() {
    // A -Cause→ B -Prevent→ A : FeedbackNegative
    let ir = make_ir(
        vec![node(0, NodeType::Action), node(1, NodeType::Etat)],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 0, RelationType::Prevent),
        ],
    );
    let result = process(ir).unwrap();
    assert_eq!(result.ir.cycles.len(), 1);
    assert_eq!(result.ir.cycles[0].cycle_type, CycleType::FeedbackNegative);
}

#[test]
fn negated_edge_in_loop_is_negative_feedback() {
    // A -Cause→ B -negated Cause→ A : FeedbackNegative
    let ir = make_ir(
        vec![node(0, NodeType::Action), node(1, NodeType::Etat)],
        vec![
            edge(0, 1, RelationType::Cause),
            edge_neg(1, 0, RelationType::Cause),
        ],
    );
    let result = process(ir).unwrap();
    assert_eq!(result.ir.cycles[0].cycle_type, CycleType::FeedbackNegative);
}

#[test]
fn concession_loop_is_oscillation() {
    let ir = make_ir(
        vec![node(0, NodeType::Etat), node(1, NodeType::Etat)],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 0, RelationType::Concession),
        ],
    );
    let result = process(ir).unwrap();
    assert_eq!(result.ir.cycles[0].cycle_type, CycleType::Oscillation);
}

#[test]
fn cycle_path_contains_both_nodes() {
    let ir = make_ir(
        vec![node(0, NodeType::Action), node(1, NodeType::Etat)],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 0, RelationType::Cause),
        ],
    );
    let result = process(ir).unwrap();
    let path = &result.ir.cycles[0].path;
    assert_eq!(path.len(), 2);
    assert!(path.contains(&NodeId(0)));
    assert!(path.contains(&NodeId(1)));
}

// ─── propagation ────────────────────────────────────────────────────────────

#[test]
fn concession_edge_gets_temporal_gap() {
    let ir = make_ir(
        vec![node(0, NodeType::Etat), node(1, NodeType::Etat)],
        vec![edge(0, 1, RelationType::Concession)],
    );
    let result = process(ir).unwrap();
    assert!(result.ir.edges[0].2.temporal_gap.is_some());
}

#[test]
fn cause_edge_no_temporal_gap() {
    let ir = make_ir(
        vec![node(0, NodeType::Action), node(1, NodeType::Etat)],
        vec![edge(0, 1, RelationType::Cause)],
    );
    let result = process(ir).unwrap();
    assert!(result.ir.edges[0].2.temporal_gap.is_none());
}

#[test]
fn sequence_temporal_violation_warns() {
    // Node 0 has temporal_index=1, Node 1 has temporal_index=0 — but Sequence says 0→1
    let mut n0 = node(0, NodeType::Action);
    n0.temporal_index = Some(1);
    let mut n1 = node(1, NodeType::Etat);
    n1.temporal_index = Some(0);

    let ir = make_ir(vec![n0, n1], vec![edge(0, 1, RelationType::Sequence)]);
    let result = process(ir).unwrap();
    let violations: Vec<_> = result
        .diagnostics
        .iter()
        .filter(|d| matches!(d.kind, DiagnosticKind::TemporalOrderViolation { .. }))
        .collect();
    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].severity, DiagnosticSeverity::Warning);
}

#[test]
fn sequence_correct_order_no_warning() {
    let mut n0 = node(0, NodeType::Action);
    n0.temporal_index = Some(0);
    let mut n1 = node(1, NodeType::Etat);
    n1.temporal_index = Some(1);

    let ir = make_ir(vec![n0, n1], vec![edge(0, 1, RelationType::Sequence)]);
    let result = process(ir).unwrap();
    assert!(
        !result
            .diagnostics
            .iter()
            .any(|d| matches!(d.kind, DiagnosticKind::TemporalOrderViolation { .. }))
    );
}

// ─── validation ─────────────────────────────────────────────────────────────

#[test]
fn self_loop_is_error() {
    let ir = make_ir(
        vec![node(0, NodeType::Action)],
        vec![edge(0, 0, RelationType::Cause)],
    );
    let result = process(ir).unwrap();
    let errors: Vec<_> = result
        .diagnostics
        .iter()
        .filter(|d| matches!(d.kind, DiagnosticKind::SelfLoop { .. }))
        .collect();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].severity, DiagnosticSeverity::Error);
}

#[test]
fn orphaned_node_warns() {
    // 3 nodes, only 0→1 connected, node 2 is orphaned
    let ir = make_ir(
        vec![
            node(0, NodeType::Action),
            node(1, NodeType::Etat),
            node(2, NodeType::Entite),
        ],
        vec![edge(0, 1, RelationType::Cause)],
    );
    let result = process(ir).unwrap();
    let orphans: Vec<_> = result
        .diagnostics
        .iter()
        .filter(|d| matches!(d.kind, DiagnosticKind::OrphanedNode { .. }))
        .collect();
    assert_eq!(orphans.len(), 1);
    if let DiagnosticKind::OrphanedNode { node } = orphans[0].kind {
        assert_eq!(node, NodeId(2));
    }
}

#[test]
fn single_node_no_orphan_warning() {
    let ir = make_ir(vec![node(0, NodeType::Action)], vec![]);
    let result = process(ir).unwrap();
    assert!(
        !result
            .diagnostics
            .iter()
            .any(|d| matches!(d.kind, DiagnosticKind::OrphanedNode { .. }))
    );
}

#[test]
fn low_confidence_edge_warns() {
    let ir = make_ir(
        vec![node(0, NodeType::Action), node(1, NodeType::Etat)],
        vec![edge_low_conf(0, 1)],
    );
    let result = process(ir).unwrap();
    let low: Vec<_> = result
        .diagnostics
        .iter()
        .filter(|d| matches!(d.kind, DiagnosticKind::LowConfidenceEdge { .. }))
        .collect();
    assert_eq!(low.len(), 1);
}

#[test]
fn dangling_condition_warns() {
    // Condition node with no outgoing edge
    let ir = make_ir(
        vec![node(0, NodeType::Action), node(1, NodeType::Condition)],
        vec![edge(0, 1, RelationType::Cause)],
    );
    let result = process(ir).unwrap();
    let dangling: Vec<_> = result
        .diagnostics
        .iter()
        .filter(|d| matches!(d.kind, DiagnosticKind::DanglingCondition { .. }))
        .collect();
    assert_eq!(dangling.len(), 1);
    if let DiagnosticKind::DanglingCondition { node } = dangling[0].kind {
        assert_eq!(node, NodeId(1));
    }
}

#[test]
fn condition_with_outgoing_no_warning() {
    let ir = make_ir(
        vec![node(0, NodeType::Condition), node(1, NodeType::Action)],
        vec![edge(0, 1, RelationType::Condition)],
    );
    let result = process(ir).unwrap();
    assert!(
        !result
            .diagnostics
            .iter()
            .any(|d| matches!(d.kind, DiagnosticKind::DanglingCondition { .. }))
    );
}

// ─── pipeline metadata ───────────────────────────────────────────────────────

#[test]
fn pipeline_appended() {
    let ir = make_ir(
        vec![node(0, NodeType::Action), node(1, NodeType::Etat)],
        vec![edge(0, 1, RelationType::Cause)],
    );
    let result = process(ir).unwrap();
    assert_eq!(
        result.ir.metadata.pipeline,
        vec!["gcn-frontend-fr", "gcn-middleend"]
    );
}
