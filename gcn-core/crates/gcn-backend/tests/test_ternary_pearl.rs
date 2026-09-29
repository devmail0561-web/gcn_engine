// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0
//! Tests Éq.12 ETUDE — requêtes Pearl ternaires.

use gcn_backend::pearl;
use gcn_ir::{
    CausalEdge, CausalIR, CausalNode, ExtractionMethod, NaturalLanguage, NodeAttributes, NodeId,
    NodeOrigin, NodeType, Provenance, RelationType, Scope, SourceLanguage, SourceSpan, TemporalRef,
    TernaryRole, TernaryThird,
};

fn node(id: u32, label: &str) -> CausalNode {
    CausalNode {
        id: NodeId(id),
        node_type: NodeType::Processus,
        label: label.to_string(),
        source_span: SourceSpan::Synthetic,
        scope: Scope::Specific,
        modifiers: smallvec::smallvec![],
        temporal_ref: TemporalRef::Unresolved,
        temporal_index: None,
        origin: NodeOrigin::Explicit,
        parent: None,
        attributes: NodeAttributes::default(),
    }
}

fn edge(rel: RelationType, joint_id: Option<&str>, third: Option<TernaryThird>) -> CausalEdge {
    CausalEdge {
        relation: rel,
        confidence: 0.9,
        temporal_gap: None,
        explicit: true,
        negated: false,
        marker_token: None,
        in_cycle: None,
        provenance: Some(Provenance {
            doc_ref: None,
            span: SourceSpan::Synthetic,
            extraction_method: ExtractionMethod::MlPython,
            model_version: "test".to_string(),
            extracted_at: "2026-09-28T00:00:00Z".to_string(),
        }),
        derivation: None,
        joint_group_id: joint_id.map(|s| s.to_string()),
        third,
    }
}

fn ir(nodes: Vec<CausalNode>, edges: Vec<(NodeId, NodeId, CausalEdge)>) -> CausalIR {
    CausalIR {
        source_lang: SourceLanguage::Natural {
            lang: NaturalLanguage::Und,
        },
        source_text: "test".to_string(),
        nodes,
        edges,
        cycles: vec![],
        unresolved: vec![],
        metadata: gcn_ir::IrMetadata::default(),
    }
}

// ---------------------------------------------------------------------------
// WHY — arête JointCause porte joint_group_id dans CausalLink
// ---------------------------------------------------------------------------

#[test]
fn test_why_joint_cause_link_has_joint_group_id() {
    let ir = ir(
        vec![node(0, "A1"), node(1, "A2"), node(2, "C")],
        vec![
            (
                NodeId(0),
                NodeId(2),
                edge(RelationType::JointCause, Some("grp1"), None),
            ),
            (
                NodeId(1),
                NodeId(2),
                edge(RelationType::JointCause, Some("grp1"), None),
            ),
        ],
    );
    let results = pearl::why(&ir, "C");
    assert!(!results.is_empty());
    let (_, links) = &results[0];
    // Au moins une arête JointCause avec joint_group_id
    let has_joint = links.iter().any(|l| {
        l.relation == RelationType::JointCause && l.joint_group_id.as_deref() == Some("grp1")
    });
    assert!(
        has_joint,
        "joint_group_id devrait être propagé dans CausalLink WHY"
    );
}

// ---------------------------------------------------------------------------
// CHAIN — lien JointCause sur le chemin porte joint_group_id
// ---------------------------------------------------------------------------

#[test]
fn test_chain_joint_cause_link_has_joint_group_id() {
    let ir = ir(
        vec![node(0, "A"), node(1, "B"), node(2, "C")],
        vec![
            (NodeId(0), NodeId(1), edge(RelationType::Cause, None, None)),
            (
                NodeId(1),
                NodeId(2),
                edge(RelationType::JointCause, Some("grp2"), None),
            ),
        ],
    );
    let path = pearl::chain(&ir, "A", "C");
    assert!(path.is_some());
    let links = path.unwrap();
    let joint_link = links
        .iter()
        .find(|l| l.relation == RelationType::JointCause);
    assert!(
        joint_link.is_some(),
        "Lien JointCause devrait être dans le chemin"
    );
    assert_eq!(joint_link.unwrap().joint_group_id.as_deref(), Some("grp2"));
}

// ---------------------------------------------------------------------------
// COUNTERFACTUAL — JointCause : suppression X → C disparaît
// ---------------------------------------------------------------------------

#[test]
fn test_counterfactual_joint_cause_removes_effect() {
    // A1 ET A2 → C (nécessité conjointe)
    let ir = ir(
        vec![node(0, "A1"), node(1, "A2"), node(2, "C")],
        vec![
            (
                NodeId(0),
                NodeId(2),
                edge(RelationType::JointCause, Some("g"), None),
            ),
            (
                NodeId(1),
                NodeId(2),
                edge(RelationType::JointCause, Some("g"), None),
            ),
        ],
    );
    let result = pearl::counterfactual(&ir, "A1");
    assert!(result.is_some());
    let (_, cf) = result.unwrap();
    // C devrait être dans les effets réels de A1
    assert!(
        cf.actual_effects.iter().any(|l| l.to_label == "C"),
        "C devrait être un effet réel de A1"
    );
    // C est unique (pas d'autre chemin vers C sans A1, A2 seul ne suffit pas)
    assert!(
        cf.unique_effects.contains(&"C".to_string()),
        "C devrait être dans les effets uniques"
    );
}

// ---------------------------------------------------------------------------
// COUNTERFACTUAL — MediatedCause : médiateur orphelin disparaît
// ---------------------------------------------------------------------------

#[test]
fn test_counterfactual_mediated_cause_propagates_on_mediator() {
    // A → C via M (MediatedCause). M n'a qu'une seule source (A).
    let third = TernaryThird {
        role: TernaryRole::Mediator,
        node: 1,
        polarity: None,
    };
    let ir = ir(
        vec![node(0, "A"), node(1, "M"), node(2, "C")],
        vec![
            (
                NodeId(0),
                NodeId(2),
                edge(RelationType::MediatedCause, None, Some(third)),
            ),
            (NodeId(1), NodeId(2), edge(RelationType::Cause, None, None)),
        ],
    );
    let result = pearl::counterfactual(&ir, "A");
    assert!(result.is_some());
    let (_, cf) = result.unwrap();
    // C devrait être dans les effets réels
    assert!(cf.actual_effects.iter().any(|l| l.to_label == "C"));
}

// ---------------------------------------------------------------------------
// SPOF — is_super_spof pour les nœuds dans joint_group_id
// ---------------------------------------------------------------------------

#[test]
fn test_spof_super_spof_flag() {
    let ir = ir(
        vec![node(0, "A"), node(1, "B"), node(2, "C")],
        vec![
            (
                NodeId(0),
                NodeId(2),
                edge(RelationType::JointCause, Some("g"), None),
            ),
            (
                NodeId(1),
                NodeId(2),
                edge(RelationType::JointCause, Some("g"), None),
            ),
        ],
    );
    let (_, scores) = pearl::spof_all(&ir).expect("spof_all sur petit graphe");
    // A et B sont dans le groupe joint → super_spof
    for s in &scores {
        if s.label == "A" || s.label == "B" {
            assert!(
                s.is_super_spof,
                "{} devrait être super_spof (dans joint_group_id)",
                s.label
            );
        }
    }
    // C n'est pas une source → pas super_spof
    let c_score = scores.iter().find(|s| s.label == "C").unwrap();
    assert!(
        !c_score.is_super_spof,
        "C (cible) ne devrait pas être super_spof"
    );
}

// ---------------------------------------------------------------------------
// Rétrocompat — arête simple sans joint ni third : champs None
// ---------------------------------------------------------------------------

#[test]
fn test_causal_link_simple_edge_none_fields() {
    let ir = ir(
        vec![node(0, "A"), node(1, "B")],
        vec![(NodeId(0), NodeId(1), edge(RelationType::Cause, None, None))],
    );
    let results = pearl::why(&ir, "B");
    assert!(!results.is_empty());
    let (_, links) = &results[0];
    assert!(!links.is_empty());
    let link = &links[0];
    assert!(link.joint_group_id.is_none());
    assert!(link.third_node.is_none());
}
