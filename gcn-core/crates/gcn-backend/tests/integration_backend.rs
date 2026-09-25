// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use gcn_backend::{BackendError, Query, QueryResult, execute, to_dot, to_json};
use gcn_ir::{
    CausalEdge, CausalIR, CausalNode, IrMetadata, NaturalLanguage, NodeAttributes, NodeId,
    NodeOrigin, NodeType, RelationType, Scope, SourceLanguage, SourceSpan, TemporalRef,
};
use gcn_middleend::process;
use smallvec::SmallVec;

fn node(id: u32, label: &str, node_type: NodeType) -> CausalNode {
    CausalNode {
        id: NodeId(id),
        node_type,
        label: label.to_string(),
        source_span: SourceSpan::Synthetic,
        scope: Scope::Unknown,
        modifiers: SmallVec::new(),
        temporal_ref: TemporalRef::Unresolved,
        temporal_index: Some(id as i32),
        origin: NodeOrigin::Explicit,
        attributes: NodeAttributes::default(),
    }
}

fn edge(src: u32, dst: u32, rel: RelationType) -> (NodeId, NodeId, CausalEdge) {
    (
        NodeId(src),
        NodeId(dst),
        CausalEdge {
            relation: rel,
            confidence: 0.9,
            temporal_gap: None,
            explicit: true,
            negated: false,
            marker_token: None,
            in_cycle: None,
        },
    )
}

fn make_ir(nodes: Vec<CausalNode>, edges: Vec<(NodeId, NodeId, CausalEdge)>) -> CausalIR {
    let ir = CausalIR {
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
    };
    // Run through middleend to populate cycles, in_cycle, etc.
    process(ir).unwrap().ir
}

// ─── Query parsing ───────────────────────────────────────────────────────────

#[test]
fn parse_why() {
    assert_eq!(
        Query::parse("WHY ventes?").unwrap(),
        Query::Why("ventes".into())
    );
}

// T-4 : WHY sans label après l'espace (ex: "WHY ") — doit retourner une erreur de parsing.
// Reason: trim() sur "WHY " produit "WHY" sans le préfixe "WHY " → aucun match → QueryParseError.
// Cela prévient un match-all silencieux si le label était vide.
#[test]
fn parse_why_trailing_space_only_returns_error() {
    assert!(
        Query::parse("WHY ").is_err(),
        "\"WHY \" (espace seul, pas de label) doit retourner une erreur de parsing"
    );
    assert!(
        Query::parse("WHY").is_err(),
        "\"WHY\" sans label doit retourner une erreur de parsing"
    );
}

#[test]
fn parse_what() {
    assert_eq!(
        Query::parse("WHAT coûts?").unwrap(),
        Query::What("coûts".into())
    );
}

#[test]
fn parse_chain() {
    assert_eq!(
        Query::parse("CHAIN qualité -> ventes?").unwrap(),
        Query::Chain("qualité".into(), "ventes".into())
    );
}

#[test]
fn parse_cycles() {
    assert_eq!(Query::parse("CYCLES?").unwrap(), Query::Cycles);
    assert_eq!(Query::parse("CYCLES").unwrap(), Query::Cycles);
}

#[test]
fn parse_gaps() {
    assert_eq!(Query::parse("GAPS?").unwrap(), Query::Gaps);
}

#[test]
fn parse_invalid() {
    assert!(Query::parse("UNKNOWN?").is_err());
    assert!(Query::parse("CHAIN x").is_err());
}

// ─── WHY / WHAT ──────────────────────────────────────────────────────────────

#[test]
fn why_finds_direct_cause() {
    // qualité → ventes
    let ir = make_ir(
        vec![
            node(0, "chute(qualité)", NodeType::Transition),
            node(1, "baisse(ventes)", NodeType::Etat),
        ],
        vec![edge(0, 1, RelationType::Cause)],
    );
    let result = execute(&Query::Why("ventes".into()), &ir).unwrap();
    if let QueryResult::Causes { links, .. } = result {
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].from, "chute(qualité)");
        assert_eq!(links[0].relation, "Cause");
    } else {
        panic!("expected Causes result");
    }
}

#[test]
fn why_node_not_found_errors() {
    let ir = make_ir(
        vec![node(0, "chute(qualité)", NodeType::Transition)],
        vec![],
    );
    let err = execute(&Query::Why("inexistant".into()), &ir);
    assert!(matches!(err, Err(BackendError::NodeNotFound(_))));
}

#[test]
fn what_finds_direct_effect() {
    // A → B → C
    let ir = make_ir(
        vec![
            node(0, "chute(qualité)", NodeType::Transition),
            node(1, "baisse(ventes)", NodeType::Etat),
            node(2, "réduction(coûts)", NodeType::Action),
        ],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 2, RelationType::Cause),
        ],
    );
    let result = execute(&Query::What("qualité".into()), &ir).unwrap();
    if let QueryResult::Effects { links, .. } = result {
        assert_eq!(links.len(), 2);
    } else {
        panic!("expected Effects result");
    }
}

// ─── CHAIN ───────────────────────────────────────────────────────────────────

#[test]
fn chain_finds_path() {
    let ir = make_ir(
        vec![
            node(0, "chute(qualité)", NodeType::Transition),
            node(1, "baisse(ventes)", NodeType::Etat),
            node(2, "réduction(coûts)", NodeType::Action),
        ],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 2, RelationType::Cause),
        ],
    );
    let result = execute(&Query::Chain("qualité".into(), "coûts".into()), &ir).unwrap();
    if let QueryResult::Path { found, links, .. } = result {
        assert!(found);
        assert_eq!(links.len(), 2);
    } else {
        panic!("expected Path result");
    }
}

#[test]
fn chain_no_path() {
    let ir = make_ir(
        vec![
            node(0, "chute(qualité)", NodeType::Transition),
            node(1, "baisse(ventes)", NodeType::Etat),
        ],
        vec![edge(0, 1, RelationType::Cause)],
    );
    let result = execute(&Query::Chain("ventes".into(), "qualité".into()), &ir).unwrap();
    if let QueryResult::Path { found, .. } = result {
        assert!(!found);
    } else {
        panic!("expected Path result");
    }
}

// ─── CYCLES ──────────────────────────────────────────────────────────────────

#[test]
fn cycles_detects_feedback_loop() {
    let ir = make_ir(
        vec![node(0, "A", NodeType::Action), node(1, "B", NodeType::Etat)],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 0, RelationType::Cause),
        ],
    );
    let result = execute(&Query::Cycles, &ir).unwrap();
    if let QueryResult::CycleList { count, .. } = result {
        assert_eq!(count, 1);
    } else {
        panic!("expected CycleList");
    }
}

#[test]
fn cycles_empty_on_acyclic() {
    let ir = make_ir(
        vec![node(0, "A", NodeType::Action), node(1, "B", NodeType::Etat)],
        vec![edge(0, 1, RelationType::Cause)],
    );
    let result = execute(&Query::Cycles, &ir).unwrap();
    if let QueryResult::CycleList { count, .. } = result {
        assert_eq!(count, 0);
    } else {
        panic!("expected CycleList");
    }
}

// ─── GAPS ────────────────────────────────────────────────────────────────────

#[test]
fn gaps_finds_concession_gap() {
    let ir = make_ir(
        vec![node(0, "A", NodeType::Etat), node(1, "B", NodeType::Etat)],
        vec![edge(0, 1, RelationType::Concession)],
    );
    let result = execute(&Query::Gaps, &ir).unwrap();
    if let QueryResult::GapList { gap_edges, .. } = result {
        assert_eq!(gap_edges.len(), 1);
        assert_eq!(gap_edges[0].from, "A");
    } else {
        panic!("expected GapList");
    }
}

#[test]
fn gaps_empty_on_clean_graph() {
    let ir = make_ir(
        vec![node(0, "A", NodeType::Action), node(1, "B", NodeType::Etat)],
        vec![edge(0, 1, RelationType::Cause)],
    );
    let result = execute(&Query::Gaps, &ir).unwrap();
    if let QueryResult::GapList { gap_edges, .. } = result {
        assert!(gap_edges.is_empty());
    } else {
        panic!("expected GapList");
    }
}

// ─── Export ──────────────────────────────────────────────────────────────────

#[test]
fn json_export_is_valid() {
    let ir = make_ir(
        vec![node(0, "A", NodeType::Action), node(1, "B", NodeType::Etat)],
        vec![edge(0, 1, RelationType::Cause)],
    );
    let json = to_json(&ir).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(parsed["nodes"].is_array());
    assert!(parsed["edges"].is_array());
    assert_eq!(parsed["nodes"].as_array().unwrap().len(), 2);
}

#[test]
fn dot_export_contains_nodes_and_edges() {
    let ir = make_ir(
        vec![
            node(0, "qualité", NodeType::Action),
            node(1, "ventes", NodeType::Etat),
        ],
        vec![edge(0, 1, RelationType::Cause)],
    );
    let dot = to_dot(&ir).unwrap();
    assert!(dot.contains("digraph"));
    assert!(dot.contains("n0"));
    assert!(dot.contains("n1"));
    assert!(dot.contains("n0 -> n1"));
    assert!(dot.contains("Cause"));
}

#[test]
fn json_roundtrip() {
    let ir = make_ir(
        vec![node(0, "A", NodeType::Action), node(1, "B", NodeType::Etat)],
        vec![edge(0, 1, RelationType::Cause)],
    );
    let json = to_json(&ir).unwrap();
    let restored: CausalIR = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.nodes.len(), ir.nodes.len());
    assert_eq!(restored.edges.len(), ir.edges.len());
}

// ─── Pearl niveau 2 — Intervention (DO) ─────────────────────────────────────

#[test]
fn parse_do() {
    assert_eq!(
        Query::parse("DO crise?").unwrap(),
        Query::Intervene("crise".into())
    );
    assert_eq!(
        Query::parse("DO hausse").unwrap(),
        Query::Intervene("hausse".into())
    );
}

#[test]
fn intervene_cuts_incoming_and_propagates_forward() {
    // Graphe : A → B → C
    // Intervention sur B : coupe A→B, propage B→C
    let ir = make_ir(
        vec![
            node(0, "A", NodeType::Action),
            node(1, "B", NodeType::Transition),
            node(2, "C", NodeType::Etat),
        ],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 2, RelationType::Cause),
        ],
    );
    let result = execute(&Query::Intervene("B".into()), &ir).unwrap();
    if let QueryResult::Intervention {
        severed_count,
        severed,
        effects,
        ..
    } = result
    {
        assert_eq!(severed_count, 1, "une arête entrante coupée (A→B)");
        assert_eq!(severed.len(), 1);
        assert_eq!(severed[0].from, "A");
        assert_eq!(severed[0].to, "B");
        assert!(!effects.is_empty(), "B→C doit être dans les effets");
        assert_eq!(effects[0].to, "C");
    } else {
        panic!("résultat inattendu : {result:?}");
    }
}

#[test]
fn intervene_no_incoming_zero_severed() {
    // Nœud racine sans arête entrante : severed_count = 0
    let ir = make_ir(
        vec![
            node(0, "racine", NodeType::Action),
            node(1, "effet", NodeType::Etat),
        ],
        vec![edge(0, 1, RelationType::Cause)],
    );
    let result = execute(&Query::Intervene("racine".into()), &ir).unwrap();
    if let QueryResult::Intervention {
        severed_count,
        effects,
        ..
    } = result
    {
        assert_eq!(severed_count, 0);
        assert_eq!(effects.len(), 1);
    } else {
        panic!("résultat inattendu");
    }
}

#[test]
fn intervene_node_not_found() {
    let ir = make_ir(vec![node(0, "A", NodeType::Action)], vec![]);
    assert!(matches!(
        execute(&Query::Intervene("INEXISTANT".into()), &ir),
        Err(BackendError::NodeNotFound(_))
    ));
}

// ─── Pearl niveau 3 — Contrefactuels ─────────────────────────────────────────

#[test]
fn parse_counterfactual() {
    assert_eq!(
        Query::parse("COUNTERFACTUAL crise?").unwrap(),
        Query::Counterfactual("crise".into())
    );
}

#[test]
fn counterfactual_unique_effect_detected() {
    // Graphe : A → B → C (chaîne simple)
    // COUNTERFACTUAL B : C est uniquement atteignable via B → unique
    let ir = make_ir(
        vec![
            node(0, "A", NodeType::Action),
            node(1, "B", NodeType::Transition),
            node(2, "C", NodeType::Etat),
        ],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 2, RelationType::Cause),
        ],
    );
    let result = execute(&Query::Counterfactual("B".into()), &ir).unwrap();
    if let QueryResult::CounterfactualDiff {
        actual_effects,
        unique_effects,
        ..
    } = result
    {
        assert!(!actual_effects.is_empty());
        assert!(
            unique_effects.contains(&"C".to_string()),
            "C est uniquement causé par B"
        );
    } else {
        panic!("résultat inattendu");
    }
}

#[test]
fn counterfactual_shared_effect_not_unique() {
    // Graphe : A → C et B → C (C a deux chemins)
    // COUNTERFACTUAL B : C n'est PAS unique (A→C existe sans B)
    let ir = make_ir(
        vec![
            node(0, "A", NodeType::Action),
            node(1, "B", NodeType::Action),
            node(2, "C", NodeType::Etat),
        ],
        vec![
            edge(0, 2, RelationType::Cause),
            edge(1, 2, RelationType::Cause),
        ],
    );
    let result = execute(&Query::Counterfactual("B".into()), &ir).unwrap();
    if let QueryResult::CounterfactualDiff { unique_effects, .. } = result {
        assert!(
            !unique_effects.contains(&"C".to_string()),
            "C a un chemin alternatif via A"
        );
    } else {
        panic!("résultat inattendu");
    }
}

#[test]
fn counterfactual_node_not_found() {
    let ir = make_ir(vec![node(0, "A", NodeType::Action)], vec![]);
    assert!(matches!(
        execute(&Query::Counterfactual("INEXISTANT".into()), &ir),
        Err(BackendError::NodeNotFound(_))
    ));
}

// fix C-4: unique_effects préserve la cardinalité quand deux nœuds distincts partagent le même label
#[test]
fn fix_c4_unique_effects_two_same_label_nodes_both_unique() {
    // trigger(1) → doublon(0) et trigger(1) → doublon(2)
    // Les deux doublons n'ont aucune autre source → tous deux uniques à trigger
    let ir = make_ir(
        vec![
            node(0, "doublon", NodeType::Etat),
            node(1, "trigger", NodeType::Action),
            node(2, "doublon", NodeType::Etat),
        ],
        vec![
            edge(1, 0, RelationType::Cause),
            edge(1, 2, RelationType::Cause),
        ],
    );
    let result = execute(&Query::Counterfactual("trigger".into()), &ir).unwrap();
    if let QueryResult::CounterfactualDiff { unique_effects, .. } = result {
        assert_eq!(
            unique_effects.len(),
            2,
            "deux nœuds 'doublon' distincts et uniques → 2 unique_effects attendus, obtenu {:?}",
            unique_effects
        );
        assert_eq!(
            unique_effects
                .iter()
                .filter(|s| s.as_str() == "doublon")
                .count(),
            2
        );
    } else {
        panic!("résultat inattendu");
    }
}

// fix BUG-2: counterfactual avec labels dupliqués (un unique, un non-unique)
#[test]
fn fix_bug2_counterfactual_duplicate_labels() {
    // doublon(id=0) racine standalone, trigger(id=1) → doublon(id=2)
    // doublon(id=2) est uniquement atteignable via trigger
    let ir = make_ir(
        vec![
            node(0, "doublon", NodeType::Etat),
            node(1, "trigger", NodeType::Action),
            node(2, "doublon", NodeType::Etat),
        ],
        vec![edge(1, 2, RelationType::Cause)],
    );
    let result = execute(&Query::Counterfactual("trigger".into()), &ir).unwrap();
    if let QueryResult::CounterfactualDiff { unique_effects, .. } = result {
        assert!(
            unique_effects.contains(&"doublon".to_string()),
            "doublon(id=2) est unique à trigger : {:?}",
            unique_effects
        );
    } else {
        panic!("résultat inattendu");
    }
}
