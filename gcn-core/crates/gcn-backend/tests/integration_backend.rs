use gcn_backend::{execute, to_dot, to_json, BackendError, Query, QueryResult};
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
        source_lang: SourceLanguage::Natural { lang: NaturalLanguage::French },
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
    assert_eq!(Query::parse("WHY ventes?").unwrap(), Query::Why("ventes".into()));
}

#[test]
fn parse_what() {
    assert_eq!(Query::parse("WHAT coûts?").unwrap(), Query::What("coûts".into()));
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
    let ir = make_ir(vec![node(0, "chute(qualité)", NodeType::Transition)], vec![]);
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
        vec![edge(0, 1, RelationType::Cause), edge(1, 2, RelationType::Cause)],
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
        vec![edge(0, 1, RelationType::Cause), edge(1, 0, RelationType::Cause)],
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
        vec![node(0, "qualité", NodeType::Action), node(1, "ventes", NodeType::Etat)],
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
