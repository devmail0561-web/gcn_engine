use std::path::Path;
use gcn_frontend_fr::FrenchParser;
use gcn_ir::{NodeType, RelationType, Scope, NodeOrigin};

fn data_dir() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR = gcn-core/crates/gcn-frontend-fr
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../gcn-references/taxonomies")
}

fn parser() -> FrenchParser {
    FrenchParser::new(&data_dir()).expect("Failed to load taxonomies")
}

// paper-001: "Si les ventes baissent, on réduit les coûts."
#[test]
fn paper_001_condition() {
    let ir = parser().parse("Si les ventes baissent, on réduit les coûts.").unwrap();
    assert_eq!(ir.nodes.len(), 2, "Expected 2 nodes, got {}: {:?}", ir.nodes.len(),
        ir.nodes.iter().map(|n| &n.label).collect::<Vec<_>>());
    assert_eq!(ir.edges.len(), 1);
    let (_, _, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Condition);
    assert!(edge.explicit);
    assert!(!edge.negated);
    for n in &ir.nodes {
        assert_eq!(n.scope, Scope::Universal, "Node '{}' should have universal scope", n.label);
    }
}

// paper-002: "Le marché a souffert parce que la demande a chuté."
#[test]
fn paper_002_cause_backward() {
    let ir = parser().parse("Le marché a souffert parce que la demande a chuté.").unwrap();
    assert_eq!(ir.nodes.len(), 2);
    assert_eq!(ir.edges.len(), 1);
    let (src, dst, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Cause);
    assert!(edge.explicit);
    assert!(!edge.negated);
    let src_ti = ir.nodes[src.0 as usize].temporal_index;
    let dst_ti = ir.nodes[dst.0 as usize].temporal_index;
    assert!(src_ti < dst_ti, "Cause must precede effect: src_ti={:?}, dst_ti={:?}", src_ti, dst_ti);
}

// paper-003a: "Il court."
#[test]
fn paper_003a_simple_action() {
    let ir = parser().parse("Il court.").unwrap();
    assert_eq!(ir.nodes.len(), 1);
    assert_eq!(ir.edges.len(), 0);
    assert_eq!(ir.nodes[0].node_type, NodeType::Action);
}

// paper-003b: "Il court depuis une heure."
#[test]
fn paper_003b_depuis_compositional() {
    let ir = parser().parse("Il court depuis une heure.").unwrap();
    assert_eq!(ir.nodes.len(), 1);
    assert_eq!(ir.edges.len(), 0);
    assert_eq!(ir.nodes[0].node_type, NodeType::Processus, "action+depuis → processus");
    let has_temporality = ir.nodes[0].modifiers.iter()
        .any(|m| matches!(m, gcn_ir::Modifier::Temporality { .. }));
    assert!(has_temporality, "Should have Temporality modifier");
}

// paper-003c: "Il courait quand il a trébuché."
#[test]
fn paper_003c_sequence() {
    let ir = parser().parse("Il courait quand il a trébuché.").unwrap();
    assert_eq!(ir.nodes.len(), 2);
    assert_eq!(ir.edges.len(), 1);
    let (_, _, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Sequence);
    let processus_count = ir.nodes.iter().filter(|n| n.node_type == NodeType::Processus).count();
    assert!(processus_count >= 1, "Should have at least 1 processus node (imparfait)");
}

// paper-004: "Il ne vend pas."
#[test]
fn paper_004_negation_on_node() {
    let ir = parser().parse("Il ne vend pas.").unwrap();
    assert_eq!(ir.nodes.len(), 1);
    assert_eq!(ir.edges.len(), 0);
    let has_neg = ir.nodes[0].modifiers.iter()
        .any(|m| matches!(m, gcn_ir::Modifier::Negation { total: true }));
    assert!(has_neg, "Should have total negation modifier");
}

// paper-005: "Bien qu'il ait travaillé, il a échoué."
#[test]
fn paper_005_concession() {
    let ir = parser().parse("Bien qu'il ait travaillé, il a échoué.").unwrap();
    assert!(ir.nodes.len() >= 2, "Expected >=2 nodes, got {}", ir.nodes.len());
    let has_concession = ir.edges.iter().any(|(_, _, e)| e.relation == RelationType::Concession);
    assert!(has_concession, "Should have a concession edge");
}

// paper-006: "Il travaille pour réussir."
#[test]
fn paper_006_motivation() {
    let ir = parser().parse("Il travaille pour réussir.").unwrap();
    assert_eq!(ir.nodes.len(), 2);
    assert_eq!(ir.edges.len(), 1);
    let (_, _, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Motivation);
}

// paper-007: "Le marché qui s'effondre entraîne le chômage."
#[test]
fn paper_007_causal_verb() {
    let ir = parser().parse("Le marché qui s'effondre entraîne le chômage.").unwrap();
    assert_eq!(ir.nodes.len(), 2, "Expected 2 nodes, got {}", ir.nodes.len());
    assert_eq!(ir.edges.len(), 1);
    let (_, _, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Cause);
    assert!(edge.explicit);
}

// paper-008: "Tous les marchés s'effondrent."
#[test]
fn paper_008_universal_scope() {
    let ir = parser().parse("Tous les marchés s'effondrent.").unwrap();
    assert_eq!(ir.nodes.len(), 1);
    assert_eq!(ir.edges.len(), 0);
    assert_eq!(ir.nodes[0].scope, Scope::Universal);
}

// paper-009: "Les ventes baissent. On réduit les coûts."
#[test]
fn paper_009_implicit_juxtaposition() {
    let ir = parser().parse("Les ventes baissent. On réduit les coûts.").unwrap();
    assert_eq!(ir.nodes.len(), 2, "Two nodes from two sentences");
    assert_eq!(ir.edges.len(), 1);
    let (_, _, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Cause);
    assert!(!edge.explicit, "Should be inferred (not explicit)");
    assert!((edge.confidence - 0.5).abs() < 0.01, "Confidence should be 0.5");
}

// paper-010: "Grâce à son travail, il a réussi."
#[test]
fn paper_010_enable() {
    let ir = parser().parse("Grâce à son travail, il a réussi.").unwrap();
    assert_eq!(ir.nodes.len(), 2, "Expected 2 nodes, got {}", ir.nodes.len());
    assert_eq!(ir.edges.len(), 1);
    let (_, _, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Enable);
}

// paper-011: "La crise n'a pas causé le chômage."
#[test]
fn paper_011_negated_edge() {
    let ir = parser().parse("La crise n'a pas causé le chômage.").unwrap();
    assert_eq!(ir.nodes.len(), 2, "Expected 2 nodes, got {}", ir.nodes.len());
    assert_eq!(ir.edges.len(), 1);
    let (_, _, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Cause);
    assert!(edge.negated, "Edge should be negated");
}

// paper-012: "Bien qu'il ait travaillé, il a échoué."
#[test]
fn paper_012_hypothetical_node() {
    let ir = parser().parse("Bien qu'il ait travaillé, il a échoué.").unwrap();
    assert_eq!(ir.nodes.len(), 3, "Expected 3 nodes, got {}: {:?}", ir.nodes.len(),
        ir.nodes.iter().map(|n| format!("{:?}:{}", n.origin, n.label)).collect::<Vec<_>>());
    let hyp = ir.nodes.iter().find(|n| n.origin == NodeOrigin::Hypothetical);
    assert!(hyp.is_some(), "Should have a hypothetical node");
    assert_eq!(hyp.unwrap().node_type, NodeType::Condition);
    let hidden_edge = ir.edges.iter()
        .find(|(_, _, e)| e.relation == RelationType::Cause && (e.confidence - 0.3).abs() < 0.01);
    assert!(hidden_edge.is_some(), "Should have cause edge with confidence 0.3");
}

// Basic: empty input → error
#[test]
fn empty_input_errors() {
    let result = parser().parse("");
    assert!(result.is_err());
}

// Basic: single action sentence
#[test]
fn single_verb_produces_one_node() {
    let ir = parser().parse("Il travaille.").unwrap();
    assert_eq!(ir.nodes.len(), 1);
    assert_eq!(ir.edges.len(), 0);
    assert_eq!(ir.nodes[0].node_type, NodeType::Action);
}
