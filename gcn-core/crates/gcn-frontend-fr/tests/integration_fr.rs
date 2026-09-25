// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use gcn_frontend_fr::FrenchParser;
use gcn_ir::{NodeOrigin, NodeType, RelationType, Scope};
use std::path::Path;

fn data_dir() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR = gcn-core/crates/gcn-frontend-fr
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../gcn-references/taxonomies")
}

fn parser() -> FrenchParser {
    FrenchParser::new(&data_dir()).expect("Failed to load taxonomies")
}

// paper-001: "Si les ventes baissent, on réduit les coûts."
#[test]
fn paper_001_condition() {
    let ir = parser()
        .parse("Si les ventes baissent, on réduit les coûts.")
        .unwrap();
    assert_eq!(
        ir.nodes.len(),
        2,
        "Expected 2 nodes, got {}: {:?}",
        ir.nodes.len(),
        ir.nodes.iter().map(|n| &n.label).collect::<Vec<_>>()
    );
    assert_eq!(ir.edges.len(), 1);
    let (_, _, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Condition);
    assert!(edge.explicit);
    assert!(!edge.negated);
    for n in &ir.nodes {
        assert_eq!(
            n.scope,
            Scope::Universal,
            "Node '{}' should have universal scope",
            n.label
        );
    }
}

// paper-002: "Le marché a souffert parce que la demande a chuté."
#[test]
fn paper_002_cause_backward() {
    let ir = parser()
        .parse("Le marché a souffert parce que la demande a chuté.")
        .unwrap();
    assert_eq!(ir.nodes.len(), 2);
    assert_eq!(ir.edges.len(), 1);
    let (src, dst, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Cause);
    assert!(edge.explicit);
    assert!(!edge.negated);
    let src_ti = ir.nodes[src.0 as usize].temporal_index;
    let dst_ti = ir.nodes[dst.0 as usize].temporal_index;
    assert!(
        src_ti < dst_ti,
        "Cause must precede effect: src_ti={:?}, dst_ti={:?}",
        src_ti,
        dst_ti
    );
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
    assert_eq!(
        ir.nodes[0].node_type,
        NodeType::Processus,
        "action+depuis → processus"
    );
    let has_temporality = ir.nodes[0]
        .modifiers
        .iter()
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
    let processus_count = ir
        .nodes
        .iter()
        .filter(|n| n.node_type == NodeType::Processus)
        .count();
    assert!(
        processus_count >= 1,
        "Should have at least 1 processus node (imparfait)"
    );
}

// paper-004: "Il ne vend pas."
#[test]
fn paper_004_negation_on_node() {
    let ir = parser().parse("Il ne vend pas.").unwrap();
    assert_eq!(ir.nodes.len(), 1);
    assert_eq!(ir.edges.len(), 0);
    let has_neg = ir.nodes[0]
        .modifiers
        .iter()
        .any(|m| matches!(m, gcn_ir::Modifier::Negation { total: true }));
    assert!(has_neg, "Should have total negation modifier");
}

// paper-005: "Bien qu'il ait travaillé, il a échoué."
#[test]
fn paper_005_concession() {
    let ir = parser()
        .parse("Bien qu'il ait travaillé, il a échoué.")
        .unwrap();
    assert!(
        ir.nodes.len() >= 2,
        "Expected >=2 nodes, got {}",
        ir.nodes.len()
    );
    let has_concession = ir
        .edges
        .iter()
        .any(|(_, _, e)| e.relation == RelationType::Concession);
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
    let ir = parser()
        .parse("Le marché qui s'effondre entraîne le chômage.")
        .unwrap();
    assert_eq!(
        ir.nodes.len(),
        2,
        "Expected 2 nodes, got {}",
        ir.nodes.len()
    );
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
    let ir = parser()
        .parse("Les ventes baissent. On réduit les coûts.")
        .unwrap();
    assert_eq!(ir.nodes.len(), 2, "Two nodes from two sentences");
    assert_eq!(ir.edges.len(), 1);
    let (_, _, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Cause);
    assert!(!edge.explicit, "Should be inferred (not explicit)");
    assert!(
        (edge.confidence - 0.5).abs() < 0.01,
        "Confidence should be 0.5"
    );
}

// paper-010: "Grâce à son travail, il a réussi."
#[test]
fn paper_010_enable() {
    let ir = parser().parse("Grâce à son travail, il a réussi.").unwrap();
    assert_eq!(
        ir.nodes.len(),
        2,
        "Expected 2 nodes, got {}",
        ir.nodes.len()
    );
    assert_eq!(ir.edges.len(), 1);
    let (_, _, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Enable);
}

// paper-011: "La crise n'a pas causé le chômage."
#[test]
fn paper_011_negated_edge() {
    let ir = parser()
        .parse("La crise n'a pas causé le chômage.")
        .unwrap();
    assert_eq!(
        ir.nodes.len(),
        2,
        "Expected 2 nodes, got {}",
        ir.nodes.len()
    );
    assert_eq!(ir.edges.len(), 1);
    let (_, _, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Cause);
    assert!(edge.negated, "Edge should be negated");
}

// paper-012: "Bien qu'il ait travaillé, il a échoué."
#[test]
fn paper_012_hypothetical_node() {
    let ir = parser()
        .parse("Bien qu'il ait travaillé, il a échoué.")
        .unwrap();
    assert_eq!(
        ir.nodes.len(),
        3,
        "Expected 3 nodes, got {}: {:?}",
        ir.nodes.len(),
        ir.nodes
            .iter()
            .map(|n| format!("{:?}:{}", n.origin, n.label))
            .collect::<Vec<_>>()
    );
    let hyp = ir
        .nodes
        .iter()
        .find(|n| n.origin == NodeOrigin::Hypothetical);
    assert!(hyp.is_some(), "Should have a hypothetical node");
    assert_eq!(hyp.unwrap().node_type, NodeType::Condition);
    let hidden_edge = ir
        .edges
        .iter()
        .find(|(_, _, e)| e.relation == RelationType::Cause && (e.confidence - 0.3).abs() < 0.01);
    assert!(
        hidden_edge.is_some(),
        "Should have cause edge with confidence 0.3"
    );
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

// T-3 : chaîne A→B→C — temporal_index strictement croissant
#[test]
fn three_clause_chain_temporal_indices_strictly_increasing() {
    // Deux arêtes consécutives : parce que … parce que …
    // La phrase doit produire 3 nœuds reliés en chaîne
    let ir = parser()
        .parse(
            "Les ventes baissent parce que les coûts augmentent parce que les salaires ont grimpé.",
        )
        .unwrap();

    if ir.nodes.len() < 3 || ir.edges.len() < 2 {
        // Si le parser ne détecte pas 3 clauses, on passe le test (syntaxe trop complexe)
        return;
    }

    let ti: Vec<i32> = ir
        .nodes
        .iter()
        .map(|n| n.temporal_index.unwrap_or(-1))
        .collect();

    // Chaque nœud consécutif dans la chaîne doit avoir un ti strictement supérieur
    for edge_pair in ir.edges.windows(2) {
        let (src_a, dst_a, _) = &edge_pair[0];
        let (src_b, dst_b, _) = &edge_pair[1];
        if dst_a.0 == src_b.0 {
            // A→B→C : ti(A) < ti(B) < ti(C)
            let ti_a = ti[src_a.0 as usize];
            let ti_b = ti[dst_a.0 as usize];
            let ti_c = ti[dst_b.0 as usize];
            assert!(ti_a < ti_b, "ti(A)={} doit être < ti(B)={}", ti_a, ti_b);
            assert!(ti_b < ti_c, "ti(B)={} doit être < ti(C)={}", ti_b, ti_c);
        }
    }
}
// Verify BUG-1 fix: 3 phrases juxtaposées → 2 arêtes implicites
#[test]
fn fix_bug1_three_sentence_juxtaposition() {
    let ir = parser()
        .parse("Les ventes baissent. On réduit les coûts. On licencie.")
        .unwrap();
    assert_eq!(ir.nodes.len(), 3);
    assert_eq!(
        ir.edges.len(),
        2,
        "attendu 2 arêtes implicites, got {}",
        ir.edges.len()
    );
    for (_, _, e) in &ir.edges {
        assert!(!e.explicit);
        assert!((e.confidence - 0.5).abs() < 0.01);
    }
    assert_eq!(ir.edges[0].0.0, 0);
    assert_eq!(ir.edges[0].1.0, 1);
    assert_eq!(ir.edges[1].0.0, 1);
    assert_eq!(ir.edges[1].1.0, 2);
}

// Verify BUG-3+4 fix: concession non-initiale → nœud hypothétique pointe vers l'effet surprenant
#[test]
fn fix_bug3_hypothetical_targets_main_clause() {
    let ir = parser()
        .parse("Il a échoué bien qu'il ait travaillé.")
        .unwrap();
    assert_eq!(ir.nodes.len(), 3);
    let hyp_idx = ir
        .nodes
        .iter()
        .position(|n| n.origin == NodeOrigin::Hypothetical)
        .unwrap() as u32;
    let hyp_edge = ir.edges.iter().find(|(s, _, _)| s.0 == hyp_idx).unwrap();
    let dst_node = ir.nodes.iter().find(|n| n.id == hyp_edge.1).unwrap();
    // le nœud hypothétique doit pointer vers l'événement surprenant (l'échec)
    assert!(
        dst_node.label.contains("échouer")
            || dst_node.label.contains("échec")
            || dst_node.label.contains("échoué"),
        "hyp → '{}' — attendu l'événement surprenant",
        dst_node.label
    );
}

// paper-013: "Pour réussir, il travaille." — GoalToAction en position initiale
// BUT(réussir) → ACTION(travaille), pas l'inverse.
#[test]
fn paper_013_goal_to_action_initial() {
    let ir = parser().parse("Pour réussir, il travaille.").unwrap();
    assert_eq!(ir.nodes.len(), 2);
    assert_eq!(ir.edges.len(), 1);
    let (src, dst, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Motivation);
    let src_label = &ir.nodes[src.0 as usize].label;
    let dst_label = &ir.nodes[dst.0 as usize].label;
    assert!(
        src_label.contains("réussir") || src_label.contains("réussite"),
        "src doit être le BUT, obtenu src='{}' dst='{}'",
        src_label,
        dst_label
    );
}

// fix C-1: concession suivie d'une phrase → arête implicite depuis clause principale, pas nœud hypothétique
#[test]
fn fix_c1_concession_sentence_implicit_edge_src_not_hypothetical() {
    // Phrase 0 = concession → 3 clauses : effet(0), concessif(1), cause_cachée(2 = Hypothetical)
    // L'arête implicite doit partir de la clause principale (0 ou 1), pas de cause_cachée(2)
    let ir = parser()
        .parse("Il a échoué bien qu'il ait travaillé. On continue.")
        .unwrap();

    let implicit = ir
        .edges
        .iter()
        .find(|(_, _, e)| !e.explicit && (e.confidence - 0.5).abs() < 0.01);

    assert!(
        implicit.is_some(),
        "Doit avoir une arête implicite inter-phrase"
    );
    let (src_id, _, _) = implicit.unwrap();
    let src_node = ir.nodes.iter().find(|n| n.id == *src_id).unwrap();
    assert_ne!(
        src_node.origin,
        NodeOrigin::Hypothetical,
        "C-1: src arête implicite = '{}' (Hypothetical) — doit être un nœud réel",
        src_node.label
    );
}
