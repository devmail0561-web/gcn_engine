// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use gcn_backend::{BackendError, Query, QueryResult, TemporalLinkDto, execute, to_dot, to_json};
use gcn_ir::{
    CausalEdge, CausalIR, CausalNode, IrMetadata, NaturalLanguage, NodeAttributes, NodeId,
    NodeOrigin, NodeType, RelationType, Scope, SourceLanguage, SourceSpan, TemporalGap,
    TemporalRef,
};
use gcn_ir::temporal::GapNature;
use gcn_middleend::process;
use smallvec::SmallVec;

fn node_with_ti(id: u32, label: &str, node_type: NodeType, ti: i32) -> CausalNode {
    let mut n = node(id, label, node_type);
    n.temporal_index = Some(ti);
    n
}

fn edge_with_gap(src: u32, dst: u32, rel: RelationType, min: i32, max: i32) -> (NodeId, NodeId, CausalEdge) {
    let (s, d, mut e) = edge(src, dst, rel);
    e.temporal_gap = Some(TemporalGap { min: Some(min), max: Some(max), nature: GapNature::Deferred });
    (s, d, e)
}

fn node(id: u32, label: &str, node_type: NodeType) -> CausalNode {
    CausalNode {
        id: NodeId(id),
        node_type,
        parent: None,
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
            provenance: None,
            derivation: None,
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

// ─── Raisonnement par analogie ───────────────────────────────────────────────

#[test]
fn analogy_same_relation_type_scores_high() {
    // Patron : A→B (Cause). Analogue attendu : C→D (Cause). E→F (Enable) = score moindre.
    let ir = make_ir(
        vec![
            node(0, "A", NodeType::Action), node(1, "B", NodeType::Etat),
            node(2, "C", NodeType::Action), node(3, "D", NodeType::Etat),
            node(4, "E", NodeType::Action), node(5, "F", NodeType::Etat),
        ],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(2, 3, RelationType::Cause),
            edge(4, 5, RelationType::Enable),
        ],
    );
    let result = execute(&Query::Analogy("A".into(), "B".into()), &ir).unwrap();
    if let QueryResult::AnalogyReport { matches, .. } = result {
        assert!(!matches.is_empty(), "au moins un analogue attendu");
        // C→D (Cause) doit scorer plus haut que E→F (Enable)
        let cd = matches.iter().find(|m| m.from_label == "C" && m.to_label == "D");
        let ef = matches.iter().find(|m| m.from_label == "E" && m.to_label == "F");
        assert!(cd.is_some(), "C→D doit être dans les matches");
        if let (Some(cd_m), Some(ef_m)) = (cd, ef) {
            assert!(cd_m.similarity_score >= ef_m.similarity_score,
                "C→D score={} doit être ≥ E→F score={}", cd_m.similarity_score, ef_m.similarity_score);
        }
    } else { panic!("résultat ANALOGY inattendu"); }
}

#[test]
fn analogy_unknown_pattern_returns_empty() {
    let ir = make_ir(
        vec![node(0, "X", NodeType::Action), node(1, "Y", NodeType::Etat)],
        vec![],
    );
    // Patron X→Y n'existe pas dans le graphe (pas d'arête)
    let result = execute(&Query::Analogy("X".into(), "Y".into()), &ir).unwrap();
    if let QueryResult::AnalogyReport { matches, .. } = result {
        assert!(matches.is_empty(), "pas d'arête patron → aucun analogue");
    } else { panic!("résultat ANALOGY inattendu"); }
}

// ─── Multi-échelle ────────────────────────────────────────────────────────────

fn make_hierarchy_ir() -> CausalIR {
    // Hiérarchie : process(0) est le parent de step_a(1) et step_b(2)
    // step_a(1) → result(3)
    let mut nodes = vec![
        node(0, "process", NodeType::Processus),
        node(1, "step_a", NodeType::Action),
        node(2, "step_b", NodeType::Action),
        node(3, "result", NodeType::Etat),
    ];
    nodes[1].parent = Some(NodeId(0));
    nodes[2].parent = Some(NodeId(0));
    make_ir(nodes, vec![
        edge(1, 3, RelationType::Cause),
        edge(2, 3, RelationType::Enable),
    ])
}

#[test]
fn zoom_in_returns_children() {
    let ir = make_hierarchy_ir();
    let result = execute(&Query::ZoomIn("process".into()), &ir).unwrap();
    if let QueryResult::HierarchyZoomIn { parent_label, children } = result {
        assert_eq!(parent_label, "process");
        assert_eq!(children.len(), 2);
        let labels: Vec<&str> = children.iter().map(|c| c.label.as_str()).collect();
        assert!(labels.contains(&"step_a"));
        assert!(labels.contains(&"step_b"));
    } else { panic!("résultat ZoomIn inattendu"); }
}

#[test]
fn zoom_out_returns_parent() {
    let ir = make_hierarchy_ir();
    let result = execute(&Query::ZoomOut("step_a".into()), &ir).unwrap();
    if let QueryResult::HierarchyZoomOut { child_label, parent_label } = result {
        assert_eq!(child_label, "step_a");
        assert_eq!(parent_label.as_deref(), Some("process"));
    } else { panic!("résultat ZoomOut inattendu"); }
}

#[test]
fn zoom_out_root_returns_none() {
    let ir = make_hierarchy_ir();
    let result = execute(&Query::ZoomOut("process".into()), &ir).unwrap();
    if let QueryResult::HierarchyZoomOut { parent_label, .. } = result {
        assert!(parent_label.is_none(), "process est à la racine");
    } else { panic!("résultat ZoomOut inattendu"); }
}

#[test]
fn aggregate_children_edges() {
    let ir = make_hierarchy_ir();
    let result = execute(&Query::Aggregate("process".into()), &ir).unwrap();
    if let QueryResult::HierarchyAggregate { parent_label, child_count, outgoing_edges, mean_confidence } = result {
        assert_eq!(parent_label, "process");
        assert_eq!(child_count, 2);
        // step_a → result et step_b → result sont les arêtes sortantes vers l'extérieur
        assert_eq!(outgoing_edges.len(), 2);
        assert!(mean_confidence > 0.0);
    } else { panic!("résultat Aggregate inattendu"); }
}

// ─── Raisonnement adversarial ────────────────────────────────────────────────

#[test]
fn centrality_hub_node() {
    // hub(0) → B(1) avec conf=0.9, hub(0) → C(2) avec conf=0.7
    let ir = make_ir(
        vec![node(0,"hub",NodeType::Action), node(1,"B",NodeType::Etat), node(2,"C",NodeType::Etat)],
        vec![edge(0,1,RelationType::Cause), edge(0,2,RelationType::Cause)],
    );
    let result = execute(&Query::Centrality("hub".into()), &ir).unwrap();
    if let QueryResult::CentralityReport { degree_out, weighted_out, centrality_score, .. } = result {
        assert_eq!(degree_out, 2);
        assert!((weighted_out - 1.8).abs() < 0.01, "weighted_out={weighted_out}");
        assert!(centrality_score > 0.0);
    } else { panic!("résultat Centrality inattendu"); }
}

#[test]
fn spof_bridge_node_detected() {
    // A→B→C : paires totales = A→B, A→C, B→C = 3
    // Sans B : seuls A→? = rien (B absent) et C→? = rien → total 0
    // B coupe 3 paires → SPOF score maximal
    let ir = make_ir(
        vec![node(0,"A",NodeType::Action), node(1,"B",NodeType::Transition), node(2,"C",NodeType::Etat)],
        vec![edge(0,1,RelationType::Cause), edge(1,2,RelationType::Cause)],
    );
    let result = execute(&Query::Spof, &ir).unwrap();
    if let QueryResult::SpofReport { nodes, total_pairs, .. } = result {
        assert_eq!(total_pairs, 3, "A→B, A→C, B→C = 3 paires");
        let b = nodes.iter().find(|n| n.label == "B").expect("B dans les résultats");
        // Supprimer B coupe A→B, A→C, B→C = 3 paires
        assert_eq!(b.paths_cut, 3, "B coupe les 3 paires : {:?}", b);
        // B est premier dans le classement (score maximal)
        assert_eq!(nodes[0].label, "B", "B doit être premier SPOF");
    } else { panic!("résultat SPOF? inattendu"); }
}

// ─── Raisonnement normatif ────────────────────────────────────────────────────

#[test]
fn diff_rgpd_gap_27_percent() {
    // Règle normative : accès → journalisation doit avoir conf=1.0
    // Fait observé : conf=0.73 → gap=0.27, gap_rate=27%
    let mut ir = make_ir(
        vec![
            node(0, "accès", NodeType::Action),
            node(1, "journalisation", NodeType::Action),
        ],
        vec![edge(0, 1, RelationType::Condition)],
    );
    ir.edges[0].2.confidence = 0.73;
    let result = execute(&Query::NormDiff("accès".into(), "journalisation".into()), &ir).unwrap();
    if let QueryResult::NormativeDiff { path_exists, observed_confidence, gap, gap_rate_pct, .. } = result {
        assert!(path_exists);
        assert!((observed_confidence - 0.73).abs() < 0.001, "conf={observed_confidence}");
        assert!((gap - 0.27).abs() < 0.001, "gap={gap}");
        assert!((gap_rate_pct - 27.0).abs() < 0.1, "gap_rate={gap_rate_pct}");
    } else { panic!("résultat NormDiff inattendu"); }
}

#[test]
fn diff_no_path_gap_100_percent() {
    let ir = make_ir(
        vec![node(0, "A", NodeType::Action), node(1, "B", NodeType::Etat)],
        vec![],
    );
    let result = execute(&Query::NormDiff("A".into(), "B".into()), &ir).unwrap();
    if let QueryResult::NormativeDiff { path_exists, gap, .. } = result {
        assert!(!path_exists);
        assert!((gap - 1.0).abs() < 0.001);
    } else { panic!("résultat NormDiff inattendu"); }
}

// ─── Méta-raisonnement ───────────────────────────────────────────────────────

#[test]
fn density_global() {
    // 3 nœuds, 2 arêtes → densité = 2/(3×2) = 0.333
    let ir = make_ir(
        vec![node(0,"A",NodeType::Action), node(1,"B",NodeType::Etat), node(2,"C",NodeType::Etat)],
        vec![edge(0,1,RelationType::Cause), edge(1,2,RelationType::Cause)],
    );
    let result = execute(&Query::Density(None), &ir).unwrap();
    if let QueryResult::DensityReport { global_density, n_nodes, n_edges, node_label, .. } = result {
        assert_eq!(n_nodes, 3);
        assert_eq!(n_edges, 2);
        assert!((global_density - 2.0/6.0).abs() < 0.001);
        assert!(node_label.is_none());
    } else { panic!("résultat Density inattendu"); }
}

#[test]
fn density_local() {
    let ir = make_ir(
        vec![node(0,"hub",NodeType::Action), node(1,"B",NodeType::Etat), node(2,"C",NodeType::Etat)],
        vec![edge(0,1,RelationType::Cause), edge(0,2,RelationType::Cause)],
    );
    let result = execute(&Query::Density(Some("hub".into())), &ir).unwrap();
    if let QueryResult::DensityReport { local_degree, node_label, .. } = result {
        assert_eq!(local_degree, Some(2));
        assert_eq!(node_label.as_deref(), Some("hub"));
    } else { panic!("résultat Density local inattendu"); }
}

#[test]
fn coverage_with_provenance() {
    use gcn_ir::{Provenance, ExtractionMethod, SourceSpan};
    let ir = make_ir(
        vec![node(0,"X",NodeType::Action), node(1,"Y",NodeType::Etat), node(2,"Z",NodeType::Etat)],
        vec![
            {
                let (s,d,mut e) = edge(0,1,RelationType::Cause);
                e.provenance = Some(Provenance { doc_ref: None, span: SourceSpan::Synthetic,
                    extraction_method: ExtractionMethod::SymbolicRust,
                    model_version: "2.5.0".into(), extracted_at: "2026-09-26T00:00:00Z".into() });
                (s,d,e)
            },
            edge(1,2,RelationType::Cause),
        ],
    );
    let result = execute(&Query::Coverage("X".into()), &ir).unwrap();
    if let QueryResult::CoverageReport { degree, n_with_provenance, provenance_ratio, .. } = result {
        assert_eq!(degree, 1);
        assert_eq!(n_with_provenance, 1);
        assert!((provenance_ratio - 1.0).abs() < 0.001);
    } else { panic!("résultat Coverage inattendu"); }
}

#[test]
fn reliability_score() {
    let ir = make_ir(
        vec![node(0,"A",NodeType::Action), node(1,"B",NodeType::Etat)],
        vec![edge(0,1,RelationType::Cause)],
    );
    let result = execute(&Query::Reliability("A".into()), &ir).unwrap();
    if let QueryResult::ReliabilityReport { mean_confidence, reliability_score, provenance_ratio, .. } = result {
        assert!((mean_confidence - 0.9).abs() < 0.001);
        // provenance=None → ratio=0.0 → reliability=0
        assert!((provenance_ratio - 0.0).abs() < 0.001);
        assert!((reliability_score - 0.0).abs() < 0.001);
    } else { panic!("résultat Reliability inattendu"); }
}

// ─── Raisonnement abductif ────────────────────────────────────────────────────

#[test]
fn explain_ranks_direct_cause_first() {
    // A→C(conf=0.9), B→C(conf=0.5) : A doit avoir un score > B
    let ir = make_ir(
        vec![
            node(0, "A", NodeType::Action),
            node(1, "B", NodeType::Etat),
            node(2, "C", NodeType::Etat),
        ],
        vec![
            edge(0, 2, RelationType::Cause),
            edge(1, 2, RelationType::Enable),
        ],
    );
    // Override confidence for A→C to 0.9
    let mut ir2 = ir.clone();
    ir2.edges[0].2.confidence = 0.9;
    ir2.edges[1].2.confidence = 0.5;
    let result = execute(&Query::Explain("C".into()), &ir2).unwrap();
    if let QueryResult::Abduction { effect, hypotheses } = result {
        assert_eq!(effect, "C");
        assert!(!hypotheses.is_empty());
        // A (conf=0.9) doit être classé avant B (conf=0.5)
        let labels: Vec<&str> = hypotheses.iter().map(|h| h.label.as_str()).collect();
        let pos_a = labels.iter().position(|&l| l == "A");
        let pos_b = labels.iter().position(|&l| l == "B");
        assert!(pos_a < pos_b, "A (conf=0.9) doit précéder B (conf=0.5): {:?}", labels);
    } else {
        panic!("résultat EXPLAIN inattendu");
    }
}

#[test]
fn explain_no_causes_returns_empty() {
    let ir = make_ir(
        vec![node(0, "isolé", NodeType::Etat)],
        vec![],
    );
    let result = execute(&Query::Explain("isolé".into()), &ir).unwrap();
    if let QueryResult::Abduction { hypotheses, .. } = result {
        assert!(hypotheses.is_empty());
    } else {
        panic!("résultat EXPLAIN inattendu");
    }
}

#[test]
fn explain_prefers_close_over_distant() {
    // A→B(0.8), B→C(0.8) : EXPLAIN(C) doit classer B (depth=1) avant A (depth=2)
    // Ancien bug : ln(1+depth) favorisait A (0.8×ln(3)=0.88 > 0.8×ln(2)=0.55)
    let mut ir = make_ir(
        vec![
            node(0, "A", NodeType::Action),
            node(1, "B", NodeType::Etat),
            node(2, "C", NodeType::Etat),
        ],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 2, RelationType::Cause),
        ],
    );
    ir.edges[0].2.confidence = 0.8;
    ir.edges[1].2.confidence = 0.8;
    let result = execute(&Query::Explain("C".into()), &ir).unwrap();
    if let QueryResult::Abduction { hypotheses, .. } = result {
        assert!(hypotheses.len() >= 2);
        let pos_b = hypotheses.iter().position(|h| h.label == "B");
        let pos_a = hypotheses.iter().position(|h| h.label == "A");
        assert!(
            pos_b < pos_a,
            "B (depth=1) doit précéder A (depth=2): {:?}",
            hypotheses.iter().map(|h| (&h.label, h.depth, h.score)).collect::<Vec<_>>()
        );
        // B: 0.8/(1+1)=0.4, A: 0.8/(1+2)=0.267
        assert!(hypotheses[pos_b.unwrap()].score > hypotheses[pos_a.unwrap()].score);
    } else {
        panic!("résultat EXPLAIN inattendu");
    }
}

// ─── Raisonnement temporel ────────────────────────────────────────────────────

#[test]
fn chain_t_ordered_path() {
    let ir = make_ir(
        vec![
            node_with_ti(0, "A", NodeType::Action, 0),
            node_with_ti(1, "B", NodeType::Transition, 1),
            node_with_ti(2, "C", NodeType::Etat, 2),
        ],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 2, RelationType::Cause),
        ],
    );
    let result = execute(&Query::ChainT("A".into(), "C".into()), &ir).unwrap();
    if let QueryResult::TemporalPath { found, temporally_ordered, links, .. } = result {
        assert!(found, "chemin A→C doit exister");
        assert!(temporally_ordered, "ti=0→1→2 est ordonné");
        assert_eq!(links.len(), 2);
    } else {
        panic!("résultat ChainT inattendu");
    }
}

#[test]
fn chain_t_unordered_path() {
    let ir = make_ir(
        vec![
            node_with_ti(0, "A", NodeType::Action, 2),
            node_with_ti(1, "B", NodeType::Etat, 1),
        ],
        vec![edge(0, 1, RelationType::Cause)],
    );
    let result = execute(&Query::ChainT("A".into(), "B".into()), &ir).unwrap();
    if let QueryResult::TemporalPath { found, temporally_ordered, .. } = result {
        assert!(found);
        assert!(!temporally_ordered, "ti=2→1 est inversé");
    } else {
        panic!("résultat ChainT inattendu");
    }
}

#[test]
fn before_a_precedes_c() {
    let ir = make_ir(
        vec![
            node_with_ti(0, "A", NodeType::Action, 0),
            node_with_ti(1, "B", NodeType::Transition, 1),
            node_with_ti(2, "C", NodeType::Etat, 2),
        ],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 2, RelationType::Cause),
        ],
    );
    let result = execute(&Query::Before("A".into(), "C".into()), &ir).unwrap();
    if let QueryResult::TemporalOrder { a_before_b, path_exists, a_index, b_index, .. } = result {
        assert!(a_before_b);
        assert!(path_exists);
        assert_eq!(a_index, Some(0));
        assert_eq!(b_index, Some(2));
    } else {
        panic!("résultat Before inattendu");
    }
}

#[test]
fn delay_sums_temporal_gaps() {
    let ir = make_ir(
        vec![
            node_with_ti(0, "A", NodeType::Action, 0),
            node_with_ti(1, "B", NodeType::Transition, 1),
            node_with_ti(2, "C", NodeType::Etat, 2),
        ],
        vec![
            edge_with_gap(0, 1, RelationType::Cause, 3, 5),
            edge_with_gap(1, 2, RelationType::Cause, 2, 4),
        ],
    );
    let result = execute(&Query::Delay("A".into(), "C".into()), &ir).unwrap();
    if let QueryResult::TemporalDelay { found, gap_min_sum, gap_max_sum, index_delta, .. } = result {
        assert!(found);
        assert_eq!(gap_min_sum, Some(5));
        assert_eq!(gap_max_sum, Some(9));
        assert_eq!(index_delta, Some(2));
    } else {
        panic!("résultat Delay inattendu");
    }
}

// ── GCN-QL parse tests (21 queries + error cases) ─────────────────────

#[test]
fn parse_all_21_query_types() {
    let cases = vec![
        ("WHY feu", Query::Why("feu".into())),
        ("WHAT feu", Query::What("feu".into())),
        ("CHAIN feu -> fumée", Query::Chain("feu".into(), "fumée".into())),
        ("CYCLES", Query::Cycles),
        ("GAPS", Query::Gaps),
        ("DO feu", Query::Intervene("feu".into())),
        ("COUNTERFACTUAL feu", Query::Counterfactual("feu".into())),
        ("EXPLAIN fumée", Query::Explain("fumée".into())),
        ("ANALOGY feu -> fumée", Query::Analogy("feu".into(), "fumée".into())),
        ("ZOOM_IN feu", Query::ZoomIn("feu".into())),
        ("ZOOM_OUT feu", Query::ZoomOut("feu".into())),
        ("AGGREGATE feu", Query::Aggregate("feu".into())),
        ("CENTRALITY feu", Query::Centrality("feu".into())),
        ("SPOF", Query::Spof),
        ("DIFF feu -> fumée", Query::NormDiff("feu".into(), "fumée".into())),
        ("DENSITY", Query::Density(None)),
        ("DENSITY feu", Query::Density(Some("feu".into()))),
        ("COVERAGE feu", Query::Coverage("feu".into())),
        ("RELIABILITY feu", Query::Reliability("feu".into())),
        ("CHAIN_T feu -> fumée", Query::ChainT("feu".into(), "fumée".into())),
        ("BEFORE? feu -> fumée", Query::Before("feu".into(), "fumée".into())),
        ("BEFORE? feu, fumée", Query::Before("feu".into(), "fumée".into())),
        ("DELAY feu -> fumée", Query::Delay("feu".into(), "fumée".into())),
    ];
    for (input, expected) in cases {
        assert_eq!(Query::parse(input).unwrap(), expected, "failed for: {input}");
    }
}

#[test]
fn parse_case_insensitive_verb() {
    assert_eq!(Query::parse("why feu").unwrap(), Query::Why("feu".into()));
    assert_eq!(Query::parse("chain feu -> fumée").unwrap(), Query::Chain("feu".into(), "fumée".into()));
}

#[test]
fn parse_rejects_empty_args() {
    assert!(Query::parse("CHAIN  -> ").is_err());
    assert!(Query::parse("CHAIN -> fumée").is_err());
    assert!(Query::parse("CHAIN feu -> ").is_err());
    assert!(Query::parse("DIFF  -> ").is_err());
    assert!(Query::parse("ANALOGY -> ").is_err());
}

#[test]
fn parse_rejects_missing_separator() {
    assert!(Query::parse("CHAIN feu fumée").is_err());
    assert!(Query::parse("DELAY feu fumée").is_err());
}

#[test]
fn parse_unknown_query() {
    let err = Query::parse("FOOBAR xyz");
    assert!(err.is_err());
    let msg = format!("{}", err.unwrap_err());
    assert!(msg.contains("unknown query"));
}

#[test]
fn parse_comma_separator_for_pair_queries() {
    assert_eq!(
        Query::parse("CHAIN feu, fumée").unwrap(),
        Query::Chain("feu".into(), "fumée".into())
    );
    assert_eq!(
        Query::parse("DELAY feu, fumée").unwrap(),
        Query::Delay("feu".into(), "fumée".into())
    );
}
