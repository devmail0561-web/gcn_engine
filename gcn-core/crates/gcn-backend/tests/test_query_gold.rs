// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! Tests gold Ph1 (négation bloqueuse + tiers) — résultats attendus en dur,
//! incluant `blocked_by`.
//!
//! Phase 4 démarrée ici (fichier créé en P0) : diamant, chaîne niée, joint,
//! cycle, temporel désordonné, prevent. Properties + fuzz = suite de la phase 4.

use gcn_backend::LinkFilter;
use gcn_backend::pearl::{DEFAULT_AMBIGUITY_LIMIT, LabelSim, why_scoped_with};
use gcn_backend::query::DEFAULT_TOPK;
use gcn_backend::{BackendError, Query, QueryResult, chain_strict, execute};
use gcn_ir::{
    Ambiguity, AmbiguityCandidate, AmbiguousField, CausalEdge, CausalIR, CausalNode, IrMetadata,
    NaturalLanguage, NodeAttributes, NodeId, NodeOrigin, NodeType, RelationType, Scope,
    SourceLanguage, SourceSpan, TernaryRole, TernaryThird,
};
use gcn_middleend::process;
use smallvec::SmallVec;

fn node(id: u32, label: &str) -> CausalNode {
    CausalNode {
        id: NodeId(id),
        node_type: NodeType::Evenement,
        parent: None,
        label: label.to_string(),
        source_span: SourceSpan::Synthetic,
        scope: Scope::Unknown,
        modifiers: SmallVec::new(),
        temporal_ref: gcn_ir::TemporalRef::Unresolved,
        temporal_index: Some(id as i32),
        origin: NodeOrigin::Explicit,
        attributes: NodeAttributes::default(),
        kind: None,
    }
}

fn node_with_ti(id: u32, label: &str, ti: i32) -> CausalNode {
    let mut n = node(id, label);
    n.temporal_index = Some(ti);
    n
}

fn edge_full(
    src: u32,
    dst: u32,
    rel: RelationType,
    negated: bool,
    third: Option<TernaryThird>,
) -> (NodeId, NodeId, CausalEdge) {
    (
        NodeId(src),
        NodeId(dst),
        CausalEdge {
            relation: rel,
            confidence: 0.9,
            temporal_gap: None,
            explicit: true,
            negated,
            marker_token: None,
            in_cycle: None,
            provenance: None,
            derivation: None,
            joint_group_id: None,
            third,
        },
    )
}

fn edge(src: u32, dst: u32, rel: RelationType) -> (NodeId, NodeId, CausalEdge) {
    edge_full(src, dst, rel, false, None)
}

fn edge_conf(src: u32, dst: u32, rel: RelationType, conf: f32) -> (NodeId, NodeId, CausalEdge) {
    let (s, d, mut e) = edge(src, dst, rel);
    e.confidence = conf;
    (s, d, e)
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
            pipeline: vec!["test_query_gold".to_string()],
            created_at: None,
        },
        intent: None,
    };
    process(ir).unwrap().ir
}

/// Jouet Ph1 : Alpha → Beta (NIÉE), Beta → Gamma.
fn jouet_nie() -> CausalIR {
    make_ir(
        vec![node(0, "Alpha"), node(1, "Beta"), node(2, "Gamma")],
        vec![
            edge_full(0, 1, RelationType::Cause, true, None),
            edge(1, 2, RelationType::Cause),
        ],
    )
}

// ─── Ph1 [D1] : WHY(C) sans A + blocked_by ──────────────────────────────────

#[test]
fn negated_chain_why_excludes_cause_lists_blocked() {
    let ir = jouet_nie();
    let result = execute(&Query::parse("WHY Gamma").unwrap(), &ir).unwrap();
    if let QueryResult::Causes {
        links, blocked_by, ..
    } = result
    {
        // Beta → Gamma présent, rien depuis Alpha (propagation coupée).
        assert!(
            links.iter().any(|l| l.from == "Beta" && l.to == "Gamma"),
            "Beta→Gamma attendu, obtenu : {links:?}"
        );
        assert!(
            links.iter().all(|l| l.from != "Alpha"),
            "aucun lien depuis Alpha attendu, obtenu : {links:?}"
        );
        // La niée est listée, pas traversée.
        assert!(
            blocked_by
                .iter()
                .any(|l| l.from == "Alpha" && l.to == "Beta"),
            "blocked_by=[Alpha→Beta] attendu, obtenu : {blocked_by:?}"
        );
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
}

// ─── Ph1 [D1] : CHAIN found:false + blocked_by ───────────────────────────────

#[test]
fn negated_chain_path_found_false_with_blocked() {
    let ir = jouet_nie();
    let result = execute(&Query::parse("CHAIN Alpha -> Gamma").unwrap(), &ir).unwrap();
    if let QueryResult::Path {
        found,
        links,
        blocked_by,
        ..
    } = result
    {
        assert!(!found, "chemin coupé par négation → found:false");
        assert!(links.is_empty(), "aucun lien traversé, obtenu : {links:?}");
        assert!(
            blocked_by
                .iter()
                .any(|l| l.from == "Alpha" && l.to == "Beta"),
            "blocked_by=[Alpha→Beta] attendu, obtenu : {blocked_by:?}"
        );
    } else {
        panic!("Path attendu, obtenu : {result:?}");
    }
}

// ─── Vraiment aucun chemin : found:false + blocked_by vide ──────────────────

#[test]
fn disconnected_path_found_false_empty_blocked() {
    let ir = make_ir(
        vec![node(0, "Alpha"), node(1, "Beta"), node(2, "Zulu")],
        vec![edge(0, 1, RelationType::Cause)],
    );
    let result = execute(&Query::parse("CHAIN Alpha -> Zulu").unwrap(), &ir).unwrap();
    if let QueryResult::Path {
        found, blocked_by, ..
    } = result
    {
        assert!(!found);
        assert!(
            blocked_by.is_empty(),
            "vraiment aucun chemin → blocked_by vide, obtenu : {blocked_by:?}"
        );
    } else {
        panic!("Path attendu, obtenu : {result:?}");
    }
}

// ─── Ph1 [D1] : DO — severed inclut la niée, effects l'excluent ──────────────

#[test]
fn do_effects_exclude_negated_severed_includes() {
    let ir = jouet_nie();
    let result = execute(&Query::parse("DO Beta").unwrap(), &ir).unwrap();
    if let QueryResult::Intervention {
        severed,
        effects,
        blocked_by,
        ..
    } = result
    {
        // Coupée 2x = coupée : l'entrante niée figure dans severed.
        assert!(
            severed.iter().any(|l| l.from == "Alpha" && l.to == "Beta"),
            "severed contient Alpha→Beta, obtenu : {severed:?}"
        );
        // Effets : Beta → Gamma oui, rien via la niée.
        assert!(
            effects.iter().any(|l| l.from == "Beta" && l.to == "Gamma"),
            "effects contient Beta→Gamma, obtenu : {effects:?}"
        );
        assert!(
            effects.iter().all(|l| l.from != "Alpha"),
            "aucun effet via la niée, obtenu : {effects:?}"
        );
        let _ = blocked_by;
    } else {
        panic!("Intervention attendue, obtenue : {result:?}");
    }
}

// ─── Ph1 : COUNTERFACTUAL ne traverse pas la niée ────────────────────────────

#[test]
fn counterfactual_actual_effects_skip_negated() {
    let ir = jouet_nie();
    let result = execute(&Query::parse("COUNTERFACTUAL Alpha").unwrap(), &ir).unwrap();
    if let QueryResult::CounterfactualDiff {
        actual_effects,
        blocked_by,
        ..
    } = result
    {
        assert!(
            actual_effects.is_empty(),
            "rien en aval pur de Alpha, obtenu : {actual_effects:?}"
        );
        assert!(
            blocked_by
                .iter()
                .any(|l| l.from == "Alpha" && l.to == "Beta"),
            "blocked_by=[Alpha→Beta] attendu, obtenu : {blocked_by:?}"
        );
    } else {
        panic!("CounterfactualDiff attendu, obtenu : {result:?}");
    }
}

// ─── Ph1 : prevent = relation positive, se propage ──────────────────────────

#[test]
fn prevent_edge_propagates_normally() {
    let ir = make_ir(
        vec![node(0, "Alpha"), node(1, "Beta")],
        vec![edge(0, 1, RelationType::Prevent)],
    );
    let result = execute(&Query::parse("WHY Beta").unwrap(), &ir).unwrap();
    if let QueryResult::Causes { links, .. } = result {
        assert!(
            links
                .iter()
                .any(|l| l.from == "Alpha" && l.relation == "prevent"),
            "WHY(Beta) contient Alpha (prevent), obtenu : {links:?}"
        );
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
}

// ─── Diamant : propagation complète sans niée ───────────────────────────────

#[test]
fn diamond_why_and_chain() {
    let ir = make_ir(
        vec![
            node(0, "Alpha"),
            node(1, "Beta"),
            node(2, "Gamma"),
            node(3, "Delta"),
        ],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(0, 2, RelationType::Cause),
            edge(1, 3, RelationType::Cause),
            edge(2, 3, RelationType::Cause),
        ],
    );
    let result = execute(&Query::parse("WHY Delta").unwrap(), &ir).unwrap();
    if let QueryResult::Causes {
        links, blocked_by, ..
    } = result
    {
        // BFS = arbre couvrant : 3 liens (chaque ancêtre lié une fois).
        // Les 2 causes directes + Alpha via l'une d'elles, sans niée.
        assert_eq!(links.len(), 3, "diamant = 3 liens, obtenu : {links:?}");
        assert!(
            links.iter().any(|l| l.from == "Beta" && l.to == "Delta")
                && links.iter().any(|l| l.from == "Gamma" && l.to == "Delta"),
            "causes directes Beta+Gamma attendues, obtenu : {links:?}"
        );
        assert!(
            links.iter().any(|l| l.from == "Alpha"),
            "Alpha atteignable, obtenu : {links:?}"
        );
        assert!(blocked_by.is_empty());
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
    let result = execute(&Query::parse("CHAIN Alpha -> Delta").unwrap(), &ir).unwrap();
    if let QueryResult::Path { found, .. } = result {
        assert!(found);
    } else {
        panic!("Path attendu, obtenu : {result:?}");
    }
}

// ─── Joint A1 ∧ A2 → C : joint_group_id propagé ─────────────────────────────

#[test]
fn joint_group_id_propagated_in_why() {
    let mut e1 = edge(0, 2, RelationType::JointCause);
    e1.2.joint_group_id = Some("g1".to_string());
    let mut e2 = edge(1, 2, RelationType::JointCause);
    e2.2.joint_group_id = Some("g1".to_string());
    let ir = make_ir(
        vec![node(0, "Un"), node(1, "Deux"), node(2, "Cible")],
        vec![e1, e2],
    );
    let result = execute(&Query::parse("WHY Cible").unwrap(), &ir).unwrap();
    if let QueryResult::Causes { links, .. } = result {
        assert!(
            links
                .iter()
                .any(|l| l.joint_group_id.as_deref() == Some("g1")),
            "joint_group_id propagé, obtenu : {links:?}"
        );
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
}

// ─── Cycle : détecté par le middleend ────────────────────────────────────────

#[test]
fn cycle_detected_by_middleend() {
    let ir = make_ir(
        vec![node(0, "Alpha"), node(1, "Beta"), node(2, "Gamma")],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 2, RelationType::Cause),
            edge(2, 0, RelationType::Cause),
        ],
    );
    let result = execute(&Query::parse("CYCLES").unwrap(), &ir).unwrap();
    if let QueryResult::CycleList { count, .. } = result {
        assert_eq!(count, 1, "1 cycle attendu, obtenu : {result:?}");
    } else {
        panic!("CycleList attendue, obtenue : {result:?}");
    }
}

// ─── Temporel désordonné : chemin trouvé, ordre rejeté ──────────────────────

#[test]
fn temporal_disordered_path_found_but_not_ordered() {
    let ir = make_ir(
        vec![node_with_ti(0, "Alpha", 2), node_with_ti(1, "Beta", 1)],
        vec![edge(0, 1, RelationType::Cause)],
    );
    let result = execute(&Query::parse("CHAIN_T Alpha -> Beta").unwrap(), &ir).unwrap();
    if let QueryResult::TemporalPath {
        found,
        temporally_ordered,
        ..
    } = result
    {
        assert!(found, "le chemin causal existe");
        assert!(
            !temporally_ordered,
            "ordre temporel rejeté (2 > 1), obtenu : {result:?}"
        );
    } else {
        panic!("TemporalPath attendu, obtenu : {result:?}");
    }
}

// ─── Tiers Éq.12 : chain_strict exige third_node sur le chemin ───────────────

#[test]
fn chain_strict_rejects_third_off_path() {
    // aa → bb direct (ConditionalCause, tiers=mm) + détour aa → mm → bb.
    // Le plus court (direct) n'a pas mm sur le chemin → NoPath en strict.
    let third = TernaryThird {
        role: TernaryRole::Condition,
        node: 1,
        polarity: None,
    };
    let ir = make_ir(
        vec![node(0, "aa"), node(1, "mm"), node(2, "bb")],
        vec![
            edge_full(0, 2, RelationType::ConditionalCause, false, Some(third)),
            edge(0, 1, RelationType::Cause),
            edge(1, 2, RelationType::Cause),
        ],
    );
    // Non-strict : chemin trouvé (direct).
    let result = execute(&Query::parse("CHAIN aa -> bb").unwrap(), &ir).unwrap();
    if let QueryResult::Path { found, .. } = result {
        assert!(found);
    } else {
        panic!("Path attendu, obtenu : {result:?}");
    }
    // Strict : tiers mm hors du chemin direct → NoPath.
    assert!(
        chain_strict(&ir, "aa", "bb").is_err(),
        "tiers hors chemin → NoPath en strict"
    );
}

#[test]
fn chain_strict_accepts_third_on_path() {
    // aa → mm → bb, arête mm → bb ternaire (tiers=aa, sur le chemin).
    let third = TernaryThird {
        role: TernaryRole::Mediator,
        node: 0,
        polarity: None,
    };
    let ir = make_ir(
        vec![node(0, "aa"), node(1, "mm"), node(2, "bb")],
        vec![
            edge(0, 1, RelationType::Cause),
            edge_full(1, 2, RelationType::MediatedCause, false, Some(third)),
        ],
    );
    let links = chain_strict(&ir, "aa", "bb").expect("tiers aa sur le chemin → OK");
    assert_eq!(links.len(), 2);
    assert!(
        links
            .iter()
            .any(|l| l.third_node_label.as_deref() == Some("aa")),
        "lien du tiers annoté, obtenu : {links:?}"
    );
}

// ─── Ph2 [D2] : exact toujours répondu, flous ignorés ───────────────────────

#[test]
fn exact_match_wins_over_fuzzy_ignores_rest() {
    // vent (exact) + ventes/ventilation (flous) : seul vent répond.
    let ir = make_ir(
        vec![node(0, "vent"), node(1, "ventes"), node(2, "ventilation")],
        vec![],
    );
    let result = execute(&Query::parse("WHY vent").unwrap(), &ir).unwrap();
    if let QueryResult::Causes {
        target,
        matched_targets,
        match_scores,
        fuzzy,
        ..
    } = result
    {
        assert_eq!(target, "vent");
        assert_eq!(matched_targets, vec!["vent".to_string()]);
        assert_eq!(match_scores, vec![("vent".to_string(), 0u8)]);
        assert!(!fuzzy, "réponse exacte → pas de warning flou");
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
}

// ─── Ph2 [D2] : 7 flous > limite 5 → Err Ambiguous + 7 candidats ────────────

fn jouet_7_flous() -> CausalIR {
    let nodes: Vec<CausalNode> = (0..7)
        .map(|i| node(i, &format!("alpha{}", i + 1)))
        .collect();
    make_ir(nodes, vec![])
}

#[test]
fn fuzzy_overflow_errors_with_all_candidates() {
    let ir = jouet_7_flous();
    let err = execute(&Query::parse("WHY alpha").unwrap(), &ir);
    if let Err(BackendError::Ambiguous { candidates }) = err {
        assert_eq!(
            candidates.len(),
            7,
            "7 candidats attendus, obtenus : {candidates:?}"
        );
    } else {
        panic!("Ambiguous attendu, obtenu : {err:?}");
    }
}

// ─── Ph2 [D2] : 7 flous ≤ limite 7 → réponse + warning ──────────────────────

#[test]
fn fuzzy_within_limit_responds_with_warning() {
    let ir = jouet_7_flous();
    let result = execute(&Query::parse("WHY alpha AMBIGUITY_LIMIT=7").unwrap(), &ir).unwrap();
    if let QueryResult::Causes {
        matched_targets,
        match_scores,
        fuzzy,
        ..
    } = result
    {
        assert_eq!(matched_targets.len(), 7);
        assert!(
            match_scores.iter().all(|(_, s)| *s == 1u8),
            "tous flous score 1, obtenus : {match_scores:?}"
        );
        assert!(fuzzy, "réponse floue → warning");
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
}

// ─── Ph2 [D2] : 6 exacts → réponse normale, quota non déclenché ─────────────

#[test]
fn exact_duplicates_ignore_fuzzy_quota() {
    let nodes: Vec<CausalNode> = (0..6).map(|i| node(i, "ventes")).collect();
    let ir = make_ir(nodes, vec![]);
    let result = execute(&Query::parse("WHY ventes").unwrap(), &ir).unwrap();
    if let QueryResult::Causes {
        matched_targets,
        match_scores,
        fuzzy,
        ..
    } = result
    {
        assert_eq!(matched_targets.len(), 6);
        assert!(
            match_scores.iter().all(|(_, s)| *s == 0u8),
            "tous exacts, obtenus : {match_scores:?}"
        );
        assert!(!fuzzy);
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
}

// ─── Ph2 [D2] : parsing AMBIGUITY_LIMIT ─────────────────────────────────────

#[test]
fn parse_ambiguity_limit_suffix() {
    assert_eq!(
        Query::parse("WHY vent AMBIGUITY_LIMIT=3").unwrap(),
        Query::Why {
            label: "vent".into(),
            ambiguity_limit: 3,
            topk: None,
            filter: LinkFilter::default()
        }
    );
    assert_eq!(
        Query::parse("WHAT vent").unwrap(),
        Query::What {
            label: "vent".into(),
            ambiguity_limit: DEFAULT_AMBIGUITY_LIMIT,
            topk: None,
            filter: LinkFilter::default()
        }
    );
    assert!(Query::parse("WHY AMBIGUITY_LIMIT=3").is_err());
    assert!(Query::parse("WHY vent AMBIGUITY_LIMIT=x").is_err());
}

// ─── Ph2 [D2] : similarité injectable (LabelSim) ────────────────────────────

/// Similarité jouet : tout label commençant par 'k' est flou-1 pour "kw".
struct KSim;

impl LabelSim for KSim {
    fn score(&self, label: &str, query: &str) -> Option<u8> {
        if query == "kw" && label.starts_with('k') {
            Some(1)
        } else {
            None
        }
    }
}

#[test]
fn custom_label_sim_injects_fuzzy() {
    let ir = make_ir(vec![node(0, "kilo"), node(1, "autre")], vec![]);
    let scoped = why_scoped_with(&ir, "kw", &KSim, 5).unwrap();
    assert_eq!(scoped.matched.len(), 1);
    assert_eq!(scoped.matched[0].0, "kilo");
    assert!(scoped.fuzzy);
}

// ─── Ph3 [D3] : Dijkstra préfère confiance-max, pas plus-court ──────────────

/// A → B direct 0.9 vs A → X → B 0.5·0.5 : rank1 = direct, conf 0.9.
fn jouet_deux_chemins() -> CausalIR {
    make_ir(
        vec![node(0, "Alpha"), node(1, "Xi"), node(2, "Beta")],
        vec![
            edge_conf(0, 2, RelationType::Cause, 0.9),
            edge_conf(0, 1, RelationType::Cause, 0.5),
            edge_conf(1, 2, RelationType::Cause, 0.5),
        ],
    )
}

#[test]
fn dijkstra_ranks_max_confidence_first() {
    let ir = jouet_deux_chemins();
    let result = execute(&Query::parse("CHAIN Alpha -> Beta").unwrap(), &ir).unwrap();
    if let QueryResult::Path {
        found,
        links,
        path_confidence,
        ranked_paths,
        ..
    } = result
    {
        assert!(found);
        assert_eq!(links.len(), 1, "direct en rank1, obtenu : {links:?}");
        assert_eq!(links[0].from, "Alpha");
        assert!(
            (path_confidence - 0.9).abs() < 1e-6,
            "conf 0.9 attendue, obtenue : {path_confidence}"
        );
        assert_eq!(
            ranked_paths.len(),
            2,
            "TOPK défaut 3 → 2 chemins, obtenus : {ranked_paths:?}"
        );
        assert_eq!(ranked_paths[0].rank, 1);
        assert!((ranked_paths[0].path_confidence - 0.9).abs() < 1e-6);
        assert_eq!(ranked_paths[1].rank, 2);
        assert!((ranked_paths[1].path_confidence - 0.25).abs() < 1e-6);
        assert_eq!(ranked_paths[1].links.len(), 2);
    } else {
        panic!("Path attendu, obtenu : {result:?}");
    }
}

#[test]
fn topk_limits_paths_topk1_hides_ranked() {
    let ir = jouet_deux_chemins();
    let result = execute(&Query::parse("CHAIN Alpha -> Beta TOPK=2").unwrap(), &ir).unwrap();
    if let QueryResult::Path { ranked_paths, .. } = result {
        assert_eq!(ranked_paths.len(), 2);
        assert!(
            ranked_paths[0].path_confidence >= ranked_paths[1].path_confidence,
            "ordonnés, obtenus : {ranked_paths:?}"
        );
    } else {
        panic!("Path attendu, obtenu : {result:?}");
    }
    let result = execute(&Query::parse("CHAIN Alpha -> Beta TOPK=1").unwrap(), &ir).unwrap();
    if let QueryResult::Path {
        found,
        links,
        ranked_paths,
        ..
    } = result
    {
        assert!(found);
        assert_eq!(links.len(), 1);
        assert!(ranked_paths.is_empty(), "TOPK=1 → pas de ranked_paths");
    } else {
        panic!("Path attendu, obtenu : {result:?}");
    }
}

// ─── Ph3 [D3] : confiance-max bat plus-court (shift BFS→Dijkstra assumé) ────

#[test]
fn dijkstra_prefers_longer_stronger_path() {
    // Direct 0.1 vs détour 0.9·0.9=0.81 : le détour gagne (pas le plus court).
    let ir = make_ir(
        vec![node(0, "Alpha"), node(1, "Xi"), node(2, "Beta")],
        vec![
            edge_conf(0, 2, RelationType::Cause, 0.1),
            edge_conf(0, 1, RelationType::Cause, 0.9),
            edge_conf(1, 2, RelationType::Cause, 0.9),
        ],
    );
    let result = execute(&Query::parse("CHAIN Alpha -> Beta").unwrap(), &ir).unwrap();
    if let QueryResult::Path {
        links,
        path_confidence,
        ..
    } = result
    {
        assert_eq!(links.len(), 2, "détour fort attendu, obtenu : {links:?}");
        assert!((path_confidence - 0.81).abs() < 1e-6);
    } else {
        panic!("Path attendu, obtenu : {result:?}");
    }
}

// ─── Ph3 [D3] : WHY trié par force (produit depuis la cible) ────────────────

#[test]
fn why_sorted_by_strength_from_target() {
    // Tgt ← A 0.9 ; Tgt ← B 0.5 ; B ← C 0.9 (force 0.45 via B).
    let ir = make_ir(
        vec![
            node(0, "Tgt"),
            node(1, "Aaa"),
            node(2, "Bbb"),
            node(3, "Ccc"),
        ],
        vec![
            edge_conf(1, 0, RelationType::Cause, 0.9),
            edge_conf(2, 0, RelationType::Cause, 0.5),
            edge_conf(3, 2, RelationType::Cause, 0.9),
        ],
    );
    let result = execute(&Query::parse("WHY Tgt").unwrap(), &ir).unwrap();
    if let QueryResult::Causes { links, .. } = result {
        let order: Vec<&str> = links.iter().map(|l| l.from.as_str()).collect();
        assert_eq!(
            order,
            vec!["Aaa", "Bbb", "Ccc"],
            "ordre 0.9 > 0.5 > 0.45 attendu, obtenu : {links:?}"
        );
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
}

// ─── Ph3 [D3] : ANALOGY paramétrable + parsing ───────────────────────────────

#[test]
fn analogy_custom_min_score_filters() {
    // 2 arêtes cause 0.9 identiques (types + degrés) : score ~1.0.
    let ir = make_ir(
        vec![node(0, "Pp"), node(1, "Qq"), node(2, "Cc"), node(3, "Dd")],
        vec![
            edge_conf(0, 1, RelationType::Cause, 0.9),
            edge_conf(2, 3, RelationType::Cause, 0.9),
        ],
    );
    let result = execute(&Query::parse("ANALOGY Pp -> Qq").unwrap(), &ir).unwrap();
    if let QueryResult::AnalogyReport { matches, .. } = result {
        assert!(!matches.is_empty(), "défauts → analogue trouvé");
    } else {
        panic!("AnalogyReport attendu, obtenu : {result:?}");
    }
    // Seuil impossible (> 1.0) → rien (poids inchangés).
    let result = execute(
        &Query::parse("ANALOGY Pp -> Qq MIN_SCORE=1.01").unwrap(),
        &ir,
    )
    .unwrap();
    if let QueryResult::AnalogyReport { matches, .. } = result {
        assert!(
            matches.is_empty(),
            "seuil 1.01 → rien, obtenu : {matches:?}"
        );
    } else {
        panic!("AnalogyReport attendu, obtenu : {result:?}");
    }
}

#[test]
fn parse_topk_weights_minscore() {
    assert_eq!(
        Query::parse("CHAIN a -> b TOPK=2").unwrap(),
        Query::Chain {
            from: "a".into(),
            to: "b".into(),
            topk: 2,
            filter: LinkFilter::default()
        }
    );
    assert_eq!(
        Query::parse("CHAIN a -> b").unwrap(),
        Query::Chain {
            from: "a".into(),
            to: "b".into(),
            topk: DEFAULT_TOPK,
            filter: LinkFilter::default()
        }
    );
    assert!(Query::parse("CHAIN a -> b TOPK=0").is_err());
    assert!(Query::parse("CHAIN a -> b TOPK=x").is_err());
    assert_eq!(
        Query::parse("ANALOGY a -> b WEIGHTS=0.4,0.3,0.2,0.1 MIN_SCORE=0.5").unwrap(),
        Query::Analogy {
            from: "a".into(),
            to: "b".into(),
            weights: [0.4, 0.3, 0.2, 0.1],
            min_score: 0.5,
            hops: 1
        }
    );
    assert!(Query::parse("ANALOGY a -> b WEIGHTS=0.4,0.3").is_err());
}

// ─── Ph6b : ASK — fiche nœud 1 saut ─────────────────────────────────────────

#[test]
fn ask_returns_node_card() {
    let ir = make_ir(
        vec![node(0, "Alpha"), node(1, "Beta"), node(2, "Gamma")],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 2, RelationType::Cause),
        ],
    );
    let result = execute(&Query::parse("ASK Beta").unwrap(), &ir).unwrap();
    if let QueryResult::AskReport {
        label,
        node_type,
        causes,
        effects,
    } = result
    {
        assert_eq!(label, "Beta");
        assert!(!node_type.is_empty());
        assert_eq!(causes.len(), 1);
        assert_eq!(causes[0].from, "Alpha");
        assert_eq!(effects.len(), 1);
        assert_eq!(effects[0].to, "Gamma");
    } else {
        panic!("AskReport attendu, obtenu : {result:?}");
    }
    let err = execute(&Query::parse("ASK Zzz").unwrap(), &ir);
    assert!(matches!(err, Err(BackendError::NodeNotFound(_))));
    assert!(Query::parse("ASK ").is_err());
}

// ─── Ph6b : WHY/WHAT TOPK — N meilleurs + total ─────────────────────────────

#[test]
fn why_topk_limits_links_reports_total() {
    // Chaîne Alpha→Beta→Gamma→Delta : WHY(Delta) = 3 liens (force décroissante).
    let ir = make_ir(
        vec![
            node(0, "Alpha"),
            node(1, "Beta"),
            node(2, "Gamma"),
            node(3, "Delta"),
        ],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 2, RelationType::Cause),
            edge(2, 3, RelationType::Cause),
        ],
    );
    let result = execute(&Query::parse("WHY Delta TOPK=2").unwrap(), &ir).unwrap();
    if let QueryResult::Causes { links, total, .. } = result {
        assert_eq!(total, 3, "total avant limite : {links:?}");
        assert_eq!(links.len(), 2, "TOPK=2 : {links:?}");
        assert_eq!(links[0].from, "Gamma", "le plus fort d'abord : {links:?}");
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
    // Sans TOPK : tout + total == len.
    let result = execute(&Query::parse("WHY Delta").unwrap(), &ir).unwrap();
    if let QueryResult::Causes { links, total, .. } = result {
        assert_eq!(links.len(), total);
        assert_eq!(total, 3);
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
    assert!(Query::parse("WHY Delta TOPK=0").is_err());
}

// ─── Ph6b : WHY-NOT — coupure, aucun lien, chemin existant ──────────────────

#[test]
fn why_not_negation_cut() {
    // Alpha → Beta NIÉE, Beta → Gamma : WHY-NOT = coupure + bloqueurs.
    let ir = make_ir(
        vec![node(0, "Alpha"), node(1, "Beta"), node(2, "Gamma")],
        vec![
            edge_full(0, 1, RelationType::Cause, true, None),
            edge(1, 2, RelationType::Cause),
        ],
    );
    let result = execute(&Query::parse("WHY-NOT Alpha -> Gamma").unwrap(), &ir).unwrap();
    if let QueryResult::WhyNot {
        verdict,
        blocked_by,
        proches,
        ..
    } = result
    {
        assert_eq!(verdict, "coupure_negation");
        assert!(
            blocked_by
                .iter()
                .any(|l| l.from == "Alpha" && l.to == "Beta"),
            "bloqueurs : {blocked_by:?}"
        );
        assert!(proches.is_empty());
    } else {
        panic!("WhyNot attendu, obtenu : {result:?}");
    }
}

#[test]
fn why_not_no_link_gives_proches() {
    let ir = make_ir(
        vec![node(0, "Alpha"), node(1, "Beta"), node(2, "Zulu")],
        vec![edge(0, 1, RelationType::Cause)],
    );
    let result = execute(&Query::parse("WHY-NOT Alpha -> Zulu").unwrap(), &ir).unwrap();
    if let QueryResult::WhyNot {
        verdict, proches, ..
    } = result
    {
        assert_eq!(verdict, "aucun_lien");
        assert!(
            proches.contains(&"Beta".to_string()),
            "proches : {proches:?}"
        );
    } else {
        panic!("WhyNot attendu, obtenu : {result:?}");
    }
}

#[test]
fn why_not_existing_path_says_so() {
    let ir = make_ir(
        vec![node(0, "Alpha"), node(1, "Beta")],
        vec![edge(0, 1, RelationType::Cause)],
    );
    let result = execute(&Query::parse("WHY-NOT Alpha -> Beta").unwrap(), &ir).unwrap();
    if let QueryResult::WhyNot { verdict, links, .. } = result {
        assert_eq!(verdict, "chemin_existe");
        assert_eq!(links.len(), 1);
    } else {
        panic!("WhyNot attendu, obtenu : {result:?}");
    }
    assert!(Query::parse("WHY-NOT Alpha").is_err());
}

// ─── Ph6b : EXPLAIN CONTRAST — distinctif vs partagé ────────────────────────

#[test]
fn explain_contrast_distinctive_vs_shared() {
    // E ← {A, B} ; F ← {B, C} : distinctif=[A], partagé=[B].
    let ir = make_ir(
        vec![
            node(0, "Aaa"),
            node(1, "Bbb"),
            node(2, "Eee"),
            node(3, "Fff"),
            node(4, "Ccc"),
        ],
        vec![
            edge(0, 2, RelationType::Cause),
            edge(1, 2, RelationType::Cause),
            edge(1, 3, RelationType::Cause),
            edge(4, 3, RelationType::Cause),
        ],
    );
    let result = execute(&Query::parse("EXPLAIN Eee CONTRAST Fff").unwrap(), &ir).unwrap();
    if let QueryResult::Abduction {
        hypotheses,
        contrast,
        ..
    } = result
    {
        assert_eq!(hypotheses.len(), 2);
        let c = contrast.expect("contraste présent");
        assert_eq!(c.alternative, "Fff");
        let dist: Vec<&str> = c.distinctive.iter().map(|h| h.label.as_str()).collect();
        assert_eq!(dist, vec!["Aaa"], "distinctif : {dist:?}");
        assert_eq!(
            c.shared,
            vec!["Bbb".to_string()],
            "partagé : {:?}",
            c.shared
        );
    } else {
        panic!("Abduction attendue, obtenue : {result:?}");
    }
    // Alternative inconnue → NodeNotFound (référence explicite).
    let err = execute(&Query::parse("EXPLAIN Eee CONTRAST Zzz").unwrap(), &ir);
    assert!(matches!(err, Err(BackendError::NodeNotFound(_))));
    // Sans contraste : inchangé (contrast None).
    let result = execute(&Query::parse("EXPLAIN Eee").unwrap(), &ir).unwrap();
    if let QueryResult::Abduction { contrast, .. } = result {
        assert!(contrast.is_none());
    } else {
        panic!("Abduction attendue, obtenue : {result:?}");
    }
}

// ─── Ph6b : ANALOGY HOPS — contexte multi-sauts ────────────────────────────

#[test]
fn analogy_hops_runs_and_parses() {
    // A→B patron, C→D analogue, E→F témoin (cf. test ANALOGY existant).
    let ir = make_ir(
        vec![
            node(0, "A"),
            node(1, "B"),
            node(2, "C"),
            node(3, "D"),
            node(4, "E"),
            node(5, "F"),
        ],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(2, 3, RelationType::Cause),
            edge(4, 5, RelationType::Enable),
        ],
    );
    let result = execute(&Query::parse("ANALOGY A -> B HOPS=2").unwrap(), &ir).unwrap();
    if let QueryResult::AnalogyReport { matches, .. } = result {
        assert!(
            matches.iter().all(|m| m.similarity_score >= 0.3),
            "seuil respecté : {matches:?}"
        );
        // Déterminisme : deux runs identiques.
        let result2 = execute(&Query::parse("ANALOGY A -> B HOPS=2").unwrap(), &ir).unwrap();
        if let QueryResult::AnalogyReport { matches: m2, .. } = result2 {
            assert_eq!(matches.len(), m2.len());
        } else {
            panic!("AnalogyReport attendu");
        }
    } else {
        panic!("AnalogyReport attendu, obtenu : {result:?}");
    }
    assert!(Query::parse("ANALOGY A -> B HOPS=0").is_err());
}

// ─── Ph6b : dérivations D3 systématiques sur les liens ─────────────────────

#[test]
fn derived_from_surfaced_on_links() {
    // A→B inférée de X→A (Derivation.source_edges).
    let mut e = edge(0, 1, RelationType::Cause);
    e.2.derivation = Some(gcn_ir::Derivation {
        source_edges: vec![(NodeId(2), NodeId(0))],
    });
    let ir = make_ir(
        vec![node(0, "Aaa"), node(1, "Bbb"), node(2, "Xxx")],
        vec![edge(2, 0, RelationType::Cause), e],
    );
    let result = execute(&Query::parse("WHY Bbb").unwrap(), &ir).unwrap();
    if let QueryResult::Causes { links, .. } = result {
        let lien = links
            .iter()
            .find(|l| l.from == "Aaa")
            .expect("lien Aaa→Bbb");
        assert_eq!(
            lien.derived_from,
            Some(vec![("Xxx".to_string(), "Aaa".to_string())]),
            "dérivation D3 : {lien:?}"
        );
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
}

// ─── Ph7a : opposition = bloqueuse, motif causal_gap ────────────────────────

#[test]
fn opposition_blocks_with_causal_gap_motif() {
    // Alpha → Beta (opposition), Beta → Gamma : WHY(Gamma) sans Alpha.
    let ir = make_ir(
        vec![node(0, "Alpha"), node(1, "Beta"), node(2, "Gamma")],
        vec![
            edge(0, 1, RelationType::Opposition),
            edge(1, 2, RelationType::Cause),
        ],
    );
    let result = execute(&Query::parse("WHY Gamma").unwrap(), &ir).unwrap();
    if let QueryResult::Causes {
        links, blocked_by, ..
    } = result
    {
        assert!(
            links.iter().any(|l| l.from == "Beta" && l.to == "Gamma"),
            "Beta→Gamma : {links:?}"
        );
        assert!(links.iter().all(|l| l.from != "Alpha"), "coupé : {links:?}");
        let bloque = blocked_by
            .iter()
            .find(|l| l.from == "Alpha" && l.to == "Beta")
            .expect("blocked_by=[Alpha→Beta]");
        assert_eq!(bloque.motif.as_deref(), Some("causal_gap"));
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
    // CHAIN coupée pareil, motif distinct de negated.
    let result = execute(&Query::parse("CHAIN Alpha -> Gamma").unwrap(), &ir).unwrap();
    if let QueryResult::Path {
        found, blocked_by, ..
    } = result
    {
        assert!(!found);
        assert_eq!(blocked_by.len(), 1);
        assert_eq!(blocked_by[0].motif.as_deref(), Some("causal_gap"));
    } else {
        panic!("Path attendu, obtenu : {result:?}");
    }
}

// ─── Ph7a : feedback négatif = damped ───────────────────────────────────────

#[test]
fn feedback_negative_links_annotated_damped() {
    // A ⇄ B avec prevent : cycle feedback-negative (classify).
    let ir = make_ir(
        vec![node(0, "Alpha"), node(1, "Beta")],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 0, RelationType::Prevent),
        ],
    );
    let result = execute(&Query::parse("CYCLES").unwrap(), &ir).unwrap();
    if let QueryResult::CycleList { cycles, .. } = result {
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0].kind, "feedback_negative");
        assert_eq!(cycles[0].effect.as_deref(), Some("damped"));
    } else {
        panic!("CycleList attendue, obtenue : {result:?}");
    }
    let result = execute(&Query::parse("WHY Beta").unwrap(), &ir).unwrap();
    if let QueryResult::Causes { links, .. } = result {
        assert!(
            links
                .iter()
                .any(|l| l.cycle_effect.as_deref() == Some("damped")),
            "lien damped : {links:?}"
        );
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
}

// ─── Ph7a : CHAIN_T exige la phase sur oscillation ──────────────────────────

fn edge_avec_gap(
    src: u32,
    dst: u32,
    rel: RelationType,
    min: i32,
    max: i32,
) -> (NodeId, NodeId, CausalEdge) {
    let (s, d, mut e) = edge(src, dst, rel);
    e.temporal_gap = Some(gcn_ir::TemporalGap {
        min: Some(min),
        max: Some(max),
        nature: gcn_ir::GapNature::Deferred,
    });
    (s, d, e)
}

#[test]
fn chain_t_rejects_unphased_oscillation() {
    // Triangle A→B→C→A avec C→A concession : cycle OSCILLATION (classify).
    // Chemin A→B→C pur (Dijkstra) mais sans phase → CHAIN_T rejette.
    let ir = make_ir(
        vec![node(0, "Alpha"), node(1, "Beta"), node(2, "Gamma")],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 2, RelationType::Cause),
            edge(2, 0, RelationType::Concession),
        ],
    );
    let result = execute(&Query::parse("CHAIN_T Alpha -> Gamma").unwrap(), &ir).unwrap();
    if let QueryResult::TemporalPath { found, .. } = result {
        assert!(!found, "oscillation sans phase → rejeté : {result:?}");
    } else {
        panic!("TemporalPath attendu, obtenu : {result:?}");
    }
    // Avec phases sur le chemin : accepté.
    let ir2 = make_ir(
        vec![node(0, "Alpha"), node(1, "Beta"), node(2, "Gamma")],
        vec![
            edge_avec_gap(0, 1, RelationType::Cause, 0, 1),
            edge_avec_gap(1, 2, RelationType::Cause, 0, 1),
            edge(2, 0, RelationType::Concession),
        ],
    );
    let result = execute(&Query::parse("CHAIN_T Alpha -> Gamma").unwrap(), &ir2).unwrap();
    if let QueryResult::TemporalPath { found, .. } = result {
        assert!(found, "oscillation phasée → accepté : {result:?}");
    } else {
        panic!("TemporalPath attendu, obtenu : {result:?}");
    }
}

// ─── Ph7b : origine — attesté 0.6 > hypothétique 0.9×0.3 ────────────────────

fn noeud_origine(id: u32, label: &str, origin: NodeOrigin) -> CausalNode {
    let mut n = node(id, label);
    n.origin = origin;
    n.scope = Scope::Universal; // pas de caveat scope ici (test 7b pur)
    n
}

#[test]
fn origin_weight_ranks_attested_first() {
    // Doc(Explicit)→Tgt 0.6 vs Hyp(Hypothetical)→Tgt 0.9·0.3=0.27.
    let ir = make_ir(
        vec![
            noeud_origine(0, "Doc", NodeOrigin::Explicit),
            noeud_origine(1, "Hyp", NodeOrigin::Hypothetical),
            noeud_origine(2, "Tgt", NodeOrigin::Explicit),
        ],
        vec![
            edge_conf(0, 2, RelationType::Cause, 0.6),
            edge_conf(1, 2, RelationType::Cause, 0.9),
        ],
    );
    let result = execute(&Query::parse("WHY Tgt").unwrap(), &ir).unwrap();
    if let QueryResult::Causes { links, .. } = result {
        assert_eq!(links.len(), 2);
        assert_eq!(links[0].from, "Doc", "attesté d'abord : {links:?}");
        assert_eq!(links[0].origin.as_deref(), Some("explicit"));
        assert_eq!(links[1].origin.as_deref(), Some("hypothetical"));
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
}

// ─── Ph7b : caveats hypothétique + DOUBTS ───────────────────────────────────

#[test]
fn caveats_hypothetical_and_doubts() {
    let mut ir = make_ir(
        vec![
            noeud_origine(0, "Doc", NodeOrigin::Explicit),
            noeud_origine(1, "Hyp", NodeOrigin::Hypothetical),
            noeud_origine(2, "Tgt", NodeOrigin::Explicit),
        ],
        vec![
            edge_conf(0, 2, RelationType::Cause, 0.6),
            edge_conf(1, 2, RelationType::Cause, 0.9),
        ],
    );
    ir.unresolved.push(Ambiguity {
        node_id: NodeId(2),
        field: AmbiguousField::NodeType,
        candidates: vec![
            AmbiguityCandidate {
                value: "processus".to_string(),
                confidence: 0.6,
                rule_source: "test".to_string(),
            },
            AmbiguityCandidate {
                value: "evenement".to_string(),
                confidence: 0.4,
                rule_source: "test".to_string(),
            },
        ],
        context_hint: Some("contexte de test".to_string()),
    });
    let result = execute(&Query::parse("WHY Tgt").unwrap(), &ir).unwrap();
    if let QueryResult::Causes { caveats, .. } = result {
        assert!(
            caveats
                .iter()
                .any(|c| c.node == "Hyp" && c.field == "origin"),
            "caveat origine : {caveats:?}"
        );
        let amb = caveats
            .iter()
            .find(|c| c.node == "Tgt")
            .expect("caveat Tgt");
        assert_eq!(amb.field, "node_type");
        assert_eq!(amb.candidates.len(), 2);
        assert_eq!(amb.hint, "contexte de test");
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
    let result = execute(&Query::parse("DOUBTS").unwrap(), &ir).unwrap();
    if let QueryResult::Doubts { count, doubts } = result {
        assert_eq!(count, 1);
        assert_eq!(doubts[0].node, "Tgt");
        assert_eq!(doubts[0].candidates.len(), 2);
    } else {
        panic!("Doubts attendu, obtenu : {result:?}");
    }
    assert!(Query::parse("doubts").is_ok(), "insensible à la casse");
}

// ─── Ph7 : filtres de liens (construits directs, parsing en 7d/7e/7f) ───────

fn requete_filtree(filtre: gcn_backend::LinkFilter) -> Query {
    Query::Why {
        label: "Tgt".into(),
        ambiguity_limit: DEFAULT_AMBIGUITY_LIMIT,
        topk: None,
        filter: filtre,
    }
}

#[test]
fn filter_scope_agent_explicit_method_since() {
    use gcn_ir::{ExtractionMethod, Provenance, SourceSpan as Span};
    let mut n0 = noeud_origine(0, "Doc", NodeOrigin::Explicit);
    n0.scope = Scope::Universal;
    n0.attributes.agent_type = Some(gcn_ir::AgentType::Human);
    let mut n1 = noeud_origine(1, "Tgt", NodeOrigin::Explicit);
    n1.scope = Scope::Universal;
    let mut e = edge(0, 1, RelationType::Cause);
    e.2.provenance = Some(Provenance {
        doc_ref: Some("doc".to_string()),
        span: Span::Synthetic,
        extraction_method: ExtractionMethod::SilverAnnotator,
        model_version: "m1".to_string(),
        extracted_at: "2026-10-01T00:00:00Z".to_string(),
    });
    let ir = make_ir(vec![n0, n1], vec![e]);
    let compte =
        |filtre: gcn_backend::LinkFilter| match execute(&requete_filtree(filtre), &ir).unwrap() {
            QueryResult::Causes { links, .. } => links.len(),
            r => panic!("Causes attendu : {r:?}"),
        };
    assert_eq!(
        compte(gcn_backend::LinkFilter {
            scope: Some("universal".to_string()),
            ..Default::default()
        }),
        1
    );
    assert_eq!(
        compte(gcn_backend::LinkFilter {
            scope: Some("null".to_string()),
            ..Default::default()
        }),
        0
    );
    assert_eq!(
        compte(gcn_backend::LinkFilter {
            agent: Some("human".to_string()),
            ..Default::default()
        }),
        1
    );
    assert_eq!(
        compte(gcn_backend::LinkFilter {
            agent: Some("collective".to_string()),
            ..Default::default()
        }),
        0
    );
    assert_eq!(
        compte(gcn_backend::LinkFilter {
            only_explicit: true,
            ..Default::default()
        }),
        1
    );
    assert_eq!(
        compte(gcn_backend::LinkFilter {
            method: Some("silver".to_string()),
            ..Default::default()
        }),
        1
    );
    assert_eq!(
        compte(gcn_backend::LinkFilter {
            method: Some("ml".to_string()),
            ..Default::default()
        }),
        0
    );
    assert_eq!(
        compte(gcn_backend::LinkFilter {
            since: Some("2026-01-01".to_string()),
            ..Default::default()
        }),
        1
    );
    assert_eq!(
        compte(gcn_backend::LinkFilter {
            since: Some("2027-01-01".to_string()),
            ..Default::default()
        }),
        0
    );
}

// ─── Ph7f : provenance sur les liens + PROVENANCE ───────────────────────────

fn ir_provenance() -> CausalIR {
    use gcn_ir::{ExtractionMethod, Provenance, SourceSpan as Span};
    let mut ir = make_ir(
        vec![node(0, "Pluie"), node(1, "Flaque")],
        vec![edge(0, 1, RelationType::Cause)],
    );
    ir.source_text = "La pluie cause la flaque.".to_string();
    if let Some((_, _, e)) = ir
        .edges
        .iter_mut()
        .find(|(s, d, _)| *s == NodeId(0) && *d == NodeId(1))
    {
        e.provenance = Some(Provenance {
            doc_ref: Some("doc1".to_string()),
            span: Span::TokenSpan { start: 3, end: 8 },
            extraction_method: ExtractionMethod::SilverAnnotator,
            model_version: "b1".to_string(),
            extracted_at: "2026-10-01T00:00:00Z".to_string(),
        });
    }
    // Re-middlend : les cycles/in_cycle recalculés (ici : aucun).
    process(ir).unwrap().ir
}

#[test]
fn provenance_fields_on_links() {
    let ir = ir_provenance();
    let result = execute(&Query::parse("WHY Flaque").unwrap(), &ir).unwrap();
    if let QueryResult::Causes { links, .. } = result {
        assert_eq!(links.len(), 1);
        let l = &links[0];
        assert!(l.explicit);
        assert_eq!(l.extraction_method.as_deref(), Some("silver"));
        assert_eq!(l.model_version.as_deref(), Some("b1"));
        assert_eq!(l.span.as_deref(), Some("pluie"), "span texte : {l:?}");
        assert_eq!(l.kind.as_deref(), Some("text"));
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
}

#[test]
fn provenance_report_and_filters() {
    let ir = ir_provenance();
    let result = execute(&Query::parse("PROVENANCE Flaque").unwrap(), &ir).unwrap();
    if let QueryResult::ProvenanceReport { label, entries } = result {
        assert_eq!(label, "Flaque");
        assert_eq!(entries.len(), 1);
        let e = &entries[0];
        assert_eq!(e.from, "Pluie");
        assert_eq!(e.extraction_method.as_deref(), Some("silver"));
        assert_eq!(e.model_version.as_deref(), Some("b1"));
        assert_eq!(e.span.as_deref(), Some("pluie"));
        assert_eq!(e.doc_ref.as_deref(), Some("doc1"));
        assert_eq!(e.extracted_at.as_deref(), Some("2026-10-01T00:00:00Z"));
    } else {
        panic!("ProvenanceReport attendu, obtenu : {result:?}");
    }
    // Filtres parsés : ONLY_EXPLICIT / METHOD / SINCE gardent ce lien silver.
    for q in [
        "WHY Flaque ONLY_EXPLICIT",
        "WHY Flaque METHOD=silver",
        "WHY Flaque SINCE=2026-01-01",
        "WHY Flaque METHOD=silver SINCE=2020-01-01 ONLY_EXPLICIT",
    ] {
        let result = execute(&Query::parse(q).unwrap(), &ir).unwrap();
        if let QueryResult::Causes { links, .. } = result {
            assert_eq!(links.len(), 1, "{q}");
        } else {
            panic!("Causes attendu ({q}) : {result:?}");
        }
    }
    // ... et l'éliminent hors critères.
    for q in ["WHY Flaque METHOD=ml", "WHY Flaque SINCE=2027-01-01"] {
        let result = execute(&Query::parse(q).unwrap(), &ir).unwrap();
        if let QueryResult::Causes { links, .. } = result {
            assert!(links.is_empty(), "{q} : {links:?}");
        } else {
            panic!("Causes attendu ({q}) : {result:?}");
        }
    }
    // Valeurs invalides → erreur typée (jamais silencieux).
    for q in [
        "WHY Flaque METHOD=unknown",
        "WHY Flaque SINCE=demain",
        "WHY Flaque SCOPE=nimporte",
        "WHY Flaque AGENT=martien",
        "PROVENANCE",
    ] {
        assert!(Query::parse(q).is_err(), "{q}");
    }
    assert!(Query::parse("PROVENANCE Flaque").is_ok());
    let err = execute(&Query::parse("PROVENANCE Zzz").unwrap(), &ir);
    assert!(matches!(err, Err(BackendError::NodeNotFound(_))));
}

// ─── Ph7d : DO irréversible → warning + effets quand même ───────────────────

#[test]
fn do_irreversible_warns_but_computes() {
    let mut tgt = node(1, "Cible");
    tgt.attributes.reversible = Some(false);
    let ir = make_ir(
        vec![node(0, "Cause"), tgt, node(2, "Effet")],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 2, RelationType::Cause),
        ],
    );
    let result = execute(&Query::parse("DO Cible").unwrap(), &ir).unwrap();
    if let QueryResult::Intervention {
        warnings, effects, ..
    } = result
    {
        assert_eq!(warnings, vec!["irreversible".to_string()], "warning FIGÉ");
        assert!(
            effects.iter().any(|l| l.to == "Effet"),
            "effets calculés quand même : {effects:?}"
        );
    } else {
        panic!("Intervention attendue, obtenue : {result:?}");
    }
    // Cible réversible (ou non renseignée) → pas de warning.
    let ir2 = make_ir(
        vec![node(0, "Cause"), node(1, "Cible"), node(2, "Effet")],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 2, RelationType::Cause),
        ],
    );
    let result = execute(&Query::parse("DO Cible").unwrap(), &ir2).unwrap();
    if let QueryResult::Intervention { warnings, .. } = result {
        assert!(warnings.is_empty(), "pas de warning : {warnings:?}");
    } else {
        panic!("Intervention attendue, obtenue : {result:?}");
    }
}

// ─── Ph7d : agent/patient/type exposés + filtre AGENT ───────────────────────

#[test]
fn agent_fields_and_filter() {
    let mut src = node(0, "Docteur");
    src.attributes.agent = Some("docteur".to_string());
    src.attributes.agent_type = Some(gcn_ir::AgentType::Human);
    let mut dst = node(1, "Patient");
    dst.attributes.patient = Some("patient".to_string());
    let ir = make_ir(vec![src, dst], vec![edge(0, 1, RelationType::Cause)]);
    let result = execute(&Query::parse("WHY Patient").unwrap(), &ir).unwrap();
    if let QueryResult::Causes { links, .. } = result {
        assert_eq!(links.len(), 1);
        let l = &links[0];
        assert_eq!(l.agent.as_deref(), Some("docteur"));
        assert_eq!(l.patient.as_deref(), Some("patient"));
        assert_eq!(l.agent_type.as_deref(), Some("human"));
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
    // Filtre AGENT=human garde, AGENT=collective élimine.
    let result = execute(&Query::parse("DO Patient AGENT=human").unwrap(), &ir).unwrap();
    if let QueryResult::Intervention { effects, .. } = result {
        assert!(
            effects.is_empty(),
            "DO Patient : pas d'effet (cible), {effects:?}"
        );
    } else {
        panic!("Intervention attendue, obtenue : {result:?}");
    }
    let result = execute(&Query::parse("DO Docteur AGENT=human").unwrap(), &ir).unwrap();
    if let QueryResult::Intervention { effects, .. } = result {
        assert_eq!(effects.len(), 1, "source humaine gardée : {effects:?}");
    } else {
        panic!("Intervention attendue, obtenue : {result:?}");
    }
    let result = execute(&Query::parse("DO Docteur AGENT=collective").unwrap(), &ir).unwrap();
    if let QueryResult::Intervention { effects, .. } = result {
        assert!(effects.is_empty(), "non-humain éliminé : {effects:?}");
    } else {
        panic!("Intervention attendue, obtenue : {result:?}");
    }
    assert!(Query::parse("DO Docteur AGENT=martien").is_err());
}

// ─── Ph7d : EXPLAIN scindé par agent_type ───────────────────────────────────

#[test]
fn explain_grouped_by_agent() {
    let mut a = node(0, "Aaa");
    a.attributes.agent_type = Some(gcn_ir::AgentType::Human);
    let mut b = node(1, "Bbb");
    b.attributes.agent_type = Some(gcn_ir::AgentType::System);
    let ir = make_ir(
        vec![a, b, node(2, "Eee")],
        vec![
            edge(0, 2, RelationType::Cause),
            edge(1, 2, RelationType::Cause),
        ],
    );
    let result = execute(&Query::parse("EXPLAIN Eee").unwrap(), &ir).unwrap();
    if let QueryResult::Abduction { by_agent, .. } = result {
        let human = by_agent
            .iter()
            .find(|g| g.agent_type.as_deref() == Some("human"));
        let system = by_agent
            .iter()
            .find(|g| g.agent_type.as_deref() == Some("system"));
        assert!(
            human.is_some_and(|g| g.hypotheses.iter().any(|h| h.label == "Aaa")),
            "groupe human : {by_agent:?}"
        );
        assert!(
            system.is_some_and(|g| g.hypotheses.iter().any(|h| h.label == "Bbb")),
            "groupe system : {by_agent:?}"
        );
    } else {
        panic!("Abduction attendue, obtenue : {result:?}");
    }
}

// ─── Ph7e : Null bloque + witness_only + WHERE scope ────────────────────────

fn noeud_scope(id: u32, label: &str, scope: Scope) -> CausalNode {
    let mut n = node(id, label);
    n.scope = scope;
    n
}

#[test]
fn null_scope_blocks_with_motif() {
    // Alpha → Beta(null) → Gamma : WHY(Gamma) garde Beta→Gamma, bloque Alpha→Beta.
    let ir = make_ir(
        vec![
            noeud_scope(0, "Alpha", Scope::Universal),
            noeud_scope(1, "Beta", Scope::Null),
            noeud_scope(2, "Gamma", Scope::Universal),
        ],
        vec![
            edge(0, 1, RelationType::Cause),
            edge(1, 2, RelationType::Cause),
        ],
    );
    let result = execute(&Query::parse("WHY Gamma").unwrap(), &ir).unwrap();
    if let QueryResult::Causes {
        links, blocked_by, ..
    } = result
    {
        assert!(
            links.iter().any(|l| l.from == "Beta" && l.to == "Gamma"),
            "Beta→Gamma gardé : {links:?}"
        );
        assert!(links.iter().all(|l| l.from != "Alpha"), "coupé : {links:?}");
        let bloque = blocked_by
            .iter()
            .find(|l| l.from == "Alpha" && l.to == "Beta")
            .expect("blocked_by=[Alpha→Beta]");
        assert_eq!(bloque.motif.as_deref(), Some("null_scope"));
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
}

#[test]
fn existential_marks_witness_only() {
    let ir = make_ir(
        vec![
            noeud_scope(0, "QuelquUn", Scope::Existential),
            noeud_scope(1, "Fievre", Scope::Universal),
        ],
        vec![edge(0, 1, RelationType::Cause)],
    );
    let result = execute(&Query::parse("WHY Fievre").unwrap(), &ir).unwrap();
    if let QueryResult::Causes { links, .. } = result {
        assert_eq!(links.len(), 1);
        assert!(links[0].witness_only, "témoin : {:?}", links[0]);
        assert_eq!(links[0].scope.as_deref(), Some("existential"));
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
    // Universel : pas témoin.
    let ir2 = make_ir(
        vec![
            noeud_scope(0, "Pluie", Scope::Universal),
            noeud_scope(1, "Flaque", Scope::Universal),
        ],
        vec![edge(0, 1, RelationType::Cause)],
    );
    let result = execute(&Query::parse("WHY Flaque").unwrap(), &ir2).unwrap();
    if let QueryResult::Causes { links, .. } = result {
        assert!(!links[0].witness_only);
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
}

#[test]
fn where_scope_filters_results() {
    let ir = make_ir(
        vec![
            noeud_scope(0, "Pluie", Scope::Universal),
            noeud_scope(1, "Flaque", Scope::Universal),
        ],
        vec![edge(0, 1, RelationType::Cause)],
    );
    let result = execute(
        &Query::parse("WHY Flaque WHERE scope=universal").unwrap(),
        &ir,
    )
    .unwrap();
    if let QueryResult::Causes { links, .. } = result {
        assert_eq!(links.len(), 1, "universel gardé : {links:?}");
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
    let result = execute(&Query::parse("WHY Flaque WHERE scope=null").unwrap(), &ir).unwrap();
    if let QueryResult::Causes { links, .. } = result {
        assert!(links.is_empty(), "non-null éliminé : {links:?}");
    } else {
        panic!("Causes attendu, obtenu : {result:?}");
    }
    assert!(Query::parse("WHY Flaque WHERE scope=nimporte").is_err());
}

// ─── Ph7e : agrégat exclut les témoins ──────────────────────────────────────

#[test]
fn aggregate_excludes_witness_links() {
    // Parent P : enfant universel C1 (→X) + enfant existentiel C2 (→Y).
    let mut c1 = node(1, "C1");
    c1.parent = Some(NodeId(0));
    let mut c2 = node(2, "C2");
    c2.parent = Some(NodeId(0));
    c2.scope = Scope::Existential;
    let result = execute(
        &Query::parse("AGGREGATE P").unwrap(),
        &make_ir(
            vec![node(0, "P"), c1, c2, node(3, "X"), node(4, "Y")],
            vec![
                edge(1, 3, RelationType::Cause),
                edge(2, 4, RelationType::Cause),
            ],
        ),
    )
    .unwrap();
    if let QueryResult::HierarchyAggregate { outgoing_edges, .. } = result {
        assert!(
            outgoing_edges.iter().any(|l| l.from == "C1"),
            "universel gardé : {outgoing_edges:?}"
        );
        assert!(
            outgoing_edges.iter().all(|l| l.from != "C2"),
            "témoin jamais généralisé : {outgoing_edges:?}"
        );
    } else {
        panic!("HierarchyAggregate attendu, obtenu : {result:?}");
    }
}
