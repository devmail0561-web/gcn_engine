// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! Tests gold Ph1 (négation bloqueuse + tiers) — résultats attendus en dur,
//! incluant `blocked_by`.
//!
//! Phase 4 démarrée ici (fichier créé en P0) : diamant, chaîne niée, joint,
//! cycle, temporel désordonné, prevent. Properties + fuzz = suite de la phase 4.

use gcn_backend::pearl::{DEFAULT_AMBIGUITY_LIMIT, LabelSim, why_scoped_with};
use gcn_backend::query::DEFAULT_TOPK;
use gcn_backend::{BackendError, Query, QueryResult, chain_strict, execute};
use gcn_ir::{
    CausalEdge, CausalIR, CausalNode, IrMetadata, NaturalLanguage, NodeAttributes, NodeId,
    NodeOrigin, NodeType, RelationType, Scope, SourceLanguage, SourceSpan, TernaryRole,
    TernaryThird,
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
            ambiguity_limit: 3
        }
    );
    assert_eq!(
        Query::parse("WHAT vent").unwrap(),
        Query::What {
            label: "vent".into(),
            ambiguity_limit: DEFAULT_AMBIGUITY_LIMIT
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
            topk: 2
        }
    );
    assert_eq!(
        Query::parse("CHAIN a -> b").unwrap(),
        Query::Chain {
            from: "a".into(),
            to: "b".into(),
            topk: DEFAULT_TOPK
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
            min_score: 0.5
        }
    );
    assert!(Query::parse("ANALOGY a -> b WEIGHTS=0.4,0.3").is_err());
}
