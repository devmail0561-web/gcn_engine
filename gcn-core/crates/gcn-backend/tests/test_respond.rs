// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! Phase 8b/8d — gold asserts sur UserResponse (phrase non-vide + statut),
//! property always-answer (jamais d'Err nue ni de None à la frontière),
//! roundtrip du format d'échange M1 (CIR JSON → UserResponse JSON).

use gcn_backend::{QueryResult, Status, UserResponse, respond};
use gcn_ir::{
    CausalEdge, CausalIR, CausalNode, IrMetadata, NaturalLanguage, NodeAttributes, NodeId,
    NodeOrigin, NodeType, RelationType, Scope, SourceLanguage, SourceSpan,
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

fn edge(src: u32, dst: u32, rel: RelationType, negated: bool) -> (NodeId, NodeId, CausalEdge) {
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
            third: None,
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
            pipeline: vec!["test_respond".to_string()],
            created_at: None,
        },
        intent: None,
    };
    process(ir).unwrap().ir
}

fn empty_ir() -> CausalIR {
    CausalIR {
        source_lang: SourceLanguage::Natural {
            lang: NaturalLanguage::French,
        },
        source_text: String::new(),
        nodes: vec![],
        edges: vec![],
        cycles: vec![],
        unresolved: vec![],
        metadata: IrMetadata {
            schema_version: "1.0".to_string(),
            pipeline: vec!["test_respond".to_string()],
            created_at: None,
        },
        intent: None,
    }
}

/// Jouet : Alpha → Beta (NIÉE), Beta → Gamma, Alpha → Gamma.
fn jouet() -> CausalIR {
    make_ir(
        vec![node(0, "Alpha"), node(1, "Beta"), node(2, "Gamma")],
        vec![
            edge(0, 1, RelationType::Cause, true),
            edge(1, 2, RelationType::Cause, false),
            edge(0, 2, RelationType::Cause, false),
        ],
    )
}

fn assert_envelope(resp: &UserResponse) {
    assert!(!resp.phrase.is_empty(), "phrase toujours non-vide");
    assert!(!resp.phrase.ends_with(char::is_whitespace), "phrase propre");
    assert_eq!(resp.lang, "fr", "Ph8 : gabarits FR");
}

// ─── 8b : WHY ok ─────────────────────────────────────────────────────────────

/// Jouet propre (zéro niée) : Alpha → Beta → Gamma.
fn jouet_propre() -> CausalIR {
    make_ir(
        vec![node(0, "Alpha"), node(1, "Beta"), node(2, "Gamma")],
        vec![
            edge(0, 1, RelationType::Cause, false),
            edge(1, 2, RelationType::Cause, false),
        ],
    )
}

#[test]
fn why_ok_full_answer() {
    let r = respond("WHY Gamma", &jouet_propre());
    assert_envelope(&r);
    assert_eq!(r.status, Status::Ok);
    assert!(r.phrase.contains("Beta"), "miettes réelles : {}", r.phrase);
    assert!(matches!(r.result, Some(QueryResult::Causes { .. })));
    assert!(!r.suggestions.is_empty());
}

#[test]
fn why_partial_when_negation_cut() {
    // La niée Alpha→Beta est rencontrée sous Gamma → partiel + mention honnête.
    let r = respond("WHY Gamma", &jouet());
    assert_envelope(&r);
    assert_eq!(r.status, Status::Partiel);
    assert!(
        r.phrase.contains("écartée"),
        "coupure nommée : {}",
        r.phrase
    );
}

// ─── 8b : WHY flou → partiel + warning ───────────────────────────────────────

#[test]
fn why_fuzzy_partial_with_warning() {
    let ir = make_ir(vec![node(0, "ventes")], vec![]);
    let r = respond("WHY vente", &ir);
    assert_envelope(&r);
    assert_eq!(r.status, Status::Partiel, "flou → partiel : {:?}", r.status);
    assert!(
        r.suggestions.iter().any(|s| s.contains("exact")),
        "warning flou : {:?}",
        r.suggestions
    );
}

// ─── 8b : introuvable → degrade + reformulation ──────────────────────────────

#[test]
fn not_found_degrade_with_reformulation() {
    let r = respond("WHY zzz", &jouet());
    assert_envelope(&r);
    assert_eq!(r.status, Status::Degrade);
    assert!(r.phrase.contains("zzz"), "label repris : {}", r.phrase);
    assert!(r.result.is_none());
    assert!(!r.suggestions.is_empty());
}

// ─── 8b : ambigu → partiel + candidats + question ────────────────────────────

#[test]
fn ambiguous_partial_with_question() {
    let nodes: Vec<CausalNode> = (0..7)
        .map(|i| node(i, &format!("alpha{}", i + 1)))
        .collect();
    let ir = make_ir(nodes, vec![]);
    let r = respond("WHY alpha", &ir);
    assert_envelope(&r);
    assert_eq!(r.status, Status::Partiel);
    assert!(r.phrase.contains("Laquelle"), "question : {}", r.phrase);
    assert!(
        r.suggestions.iter().any(|s| s.contains("alpha1")),
        "candidats : {:?}",
        r.suggestions
    );
    assert!(r.result.is_none());
}

// ─── 8b : chemin coupé → partiel + bloqueurs + WHY-NOT ───────────────────────

#[test]
fn blocked_path_partial_with_blockers() {
    let ir = make_ir(
        vec![node(0, "Alpha"), node(1, "Beta"), node(2, "Gamma")],
        vec![
            edge(0, 1, RelationType::Cause, true),
            edge(1, 2, RelationType::Cause, false),
        ],
    );
    let r = respond("CHAIN Alpha -> Gamma", &ir);
    assert_envelope(&r);
    assert_eq!(r.status, Status::Partiel);
    assert!(r.phrase.contains("coupe"), "bloqueur nommé : {}", r.phrase);
    assert!(
        r.suggestions.iter().any(|s| s.contains("WHY Gamma")),
        "WHY-NOT : {:?}",
        r.suggestions
    );
    assert!(matches!(
        r.result,
        Some(QueryResult::Path { found: false, .. })
    ));
}

// ─── 8b : aucun chemin → partiel + proches ────────────────────────────────────

#[test]
fn no_path_partial_with_proches() {
    let ir = make_ir(
        vec![node(0, "Alpha"), node(1, "Beta"), node(2, "Zulu")],
        vec![edge(0, 1, RelationType::Cause, false)],
    );
    let r = respond("CHAIN Alpha -> Zulu", &ir);
    assert_envelope(&r);
    assert_eq!(r.status, Status::Partiel);
    assert!(r.phrase.contains("Beta"), "proche nommé : {}", r.phrase);
}

// ─── 8b : requête incomprise → degrade + aide ────────────────────────────────

#[test]
fn bad_query_degrade_with_help() {
    let r = respond("WHEY ventes", &jouet());
    assert_envelope(&r);
    assert_eq!(r.status, Status::Degrade);
    assert!(r.phrase.contains("WHY"), "vouliez-vous dire : {}", r.phrase);
    assert!(r.result.is_none());
    let r = respond("FOOBAR xyz", &jouet());
    assert_eq!(r.status, Status::Degrade);
    assert!(r.phrase.contains("WHY"), "aide : {}", r.phrase);
}

// ─── 8b : graphe vide → degrade ──────────────────────────────────────────────

#[test]
fn empty_ir_degrade() {
    let r = respond("WHY x", &empty_ir());
    assert_envelope(&r);
    assert_eq!(r.status, Status::Degrade);
    assert!(r.phrase.contains("vide"), "cir-vide : {}", r.phrase);
}

// ─── 8b : SPOF > 500 → partiel + top-k ────────────────────────────────────────

#[test]
fn too_big_partial_with_top() {
    let nodes: Vec<CausalNode> = (0..501).map(|i| node(i, &format!("n{i}"))).collect();
    let edges = vec![
        edge(0, 1, RelationType::Cause, false),
        edge(0, 2, RelationType::Cause, false),
        edge(0, 3, RelationType::Cause, false),
    ];
    let ir = make_ir(nodes, edges);
    let r = respond("SPOF", &ir);
    assert_envelope(&r);
    assert_eq!(r.status, Status::Partiel);
    assert!(r.phrase.contains("501"), "annonce : {}", r.phrase);
    assert!(!r.suggestions.is_empty(), "top-k en suggestions");
}

// ─── M1 : roundtrip CIR JSON → UserResponse JSON ──────────────────────────────

#[test]
fn exchange_format_roundtrip() {
    let ir = jouet_propre();
    let ir_json = serde_json::to_string(&ir).unwrap();
    let ir2: CausalIR = serde_json::from_str(&ir_json).unwrap();
    let direct = respond("WHY Gamma", &ir);
    let via_json = respond("WHY Gamma", &ir2);
    assert_eq!(direct.phrase, via_json.phrase);
    assert_eq!(direct.status, via_json.status);
    let resp_json = serde_json::to_string(&via_json).unwrap();
    let back: UserResponse = serde_json::from_str(&resp_json).unwrap();
    assert_eq!(back.phrase, direct.phrase);
    let v: serde_json::Value = serde_json::from_str(&resp_json).unwrap();
    assert_eq!(v["status"], "ok");
    assert_eq!(v["lang"], "fr");
    assert!(v["phrase"].is_string());
}

// ─── 8d : property always-answer ──────────────────────────────────────────────

#[test]
fn always_answer_matrix() {
    let small = jouet();
    let empty = empty_ir();
    let big_nodes: Vec<CausalNode> = (0..501).map(|i| node(i, &format!("b{i}"))).collect();
    let big = make_ir(big_nodes, vec![]);
    let uni = make_ir(vec![node(0, "日本語"), node(1, "🎉")], vec![]);
    let queries = vec![
        "WHY x",
        "WHAT y",
        "CHAIN a -> b",
        "DO z",
        "COUNTERFACTUAL w",
        "EXPLAIN e",
        "CYCLES",
        "GAPS",
        "SPOF",
        "DENSITY",
        "FOOBAR",
        "",
        "WHY ",
        "WHY 日本語?",
        "CHAIN a->b TOPK=99",
        "WHY b1 AMBIGUITY_LIMIT=1",
        "ASK x",
        "ASK",
        "WHY-NOT a -> b",
        "WHY-NOT a",
        "WHY x TOPK=2",
        "WHY x TOPK=0",
        "EXPLAIN e CONTRAST f",
        "EXPLAIN e CONTRAST",
        "ANALOGY a -> b HOPS=2",
        "ANALOGY a -> b HOPS=0",
    ];
    for ir in [&small, &empty, &big, &uni] {
        for q in &queries {
            let r = respond(q, ir);
            assert_envelope(&r);
            assert!(
                matches!(r.status, Status::Ok | Status::Partiel | Status::Degrade),
                "statut fermé pour {q:?}"
            );
        }
    }
}

// ─── Ph6b : enveloppes ASK / WHY-NOT / contraste ───────────────────────────

#[test]
fn ask_envelope_card() {
    let r = respond("ASK Beta", &jouet());
    assert_envelope(&r);
    assert_eq!(r.status, Status::Ok);
    assert!(r.phrase.contains("Beta"), "fiche nommée : {}", r.phrase);
    assert!(matches!(r.result, Some(QueryResult::AskReport { .. })));
}

#[test]
fn whynot_cut_envelope() {
    let ir = make_ir(
        vec![node(0, "Alpha"), node(1, "Beta"), node(2, "Gamma")],
        vec![
            edge(0, 1, RelationType::Cause, true),
            edge(1, 2, RelationType::Cause, false),
        ],
    );
    let r = respond("WHY-NOT Alpha -> Gamma", &ir);
    assert_envelope(&r);
    assert_eq!(r.status, Status::Partiel);
    assert!(r.phrase.contains("coupe"), "bloqueur nommé : {}", r.phrase);
    assert!(r.suggestions.iter().any(|s| s.contains("WHY Gamma")));
}

#[test]
fn whynot_empty_envelope() {
    let ir = make_ir(
        vec![node(0, "Alpha"), node(1, "Beta"), node(2, "Zulu")],
        vec![edge(0, 1, RelationType::Cause, false)],
    );
    let r = respond("WHY-NOT Alpha -> Zulu", &ir);
    assert_envelope(&r);
    assert_eq!(r.status, Status::Partiel);
    assert!(r.phrase.contains("Beta"), "proche nommé : {}", r.phrase);
}

#[test]
fn explain_contrast_envelope() {
    let ir = make_ir(
        vec![
            node(0, "Aaa"),
            node(1, "Bbb"),
            node(2, "Eee"),
            node(3, "Fff"),
            node(4, "Ccc"),
        ],
        vec![
            edge(0, 2, RelationType::Cause, false),
            edge(1, 2, RelationType::Cause, false),
            edge(1, 3, RelationType::Cause, false),
            edge(4, 3, RelationType::Cause, false),
        ],
    );
    let r = respond("EXPLAIN Eee CONTRAST Fff", &ir);
    assert_envelope(&r);
    assert_eq!(r.status, Status::Ok);
    assert!(r.phrase.contains("Aaa"), "distinctif nommé : {}", r.phrase);
}

// ─── Ph7b : enveloppe DOUBTS + propagation caveats ──────────────────────────

#[test]
fn doubts_envelope_empty() {
    // Jouet sans unresolved : aucun doute, statut Ok.
    let r = respond("DOUBTS", &jouet());
    assert_envelope(&r);
    assert_eq!(r.status, Status::Ok);
    assert!(
        r.phrase.contains("Aucun doute"),
        "vide annoncé : {}",
        r.phrase
    );
}

#[test]
fn caveats_propagated_to_envelope() {
    // Scopes Unknown des fixtures → caveats portées dans l'enveloppe.
    let r = respond("WHY Gamma", &jouet());
    assert_envelope(&r);
    assert!(!r.caveats.is_empty(), "caveats propagées : {:?}", r.caveats);
    assert!(r.caveats.iter().all(|c| c.contains('—')));
}

#[test]
fn invalid_ir_errors_refuse_response() {
    // P1-rigidité : arête pendante (nœud 99 absent) → diagnostic Error →
    // respond_validated refuse (Degrade invalid_ir), respond() historique répond.
    use gcn_backend::respond_validated;
    use gcn_middleend::error::DiagnosticSeverity;
    let ir = make_ir(
        vec![node(0, "Alpha"), node(1, "Beta")],
        vec![edge(0, 99, RelationType::Cause, false)],
    );
    let processed = process(ir).expect("process ne refuse pas (diagnostics)");
    assert!(
        processed
            .diagnostics
            .iter()
            .any(|d| d.severity == DiagnosticSeverity::Error),
        "pendante doit être Error"
    );
    let r = respond_validated("WHY Beta", &processed.ir, &processed.diagnostics);
    assert_envelope(&r);
    assert_eq!(r.status, Status::Degrade);
    assert!(r.phrase.contains("invalide"), "motif nommé : {}", r.phrase);
    // Warnings seuls : réponse normale.
    let clean = process(jouet_propre()).expect("propre");
    let r2 = respond_validated("WHY Gamma", &clean.ir, &clean.diagnostics);
    assert_eq!(r2.status, Status::Ok);
}
