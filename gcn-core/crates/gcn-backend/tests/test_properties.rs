// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! Phase 4b/4c — propriétés (déterminisme, monotonie, joint, négation,
//! temporel, ambiguïté) + fuzz `Query::parse` (erreur typée, jamais panic).
//! Snapshots JSON : `tests/snapshots/*.json`, comparés à l'octet près ;
//! régénérer (diff revu !) avec `SNAPSHOT_UPDATE=1`.

use std::collections::HashSet;
use std::path::Path;

use gcn_backend::query::LinkDto;
use gcn_backend::{BackendError, Query, QueryResult, execute};
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
            pipeline: vec!["test_properties".to_string()],
            created_at: None,
        },
        intent: None,
    };
    process(ir).unwrap().ir
}

/// Jouet mixte : Alpha → Beta (NIÉE), Beta → Gamma, Alpha → Gamma, Delta isolé.
fn jouet_mixte() -> CausalIR {
    make_ir(
        vec![
            node(0, "Alpha"),
            node(1, "Beta"),
            node(2, "Gamma"),
            node(3, "Delta"),
        ],
        vec![
            edge(0, 1, RelationType::Cause, true),
            edge(1, 2, RelationType::Cause, false),
            edge(0, 2, RelationType::Cause, false),
        ],
    )
}

fn links_of(result: &QueryResult) -> &[LinkDto] {
    match result {
        QueryResult::Causes { links, .. } => links,
        QueryResult::Effects { links, .. } => links,
        QueryResult::Path { links, .. } => links,
        QueryResult::Intervention { effects, .. } => effects,
        QueryResult::CounterfactualDiff { actual_effects, .. } => actual_effects,
        _ => panic!("résultat sans liens : {result:?}"),
    }
}

// ─── Déterminisme : 2 runs = mêmes octets ───────────────────────────────────

#[test]
fn deterministic_same_bytes_twice() {
    let ir = jouet_mixte();
    for q in [
        "WHY Gamma",
        "WHAT Alpha",
        "CHAIN Alpha -> Gamma",
        "DO Beta",
        "COUNTERFACTUAL Alpha",
    ] {
        let query = Query::parse(q).unwrap();
        let a = serde_json::to_string(&execute(&query, &ir).unwrap()).unwrap();
        let b = serde_json::to_string(&execute(&query, &ir).unwrap()).unwrap();
        assert_eq!(a, b, "instable entre 2 runs : {q}");
    }
}

// ─── Monotonie : couper une arête ne crée jamais de lien ────────────────────

#[test]
fn removing_edge_never_adds_links() {
    let full = jouet_mixte();
    let mut cut = full.clone();
    cut.edges
        .retain(|(s, d, _)| !(*s == NodeId(0) && *d == NodeId(2)));
    let cut = process(cut).unwrap().ir;
    for q in ["WHY Gamma", "WHAT Alpha", "CHAIN Alpha -> Gamma"] {
        let query = Query::parse(q).unwrap();
        let before: HashSet<(String, String)> = links_of(&execute(&query, &full).unwrap())
            .iter()
            .map(|l| (l.from.clone(), l.to.clone()))
            .collect();
        let after: HashSet<(String, String)> = links_of(&execute(&query, &cut).unwrap())
            .iter()
            .map(|l| (l.from.clone(), l.to.clone()))
            .collect();
        assert!(
            after.is_subset(&before),
            "coupure a créé des liens ({q}) : {after:?} ⊄ {before:?}"
        );
    }
}

// ─── Joint-nécessité : 1 co-source retirée = joint invalide ─────────────────

#[test]
fn joint_invalid_without_one_cosource() {
    let mut e1 = edge(0, 2, RelationType::JointCause, false);
    e1.2.joint_group_id = Some("g".to_string());
    let mut e2 = edge(1, 2, RelationType::JointCause, false);
    e2.2.joint_group_id = Some("g".to_string());
    let full = make_ir(
        vec![node(0, "Un"), node(1, "Deux"), node(2, "Cible")],
        vec![e1, e2],
    );
    let result = execute(&Query::parse("WHY Cible").unwrap(), &full).unwrap();
    if let QueryResult::Causes { links, .. } = result {
        assert_eq!(links.len(), 2);
        assert!(
            links
                .iter()
                .all(|l| l.joint_group_id.as_deref() == Some("g"))
        );
    } else {
        panic!("Causes attendu : {result:?}");
    }
    // Sans la 2e co-source : plus de groupe joint (joint invalide).
    let mut e1 = edge(0, 2, RelationType::JointCause, false);
    e1.2.joint_group_id = Some("g".to_string());
    let cut = make_ir(
        vec![node(0, "Un"), node(1, "Deux"), node(2, "Cible")],
        vec![e1],
    );
    let result = execute(&Query::parse("WHY Cible").unwrap(), &cut).unwrap();
    if let QueryResult::Causes { links, .. } = result {
        assert_eq!(links.len(), 1);
        assert!(
            links.iter().all(|l| l.joint_group_id.is_none()),
            "groupe orphelin interdit : {links:?}"
        );
    } else {
        panic!("Causes attendu : {result:?}");
    }
}

// ─── Négation-bloque : aucun lien traversé n'est nié ─────────────────────────

#[test]
fn no_traversed_link_is_negated() {
    let ir = jouet_mixte();
    for q in [
        "WHY Gamma",
        "WHAT Alpha",
        "CHAIN Alpha -> Gamma",
        "DO Beta",
        "COUNTERFACTUAL Alpha",
    ] {
        let query = Query::parse(q).unwrap();
        for l in links_of(&execute(&query, &ir).unwrap()) {
            assert!(!l.negated, "lien nié traversé ({q}) : {}→{}", l.from, l.to);
        }
    }
}

// ─── Abduction : ancêtre via niée exclu ─────────────────────────────────────

#[test]
fn abduct_skips_negated_ancestors() {
    // Alpha → Beta NIÉE, Beta → Gamma : Alpha inatteignable sans la niée.
    let ir = make_ir(
        vec![node(0, "Alpha"), node(1, "Beta"), node(2, "Gamma")],
        vec![
            edge(0, 1, RelationType::Cause, true),
            edge(1, 2, RelationType::Cause, false),
        ],
    );
    let result = execute(&Query::parse("EXPLAIN Gamma").unwrap(), &ir).unwrap();
    if let QueryResult::Abduction { hypotheses, .. } = result {
        let labels: Vec<&str> = hypotheses.iter().map(|h| h.label.as_str()).collect();
        assert_eq!(labels, vec!["Beta"], "Alpha via niée exclu : {labels:?}");
    } else {
        panic!("Abduction attendue : {result:?}");
    }
}

// ─── Temporel : indices ordonnés ⟺ temporally_ordered ───────────────────────

#[test]
fn temporal_order_matches_indices() {
    // Chaîne linéaire (le meilleur Dijkstra = la chaîne elle-même).
    let linear = || {
        make_ir(
            vec![node(0, "Alpha"), node(1, "Beta"), node(2, "Gamma")],
            vec![
                edge(0, 1, RelationType::Cause, false),
                edge(1, 2, RelationType::Cause, false),
            ],
        )
    };
    let mut ordered = linear();
    for (n, ti) in [(0usize, 1i32), (1, 2), (2, 3)] {
        ordered.nodes[n].temporal_index = Some(ti);
    }
    let result = execute(&Query::parse("CHAIN_T Alpha -> Gamma").unwrap(), &ordered).unwrap();
    if let QueryResult::TemporalPath {
        found,
        temporally_ordered,
        ..
    } = result
    {
        assert!(found && temporally_ordered, "ordonné → true : {result:?}");
    } else {
        panic!("TemporalPath attendu : {result:?}");
    }
    let mut disordered = ordered.clone();
    disordered.nodes[1].temporal_index = Some(9);
    let result = execute(
        &Query::parse("CHAIN_T Alpha -> Gamma").unwrap(),
        &disordered,
    )
    .unwrap();
    if let QueryResult::TemporalPath {
        found,
        temporally_ordered,
        ..
    } = result
    {
        assert!(
            found && !temporally_ordered,
            "désordonné → false : {result:?}"
        );
    } else {
        panic!("TemporalPath attendu : {result:?}");
    }
}

// ─── Ambiguïté : limite = frontière exacte ──────────────────────────────────

fn jouet_n_flous(n: u32) -> CausalIR {
    let nodes: Vec<CausalNode> = (0..n).map(|i| node(i, &format!("w{}", i + 1))).collect();
    make_ir(nodes, vec![])
}

#[test]
fn ambiguity_limit_is_exact_boundary() {
    // 5 flous = limite défaut → OK ; 6 → Err + 6 candidats.
    let ok = execute(&Query::parse("WHY w").unwrap(), &jouet_n_flous(5)).unwrap();
    if let QueryResult::Causes {
        matched_targets,
        fuzzy,
        ..
    } = ok
    {
        assert_eq!(matched_targets.len(), 5);
        assert!(fuzzy);
    } else {
        panic!("Causes attendu : {ok:?}");
    }
    let err = execute(&Query::parse("WHY w").unwrap(), &jouet_n_flous(6));
    match err {
        Err(BackendError::Ambiguous { candidates }) => assert_eq!(candidates.len(), 6),
        other => panic!("Ambiguous(6) attendu, obtenu : {other:?}"),
    }
}

// ─── Fuzz Query::parse : erreur typée, jamais panic ─────────────────────────

#[test]
fn fuzz_parse_never_panics_typed_errors() {
    let long = "x".repeat(10_000);
    let invalid = vec![
        "",
        " ",
        "\t",
        "WHY",
        "WHY ",
        "->",
        "CHAIN ->",
        "CHAIN A",
        "A -> B",
        "WHY A AMBIGUITY_LIMIT=",
        "WHY A AMBIGUITY_LIMIT=x",
        "CHAIN A -> B TOPK=",
        "CHAIN A -> B TOPK=0",
        "ANALOGY A -> B WEIGHTS=,",
        "ANALOGY A -> B WEIGHTS=0.4,0.3",
        "ANALOGY -> ",
        "FOOBAR xyz",
        "CHAIN A -> B TOPK=99999999999999999999",
        "WHY A AMBIGUITY_LIMIT=99999999999999999999",
    ];
    for input in &invalid {
        assert!(
            Query::parse(input).is_err(),
            "erreur typée attendue pour : {input:?}"
        );
    }
    let mut valid: Vec<String> = vec![
        "wHy VeNtEs",
        "chain a -> b topk=2",
        "WHY 日本語?",
        "CHAIN café -> naïve",
        "WHY 🎉?",
        "WHY A AMBIGUITY_LIMIT=0",
        // Label exotique accepté au parse (NodeNotFound à l'exécution).
        "WHY A\x00B",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    valid.push(format!("WHY {long}"));
    for input in &valid {
        assert!(
            Query::parse(input).is_ok(),
            "accepté attendu pour : {:?}...",
            &input[..input.len().min(40)]
        );
    }
}

// ─── Snapshots JSON ──────────────────────────────────────────────────────────

fn check_snapshot(name: &str, value: &QueryResult) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots");
    let path = dir.join(format!("{name}.json"));
    let actual = serde_json::to_string_pretty(value).unwrap() + "\n";
    if std::env::var("SNAPSHOT_UPDATE").is_ok() {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, &actual).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "snapshot manquant : {} (générer avec SNAPSHOT_UPDATE=1, diff revu !)",
            path.display()
        )
    });
    assert_eq!(actual, expected, "snapshot {name} instable");
}

#[test]
fn snapshot_why_negated() {
    let ir = jouet_mixte();
    let result = execute(&Query::parse("WHY Gamma").unwrap(), &ir).unwrap();
    check_snapshot("why_gamma_negated", &result);
}

#[test]
fn snapshot_chain_ranked() {
    let ir = make_ir(
        vec![node(0, "Alpha"), node(1, "Xi"), node(2, "Beta")],
        vec![
            edge(0, 2, RelationType::Cause, false),
            edge(0, 1, RelationType::Cause, false),
            edge(1, 2, RelationType::Cause, false),
        ],
    );
    let result = execute(&Query::parse("CHAIN Alpha -> Beta").unwrap(), &ir).unwrap();
    check_snapshot("chain_alpha_beta_ranked", &result);
}

#[test]
fn snapshot_do_negated() {
    let ir = jouet_mixte();
    let result = execute(&Query::parse("DO Beta").unwrap(), &ir).unwrap();
    check_snapshot("do_beta_negated", &result);
}
