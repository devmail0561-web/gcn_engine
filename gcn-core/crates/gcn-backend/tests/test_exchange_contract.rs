// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! Socle 0b (D-S03) — contrat d'échange M1 : le lecteur Rust tolère le CIR
//! émis par Python (schema 2.0), ignore les clés futures inconnues (s1-s3),
//! accepte les formes historiques. Gate de version en CI (ce fichier).

use gcn_backend::{Status, respond};
use gcn_ir::{CausalIR, EXCHANGE_SCHEMA_VERSION};

fn fixture() -> CausalIR {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/python_emitter_s0.json");
    let raw = std::fs::read_to_string(&path).expect("fixture M1 manquante");
    serde_json::from_str(&raw).expect("CIR Python illisible par Rust")
}

#[test]
fn exchange_contract_version_and_shape() {
    let ir = fixture();
    assert_eq!(
        ir.metadata.schema_version, EXCHANGE_SCHEMA_VERSION,
        "contrat M1 : schema_version == 2.0 des deux côtés"
    );
    assert_eq!(ir.node_count(), 3);
    assert_eq!(ir.edge_count(), 3);
    // Clés futures inconnues (s2 : tok_vocab ; s3 : champs ad hoc) ignorées.
    let raw = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/python_emitter_s0.json"),
    )
    .unwrap();
    assert!(
        raw.contains("tok_vocab"),
        "fixture doit porter une clé future"
    );
    assert!(
        raw.contains("future_field"),
        "fixture doit porter une clé future"
    );
}

#[test]
fn exchange_roundtrip_m1_respond() {
    // Pont M1 : CIR Python → respond() Rust (enveloppe UserResponse).
    let ir = fixture();
    let r = respond("WHY inondations", &ir);
    assert_eq!(r.status, Status::Ok);
    assert!(r.phrase.contains("pluie"), "miettes réelles : {}", r.phrase);
    let json = serde_json::to_string(&r).unwrap();
    let back: gcn_backend::UserResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(back.phrase, r.phrase);
}

#[test]
fn tolerant_reader_legacy_shapes() {
    // Formes historiques Rust (PascalCase) toujours lisibles (aliases D-S03).
    let span: gcn_ir::SourceSpan = serde_json::from_str(r#""Synthetic""#).unwrap();
    assert!(matches!(span, gcn_ir::SourceSpan::Synthetic));
    let span: gcn_ir::SourceSpan =
        serde_json::from_str(r#"{"TokenSpan":{"start":1,"end":2}}"#).unwrap();
    assert!(matches!(span, gcn_ir::SourceSpan::TokenSpan { .. }));
    // Clés optionnelles absentes → défauts (tolérance s0→s3).
    let edge: gcn_ir::CausalEdge = serde_json::from_str(
        r#"{"relation":"cause","confidence":0.5,"explicit":true,"negated":false}"#,
    )
    .unwrap();
    assert!(edge.temporal_gap.is_none());
    assert!(edge.marker_token.is_none());
    assert!(edge.in_cycle.is_none());
    assert!(edge.provenance.is_none());
    let meta: gcn_ir::IrMetadata = serde_json::from_str(r#"{}"#).unwrap();
    assert_eq!(meta.schema_version, "");
}

#[test]
fn cir_intent_optional_and_roundtrip() {
    // R9 : PR Rust Option<CirIntent> + roundtrip AVANT la tête.
    // Absent → None (CIR legacy / non annoté).
    let bare: gcn_ir::CausalIR = serde_json::from_str(
        r#"{"source_lang":{"natural":{"lang":"french"}},"source_text":"",
            "nodes":[],"edges":[],"cycles":[],"unresolved":[],
            "metadata":{"schema_version":"2.0","pipeline":[],"created_at":null}}"#,
    )
    .unwrap();
    assert!(bare.intent.is_none());
    // Présent → porté (acte + langue + saillance).
    let full: gcn_ir::CausalIR = serde_json::from_str(
        r#"{"source_lang":{"natural":{"lang":"french"}},"source_text":"",
            "nodes":[],"edges":[],"cycles":[],"unresolved":[],
            "metadata":{"schema_version":"2.0","pipeline":[],"created_at":null},
            "intent":{"act":"question","lang":"fr","saillant":false}}"#,
    )
    .unwrap();
    let intent = full.intent.as_ref().expect("intent présent");
    assert!(matches!(intent.act, gcn_ir::Act::Question));
    assert_eq!(intent.lang, "fr");
    assert!(!intent.saillant);
    // Roundtrip : l'intent survit à la sérialisation (pont M1).
    let back: gcn_ir::CausalIR =
        serde_json::from_str(&serde_json::to_string(&full).unwrap()).unwrap();
    let intent = back.intent.expect("intent après roundtrip");
    assert!(matches!(intent.act, gcn_ir::Act::Question));
    // Langue absente → défaut français ([R-LANG]).
    let no_lang: gcn_ir::CausalIR = serde_json::from_str(
        r#"{"source_lang":{"natural":{"lang":"french"}},"source_text":"",
            "nodes":[],"edges":[],"cycles":[],"unresolved":[],
            "metadata":{"schema_version":"2.0","pipeline":[],"created_at":null},
            "intent":{"act":"assert"}}"#,
    )
    .unwrap();
    let intent = no_lang.intent.expect("intent présent");
    assert_eq!(intent.lang, "fr");
    assert!(!intent.saillant);
}
