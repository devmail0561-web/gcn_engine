// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use gcn_frontend_graph::{
    is_supported_relationship, parse_stix_bundle_with_report, stix_object_type_to_node_type,
    stix_relationship_to_relation_type,
};
use gcn_ir::{GraphFormat, NodeType, RelationType, SourceLanguage};

const BUNDLE: &str = r#"{
  "type": "bundle",
  "id": "bundle--abc123",
  "objects": [
    {"type": "threat-actor", "id": "threat-actor--a1", "name": "Fancy Bear"},
    {"type": "attack-pattern", "id": "attack-pattern--b2", "name": "Spearphishing"},
    {"type": "malware", "id": "malware--c3", "name": "Zeus"},
    {"type": "identity", "id": "identity--d4", "name": "Victim Corp"},
    {"type": "vulnerability", "id": "vulnerability--e5", "name": "CVE-2024-0001"},
    {"type": "ipv4-addr", "id": "ipv4-addr--f6", "value": "10.0.0.1"},
    {"type": "relationship", "id": "relationship--r1",
     "relationship_type": "uses", "source_ref": "threat-actor--a1", "target_ref": "malware--c3"},
    {"type": "relationship", "id": "relationship--r2",
     "relationship_type": "targets", "source_ref": "threat-actor--a1", "target_ref": "identity--d4"},
    {"type": "relationship", "id": "relationship--r3",
     "relationship_type": "mitigates", "source_ref": "attack-pattern--b2", "target_ref": "vulnerability--e5"},
    {"type": "relationship", "id": "relationship--r4",
     "relationship_type": "related-to", "source_ref": "malware--c3", "target_ref": "ipv4-addr--f6"},
    {"type": "relationship", "id": "relationship--r5",
     "relationship_type": "uses", "source_ref": "threat-actor--a1", "target_ref": "missing--zzz"}
  ]
}"#;

#[test]
fn stix_bundle_produces_expected_nodes_and_edges() {
    let (ir, report) = parse_stix_bundle_with_report(BUNDLE).expect("bundle valide");
    assert_eq!(ir.nodes.len(), 6, "6 objets non-relationship → 6 nœuds");
    assert_eq!(ir.edges.len(), 3, "uses+targets+mitigates → 3 arêtes");
    assert_eq!(report.skipped_unknown_relationships, 1, "related-to ignoré");
    assert_eq!(report.skipped_dangling, 1, "target_ref absent ignoré");
    assert_eq!(report.skipped_objects, 0);

    assert!(matches!(
        ir.source_lang,
        SourceLanguage::Graph {
            format: GraphFormat::Stix21
        }
    ));

    let rels: Vec<RelationType> = ir.edges.iter().map(|(_, _, e)| e.relation).collect();
    assert!(rels.contains(&RelationType::Enable), "uses → enable");
    assert!(rels.contains(&RelationType::Cause), "targets → cause");
    assert!(rels.contains(&RelationType::Prevent), "mitigates → prevent");
    for (_, _, e) in &ir.edges {
        assert_eq!(e.confidence, 1.0);
        assert!(e.explicit);
        assert!(e.provenance.is_some(), "provenance tracée (bundle id)");
    }
}

#[test]
fn stix_node_types_follow_mapping_table() {
    assert_eq!(
        stix_object_type_to_node_type("threat-actor"),
        NodeType::Entite
    );
    assert_eq!(
        stix_object_type_to_node_type("attack-pattern"),
        NodeType::Processus
    );
    assert_eq!(
        stix_object_type_to_node_type("vulnerability"),
        NodeType::Condition
    );
    assert_eq!(
        stix_object_type_to_node_type("indicator"),
        NodeType::Concept
    );
    assert_eq!(
        stix_object_type_to_node_type("observed-data"),
        NodeType::Evenement
    );
    assert_eq!(stix_object_type_to_node_type("file"), NodeType::Entite);
    assert_eq!(
        stix_object_type_to_node_type("x-stix-future-thing"),
        NodeType::Concept,
        "inconnu → concept, jamais d'échec"
    );
}

#[test]
fn relationship_table_covers_causal_types_only() {
    assert_eq!(
        stix_relationship_to_relation_type("uses"),
        Some(RelationType::Enable)
    );
    assert_eq!(
        stix_relationship_to_relation_type("variant-of"),
        Some(RelationType::Analogy)
    );
    assert_eq!(
        stix_relationship_to_relation_type("detects"),
        Some(RelationType::Enable),
        "F1 audit : detects → enable (couverture ATT&CK)"
    );
    assert_eq!(stix_relationship_to_relation_type("related-to"), None);
    assert_eq!(stix_relationship_to_relation_type("located-at"), None);
    assert!(!is_supported_relationship("related-to"));
    assert!(is_supported_relationship("mitigates"));
}

#[test]
fn non_bundle_rejected_with_typed_error() {
    // P0-5 : via with_report (l'alias aveugle parse_stix_bundle est déprécié).
    let err = parse_stix_bundle_with_report(r#"{"type": "indicator"}"#).unwrap_err();
    assert!(err.to_string().contains("bundle"));
}

#[test]
fn invalid_json_rejected() {
    assert!(parse_stix_bundle_with_report("{pas du json").is_err());
}

#[test]
fn cir_json_roundtrip() {
    let (ir, _) = parse_stix_bundle_with_report(BUNDLE).expect("bundle valide");
    let s = serde_json::to_string(&ir).expect("serialize CIR");
    assert!(s.contains("\"stix21\""), "format graphe tracé en JSON");
    let back: gcn_ir::CausalIR = serde_json::from_str(&s).expect("deserialize CIR");
    assert_eq!(back.nodes.len(), 6);
    assert_eq!(back.edges.len(), 3);
}
