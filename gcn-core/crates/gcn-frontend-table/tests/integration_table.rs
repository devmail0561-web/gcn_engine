// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use gcn_frontend_table::{TableParseReport, TableParserError, TableSchema, parse_table};
use gcn_ir::{GraphFormat, NodeType, RelationType, SourceLanguage};

const CSV: &str = "facteur,resultat,relation,confiance\n\
    pluie,inondation,cause,0.9\n\
    \"sécheresse, prolongée\",incendie,enable,0.7\n\
    vent,tempête,prevent,\n\
    soleil,pluie,magie,0.5\n\
    ,vide,cause,0.5\n\
    brouillard,verglas,cause,abc\n";

fn schema() -> TableSchema {
    TableSchema {
        cause_col: "facteur".to_string(),
        effect_col: "resultat".to_string(),
        relation_col: Some("relation".to_string()),
        confidence_col: Some("confiance".to_string()),
        ..TableSchema::default()
    }
}

#[test]
fn csv_lignes_produisent_aretes_avec_rapport() {
    let (ir, report) =
        parse_table(CSV, &schema(), Some("risques.csv".to_string())).expect("csv valide");
    // 3 lignes valides (la quotée compte), 1 relation inconnue, 1 vide, 1 confiance HS.
    assert_eq!(ir.edges.len(), 3, "3 arêtes");
    assert_eq!(
        report,
        TableParseReport {
            skipped_empty: 1,
            skipped_unknown_relation: 1,
            skipped_bad_confidence: 1,
            skipped_field_count: 0,
        }
    );
    assert!(matches!(
        ir.source_lang,
        SourceLanguage::Graph {
            format: GraphFormat::Table
        }
    ));

    let by_label = |label: &str| {
        ir.nodes
            .iter()
            .find(|n| n.label == label)
            .unwrap_or_else(|| panic!("nœud {label} absent"))
            .id
    };
    let edge = |s: &str, d: &str| {
        ir.edges
            .iter()
            .find(|(a, b, _)| *a == by_label(s) && *b == by_label(d))
            .unwrap_or_else(|| panic!("arête {s}->{d} absente"))
            .2
            .clone()
    };
    let e1 = edge("pluie", "inondation");
    assert_eq!(e1.relation, RelationType::Cause);
    assert!((e1.confidence - 0.9).abs() < 1e-6);
    // Champ quoté avec virgule interne = UNE cellule.
    let e2 = edge("sécheresse, prolongée", "incendie");
    assert_eq!(e2.relation, RelationType::Enable);
    // Confiance vide → 1.0.
    assert_eq!(edge("vent", "tempête").confidence, 1.0);
    // Déduplication : 3 arêtes, 6 nœuds distincts.
    assert_eq!(ir.nodes.len(), 6);
    for n in &ir.nodes {
        assert_eq!(n.node_type, NodeType::Entite, "défaut documenté");
    }
}

#[test]
fn defauts_sans_colonnes_optionnelles() {
    let csv = "a,b\nx,y\n";
    let schema = TableSchema {
        cause_col: "a".to_string(),
        effect_col: "b".to_string(),
        ..TableSchema::default()
    };
    let (ir, report) = parse_table(csv, &schema, None).expect("csv valide");
    assert_eq!(ir.edges.len(), 1);
    assert_eq!(ir.edges[0].2.relation, RelationType::Cause);
    assert_eq!(report.total_skipped(), 0);
}

#[test]
fn relations_insensibles_a_la_casse() {
    let csv = "a,b,r\nx,y,Prevent\n";
    let schema = TableSchema {
        cause_col: "a".to_string(),
        effect_col: "b".to_string(),
        relation_col: Some("r".to_string()),
        ..TableSchema::default()
    };
    let (ir, _) = parse_table(csv, &schema, None).expect("csv valide");
    assert_eq!(ir.edges[0].2.relation, RelationType::Prevent);
}

#[test]
fn confiance_hors_bornes_ignoree_comptee() {
    // F2 audit : 1e3, -0.5, NaN, inf → skipped_bad_confidence (pas de clamp).
    let csv = "cause,effect,conf\nx,y,1e3\na,b,-0.5\nc,d,NaN\ne,f,inf\ng,h,0.4\n";
    let schema = TableSchema {
        confidence_col: Some("conf".to_string()),
        ..TableSchema::default()
    };
    let (ir, report) = parse_table(csv, &schema, None).expect("csv valide");
    assert_eq!(ir.edges.len(), 1, "seule 0.4 produit une arête");
    assert_eq!(report.skipped_bad_confidence, 4);
    assert!((ir.edges[0].2.confidence - 0.4).abs() < 1e-6);
}

#[test]
fn erreurs_typees() {
    let schema = TableSchema::default();
    assert!(matches!(
        parse_table("", &schema, None).unwrap_err(),
        TableParserError::EmptyInput
    ));
    assert!(matches!(
        parse_table("x,y\n", &schema, None).unwrap_err(),
        TableParserError::MissingColumn(_)
    ));
    assert!(matches!(
        parse_table("cause,effect\na,b,c\n", &schema, None).unwrap_err(),
        TableParserError::FieldCount(..)
    ));
    assert!(matches!(
        parse_table("cause,effect\n\"oups\n", &schema, None).unwrap_err(),
        TableParserError::UnterminatedQuote(_)
    ));
    // F3 audit : en-tête dupliqué → DuplicateColumn, pas de choix silencieux.
    assert!(matches!(
        parse_table("cause,cause,effect\na,b,c\n", &schema, None).unwrap_err(),
        TableParserError::DuplicateColumn(_)
    ));
}

#[test]
fn cir_json_roundtrip() {
    let (ir, _) = parse_table(CSV, &schema(), None).expect("csv valide");
    let s = serde_json::to_string(&ir).expect("serialize CIR");
    assert!(s.contains("\"table\""), "format table tracé");
    let back: gcn_ir::CausalIR = serde_json::from_str(&s).expect("deserialize CIR");
    assert_eq!(back.edges.len(), 3);
}

// ─── P0-4 : ligne malformée → partiel compté, erreur seulement si tout est HS ─

#[test]
fn p04_field_count_partiel_lignes_valides_conservees() {
    // Ligne 2 : 3 champs au lieu de 2 → ignorée + comptée, les autres passent.
    let csv = "cause,effect\npluie,inondation\ntrop,peu,champs\nsoleil,secheresse\n";
    let (ir, report) = parse_table(csv, &TableSchema::default(), None).expect("partiel OK");
    assert_eq!(ir.edges.len(), 2, "2 lignes valides conservées : {ir:?}");
    assert_eq!(report.skipped_field_count, 1);
    assert_eq!(report.total_skipped(), 1);
}

#[test]
fn p04_field_count_erreur_si_aucune_ligne_valide() {
    // Toutes les lignes malformées → Err (pas un IR vide silencieux).
    let csv = "cause,effect\na,b,c\nd,e,f\n";
    let err = parse_table(csv, &TableSchema::default(), None).unwrap_err();
    assert!(
        matches!(err, TableParserError::FieldCount(..)),
        "attendu FieldCount, obtenu {err:?}"
    );
}
