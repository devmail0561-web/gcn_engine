// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! FR lattice integration — état réel, zéro dictionnaire.
//! Remplace l'ancien suite symbolique FrenchParser (supprimé avec gcn-knowledge).

use gcn_frontend_fr::parse_lattice;
use gcn_ir::{LatticeDep, LatticePos};

// ─── Basic API ───────────────────────────────────────────────────────────────

#[test]
fn empty_input_gives_empty_lattice() {
    let lat = parse_lattice("");
    assert!(lat.tokens.is_empty());
    assert!(lat.clauses.is_empty());
    let lat2 = parse_lattice("   ");
    assert!(lat2.tokens.is_empty());
}

#[test]
fn simple_sentence_roles_by_position() {
    let lat = parse_lattice("Le médicament réduit la douleur.");
    let by_form = |f: &str| lat.tokens.iter().find(|t| t.form == f).unwrap().clone();
    assert_eq!(by_form("réduit").pos, LatticePos::Verb);
    assert_eq!(by_form("réduit").dep_rel, LatticeDep::Root);
    assert_eq!(by_form("médicament").dep_rel, LatticeDep::Nsubj);
    assert_eq!(by_form("douleur").dep_rel, LatticeDep::Obj);
    assert!(lat.tokens.iter().all(|t| !t.lemma.is_empty()));
}

#[test]
fn subordinate_mark_and_advcl() {
    // "si" seul token court entre les verbes → marqueur non ambigu.
    let lat = parse_lattice("Il venait si tu venais.");
    let si = lat.tokens.iter().find(|t| t.form == "si").unwrap();
    assert_eq!(si.dep_rel, LatticeDep::Mark);
    let sub = lat.tokens.iter().find(|t| t.form == "venais").unwrap();
    assert_eq!(sub.dep_rel, LatticeDep::Advcl);
    let main = lat
        .tokens
        .iter()
        .find(|t| t.dep_rel == LatticeDep::Root)
        .unwrap();
    assert_eq!(main.form, "venait");
}

#[test]
fn interrogative_inversion_and_true_mark() {
    let lat = parse_lattice("Que se passe-t-il si la demande baisse ?");
    let passe = lat.tokens.iter().find(|t| t.form == "passe").unwrap();
    assert_eq!(passe.pos, LatticePos::Verb);
    let si = lat.tokens.iter().find(|t| t.form == "si").unwrap();
    assert_eq!(si.dep_rel, LatticeDep::Mark);
    assert!(lat.tokens.iter().any(|t| t.form == "?"));
}

#[test]
fn lattice_carries_clauses_and_indexing() {
    let lat = parse_lattice("Si les ventes baissent, on réduit les coûts.");
    assert!(!lat.clauses.is_empty());
    for (i, t) in lat.tokens.iter().enumerate() {
        assert_eq!(t.index, (i + 1) as u32);
    }
}

#[test]
fn no_lexical_decision_in_lattice() {
    // Connecteurs à mot long ("parce") invisibles — limite documentée, pas d'erreur.
    let lat = parse_lattice("Le marché a souffert parce que la demande a chuté.");
    assert!(!lat.tokens.is_empty());
    assert!(lat.tokens.iter().all(|t| !t.lemma.is_empty()));
}
