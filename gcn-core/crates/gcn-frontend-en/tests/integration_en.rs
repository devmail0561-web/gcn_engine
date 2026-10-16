// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! EN lattice integration — état réel, zéro dictionnaire.
//! Remplace l'ancien suite symbolique EnglishParser (supprimé avec gcn-knowledge).

use gcn_frontend_en::parse_lattice;
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
    let lat = parse_lattice("The drug reduces pain.");
    let by_form = |f: &str| lat.tokens.iter().find(|t| t.form == f).unwrap().clone();
    assert_eq!(by_form("reduces").pos, LatticePos::Verb);
    assert_eq!(by_form("reduces").dep_rel, LatticeDep::Root);
    assert!(lat.tokens.iter().all(|t| !t.lemma.is_empty()));
}

#[test]
fn subordinate_mark_and_advcl() {
    // "we" court entre les verbes → "if" marqueur positionnel, subordonnée advcl.
    // (Verbes réguliers -ed : les irréguliers "fell/rose" sont invisibles
    // sans liste — limite documentée du lattice.)
    let lat = parse_lattice("Sales dropped if we crashed.");
    let marks: Vec<_> = lat
        .tokens
        .iter()
        .filter(|t| t.dep_rel == LatticeDep::Mark)
        .collect();
    assert_eq!(marks.len(), 1, "un seul marqueur attendu: {:?}", lat.tokens);
    assert_eq!(marks[0].form, "if");
    let sub: Vec<_> = lat
        .tokens
        .iter()
        .filter(|t| t.dep_rel == LatticeDep::Advcl)
        .collect();
    assert!(!sub.is_empty(), "subordonnée advcl attendue");
}

#[test]
fn lattice_carries_clauses_and_indexing() {
    let lat = parse_lattice("Sales fell, costs rose.");
    assert!(!lat.clauses.is_empty());
    for (i, t) in lat.tokens.iter().enumerate() {
        assert_eq!(t.index, (i + 1) as u32);
    }
}

#[test]
fn no_lexical_decision_in_lattice() {
    // Le lattice ne décide ni type de nœud ni relation : que forme + positions.
    let lat = parse_lattice("Sales fell because costs rose.");
    assert!(!lat.tokens.is_empty());
    // `because` long : invisible comme marqueur (limite documentée), pas d'erreur.
    assert!(lat.tokens.iter().all(|t| !t.lemma.is_empty()));
}
