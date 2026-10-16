// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! Frontend anglais : lattice de tokens sans dictionnaire.
//!
//! Zéro YAML, zéro liste de lemmes. `parse_lattice` produit des tokens
//! observés (forme, POS morphologique, pseudo-dep_rel positionnels) et
//! des découpes de clauses. AUCUNE décision linguistique (ni type de
//! nœud, ni relation) : le ML décide seul en aval.

pub mod lattice;

pub use gcn_ir::Lattice;

/// Analyse un texte anglais → lattice (jamais d'erreur : vide → vide).
/// La langue voyage par le choix de sous-commande CLI (analyze vs analyze-en),
/// jamais dans les données (moteur langue-agnostique).
pub fn parse_lattice(text: &str) -> Lattice {
    lattice::parse_lattice(text)
}
