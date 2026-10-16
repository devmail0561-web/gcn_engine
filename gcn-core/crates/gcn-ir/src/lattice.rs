// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! Token lattice — sortie du frontend sans dictionnaire.
//!
//! Le lattice transporte des tokens observés (forme, POS morphologique,
//! rôles syntaxiques lus par POSITION) sans aucune décision linguistique :
//! ni type de nœud, ni relation, ni matching de lemmes. Le ML décide seul
//! en aval. Doctrine : ETUDE_LINGUISTIQUE_NLU §14 (positions), P2.

use serde::{Deserialize, Serialize};

/// Un token observé : forme + POS dérivée de la forme + rôle positionnel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeToken {
    /// Index 1-based dans la phrase.
    pub index: u32,
    /// Forme de surface.
    pub form: String,
    /// POS morpho-positionnelle (jamais issue d'un dictionnaire).
    pub pos: LatticePos,
    /// Rôle syntaxique lu par position relative aux verbes (pseudo-dep_rel).
    /// Bruités mais réels : le ML apprend à les pondérer, le parseur UD
    /// (couche B) les remplacera par des dep_rel authentiques.
    pub dep_rel: LatticeDep,
    /// Tête syntaxique (index 1-based, 0 = racine, -1 = non rattaché).
    pub dep_head: i32,
    /// Lemme dérivé par règles de forme (suffixes), jamais par table.
    pub lemma: String,
    /// Id de la clause (découpe virgule + groupes verbaux).
    pub clause: u32,
    /// Flags morphologiques de forme (imparfait, infinitif...).
    pub flags: Vec<String>,
}

/// POS fermée, dérivée de la forme et de la position uniquement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatticePos {
    Verb,
    Noun,
    Adv,
    Punct,
    Other,
}

/// Rôles syntaxiques positionnels (pseudo-dep_rel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatticeDep {
    /// Verbe principal (tête de phrase).
    Root,
    /// Premier Nom pré-verbal.
    Nsubj,
    /// Premier Nom post-verbal.
    Obj,
    /// Token court devant un verbe subordonné.
    Mark,
    /// Verbe subordonné (tête de clause non principale).
    Advcl,
    /// Token court en position déterminant (devant un Nom pré-verbal).
    Det,
    /// Token court directement pré-verbal (modificateur).
    Advmod,
    /// Ponctuation conservée (., ,, ?, !).
    Punct,
    /// Non rattaché.
    Other,
}

/// Lattice complet d'une phrase : tokens + découpes de clauses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lattice {
    pub source_text: String,
    pub tokens: Vec<LatticeToken>,
    /// Spans de clauses (index 1-based inclusifs).
    pub clauses: Vec<(u32, u32)>,
}

impl LatticeDep {
    /// Mapping vers les dep_rel UD consommées par le ML Python.
    pub fn as_ud(&self) -> &'static str {
        match self {
            LatticeDep::Root => "root",
            LatticeDep::Nsubj => "nsubj",
            LatticeDep::Obj => "obj",
            LatticeDep::Mark => "mark",
            LatticeDep::Advcl => "advcl",
            LatticeDep::Det => "det",
            LatticeDep::Advmod => "advmod",
            LatticeDep::Punct => "punct",
            LatticeDep::Other => "dep",
        }
    }
}

impl LatticePos {
    /// Mapping vers les UPOS consommés par le ML Python.
    pub fn as_upos(&self) -> &'static str {
        match self {
            LatticePos::Verb => "VERB",
            LatticePos::Noun => "NOUN",
            LatticePos::Adv => "ADV",
            LatticePos::Punct => "PUNCT",
            LatticePos::Other => "X",
        }
    }
}
