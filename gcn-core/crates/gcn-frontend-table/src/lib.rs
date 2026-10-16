// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! Frontend tabulaire → CausalIR (P2.3 ETUDE).
//!
//! Un tableau avec colonnes cause/effet est une relation causale structurée :
//! chaque ligne produit un couple (source, target, edge). Le schéma vient des
//! paramètres d'appel — jamais deviné, jamais configuré en dur. Symbolique
//! pur, zéro ML. Savoir compilé dans le moteur (SANS YAML : le moteur
//! n'apprend jamais de configs).
//!
//! Lignes sans arête (cellule vide, relation inconnue, confiance illisible) :
//! **ignorées mais comptées** dans [`TableParseReport`].

mod csv;
mod error;
mod table;

pub use error::TableParserError;
pub use table::{TableParseReport, TableSchema, parse_table};
