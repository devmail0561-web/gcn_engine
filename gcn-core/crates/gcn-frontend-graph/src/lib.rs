// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! Frontend graphes typés → CausalIR (P2.1 ETUDE).
//!
//! Couvre STIX 2.x (MITRE ATT&CK et tout bundle STIX) : les relations causales
//! y sont déjà explicites et typées (`relationship_type`), donc pas de ML et
//! pas d'ambiguïté — même patron que `gcn-frontend-code` avec `serde_json`
//! à la place de tree-sitter. Mapping SANS YAML, compilé dans le moteur
//! (le moteur n'apprend jamais de configs).
//!
//! Transparence : les relations non causales (`related-to`, `located-at`, ...)
//! et les références pendantes sont ignorées mais **comptées** — utilisez
//! [`parse_stix_bundle_with_report`] pour les obtenir au lieu de
//! [`parse_stix_bundle`].

mod error;
mod mapper;
mod stix;

pub use error::GraphParserError;
pub use mapper::{
    is_supported_relationship, stix_object_type_to_node_type, stix_relationship_to_relation_type,
};
pub use stix::parse_bundle_with_report as parse_stix_bundle_with_report;
// P0-5 : ré-export de l'alias déprécié (compat) — warning assumé ici, pas chez l'appelant.
#[allow(deprecated)]
pub use stix::parse_bundle as parse_stix_bundle;

use gcn_ir::CausalIR;

use crate::error::GraphParserError as GraphError;

/// Alias conservé pour symétrie avec `gcn-frontend-code::parse_python` et al.
/// P0-5 : aveugle aux ignorés — préférer `parse_stix_bundle_with_report`.
#[deprecated(
    since = "4.0.0",
    note = "Aveugle aux ignorés : utilisez `parse_stix_bundle_with_report`."
)]
pub fn parse_stix(bundle_json: &str) -> Result<CausalIR, GraphError> {
    Ok(parse_stix_bundle_with_report(bundle_json)?.0)
}
