// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! Bundle STIX 2.x → CausalIR.
//!
//! Patron `gcn-frontend-code` avec `serde_json` à la place de tree-sitter :
//! 1. unités causales = objets STIX du bundle (SDO/SCO) ;
//! 2. signal causal = champ `relationship_type` des objets `relationship` ;
//! 3. `RelationType` via `mapper.rs` (hardcodé, types STIX stables) ;
//! 4. même `CausalIR` que tous les frontends (`SourceSpan::Synthetic`, confiance 1.0).

use std::collections::HashMap;

use gcn_ir::{
    CausalEdge, CausalIR, CausalNode, GraphFormat, IrMetadata, NodeId, NodeOrigin, Provenance,
    Scope, SourceLanguage, SourceSpan, TemporalRef,
};
use smallvec::SmallVec;

use crate::error::GraphParserError;
use crate::mapper::{stix_object_type_to_node_type, stix_relationship_to_relation_type};

/// Compteurs de ce qui n'a PAS produit d'arête — transparence anti-silence (§2.12 ETUDE).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GraphParseReport {
    /// Relations ignorées car `relationship_type` non causal (`related-to`, ...).
    pub skipped_unknown_relationships: usize,
    /// Relations ignorées car `source_ref`/`target_ref` absent du bundle.
    pub skipped_dangling: usize,
    /// Objets ignorés (sans `id` ou sans `type`).
    pub skipped_objects: usize,
}

impl GraphParseReport {
    pub fn total_skipped(&self) -> usize {
        self.skipped_unknown_relationships + self.skipped_dangling + self.skipped_objects
    }
}

/// P0-5 : alias aveugle — jette le `GraphParseReport` (ignorés non visibles).
/// Préférer `parse_bundle_with_report`. Conservé pour compatibilité.
#[deprecated(
    since = "4.0.0",
    note = " Aveugle aux ignorés : utilisez `parse_bundle_with_report` pour obtenir le `GraphParseReport`."
)]
pub fn parse_bundle(json_text: &str) -> Result<CausalIR, GraphParserError> {
    Ok(parse_bundle_with_report(json_text)?.0)
}

pub fn parse_bundle_with_report(
    json_text: &str,
) -> Result<(CausalIR, GraphParseReport), GraphParserError> {
    let bundle: serde_json::Value = serde_json::from_str(json_text)?;
    if bundle.get("type").and_then(|t| t.as_str()) != Some("bundle") {
        return Err(GraphParserError::NotABundle(
            bundle
                .get("type")
                .and_then(|t| t.as_str())
                .map(|s| s.to_string()),
        ));
    }
    let bundle_id = bundle
        .get("id")
        .and_then(|i| i.as_str())
        .map(|s| s.to_string());
    let objects = bundle
        .get("objects")
        .and_then(|o| o.as_array())
        .ok_or(GraphParserError::MissingObjects)?;

    let mut report = GraphParseReport::default();
    let mut nodes: Vec<CausalNode> = Vec::new();
    let mut id_to_node: HashMap<String, NodeId> = HashMap::new();
    let mut next_id: u32 = 0;

    // Passe 1 — nœuds : tout objet non-`relationship` avec id + type.
    for obj in objects {
        if obj.get("type").and_then(|t| t.as_str()) == Some("relationship") {
            continue;
        }
        let (Some(id), Some(stix_type)) = (
            obj.get("id").and_then(|i| i.as_str()),
            obj.get("type").and_then(|t| t.as_str()),
        ) else {
            report.skipped_objects += 1;
            continue;
        };
        let node_id = NodeId(next_id);
        next_id += 1;
        nodes.push(CausalNode {
            id: node_id,
            node_type: stix_object_type_to_node_type(stix_type),
            label: stix_label(obj, id),
            source_span: SourceSpan::Synthetic,
            scope: Scope::Unknown,
            modifiers: SmallVec::new(),
            temporal_ref: TemporalRef::Unresolved,
            temporal_index: Some(node_id.0 as i32),
            origin: NodeOrigin::Explicit,
            attributes: Default::default(),
            parent: None,
            kind: None,
        });
        // Premier objet gagne en cas d'id dupliqué (bundle malformé).
        id_to_node.entry(id.to_string()).or_insert(node_id);
    }

    // Passe 2 — arêtes : objets `relationship` entre nœuds connus.
    let mut edges: Vec<(NodeId, NodeId, CausalEdge)> = Vec::new();
    for obj in objects {
        if obj.get("type").and_then(|t| t.as_str()) != Some("relationship") {
            continue;
        }
        let rel_type = obj
            .get("relationship_type")
            .and_then(|r| r.as_str())
            .unwrap_or("");
        let Some(relation) = stix_relationship_to_relation_type(rel_type) else {
            report.skipped_unknown_relationships += 1;
            continue;
        };
        let (Some(src), Some(dst)) = (
            obj.get("source_ref")
                .and_then(|s| s.as_str())
                .and_then(|s| id_to_node.get(s)),
            obj.get("target_ref")
                .and_then(|t| t.as_str())
                .and_then(|t| id_to_node.get(t)),
        ) else {
            report.skipped_dangling += 1;
            continue;
        };
        edges.push((
            *src,
            *dst,
            CausalEdge {
                relation,
                confidence: 1.0,
                temporal_gap: None,
                explicit: true,
                negated: false,
                marker_token: None,
                in_cycle: None,
                provenance: Some(Provenance::with_ref(
                    bundle_id.clone(),
                    SourceSpan::Synthetic,
                )),
                derivation: None,
                joint_group_id: None,
                third: None,
            },
        ));
    }

    // Ordre déterministe (tri par ids source/destination).
    edges.sort_by_key(|(s, d, _)| (s.0, d.0));

    Ok((
        CausalIR {
            source_lang: SourceLanguage::Graph {
                format: GraphFormat::Stix21,
            },
            source_text: bundle_id.clone().unwrap_or_default(),
            nodes,
            edges,
            cycles: vec![],
            unresolved: vec![],
            metadata: IrMetadata {
                schema_version: "2.0".to_string(),
                pipeline: vec!["gcn-frontend-graph:stix-2.x".to_string()],
                created_at: None,
            },
        },
        report,
    ))
}

/// Libellé de nœud : `name`, sinon premier `labels`, sinon `id` (tronqué à 128 caractères).
fn stix_label(obj: &serde_json::Value, id: &str) -> String {
    let raw = obj
        .get("name")
        .and_then(|n| n.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            obj.get("labels")
                .and_then(|l| l.as_array())
                .and_then(|arr| arr.first())
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| id.to_string());
    raw.trim().chars().take(128).collect()
}
