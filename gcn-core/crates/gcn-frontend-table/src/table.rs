// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! CSV cause/effet → CausalIR (P2.3 ETUDE).
//!
//! Le schéma (quelles colonnes sont cause/effet/relation/confiance) vient des
//! PARAMÈTRES d'appel (`TableSchema`, exposés en CLI) — jamais du code en dur
//! et jamais d'un fichier de config : une table sans schéma explicite
//! n'a pas de lecture causale définie.
//!
//! Règle ligne → arête : chaque ligne non vide produit un couple
//! (source, target, edge). Symbolique pur, zéro ML.

use std::collections::HashMap;

use gcn_ir::{
    CausalEdge, CausalIR, CausalNode, GraphFormat, IrMetadata, NodeId, NodeOrigin, NodeType,
    Provenance, RelationType, Scope, SourceLanguage, SourceSpan, TemporalRef,
};
use smallvec::SmallVec;

use crate::csv::parse_csv;
use crate::error::TableParserError;

/// Schéma causal d'un CSV — fourni par l'appelant (CLI), pas deviné.
#[derive(Debug, Clone)]
pub struct TableSchema {
    /// Nom exact de la colonne des causes (en-tête CSV).
    pub cause_col: String,
    /// Nom exact de la colonne des effets (en-tête CSV).
    pub effect_col: String,
    /// Colonne optionnelle des relations (valeurs snake_case : `cause`, ...).
    pub relation_col: Option<String>,
    /// Colonne optionnelle des confiances (flottants dans [0, 1]).
    pub confidence_col: Option<String>,
    /// Relation si pas de colonne (ou cellule vide).
    pub default_relation: RelationType,
    /// Type de nœud pour toutes les cellules (les tables ne typent pas).
    pub default_node_type: NodeType,
    /// Délimiteur CSV (défaut `,`).
    pub delimiter: u8,
}

impl Default for TableSchema {
    fn default() -> Self {
        TableSchema {
            cause_col: "cause".to_string(),
            effect_col: "effect".to_string(),
            relation_col: None,
            confidence_col: None,
            default_relation: RelationType::Cause,
            default_node_type: NodeType::Entite,
            delimiter: b',',
        }
    }
}

/// Compteurs de lignes SANS arête — transparence anti-silence.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TableParseReport {
    /// Lignes à cause ou effet vide.
    pub skipped_empty: usize,
    /// Lignes à valeur de relation inconnue.
    pub skipped_unknown_relation: usize,
    /// Lignes à confiance illisible.
    pub skipped_bad_confidence: usize,
}

impl TableParseReport {
    pub fn total_skipped(&self) -> usize {
        self.skipped_empty + self.skipped_unknown_relation + self.skipped_bad_confidence
    }
}

pub fn parse_table(
    csv_text: &str,
    schema: &TableSchema,
    doc_ref: Option<String>,
) -> Result<(CausalIR, TableParseReport), TableParserError> {
    let rows = parse_csv(csv_text, schema.delimiter)?;
    let Some((header, data)) = rows.split_first() else {
        return Err(TableParserError::EmptyInput);
    };
    if header.is_empty() {
        return Err(TableParserError::EmptyInput);
    }

    // F3 audit : en-tête dupliqué → erreur (pas de "première occurrence gagne" silencieux).
    let col = |name: &str| {
        let mut positions = header
            .iter()
            .enumerate()
            .filter(|(_, h)| h.trim() == name)
            .map(|(i, _)| i);
        match (positions.next(), positions.next()) {
            (Some(first), None) => Ok(first),
            (Some(_), Some(_)) => Err(TableParserError::DuplicateColumn(name.to_string())),
            (None, _) => Err(TableParserError::MissingColumn(name.to_string())),
        }
    };
    let cause_idx = col(&schema.cause_col)?;
    let effect_idx = col(&schema.effect_col)?;
    let relation_idx = schema.relation_col.as_deref().map(col).transpose()?;
    let confidence_idx = schema.confidence_col.as_deref().map(col).transpose()?;

    let mut report = TableParseReport::default();
    let mut nodes: Vec<CausalNode> = Vec::new();
    let mut id_by_label: HashMap<String, NodeId> = HashMap::new();
    let mut edges: Vec<(NodeId, NodeId, CausalEdge)> = Vec::new();

    let node_for = |label: &str,
                    nodes: &mut Vec<CausalNode>,
                    id_by_label: &mut HashMap<String, NodeId>|
     -> NodeId {
        if let Some(&id) = id_by_label.get(label) {
            return id;
        }
        let id = NodeId(nodes.len() as u32);
        nodes.push(CausalNode {
            id,
            node_type: schema.default_node_type,
            label: label.chars().take(128).collect(),
            source_span: SourceSpan::Synthetic,
            scope: Scope::Unknown,
            modifiers: SmallVec::new(),
            temporal_ref: TemporalRef::Unresolved,
            temporal_index: Some(id.0 as i32),
            origin: NodeOrigin::Explicit,
            attributes: Default::default(),
            parent: None,
        });
        id_by_label.insert(label.to_string(), id);
        id
    };

    for (row_no, row) in data.iter().enumerate() {
        if row.len() != header.len() {
            return Err(TableParserError::FieldCount(
                row_no + 2,
                header.len(),
                row.len(),
            ));
        }
        let cause = row[cause_idx].trim();
        let effect = row[effect_idx].trim();
        if cause.is_empty() || effect.is_empty() {
            report.skipped_empty += 1;
            continue;
        }
        let relation = match relation_idx.map(|i| row[i].trim()) {
            None | Some("") => schema.default_relation,
            Some(name) => match RelationType::from_name(&name.to_lowercase()) {
                Some(r) => r,
                None => {
                    report.skipped_unknown_relation += 1;
                    continue;
                }
            },
        };
        // F2 audit : hors [0, 1] → ignorée + comptée (pas de clamp silencieux).
        let confidence = match confidence_idx.map(|i| row[i].trim()) {
            None | Some("") => 1.0,
            Some(raw) => match raw.parse::<f32>() {
                Ok(c) if c.is_finite() && (0.0..=1.0).contains(&c) => c,
                _ => {
                    report.skipped_bad_confidence += 1;
                    continue;
                }
            },
        };
        let src = node_for(cause, &mut nodes, &mut id_by_label);
        let dst = node_for(effect, &mut nodes, &mut id_by_label);
        edges.push((
            src,
            dst,
            CausalEdge {
                relation,
                confidence,
                temporal_gap: None,
                explicit: true,
                negated: false,
                marker_token: None,
                in_cycle: None,
                provenance: Some(Provenance::with_ref(doc_ref.clone(), SourceSpan::Synthetic)),
                derivation: None,
                joint_group_id: None,
                third: None,
            },
        ));
    }

    edges.sort_by_key(|(s, d, _)| (s.0, d.0));

    Ok((
        CausalIR {
            source_lang: SourceLanguage::Graph {
                format: GraphFormat::Table,
            },
            source_text: doc_ref.clone().unwrap_or_default(),
            nodes,
            edges,
            cycles: vec![],
            unresolved: vec![],
            metadata: IrMetadata {
                schema_version: "2.0".to_string(),
                pipeline: vec!["gcn-frontend-table:csv".to_string()],
                created_at: None,
            },
        },
        report,
    ))
}
