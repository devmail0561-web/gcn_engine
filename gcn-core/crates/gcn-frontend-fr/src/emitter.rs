// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use crate::annotator::{ClauseAnnotation, SentenceAnnotation};
use gcn_ir::{
    CausalEdge, CausalIR, CausalNode, IrMetadata, Modifier, NaturalLanguage, NodeAttributes,
    NodeId, NodeOrigin, SourceLanguage, SourceSpan, TemporalAnchor, TemporalRef,
};
use smallvec::SmallVec;

pub fn emit(ann: SentenceAnnotation, source_text: String) -> CausalIR {
    let n = ann.clauses.len();
    let mut temporal_indices: Vec<Option<i32>> = vec![None; n];

    // Assign temporal indices from edges (source = earlier in causal order).
    // Propagate max from source so that A→B→C yields 0,1,2 and not 0,1,1.
    for edge in &ann.edges {
        if edge.src_clause >= n || edge.dst_clause >= n {
            continue;
        }
        if temporal_indices[edge.src_clause].is_none() {
            temporal_indices[edge.src_clause] = Some(0);
        }
        let src_ti = temporal_indices[edge.src_clause].unwrap_or(0);
        match temporal_indices[edge.dst_clause] {
            None => temporal_indices[edge.dst_clause] = Some(src_ti + 1),
            Some(existing) if existing <= src_ti => {
                temporal_indices[edge.dst_clause] = Some(src_ti + 1)
            }
            _ => {}
        }
    }
    let mut next_ti = 0i32;
    for ti in temporal_indices.iter_mut() {
        if ti.is_none() {
            *ti = Some(next_ti);
            next_ti += 1;
        }
    }

    let nodes: Vec<CausalNode> = ann
        .clauses
        .iter()
        .enumerate()
        .map(|(i, clause)| build_node(NodeId(i as u32), clause, temporal_indices[i]))
        .collect();

    let mut edges: Vec<(NodeId, NodeId, CausalEdge)> = Vec::new();
    for ea in &ann.edges {
        if ea.src_clause >= n || ea.dst_clause >= n {
            continue;
        }
        edges.push((
            NodeId(ea.src_clause as u32),
            NodeId(ea.dst_clause as u32),
            CausalEdge {
                relation: ea.relation,
                confidence: ea.confidence,
                temporal_gap: None,
                explicit: ea.explicit,
                negated: ea.negated,
                marker_token: ea.marker_token_idx,
                in_cycle: None,
            },
        ));
    }

    CausalIR {
        source_lang: SourceLanguage::Natural {
            lang: NaturalLanguage::French,
        },
        source_text,
        nodes,
        edges,
        cycles: vec![],
        unresolved: vec![],
        metadata: IrMetadata {
            schema_version: "2.0".to_string(),
            pipeline: vec!["gcn-frontend-fr".to_string()],
            created_at: None,
        },
    }
}

fn build_node(id: NodeId, clause: &ClauseAnnotation, temporal_index: Option<i32>) -> CausalNode {
    let mut modifiers: SmallVec<[Modifier; 4]> = SmallVec::new();

    if clause.neg_on_node {
        modifiers.push(Modifier::Negation { total: true });
    }

    let temporal_ref = if clause.has_depuis {
        modifiers.push(Modifier::Temporality {
            anchor: TemporalAnchor::Past,
            source_form: "depuis ...".to_string(),
        });
        TemporalRef::Range { start: -60, end: 0 }
    } else if clause.is_imparfait {
        modifiers.push(Modifier::Temporality {
            anchor: TemporalAnchor::Past,
            source_form: clause.label.clone(),
        });
        TemporalRef::Unresolved
    } else {
        TemporalRef::Unresolved
    };

    let source_span = if clause.origin == NodeOrigin::Hypothetical || clause.span == (0, 0) {
        SourceSpan::Synthetic
    } else {
        SourceSpan::TokenSpan {
            start: clause.span.0,
            end: clause.span.1,
        }
    };

    CausalNode {
        id,
        node_type: clause.node_type,
        label: clause.label.clone(),
        source_span,
        scope: clause.scope,
        modifiers,
        temporal_ref,
        temporal_index,
        origin: clause.origin,
        attributes: NodeAttributes {
            entity: clause.entity.clone(),
            quality: clause.quality.clone(),
            agent: clause.agent.clone(),
            patient: clause.patient.clone(),
            agent_type: clause.agent_type,
            reversible: None,
        },
    }
}
