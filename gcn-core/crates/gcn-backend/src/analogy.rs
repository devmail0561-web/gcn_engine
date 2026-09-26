// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! Raisonnement par analogie — matching de patterns causaux par signatures topologiques.
//!
//! Algorithme : O(E), pas d'isomorphisme exact (pas de VF2).
//! Pour chaque arête (A→B) du patron, on cherche les arêtes (X→Y) avec la
//! signature la plus proche : même type de relation, mêmes types de nœuds,
//! confiance proche, degré similaire.

use gcn_ir::{CausalIR, NodeId, NodeType, RelationType};

/// Signature topologique d'une arête causale.
#[derive(Debug, Clone)]
pub struct EdgeSignature {
    pub relation: RelationType,
    pub src_type: NodeType,
    pub dst_type: NodeType,
    pub confidence: f32,
    pub src_out_degree: usize,
    pub dst_in_degree: usize,
}

/// Un match analogique : arête (from→to) similaire au patron.
#[derive(Debug, Clone)]
pub struct AnalogyMatch {
    pub from_label: String,
    pub to_label: String,
    pub relation: RelationType,
    pub confidence: f32,
    pub similarity_score: f32,
}

struct DegreeMaps {
    out_deg: std::collections::HashMap<NodeId, usize>,
    in_deg: std::collections::HashMap<NodeId, usize>,
    type_map: std::collections::HashMap<NodeId, NodeType>,
}

impl DegreeMaps {
    fn from_ir(ir: &CausalIR) -> Self {
        let mut out_deg = std::collections::HashMap::new();
        let mut in_deg = std::collections::HashMap::new();
        for (s, d, _) in &ir.edges {
            *out_deg.entry(*s).or_insert(0usize) += 1;
            *in_deg.entry(*d).or_insert(0usize) += 1;
        }
        let type_map = ir.nodes.iter().map(|n| (n.id, n.node_type)).collect();
        Self { out_deg, in_deg, type_map }
    }
}

fn edge_signature(dm: &DegreeMaps, src: NodeId, dst: NodeId, confidence: f32, relation: RelationType) -> EdgeSignature {
    let src_type = dm.type_map.get(&src).copied().unwrap_or(NodeType::Entite);
    let dst_type = dm.type_map.get(&dst).copied().unwrap_or(NodeType::Entite);
    let src_out_degree = dm.out_deg.get(&src).copied().unwrap_or(0);
    let dst_in_degree = dm.in_deg.get(&dst).copied().unwrap_or(0);
    EdgeSignature { relation, src_type, dst_type, confidence, src_out_degree, dst_in_degree }
}

fn relation_similarity(a: RelationType, b: RelationType) -> f32 {
    if a == b { return 1.0; }
    // Famille causale directe : cause, enable, condition
    let causal = [RelationType::Cause, RelationType::Enable, RelationType::Condition];
    // Famille de flux : data_dependency, control_dependency, sequence
    let flow = [RelationType::DataDependency, RelationType::ControlDependency, RelationType::Sequence];
    if causal.contains(&a) && causal.contains(&b) { return 0.6; }
    if flow.contains(&a) && flow.contains(&b) { return 0.6; }
    0.0
}

fn type_similarity(a: NodeType, b: NodeType) -> f32 {
    if a == b { return 1.0; }
    // Action et Transition sont proches
    let dynamic = [NodeType::Action, NodeType::Transition];
    // Etat et EtatSystemique sont proches
    let state = [NodeType::Etat, NodeType::EtatSystemique];
    if dynamic.contains(&a) && dynamic.contains(&b) { return 0.5; }
    if state.contains(&a) && state.contains(&b) { return 0.5; }
    0.0
}

fn similarity(pattern: &EdgeSignature, candidate: &EdgeSignature) -> f32 {
    let rel_sim = relation_similarity(pattern.relation, candidate.relation);
    let src_sim = type_similarity(pattern.src_type, candidate.src_type);
    let dst_sim = type_similarity(pattern.dst_type, candidate.dst_type);
    let type_sim = (src_sim + dst_sim) / 2.0;
    let conf_prox = 1.0 - (pattern.confidence - candidate.confidence).abs().min(1.0);
    let degree_prox = {
        let d_out = (pattern.src_out_degree as i32 - candidate.src_out_degree as i32).unsigned_abs() as f32;
        let d_in  = (pattern.dst_in_degree  as i32 - candidate.dst_in_degree  as i32).unsigned_abs() as f32;
        1.0 / (1.0 + d_out + d_in)
    };
    // Poids : relation > type > confiance > degré
    0.40 * rel_sim + 0.30 * type_sim + 0.20 * conf_prox + 0.10 * degree_prox
}

/// Trouve les arêtes du graphe analogues à l'arête (from→to).
///
/// Retourne les matches triés par score décroissant, en excluant l'arête patron elle-même.
/// `min_score` filtre les résultats trop faibles (défaut recommandé : 0.3).
pub fn find_analogies(ir: &CausalIR, from: &str, to: &str, min_score: f32) -> Vec<AnalogyMatch> {
    use gcn_ir::normalize_label;
    let from_n = normalize_label(from);
    let to_n = normalize_label(to);

    // Trouver l'arête patron
    let src_node = ir.nodes.iter()
        .find(|n| normalize_label(&n.label) == from_n);
    let dst_node = ir.nodes.iter()
        .find(|n| normalize_label(&n.label) == to_n);
    let (src_id, dst_id) = match (src_node, dst_node) {
        (Some(s), Some(d)) => (s.id, d.id),
        _ => return vec![],
    };

    let pattern_edge = ir.edges.iter()
        .find(|(s, d, _)| *s == src_id && *d == dst_id);
    let (_, _, pattern_e) = match pattern_edge {
        Some(e) => e,
        None => return vec![],
    };
    let dm = DegreeMaps::from_ir(ir);
    let pattern_sig = edge_signature(&dm, src_id, dst_id, pattern_e.confidence, pattern_e.relation);

    let nm: std::collections::HashMap<NodeId, &gcn_ir::CausalNode> =
        ir.nodes.iter().map(|n| (n.id, n)).collect();

    let mut matches: Vec<AnalogyMatch> = ir.edges.iter()
        .filter(|(s, d, _)| !(*s == src_id && *d == dst_id))
        .filter_map(|(s, d, e)| {
            let cand_sig = edge_signature(&dm, *s, *d, e.confidence, e.relation);
            let score = similarity(&pattern_sig, &cand_sig);
            if score < min_score { return None; }
            let from_label = nm.get(s).map(|n| n.label.clone()).unwrap_or_default();
            let to_label   = nm.get(d).map(|n| n.label.clone()).unwrap_or_default();
            Some(AnalogyMatch {
                from_label, to_label,
                relation: e.relation,
                confidence: e.confidence,
                similarity_score: score,
            })
        })
        .collect();

    matches.sort_by(|a, b|
        b.similarity_score.partial_cmp(&a.similarity_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.from_label.cmp(&b.from_label))
    );
    matches
}
