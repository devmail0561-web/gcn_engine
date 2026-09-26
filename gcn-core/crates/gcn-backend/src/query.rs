// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use gcn_ir::{CausalCycle, CausalIR};
use serde::{Deserialize, Serialize};

use crate::error::BackendError;
use crate::pearl::{self, CausalLink, TemporalChainResult, AbductionHypothesis};

fn split_pair(rest: &str, cmd: &str) -> Result<(String, String), BackendError> {
    let (a, b) = rest.split_once("->")
        .or_else(|| rest.split_once(','))
        .ok_or_else(|| BackendError::QueryParseError(
            format!("{cmd} query requires: {cmd} <from> -> <to>")
        ))?;
    let a = a.trim().to_string();
    let b = b.trim().to_string();
    if a.is_empty() || b.is_empty() {
        return Err(BackendError::QueryParseError(
            format!("{cmd}: both arguments must be non-empty")
        ));
    }
    Ok((a, b))
}

#[derive(Debug, Clone, PartialEq)]
pub enum Query {
    Why(String),
    What(String),
    Chain(String, String),
    Cycles,
    Gaps,
    /// Pearl niveau 2 — `DO <nœud>` : intervention do-calculus
    Intervene(String),
    /// Pearl niveau 3 — `COUNTERFACTUAL <nœud>` : raisonnement contrefactuel
    Counterfactual(String),
    /// Abductif — `EXPLAIN <effet>` : hypothèses causales les plus plausibles pour E
    Explain(String),
    /// Analogie — `ANALOGY <from> -> <to>` : patterns causaux similaires au patron (from→to)
    Analogy(String, String),
    /// Multi-échelle — `ZOOM_IN <label>` : enfants directs du nœud dans la hiérarchie
    ZoomIn(String),
    /// Multi-échelle — `ZOOM_OUT <label>` : parent du nœud dans la hiérarchie
    ZoomOut(String),
    /// Multi-échelle — `AGGREGATE <label>` : vue agrégée d'un nœud parent et ses enfants
    Aggregate(String),
    /// Adversarial — `CENTRALITY <label>` : centralité causale pondérée confiance
    Centrality(String),
    /// Adversarial — `SPOF?` : nœuds dont la suppression coupe le plus de paires causales
    Spof,
    /// Normatif — `DIFF <from> -> <to>` : écart entre confiance observée et attente normative (1.0)
    NormDiff(String, String),
    /// Méta — `DENSITY` ou `DENSITY <label>` : densité globale ou locale du graphe
    Density(Option<String>),
    /// Méta — `COVERAGE <label>` : couverture causale autour d'un concept
    Coverage(String),
    /// Méta — `RELIABILITY <label>` : fiabilité des réponses causales pour un concept
    Reliability(String),
    /// Temporal — `CHAIN_T <a> -> <b>` : chemin causal respectant l'ordre temporel
    ChainT(String, String),
    /// Temporal — `BEFORE? <a>, <b>` : A précède-t-il B (temporal_index + chemin orienté) ?
    Before(String, String),
    /// Temporal — `DELAY <a> -> <b>` : délai estimé A→B sur le meilleur chemin
    Delay(String, String),
}

impl Query {
    pub fn parse(input: &str) -> Result<Self, BackendError> {
        let trimmed = input.trim().trim_end_matches('?').trim();
        // Normalize only the keyword (first word) to uppercase — label case is preserved.
        let normalized = match trimmed.find(' ') {
            Some(pos) => format!("{}{}", trimmed[..pos].to_ascii_uppercase(), &trimmed[pos..]),
            None => trimmed.to_ascii_uppercase(),
        };
        let s = normalized.as_str();

        if let Some(label) = s.strip_prefix("WHY ") {
            return Ok(Query::Why(label.trim().to_string()));
        }
        if let Some(label) = s.strip_prefix("WHAT ") {
            return Ok(Query::What(label.trim().to_string()));
        }
        if let Some(rest) = s.strip_prefix("CHAIN ") {
            let (from, to) = split_pair(rest, "CHAIN")?;
            return Ok(Query::Chain(from, to));
        }
        if s == "CYCLES" {
            return Ok(Query::Cycles);
        }
        if s == "GAPS" {
            return Ok(Query::Gaps);
        }
        if let Some(label) = s.strip_prefix("DO ") {
            return Ok(Query::Intervene(label.trim().to_string()));
        }
        if let Some(label) = s.strip_prefix("COUNTERFACTUAL ") {
            return Ok(Query::Counterfactual(label.trim().to_string()));
        }
        if let Some(label) = s.strip_prefix("EXPLAIN ") {
            return Ok(Query::Explain(label.trim().to_string()));
        }
        if let Some(rest) = s.strip_prefix("ANALOGY ") {
            let (from, to) = split_pair(rest, "ANALOGY")?;
            return Ok(Query::Analogy(from, to));
        }
        if let Some(label) = s.strip_prefix("ZOOM_IN ") {
            return Ok(Query::ZoomIn(label.trim().to_string()));
        }
        if let Some(label) = s.strip_prefix("ZOOM_OUT ") {
            return Ok(Query::ZoomOut(label.trim().to_string()));
        }
        if let Some(label) = s.strip_prefix("AGGREGATE ") {
            return Ok(Query::Aggregate(label.trim().to_string()));
        }
        if let Some(label) = s.strip_prefix("CENTRALITY ") {
            return Ok(Query::Centrality(label.trim().to_string()));
        }
        if s == "SPOF" {
            return Ok(Query::Spof);
        }
        if let Some(rest) = s.strip_prefix("DIFF ") {
            let (from, to) = split_pair(rest, "DIFF")?;
            return Ok(Query::NormDiff(from, to));
        }
        if s == "DENSITY" {
            return Ok(Query::Density(None));
        }
        if let Some(label) = s.strip_prefix("DENSITY ") {
            return Ok(Query::Density(Some(label.trim().to_string())));
        }
        if let Some(label) = s.strip_prefix("COVERAGE ") {
            return Ok(Query::Coverage(label.trim().to_string()));
        }
        if let Some(label) = s.strip_prefix("RELIABILITY ") {
            return Ok(Query::Reliability(label.trim().to_string()));
        }
        if let Some(rest) = s.strip_prefix("CHAIN_T ") {
            let (from, to) = split_pair(rest, "CHAIN_T")?;
            return Ok(Query::ChainT(from, to));
        }
        if let Some(rest) = s.strip_prefix("BEFORE? ") {
            let (a, b) = split_pair(rest, "BEFORE?")?;
            return Ok(Query::Before(a, b));
        }
        if let Some(rest) = s.strip_prefix("DELAY ") {
            let (from, to) = split_pair(rest, "DELAY")?;
            return Ok(Query::Delay(from, to));
        }

        Err(BackendError::QueryParseError(format!(
            "unknown query '{}'. Valid: WHY, WHAT, CHAIN, CYCLES, GAPS, DO, COUNTERFACTUAL, EXPLAIN, ANALOGY, CENTRALITY, SPOF?, DIFF, DENSITY, COVERAGE, RELIABILITY, CHAIN_T, BEFORE?, DELAY, ZOOM_IN, ZOOM_OUT, AGGREGATE",
            if input.len() > 200 { &input[..200] } else { input }
        )))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum QueryResult {
    Causes {
        target: String,
        links: Vec<LinkDto>,
    },
    Effects {
        source: String,
        links: Vec<LinkDto>,
    },
    Path {
        from: String,
        to: String,
        found: bool,
        links: Vec<LinkDto>,
    },
    CycleList {
        count: usize,
        cycles: Vec<CycleDto>,
    },
    GapList {
        gap_edges: Vec<GapDto>,
        unresolved_nodes: usize,
    },
    /// Pearl niveau 2 : résultat d'une intervention do-calculus.
    Intervention {
        target: String,
        /// Nombre d'arêtes entrantes coupées par l'intervention.
        severed_count: usize,
        /// Arêtes entrantes supprimées (causes naturelles de `target` court-circuitées).
        severed: Vec<LinkDto>,
        /// Effets en aval de `target` après l'intervention.
        effects: Vec<LinkDto>,
    },
    /// Pearl niveau 3 : résultat d'une requête contrefactuelle.
    CounterfactualDiff {
        target: String,
        /// Tous les effets réels de `target` dans le monde actuel.
        actual_effects: Vec<LinkDto>,
        /// Effets counterfactuellement dépendants de `target` (n'auraient pas eu lieu sans X).
        unique_effects: Vec<String>,
    },
    /// Analogie — patterns causaux similaires au patron (from→to).
    AnalogyReport {
        pattern_from: String,
        pattern_to: String,
        /// Matches triés par score décroissant. Vide si patron introuvable ou graphe sans analogue.
        matches: Vec<AnalogyMatchDto>,
    },
    /// Multi-échelle — enfants directs dans la hiérarchie.
    HierarchyZoomIn {
        parent_label: String,
        children: Vec<NodeSummaryDto>,
    },
    /// Multi-échelle — parent dans la hiérarchie.
    HierarchyZoomOut {
        child_label: String,
        /// None si le nœud est déjà à la racine.
        parent_label: Option<String>,
    },
    /// Multi-échelle — vue agrégée d'un parent et ses enfants.
    HierarchyAggregate {
        parent_label: String,
        child_count: usize,
        /// Toutes les arêtes sortantes des enfants (union).
        outgoing_edges: Vec<LinkDto>,
        /// Confiance moyenne sur ces arêtes.
        mean_confidence: f32,
    },
    /// Adversarial — centralité causale d'un nœud.
    CentralityReport {
        label: String,
        degree_in: usize,
        degree_out: usize,
        /// Somme des confidences sur arêtes entrantes.
        weighted_in: f32,
        /// Somme des confidences sur arêtes sortantes.
        weighted_out: f32,
        /// Centralité composite : (weighted_in + weighted_out) / (degree_in + degree_out). 0 si isolé.
        centrality_score: f32,
    },
    /// Adversarial — SPOF : nœuds dont la suppression coupe le plus de paires.
    SpofReport {
        /// Nombre total de paires (s,d) avec chemin dans le graphe complet.
        total_pairs: usize,
        nodes: Vec<SpofNodeDto>,
    },
    /// Normatif — écart observé vs attente normative (1.0) sur le chemin A→B.
    NormativeDiff {
        from: String,
        to: String,
        /// Le chemin A→B existe-t-il dans le graphe ?
        path_exists: bool,
        /// Nombre d'arêtes sur le chemin.
        n_edges_on_path: usize,
        /// Confiance minimale observée sur le chemin (proxy de la force causale).
        observed_confidence: f32,
        /// Écart absolu : 1.0 − observed_confidence.
        gap: f32,
        /// Taux d'écart en % : gap × 100.
        gap_rate_pct: f32,
        /// Liste des relations sur le chemin avec leur confiance.
        path_links: Vec<LinkDto>,
    },
    /// Méta — densité du graphe ou d'un voisinage local.
    DensityReport {
        /// Densité globale : n_edges / (n_nodes × (n_nodes-1)). 0 si ≤ 1 nœud.
        global_density: f32,
        n_nodes: usize,
        n_edges: usize,
        /// Présent si DENSITY <label> : label du nœud trouvé.
        node_label: Option<String>,
        /// Degré total (entrant + sortant) du nœud si label fourni.
        local_degree: Option<usize>,
        /// Densité locale : local_degree / (2 × (n_nodes-1)). None si global.
        local_density: Option<f32>,
    },
    /// Méta — couverture causale d'un concept.
    CoverageReport {
        label: String,
        /// Nombre d'arêtes connectées à ce nœud (entrant + sortant).
        degree: usize,
        /// Nombre d'arêtes avec provenance renseignée.
        n_with_provenance: usize,
        /// coverage_score = degree / n_edges (fraction du graphe couverte). 0 si aucune arête.
        coverage_score: f32,
        /// provenance_ratio = n_with_provenance / degree. 1.0 si degree=0.
        provenance_ratio: f32,
    },
    /// Méta — fiabilité des réponses causales pour un concept.
    ReliabilityReport {
        label: String,
        /// Confiance minimale sur les arêtes incidentes.
        min_confidence: f32,
        /// Confiance moyenne sur les arêtes incidentes.
        mean_confidence: f32,
        /// Fraction d'arêtes incidentes avec provenance. 1.0 si aucune arête.
        provenance_ratio: f32,
        /// reliability_score = mean_confidence × provenance_ratio.
        reliability_score: f32,
    },
    /// Abductif — hypothèses causales pour un effet observé, triées par score.
    Abduction {
        effect: String,
        hypotheses: Vec<AbductionHypothesisDto>,
    },
    /// Temporal — chemin causal respectant l'ordre temporal_index.
    TemporalPath {
        from: String,
        to: String,
        found: bool,
        temporally_ordered: bool,
        links: Vec<TemporalLinkDto>,
    },
    /// Temporal — ordre relatif de deux nœuds.
    TemporalOrder {
        a: String,
        b: String,
        /// true si a.temporal_index < b.temporal_index ET chemin orienté a→b existe.
        a_before_b: bool,
        a_index: Option<i32>,
        b_index: Option<i32>,
        path_exists: bool,
    },
    /// Temporal — délai estimé A→B.
    TemporalDelay {
        from: String,
        to: String,
        found: bool,
        /// Somme des TemporalGap.min sur le chemin (unités = unités du domaine).
        gap_min_sum: Option<i32>,
        gap_max_sum: Option<i32>,
        /// Différence de temporal_index comme proxy si pas de TemporalGap.
        index_delta: Option<i32>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkDto {
    pub from: String,
    pub to: String,
    pub relation: String,
    pub confidence: f32,
    pub negated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleDto {
    pub id: u32,
    pub kind: String,
    pub nodes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapDto {
    pub from: String,
    pub to: String,
    pub relation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalogyMatchDto {
    pub from_label: String,
    pub to_label: String,
    pub relation: String,
    pub confidence: f32,
    pub similarity_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSummaryDto {
    pub label: String,
    pub node_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpofNodeDto {
    pub label: String,
    /// Paires (s,d) déconnectées par la suppression de ce nœud.
    pub paths_cut: usize,
    /// spof_score = paths_cut / total_pairs. 0 si total_pairs=0.
    pub spof_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbductionHypothesisDto {
    /// Label du nœud hypothèse (cause candidate).
    pub label: String,
    /// Relation de ce nœud vers son successeur immédiat dans le chemin vers l'effet.
    pub relation: String,
    /// Confiance de l'arête immédiate de ce nœud vers son successeur dans le chemin.
    pub edge_confidence: f32,
    /// Score = min_conf_on_path / (1 + depth).
    pub score: f32,
    /// Nombre de sauts depuis cette hypothèse jusqu'à l'effet.
    pub depth: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalLinkDto {
    pub from: String,
    pub to: String,
    pub relation: String,
    pub confidence: f32,
    pub negated: bool,
    pub gap_min: Option<i32>,
    pub gap_max: Option<i32>,
    pub index_delta: Option<i32>,
}

pub fn execute(query: &Query, ir: &CausalIR) -> Result<QueryResult, BackendError> {
    match query {
        Query::Why(label) => {
            let mut results = pearl::why(ir, label);
            if results.is_empty() {
                return Err(BackendError::NodeNotFound(label.clone()));
            }
            let (target, links) = results
                .drain(..)
                .next()
                .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
            Ok(QueryResult::Causes {
                target,
                links: links.iter().map(link_to_dto).collect(),
            })
        }

        Query::What(label) => {
            let mut results = pearl::what(ir, label);
            if results.is_empty() {
                return Err(BackendError::NodeNotFound(label.clone()));
            }
            let (source, links) = results
                .drain(..)
                .next()
                .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
            Ok(QueryResult::Effects {
                source,
                links: links.iter().map(link_to_dto).collect(),
            })
        }

        Query::Chain(from, to) => {
            // Distingue nœud introuvable (NodeNotFound) de l'absence de chemin (found:false).
            if pearl::find_best(ir, from).is_none() {
                return Err(BackendError::NodeNotFound(from.clone()));
            }
            if pearl::find_best(ir, to).is_none() {
                return Err(BackendError::NodeNotFound(to.clone()));
            }
            let links = pearl::chain(ir, from, to);
            let found = links.is_some();
            Ok(QueryResult::Path {
                from: from.clone(),
                to: to.clone(),
                found,
                links: links.unwrap_or_default().iter().map(link_to_dto).collect(),
            })
        }

        Query::Cycles => Ok(QueryResult::CycleList {
            count: ir.cycles.len(),
            cycles: ir.cycles.iter().map(cycle_to_dto(ir)).collect(),
        }),

        Query::Gaps => {
            let gap_edges: Vec<GapDto> = ir
                .edges
                .iter()
                .filter(|(_, _, e)| e.temporal_gap.is_some())
                .map(|(s, d, e)| GapDto {
                    from: pearl::node_label(ir, *s),
                    to: pearl::node_label(ir, *d),
                    relation: format!("{:?}", e.relation),
                })
                .collect();
            Ok(QueryResult::GapList {
                gap_edges,
                unresolved_nodes: ir.unresolved.len(),
            })
        }

        Query::Intervene(label) => {
            let (target, result) = pearl::intervene(ir, label)
                .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
            Ok(QueryResult::Intervention {
                target,
                severed_count: result.severed.len(),
                severed: result.severed.iter().map(link_to_dto).collect(),
                effects: result.effects.iter().map(link_to_dto).collect(),
            })
        }

        Query::Counterfactual(label) => {
            let (target, result) = pearl::counterfactual(ir, label)
                .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
            Ok(QueryResult::CounterfactualDiff {
                target,
                actual_effects: result.actual_effects.iter().map(link_to_dto).collect(),
                unique_effects: result.unique_effects,
            })
        }

        Query::Analogy(from, to) => {
            let matches = crate::analogy::find_analogies(ir, from, to, 0.3);
            Ok(QueryResult::AnalogyReport {
                pattern_from: from.clone(),
                pattern_to: to.clone(),
                matches: matches.into_iter().map(|m| AnalogyMatchDto {
                    from_label: m.from_label,
                    to_label: m.to_label,
                    relation: format!("{:?}", m.relation),
                    confidence: m.confidence,
                    similarity_score: m.similarity_score,
                }).collect(),
            })
        }

        Query::ZoomIn(label) => {
            let node = pearl::find_best(ir, label)
                .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
            let children: Vec<NodeSummaryDto> = ir.nodes.iter()
                .filter(|n| n.parent == Some(node.id))
                .map(|n| NodeSummaryDto {
                    label: n.label.clone(),
                    node_type: format!("{:?}", n.node_type),
                })
                .collect();
            Ok(QueryResult::HierarchyZoomIn {
                parent_label: node.label.clone(),
                children,
            })
        }

        Query::ZoomOut(label) => {
            let node = pearl::find_best(ir, label)
                .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
            let parent_label = node.parent
                .and_then(|pid| ir.nodes.iter().find(|n| n.id == pid))
                .map(|n| n.label.clone());
            Ok(QueryResult::HierarchyZoomOut {
                child_label: node.label.clone(),
                parent_label,
            })
        }

        Query::Aggregate(label) => {
            let node = pearl::find_best(ir, label)
                .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
            let child_ids: std::collections::HashSet<_> = ir.nodes.iter()
                .filter(|n| n.parent == Some(node.id))
                .map(|n| n.id)
                .collect();
            let child_count = child_ids.len();
            let (nm, _) = pearl::build_maps_pub(ir);
            let outgoing: Vec<LinkDto> = ir.edges.iter()
                .filter(|(s, _, _)| child_ids.contains(s))
                .filter(|(_, d, _)| !child_ids.contains(d))
                .map(|(s, d, e)| LinkDto {
                    from: nm.get(s).map(|n| n.label.clone()).unwrap_or_default(),
                    to: nm.get(d).map(|n| n.label.clone()).unwrap_or_default(),
                    relation: format!("{:?}", e.relation),
                    confidence: e.confidence,
                    negated: e.negated,
                })
                .collect();
            let mean_confidence = if outgoing.is_empty() { 0.0 } else {
                outgoing.iter().map(|l| l.confidence).sum::<f32>() / outgoing.len() as f32
            };
            Ok(QueryResult::HierarchyAggregate {
                parent_label: node.label.clone(),
                child_count,
                outgoing_edges: outgoing,
                mean_confidence,
            })
        }

        Query::Centrality(label) => {
            let node = pearl::find_best(ir, label)
                .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
            let in_edges: Vec<f32> = ir.edges.iter()
                .filter(|(_, d, _)| *d == node.id)
                .map(|(_, _, e)| e.confidence)
                .collect();
            let out_edges: Vec<f32> = ir.edges.iter()
                .filter(|(s, _, _)| *s == node.id)
                .map(|(_, _, e)| e.confidence)
                .collect();
            let degree_in = in_edges.len();
            let degree_out = out_edges.len();
            let weighted_in: f32 = in_edges.iter().sum();
            let weighted_out: f32 = out_edges.iter().sum();
            let total_degree = degree_in + degree_out;
            let centrality_score = if total_degree > 0 {
                (weighted_in + weighted_out) / total_degree as f32
            } else { 0.0 };
            Ok(QueryResult::CentralityReport {
                label: node.label.clone(),
                degree_in, degree_out, weighted_in, weighted_out, centrality_score,
            })
        }

        Query::Spof => {
            const SPOF_MAX_NODES: usize = 500;
            if ir.nodes.len() > SPOF_MAX_NODES {
                return Err(BackendError::QueryParseError(format!(
                    "SPOF: graph too large ({} nodes > {}). Use CENTRALITY on specific nodes instead.",
                    ir.nodes.len(), SPOF_MAX_NODES
                )));
            }
            let (total_pairs, scores) = pearl::spof_all(ir);
            let spof_nodes: Vec<SpofNodeDto> = scores.into_iter()
                .map(|(label, paths_cut)| {
                    let spof_score = if total_pairs > 0 {
                        paths_cut as f32 / total_pairs as f32
                    } else { 0.0 };
                    SpofNodeDto { label, paths_cut, spof_score }
                })
                .collect();
            Ok(QueryResult::SpofReport { total_pairs, nodes: spof_nodes })
        }

        Query::NormDiff(from, to) => {
            if pearl::find_best(ir, from).is_none() {
                return Err(BackendError::NodeNotFound(from.clone()));
            }
            if pearl::find_best(ir, to).is_none() {
                return Err(BackendError::NodeNotFound(to.clone()));
            }
            let links = pearl::chain(ir, from, to);
            let path_exists = links.is_some();
            let path_links_dto: Vec<LinkDto> = links
                .as_ref()
                .map(|ls| ls.iter().map(link_to_dto).collect())
                .unwrap_or_default();
            let n_edges_on_path = path_links_dto.len();
            let observed_confidence = links
                .as_ref()
                .and_then(|ls| ls.iter().map(|l| l.confidence).reduce(f32::min))
                .unwrap_or(0.0);
            let gap = if path_exists { 1.0 - observed_confidence } else { 1.0 };
            Ok(QueryResult::NormativeDiff {
                from: from.clone(),
                to: to.clone(),
                path_exists,
                n_edges_on_path,
                observed_confidence,
                gap,
                gap_rate_pct: gap * 100.0,
                path_links: path_links_dto,
            })
        }

        Query::Density(label_opt) => {
            let n_nodes = ir.nodes.len();
            let n_edges = ir.edges.len();
            let global_density = if n_nodes > 1 {
                n_edges as f32 / (n_nodes * (n_nodes - 1)) as f32
            } else {
                0.0
            };
            if let Some(label) = label_opt {
                let node = pearl::find_best(ir, label)
                    .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
                let local_degree = ir.edges.iter()
                    .filter(|(s, d, _)| *s == node.id || *d == node.id)
                    .count();
                let local_density = if n_nodes > 1 {
                    Some(local_degree as f32 / (2 * (n_nodes - 1)) as f32)
                } else {
                    None
                };
                Ok(QueryResult::DensityReport {
                    global_density, n_nodes, n_edges,
                    node_label: Some(node.label.clone()),
                    local_degree: Some(local_degree),
                    local_density,
                })
            } else {
                Ok(QueryResult::DensityReport {
                    global_density, n_nodes, n_edges,
                    node_label: None, local_degree: None, local_density: None,
                })
            }
        }

        Query::Coverage(label) => {
            let node = pearl::find_best(ir, label)
                .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
            let incident: Vec<_> = ir.edges.iter()
                .filter(|(s, d, _)| *s == node.id || *d == node.id)
                .collect();
            let degree = incident.len();
            let n_with_provenance = incident.iter()
                .filter(|(_, _, e)| e.provenance.is_some())
                .count();
            let n_edges = ir.edges.len();
            let coverage_score = if n_edges > 0 {
                degree as f32 / n_edges as f32
            } else { 0.0 };
            let provenance_ratio = if degree > 0 {
                n_with_provenance as f32 / degree as f32
            } else { 1.0 };
            Ok(QueryResult::CoverageReport {
                label: node.label.clone(),
                degree, n_with_provenance, coverage_score, provenance_ratio,
            })
        }

        Query::Reliability(label) => {
            let node = pearl::find_best(ir, label)
                .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
            let incident: Vec<_> = ir.edges.iter()
                .filter(|(s, d, _)| *s == node.id || *d == node.id)
                .map(|(_, _, e)| e)
                .collect();
            let degree = incident.len();
            let (min_confidence, mean_confidence) = if degree > 0 {
                let min = incident.iter().map(|e| e.confidence)
                    .fold(f32::INFINITY, f32::min);
                let mean = incident.iter().map(|e| e.confidence).sum::<f32>() / degree as f32;
                (min, mean)
            } else { (0.0, 0.0) };
            let provenance_ratio = if degree > 0 {
                incident.iter().filter(|e| e.provenance.is_some()).count() as f32 / degree as f32
            } else { 1.0 };
            let reliability_score = mean_confidence * provenance_ratio;
            Ok(QueryResult::ReliabilityReport {
                label: node.label.clone(),
                min_confidence, mean_confidence, provenance_ratio, reliability_score,
            })
        }

        Query::Explain(effect) => {
            let node = pearl::find_best(ir, effect)
                .ok_or_else(|| BackendError::NodeNotFound(effect.clone()))?;
            let hypotheses = pearl::abduct(ir, node.id);
            Ok(QueryResult::Abduction {
                effect: node.label.clone(),
                hypotheses: hypotheses
                    .into_iter()
                    .map(|h| AbductionHypothesisDto {
                        label: h.label,
                        relation: format!("{:?}", h.relation),
                        edge_confidence: h.edge_confidence,
                        score: h.score,
                        depth: h.depth,
                    })
                    .collect(),
            })
        }

        Query::ChainT(from, to) => {
            if pearl::find_best(ir, from).is_none() {
                return Err(BackendError::NodeNotFound(from.clone()));
            }
            if pearl::find_best(ir, to).is_none() {
                return Err(BackendError::NodeNotFound(to.clone()));
            }
            let result = pearl::chain_temporal(ir, from, to);
            let found = result.links.is_some();
            Ok(QueryResult::TemporalPath {
                from: from.clone(),
                to: to.clone(),
                found,
                temporally_ordered: result.temporally_ordered,
                links: result.links.unwrap_or_default().iter().map(temporal_link_to_dto).collect(),
            })
        }

        Query::Before(a, b) => {
            let node_a = pearl::find_best(ir, a)
                .ok_or_else(|| BackendError::NodeNotFound(a.clone()))?;
            let node_b = pearl::find_best(ir, b)
                .ok_or_else(|| BackendError::NodeNotFound(b.clone()))?;
            let a_index = node_a.temporal_index;
            let b_index = node_b.temporal_index;
            let path_exists = pearl::chain(ir, a, b).is_some();
            let a_before_b = match (a_index, b_index) {
                (Some(ai), Some(bi)) => ai < bi && path_exists,
                _ => path_exists,
            };
            Ok(QueryResult::TemporalOrder {
                a: node_a.label.clone(),
                b: node_b.label.clone(),
                a_before_b,
                a_index,
                b_index,
                path_exists,
            })
        }

        Query::Delay(from, to) => {
            if pearl::find_best(ir, from).is_none() {
                return Err(BackendError::NodeNotFound(from.clone()));
            }
            if pearl::find_best(ir, to).is_none() {
                return Err(BackendError::NodeNotFound(to.clone()));
            }
            let result = pearl::chain_temporal(ir, from, to);
            let (gap_min_sum, gap_max_sum, index_delta) = if let Some(ref links) = result.links {
                let min_sum: Option<i32> = links.iter()
                    .map(|l| l.gap_min)
                    .try_fold(0i32, |acc, v| v.map(|x| acc + x));
                let max_sum: Option<i32> = links.iter()
                    .map(|l| l.gap_max)
                    .try_fold(0i32, |acc, v| v.map(|x| acc + x));
                let idx_delta = links.iter()
                    .filter_map(|l| l.index_delta)
                    .reduce(|a, b| a + b);
                (min_sum, max_sum, idx_delta)
            } else {
                (None, None, None)
            };
            Ok(QueryResult::TemporalDelay {
                from: from.clone(),
                to: to.clone(),
                found: result.links.is_some(),
                gap_min_sum,
                gap_max_sum,
                index_delta,
            })
        }
    }
}

/// Variante stricte de CHAIN : erreur `NoPath` si les nœuds existent mais sans
/// chemin causal (rend `BackendError::NoPath` atteignable).
pub fn chain_strict(ir: &CausalIR, from: &str, to: &str) -> Result<Vec<LinkDto>, BackendError> {
    if pearl::find_best(ir, from).is_none() {
        return Err(BackendError::NodeNotFound(from.to_string()));
    }
    if pearl::find_best(ir, to).is_none() {
        return Err(BackendError::NodeNotFound(to.to_string()));
    }
    pearl::chain(ir, from, to)
        .map(|links| links.iter().map(link_to_dto).collect())
        .ok_or_else(|| BackendError::NoPath(from.to_string(), to.to_string()))
}

fn temporal_link_to_dto(l: &pearl::TemporalLink) -> TemporalLinkDto {
    TemporalLinkDto {
        from: l.from_label.clone(),
        to: l.to_label.clone(),
        relation: format!("{:?}", l.relation),
        confidence: l.confidence,
        negated: l.negated,
        gap_min: l.gap_min,
        gap_max: l.gap_max,
        index_delta: l.index_delta,
    }
}

fn link_to_dto(l: &CausalLink) -> LinkDto {
    LinkDto {
        from: l.from_label.clone(),
        to: l.to_label.clone(),
        relation: format!("{:?}", l.relation),
        confidence: l.confidence,
        negated: l.negated,
    }
}

fn cycle_to_dto(ir: &CausalIR) -> impl Fn(&CausalCycle) -> CycleDto + '_ {
    move |c| CycleDto {
        id: c.id.0,
        kind: format!("{:?}", c.cycle_type),
        nodes: c.path.iter().map(|&id| pearl::node_label(ir, id)).collect(),
    }
}
