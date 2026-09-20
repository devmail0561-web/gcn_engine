// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use gcn_ir::{CausalCycle, CausalIR};
use serde::{Deserialize, Serialize};

use crate::error::BackendError;
use crate::pearl::{self, CausalLink};

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
}

impl Query {
    pub fn parse(input: &str) -> Result<Self, BackendError> {
        let s = input.trim().trim_end_matches('?').trim();

        if let Some(label) = s.strip_prefix("WHY ") {
            return Ok(Query::Why(label.trim().to_string()));
        }
        if let Some(label) = s.strip_prefix("WHAT ") {
            return Ok(Query::What(label.trim().to_string()));
        }
        if let Some(rest) = s.strip_prefix("CHAIN ") {
            if let Some((from, to)) = rest.split_once("->") {
                return Ok(Query::Chain(from.trim().to_string(), to.trim().to_string()));
            }
            return Err(BackendError::QueryParseError(
                "CHAIN query requires: CHAIN <from> -> <to>".to_string(),
            ));
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

        Err(BackendError::QueryParseError(format!(
            "unknown query '{}'. Valid: WHY <label>, WHAT <label>, CHAIN <a> -> <b>, CYCLES, GAPS, DO <label>, COUNTERFACTUAL <label>",
            input
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

pub fn execute(query: &Query, ir: &CausalIR) -> Result<QueryResult, BackendError> {
    match query {
        Query::Why(label) => {
            let mut results = pearl::why(ir, label);
            if results.is_empty() {
                return Err(BackendError::NodeNotFound(label.clone()));
            }
            let (target, links) = results.drain(..).next().ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
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
            let (source, links) = results.drain(..).next().ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
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
