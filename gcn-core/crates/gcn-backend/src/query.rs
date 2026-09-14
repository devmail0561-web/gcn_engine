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

        Err(BackendError::QueryParseError(format!(
            "unknown query '{}'. Valid: WHY <label>, WHAT <label>, CHAIN <a> -> <b>, CYCLES, GAPS",
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
            let results = pearl::why(ir, label);
            if results.is_empty() {
                return Err(BackendError::NodeNotFound(label.clone()));
            }
            let (target, links) = results.into_iter().next().unwrap();
            Ok(QueryResult::Causes {
                target,
                links: links.iter().map(link_to_dto).collect(),
            })
        }

        Query::What(label) => {
            let results = pearl::what(ir, label);
            if results.is_empty() {
                return Err(BackendError::NodeNotFound(label.clone()));
            }
            let (source, links) = results.into_iter().next().unwrap();
            Ok(QueryResult::Effects {
                source,
                links: links.iter().map(link_to_dto).collect(),
            })
        }

        Query::Chain(from, to) => {
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
