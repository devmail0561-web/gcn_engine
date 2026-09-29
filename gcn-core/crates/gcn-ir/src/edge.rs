// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::node::{NodeId, SourceSpan};
use crate::temporal::TemporalGap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationType {
    // 11 relations directes (existantes)
    Cause,
    Enable,
    Prevent,
    Condition,
    Concession,
    Sequence,
    Motivation,
    Filter,
    Opposition,
    DataDependency,
    ControlDependency,
    // 8 nouvelles relations v3.0
    Analogy,
    Counterfactual,
    ConditionalCause,
    MediatedCause,
    JointCause,
    ConditionalPrevent,
    MediatedPrevent,
    JointPrevent,
}

impl RelationType {
    pub fn signals_causal_gap(&self) -> bool {
        matches!(self, RelationType::Concession | RelationType::Opposition)
    }

    pub fn is_joint(&self) -> bool {
        matches!(self, RelationType::JointCause | RelationType::JointPrevent)
    }

    /// Nom canonique snake_case (identique au serde) → `RelationType`.
    /// Source unique des conversions dans le moteur (D1 : pas de match
    /// dupliqué dans les frontends).
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "cause" => Some(RelationType::Cause),
            "enable" => Some(RelationType::Enable),
            "prevent" => Some(RelationType::Prevent),
            "condition" => Some(RelationType::Condition),
            "concession" => Some(RelationType::Concession),
            "sequence" => Some(RelationType::Sequence),
            "motivation" => Some(RelationType::Motivation),
            "filter" => Some(RelationType::Filter),
            "opposition" => Some(RelationType::Opposition),
            "data_dependency" => Some(RelationType::DataDependency),
            "control_dependency" => Some(RelationType::ControlDependency),
            "analogy" => Some(RelationType::Analogy),
            "counterfactual" => Some(RelationType::Counterfactual),
            "conditional_cause" => Some(RelationType::ConditionalCause),
            "mediated_cause" => Some(RelationType::MediatedCause),
            "joint_cause" => Some(RelationType::JointCause),
            "conditional_prevent" => Some(RelationType::ConditionalPrevent),
            "mediated_prevent" => Some(RelationType::MediatedPrevent),
            "joint_prevent" => Some(RelationType::JointPrevent),
            _ => None,
        }
    }
}

/// D2/Éq.12 ETUDE — rôle du tiers dans une relation ternaire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TernaryRole {
    Condition,
    Mediator,
}

/// D2/Éq.12 ETUDE — tiers d'une arête ternaire (condition ou médiateur).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryThird {
    pub role: TernaryRole,
    /// Identifiant du nœud tiers (u64 = NodeId.0).
    pub node: u64,
    /// Polarité du tiers si négation R2 (§9.4) : "negative" ou absent.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub polarity: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CycleId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractionMethod {
    SymbolicRust,
    MlPython,
    SilverAnnotator,
    Derived,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    #[serde(rename = "ref")]
    pub doc_ref: Option<String>,
    pub span: SourceSpan,
    pub extraction_method: ExtractionMethod,
    pub model_version: String,
    pub extracted_at: String,
}

/// Justification d'une relation inférée (A→C déduit de A→B + B→C).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Derivation {
    pub source_edges: Vec<(NodeId, NodeId)>,
}

fn now_utc_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let sec = (secs % 60) as u32;
    let min = ((secs / 60) % 60) as u32;
    let hour = ((secs / 3600) % 24) as u32;
    let days = secs / 86400;
    let (y, m, d) = unix_days_to_date(days);
    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{min:02}:{sec:02}Z")
}

fn is_leap(y: u32) -> bool {
    y.is_multiple_of(4) && (!y.is_multiple_of(100) || y.is_multiple_of(400))
}

fn unix_days_to_date(days: u64) -> (u32, u32, u32) {
    let mut year = 1970u32;
    let mut rem = days as i64;
    loop {
        let dy = if is_leap(year) { 366 } else { 365 };
        if rem < dy {
            break;
        }
        rem -= dy;
        year += 1;
    }
    let md = if is_leap(year) {
        [31u32, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1u32;
    for days_in_month in md {
        if rem < days_in_month as i64 {
            break;
        }
        rem -= days_in_month as i64;
        month += 1;
    }
    (year, month, rem as u32 + 1)
}

impl Provenance {
    pub fn new_symbolic_rust(span: SourceSpan) -> Self {
        Self::with_ref(None, span)
    }

    pub fn with_ref(doc_ref: Option<String>, span: SourceSpan) -> Self {
        Self {
            doc_ref,
            span,
            extraction_method: ExtractionMethod::SymbolicRust,
            model_version: env!("CARGO_PKG_VERSION").to_string(),
            extracted_at: now_utc_iso8601(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEdge {
    pub relation: RelationType,
    pub confidence: f32,
    pub temporal_gap: Option<TemporalGap>,
    pub explicit: bool,
    pub negated: bool,
    pub marker_token: Option<u32>,
    pub in_cycle: Option<CycleId>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub provenance: Option<Provenance>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub derivation: Option<Derivation>,
    /// sha256(target|sorted_sources)[:16] — déterministe, identique sur les 2 arêtes JointCause/JointPrevent
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub joint_group_id: Option<String>,
    /// D2/Éq.12 — tiers de la relation ternaire (condition ou médiateur).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub third: Option<TernaryThird>,
}
