use serde::{Deserialize, Serialize};

use crate::temporal::TemporalGap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationType {
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
}

impl RelationType {
    pub fn signals_causal_gap(&self) -> bool {
        matches!(self, RelationType::Concession | RelationType::Opposition)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CycleId(pub u32);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEdge {
    pub relation: RelationType,
    pub confidence: f32,
    pub temporal_gap: Option<TemporalGap>,
    pub explicit: bool,
    pub negated: bool,
    pub marker_token: Option<u32>,
    pub in_cycle: Option<CycleId>,
}
