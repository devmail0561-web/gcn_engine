use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TemporalRef {
    Absolute(i64),
    Relative {
        base: i32,
        offset: i32,
    },
    Epsilon,
    Range {
        start: i32,
        end: i32,
    },
    #[default]
    Unresolved,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalGap {
    pub min: Option<i32>,
    pub max: Option<i32>,
    pub nature: GapNature,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GapNature {
    Immediate,
    Deferred,
    Continuous,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalAnchor {
    Past,
    Present,
    Future,
}
