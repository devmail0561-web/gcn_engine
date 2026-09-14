use serde::{Deserialize, Serialize};

use crate::temporal::TemporalAnchor;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Modifier {
    Intensity {
        value: f32,
        source_form: String,
    },
    Negation {
        total: bool,
    },
    Probability {
        value: f32,
        source_form: String,
    },
    Temporality {
        anchor: TemporalAnchor,
        source_form: String,
    },
    Frequency {
        #[serde(rename = "frequency_kind")]
        kind: FrequencyKind,
        source_form: String,
    },
    Manner {
        description: String,
        source_form: String,
    },
    Location {
        scope: LocationScope,
        source_form: String,
    },
    IntrinsicProperty {
        property: String,
        source_form: String,
    },
    TemporaryState {
        state: String,
        source_form: String,
    },
    Relational {
        relation: String,
        source_form: String,
    },
    Quantitative {
        impact: f32,
        source_form: String,
    },
    TemporalQuality {
        maturity: Maturity,
        source_form: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrequencyKind {
    Always,
    Often,
    Sometimes,
    Rarely,
    Never,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocationScope {
    Local,
    Regional,
    Global,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Maturity {
    New,
    Recent,
    Mature,
    Old,
}
