use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    Universal,
    Existential,
    Partial,
    Null,
    Specific,
    #[default]
    Unknown,
}
