use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Taxonomy {
    pub taxonomy: String,
    pub version: String,
    pub description: String,
    pub classes: HashMap<String, TaxonomyClass>,
    #[serde(default)]
    pub compositional_rules: Option<Vec<CompositionalRule>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxonomyClass {
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    // Valeurs textuelles (ex: "maintien", "production") — Phase 2 mappera vers CausalDirection enum.
    #[serde(default)]
    pub causal_direction: Option<String>,
    #[serde(default)]
    pub causal_effect: Option<String>,
    #[serde(default)]
    pub dimension: Option<String>,
    #[serde(default)]
    pub relation_type: Option<String>,
    #[serde(default)]
    pub signals_gap: Option<bool>,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub formal: Option<String>,
    #[serde(default)]
    pub position: Option<String>,
    #[serde(default)]
    pub examples_fr: Option<Vec<LexicalEntry>>,
    #[serde(default)]
    pub subtypes: Option<HashMap<String, SubType>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalEntry {
    pub lemma: String,
    #[serde(default)]
    pub note: Option<String>,
    #[serde(default)]
    pub value: Option<f32>,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub anchor: Option<String>,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub total: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubType {
    pub description: String,
    #[serde(default)]
    pub examples_fr: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionalRule {
    pub pattern: String,
    pub result: String,
    pub description: String,
}
