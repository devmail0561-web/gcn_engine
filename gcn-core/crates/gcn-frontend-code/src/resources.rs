// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::path::Path;

use gcn_ir::{NodeType, RelationType};
use serde::Deserialize;

use crate::error::CodeParserError;
use crate::mapper::{
    label_strategy_str_to_enum, node_type_str_to_enum, relation_type_str_to_enum, LabelStrategy,
};

#[derive(Debug, Deserialize)]
struct AstMapping {
    node_kind: String,
    node_type: String,
    edge_type: String,
    #[serde(default)]
    label_strategy: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AstTaxonomy {
    #[serde(default)]
    transparent: Vec<String>,
    mappings: Vec<AstMapping>,
}

/// Loaded AST-to-causal mapping tables, built from a YAML taxonomy file.
#[derive(Debug)]
pub struct CodeResources {
    pub kind_to_node_type: HashMap<String, NodeType>,
    pub kind_to_edge_type: HashMap<String, RelationType>,
    pub kind_to_label_strategy: HashMap<String, LabelStrategy>,
    /// Grammar wrapper nodes with no causal semantics — recurse into their first named child.
    pub transparent: HashSet<String>,
}

impl CodeResources {
    fn load_from(path: &Path) -> Result<Self, CodeParserError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| CodeParserError::Language(format!("{}: {e}", path.display())))?;
        let taxonomy: AstTaxonomy = yaml_serde::from_str(&content)
            .map_err(|e| CodeParserError::Language(format!("YAML parse error: {e}")))?;

        let mut kind_to_node_type = HashMap::new();
        let mut kind_to_edge_type = HashMap::new();
        let mut kind_to_label_strategy = HashMap::new();

        for m in &taxonomy.mappings {
            if let Some(nt) = node_type_str_to_enum(&m.node_type) {
                kind_to_node_type.insert(m.node_kind.clone(), nt);
            }
            if let Some(rt) = relation_type_str_to_enum(&m.edge_type) {
                kind_to_edge_type.insert(m.node_kind.clone(), rt);
            }
            if let Some(ls) = m.label_strategy.as_deref().and_then(label_strategy_str_to_enum) {
                kind_to_label_strategy.insert(m.node_kind.clone(), ls);
            }
        }

        let transparent = taxonomy.transparent.into_iter().collect();

        Ok(CodeResources { kind_to_node_type, kind_to_edge_type, kind_to_label_strategy, transparent })
    }

    pub fn load_python(taxonomies_root: &Path) -> Result<Self, CodeParserError> {
        Self::load_from(&taxonomies_root.join("python").join("python_ast.yaml"))
    }

    pub fn load_rust(taxonomies_root: &Path) -> Result<Self, CodeParserError> {
        Self::load_from(&taxonomies_root.join("rust").join("rust_ast.yaml"))
    }

    pub fn load_js(taxonomies_root: &Path) -> Result<Self, CodeParserError> {
        Self::load_from(&taxonomies_root.join("js").join("js_ast.yaml"))
    }
}
