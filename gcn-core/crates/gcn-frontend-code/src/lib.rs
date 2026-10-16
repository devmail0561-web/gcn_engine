// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

mod common;
mod error;
mod js;
mod kinds;
mod python;
mod rust;

pub use common::{CodeParseReport, MAX_LABEL_CHARS};
pub use error::CodeParserError;

use gcn_ir::CausalIR;
/// Source → CausalIR via tree-sitter (structure seule, zéro YAML).
///
/// P0-3 : les kinds AST non mappés sont ignorés — utilisez
/// `parse_python_with_report` pour les compter.
pub fn parse_python(source: &str) -> Result<CausalIR, CodeParserError> {
    python::parse(source)
}

/// P0-3 : variante avec `CodeParseReport`.
pub fn parse_python_with_report(
    source: &str,
) -> Result<(CausalIR, CodeParseReport), CodeParserError> {
    python::parse_with_report(source)
}

/// Bootstrap annotation tool: auto-annotates Rust source → CausalIR.
/// P0-3 : voir `parse_rust_with_report` pour le rapport d'ignorés.
pub fn parse_rust(source: &str) -> Result<CausalIR, CodeParserError> {
    rust::parse(source)
}

/// P0-3 : variante avec `CodeParseReport`.
pub fn parse_rust_with_report(
    source: &str,
) -> Result<(CausalIR, CodeParseReport), CodeParserError> {
    rust::parse_with_report(source)
}

/// Bootstrap annotation tool: auto-annotates JavaScript source → CausalIR.
/// P0-3 : voir `parse_js_with_report` pour le rapport d'ignorés.
pub fn parse_js(source: &str) -> Result<CausalIR, CodeParserError> {
    js::parse(source)
}

/// P0-3 : variante avec `CodeParseReport`.
pub fn parse_js_with_report(source: &str) -> Result<(CausalIR, CodeParseReport), CodeParserError> {
    js::parse_with_report(source)
}

#[cfg(test)]
mod tests {
    #[test]
    fn python_parser_produces_expected_ast_nodes() {
        let mut p = tree_sitter::Parser::new();
        p.set_language(&tree_sitter_python::LANGUAGE.into())
            .unwrap();

        let tree = p.parse("if x < y:\n    reduce(z)", None).unwrap();
        let root = tree.root_node();
        assert_eq!(root.kind(), "module", "root node doit être 'module'");

        let if_node = (0..root.child_count())
            .filter_map(|i| root.child(i))
            .find(|n| n.kind() == "if_statement");
        assert!(
            if_node.is_some(),
            "if_statement attendu comme enfant du module"
        );

        let tree2 = p.parse("x = compute()", None).unwrap();
        let root2 = tree2.root_node();
        assert_eq!(root2.kind(), "module");
        let assign = (0..root2.child_count())
            .filter_map(|i| root2.child(i))
            .find(|n| n.kind() == "expression_statement" || n.kind() == "assignment");
        assert!(
            assign.is_some(),
            "expression/affectation attendue pour x = compute()"
        );
    }
}
