mod common;
mod error;
mod js;
mod mapper;
mod python;
mod resources;
mod rust;

pub use error::CodeParserError;

use std::path::Path;
use gcn_ir::CausalIR;

/// Bootstrap annotation tool: auto-annotates Python source → CausalIR.
/// `taxonomies_root` points to `gcn-references/taxonomies/`.
pub fn parse_python(source: &str, taxonomies_root: &Path) -> Result<CausalIR, CodeParserError> {
    python::parse(source, taxonomies_root)
}

/// Bootstrap annotation tool: auto-annotates Rust source → CausalIR.
pub fn parse_rust(source: &str, taxonomies_root: &Path) -> Result<CausalIR, CodeParserError> {
    rust::parse(source, taxonomies_root)
}

/// Bootstrap annotation tool: auto-annotates JavaScript source → CausalIR.
pub fn parse_js(source: &str, taxonomies_root: &Path) -> Result<CausalIR, CodeParserError> {
    js::parse(source, taxonomies_root)
}

#[cfg(test)]
mod tests {
    #[test]
    fn python_parser_produces_expected_ast_nodes() {
        let mut p = tree_sitter::Parser::new();
        p.set_language(&tree_sitter_python::LANGUAGE.into()).unwrap();

        let tree = p.parse("if x < y:\n    reduce(z)", None).unwrap();
        let root = tree.root_node();
        assert_eq!(root.kind(), "module", "root node doit être 'module'");

        let if_node = (0..root.child_count())
            .filter_map(|i| root.child(i))
            .find(|n| n.kind() == "if_statement");
        assert!(if_node.is_some(), "if_statement attendu comme enfant du module");

        let tree2 = p.parse("x = compute()", None).unwrap();
        let root2 = tree2.root_node();
        assert_eq!(root2.kind(), "module");
        let assign = (0..root2.child_count())
            .filter_map(|i| root2.child(i))
            .find(|n| n.kind() == "expression_statement" || n.kind() == "assignment");
        assert!(assign.is_some(), "expression/affectation attendue pour x = compute()");
    }
}
