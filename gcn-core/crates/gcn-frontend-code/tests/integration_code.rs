// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use gcn_frontend_code::{parse_js, parse_python, parse_rust};
use gcn_ir::{NodeType, RelationType};

// ---------------------------------------------------------------------------
// Fonctionnalité de base du parseur Python
// ---------------------------------------------------------------------------

#[test]
fn test_python_parses_if_statement() {
    let ir = parse_python("if x < y:\n    reduce(z)").expect("parse failed");
    assert!(!ir.nodes.is_empty(), "should produce nodes");
    assert!(!ir.edges.is_empty(), "should produce edges");
}

#[test]
fn test_python_if_produces_condition_node() {
    let ir = parse_python("if x < y:\n    reduce(z)").expect("parse failed");
    let has_condition = ir
        .nodes
        .iter()
        .any(|n| n.node_type == gcn_ir::NodeType::Condition);
    assert!(
        has_condition,
        "if_statement should produce a Condition node"
    );
}

#[test]
fn test_python_if_produces_condition_edge() {
    let ir = parse_python("if x < y:\n    reduce(z)").expect("parse failed");
    let has_cond_edge = ir
        .edges
        .iter()
        .any(|(_, _, e)| e.relation == RelationType::Condition);
    assert!(
        has_cond_edge,
        "if_statement should produce a Condition edge"
    );
}

#[test]
fn test_python_for_produces_processus_node() {
    let ir = parse_python("for i in items:\n    process(i)").expect("parse failed");
    let has_proc = ir
        .nodes
        .iter()
        .any(|n| n.node_type == gcn_ir::NodeType::Processus);
    assert!(has_proc, "for_statement should produce a Processus node");
}

#[test]
fn test_python_assignment_produces_transition() {
    let ir = parse_python("x = compute()").expect("parse failed");
    let has_trans = ir
        .nodes
        .iter()
        .any(|n| n.node_type == gcn_ir::NodeType::Processus);
    assert!(has_trans, "assignment should produce a Processus node");
}

#[test]
fn test_python_source_lang_is_python() {
    let ir = parse_python("x = 1").expect("parse failed");
    assert!(matches!(
        ir.source_lang,
        gcn_ir::SourceLanguage::Programming {
            lang: gcn_ir::ProgrammingLanguage::Python
        }
    ));
}

#[test]
fn test_python_empty_produces_empty_ir() {
    let ir = parse_python("").expect("parse failed");
    assert!(ir.nodes.is_empty());
    assert!(ir.edges.is_empty());
}

// ---------------------------------------------------------------------------
// Isomorphisme fr ↔ python : critère SAD Phase 6
// Isomorphisme NL retiré : le frontend FR émet un lattice (zéro décision),
// pas un CIR — aucune comparaison CIR py↔fr possible. Le cross-langage
// code est couvert par test_isomorphism_python_rust_js_if_structure.

// ---------------------------------------------------------------------------
// Arêtes séquentielles : deux instructions consécutives → Sequence, pas Condition
// ---------------------------------------------------------------------------

#[test]
fn test_sequential_statements_use_sequence_edge() {
    let ir = parse_python("x = compute()\nif x < y:\n    act()").expect("parse failed");

    let transition_ids: Vec<_> = ir
        .nodes
        .iter()
        .filter(|n| n.node_type == NodeType::Processus)
        .map(|n| n.id)
        .collect();
    let condition_ids: Vec<_> = ir
        .nodes
        .iter()
        .filter(|n| n.node_type == NodeType::Condition)
        .map(|n| n.id)
        .collect();

    assert!(
        !transition_ids.is_empty(),
        "should have Processus node (x = compute())"
    );
    assert!(
        !condition_ids.is_empty(),
        "should have Condition node (if x < y)"
    );

    // The edge between the assignment and the if should be Sequence, not Condition
    let seq_edge = ir.edges.iter().any(|(src, dst, e)| {
        transition_ids.contains(src)
            && condition_ids.contains(dst)
            && e.relation == RelationType::Sequence
    });
    assert!(
        seq_edge,
        "consecutive statements should be linked by a Sequence edge"
    );
}

// ---------------------------------------------------------------------------
// walk_try : les statements du bloc except doivent apparaître dans le CIR
// ---------------------------------------------------------------------------

#[test]
fn test_try_except_body_in_ir() {
    let ir = parse_python("try:\n    x = risky()\nexcept ValueError:\n    handle_error()")
        .expect("parse failed");

    // handle_error() should appear as a Processus node in the IR
    let has_handle = ir.nodes.iter().any(|n| n.label.contains("handle_error"));
    assert!(
        has_handle,
        "handler statement inside except should produce a node in the CIR"
    );
}

// ---------------------------------------------------------------------------
// node_label : pas de troncature sur le premier ':' pour les fonctions typées
// ---------------------------------------------------------------------------

#[test]
fn test_function_label_uses_name_not_colon_truncation() {
    let ir = parse_python("def compute(x: int) -> int:\n    return x * 2").expect("parse failed");

    let fn_node = ir.nodes.iter().find(|n| n.node_type == NodeType::Processus);
    assert!(
        fn_node.is_some(),
        "function_definition should produce a Processus node"
    );

    let label = &fn_node.unwrap().label;
    // Must NOT be "def compute(x" (truncated at first colon inside parameters)
    assert!(
        !label.starts_with("def compute(x") || label.len() > "def compute(x".len(),
        "label should not be truncated at the type annotation colon: got {label:?}"
    );
    assert!(
        label.contains("compute"),
        "label should contain the function name"
    );
}

// ---------------------------------------------------------------------------
// Frontend Rust
// ---------------------------------------------------------------------------

#[test]
fn test_rust_parses_if_expression() {
    let ir = parse_rust("fn main() {\n    if x < y {\n        reduce(z);\n    }\n}")
        .expect("parse failed");
    assert!(!ir.nodes.is_empty());
    assert!(!ir.edges.is_empty());
}

#[test]
fn test_rust_if_produces_condition_node() {
    let ir = parse_rust("fn main() {\n    if condition {\n        action();\n    }\n}")
        .expect("parse failed");
    assert!(ir.nodes.iter().any(|n| n.node_type == NodeType::Condition));
}

#[test]
fn test_rust_if_produces_condition_edge() {
    let ir =
        parse_rust("fn main() {\n    if x {\n        do_it();\n    }\n}").expect("parse failed");
    assert!(
        ir.edges
            .iter()
            .any(|(_, _, e)| e.relation == RelationType::Condition)
    );
}

#[test]
fn test_rust_fn_item_produces_action_node() {
    let ir = parse_rust("fn compute(x: i32) -> i32 { x * 2 }").expect("parse failed");
    assert!(ir.nodes.iter().any(|n| n.node_type == NodeType::Processus));
}

#[test]
fn test_rust_let_produces_transition_node() {
    let ir = parse_rust("fn main() { let x = compute(); }").expect("parse failed");
    assert!(ir.nodes.iter().any(|n| n.node_type == NodeType::Processus));
}

#[test]
fn test_rust_source_lang_is_rust() {
    let ir = parse_rust("fn main() {}").expect("parse failed");
    assert!(matches!(
        ir.source_lang,
        gcn_ir::SourceLanguage::Programming {
            lang: gcn_ir::ProgrammingLanguage::Rust
        }
    ));
}

#[test]
fn test_rust_empty_produces_empty_ir() {
    let ir = parse_rust("").expect("parse failed");
    assert!(ir.nodes.is_empty());
    assert!(ir.edges.is_empty());
}

// ---------------------------------------------------------------------------
// Frontend JavaScript
// ---------------------------------------------------------------------------

#[test]
fn test_js_parses_if_statement() {
    let ir = parse_js("if (x < y) {\n    reduce(z);\n}").expect("parse failed");
    assert!(!ir.nodes.is_empty());
    assert!(!ir.edges.is_empty());
}

#[test]
fn test_js_if_produces_condition_node() {
    let ir = parse_js("if (condition) {\n    action();\n}").expect("parse failed");
    assert!(ir.nodes.iter().any(|n| n.node_type == NodeType::Condition));
}

#[test]
fn test_js_if_produces_condition_edge() {
    let ir = parse_js("if (x) {\n    doIt();\n}").expect("parse failed");
    assert!(
        ir.edges
            .iter()
            .any(|(_, _, e)| e.relation == RelationType::Condition)
    );
}

#[test]
fn test_js_function_produces_action_node() {
    let ir = parse_js("function compute(x) { return x * 2; }").expect("parse failed");
    assert!(ir.nodes.iter().any(|n| n.node_type == NodeType::Processus));
}

#[test]
fn test_js_variable_declaration_produces_transition() {
    let ir = parse_js("const x = compute();").expect("parse failed");
    assert!(ir.nodes.iter().any(|n| n.node_type == NodeType::Processus));
}

#[test]
fn test_js_source_lang_is_javascript() {
    let ir = parse_js("const x = 1;").expect("parse failed");
    assert!(matches!(
        ir.source_lang,
        gcn_ir::SourceLanguage::Programming {
            lang: gcn_ir::ProgrammingLanguage::JavaScript
        }
    ));
}

#[test]
fn test_js_empty_produces_empty_ir() {
    let ir = parse_js("").expect("parse failed");
    assert!(ir.nodes.is_empty());
    assert!(ir.edges.is_empty());
}

// ---------------------------------------------------------------------------
// Rust : impl_item — les méthodes internes doivent apparaître dans le CIR
// ---------------------------------------------------------------------------

#[test]
fn test_rust_impl_methods_in_ir() {
    let ir = parse_rust("impl Foo { fn bar(&self) {} fn baz(&self) {} }").expect("parse failed");

    let action_count = ir
        .nodes
        .iter()
        .filter(|n| n.node_type == NodeType::Processus)
        .count();
    assert!(
        action_count >= 2,
        "impl_item should expose its function_item children as Processus nodes, got {action_count}"
    );
}

// ---------------------------------------------------------------------------
// Isomorphisme cross-langage : if(cond) { action } ≡ en Python et Rust et JS
// ---------------------------------------------------------------------------

#[test]
fn test_isomorphism_python_rust_js_if_structure() {
    let ir_py = parse_python("if condition:\n    action()").expect("python parse failed");
    let ir_rs = parse_rust("fn main() {\n    if condition {\n        action();\n    }\n}")
        .expect("rust parse failed");
    let ir_js = parse_js("if (condition) {\n    action();\n}").expect("js parse failed");

    for (lang, ir) in [("python", &ir_py), ("rust", &ir_rs), ("js", &ir_js)] {
        let cond_edges = ir
            .edges
            .iter()
            .filter(|(_, _, e)| e.relation == RelationType::Condition)
            .count();
        assert_eq!(
            cond_edges, 1,
            "{lang}: expected 1 Condition edge, got {cond_edges}"
        );

        assert!(
            ir.nodes.iter().any(|n| n.node_type == NodeType::Condition),
            "{lang}: missing Condition node"
        );
    }
}

// ---------------------------------------------------------------------------
// Rejet du code invalide (has_error — guard P0-4)
// ---------------------------------------------------------------------------

#[test]
fn test_python_invalid_syntax_is_err() {
    assert!(
        parse_python("def foo(").is_err(),
        "syntaxe Python invalide (unclosed def) doit retourner Err"
    );
}

#[test]
fn test_rust_invalid_syntax_is_err() {
    assert!(
        parse_rust("fn foo(").is_err(),
        "syntaxe Rust invalide (unclosed fn) doit retourner Err"
    );
}

#[test]
fn test_js_invalid_syntax_is_err() {
    assert!(
        parse_js("function foo(").is_err(),
        "syntaxe JS invalide (unclosed function) doit retourner Err"
    );
}

// ---------------------------------------------------------------------------
// P0-3 : CodeParseReport — kinds non mappés comptés, plus de `continue` aveugle
// ---------------------------------------------------------------------------

#[test]
fn p03_unmapped_kinds_counted_in_report() {
    use gcn_frontend_code::parse_python_with_report;
    // `assert x` (expression_statement simple) : selon kinds.rs, certains
    // kinds passent, d'autres non — l'essentiel : le rapport existe et est cohérent.
    let (ir, report) =
        parse_python_with_report("if x < y:\n    reduce(z)\n").expect("parse failed");
    assert!(!ir.nodes.is_empty());
    assert_eq!(
        report.total_skipped(),
        report.skipped_unmapped + report.truncated_labels
    );
    // Cas avec des kinds hors mapping (break/pass) : ils doivent être comptés,
    // pas silencieusement ignorés.
    let (_, report2) = parse_python_with_report("while True:\n    break\n").expect("parse failed");
    assert!(
        report2.skipped_unmapped > 0,
        "kinds non mappés (break) comptés, obtenu {report2:?}"
    );
}

#[test]
fn p03_function_name_populates_agent_attribute() {
    // P3 : attributes.agent = nom défini via le champ AST `name`.
    let ir = parse_python("def compute(x):\n    return x * 2\n").expect("parse failed");
    let f = ir
        .nodes
        .iter()
        .find(|n| n.label.contains("compute"))
        .expect("nœud fonction attendu");
    assert_eq!(
        f.attributes.agent.as_deref(),
        Some("compute"),
        "agent = nom défini, obtenu {:?}",
        f.attributes
    );
}

#[test]
fn p03_long_labels_truncated_at_256_and_counted() {
    use gcn_frontend_code::{MAX_LABEL_CHARS, parse_python_with_report};
    assert_eq!(MAX_LABEL_CHARS, 256);
    // Une expression très longue en une ligne → label tronqué + compté.
    let long_call = format!("result = some_function_name({})", "x, ".repeat(200));
    let (ir, report) = parse_python_with_report(&long_call).expect("parse failed");
    assert!(
        report.truncated_labels > 0,
        "troncature comptée, obtenu {report:?}"
    );
    for n in &ir.nodes {
        assert!(
            n.label.chars().count() <= MAX_LABEL_CHARS,
            "label > 256 chars : {:?}...",
            n.label.chars().take(40).collect::<String>()
        );
    }
}
