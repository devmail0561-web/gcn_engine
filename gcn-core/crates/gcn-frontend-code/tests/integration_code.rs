use std::path::PathBuf;
use gcn_frontend_code::{parse_js, parse_python, parse_rust};
use gcn_frontend_fr::FrenchParser;
use gcn_ir::{NodeType, RelationType};

fn taxonomies_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../gcn-references/taxonomies")
}

fn fr_parser() -> FrenchParser {
    FrenchParser::new(&taxonomies_root()).expect("Failed to load French taxonomies")
}

// ---------------------------------------------------------------------------
// Fonctionnalité de base du parseur Python
// ---------------------------------------------------------------------------

#[test]
fn test_python_parses_if_statement() {
    let ir = parse_python("if x < y:\n    reduce(z)", &taxonomies_root())
        .expect("parse failed");
    assert!(!ir.nodes.is_empty(), "should produce nodes");
    assert!(!ir.edges.is_empty(), "should produce edges");
}

#[test]
fn test_python_if_produces_condition_node() {
    let ir = parse_python("if x < y:\n    reduce(z)", &taxonomies_root())
        .expect("parse failed");
    let has_condition = ir.nodes.iter()
        .any(|n| n.node_type == gcn_ir::NodeType::Condition);
    assert!(has_condition, "if_statement should produce a Condition node");
}

#[test]
fn test_python_if_produces_condition_edge() {
    let ir = parse_python("if x < y:\n    reduce(z)", &taxonomies_root())
        .expect("parse failed");
    let has_cond_edge = ir.edges.iter()
        .any(|(_, _, e)| e.relation == RelationType::Condition);
    assert!(has_cond_edge, "if_statement should produce a Condition edge");
}

#[test]
fn test_python_for_produces_processus_node() {
    let ir = parse_python("for i in items:\n    process(i)", &taxonomies_root())
        .expect("parse failed");
    let has_proc = ir.nodes.iter()
        .any(|n| n.node_type == gcn_ir::NodeType::Processus);
    assert!(has_proc, "for_statement should produce a Processus node");
}

#[test]
fn test_python_assignment_produces_transition() {
    let ir = parse_python("x = compute()", &taxonomies_root())
        .expect("parse failed");
    let has_trans = ir.nodes.iter()
        .any(|n| n.node_type == gcn_ir::NodeType::Transition);
    assert!(has_trans, "assignment should produce a Transition node");
}

#[test]
fn test_python_source_lang_is_python() {
    let ir = parse_python("x = 1", &taxonomies_root()).expect("parse failed");
    assert!(matches!(
        ir.source_lang,
        gcn_ir::SourceLanguage::Programming { lang: gcn_ir::ProgrammingLanguage::Python }
    ));
}

#[test]
fn test_python_empty_produces_empty_ir() {
    let ir = parse_python("", &taxonomies_root()).expect("parse failed");
    assert!(ir.nodes.is_empty());
    assert!(ir.edges.is_empty());
}

// ---------------------------------------------------------------------------
// Isomorphisme fr ↔ python : critère SAD Phase 6
// "if x < y: reduce(z)" ≡ "Si x est inférieur à y, on réduit z"
// ---------------------------------------------------------------------------

#[test]
fn test_isomorphism_if_vs_si() {
    let ir_py = parse_python("if x < y:\n    reduce(z)", &taxonomies_root())
        .expect("python parse failed");
    let ir_fr = fr_parser()
        .parse("Si x est inférieur à y, on réduit z.")
        .expect("french parse failed");

    // Both have at least one Condition edge
    let py_cond_edges = ir_py.edges.iter()
        .filter(|(_, _, e)| e.relation == RelationType::Condition)
        .count();
    let fr_cond_edges = ir_fr.edges.iter()
        .filter(|(_, _, e)| e.relation == RelationType::Condition)
        .count();
    assert_eq!(
        py_cond_edges, fr_cond_edges,
        "Same number of Condition edges: python={py_cond_edges}, french={fr_cond_edges}"
    );
    assert!(py_cond_edges > 0, "both IRs should have at least one Condition edge");

    // Both have exactly 2 nodes for a simple if/si (condition + effect)
    assert_eq!(
        ir_py.nodes.len(), ir_fr.nodes.len(),
        "Same node count: python={}, french={}", ir_py.nodes.len(), ir_fr.nodes.len()
    );

    // In both, every Condition edge connects two existing nodes (source → destination)
    let all_node_ids = |ir: &gcn_ir::CausalIR| {
        ir.nodes.iter().map(|n| n.id).collect::<std::collections::HashSet<_>>()
    };
    for (src, dst, e) in &ir_py.edges {
        if e.relation == RelationType::Condition {
            let ids = all_node_ids(&ir_py);
            assert!(ids.contains(src) && ids.contains(dst),
                "Python Condition edge endpoints must be valid node IDs");
        }
    }
    for (src, dst, e) in &ir_fr.edges {
        if e.relation == RelationType::Condition {
            let ids = all_node_ids(&ir_fr);
            assert!(ids.contains(src) && ids.contains(dst),
                "French Condition edge endpoints must be valid node IDs");
        }
    }

    // Direction: condition edge goes from cause (lower temporal_index) to effect (higher)
    // For Python: if_statement (id=0) → body_call (id=1), so src.id < dst.id
    for (src_id, dst_id, e) in &ir_py.edges {
        if e.relation == RelationType::Condition {
            assert!(src_id.0 < dst_id.0,
                "Python Condition edge should go from cause (lower id) to effect (higher id): {src_id:?} → {dst_id:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// Arêtes séquentielles : deux instructions consécutives → Sequence, pas Condition
// ---------------------------------------------------------------------------

#[test]
fn test_sequential_statements_use_sequence_edge() {
    let ir = parse_python("x = compute()\nif x < y:\n    act()", &taxonomies_root())
        .expect("parse failed");

    let transition_ids: Vec<_> = ir.nodes.iter()
        .filter(|n| n.node_type == NodeType::Transition)
        .map(|n| n.id)
        .collect();
    let condition_ids: Vec<_> = ir.nodes.iter()
        .filter(|n| n.node_type == NodeType::Condition)
        .map(|n| n.id)
        .collect();

    assert!(!transition_ids.is_empty(), "should have Transition node (x = compute())");
    assert!(!condition_ids.is_empty(), "should have Condition node (if x < y)");

    // The edge between the assignment and the if should be Sequence, not Condition
    let seq_edge = ir.edges.iter().any(|(src, dst, e)| {
        transition_ids.contains(src)
            && condition_ids.contains(dst)
            && e.relation == RelationType::Sequence
    });
    assert!(seq_edge, "consecutive statements should be linked by a Sequence edge");
}

// ---------------------------------------------------------------------------
// walk_try : les statements du bloc except doivent apparaître dans le CIR
// ---------------------------------------------------------------------------

#[test]
fn test_try_except_body_in_ir() {
    let ir = parse_python(
        "try:\n    x = risky()\nexcept ValueError:\n    handle_error()",
        &taxonomies_root(),
    ).expect("parse failed");

    // handle_error() should appear as an Action node in the IR
    let has_handle = ir.nodes.iter().any(|n| n.label.contains("handle_error"));
    assert!(has_handle, "handler statement inside except should produce a node in the CIR");
}

// ---------------------------------------------------------------------------
// node_label : pas de troncature sur le premier ':' pour les fonctions typées
// ---------------------------------------------------------------------------

#[test]
fn test_function_label_uses_name_not_colon_truncation() {
    let ir = parse_python("def compute(x: int) -> int:\n    return x * 2", &taxonomies_root())
        .expect("parse failed");

    let fn_node = ir.nodes.iter().find(|n| n.node_type == NodeType::Action);
    assert!(fn_node.is_some(), "function_definition should produce an Action node");

    let label = &fn_node.unwrap().label;
    // Must NOT be "def compute(x" (truncated at first colon inside parameters)
    assert!(
        !label.starts_with("def compute(x") || label.len() > "def compute(x".len(),
        "label should not be truncated at the type annotation colon: got {label:?}"
    );
    assert!(label.contains("compute"), "label should contain the function name");
}

// ---------------------------------------------------------------------------
// Frontend Rust
// ---------------------------------------------------------------------------

#[test]
fn test_rust_parses_if_expression() {
    let ir = parse_rust("fn main() {\n    if x < y {\n        reduce(z);\n    }\n}", &taxonomies_root())
        .expect("parse failed");
    assert!(!ir.nodes.is_empty());
    assert!(!ir.edges.is_empty());
}

#[test]
fn test_rust_if_produces_condition_node() {
    let ir = parse_rust("fn main() {\n    if condition {\n        action();\n    }\n}", &taxonomies_root())
        .expect("parse failed");
    assert!(ir.nodes.iter().any(|n| n.node_type == NodeType::Condition));
}

#[test]
fn test_rust_if_produces_condition_edge() {
    let ir = parse_rust("fn main() {\n    if x {\n        do_it();\n    }\n}", &taxonomies_root())
        .expect("parse failed");
    assert!(ir.edges.iter().any(|(_, _, e)| e.relation == RelationType::Condition));
}

#[test]
fn test_rust_fn_item_produces_action_node() {
    let ir = parse_rust("fn compute(x: i32) -> i32 { x * 2 }", &taxonomies_root())
        .expect("parse failed");
    assert!(ir.nodes.iter().any(|n| n.node_type == NodeType::Action));
}

#[test]
fn test_rust_let_produces_transition_node() {
    let ir = parse_rust("fn main() { let x = compute(); }", &taxonomies_root())
        .expect("parse failed");
    assert!(ir.nodes.iter().any(|n| n.node_type == NodeType::Transition));
}

#[test]
fn test_rust_source_lang_is_rust() {
    let ir = parse_rust("fn main() {}", &taxonomies_root()).expect("parse failed");
    assert!(matches!(
        ir.source_lang,
        gcn_ir::SourceLanguage::Programming { lang: gcn_ir::ProgrammingLanguage::Rust }
    ));
}

#[test]
fn test_rust_empty_produces_empty_ir() {
    let ir = parse_rust("", &taxonomies_root()).expect("parse failed");
    assert!(ir.nodes.is_empty());
    assert!(ir.edges.is_empty());
}

// ---------------------------------------------------------------------------
// Frontend JavaScript
// ---------------------------------------------------------------------------

#[test]
fn test_js_parses_if_statement() {
    let ir = parse_js("if (x < y) {\n    reduce(z);\n}", &taxonomies_root())
        .expect("parse failed");
    assert!(!ir.nodes.is_empty());
    assert!(!ir.edges.is_empty());
}

#[test]
fn test_js_if_produces_condition_node() {
    let ir = parse_js("if (condition) {\n    action();\n}", &taxonomies_root())
        .expect("parse failed");
    assert!(ir.nodes.iter().any(|n| n.node_type == NodeType::Condition));
}

#[test]
fn test_js_if_produces_condition_edge() {
    let ir = parse_js("if (x) {\n    doIt();\n}", &taxonomies_root())
        .expect("parse failed");
    assert!(ir.edges.iter().any(|(_, _, e)| e.relation == RelationType::Condition));
}

#[test]
fn test_js_function_produces_action_node() {
    let ir = parse_js("function compute(x) { return x * 2; }", &taxonomies_root())
        .expect("parse failed");
    assert!(ir.nodes.iter().any(|n| n.node_type == NodeType::Action));
}

#[test]
fn test_js_variable_declaration_produces_transition() {
    let ir = parse_js("const x = compute();", &taxonomies_root())
        .expect("parse failed");
    assert!(ir.nodes.iter().any(|n| n.node_type == NodeType::Transition));
}

#[test]
fn test_js_source_lang_is_javascript() {
    let ir = parse_js("const x = 1;", &taxonomies_root()).expect("parse failed");
    assert!(matches!(
        ir.source_lang,
        gcn_ir::SourceLanguage::Programming { lang: gcn_ir::ProgrammingLanguage::JavaScript }
    ));
}

#[test]
fn test_js_empty_produces_empty_ir() {
    let ir = parse_js("", &taxonomies_root()).expect("parse failed");
    assert!(ir.nodes.is_empty());
    assert!(ir.edges.is_empty());
}

// ---------------------------------------------------------------------------
// Rust : impl_item — les méthodes internes doivent apparaître dans le CIR
// ---------------------------------------------------------------------------

#[test]
fn test_rust_impl_methods_in_ir() {
    let ir = parse_rust(
        "impl Foo { fn bar(&self) {} fn baz(&self) {} }",
        &taxonomies_root(),
    ).expect("parse failed");

    let action_count = ir.nodes.iter().filter(|n| n.node_type == NodeType::Action).count();
    assert!(action_count >= 2,
        "impl_item should expose its function_item children as Action nodes, got {action_count}");
}

// ---------------------------------------------------------------------------
// Isomorphisme cross-langage : if(cond) { action } ≡ en Python et Rust et JS
// ---------------------------------------------------------------------------

#[test]
fn test_isomorphism_python_rust_js_if_structure() {
    let ir_py = parse_python("if condition:\n    action()", &taxonomies_root())
        .expect("python parse failed");
    let ir_rs = parse_rust("fn main() {\n    if condition {\n        action();\n    }\n}", &taxonomies_root())
        .expect("rust parse failed");
    let ir_js = parse_js("if (condition) {\n    action();\n}", &taxonomies_root())
        .expect("js parse failed");

    for (lang, ir) in [("python", &ir_py), ("rust", &ir_rs), ("js", &ir_js)] {
        let cond_edges = ir.edges.iter()
            .filter(|(_, _, e)| e.relation == RelationType::Condition)
            .count();
        assert_eq!(cond_edges, 1, "{lang}: expected 1 Condition edge, got {cond_edges}");

        assert!(
            ir.nodes.iter().any(|n| n.node_type == NodeType::Condition),
            "{lang}: missing Condition node"
        );
    }
}
