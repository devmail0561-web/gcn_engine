// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use gcn_ir::{
    CausalEdge, CausalIR, CausalNode, IrMetadata, NodeId, ProgrammingLanguage, RelationType,
    SourceLanguage,
};
use tree_sitter::Parser;

use crate::common::{CodeParseReport, control_edge, emit_node, walk_dataflow};
use crate::error::CodeParserError;
use crate::kinds::{Lang, is_transparent, kind_info};

pub fn parse(source: &str) -> Result<CausalIR, CodeParserError> {
    Ok(parse_with_report(source)?.0)
}

/// P0-3 : variante avec `CodeParseReport` (kinds ignorés + troncatures comptés).
pub fn parse_with_report(source: &str) -> Result<(CausalIR, CodeParseReport), CodeParserError> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .map_err(|e| CodeParserError::Language(e.to_string()))?;

    let tree = parser
        .parse(source, None)
        .ok_or(CodeParserError::ParseFailed)?;
    let root = tree.root_node();
    if root.has_error() {
        return Err(CodeParserError::ParseFailed);
    }
    let src_bytes = source.as_bytes();

    let mut nodes: Vec<CausalNode> = Vec::new();
    let mut edges: Vec<(NodeId, NodeId, CausalEdge)> = Vec::new();
    let mut next_id: u32 = 0;
    let mut report = CodeParseReport::default();

    walk_block(
        root,
        src_bytes,
        &mut nodes,
        &mut edges,
        &mut next_id,
        &mut report,
    );

    Ok((
        CausalIR {
            source_lang: SourceLanguage::Programming {
                lang: ProgrammingLanguage::Rust,
            },
            source_text: source.to_string(),
            nodes,
            edges,
            cycles: vec![],
            unresolved: vec![],
            metadata: IrMetadata {
                schema_version: "2.0".to_string(),
                pipeline: vec!["gcn-frontend-code".to_string()],
                created_at: None,
            },
        },
        report,
    ))
}

fn walk_block(
    node: tree_sitter::Node<'_>,
    src: &[u8],
    nodes: &mut Vec<CausalNode>,
    edges: &mut Vec<(NodeId, NodeId, CausalEdge)>,
    next_id: &mut u32,
    report: &mut CodeParseReport,
) -> Vec<NodeId> {
    let mut block_ids: Vec<NodeId> = Vec::new();
    let mut cursor = node.walk();

    for raw_child in node.children(&mut cursor) {
        if raw_child.is_extra() || !raw_child.is_named() {
            continue;
        }
        let child = if is_transparent(Lang::Rust, raw_child.kind()) {
            let mut c = raw_child.walk();
            raw_child.named_children(&mut c).next().unwrap_or(raw_child)
        } else {
            raw_child
        };
        let kind = child.kind();
        let Some(info) = kind_info(Lang::Rust, kind) else {
            // P0-3 : kind non mappé → ignoré MAIS compté (était `continue` silencieux).
            report.skipped_unmapped += 1;
            continue;
        };

        let id = emit_node(
            child,
            src,
            info.node_type,
            nodes,
            next_id,
            info.label,
            report,
        );
        let edge_rel = kind_info(Lang::Rust, kind)
            .map(|i| i.edge_type)
            .unwrap_or(RelationType::ControlDependency);

        let body_ids = match kind {
            "if_expression" | "match_expression" => {
                walk_if(child, src, nodes, edges, next_id, report)
            }
            "while_expression" | "for_expression" | "loop_expression" | "function_item" => {
                walk_body_of(child, src, nodes, edges, next_id, report)
            }
            // impl_item body is declaration_list, not block
            "impl_item" | "trait_item" => {
                walk_body_of_decl(child, src, nodes, edges, next_id, report)
            }
            // C2a : récursion data-flow (voir python.rs).            "assignment_expression"
            "compound_assignment_expr"
            | "call_expression"
            | "let_declaration"
            | "return_expression" => {
                walk_dataflow(Lang::Rust, id, child, src, nodes, edges, next_id, report);
                vec![]
            }
            _ => vec![],
        };
        for body_id in body_ids {
            edges.push((id, body_id, control_edge(edge_rel)));
        }

        if let Some(&prev) = block_ids.last() {
            edges.push((prev, id, control_edge(RelationType::Sequence)));
        }

        block_ids.push(id);
    }
    block_ids
}

fn walk_if(
    node: tree_sitter::Node<'_>,
    src: &[u8],
    nodes: &mut Vec<CausalNode>,
    edges: &mut Vec<(NodeId, NodeId, CausalEdge)>,
    next_id: &mut u32,
    report: &mut CodeParseReport,
) -> Vec<NodeId> {
    let mut body_ids = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "block" => body_ids.extend(walk_block(child, src, nodes, edges, next_id, report)),
            "else_clause" => {
                body_ids.extend(walk_body_of(child, src, nodes, edges, next_id, report));
            }
            _ => {}
        }
    }
    body_ids
}

/// Walk the `block` body of a node (functions, loops, if bodies).
fn walk_body_of(
    node: tree_sitter::Node<'_>,
    src: &[u8],
    nodes: &mut Vec<CausalNode>,
    edges: &mut Vec<(NodeId, NodeId, CausalEdge)>,
    next_id: &mut u32,
    report: &mut CodeParseReport,
) -> Vec<NodeId> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "block" {
            return walk_block(child, src, nodes, edges, next_id, report);
        }
    }
    vec![]
}

/// Walk the `declaration_list` body of impl/trait items.
fn walk_body_of_decl(
    node: tree_sitter::Node<'_>,
    src: &[u8],
    nodes: &mut Vec<CausalNode>,
    edges: &mut Vec<(NodeId, NodeId, CausalEdge)>,
    next_id: &mut u32,
    report: &mut CodeParseReport,
) -> Vec<NodeId> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "declaration_list" {
            return walk_block(child, src, nodes, edges, next_id, report);
        }
    }
    vec![]
}
