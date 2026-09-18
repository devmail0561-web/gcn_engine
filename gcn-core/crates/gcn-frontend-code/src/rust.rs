use std::path::Path;

use gcn_ir::{CausalEdge, CausalIR, CausalNode, IrMetadata, NodeId,
    ProgrammingLanguage, RelationType, SourceLanguage};
use tree_sitter::Parser;

use crate::common::{control_edge, emit_node};
use crate::error::CodeParserError;
use crate::resources::CodeResources;

pub fn parse(source: &str, taxonomies_root: &Path) -> Result<CausalIR, CodeParserError> {
    let res = CodeResources::load_rust(taxonomies_root)?;

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .map_err(|e| CodeParserError::Language(e.to_string()))?;

    let tree = parser.parse(source, None).ok_or(CodeParserError::ParseFailed)?;
    let root = tree.root_node();
    let src_bytes = source.as_bytes();

    let mut nodes: Vec<CausalNode> = Vec::new();
    let mut edges: Vec<(NodeId, NodeId, CausalEdge)> = Vec::new();
    let mut next_id: u32 = 0;

    walk_block(root, src_bytes, &res, &mut nodes, &mut edges, &mut next_id);

    Ok(CausalIR {
        source_lang: SourceLanguage::Programming { lang: ProgrammingLanguage::Rust },
        source_text: source.to_string(),
        nodes,
        edges,
        cycles: vec![],
        unresolved: vec![],
        metadata: IrMetadata {
            schema_version: "1.0".to_string(),
            pipeline: vec!["gcn-frontend-code".to_string()],
            created_at: None,
        },
    })
}

fn walk_block(
    node: tree_sitter::Node<'_>,
    src: &[u8],
    res: &CodeResources,
    nodes: &mut Vec<CausalNode>,
    edges: &mut Vec<(NodeId, NodeId, CausalEdge)>,
    next_id: &mut u32,
) -> Vec<NodeId> {
    let mut block_ids: Vec<NodeId> = Vec::new();
    let mut cursor = node.walk();

    for raw_child in node.children(&mut cursor) {
        if raw_child.is_extra() || !raw_child.is_named() {
            continue;
        }
        let child = if res.transparent.contains(raw_child.kind()) {
            let mut c = raw_child.walk();
            raw_child.named_children(&mut c).next().unwrap_or(raw_child)
        } else {
            raw_child
        };
        let kind = child.kind();
        let Some(&node_type) = res.kind_to_node_type.get(kind) else { continue };

        let id = emit_node(child, src, node_type, nodes, next_id, res);
        let edge_rel = res.kind_to_edge_type.get(kind).copied()
            .unwrap_or(RelationType::ControlDependency);

        let body_ids = match kind {
            "if_expression" | "match_expression" => {
                walk_if(child, src, res, nodes, edges, next_id)
            }
            "while_expression" | "for_expression" | "loop_expression" | "function_item" => {
                walk_body_of(child, src, res, nodes, edges, next_id)
            }
            // impl_item body is declaration_list, not block
            "impl_item" | "trait_item" => {
                walk_body_of_decl(child, src, res, nodes, edges, next_id)
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
    res: &CodeResources,
    nodes: &mut Vec<CausalNode>,
    edges: &mut Vec<(NodeId, NodeId, CausalEdge)>,
    next_id: &mut u32,
) -> Vec<NodeId> {
    let mut body_ids = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "block" => body_ids.extend(walk_block(child, src, res, nodes, edges, next_id)),
            "else_clause" => {
                body_ids.extend(walk_body_of(child, src, res, nodes, edges, next_id));
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
    res: &CodeResources,
    nodes: &mut Vec<CausalNode>,
    edges: &mut Vec<(NodeId, NodeId, CausalEdge)>,
    next_id: &mut u32,
) -> Vec<NodeId> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "block" {
            return walk_block(child, src, res, nodes, edges, next_id);
        }
    }
    vec![]
}

/// Walk the `declaration_list` body of impl/trait items.
fn walk_body_of_decl(
    node: tree_sitter::Node<'_>,
    src: &[u8],
    res: &CodeResources,
    nodes: &mut Vec<CausalNode>,
    edges: &mut Vec<(NodeId, NodeId, CausalEdge)>,
    next_id: &mut u32,
) -> Vec<NodeId> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "declaration_list" {
            return walk_block(child, src, res, nodes, edges, next_id);
        }
    }
    vec![]
}
