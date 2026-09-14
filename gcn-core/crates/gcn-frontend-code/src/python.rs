use std::path::Path;

use gcn_ir::{CausalEdge, CausalIR, CausalNode, IrMetadata, NodeId, NodeType,
    ProgrammingLanguage, RelationType, SourceLanguage};
use tree_sitter::Parser;

use crate::common::{control_edge, emit_node};
use crate::error::CodeParserError;
use crate::resources::CodeResources;

pub fn parse(source: &str, taxonomies_root: &Path) -> Result<CausalIR, CodeParserError> {
    let res = CodeResources::load_python(taxonomies_root)?;

    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_python::LANGUAGE.into())
        .map_err(|e| CodeParserError::Language(e.to_string()))?;

    let tree = parser.parse(source, None).ok_or(CodeParserError::ParseFailed)?;
    let root = tree.root_node();
    let src_bytes = source.as_bytes();

    let mut nodes: Vec<CausalNode> = Vec::new();
    let mut edges: Vec<(NodeId, NodeId, CausalEdge)> = Vec::new();
    let mut next_id: u32 = 0;

    walk_block(root, src_bytes, &res, &mut nodes, &mut edges, &mut next_id);

    Ok(CausalIR {
        source_lang: SourceLanguage::Programming { lang: ProgrammingLanguage::Python },
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

        // Dispatch structurel sur la grammaire tree-sitter Python (pas de données lexicales) :
        // sélectionne la fonction de traversée selon la topologie de l'AST, pas le sens causal.
        // Les types de nœuds et relations causaux viennent de gcn-knowledge (YAML).
        let body_ids = match kind {
            "if_statement" => walk_if(child, src, res, nodes, edges, next_id),
            "for_statement" | "while_statement" | "function_definition" | "with_statement" => {
                walk_body_of(child, src, res, nodes, edges, next_id)
            }
            "try_statement" => walk_try(child, src, res, nodes, edges, next_id),
            _ => vec![],
        };
        for body_id in body_ids {
            edges.push((id, body_id, control_edge(edge_rel)));
        }

        if !block_ids.is_empty() {
            let prev = *block_ids.last().expect("non-empty");
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
            "elif_clause" | "else_clause" => {
                body_ids.extend(walk_body_of(child, src, res, nodes, edges, next_id));
            }
            _ => {}
        }
    }
    body_ids
}

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

fn walk_try(
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
            "except_clause" => {
                let node_type = res.kind_to_node_type.get("except_clause").copied()
                    .unwrap_or(NodeType::Etat);
                let except_id = emit_node(child, src, node_type, nodes, next_id, res);
                let edge_rel = res.kind_to_edge_type.get("except_clause").copied()
                    .unwrap_or(RelationType::Concession);
                for hid in walk_body_of(child, src, res, nodes, edges, next_id) {
                    edges.push((except_id, hid, control_edge(edge_rel)));
                }
                body_ids.push(except_id);
            }
            _ => {}
        }
    }
    body_ids
}
