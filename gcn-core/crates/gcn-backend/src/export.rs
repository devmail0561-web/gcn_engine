use gcn_ir::CausalIR;

use crate::error::BackendError;

pub fn to_json(ir: &CausalIR) -> Result<String, BackendError> {
    Ok(serde_json::to_string_pretty(ir)?)
}

pub fn to_dot(ir: &CausalIR) -> Result<String, BackendError> {
    let mut out = String::from("digraph CausalGraph {\n");
    out.push_str("  rankdir=LR;\n");
    out.push_str("  node [shape=box, style=filled, fillcolor=lightblue];\n\n");

    for node in &ir.nodes {
        let label = escape_dot(&node.label);
        let shape = match node.node_type {
            gcn_ir::NodeType::Condition => "diamond",
            gcn_ir::NodeType::EtatSystemique => "ellipse",
            _ => "box",
        };
        out.push_str(&format!(
            "  n{} [label=\"{}\", shape={}, tooltip=\"{:?}\"];\n",
            node.id.0, label, shape, node.node_type
        ));
    }

    out.push('\n');

    for (src, dst, edge) in &ir.edges {
        let rel = format!("{:?}", edge.relation);
        let color = edge_color(edge);
        let style = if edge.negated { "dashed" } else { "solid" };
        let conf = format!("{:.2}", edge.confidence);
        out.push_str(&format!(
            "  n{} -> n{} [label=\"{} ({})\" color=\"{}\" style=\"{}\"];\n",
            src.0, dst.0, rel, conf, color, style
        ));
    }

    out.push_str("}\n");
    Ok(out)
}

fn edge_color(edge: &gcn_ir::CausalEdge) -> &'static str {
    use gcn_ir::RelationType::*;
    match edge.relation {
        Cause => "#2ecc71",
        Enable => "#3498db",
        Prevent => "#e74c3c",
        Condition => "#f39c12",
        Concession | Opposition => "#9b59b6",
        Sequence => "#1abc9c",
        Motivation => "#e67e22",
        _ => "#95a5a6",
    }
}

fn escape_dot(s: &str) -> String {
    s.replace('"', "\\\"").replace('\n', "\\n")
}
