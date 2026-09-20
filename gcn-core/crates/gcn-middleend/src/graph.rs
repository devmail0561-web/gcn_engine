// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use gcn_ir::{CausalIR, NodeId};
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;

pub struct CausalGraph {
    pub g: DiGraph<NodeId, usize>,
    pub node_indices: HashMap<NodeId, NodeIndex>,
}

impl CausalGraph {
    pub fn node_index(&self, id: NodeId) -> Option<NodeIndex> {
        self.node_indices.get(&id).copied()
    }
}

pub fn build(ir: &CausalIR) -> CausalGraph {
    let mut g = DiGraph::new();
    let mut node_indices = HashMap::new();

    for node in &ir.nodes {
        let idx = g.add_node(node.id);
        node_indices.insert(node.id, idx);
    }

    for (i, (src, dst, _edge)) in ir.edges.iter().enumerate() {
        if let (Some(&si), Some(&di)) = (node_indices.get(src), node_indices.get(dst)) {
            g.add_edge(si, di, i);
        } else {
            eprintln!("gcn-middleend: arête #{i} dangling ({src:?} → {dst:?}) ignorée");
        }
    }

    CausalGraph { g, node_indices }
}
