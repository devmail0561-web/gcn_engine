use std::collections::{HashMap, HashSet, VecDeque};

use gcn_ir::{CausalIR, NodeId, RelationType};
use gcn_middleend::graph::{build, CausalGraph};
use petgraph::graph::NodeIndex;
use petgraph::Direction;

#[derive(Debug, Clone)]
pub struct CausalLink {
    pub from_label: String,
    pub to_label: String,
    pub relation: RelationType,
    pub confidence: f32,
    pub negated: bool,
}

/// WHY: reverse BFS — find all causal ancestors of nodes matching `label`
pub fn why(ir: &CausalIR, label: &str) -> Vec<(String, Vec<CausalLink>)> {
    let g = build(ir);
    let lower = label.to_lowercase();
    ir.nodes
        .iter()
        .filter(|n| n.label.to_lowercase().contains(&lower))
        .map(|n| {
            let links = bfs_ancestors(ir, &g, n.id);
            (n.label.clone(), links)
        })
        .collect()
}

/// WHAT: forward BFS — find all causal descendants of nodes matching `label`
pub fn what(ir: &CausalIR, label: &str) -> Vec<(String, Vec<CausalLink>)> {
    let g = build(ir);
    let lower = label.to_lowercase();
    ir.nodes
        .iter()
        .filter(|n| n.label.to_lowercase().contains(&lower))
        .map(|n| {
            let links = bfs_descendants(ir, &g, n.id);
            (n.label.clone(), links)
        })
        .collect()
}

/// CHAIN: find causal path from the first node matching `from` to `to`
pub fn chain(ir: &CausalIR, from: &str, to: &str) -> Option<Vec<CausalLink>> {
    let g = build(ir);
    let src = ir.nodes.iter().find(|n| n.label.to_lowercase().contains(&from.to_lowercase()))?;
    let dst = ir.nodes.iter().find(|n| n.label.to_lowercase().contains(&to.to_lowercase()))?;
    let si = g.node_index(src.id)?;
    let di = g.node_index(dst.id)?;

    if !petgraph::algo::has_path_connecting(&g.g, si, di, None) {
        return None;
    }
    Some(bfs_path_links(ir, &g, src.id, dst.id))
}

fn bfs_ancestors(ir: &CausalIR, g: &CausalGraph, start: NodeId) -> Vec<CausalLink> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut links = Vec::new();

    if let Some(&si) = g.node_indices.get(&start) {
        queue.push_back(si);
        visited.insert(si);
    }

    while let Some(ni) = queue.pop_front() {
        let cur_id = g.g[ni];
        for pred in g.g.neighbors_directed(ni, Direction::Incoming) {
            if visited.insert(pred) {
                queue.push_back(pred);
                if let Some(link) = edge_link(ir, g.g[pred], cur_id) {
                    links.push(link);
                }
            }
        }
    }
    links
}

fn bfs_descendants(ir: &CausalIR, g: &CausalGraph, start: NodeId) -> Vec<CausalLink> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut links = Vec::new();

    if let Some(&si) = g.node_indices.get(&start) {
        queue.push_back(si);
        visited.insert(si);
    }

    while let Some(ni) = queue.pop_front() {
        let cur_id = g.g[ni];
        for succ in g.g.neighbors_directed(ni, Direction::Outgoing) {
            if visited.insert(succ) {
                queue.push_back(succ);
                if let Some(link) = edge_link(ir, cur_id, g.g[succ]) {
                    links.push(link);
                }
            }
        }
    }
    links
}

fn bfs_path_links(ir: &CausalIR, g: &CausalGraph, from: NodeId, to: NodeId) -> Vec<CausalLink> {
    let si = match g.node_indices.get(&from) { Some(&x) => x, None => return vec![] };
    let di = match g.node_indices.get(&to) { Some(&x) => x, None => return vec![] };

    let mut prev: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();
    queue.push_back(si);
    visited.insert(si);

    'bfs: while let Some(ni) = queue.pop_front() {
        if ni == di { break 'bfs; }
        for nb in g.g.neighbors_directed(ni, Direction::Outgoing) {
            if visited.insert(nb) {
                prev.insert(nb, ni);
                queue.push_back(nb);
            }
        }
    }

    let mut path_ids = Vec::new();
    let mut cur = di;
    path_ids.push(g.g[cur]);
    while cur != si {
        match prev.get(&cur) {
            Some(&p) => { path_ids.push(g.g[p]); cur = p; }
            None => return vec![],
        }
    }
    path_ids.reverse();

    path_ids
        .windows(2)
        .filter_map(|w| edge_link(ir, w[0], w[1]))
        .collect()
}

fn edge_link(ir: &CausalIR, src: NodeId, dst: NodeId) -> Option<CausalLink> {
    ir.edges.iter()
        .find(|(s, d, _)| *s == src && *d == dst)
        .map(|(s, d, e)| CausalLink {
            from_label: node_label(ir, *s),
            to_label: node_label(ir, *d),
            relation: e.relation,
            confidence: e.confidence,
            negated: e.negated,
        })
}

pub fn node_label(ir: &CausalIR, id: NodeId) -> String {
    ir.nodes
        .iter()
        .find(|n| n.id == id)
        .map(|n| n.label.clone())
        .unwrap_or_else(|| format!("node_{}", id.0))
}
