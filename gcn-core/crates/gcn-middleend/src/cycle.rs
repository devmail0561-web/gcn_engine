use gcn_ir::{CausalCycle, CausalIR, CycleId, CycleType, NodeId, RelationType};
use petgraph::algo::tarjan_scc;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet};

use crate::graph::CausalGraph;

pub fn detect_and_classify(
    ir: &CausalIR,
    g: &CausalGraph,
) -> (Vec<CausalCycle>, HashMap<usize, CycleId>) {
    let sccs = tarjan_scc(&g.g);
    let mut cycles = Vec::new();
    let mut edge_cycle_map: HashMap<usize, CycleId> = HashMap::new();
    let mut counter = 0u32;

    for scc in &sccs {
        let is_cycle = scc.len() > 1 || has_self_loop(&g.g, scc[0]);
        if !is_cycle {
            continue;
        }

        let cycle_id = CycleId(counter);
        counter += 1;

        let node_ids: Vec<NodeId> = scc.iter().map(|&ni| g.g[ni]).collect();
        let node_id_set: HashSet<NodeId> = node_ids.iter().copied().collect();

        let cycle_edges: Vec<(usize, &gcn_ir::CausalEdge)> = ir
            .edges
            .iter()
            .enumerate()
            .filter(|(_, (src, dst, _))| node_id_set.contains(src) && node_id_set.contains(dst))
            .map(|(i, (_, _, e))| (i, e))
            .collect();

        for (idx, _) in &cycle_edges {
            edge_cycle_map.insert(*idx, cycle_id);
        }

        cycles.push(CausalCycle {
            id: cycle_id,
            path: node_ids,
            cycle_type: classify(&cycle_edges),
        });
    }

    (cycles, edge_cycle_map)
}

fn has_self_loop(g: &petgraph::graph::DiGraph<NodeId, usize>, ni: NodeIndex) -> bool {
    g.edges(ni).any(|e| e.target() == ni)
}

fn classify(edges: &[(usize, &gcn_ir::CausalEdge)]) -> CycleType {
    let has_concession = edges
        .iter()
        .any(|(_, e)| matches!(e.relation, RelationType::Concession));
    let has_negative = edges.iter().any(|(_, e)| {
        e.negated || matches!(e.relation, RelationType::Prevent | RelationType::Opposition)
    });

    if has_concession {
        CycleType::Oscillation
    } else if has_negative {
        CycleType::FeedbackNegative
    } else {
        CycleType::FeedbackPositive
    }
}
