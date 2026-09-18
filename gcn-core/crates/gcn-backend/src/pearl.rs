use std::collections::{HashMap, HashSet, VecDeque};

use gcn_ir::{CausalIR, NodeId, RelationType};
use gcn_middleend::graph::{build, CausalGraph};
use petgraph::graph::NodeIndex;
use petgraph::Direction;

#[derive(Debug, Clone)]
pub struct CausalLink {
    pub from_label: String,
    pub to_label: String,
    pub to_id: NodeId,
    pub relation: RelationType,
    pub confidence: f32,
    pub negated: bool,
}

/// Pearl niveau 2 — résultat d'une intervention do-calculus sur un nœud.
#[derive(Debug, Clone)]
pub struct InterventionResult {
    /// Arêtes entrantes coupées par l'intervention (les causes naturelles de X sont court-circuitées).
    pub severed: Vec<CausalLink>,
    /// Effets qui se propagent en aval depuis X une fois X forcé.
    pub effects: Vec<CausalLink>,
}

/// Pearl niveau 3 — résultat d'une requête contrefactuelle sur un nœud X.
#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    /// Effets réels de X dans le monde actuel (BFS avant depuis X).
    pub actual_effects: Vec<CausalLink>,
    /// Sous-ensemble des effets réels qui n'ont AUCUN autre chemin causal sans X.
    /// Ce sont les effets qui n'auraient pas eu lieu si X n'avait pas eu lieu.
    pub unique_effects: Vec<String>,
}

/// Score de correspondance label ↔ requête : 0 exact, 1 préfixe, 2 contient.
/// Retourne None si aucune correspondance (insensible à la casse).
pub fn match_score(label: &str, query: &str) -> Option<u8> {
    let l = label.to_lowercase();
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return None;
    }
    if l == q {
        Some(0)
    } else if l.starts_with(&q) {
        Some(1)
    } else if l.contains(&q) {
        Some(2)
    } else {
        None
    }
}

/// Meilleur nœud pour une requête : score minimal, puis label le plus court
/// (déterministe en cas d'égalité).
pub fn find_best<'a>(ir: &'a CausalIR, query: &str) -> Option<&'a gcn_ir::CausalNode> {
    ir.nodes
        .iter()
        .filter_map(|n| match_score(&n.label, query).map(|s| (s, n.label.len(), n)))
        .min_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)))
        .map(|(_, _, n)| n)
}

/// Tous les nœuds correspondants, triés par pertinence (exact > préfixe > contient).
pub fn find_all_ranked<'a>(ir: &'a CausalIR, query: &str) -> Vec<&'a gcn_ir::CausalNode> {
    let mut v: Vec<(u8, usize, &'a gcn_ir::CausalNode)> = ir
        .nodes
        .iter()
        .filter_map(|n| match_score(&n.label, query).map(|s| (s, n.label.len(), n)))
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    v.into_iter().map(|(_, _, n)| n).collect()
}

type NodeMap<'a> = HashMap<NodeId, &'a gcn_ir::CausalNode>;
type EdgeMap<'a> = HashMap<(u32, u32), &'a gcn_ir::CausalEdge>;

fn build_maps(ir: &CausalIR) -> (NodeMap<'_>, EdgeMap<'_>) {
    let nm: NodeMap = ir.nodes.iter().map(|n| (n.id, n)).collect();
    let em: EdgeMap = ir.edges.iter().map(|(s, d, e)| ((s.0, d.0), e)).collect();
    (nm, em)
}

/// WHY: reverse BFS — find all causal ancestors of nodes matching `label`
pub fn why(ir: &CausalIR, label: &str) -> Vec<(String, Vec<CausalLink>)> {
    let g = build(ir);
    let (nm, em) = build_maps(ir);
    find_all_ranked(ir, label)
        .into_iter()
        .map(|n| {
            let links = bfs_ancestors(&nm, &em, &g, n.id);
            (n.label.clone(), links)
        })
        .collect()
}

/// WHAT: forward BFS — find all causal descendants of nodes matching `label`
pub fn what(ir: &CausalIR, label: &str) -> Vec<(String, Vec<CausalLink>)> {
    let g = build(ir);
    let (nm, em) = build_maps(ir);
    find_all_ranked(ir, label)
        .into_iter()
        .map(|n| {
            let links = bfs_descendants(&nm, &em, &g, n.id);
            (n.label.clone(), links)
        })
        .collect()
}

/// CHAIN: find causal path from the best node matching `from` to `to`
pub fn chain(ir: &CausalIR, from: &str, to: &str) -> Option<Vec<CausalLink>> {
    let g = build(ir);
    let src = find_best(ir, from)?;
    let dst = find_best(ir, to)?;
    let si = g.node_index(src.id)?;
    let di = g.node_index(dst.id)?;

    if !petgraph::algo::has_path_connecting(&g.g, si, di, None) {
        return None;
    }
    let (nm, em) = build_maps(ir);
    Some(bfs_path_links(&nm, &em, &g, src.id, dst.id))
}

fn bfs_ancestors(nm: &NodeMap, em: &EdgeMap, g: &CausalGraph, start: NodeId) -> Vec<CausalLink> {
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
                if let Some(link) = edge_link(nm, em, g.g[pred], cur_id) {
                    links.push(link);
                }
            }
        }
    }
    links
}

fn bfs_descendants(nm: &NodeMap, em: &EdgeMap, g: &CausalGraph, start: NodeId) -> Vec<CausalLink> {
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
                if let Some(link) = edge_link(nm, em, cur_id, g.g[succ]) {
                    links.push(link);
                }
            }
        }
    }
    links
}

fn bfs_path_links(nm: &NodeMap, em: &EdgeMap, g: &CausalGraph, from: NodeId, to: NodeId) -> Vec<CausalLink> {
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
        .filter_map(|w| edge_link(nm, em, w[0], w[1]))
        .collect()
}

fn edge_link(nm: &NodeMap, em: &EdgeMap, src: NodeId, dst: NodeId) -> Option<CausalLink> {
    em.get(&(src.0, dst.0)).map(|e| CausalLink {
        from_label: nm.get(&src).map(|n| n.label.clone()).unwrap_or_else(|| format!("node_{}", src.0)),
        to_label: nm.get(&dst).map(|n| n.label.clone()).unwrap_or_else(|| format!("node_{}", dst.0)),
        to_id: dst,
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

// ---------------------------------------------------------------------------
// Pearl niveau 2 — Intervention (do-calculus)
// ---------------------------------------------------------------------------

/// `DO(X)` : intervention sur le nœud X.
/// Coupe toutes les arêtes entrantes (les causes de X sont court-circuitées),
/// puis propage en avant depuis X.
/// Retourne None si aucun nœud ne correspond à `target_label`.
pub fn intervene(ir: &CausalIR, target_label: &str) -> Option<(String, InterventionResult)> {
    let target = find_best(ir, target_label)?;
    let (nm, em) = build_maps(ir);

    // Arêtes entrantes coupées par l'intervention
    let severed: Vec<CausalLink> = ir
        .edges
        .iter()
        .filter(|(_, d, _)| *d == target.id)
        .filter_map(|(s, d, _)| edge_link(&nm, &em, *s, *d))
        .collect();

    // Propagation en avant depuis X (les effets de X persistent même après intervention)
    let g = build(ir);
    let effects = bfs_descendants(&nm, &em, &g, target.id);

    Some((target.label.clone(), InterventionResult { severed, effects }))
}

// ---------------------------------------------------------------------------
// Pearl niveau 3 — Contrefactuels
// ---------------------------------------------------------------------------

/// `COUNTERFACTUAL(X)` : "Que se serait-il passé si X n'avait pas eu lieu ?"
///
/// Algorithme :
/// 1. Effets réels : BFS avant depuis X → `actual_effects`
/// 2. On retire X du graphe et on calcule les nœuds atteignables depuis les racines
/// 3. Les effets réels qui ne sont plus atteignables sans X sont counterfactuellement dépendants de X
pub fn counterfactual(ir: &CausalIR, target_label: &str) -> Option<(String, CounterfactualResult)> {
    let target = find_best(ir, target_label)?;

    let g = build(ir);
    let (nm, em) = build_maps(ir);

    let actual_effects = bfs_descendants(&nm, &em, &g, target.id);

    // Nœuds atteignables depuis les racines SANS X
    let reachable_without_x = reachable_from_roots_without(&g, target.id);

    // Effets uniques = atteignables depuis X mais pas depuis les racines sans X.
    // Dédup par NodeId (pas par label) pour préserver la cardinalité quand deux nœuds
    // distincts partagent le même label.
    let mut seen_ids: HashSet<NodeId> = HashSet::new();
    let unique_effects: Vec<String> = actual_effects
        .iter()
        .filter(|link| !reachable_without_x.contains(&link.to_id))
        .filter(|link| seen_ids.insert(link.to_id))
        .map(|link| link.to_label.clone())
        .collect();

    Some((target.label.clone(), CounterfactualResult { actual_effects, unique_effects }))
}

/// Calcule l'ensemble des nœuds atteignables depuis les VRAIES racines originelles du graphe
/// en sautant le nœud `excluded` et ses arêtes sortantes.
///
/// "Vraie racine" = nœud sans AUCUNE arête entrante dans le graphe complet (variable exogène).
/// On ne repart pas des nœuds qui deviendraient des racines artificielles après suppression de X.
fn reachable_from_roots_without(g: &CausalGraph, excluded: NodeId) -> HashSet<NodeId> {
    let excluded_idx = match g.node_indices.get(&excluded) {
        Some(&x) => x,
        None => {
            return g.node_indices.keys().copied().collect();
        }
    };

    let mut reachable: HashSet<NodeId> = HashSet::new();
    let mut queue: VecDeque<NodeIndex> = VecDeque::new();

    // Racines originelles : nœuds sans AUCUNE arête entrante dans le graphe complet
    for ni in g.g.node_indices() {
        if ni == excluded_idx {
            continue;
        }
        let total_incoming = g.g.neighbors_directed(ni, Direction::Incoming).count();
        if total_incoming == 0 {
            let id = g.g[ni];
            if reachable.insert(id) {
                queue.push_back(ni);
            }
        }
    }

    // BFS en avant depuis les vraies racines, sans traverser X
    while let Some(ni) = queue.pop_front() {
        for nb in g.g.neighbors_directed(ni, Direction::Outgoing) {
            if nb == excluded_idx {
                continue;
            }
            let nb_id = g.g[nb];
            if reachable.insert(nb_id) {
                queue.push_back(nb);
            }
        }
    }
    reachable
}
