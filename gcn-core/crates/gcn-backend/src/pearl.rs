// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet, VecDeque};

use gcn_ir::{CausalIR, NodeId, RelationType, normalize_label};
use gcn_middleend::graph::{CausalGraph, build};
use petgraph::Direction;
use petgraph::graph::NodeIndex;

#[derive(Debug, Clone)]
pub struct CausalLink {
    pub from_label: String,
    pub to_label: String,
    pub from_id: NodeId,
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

/// Lien causal avec métadonnées temporelles.
#[derive(Debug, Clone)]
pub struct TemporalLink {
    pub from_label: String,
    pub to_label: String,
    pub relation: RelationType,
    pub confidence: f32,
    pub negated: bool,
    /// TemporalGap.min sur cette arête (unités domaine).
    pub gap_min: Option<i32>,
    /// TemporalGap.max sur cette arête.
    pub gap_max: Option<i32>,
    /// dst.temporal_index - src.temporal_index (proxy si pas de TemporalGap).
    pub index_delta: Option<i32>,
}

/// Résultat de chain_temporal.
#[derive(Debug, Clone)]
pub struct TemporalChainResult {
    pub links: Option<Vec<TemporalLink>>,
    /// true si tous les sauts du chemin respectent temporal_index (src ≤ dst).
    pub temporally_ordered: bool,
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
/// Retourne None si aucune correspondance. Applique normalize_label (accents, casse, ponctuation).
pub fn match_score(label: &str, query: &str) -> Option<u8> {
    let l = normalize_label(label);
    let q = normalize_label(query);
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

pub fn build_maps_pub(ir: &CausalIR) -> (NodeMap<'_>, EdgeMap<'_>) {
    build_maps(ir)
}

fn build_maps(ir: &CausalIR) -> (NodeMap<'_>, EdgeMap<'_>) {
    let nm: NodeMap = ir.nodes.iter().map(|n| (n.id, n)).collect();
    // or_insert keeps the first occurrence for duplicate (src, dst) pairs;
    // collect() would silently keep the last (non-deterministic HashMap ordering).
    let mut em: EdgeMap = HashMap::new();
    for (s, d, e) in &ir.edges {
        em.entry((s.0, d.0)).or_insert(e);
    }
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

fn bfs_path_links(
    nm: &NodeMap,
    em: &EdgeMap,
    g: &CausalGraph,
    from: NodeId,
    to: NodeId,
) -> Vec<CausalLink> {
    let si = match g.node_indices.get(&from) {
        Some(&x) => x,
        None => return vec![],
    };
    let di = match g.node_indices.get(&to) {
        Some(&x) => x,
        None => return vec![],
    };

    let mut prev: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();
    queue.push_back(si);
    visited.insert(si);

    'bfs: while let Some(ni) = queue.pop_front() {
        if ni == di {
            break 'bfs;
        }
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
            Some(&p) => {
                path_ids.push(g.g[p]);
                cur = p;
            }
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
        from_label: nm
            .get(&src)
            .map(|n| n.label.clone())
            .unwrap_or_else(|| format!("node_{}", src.0)),
        to_label: nm
            .get(&dst)
            .map(|n| n.label.clone())
            .unwrap_or_else(|| format!("node_{}", dst.0)),
        from_id: src,
        to_id: dst,
        relation: e.relation,
        confidence: e.confidence,
        negated: e.negated,
    })
}

fn reachable_pairs_with_graph(g: &CausalGraph, nodes: &[gcn_ir::CausalNode], exclude: Option<NodeIndex>) -> usize {
    let mut count = 0;
    for src in nodes {
        if let Some(ex) = exclude {
            if g.node_indices.get(&src.id) == Some(&ex) { continue; }
        }
        let si = match g.node_indices.get(&src.id) { Some(&x) => x, None => continue };
        let mut visited = HashSet::new();
        visited.insert(si);
        if let Some(ex) = exclude { visited.insert(ex); }
        let mut queue = VecDeque::new();
        queue.push_back(si);
        while let Some(ni) = queue.pop_front() {
            for nb in g.g.neighbors_directed(ni, Direction::Outgoing) {
                if visited.insert(nb) {
                    queue.push_back(nb);
                    count += 1;
                }
            }
        }
    }
    count
}

/// Compte le nombre de paires (s,d) avec s≠d qui ont un chemin orienté dans le graphe.
pub fn count_reachable_pairs(ir: &CausalIR) -> usize {
    let g = build(ir);
    reachable_pairs_with_graph(&g, &ir.nodes, None)
}

/// Compte les paires atteignables après suppression virtuelle de `excluded`.
pub fn count_reachable_pairs_without(ir: &CausalIR, excluded: NodeId) -> usize {
    let g = build(ir);
    let excluded_idx = match g.node_indices.get(&excluded) {
        Some(&x) => x,
        None => return reachable_pairs_with_graph(&g, &ir.nodes, None),
    };
    reachable_pairs_with_graph(&g, &ir.nodes, Some(excluded_idx))
}

/// Calcule SPOF pour tous les nœuds en un seul build du graphe.
pub fn spof_all(ir: &CausalIR) -> (usize, Vec<(String, usize)>) {
    let g = build(ir);
    let total = reachable_pairs_with_graph(&g, &ir.nodes, None);
    let mut scores: Vec<(String, usize)> = ir.nodes.iter()
        .map(|n| {
            let ex = g.node_indices.get(&n.id).copied();
            let without = match ex {
                Some(idx) => reachable_pairs_with_graph(&g, &ir.nodes, Some(idx)),
                None => total,
            };
            (n.label.clone(), total.saturating_sub(without))
        })
        .collect();
    scores.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    (total, scores)
}

/// Hypothèse abductive — cause candidate d'un effet observé.
#[derive(Debug, Clone)]
pub struct AbductionHypothesis {
    pub label: String,
    pub relation: RelationType,
    /// Confiance de l'arête directe vers l'effet (ou vers le nœud intermédiaire).
    pub edge_confidence: f32,
    /// score = min_confidence_on_path / (1 + depth) — favorise les causes proches et fiables.
    pub score: f32,
    pub depth: usize,
}

/// EXPLAIN(E) — raisonnement abductif : quelles causes expliquent E ?
///
/// Algorithme : BFS inverse depuis E (ancêtres). Score de chaque ancêtre :
///   score = min_confidence_on_path / (1 + depth)
///
/// Le score favorise les causes proches (depth faible) et fiables (confiance élevée).
/// `confirmations` sera intégré quand le SA fournira les compteurs — pour l'instant = 1.
pub fn abduct(ir: &CausalIR, effect_id: NodeId) -> Vec<AbductionHypothesis> {
    let g = build(ir);
    let (nm, em) = build_maps(ir);
    let mut hypotheses: Vec<AbductionHypothesis> = Vec::new();

    let start = match g.node_indices.get(&effect_id) {
        Some(&si) => si,
        None => return vec![],
    };

    // BFS inverse avec tracking de profondeur et confiance min sur le chemin
    let mut queue: VecDeque<(NodeIndex, usize, f32)> = VecDeque::new();
    let mut visited: HashSet<NodeIndex> = HashSet::new();
    queue.push_back((start, 0, 1.0_f32));
    visited.insert(start);

    while let Some((ni, depth, path_conf)) = queue.pop_front() {
        if depth == 0 {
            // Le nœud de départ (effet) n'est pas une hypothèse
            for pred in g.g.neighbors_directed(ni, Direction::Incoming) {
                if visited.insert(pred) {
                    let pred_id = g.g[pred];
                    let edge_conf = em.get(&(pred_id.0, effect_id.0))
                        .map(|e| e.confidence)
                        .unwrap_or(0.5);
                    let new_path_conf = edge_conf;
                    queue.push_back((pred, 1, new_path_conf));
                    let score = new_path_conf / (1.0_f32 + 1.0_f32);
                    if let Some(node) = nm.get(&pred_id) {
                        hypotheses.push(AbductionHypothesis {
                            label: node.label.clone(),
                            relation: em.get(&(pred_id.0, g.g[ni].0))
                                .map(|e| e.relation)
                                .unwrap_or(RelationType::Cause),
                            edge_confidence: edge_conf,
                            score,
                            depth: 1,
                        });
                    }
                }
            }
            continue;
        }
        // Pour les nœuds plus profonds : propager le BFS mais ne pas ajouter comme
        // hypothèse additionnelle (on garde seulement les ancêtres directs de E pour
        // l'instant — extension future : ancêtres indirects avec score actualisé)
        for pred in g.g.neighbors_directed(ni, Direction::Incoming) {
            if visited.insert(pred) {
                let pred_id = g.g[pred];
                let edge_conf = em.get(&(pred_id.0, g.g[ni].0))
                    .map(|e| e.confidence)
                    .unwrap_or(0.5);
                let new_path_conf = path_conf.min(edge_conf);
                let new_depth = depth + 1;
                let score = new_path_conf / (1.0_f32 + new_depth as f32);
                if let Some(node) = nm.get(&pred_id) {
                    hypotheses.push(AbductionHypothesis {
                        label: node.label.clone(),
                        relation: em.get(&(pred_id.0, g.g[ni].0))
                            .map(|e| e.relation)
                            .unwrap_or(RelationType::Cause),
                        edge_confidence: edge_conf,
                        score,
                        depth: new_depth,
                    });
                }
                queue.push_back((pred, new_depth, new_path_conf));
            }
        }
    }

    // Trier par score décroissant, puis label pour déterminisme
    hypotheses.sort_by(|a, b| {
        b.score.partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.label.cmp(&b.label))
    });
    hypotheses
}

/// CHAIN_T — chemin causal avec info temporelle sur chaque saut.
/// Utilise le même BFS que chain() mais enrichit chaque lien avec temporal_gap
/// et index_delta. Vérifie si le chemin est temporellement ordonné.
pub fn chain_temporal(ir: &CausalIR, from: &str, to: &str) -> TemporalChainResult {
    let path = chain(ir, from, to);
    let (nm, em) = build_maps(ir);
    let idx_map: HashMap<NodeId, Option<i32>> =
        ir.nodes.iter().map(|n| (n.id, n.temporal_index)).collect();

    match path {
        None => TemporalChainResult { links: None, temporally_ordered: false },
        Some(links) => {
            let mut temporally_ordered = true;
            let temporal_links: Vec<TemporalLink> = links
                .iter()
                .map(|l| {
                    let (gap_min, gap_max) = em.get(&(l.from_id.0, l.to_id.0))
                        .and_then(|e| e.temporal_gap.as_ref())
                        .map(|g| (g.min, g.max))
                        .unwrap_or((None, None));
                    let src_ti = idx_map.get(&l.from_id).copied().flatten();
                    let dst_ti = idx_map.get(&l.to_id).copied().flatten();
                    let index_delta = match (src_ti, dst_ti) {
                        (Some(s), Some(d)) => {
                            if s > d {
                                temporally_ordered = false;
                            }
                            Some(d - s)
                        }
                        _ => None,
                    };
                    TemporalLink {
                        from_label: l.from_label.clone(),
                        to_label: l.to_label.clone(),
                        relation: l.relation,
                        confidence: l.confidence,
                        negated: l.negated,
                        gap_min,
                        gap_max,
                        index_delta,
                    }
                })
                .collect();
            TemporalChainResult {
                links: Some(temporal_links),
                temporally_ordered,
            }
        }
    }
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

    Some((
        target.label.clone(),
        InterventionResult { severed, effects },
    ))
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

    Some((
        target.label.clone(),
        CounterfactualResult {
            actual_effects,
            unique_effects,
        },
    ))
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
