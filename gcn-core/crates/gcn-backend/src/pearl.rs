// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use gcn_ir::{CausalIR, CycleType, NodeId, NodeOrigin, RelationType, Scope, normalize_label};
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
    /// Éq.12 — identifiant de groupe JOINT_CAUSE/JOINT_PREVENT (sha256[:16], déterministe).
    pub joint_group_id: Option<String>,
    /// Éq.12 — NodeId du tiers (médiateur ou condition) pour les arêtes ternaires.
    pub third_node: Option<NodeId>,
    /// Ph7 — motif de coupure (None = lien traversé ; Some = collecté `blocked_by`).
    pub motif: Option<BlockMotif>,
}

/// Ph7 — raison de coupure d'une arête (``blocked_by``, motif distinct par cause).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockMotif {
    /// [D1] Arête niée (propagation coupée, pas d'inversion).
    Negated,
    /// [7a] Concession/opposition (gaps causaux, comme les niées).
    CausalGap,
    /// [7-i] Nœud cible sans référent (scope null).
    NullScope,
}

impl BlockMotif {
    /// Nom snake_case pour ``LinkDto.motif`` (stable, snapshoté).
    pub fn as_str(self) -> &'static str {
        match self {
            BlockMotif::Negated => "negated",
            BlockMotif::CausalGap => "causal_gap",
            BlockMotif::NullScope => "null_scope",
        }
    }
}

/// Pearl niveau 2 — résultat d'une intervention do-calculus sur un nœud.
#[derive(Debug, Clone)]
pub struct InterventionResult {
    /// Arêtes entrantes coupées par l'intervention (les causes naturelles de X sont court-circuitées).
    /// Ph1 [D1] : inclut les entrantes niées (coupées 2x = coupées).
    pub severed: Vec<CausalLink>,
    /// Effets qui se propagent en aval depuis X une fois X forcé.
    /// Ph1 [D1] : jamais via une arête niée.
    pub effects: Vec<CausalLink>,
    /// Arêtes niées rencontrées en aval (pour `blocked_by`).
    pub blocked: Vec<CausalLink>,
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
    /// Ph1 [D1] : jamais via une arête niée.
    pub actual_effects: Vec<CausalLink>,
    /// Sous-ensemble des effets réels qui n'ont AUCUN autre chemin causal sans X.
    /// Ce sont les effets qui n'auraient pas eu lieu si X n'avait pas eu lieu.
    pub unique_effects: Vec<String>,
    /// Arêtes niées rencontrées depuis X (pour `blocked_by`).
    pub blocked: Vec<CausalLink>,
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

/// Ph2 [D2] — similarité de labels injectable (backend pur Rust, sans ML).
///
/// Score : 0 = exact, >0 = flou croissant, None = aucun rapport.
/// Le défaut (`DefaultLabelSim`) = paliers normalisés actuels (`match_score`).
/// Une implémentation cosinus recevrait les embeddings de l'appelant et
/// mapperait la similarité vers ces paliers (ex. sim ≥ 0.9 → 1, ≥ 0.7 → 2).
pub trait LabelSim {
    fn score(&self, label: &str, query: &str) -> Option<u8>;
}

/// Similarité par défaut : paliers normalisés (`match_score`).
pub struct DefaultLabelSim;

impl LabelSim for DefaultLabelSim {
    fn score(&self, label: &str, query: &str) -> Option<u8> {
        match_score(label, query)
    }
}

/// Tous les nœuds correspondants, triés par pertinence (exact > préfixe > contient).
pub fn find_all_ranked<'a>(ir: &'a CausalIR, query: &str) -> Vec<&'a gcn_ir::CausalNode> {
    find_all_ranked_with(ir, query, &DefaultLabelSim)
        .into_iter()
        .map(|(_, n)| n)
        .collect()
}

/// Tous les nœuds correspondants + leur score, triés (exact > flou, puis label court).
pub fn find_all_ranked_with<'a>(
    ir: &'a CausalIR,
    query: &str,
    sim: &dyn LabelSim,
) -> Vec<(u8, &'a gcn_ir::CausalNode)> {
    let mut v: Vec<(u8, usize, &'a gcn_ir::CausalNode)> = ir
        .nodes
        .iter()
        .filter_map(|n| sim.score(&n.label, query).map(|s| (s, n.label.len(), n)))
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    v.into_iter().map(|(s, _, n)| (s, n)).collect()
}

/// Ph2 [D2] — limite floue par défaut (`WHY`/`WHAT` sans `AMBIGUITY_LIMIT=`).
/// Compte les FLOUS seuls (les exacts multiples passent toujours).
pub const DEFAULT_AMBIGUITY_LIMIT: usize = 5;

/// Matchs cadrés exact-puis-flou pour WHY/WHAT (Ph2 [D2]).
#[derive(Debug, Clone)]
pub struct ScopedMatches {
    /// (label, liens, bloqués) des nœuds retenus.
    pub matched: Vec<(String, Vec<CausalLink>, Vec<CausalLink>)>,
    /// (label, score) des retenus : 0 = exact, >0 = flou.
    pub match_scores: Vec<(String, u8)>,
    /// Vrai si la réponse repose sur des flous (warning).
    pub fuzzy: bool,
}

/// Politique exact-puis-flou [D2] : exacts (tous) > flous (≤ `limit`) > `Err`.
///
/// `Err` = candidats flous en trop (appelant : `BackendError::Ambiguous`).
/// Sans aucun match : `Ok` vide (appelant : `NodeNotFound`, comme avant).
fn scope_matches_ctx(
    ctx: &GraphCtx<'_>,
    label: &str,
    sim: &dyn LabelSim,
    limit: usize,
    reverse: bool,
) -> Result<ScopedMatches, Vec<String>> {
    let ir = ctx.ir;
    let g = &ctx.g;
    let nm = &ctx.nm;
    let em = &ctx.em;
    let ranked = find_all_ranked_with(ir, label, sim);
    let (exacts, flous): (Vec<_>, Vec<_>) = ranked.into_iter().partition(|(s, _)| *s == 0);
    // Exact (score 0) toujours répondu, même multiples ; flou seulement si 0 exact.
    let picked: Vec<(u8, &gcn_ir::CausalNode)> = if !exacts.is_empty() {
        exacts
    } else if flous.len() <= limit {
        flous
    } else {
        return Err(flous.iter().map(|(_, n)| n.label.clone()).collect());
    };
    let fuzzy = picked.iter().any(|(s, _)| *s > 0);
    let mut matched = Vec::with_capacity(picked.len());
    let mut match_scores = Vec::with_capacity(picked.len());
    for (s, n) in picked {
        let (links, blocked) = if reverse {
            bfs_ancestors(nm, em, g, n.id)
        } else {
            bfs_descendants(nm, em, g, n.id)
        };
        matched.push((n.label.clone(), links, blocked));
        match_scores.push((n.label.clone(), s));
    }
    Ok(ScopedMatches {
        matched,
        match_scores,
        fuzzy,
    })
}

/// WHY cadré [D2] : exacts, sinon flous ≤ `limit`, sinon `Err(candidats)`.
pub fn why_scoped(ir: &CausalIR, label: &str, limit: usize) -> Result<ScopedMatches, Vec<String>> {
    why_scoped_with(ir, label, &DefaultLabelSim, limit)
}

/// Variante à similarité injectable (cosinus appelant, cf. `LabelSim`).
pub fn why_scoped_with(
    ir: &CausalIR,
    label: &str,
    sim: &dyn LabelSim,
    limit: usize,
) -> Result<ScopedMatches, Vec<String>> {
    scope_matches_ctx(&GraphCtx::new(ir), label, sim, limit, true)
}

/// M4 — WHY cadré sur contexte partagé (un seul build par requête).
pub(crate) fn why_scoped_ctx(
    ctx: &GraphCtx<'_>,
    label: &str,
    sim: &dyn LabelSim,
    limit: usize,
) -> Result<ScopedMatches, Vec<String>> {
    scope_matches_ctx(ctx, label, sim, limit, true)
}

/// WHAT cadré [D2] : même politique que `why_scoped`, propagation avant.
pub fn what_scoped(ir: &CausalIR, label: &str, limit: usize) -> Result<ScopedMatches, Vec<String>> {
    what_scoped_with(ir, label, &DefaultLabelSim, limit)
}

/// Variante à similarité injectable (cosinus appelant, cf. `LabelSim`).
pub fn what_scoped_with(
    ir: &CausalIR,
    label: &str,
    sim: &dyn LabelSim,
    limit: usize,
) -> Result<ScopedMatches, Vec<String>> {
    scope_matches_ctx(&GraphCtx::new(ir), label, sim, limit, false)
}

/// M4 — WHAT cadré sur contexte partagé (un seul build par requête).
pub(crate) fn what_scoped_ctx(
    ctx: &GraphCtx<'_>,
    label: &str,
    sim: &dyn LabelSim,
    limit: usize,
) -> Result<ScopedMatches, Vec<String>> {
    scope_matches_ctx(ctx, label, sim, limit, false)
}

type NodeMap<'a> = HashMap<NodeId, &'a gcn_ir::CausalNode>;
type EdgeMap<'a> = HashMap<(u32, u32), &'a gcn_ir::CausalEdge>;

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

/// Ph10-M4 — contexte de graphe partagé : UN SEUL calcul par requête.
///
/// `build` (petgraph) + `build_maps` (index nœuds/arêtes) sont purs en `ir` ;
/// `execute()` construit un `GraphCtx` à l'arrivée de la requête et le passe
/// à toutes les fonctions d'analyse (`*_ctx`). Aucun changement de résultat :
/// les variantes `ir` restent en wrappers fins (appelants directs + tests).
pub struct GraphCtx<'a> {
    pub(crate) ir: &'a CausalIR,
    pub(crate) g: CausalGraph,
    pub(crate) nm: NodeMap<'a>,
    pub(crate) em: EdgeMap<'a>,
}

impl<'a> GraphCtx<'a> {
    pub fn new(ir: &'a CausalIR) -> Self {
        let g = build(ir);
        let (nm, em) = build_maps(ir);
        GraphCtx { ir, g, nm, em }
    }

    /// Label d'un nœud via l'index (remplace les scans linéaires `node_label`).
    pub(crate) fn node_label(&self, id: NodeId) -> String {
        self.nm
            .get(&id)
            .map(|n| n.label.clone())
            .unwrap_or_else(|| format!("node_{}", id.0))
    }
}

/// Ph7 — raison de coupure d'une arête (None = traversable).
/// [D1] niée > [7a] gap concessif/oppositif > [7-i] cible sans référent.
/// Le premier motif applicable gagne (ordre déterministe, documenté).
fn edge_block_motif(nm: &NodeMap, em: &EdgeMap, src: NodeId, dst: NodeId) -> Option<BlockMotif> {
    let e = em.get(&(src.0, dst.0))?;
    if e.negated {
        return Some(BlockMotif::Negated);
    }
    if e.relation.signals_causal_gap() {
        return Some(BlockMotif::CausalGap);
    }
    if nm.get(&dst).is_some_and(|n| n.scope == Scope::Null) {
        return Some(BlockMotif::NullScope);
    }
    None
}

/// Tri déterministe des liens bloqués (snapshots stables).
fn sort_blocked(blocked: &mut [CausalLink]) {
    blocked.sort_by(|a, b| {
        (a.from_id.0, a.to_id.0)
            .cmp(&(b.from_id.0, b.to_id.0))
            .then(a.from_label.cmp(&b.from_label))
    });
}

/// WHY: reverse BFS — find all causal ancestors of nodes matching `label`
///
/// Éq.12 : pour les arêtes JointCause/JointPrevent, joint_group_id est propagé
/// dans chaque CausalLink — les co-sources partageant le même joint_group_id
/// représentent une nécessité conjointe (A₁ ET A₂ ensemble).
///
/// Ph1 [D1] : les arêtes niées ne sont pas traversées (propagation coupée) ;
/// elles sont collectées en 3e position pour `blocked_by`.
pub fn why(ir: &CausalIR, label: &str) -> Vec<(String, Vec<CausalLink>, Vec<CausalLink>)> {
    why_ctx(&GraphCtx::new(ir), label)
}

/// M4 — cœur partageable (contexte construit une fois par requête).
pub(crate) fn why_ctx(
    ctx: &GraphCtx<'_>,
    label: &str,
) -> Vec<(String, Vec<CausalLink>, Vec<CausalLink>)> {
    let ir = ctx.ir;
    let g = &ctx.g;
    let nm = &ctx.nm;
    let em = &ctx.em;
    find_all_ranked(ir, label)
        .into_iter()
        .map(|n| {
            let (links, blocked) = bfs_ancestors(nm, em, g, n.id);
            (n.label.clone(), links, blocked)
        })
        .collect()
}

/// WHAT: forward BFS — find all causal descendants of nodes matching `label`
///
/// Ph1 [D1] : même coupure par négation que WHY (3e position = `blocked_by`).
pub fn what(ir: &CausalIR, label: &str) -> Vec<(String, Vec<CausalLink>, Vec<CausalLink>)> {
    what_ctx(&GraphCtx::new(ir), label)
}

/// M4 — cœur partageable (contexte construit une fois par requête).
pub(crate) fn what_ctx(
    ctx: &GraphCtx<'_>,
    label: &str,
) -> Vec<(String, Vec<CausalLink>, Vec<CausalLink>)> {
    let ir = ctx.ir;
    let g = &ctx.g;
    let nm = &ctx.nm;
    let em = &ctx.em;
    find_all_ranked(ir, label)
        .into_iter()
        .map(|n| {
            let (links, blocked) = bfs_descendants(nm, em, g, n.id);
            (n.label.clone(), links, blocked)
        })
        .collect()
}

/// CHAIN: find causal path from the best node matching `from` to `to`
///
/// Éq.12 : si un lien JointCause est sur le chemin, joint_group_id est inclus
/// dans le CausalLink. L'appelant peut inspecter joint_group_id pour vérifier
/// que les 2 co-sources sont présentes (partial_joint = une seule source trouvée).
///
/// Ph1 [D1] : le chemin ne traverse jamais d'arête niée. Voir
/// `chain_with_blocked` pour le détail des coupures.
pub fn chain(ir: &CausalIR, from: &str, to: &str) -> Option<Vec<CausalLink>> {
    chain_ctx(&GraphCtx::new(ir), from, to)
}

/// M4 — cœur partageable (contexte construit une fois par requête).
pub(crate) fn chain_ctx(ctx: &GraphCtx<'_>, from: &str, to: &str) -> Option<Vec<CausalLink>> {
    chain_with_blocked_ctx(ctx, from, to).0
}

/// Ph3 [D3] — coût d'arête pour Dijkstra confiance-max : −ln(conf),
/// conf ≤ 0 → epsilon (jamais d'infini).
fn edge_cost(conf: f32) -> f32 {
    -(conf.max(1e-6)).ln()
}

/// Coût totalement ordonné (file de priorité déterministe).
#[derive(Clone, Copy, PartialEq)]
struct Cost(f32);

impl Eq for Cost {}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cost {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Entrée de file Dijkstra : (coût, id nœud, séquence) — total + déterministe.
type HeapEntry = (std::cmp::Reverse<(Cost, u32, u64)>, NodeIndex);

/// Ph3 [D3] — chemin classé : liens + confiance produit.
#[derive(Debug, Clone)]
pub struct ScoredPath {
    pub links: Vec<CausalLink>,
    /// Produit des confidences (1.0 si chemin vide from == to).
    pub confidence: f32,
}

/// Dijkstra confiance-max (Ph3 [D3]) : coût −ln(conf), coupures jamais traversées.
/// `removed_edges`/`removed_nodes` : exclusions temporaires (Yen).
/// Retourne (nœuds du chemin, coût total). `None` si inatteignable.
fn dijkstra_path(
    nm: &NodeMap,
    em: &EdgeMap,
    g: &CausalGraph,
    src: NodeId,
    dst: NodeId,
    removed_edges: &HashSet<(u32, u32)>,
    removed_nodes: &HashSet<NodeId>,
) -> Option<(Vec<NodeId>, f32)> {
    let start = *g.node_indices.get(&src)?;
    let goal = *g.node_indices.get(&dst)?;
    if removed_nodes.contains(&src) || removed_nodes.contains(&dst) {
        return None;
    }
    let mut dist: HashMap<NodeIndex, f32> = HashMap::new();
    let mut prev: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    // (coût, id nœud, séquence) : total + déterministe.
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
    let mut seq: u64 = 0;
    dist.insert(start, 0.0);
    heap.push((std::cmp::Reverse((Cost(0.0), g.g[start].0, seq)), start));
    while let Some((std::cmp::Reverse((cost, _, _)), ni)) = heap.pop() {
        if ni == goal {
            let mut path = vec![g.g[ni]];
            let mut cur = ni;
            while cur != start {
                cur = *prev.get(&cur)?;
                path.push(g.g[cur]);
            }
            path.reverse();
            return Some((path, cost.0));
        }
        if cost.0 > dist.get(&ni).copied().unwrap_or(f32::INFINITY) {
            continue;
        }
        // Voisins triés par NodeId : exploration déterministe.
        let mut succs: Vec<NodeIndex> = g.g.neighbors_directed(ni, Direction::Outgoing).collect();
        succs.sort_by_key(|n| g.g[*n].0);
        for nb in succs {
            let (a, b) = (g.g[ni], g.g[nb]);
            if removed_nodes.contains(&b) || removed_edges.contains(&(a.0, b.0)) {
                continue;
            }
            let e = match em.get(&(a.0, b.0)) {
                Some(e) => e,
                None => continue,
            };
            // Ph1 [D1] + Ph7 : coupures (niées, gaps, null) jamais traversées.
            if edge_block_motif(nm, em, a, b).is_some() {
                continue;
            }
            // Ph7b : coût × poids d'origine (cause attestée > hypothétique).
            let c = cost.0 + edge_cost(e.confidence * node_origin_weight(nm, a));
            if c < dist.get(&nb).copied().unwrap_or(f32::INFINITY) {
                dist.insert(nb, c);
                prev.insert(nb, ni);
                seq += 1;
                heap.push((std::cmp::Reverse((Cost(c), b.0, seq)), nb));
            }
        }
    }
    None
}

/// Coût total −ln(conf × poids origine) d'un chemin de nœuds (comparaison Yen).
fn path_cost(nm: &NodeMap, em: &EdgeMap, path: &[NodeId]) -> f32 {
    path.windows(2)
        .map(|w| {
            em.get(&(w[0].0, w[1].0))
                .map(|e| edge_cost(e.confidence * node_origin_weight(nm, w[0])))
                .unwrap_or(f32::INFINITY)
        })
        .sum()
}

fn links_of_path(nm: &NodeMap, em: &EdgeMap, path: &[NodeId]) -> (Vec<CausalLink>, f32) {
    let mut links = Vec::new();
    let mut prod = 1.0f32;
    for w in path.windows(2) {
        if let Some(l) = edge_link(nm, em, w[0], w[1]) {
            prod *= l.confidence;
            links.push(l);
        }
    }
    (links, prod)
}

/// K meilleurs chemins (Yen, Ph3 [D3]) : Dijkstra itéré avec exclusions.
/// `k ≥ 1`. Chemins ordonnés (confiance décroissante), déterministes.
fn k_shortest_paths(
    nm: &NodeMap,
    em: &EdgeMap,
    g: &CausalGraph,
    src: NodeId,
    dst: NodeId,
    k: usize,
) -> Vec<ScoredPath> {
    let empty_e: HashSet<(u32, u32)> = HashSet::new();
    let empty_n: HashSet<NodeId> = HashSet::new();
    let Some(first) = dijkstra_path(nm, em, g, src, dst, &empty_e, &empty_n) else {
        return vec![];
    };
    let mut known: Vec<Vec<NodeId>> = vec![first.0];
    let mut candidates: Vec<(f32, Vec<NodeId>)> = Vec::new();
    for _ in 1..k {
        let prev_path = &known[known.len() - 1];
        if prev_path.len() < 2 {
            break; // from == to : un seul chemin vide.
        }
        for spur_idx in 0..prev_path.len() - 1 {
            let spur = prev_path[spur_idx];
            let root: Vec<NodeId> = prev_path[..=spur_idx].to_vec();
            let mut rem_e: HashSet<(u32, u32)> = HashSet::new();
            let mut rem_n: HashSet<NodeId> = HashSet::new();
            for p in &known {
                if p.len() > spur_idx && p[..=spur_idx] == root[..] {
                    rem_e.insert((p[spur_idx].0, p[spur_idx + 1].0));
                }
            }
            for n in &root[..root.len() - 1] {
                rem_n.insert(*n);
            }
            if let Some((spur_path, _)) = dijkstra_path(nm, em, g, spur, dst, &rem_e, &rem_n) {
                let mut full = root[..root.len() - 1].to_vec();
                full.extend(spur_path);
                if !known.contains(&full) && !candidates.iter().any(|(_, p)| *p == full) {
                    candidates.push((path_cost(nm, em, &full), full));
                }
            }
        }
        if candidates.is_empty() {
            break;
        }
        candidates.sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then(a.1.len().cmp(&b.1.len()))
                .then(a.1.iter().map(|n| n.0).cmp(b.1.iter().map(|n| n.0)))
        });
        let (_, best) = candidates.remove(0);
        known.push(best);
    }
    known
        .into_iter()
        .map(|p| {
            let (links, confidence) = links_of_path(nm, em, &p);
            ScoredPath { links, confidence }
        })
        .collect()
}

/// CHAIN avec détail des coupures : chemin sans arête niée + arêtes niées
/// qui auraient porté un chemin (pour `blocked_by`).
///
/// - `(Some(path), [])` : chemin causal pur trouvé.
/// - `(None, bloqués non vides)` : nœuds reliés MAIS uniquement via des niées
///   (coupure par négation — `found:false` + `blocked_by` rempli).
/// - `(None, [])` : vraiment aucun chemin (ni pur ni nié).
pub fn chain_with_blocked(
    ir: &CausalIR,
    from: &str,
    to: &str,
) -> (Option<Vec<CausalLink>>, Vec<CausalLink>) {
    chain_with_blocked_ctx(&GraphCtx::new(ir), from, to)
}

/// M4 — cœur partageable (contexte construit une fois par requête).
pub(crate) fn chain_with_blocked_ctx(
    ctx: &GraphCtx<'_>,
    from: &str,
    to: &str,
) -> (Option<Vec<CausalLink>>, Vec<CausalLink>) {
    let (paths, blocked) = chain_topk_ctx(ctx, from, to, 1);
    (paths.into_iter().next().map(|p| p.links), blocked)
}

/// CHAIN top-k (Ph3 [D3]) : jusqu'à `k` chemins confiance-max + niées de coupure.
///
/// - chemins : `k` meilleurs (Dijkstra + Yen), vide si inatteignable sans coupure ;
/// - bloqués : coupures (niées, gaps, null) du chemin non filtré ssi aucun
///   chemin pur (cf. `chain_with_blocked`), avec leur motif.
pub fn chain_topk(
    ir: &CausalIR,
    from: &str,
    to: &str,
    k: usize,
) -> (Vec<ScoredPath>, Vec<CausalLink>) {
    chain_topk_ctx(&GraphCtx::new(ir), from, to, k)
}

/// M4 — cœur partageable (contexte construit une fois par requête).
pub(crate) fn chain_topk_ctx(
    ctx: &GraphCtx<'_>,
    from: &str,
    to: &str,
    k: usize,
) -> (Vec<ScoredPath>, Vec<CausalLink>) {
    let ir = ctx.ir;
    let g = &ctx.g;
    let nm = &ctx.nm;
    let em = &ctx.em;
    let (Some(src), Some(dst)) = (find_best(ir, from), find_best(ir, to)) else {
        return (vec![], vec![]);
    };
    let paths = k_shortest_paths(nm, em, g, src.id, dst.id, k.max(1));
    if !paths.is_empty() {
        return (paths, vec![]);
    }
    // Pas de chemin pur : coupures du chemin non filtré (borne : BFS historique).
    let full = bfs_path_links(nm, em, g, src.id, dst.id, false);
    if full.is_empty() {
        return (vec![], vec![]);
    }
    let mut blocked: Vec<CausalLink> = full
        .into_iter()
        .filter_map(|mut l| {
            l.motif = edge_block_motif(nm, em, l.from_id, l.to_id);
            l.motif.map(|_| l)
        })
        .collect();
    sort_blocked(&mut blocked);
    (vec![], blocked)
}

/// BFS ancêtres avec force (Ph3 [D3]) : chaque lien porte le produit des
/// confidences depuis le départ (chaînes fortes d'abord, déterministe).
fn bfs_ancestors(
    nm: &NodeMap,
    em: &EdgeMap,
    g: &CausalGraph,
    start: NodeId,
) -> (Vec<CausalLink>, Vec<CausalLink>) {
    let mut visited = HashSet::new();
    let mut queue: VecDeque<(NodeIndex, f32)> = VecDeque::new();
    let mut links: Vec<(CausalLink, f32)> = Vec::new();
    let mut blocked = Vec::new();

    if let Some(&si) = g.node_indices.get(&start) {
        queue.push_back((si, 1.0));
        visited.insert(si);
    }

    while let Some((ni, prod)) = queue.pop_front() {
        let cur_id = g.g[ni];
        for pred in g.g.neighbors_directed(ni, Direction::Incoming) {
            let pred_id = g.g[pred];
            // Ph1 [D1] + Ph7 : arête coupée (niée, gap, null) = collectée
            // `blocked_by` avec son motif (le prédécesseur reste atteignable
            // par un autre chemin pur).
            if let Some(motif) = edge_block_motif(nm, em, pred_id, cur_id) {
                if let Some(mut link) = edge_link(nm, em, pred_id, cur_id) {
                    link.motif = Some(motif);
                    blocked.push(link);
                }
                continue;
            }
            if visited.insert(pred) {
                if let Some(link) = edge_link(nm, em, pred_id, cur_id) {
                    // Ph7b : force × poids d'origine (cause attestée > hypothétique).
                    let strength = prod * link.confidence * node_origin_weight(nm, pred_id);
                    queue.push_back((pred, strength));
                    links.push((link, strength));
                } else {
                    queue.push_back((pred, prod));
                }
            }
        }
    }
    sort_by_strength(&mut links);
    sort_blocked(&mut blocked);
    (links.into_iter().map(|(l, _)| l).collect(), blocked)
}

/// Tri décroissant par force, égalités par (from, to) — déterministe.
fn sort_by_strength(links: &mut [(CausalLink, f32)]) {
    links.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then((a.0.from_id.0, a.0.to_id.0).cmp(&(b.0.from_id.0, b.0.to_id.0)))
    });
}

/// BFS descendants avec force (Ph3 [D3]) : voir `bfs_ancestors`.
fn bfs_descendants(
    nm: &NodeMap,
    em: &EdgeMap,
    g: &CausalGraph,
    start: NodeId,
) -> (Vec<CausalLink>, Vec<CausalLink>) {
    let mut visited = HashSet::new();
    let mut queue: VecDeque<(NodeIndex, f32)> = VecDeque::new();
    let mut links: Vec<(CausalLink, f32)> = Vec::new();
    let mut blocked = Vec::new();

    if let Some(&si) = g.node_indices.get(&start) {
        queue.push_back((si, 1.0));
        visited.insert(si);
    }

    while let Some((ni, prod)) = queue.pop_front() {
        let cur_id = g.g[ni];
        for succ in g.g.neighbors_directed(ni, Direction::Outgoing) {
            let succ_id = g.g[succ];
            // Ph1 [D1] + Ph7 : voir bfs_ancestors.
            if let Some(motif) = edge_block_motif(nm, em, cur_id, succ_id) {
                if let Some(mut link) = edge_link(nm, em, cur_id, succ_id) {
                    link.motif = Some(motif);
                    blocked.push(link);
                }
                continue;
            }
            if visited.insert(succ) {
                if let Some(link) = edge_link(nm, em, cur_id, succ_id) {
                    // Ph7b : voir bfs_ancestors (poids du nœud source).
                    let strength = prod * link.confidence * node_origin_weight(nm, cur_id);
                    queue.push_back((succ, strength));
                    links.push((link, strength));
                } else {
                    queue.push_back((succ, prod));
                }
            }
        }
    }
    sort_by_strength(&mut links);
    sort_blocked(&mut blocked);
    (links.into_iter().map(|(l, _)| l).collect(), blocked)
}

fn bfs_path_links(
    nm: &NodeMap,
    em: &EdgeMap,
    g: &CausalGraph,
    from: NodeId,
    to: NodeId,
    avoid_blocked: bool,
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
            // Ph1 [D1] + Ph7 : en mode strict, ne jamais router à travers
            // une arête coupée (niée, gap, null).
            if avoid_blocked && edge_block_motif(nm, em, g.g[ni], g.g[nb]).is_some() {
                continue;
            }
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

/// Ph6b — liens directs 1 saut d'un nœud : (entrants, sortants).
/// Sert ASK (fiche nœud) sans traversée (pas de propagation, pas de niée).
pub fn direct_links(ir: &CausalIR, id: NodeId) -> (Vec<CausalLink>, Vec<CausalLink>) {
    direct_links_ctx(&GraphCtx::new(ir), id)
}

/// M4 — cœur partageable (contexte construit une fois par requête).
pub(crate) fn direct_links_ctx(
    ctx: &GraphCtx<'_>,
    id: NodeId,
) -> (Vec<CausalLink>, Vec<CausalLink>) {
    let ir = ctx.ir;
    let nm = &ctx.nm;
    let em = &ctx.em;
    let mut entrants = Vec::new();
    let mut sortants = Vec::new();
    for (s, d, _) in &ir.edges {
        if *d == id
            && let Some(link) = edge_link(nm, em, *s, *d)
        {
            entrants.push(link);
        }
        if *s == id
            && let Some(link) = edge_link(nm, em, *s, *d)
        {
            sortants.push(link);
        }
    }
    sort_by_strength_pairs(&mut entrants);
    sort_by_strength_pairs(&mut sortants);
    (entrants, sortants)
}

/// Tri décroissant par confiance (liens directs, déterministe).
fn sort_by_strength_pairs(links: &mut [CausalLink]) {
    links.sort_by(|a, b| {
        b.confidence
            .total_cmp(&a.confidence)
            .then((a.from_id.0, a.to_id.0).cmp(&(b.from_id.0, b.to_id.0)))
    });
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
        // Ph4 : nécessité conjointe — un groupe joint n'est valide qu'à ≥ 2 arêtes ;
        // orphelin = invalide → arête simple (joint_group_id effacé).
        joint_group_id: e
            .joint_group_id
            .as_ref()
            .filter(|gid| {
                em.values()
                    .filter(|x| x.joint_group_id.as_deref() == Some(gid.as_str()))
                    .take(2)
                    .count()
                    == 2
            })
            .cloned(),
        third_node: e.third.as_ref().map(|t| NodeId(t.node as u32)),
        // Ph7 : motif positionné par les sites de collecte `blocked_by`.
        motif: None,
    })
}

/// Éq.12 — groupe les arêtes JointCause/JointPrevent par joint_group_id.
/// Ph4 : groupes orphelins (< 2 arêtes) exclus (joints invalides, cf. `edge_link`).
fn group_by_joint_id(em: &EdgeMap) -> HashMap<String, Vec<(NodeId, NodeId)>> {
    let mut groups: HashMap<String, Vec<(NodeId, NodeId)>> = em
        .iter()
        .filter_map(|((s, d), e)| {
            e.joint_group_id
                .as_ref()
                .map(|g| (g.clone(), (NodeId(*s), NodeId(*d))))
        })
        .fold(HashMap::new(), |mut m, (g, p)| {
            m.entry(g).or_default().push(p);
            m
        });
    groups.retain(|_, v| v.len() >= 2);
    groups
}

fn reachable_pairs_with_graph(
    g: &CausalGraph,
    nodes: &[gcn_ir::CausalNode],
    exclude: Option<NodeIndex>,
) -> usize {
    let mut count = 0;
    for src in nodes {
        if let Some(ex) = exclude
            && g.node_indices.get(&src.id) == Some(&ex)
        {
            continue;
        }
        let si = match g.node_indices.get(&src.id) {
            Some(&x) => x,
            None => continue,
        };
        let mut visited = HashSet::new();
        visited.insert(si);
        if let Some(ex) = exclude {
            visited.insert(ex);
        }
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

/// Garde anti-DoS SPOF (source unique — query.rs la réutilise, pas de doublon).
pub const SPOF_MAX_NODES: usize = 500;

/// Ph7b [7-ii] — poids d'origine FIGÉS (tri WHY/WHAT/CHAIN, documentés) :
/// Explicit=1.0 > Inferred=0.7 > Hypothetical=0.3. Nommés (pas de magiques) ;
/// la composition session (9a) vit côté Python (D-ORIG).
pub const ORIGIN_WEIGHT_EXPLICIT: f32 = 1.0;
pub const ORIGIN_WEIGHT_INFERRED: f32 = 0.7;
pub const ORIGIN_WEIGHT_HYPOTHETICAL: f32 = 0.3;

/// Poids d'origine d'un nœud (1.0 par défaut si inconnu).
pub fn origin_weight(origin: NodeOrigin) -> f32 {
    match origin {
        NodeOrigin::Explicit => ORIGIN_WEIGHT_EXPLICIT,
        NodeOrigin::Inferred => ORIGIN_WEIGHT_INFERRED,
        NodeOrigin::Hypothetical => ORIGIN_WEIGHT_HYPOTHETICAL,
    }
}

/// Poids d'origine d'un nœud par id (1.0 si nœud inconnu).
fn node_origin_weight(nm: &NodeMap, id: NodeId) -> f32 {
    nm.get(&id).map(|n| origin_weight(n.origin)).unwrap_or(1.0)
}

/// Score SPOF étendu avec flag super_spof (Éq.12).
#[derive(Debug, Clone)]
pub struct SpofScore {
    pub label: String,
    pub score: usize,
    /// true si le nœud est source d'au moins 1 arête JointCause/JointPrevent.
    pub is_super_spof: bool,
}

/// Calcule SPOF pour tous les nœuds en un seul build du graphe.
///
/// Éq.12 : `is_super_spof` = vrai si le nœud apparaît dans au moins un groupe joint_group_id.
///
/// §2.13 ETUDE : garde anti-DoS — O(n²·(n+e)) sans garde exposait tout
/// appelant direct de la crate (le CLI garde déjà 500 nœuds dans query.rs).
/// Retourne `Err` au-delà de `SPOF_MAX_NODES` au lieu de calculer.
pub fn spof_all(ir: &CausalIR) -> Result<(usize, Vec<SpofScore>), crate::error::BackendError> {
    spof_all_ctx(&GraphCtx::new(ir))
}

/// M4 — cœur partageable (contexte construit une fois par requête).
pub(crate) fn spof_all_ctx(
    ctx: &GraphCtx<'_>,
) -> Result<(usize, Vec<SpofScore>), crate::error::BackendError> {
    let ir = ctx.ir;
    if ir.nodes.len() > SPOF_MAX_NODES {
        return Err(crate::error::BackendError::QueryParseError(format!(
            "SPOF: graph too large ({} nodes > {})",
            ir.nodes.len(),
            SPOF_MAX_NODES
        )));
    }
    let g = &ctx.g;
    let em = &ctx.em;
    let total = reachable_pairs_with_graph(g, &ir.nodes, None);

    // Éq.12 — nœuds impliqués dans un joint_group
    let joint_sources: HashSet<NodeId> = em
        .iter()
        .filter(|(_, e)| e.joint_group_id.is_some())
        .map(|((s, _), _)| NodeId(*s))
        .collect();

    let mut scores: Vec<SpofScore> = ir
        .nodes
        .iter()
        .map(|n| {
            let ex = g.node_indices.get(&n.id).copied();
            let without = match ex {
                Some(idx) => reachable_pairs_with_graph(g, &ir.nodes, Some(idx)),
                None => total,
            };
            SpofScore {
                label: n.label.clone(),
                score: total.saturating_sub(without),
                is_super_spof: joint_sources.contains(&n.id),
            }
        })
        .collect();
    scores.sort_by(|a, b| b.score.cmp(&a.score).then(a.label.cmp(&b.label)));
    Ok((total, scores))
}

/// Hypothèse abductive — cause candidate d'un effet observé.
#[derive(Debug, Clone)]
pub struct AbductionHypothesis {
    /// Nœud cause (regroupement Ph7d par agent_type, sans ambiguïté de label).
    pub node_id: NodeId,
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
/// Ph7d — `group_by_agent` scinde les hypothèses par agent_type (tri stable,
/// `None` = non renseigné en dernier).
pub fn group_by_agent(
    ir: &CausalIR,
    hypotheses: &[AbductionHypothesis],
) -> Vec<(Option<String>, Vec<AbductionHypothesis>)> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<Option<String>, Vec<AbductionHypothesis>> = BTreeMap::new();
    for h in hypotheses {
        let key = ir
            .nodes
            .iter()
            .find(|n| n.id == h.node_id)
            .and_then(|n| n.attributes.agent_type)
            .map(|t| serde_agent_type(&t));
        groups.entry(key).or_default().push(h.clone());
    }
    groups.into_iter().collect()
}

fn serde_agent_type(t: &gcn_ir::AgentType) -> String {
    match t {
        gcn_ir::AgentType::Human => "human".to_string(),
        gcn_ir::AgentType::Collective => "collective".to_string(),
        gcn_ir::AgentType::Institutional => "institutional".to_string(),
        gcn_ir::AgentType::Natural => "natural".to_string(),
        gcn_ir::AgentType::System => "system".to_string(),
    }
}
pub fn abduct(ir: &CausalIR, effect_id: NodeId) -> Vec<AbductionHypothesis> {
    abduct_ctx(&GraphCtx::new(ir), effect_id)
}

/// M4 — cœur partageable (contexte construit une fois par requête).
pub(crate) fn abduct_ctx(ctx: &GraphCtx<'_>, effect_id: NodeId) -> Vec<AbductionHypothesis> {
    let g = &ctx.g;
    let nm = &ctx.nm;
    let em = &ctx.em;
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
                let pred_id = g.g[pred];
                // Ph1 [D1] + Ph7 : pas d'hypothèse à travers une coupure.
                if edge_block_motif(nm, em, pred_id, effect_id).is_some() {
                    continue;
                }
                if visited.insert(pred) {
                    let edge_conf = em
                        .get(&(pred_id.0, effect_id.0))
                        .map(|e| e.confidence)
                        .unwrap_or(0.5);
                    let new_path_conf = edge_conf;
                    queue.push_back((pred, 1, new_path_conf));
                    let score = new_path_conf / (1.0_f32 + 1.0_f32);
                    if let Some(node) = nm.get(&pred_id) {
                        hypotheses.push(AbductionHypothesis {
                            node_id: pred_id,
                            label: node.label.clone(),
                            relation: em
                                .get(&(pred_id.0, effect_id.0))
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
        let cur_id = g.g[ni];
        for pred in g.g.neighbors_directed(ni, Direction::Incoming) {
            let pred_id = g.g[pred];
            // Ph1 [D1] + Ph7 : pas de propagation à travers une coupure.
            if edge_block_motif(nm, em, pred_id, cur_id).is_some() {
                continue;
            }
            if visited.insert(pred) {
                let edge_conf = em
                    .get(&(pred_id.0, cur_id.0))
                    .map(|e| e.confidence)
                    .unwrap_or(0.5);
                let new_path_conf = path_conf.min(edge_conf);
                let new_depth = depth + 1;
                let score = new_path_conf / (1.0_f32 + new_depth as f32);
                if let Some(node) = nm.get(&pred_id) {
                    hypotheses.push(AbductionHypothesis {
                        node_id: pred_id,
                        label: node.label.clone(),
                        relation: em
                            .get(&(pred_id.0, g.g[ni].0))
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
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.label.cmp(&b.label))
    });
    hypotheses
}

/// CHAIN_T — chemin causal avec info temporelle sur chaque saut.
/// Utilise le même BFS que chain() mais enrichit chaque lien avec temporal_gap
/// et index_delta. Vérifie si le chemin est temporellement ordonné.
///
/// Ph7a : un chemin traversant une boucle OSCILLATION sans info de phase
/// (aucun TemporalGap sur ses arêtes) est rejeté (None) — CHAIN_T exige
/// la phase pour raisonner sur une oscillation.
pub fn chain_temporal(ir: &CausalIR, from: &str, to: &str) -> TemporalChainResult {
    chain_temporal_ctx(&GraphCtx::new(ir), from, to)
}

/// M4 — cœur partageable (contexte construit une fois par requête).
pub(crate) fn chain_temporal_ctx(ctx: &GraphCtx<'_>, from: &str, to: &str) -> TemporalChainResult {
    let ir = ctx.ir;
    let em = &ctx.em;
    let path = chain_ctx(ctx, from, to);
    let idx_map: HashMap<NodeId, Option<i32>> =
        ir.nodes.iter().map(|n| (n.id, n.temporal_index)).collect();

    match path {
        None => TemporalChainResult {
            links: None,
            temporally_ordered: false,
        },
        Some(links) => {
            // Ph7a : oscillation sans phase → rejet (pas de raisonnement
            // temporel fiable sur une boucle oscillante non phasée).
            let oscille_sans_phase = links.iter().any(|l| {
                em.get(&(l.from_id.0, l.to_id.0)).is_some_and(|e| {
                    e.temporal_gap.is_none()
                        && e.in_cycle.is_some_and(|cid| {
                            ir.cycles.iter().any(|c| {
                                c.id == cid && matches!(c.cycle_type, CycleType::Oscillation)
                            })
                        })
                })
            });
            if oscille_sans_phase {
                return TemporalChainResult {
                    links: None,
                    temporally_ordered: false,
                };
            }
            let mut temporally_ordered = true;
            let mut has_temporal_data = false;
            let temporal_links: Vec<TemporalLink> = links
                .iter()
                .map(|l| {
                    let (gap_min, gap_max) = em
                        .get(&(l.from_id.0, l.to_id.0))
                        .and_then(|e| e.temporal_gap.as_ref())
                        .map(|g| (g.min, g.max))
                        .unwrap_or((None, None));
                    let src_ti = idx_map.get(&l.from_id).copied().flatten();
                    let dst_ti = idx_map.get(&l.to_id).copied().flatten();
                    let index_delta = match (src_ti, dst_ti) {
                        (Some(s), Some(d)) => {
                            has_temporal_data = true;
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
                temporally_ordered: has_temporal_data && temporally_ordered,
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
///
/// Ph1 [D1] : les entrantes niées sont incluses dans `severed` (coupées 2x =
/// coupées) ; les effets ne traversent aucune arête niée.
pub fn intervene(ir: &CausalIR, target_label: &str) -> Option<(String, InterventionResult)> {
    intervene_ctx(&GraphCtx::new(ir), target_label)
}

/// M4 — cœur partageable (contexte construit une fois par requête).
pub(crate) fn intervene_ctx(
    ctx: &GraphCtx<'_>,
    target_label: &str,
) -> Option<(String, InterventionResult)> {
    let ir = ctx.ir;
    let g = &ctx.g;
    let nm = &ctx.nm;
    let em = &ctx.em;
    let target = find_best(ir, target_label)?;

    // Arêtes entrantes coupées par l'intervention (niées incluses, cf. ci-dessus)
    let severed: Vec<CausalLink> = ir
        .edges
        .iter()
        .filter(|(_, d, _)| *d == target.id)
        .filter_map(|(s, d, _)| edge_link(nm, em, *s, *d))
        .collect();

    // Propagation en avant depuis X (les effets de X persistent même après intervention)
    let (effects, blocked) = bfs_descendants(nm, em, g, target.id);

    Some((
        target.label.clone(),
        InterventionResult {
            severed,
            effects,
            blocked,
        },
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
    counterfactual_ctx(&GraphCtx::new(ir), target_label)
}

/// M4 — cœur partageable (contexte construit une fois par requête).
pub(crate) fn counterfactual_ctx(
    ctx: &GraphCtx<'_>,
    target_label: &str,
) -> Option<(String, CounterfactualResult)> {
    let ir = ctx.ir;
    let g = &ctx.g;
    let nm = &ctx.nm;
    let em = &ctx.em;
    let target = find_best(ir, target_label)?;

    let (actual_effects, blocked) = bfs_descendants(nm, em, g, target.id);

    // Éq.12 — nœuds à exclure du graphe résiduel (en plus de X)
    let mut extra_excluded: HashSet<NodeId> = HashSet::new();

    // JointCause([X,A₂]→C) : nécessité conjointe — si X est supprimé, C disparaît
    // même si A₂ reste. Identifier les groupes joint où X est membre → exclure leur cible.
    let joint_groups = group_by_joint_id(em);
    for (gid, members) in &joint_groups {
        let x_in_group = members.iter().any(|(s, _)| *s == target.id);
        if x_in_group {
            // La cible du groupe = dst commun à tous les membres
            if let Some((_, dst)) = members.first() {
                extra_excluded.insert(*dst);
            }
        }
        let _ = gid; // utilisé seulement pour le groupement
    }

    // MediatedCause(X→C via M) : si M n'a pas d'autre source → M disparaît aussi
    for (_, _dst, edge) in &ir.edges {
        if edge.relation == RelationType::MediatedCause
            && let Some(third) = &edge.third
        {
            let mediator_id = NodeId(third.node as u32);
            let other_sources = ir
                .edges
                .iter()
                .filter(|(s, d, _e)| *d == mediator_id && *s != target.id)
                .count();
            if other_sources == 0 {
                extra_excluded.insert(mediator_id);
            }
        }
    }

    // Nœuds atteignables depuis les racines SANS X (niées non traversées)
    // et sans les nœuds dépendants.
    let mut reachable_without_x = reachable_from_roots_without(g, nm, em, target.id);
    for ex in &extra_excluded {
        reachable_without_x.remove(ex);
    }

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
            blocked,
        },
    ))
}

/// Calcule l'ensemble des nœuds atteignables depuis les VRAIES racines originelles du graphe
/// en sautant le nœud `excluded` et ses arêtes sortantes, SANS traverser d'arête niée (Ph1 [D1]).
///
/// "Vraie racine" = nœud sans AUCUNE arête entrante dans le graphe complet (variable exogène).
/// On ne repart pas des nœuds qui deviendraient des racines artificielles après suppression de X.
fn reachable_from_roots_without(
    g: &CausalGraph,
    nm: &NodeMap,
    em: &EdgeMap,
    excluded: NodeId,
) -> HashSet<NodeId> {
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

    // BFS en avant depuis les vraies racines, sans traverser X ni les coupures
    while let Some(ni) = queue.pop_front() {
        for nb in g.g.neighbors_directed(ni, Direction::Outgoing) {
            if nb == excluded_idx {
                continue;
            }
            if edge_block_motif(nm, em, g.g[ni], g.g[nb]).is_some() {
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

// ---------------------------------------------------------------------------
// Tests M4 — équivalence wrappers (build par appel) / cœurs ctx (partagé).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod ctx_tests {
    use super::*;
    use gcn_ir::{
        CausalEdge, CausalIR, CausalNode, IrMetadata, NaturalLanguage, NodeAttributes, NodeId,
        NodeOrigin, NodeType, RelationType, Scope, SourceLanguage,
    };
    use smallvec::SmallVec;

    fn tnode(id: u32, label: &str) -> CausalNode {
        CausalNode {
            id: NodeId(id),
            node_type: NodeType::Evenement,
            parent: None,
            label: label.to_string(),
            source_span: gcn_ir::SourceSpan::Synthetic,
            scope: Scope::Unknown,
            modifiers: SmallVec::new(),
            temporal_ref: gcn_ir::TemporalRef::Unresolved,
            temporal_index: Some(id as i32),
            origin: NodeOrigin::Explicit,
            attributes: NodeAttributes::default(),
            kind: None,
        }
    }

    fn tedge(src: u32, dst: u32, conf: f32, negated: bool) -> (NodeId, NodeId, CausalEdge) {
        (
            NodeId(src),
            NodeId(dst),
            CausalEdge {
                relation: RelationType::Cause,
                confidence: conf,
                temporal_gap: None,
                explicit: true,
                negated,
                marker_token: None,
                in_cycle: None,
                provenance: None,
                derivation: None,
                joint_group_id: None,
                third: None,
            },
        )
    }

    fn fixture() -> CausalIR {
        CausalIR {
            source_lang: SourceLanguage::Natural {
                lang: NaturalLanguage::Und,
            },
            source_text: "m4".to_string(),
            nodes: vec![
                tnode(0, "alpha"),
                tnode(1, "beta"),
                tnode(2, "gamma"),
                tnode(3, "delta"),
            ],
            // Chaîne alpha→beta→gamma→delta + raccourci faible + branche niée.
            edges: vec![
                tedge(0, 1, 0.9, false),
                tedge(1, 2, 0.8, false),
                tedge(2, 3, 0.7, false),
                tedge(0, 3, 0.4, false),
                tedge(1, 3, 0.9, true),
            ],
            cycles: vec![],
            unresolved: vec![],
            metadata: IrMetadata::default(),
            intent: None,
        }
    }

    fn same(a: impl std::fmt::Debug, b: impl std::fmt::Debug) {
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }

    #[test]
    fn ctx_matches_direct_on_fixture() {
        let ir = fixture();
        let ctx = GraphCtx::new(&ir);
        same(why(&ir, "delta"), why_ctx(&ctx, "delta"));
        same(what(&ir, "alpha"), what_ctx(&ctx, "alpha"));
        same(
            why_scoped_with(&ir, "delta", &DefaultLabelSim, DEFAULT_AMBIGUITY_LIMIT),
            why_scoped_ctx(&ctx, "delta", &DefaultLabelSim, DEFAULT_AMBIGUITY_LIMIT),
        );
        same(
            what_scoped_with(&ir, "alpha", &DefaultLabelSim, DEFAULT_AMBIGUITY_LIMIT),
            what_scoped_ctx(&ctx, "alpha", &DefaultLabelSim, DEFAULT_AMBIGUITY_LIMIT),
        );
        same(
            chain_topk(&ir, "alpha", "delta", 3),
            chain_topk_ctx(&ctx, "alpha", "delta", 3),
        );
        same(
            chain_with_blocked(&ir, "alpha", "delta"),
            chain_with_blocked_ctx(&ctx, "alpha", "delta"),
        );
        same(
            chain(&ir, "alpha", "delta"),
            chain_ctx(&ctx, "alpha", "delta"),
        );
        same(
            chain_temporal(&ir, "alpha", "delta"),
            chain_temporal_ctx(&ctx, "alpha", "delta"),
        );
        same(intervene(&ir, "beta"), intervene_ctx(&ctx, "beta"));
        same(
            counterfactual(&ir, "beta"),
            counterfactual_ctx(&ctx, "beta"),
        );
        same(abduct(&ir, NodeId(3)), abduct_ctx(&ctx, NodeId(3)));
        same(
            direct_links(&ir, NodeId(1)),
            direct_links_ctx(&ctx, NodeId(1)),
        );
        same(spof_all(&ir), spof_all_ctx(&ctx));
    }
}
