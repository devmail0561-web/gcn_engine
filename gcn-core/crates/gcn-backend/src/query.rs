// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use gcn_ir::{CausalCycle, CausalIR, NodeId, NodeOrigin, Scope};
use serde::{Deserialize, Serialize};

use crate::analogy::{DEFAULT_ANALOGY_MIN_SCORE, DEFAULT_ANALOGY_WEIGHTS};
use crate::error::BackendError;
use crate::pearl::{self, CausalLink, SpofScore};

fn split_pair(rest: &str, cmd: &str) -> Result<(String, String), BackendError> {
    let (a, b) = rest
        .split_once("->")
        .or_else(|| rest.split_once(','))
        .ok_or_else(|| {
            BackendError::QueryParseError(format!("{cmd} query requires: {cmd} <from> -> <to>"))
        })?;
    let a = a.trim().to_string();
    let b = b.trim().to_string();
    if a.is_empty() || b.is_empty() {
        return Err(BackendError::QueryParseError(format!(
            "{cmd}: both arguments must be non-empty"
        )));
    }
    Ok((a, b))
}

/// Options WHY/WHAT (Ph2 + Ph6b) : suffixes `AMBIGUITY_LIMIT=N` et `TOPK=N`
/// en ordre libre, optionnels. Retourne (label, limite, topk).
/// Label vide ou option mal formée → `QueryParseError` (jamais de match-all).
fn split_why_options(
    rest: &str,
    cmd: &str,
) -> Result<(String, usize, Option<usize>, LinkFilter), BackendError> {
    let mut body = rest.trim().to_string();
    let mut limit = pearl::DEFAULT_AMBIGUITY_LIMIT;
    let mut topk: Option<usize> = None;
    let mut filter = LinkFilter::default();
    while let Some((head, tail)) = body.rsplit_once(char::is_whitespace) {
        let up = tail.to_ascii_uppercase();
        if let Some(v) = up.strip_prefix("AMBIGUITY_LIMIT=") {
            limit = v.parse().map_err(|_| {
                BackendError::QueryParseError(format!(
                    "{cmd}: AMBIGUITY_LIMIT entier attendu, reçu '{tail}'"
                ))
            })?;
            body = head.trim().to_string();
        } else if let Some(v) = up.strip_prefix("TOPK=") {
            let n: usize = v.parse().map_err(|_| {
                BackendError::QueryParseError(format!(
                    "{cmd}: TOPK entier ≥ 1 attendu, reçu '{tail}'"
                ))
            })?;
            if n == 0 {
                return Err(BackendError::QueryParseError(format!(
                    "{cmd}: TOPK entier ≥ 1 attendu, reçu '{tail}'"
                )));
            }
            topk = Some(n);
            body = head.trim().to_string();
        } else if consume_filter_suffix(tail, cmd, &mut filter)? {
            body = head.trim().to_string();
        } else {
            break;
        }
    }
    if is_lone_option(&body) {
        return Err(BackendError::QueryParseError(format!(
            "{cmd}: label manquant"
        )));
    }
    Ok((body, limit, topk, filter))
}

/// Ph7 — vrai si `body` n'est qu'une option orpheline (pas un label).
fn is_lone_option(body: &str) -> bool {
    if body.is_empty() {
        return true;
    }
    let up = body.to_ascii_uppercase();
    up.starts_with("AMBIGUITY_LIMIT=")
        || up.starts_with("TOPK=")
        || up.starts_with("SCOPE=")
        || up.starts_with("AGENT=")
        || up.starts_with("METHOD=")
        || up.starts_with("SINCE=")
        || up == "ONLY_EXPLICIT"
}

/// Ph7 — valeurs valides des suffixes de filtre (erreur sinon, jamais silencieux).
fn valid_scope(v: &str) -> Result<String, BackendError> {
    let s = v.to_ascii_lowercase();
    match s.as_str() {
        "universal" | "existential" | "specific" | "null" | "unknown" | "partial" => Ok(s),
        _ => Err(BackendError::QueryParseError(format!(
            "SCOPE ∈ universal/existential/specific/null/unknown/partial attendu, reçu '{v}'"
        ))),
    }
}

fn valid_agent(v: &str) -> Result<String, BackendError> {
    let s = v.to_ascii_lowercase();
    match s.as_str() {
        "human" | "collective" | "institutional" | "natural" | "system" => Ok(s),
        _ => Err(BackendError::QueryParseError(format!(
            "AGENT ∈ human/collective/institutional/natural/system attendu, reçu '{v}'"
        ))),
    }
}

fn valid_method(v: &str) -> Result<String, BackendError> {
    let s = v.to_ascii_lowercase();
    match s.as_str() {
        "silver" | "ml" | "symbolic" => Ok(s),
        _ => Err(BackendError::QueryParseError(format!(
            "METHOD ∈ silver/ml/symbolic attendu, reçu '{v}'"
        ))),
    }
}

fn valid_since(v: &str) -> Result<String, BackendError> {
    let b = v.as_bytes();
    let ok = b.len() >= 10
        && b[0..4].iter().all(|c| c.is_ascii_digit())
        && b[4] == b'-'
        && b[5..7].iter().all(|c| c.is_ascii_digit())
        && b[7] == b'-'
        && b[8..10].iter().all(|c| c.is_ascii_digit());
    if ok {
        Ok(v.to_string())
    } else {
        Err(BackendError::QueryParseError(format!(
            "SINCE date ISO (AAAA-MM-JJ) attendue, reçue '{v}'"
        )))
    }
}

/// Ph7 — consomme UN suffixe de filtre sur `tail` (token original).
/// Retourne true si consommé. Valeurs validées (erreur sinon, jamais silencieux).
fn consume_filter_suffix(
    tail: &str,
    cmd: &str,
    filter: &mut LinkFilter,
) -> Result<bool, BackendError> {
    const SCOPE_EQ: &str = "SCOPE=";
    const AGENT_EQ: &str = "AGENT=";
    const METHOD_EQ: &str = "METHOD=";
    const SINCE_EQ: &str = "SINCE=";
    let up = tail.to_ascii_uppercase();
    if up == "ONLY_EXPLICIT" {
        filter.only_explicit = true;
        return Ok(true);
    }
    // Mot-clé WHERE : `... WHERE SCOPE=u` (le filtre suit ; seul, erreur).
    if up == "WHERE" {
        if filter == &LinkFilter::default() {
            return Err(BackendError::QueryParseError(format!(
                "{cmd}: WHERE sans filtre (SCOPE=/AGENT=/METHOD=/SINCE=/ONLY_EXPLICIT)"
            )));
        }
        return Ok(true);
    }
    if up.starts_with(SCOPE_EQ) {
        let v = &tail[SCOPE_EQ.len()..];
        filter.scope = Some(valid_scope(v).map_err(|_| {
            BackendError::QueryParseError(format!("{cmd}: SCOPE invalide ('{tail}')"))
        })?);
        return Ok(true);
    }
    if up.starts_with(AGENT_EQ) {
        let v = &tail[AGENT_EQ.len()..];
        filter.agent = Some(valid_agent(v).map_err(|_| {
            BackendError::QueryParseError(format!("{cmd}: AGENT invalide ('{tail}')"))
        })?);
        return Ok(true);
    }
    if up.starts_with(METHOD_EQ) {
        let v = &tail[METHOD_EQ.len()..];
        filter.method = Some(valid_method(v).map_err(|_| {
            BackendError::QueryParseError(format!("{cmd}: METHOD invalide ('{tail}')"))
        })?);
        return Ok(true);
    }
    if up.starts_with(SINCE_EQ) {
        let v = &tail[SINCE_EQ.len()..];
        filter.since = Some(valid_since(v.trim()).map_err(|_| {
            BackendError::QueryParseError(format!("{cmd}: SINCE invalide ('{tail}')"))
        })?);
        return Ok(true);
    }
    Ok(false)
}

/// Ph3 [D3] — chemins retournés par CHAIN sans `TOPK=` (documenté).
pub const DEFAULT_TOPK: usize = 3;

/// Ph7 — filtres de liens (7d AGENT, 7e SCOPE, 7f ONLY_EXPLICIT/METHOD/SINCE).
/// Filtre en RÉSULTAT (pas de coupure : niées/gaps/null restent en `blocked_by`).
/// Parsés par suffixes (un suffixe par phase) ; `Default` = aucun filtre.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LinkFilter {
    /// `SCOPE=<scope>` (7e) : les deux extrémités doivent matcher.
    pub scope: Option<String>,
    /// `AGENT=<type>` (7d, DO) : le nœud source doit matcher.
    pub agent: Option<String>,
    /// `ONLY_EXPLICIT` (7f) : arêtes explicites seules.
    pub only_explicit: bool,
    /// `METHOD=silver|ml|symbolic` (7f) : méthode d'extraction requise.
    pub method: Option<String>,
    /// `SINCE=<date>` (7f) : `extracted_at` ISO ≥ date.
    pub since: Option<String>,
}

/// Ph7 — applique un filtre à des liens (résultat, jamais de coupure).
/// Champs inconnus/sans provenance = exclus quand le filtre l'exige.
pub fn filter_links(ir: &CausalIR, links: Vec<CausalLink>, filter: &LinkFilter) -> Vec<CausalLink> {
    filter_links_ctx(&pearl::GraphCtx::new(ir), links, filter)
}

/// M4 — filtre sur contexte partagé (un seul build par requête).
pub(crate) fn filter_links_ctx(
    ctx: &pearl::GraphCtx<'_>,
    links: Vec<CausalLink>,
    filter: &LinkFilter,
) -> Vec<CausalLink> {
    if filter == &LinkFilter::default() {
        return links;
    }
    let nm = &ctx.nm;
    let em = &ctx.em;
    links
        .into_iter()
        .filter(|l| {
            if let Some(scope) = &filter.scope {
                let ok = |id: NodeId| nm.get(&id).is_some_and(|n| serde_name(&n.scope) == *scope);
                if !(ok(l.from_id) && ok(l.to_id)) {
                    return false;
                }
            }
            if let Some(agent) = &filter.agent {
                let ok = nm
                    .get(&l.from_id)
                    .and_then(|n| n.attributes.agent_type)
                    .is_some_and(|t| serde_name(&t) == *agent);
                if !ok {
                    return false;
                }
            }
            let edge = em.get(&(l.from_id.0, l.to_id.0));
            if filter.only_explicit && !edge.is_some_and(|e| e.explicit) {
                return false;
            }
            if let Some(method) = &filter.method {
                let ok = edge
                    .and_then(|e| e.provenance.as_ref())
                    .is_some_and(|p| method_name(p.extraction_method) == method);
                if !ok {
                    return false;
                }
            }
            if let Some(since) = &filter.since {
                let ok = edge
                    .and_then(|e| e.provenance.as_ref())
                    .is_some_and(|p| p.extracted_at.as_str() >= since.as_str());
                if !ok {
                    return false;
                }
            }
            true
        })
        .collect()
}

/// Ph7b — caveat : point d'honnêteté sur un nœud de chemin (origine
/// hypothétique, ambiguïté non levée, portée incertaine).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Caveat {
    pub node: String,
    pub field: String,
    pub candidates: Vec<String>,
    pub hint: String,
}

/// Ph7b — caveats systématiques d'un ensemble de nœuds (ordre stable, dédup).
/// Déclencheurs : origine Hypothetical, `Ambiguity` (ir.unresolved), scope
/// Unknown/Partial (Ph7e). Vide = chemin propre.
pub fn caveats_for(ir: &CausalIR, ids: &[NodeId]) -> Vec<Caveat> {
    caveats_for_ctx(&pearl::GraphCtx::new(ir), ids)
}

/// M4 — caveats sur contexte partagé (un seul build par requête).
pub(crate) fn caveats_for_ctx(ctx: &pearl::GraphCtx<'_>, ids: &[NodeId]) -> Vec<Caveat> {
    let ir = ctx.ir;
    let nm = &ctx.nm;
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for id in ids {
        if !seen.insert(*id) {
            continue;
        }
        let Some(node) = nm.get(id) else {
            continue;
        };
        if node.origin == NodeOrigin::Hypothetical {
            out.push(Caveat {
                node: node.label.clone(),
                field: "origin".to_string(),
                candidates: vec!["hypothetical".to_string()],
                hint: "chemin hypothétique : à confirmer avant d'agir (poids 0.3)".to_string(),
            });
        }
        match node.scope {
            Scope::Unknown | Scope::Partial => out.push(Caveat {
                node: node.label.clone(),
                field: "scope".to_string(),
                candidates: vec![serde_name(&node.scope)],
                hint: "portée incertaine : généralisation suspendue".to_string(),
            }),
            _ => {}
        }
        for amb in ir.unresolved.iter().filter(|a| a.node_id == *id) {
            out.push(Caveat {
                node: node.label.clone(),
                field: serde_name(&amb.field),
                candidates: amb.candidates.iter().map(|c| c.value.clone()).collect(),
                hint: amb
                    .context_hint
                    .clone()
                    .unwrap_or_else(|| "levée d'ambiguïté requise".to_string()),
            });
        }
    }
    out
}

/// Ph7b — extrémités (from+to) de liens, dans l'ordre (dédup en aval).
fn endpoint_ids(links: &[CausalLink]) -> Vec<NodeId> {
    let mut ids = Vec::with_capacity(links.len() * 2);
    for l in links {
        ids.push(l.from_id);
        ids.push(l.to_id);
    }
    ids
}

/// Options CHAIN (Ph3 + Ph7) : `TOPK=N` (défaut 3) + suffixes de filtre,
/// ordre libre. Retourne (corps, topk, filtre).
fn split_chain_options(rest: &str) -> Result<(String, usize, LinkFilter), BackendError> {
    let mut body = rest.trim().to_string();
    let mut topk = DEFAULT_TOPK;
    let mut filter = LinkFilter::default();
    while let Some((head, tail)) = body.rsplit_once(char::is_whitespace) {
        let up = tail.to_ascii_uppercase();
        if up.starts_with("TOPK=") {
            let n: usize = tail["TOPK=".len()..].parse().map_err(|_| {
                BackendError::QueryParseError(format!(
                    "CHAIN: TOPK entier ≥ 1 attendu, reçu '{tail}'"
                ))
            })?;
            if n == 0 {
                return Err(BackendError::QueryParseError(format!(
                    "CHAIN: TOPK entier ≥ 1 attendu, reçu '{tail}'"
                )));
            }
            topk = n;
            body = head.trim().to_string();
        } else if consume_filter_suffix(tail, "CHAIN", &mut filter)? {
            body = head.trim().to_string();
        } else {
            break;
        }
    }
    Ok((body, topk, filter))
}

/// Variante Ph6b : extrait en plus `HOPS=N` (profondeur de contexte, ≥ 1,
/// défaut 1 = mono-saut). Retourne (corps, poids, seuil, hops).
fn split_analogy_options_hops(rest: &str) -> Result<(String, [f32; 4], f32, u32), BackendError> {
    let mut body = rest.trim().to_string();
    let mut weights = DEFAULT_ANALOGY_WEIGHTS;
    let mut min_score = DEFAULT_ANALOGY_MIN_SCORE;
    let mut hops: u32 = 1;
    while let Some((head, tail)) = body.rsplit_once(char::is_whitespace) {
        let up = tail.to_ascii_uppercase();
        if let Some(v) = up.strip_prefix("WEIGHTS=") {
            weights = parse_weights(v)?;
            body = head.trim().to_string();
        } else if let Some(v) = up.strip_prefix("MIN_SCORE=") {
            min_score = v.parse().map_err(|_| {
                BackendError::QueryParseError(format!(
                    "ANALOGY: MIN_SCORE flottant attendu, reçu '{tail}'"
                ))
            })?;
            body = head.trim().to_string();
        } else if let Some(v) = up.strip_prefix("HOPS=") {
            hops = v.parse().map_err(|_| {
                BackendError::QueryParseError(format!(
                    "ANALOGY: HOPS entier ≥ 1 attendu, reçu '{tail}'"
                ))
            })?;
            if hops == 0 {
                return Err(BackendError::QueryParseError(format!(
                    "ANALOGY: HOPS entier ≥ 1 attendu, reçu '{tail}'"
                )));
            }
            body = head.trim().to_string();
        } else {
            break;
        }
    }
    Ok((body, weights, min_score, hops))
}

/// `WEIGHTS=a,b,c,d` : exactement 4 flottants (relation, type, confiance, degré).
fn parse_weights(v: &str) -> Result<[f32; 4], BackendError> {
    let parts: Vec<f32> = v
        .split(',')
        .map(|x| x.trim().parse())
        .collect::<Result<Vec<f32>, _>>()
        .map_err(|_| {
            BackendError::QueryParseError(format!(
                "ANALOGY: WEIGHTS=a,b,c,d (4 flottants) attendu, reçu '{v}'"
            ))
        })?;
    if parts.len() != 4 {
        return Err(BackendError::QueryParseError(format!(
            "ANALOGY: WEIGHTS=a,b,c,d (4 flottants) attendu, reçu '{v}'"
        )));
    }
    Ok([parts[0], parts[1], parts[2], parts[3]])
}

#[derive(Debug, Clone, PartialEq)]
pub enum Query {
    /// Ph2 [D2] : `ambiguity_limit` = max de matchs FLOUS (défaut 5) ;
    /// les exacts passent toujours, l'excès flou → `Err(Ambiguous)`.
    /// Ph6b : `topk` = N meilleurs liens (None = tous).
    /// Ph7 : `filter` = filtres de liens (défaut : aucun).
    Why {
        label: String,
        ambiguity_limit: usize,
        topk: Option<usize>,
        filter: LinkFilter,
    },
    /// Ph2 [D2] : voir `Why`. Ph6b : voir `Why.topk`. Ph7 : voir `Why.filter`.
    What {
        label: String,
        ambiguity_limit: usize,
        topk: Option<usize>,
        filter: LinkFilter,
    },
    /// Ph3 [D3] : `topk` = chemins retournés, confiance-max (défaut 3, ≥ 1).
    /// Ph7 : `filter` = filtres de liens (défaut : aucun).
    Chain {
        from: String,
        to: String,
        topk: usize,
        filter: LinkFilter,
    },
    Cycles,
    Gaps,
    /// Pearl niveau 2 — `DO <nœud>` : intervention do-calculus.
    /// Ph7 : `filter` = filtres de liens (défaut : aucun ; `AGENT=` en 7d).
    Intervene {
        label: String,
        filter: LinkFilter,
    },
    /// Pearl niveau 3 — `COUNTERFACTUAL <nœud>` : raisonnement contrefactuel
    Counterfactual(String),
    /// Abductif — `EXPLAIN <effet>` : hypothèses causales les plus plausibles pour E.
    /// Ph6b : `EXPLAIN <effet> CONTRAST <alternative>` = causes de E hors F.
    Explain {
        effect: String,
        contrast: Option<String>,
    },
    /// Analogie — `ANALOGY <from> -> <to>` : patterns causaux similaires au patron (from→to).
    /// Ph3 [D3] : `weights` = pondération (relation, type, confiance, degré),
    /// `min_score` = seuil (défauts documentés).
    /// Ph6b : `hops` = profondeur de contexte (1 = mono-saut historique).
    Analogy {
        from: String,
        to: String,
        weights: [f32; 4],
        min_score: f32,
        hops: u32,
    },
    /// Ph6b — `ASK <label>` : fiche d'un nœud (type + causes/effets directs).
    Ask {
        label: String,
    },
    /// Ph6b — `WHY-NOT <from> -> <to>` : pourquoi aucun chemin causal ?
    /// Réutilise `blocked_by` (coupure niée) + proches (vraiment aucun lien).
    WhyNot {
        from: String,
        to: String,
    },
    /// Multi-échelle — `ZOOM_IN <label>` : enfants directs du nœud dans la hiérarchie
    ZoomIn(String),
    /// Multi-échelle — `ZOOM_OUT <label>` : parent du nœud dans la hiérarchie
    ZoomOut(String),
    /// Multi-échelle — `AGGREGATE <label>` : vue agrégée d'un nœud parent et ses enfants
    Aggregate(String),
    /// Adversarial — `CENTRALITY <label>` : centralité causale pondérée confiance
    Centrality(String),
    /// Adversarial — `SPOF?` : nœuds dont la suppression coupe le plus de paires causales
    Spof,
    /// Ph7b — `DOUBTS` : liste brute des non-résolus (`ir.unresolved`).
    Doubts,
    /// Ph7f — `PROVENANCE <label>` : fiche complète des arêtes incidentes
    /// (qui, quand, comment, span texte).
    Provenance(String),
    /// Normatif — `DIFF <from> -> <to>` : écart entre confiance observée et attente normative (1.0)
    NormDiff(String, String),
    /// Méta — `DENSITY` ou `DENSITY <label>` : densité globale ou locale du graphe
    Density(Option<String>),
    /// Méta — `COVERAGE <label>` : couverture causale autour d'un concept
    Coverage(String),
    /// Méta — `RELIABILITY <label>` : fiabilité des réponses causales pour un concept
    Reliability(String),
    /// Temporal — `CHAIN_T <a> -> <b>` : chemin causal respectant l'ordre temporel
    ChainT(String, String),
    /// Temporal — `BEFORE? <a>, <b>` : A précède-t-il B (temporal_index + chemin orienté) ?
    Before(String, String),
    /// Temporal — `DELAY <a> -> <b>` : délai estimé A→B sur le meilleur chemin
    Delay(String, String),
}

impl Query {
    pub fn parse(input: &str) -> Result<Self, BackendError> {
        let trimmed = input.trim().trim_end_matches('?').trim();
        // Normalize only the keyword (first word) to uppercase — label case is preserved.
        let normalized = match trimmed.find(' ') {
            Some(pos) => format!("{}{}", trimmed[..pos].to_ascii_uppercase(), &trimmed[pos..]),
            None => trimmed.to_ascii_uppercase(),
        };
        let s = normalized.as_str();

        if let Some(rest) = s.strip_prefix("WHY ") {
            let (label, ambiguity_limit, topk, filter) = split_why_options(rest, "WHY")?;
            return Ok(Query::Why {
                label,
                ambiguity_limit,
                topk,
                filter,
            });
        }
        if let Some(rest) = s.strip_prefix("WHAT ") {
            let (label, ambiguity_limit, topk, filter) = split_why_options(rest, "WHAT")?;
            return Ok(Query::What {
                label,
                ambiguity_limit,
                topk,
                filter,
            });
        }
        if let Some(rest) = s.strip_prefix("CHAIN ") {
            let (body, topk, filter) = split_chain_options(rest)?;
            let (from, to) = split_pair(&body, "CHAIN")?;
            return Ok(Query::Chain {
                from,
                to,
                topk,
                filter,
            });
        }
        if s == "CYCLES" {
            return Ok(Query::Cycles);
        }
        if s == "GAPS" {
            return Ok(Query::Gaps);
        }
        if let Some(rest) = s.strip_prefix("DO ") {
            // Ph7 : suffixes de filtre (`AGENT=` en 7d), ordre libre.
            let mut body = rest.trim().to_string();
            let mut filter = LinkFilter::default();
            while let Some((head, tail)) = body.rsplit_once(char::is_whitespace) {
                if consume_filter_suffix(tail, "DO", &mut filter)? {
                    body = head.trim().to_string();
                } else {
                    break;
                }
            }
            if is_lone_option(&body) {
                return Err(BackendError::QueryParseError(
                    "DO: label manquant".to_string(),
                ));
            }
            return Ok(Query::Intervene {
                label: body,
                filter,
            });
        }
        if let Some(label) = s.strip_prefix("COUNTERFACTUAL ") {
            return Ok(Query::Counterfactual(label.trim().to_string()));
        }
        if let Some(rest) = s.strip_prefix("EXPLAIN ") {
            // Ph6b : `EXPLAIN <effet> CONTRAST <alternative>` (insensible à la casse).
            let up = rest.to_ascii_uppercase();
            if let Some(pos) = up.find(" CONTRAST ") {
                let effect = rest[..pos].trim().to_string();
                let contrast = rest[pos + " CONTRAST ".len()..].trim().to_string();
                if effect.is_empty() || contrast.is_empty() {
                    return Err(BackendError::QueryParseError(
                        "EXPLAIN: effet et alternative requis (EXPLAIN <effet> CONTRAST <alternative>)"
                            .to_string(),
                    ));
                }
                return Ok(Query::Explain {
                    effect,
                    contrast: Some(contrast),
                });
            }
            let effect = rest.trim().to_string();
            if effect.is_empty() {
                return Err(BackendError::QueryParseError(
                    "EXPLAIN: effet manquant".to_string(),
                ));
            }
            return Ok(Query::Explain {
                effect,
                contrast: None,
            });
        }
        if let Some(rest) = s.strip_prefix("ANALOGY ") {
            let (body, weights, min_score, hops) = split_analogy_options_hops(rest)?;
            let (from, to) = split_pair(&body, "ANALOGY")?;
            return Ok(Query::Analogy {
                from,
                to,
                weights,
                min_score,
                hops,
            });
        }
        if let Some(label) = s.strip_prefix("ASK ") {
            // Ph6b : fiche nœud (label requis, pas de match-all silencieux).
            let label = label.trim().to_string();
            if label.is_empty() {
                return Err(BackendError::QueryParseError(
                    "ASK: label manquant".to_string(),
                ));
            }
            return Ok(Query::Ask { label });
        }
        if let Some(rest) = s.strip_prefix("WHY-NOT ") {
            // Ph6b : pourquoi aucun chemin causal ? (réutilise blocked_by).
            let (from, to) = split_pair(rest, "WHY-NOT")?;
            return Ok(Query::WhyNot { from, to });
        }
        if let Some(label) = s.strip_prefix("ZOOM_IN ") {
            return Ok(Query::ZoomIn(label.trim().to_string()));
        }
        if let Some(label) = s.strip_prefix("ZOOM_OUT ") {
            return Ok(Query::ZoomOut(label.trim().to_string()));
        }
        if let Some(label) = s.strip_prefix("AGGREGATE ") {
            return Ok(Query::Aggregate(label.trim().to_string()));
        }
        if let Some(label) = s.strip_prefix("CENTRALITY ") {
            return Ok(Query::Centrality(label.trim().to_string()));
        }
        if s == "SPOF" {
            return Ok(Query::Spof);
        }
        if s == "DOUBTS" {
            return Ok(Query::Doubts);
        }
        if let Some(label) = s.strip_prefix("PROVENANCE ") {
            // Ph7f : fiche provenance (label requis).
            let label = label.trim().to_string();
            if label.is_empty() {
                return Err(BackendError::QueryParseError(
                    "PROVENANCE: label manquant".to_string(),
                ));
            }
            return Ok(Query::Provenance(label));
        }
        if let Some(rest) = s.strip_prefix("DIFF ") {
            let (from, to) = split_pair(rest, "DIFF")?;
            return Ok(Query::NormDiff(from, to));
        }
        if s == "DENSITY" {
            return Ok(Query::Density(None));
        }
        if let Some(label) = s.strip_prefix("DENSITY ") {
            return Ok(Query::Density(Some(label.trim().to_string())));
        }
        if let Some(label) = s.strip_prefix("COVERAGE ") {
            return Ok(Query::Coverage(label.trim().to_string()));
        }
        if let Some(label) = s.strip_prefix("RELIABILITY ") {
            return Ok(Query::Reliability(label.trim().to_string()));
        }
        if let Some(rest) = s.strip_prefix("CHAIN_T ") {
            let (from, to) = split_pair(rest, "CHAIN_T")?;
            return Ok(Query::ChainT(from, to));
        }
        if let Some(rest) = s.strip_prefix("BEFORE? ") {
            let (a, b) = split_pair(rest, "BEFORE?")?;
            return Ok(Query::Before(a, b));
        }
        if let Some(rest) = s.strip_prefix("DELAY ") {
            let (from, to) = split_pair(rest, "DELAY")?;
            return Ok(Query::Delay(from, to));
        }

        let preview = {
            let end = input
                .char_indices()
                .nth(200)
                .map(|(i, _)| i)
                .unwrap_or(input.len());
            &input[..end]
        };
        Err(BackendError::QueryParseError(format!(
            "unknown query '{}'. Valid: WHY, WHAT, CHAIN, CYCLES, GAPS, DO, COUNTERFACTUAL, EXPLAIN, ANALOGY, ASK, WHY-NOT, CENTRALITY, SPOF, DOUBTS, DIFF, DENSITY, COVERAGE, RELIABILITY, CHAIN_T, BEFORE?, DELAY, ZOOM_IN, ZOOM_OUT, AGGREGATE",
            preview
        )))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum QueryResult {
    Causes {
        target: String,
        /// Liens causaux de TOUS les nœuds matchés (concaténés).
        /// P0-1 : `execute()` ne jetait que la première correspondance —
        /// désormais `links` couvre tous les matchs, `matched_targets` les nomme.
        links: Vec<LinkDto>,
        /// Labels de tous les nœuds matchés par la requête (le 1er = `target`).
        #[serde(default)]
        matched_targets: Vec<String>,
        /// Ph1 [D1] : arêtes niées qui auraient porté une propagation
        /// (coupées, listées au lieu d'être traversées).
        #[serde(default)]
        blocked_by: Vec<LinkDto>,
        /// Ph2 [D2] : (label, score) des nœuds retenus — 0 = exact, >0 = flou.
        #[serde(default)]
        match_scores: Vec<(String, u8)>,
        /// Ph2 [D2] : vrai si réponse sur flous (warning ; voir `match_scores`).
        #[serde(default)]
        fuzzy: bool,
        /// Ph6b : liens avant limite TOPK (== `links.len()` sans TOPK).
        #[serde(default)]
        total: usize,
        /// Ph7b : caveats systématiques (hypothétiques, ambiguïtés, portées).
        #[serde(default)]
        caveats: Vec<Caveat>,
    },
    Effects {
        source: String,
        /// Effets de TOUS les nœuds matchés (concaténés, cf. P0-1).
        links: Vec<LinkDto>,
        /// Labels de tous les nœuds matchés par la requête (le 1er = `source`).
        #[serde(default)]
        matched_sources: Vec<String>,
        /// Ph1 [D1] : voir `Causes::blocked_by`.
        #[serde(default)]
        blocked_by: Vec<LinkDto>,
        /// Ph2 [D2] : voir `Causes::match_scores`.
        #[serde(default)]
        match_scores: Vec<(String, u8)>,
        /// Ph2 [D2] : voir `Causes::fuzzy`.
        #[serde(default)]
        fuzzy: bool,
        /// Ph6b : voir `Causes::total`.
        #[serde(default)]
        total: usize,
        /// Ph7b : voir `Causes::caveats`.
        #[serde(default)]
        caveats: Vec<Caveat>,
    },
    Path {
        from: String,
        to: String,
        found: bool,
        links: Vec<LinkDto>,
        /// Ph1 [D1] : `found:false` + `blocked_by` rempli = chemin existant
        /// mais coupé par négation ; `found:false` + vide = vraiment aucun chemin.
        #[serde(default)]
        blocked_by: Vec<LinkDto>,
        /// Ph3 [D3] : produit des confidences du meilleur chemin (0.0 si !found).
        #[serde(default)]
        path_confidence: f32,
        /// Ph3 [D3] : chemins classés (rang 1 = `links`), présent ssi TOPK > 1.
        #[serde(default)]
        ranked_paths: Vec<RankedPathDto>,
        /// Ph7b : voir `Causes::caveats`.
        #[serde(default)]
        caveats: Vec<Caveat>,
    },
    CycleList {
        count: usize,
        cycles: Vec<CycleDto>,
    },
    GapList {
        gap_edges: Vec<GapDto>,
        unresolved_nodes: usize,
    },
    /// Pearl niveau 2 : résultat d'une intervention do-calculus.
    Intervention {
        target: String,
        /// Nombre d'arêtes entrantes coupées par l'intervention.
        severed_count: usize,
        /// Arêtes entrantes supprimées (causes naturelles de `target` court-circuitées).
        severed: Vec<LinkDto>,
        /// Effets en aval de `target` après l'intervention.
        effects: Vec<LinkDto>,
        /// Ph1 [D1] : niées rencontrées en aval (voir `Causes::blocked_by`).
        #[serde(default)]
        blocked_by: Vec<LinkDto>,
        /// Ph7b : voir `Causes::caveats`.
        #[serde(default)]
        caveats: Vec<Caveat>,
        /// Ph7d : avertissements FIGÉS (`irreversible` si cible irréversible —
        /// effets calculés quand même, DO informatif).
        #[serde(default)]
        warnings: Vec<String>,
    },
    /// Pearl niveau 3 : résultat d'une requête contrefactuelle.
    CounterfactualDiff {
        target: String,
        /// Tous les effets réels de `target` dans le monde actuel.
        actual_effects: Vec<LinkDto>,
        /// Effets counterfactuellement dépendants de `target` (n'auraient pas eu lieu sans X).
        unique_effects: Vec<String>,
        /// Ph1 [D1] : niées rencontrées depuis `target` (voir `Causes::blocked_by`).
        #[serde(default)]
        blocked_by: Vec<LinkDto>,
        /// Ph7b : voir `Causes::caveats`.
        #[serde(default)]
        caveats: Vec<Caveat>,
    },
    /// Analogie — patterns causaux similaires au patron (from→to).
    AnalogyReport {
        pattern_from: String,
        pattern_to: String,
        /// Matches triés par score décroissant. Vide si patron introuvable ou graphe sans analogue.
        matches: Vec<AnalogyMatchDto>,
    },
    /// Multi-échelle — enfants directs dans la hiérarchie.
    HierarchyZoomIn {
        parent_label: String,
        children: Vec<NodeSummaryDto>,
    },
    /// Multi-échelle — parent dans la hiérarchie.
    HierarchyZoomOut {
        child_label: String,
        /// None si le nœud est déjà à la racine.
        parent_label: Option<String>,
    },
    /// Multi-échelle — vue agrégée d'un parent et ses enfants.
    HierarchyAggregate {
        parent_label: String,
        child_count: usize,
        /// Toutes les arêtes sortantes des enfants (union).
        outgoing_edges: Vec<LinkDto>,
        /// Confiance moyenne sur ces arêtes.
        mean_confidence: f32,
    },
    /// Adversarial — centralité causale d'un nœud.
    CentralityReport {
        label: String,
        degree_in: usize,
        degree_out: usize,
        /// Somme des confidences sur arêtes entrantes.
        weighted_in: f32,
        /// Somme des confidences sur arêtes sortantes.
        weighted_out: f32,
        /// Centralité composite : (weighted_in + weighted_out) / (degree_in + degree_out). 0 si isolé.
        centrality_score: f32,
    },
    /// Adversarial — SPOF : nœuds dont la suppression coupe le plus de paires.
    SpofReport {
        /// Nombre total de paires (s,d) avec chemin dans le graphe complet.
        total_pairs: usize,
        nodes: Vec<SpofNodeDto>,
    },
    /// Ph7b — DOUBTS : liste brute des non-résolus.
    Doubts {
        count: usize,
        doubts: Vec<Caveat>,
    },
    /// Ph7f — PROVENANCE : fiche des arêtes incidentes (ancrage confiance).
    ProvenanceReport {
        label: String,
        entries: Vec<ProvenanceEntryDto>,
    },
    /// Normatif — écart observé vs attente normative (1.0) sur le chemin A→B.
    NormativeDiff {
        from: String,
        to: String,
        /// Le chemin A→B existe-t-il dans le graphe ?
        path_exists: bool,
        /// Nombre d'arêtes sur le chemin.
        n_edges_on_path: usize,
        /// Confiance minimale observée sur le chemin (proxy de la force causale).
        observed_confidence: f32,
        /// Écart absolu : 1.0 − observed_confidence.
        gap: f32,
        /// Taux d'écart en % : gap × 100.
        gap_rate_pct: f32,
        /// Liste des relations sur le chemin avec leur confiance.
        path_links: Vec<LinkDto>,
    },
    /// Méta — densité du graphe ou d'un voisinage local.
    DensityReport {
        /// Densité globale : n_edges / (n_nodes × (n_nodes-1)). 0 si ≤ 1 nœud.
        global_density: f32,
        n_nodes: usize,
        n_edges: usize,
        /// Présent si DENSITY <label> : label du nœud trouvé.
        node_label: Option<String>,
        /// Degré total (entrant + sortant) du nœud si label fourni.
        local_degree: Option<usize>,
        /// Densité locale : local_degree / (2 × (n_nodes-1)). None si global.
        local_density: Option<f32>,
    },
    /// Méta — couverture causale d'un concept.
    CoverageReport {
        label: String,
        /// Nombre d'arêtes connectées à ce nœud (entrant + sortant).
        degree: usize,
        /// Nombre d'arêtes avec provenance renseignée.
        n_with_provenance: usize,
        /// coverage_score = degree / n_edges (fraction du graphe couverte). 0 si aucune arête.
        coverage_score: f32,
        /// provenance_ratio = n_with_provenance / degree. 1.0 si degree=0.
        provenance_ratio: f32,
    },
    /// Méta — fiabilité des réponses causales pour un concept.
    ReliabilityReport {
        label: String,
        /// Confiance minimale sur les arêtes incidentes.
        min_confidence: f32,
        /// Confiance moyenne sur les arêtes incidentes.
        mean_confidence: f32,
        /// Fraction d'arêtes incidentes avec provenance. 1.0 si aucune arête.
        provenance_ratio: f32,
        /// reliability_score = mean_confidence × provenance_ratio.
        reliability_score: f32,
    },
    /// Abductif — hypothèses causales pour un effet observé, triées par score.
    Abduction {
        effect: String,
        hypotheses: Vec<AbductionHypothesisDto>,
        /// Ph6b : contraste vs une alternative (None = abduction simple).
        #[serde(default)]
        contrast: Option<AbductionContrastDto>,
        /// Ph7d : hypothèses scindées par type d'agent (None = non renseigné).
        #[serde(default)]
        by_agent: Vec<AgentGroupDto>,
    },
    /// Ph6b — fiche nœud (ASK) : type + causes/effets directs 1 saut.
    AskReport {
        label: String,
        node_type: String,
        causes: Vec<LinkDto>,
        effects: Vec<LinkDto>,
    },
    /// Ph6b — WHY-NOT : pourquoi aucun chemin causal (blocked_by ou proches).
    WhyNot {
        from: String,
        to: String,
        /// `chemin_existe` | `coupure_negation` | `aucun_lien`.
        verdict: String,
        /// Chemin existant ssi `chemin_existe` (sinon vide).
        links: Vec<LinkDto>,
        /// Niées de coupure ssi `coupure_negation` (WHY-NOT = les bloqueurs).
        #[serde(default)]
        blocked_by: Vec<LinkDto>,
        /// Voisins 1 saut ssi `aucun_lien` (rebond).
        #[serde(default)]
        proches: Vec<String>,
    },
    /// Temporal — chemin causal respectant l'ordre temporal_index.
    TemporalPath {
        from: String,
        to: String,
        found: bool,
        temporally_ordered: bool,
        links: Vec<TemporalLinkDto>,
    },
    /// Temporal — ordre relatif de deux nœuds.
    TemporalOrder {
        a: String,
        b: String,
        /// true si a.temporal_index < b.temporal_index ET chemin orienté a→b existe.
        a_before_b: bool,
        a_index: Option<i32>,
        b_index: Option<i32>,
        path_exists: bool,
    },
    /// Temporal — délai estimé A→B.
    TemporalDelay {
        from: String,
        to: String,
        found: bool,
        /// Somme des TemporalGap.min sur le chemin (unités = unités du domaine).
        gap_min_sum: Option<i32>,
        gap_max_sum: Option<i32>,
        /// Différence de temporal_index comme proxy si pas de TemporalGap.
        index_delta: Option<i32>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkDto {
    pub from: String,
    pub to: String,
    pub relation: String,
    pub confidence: f32,
    pub negated: bool,
    /// C4 fix — joint_group_id pour co-nécessité JointCause/JointPrevent
    #[serde(skip_serializing_if = "Option::is_none")]
    pub joint_group_id: Option<String>,
    /// C4 fix — third_node pour les relations ternaires
    #[serde(skip_serializing_if = "Option::is_none")]
    pub third_node_label: Option<String>,
    /// Ph6b — dérivation D3 systématique : (from, to) sources ayant inféré
    /// ce lien (`Derivation.source_edges` résolues en labels).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived_from: Option<Vec<(String, String)>>,
    /// Ph7 — motif de coupure pour `blocked_by` (None = lien traversé).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub motif: Option<String>,
    /// Ph7a — effet de cycle : damped/amplifying (None sinon).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cycle_effect: Option<String>,
    /// Ph7b — origine du nœud source (explicit/inferred/hypothetical).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<String>,
    /// Ph7f — arête explicite (False si clé absente : conservateur).
    #[serde(default, skip_serializing_if = "is_false")]
    pub explicit: bool,
    /// Ph7f — méthode d'extraction (silver/ml/symbolic).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extraction_method: Option<String>,
    /// Ph7f — version du modèle extracteur.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    /// Ph7f — span texte/code (texte extrait si possible, sinon coordonnées).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<String>,
    /// Ph7f — nature du span (text/code/synthetic).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Ph7e — scope des extrémités (universal/...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Ph7d — agent (nœud source) / patient (nœud cible).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patient: Option<String>,
    /// Ph7d — type d'agent du nœud source (human/...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_type: Option<String>,
    /// Ph7e — lien témoin (existential/specific : généralisation interdite).
    #[serde(default, skip_serializing_if = "is_false")]
    pub witness_only: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleDto {
    pub id: u32,
    pub kind: String,
    pub nodes: Vec<String>,
    /// Ph7a — effet du cycle : damped/amplifying/oscillating.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect: Option<String>,
}

/// Ph3 [D3] — un chemin classé de CHAIN TOPK.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedPathDto {
    /// Rang 1-based (1 = meilleur = `Path.links`).
    pub rank: usize,
    /// Produit des confidences du chemin.
    pub path_confidence: f32,
    pub links: Vec<LinkDto>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapDto {
    pub from: String,
    pub to: String,
    pub relation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalogyMatchDto {
    pub from_label: String,
    pub to_label: String,
    pub relation: String,
    pub confidence: f32,
    pub similarity_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSummaryDto {
    pub label: String,
    pub node_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpofNodeDto {
    pub label: String,
    /// Paires (s,d) déconnectées par la suppression de ce nœud.
    pub paths_cut: usize,
    /// spof_score = paths_cut / total_pairs. 0 si total_pairs=0.
    pub spof_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbductionHypothesisDto {
    /// Label du nœud hypothèse (cause candidate).
    pub label: String,
    /// Relation de ce nœud vers son successeur immédiat dans le chemin vers l'effet.
    pub relation: String,
    /// Confiance de l'arête immédiate de ce nœud vers son successeur dans le chemin.
    pub edge_confidence: f32,
    /// Score = min_conf_on_path / (1 + depth).
    pub score: f32,
    /// Nombre de sauts depuis cette hypothèse jusqu'à l'effet.
    pub depth: usize,
}

/// Ph6b — contraste abductif : causes de l'effet hors alternative.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbductionContrastDto {
    /// Label de l'alternative comparée.
    pub alternative: String,
    /// Hypothèses de l'effet absentes de l'alternative (distinctives).
    pub distinctive: Vec<AbductionHypothesisDto>,
    /// Labels causes des deux (contexte partagé).
    pub shared: Vec<String>,
}

/// Ph7d — groupe d'hypothèses par type d'agent (EXPLAIN scindé).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentGroupDto {
    /// Type d'agent (human/...), None = non renseigné sur les nœuds.
    pub agent_type: Option<String>,
    pub hypotheses: Vec<AbductionHypothesisDto>,
}

/// Ph7f — une arête incidente et sa provenance (None = non renseignée).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceEntryDto {
    pub from: String,
    pub to: String,
    pub relation: String,
    pub explicit: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extraction_method: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extracted_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalLinkDto {
    pub from: String,
    pub to: String,
    pub relation: String,
    pub confidence: f32,
    pub negated: bool,
    pub gap_min: Option<i32>,
    pub gap_max: Option<i32>,
    pub index_delta: Option<i32>,
}

/// S1 — réponse cadrée WHY/WHAT (corps commun extrait d'`execute`).
struct ScopedAnswer {
    names: Vec<String>,
    first: String,
    links: Vec<LinkDto>,
    blocked_by: Vec<LinkDto>,
    match_scores: Vec<(String, u8)>,
    fuzzy: bool,
    total: usize,
    caveats: Vec<Caveat>,
}

/// S1 — WHY (`reverse`) / WHAT sur contexte partagé : exacts (tous) > flous
/// (≤ limite) > `Err(Ambiguous)` [D2] ; niées coupées [D1] ; filtre résultat
/// (jamais de coupure) puis TOPK [Ph7/6b] ; caveats [Ph7b].
fn scoped_why_what(
    ctx: &pearl::GraphCtx<'_>,
    label: &str,
    ambiguity_limit: usize,
    topk: Option<usize>,
    filter: &LinkFilter,
    reverse: bool,
) -> Result<ScopedAnswer, BackendError> {
    let scoped = if reverse {
        pearl::why_scoped_ctx(ctx, label, &pearl::DefaultLabelSim, ambiguity_limit)
    } else {
        pearl::what_scoped_ctx(ctx, label, &pearl::DefaultLabelSim, ambiguity_limit)
    }
    .map_err(|candidates| BackendError::Ambiguous { candidates })?;
    if scoped.matched.is_empty() {
        return Err(BackendError::NodeNotFound(label.to_string()));
    }
    // P0-1 : agréger TOUS les matchs au lieu du premier seul.
    let names: Vec<String> = scoped.matched.iter().map(|(t, _, _)| t.clone()).collect();
    let first = names
        .first()
        .cloned()
        .ok_or_else(|| BackendError::NodeNotFound(label.to_string()))?;
    let raw: Vec<CausalLink> = scoped
        .matched
        .iter()
        .flat_map(|(_, ls, _)| ls.iter().cloned())
        .collect();
    // Ph1 [D1] : niées coupées sur tous les matchs (triées, cf. pearl).
    let blocked_by: Vec<LinkDto> = scoped
        .matched
        .iter()
        .flat_map(|(_, _, b)| b.iter().map(|l| link_to_dto(ctx, l)))
        .collect();
    // Ph7 : filtre résultat (jamais de coupure), puis Ph6b TOPK.
    let kept = filter_links_ctx(ctx, raw, filter);
    let total = kept.len();
    let links: Vec<LinkDto> = kept
        .iter()
        .take(topk.unwrap_or(usize::MAX))
        .map(|l| link_to_dto(ctx, l))
        .collect();
    // Ph7b : caveats des chemins retenus.
    let caveats = caveats_for_ctx(ctx, &endpoint_ids(&kept));
    Ok(ScopedAnswer {
        names,
        first,
        links,
        blocked_by,
        match_scores: scoped.match_scores,
        fuzzy: scoped.fuzzy,
        total,
        caveats,
    })
}

pub fn execute(query: &Query, ir: &CausalIR) -> Result<QueryResult, BackendError> {
    // Ph10-M4 : UN SEUL calcul de graphe par requête — contexte partagé
    // (petgraph + index) passé à toutes les fonctions d'analyse ci-dessous.
    let ctx = pearl::GraphCtx::new(ir);
    match query {
        Query::Why {
            label,
            ambiguity_limit,
            topk,
            filter,
        } => {
            let a = scoped_why_what(&ctx, label, *ambiguity_limit, *topk, filter, true)?;
            Ok(QueryResult::Causes {
                target: a.first,
                links: a.links,
                matched_targets: a.names,
                blocked_by: a.blocked_by,
                match_scores: a.match_scores,
                fuzzy: a.fuzzy,
                total: a.total,
                caveats: a.caveats,
            })
        }

        Query::What {
            label,
            ambiguity_limit,
            topk,
            filter,
        } => {
            let a = scoped_why_what(&ctx, label, *ambiguity_limit, *topk, filter, false)?;
            Ok(QueryResult::Effects {
                source: a.first,
                links: a.links,
                matched_sources: a.names,
                blocked_by: a.blocked_by,
                match_scores: a.match_scores,
                fuzzy: a.fuzzy,
                total: a.total,
                caveats: a.caveats,
            })
        }

        Query::Chain {
            from,
            to,
            topk,
            filter,
        } => {
            // Distingue nœud introuvable (NodeNotFound) de l'absence de chemin (found:false).
            if pearl::find_best(ctx.ir, from).is_none() {
                return Err(BackendError::NodeNotFound(from.clone()));
            }
            if pearl::find_best(ctx.ir, to).is_none() {
                return Err(BackendError::NodeNotFound(to.clone()));
            }
            // Ph3 [D3] : jusqu'à `topk` chemins confiance-max (Dijkstra + Yen).
            let (paths, blocked) = pearl::chain_topk_ctx(&ctx, from, to, *topk);
            // Ph7 : filtre résultat — chemin écarté si un lien ne passe pas
            // (le chemin n'est valide que d'un bloc ; pas de coupure).
            let paths: Vec<pearl::ScoredPath> = paths
                .into_iter()
                .filter(|p| filter_links_ctx(&ctx, p.links.clone(), filter).len() == p.links.len())
                .collect();
            let found = !paths.is_empty();
            let best = paths.first();
            // Ph7b : caveats du meilleur chemin retenu.
            let caveats = best
                .map(|p| {
                    caveats_for(
                        ir,
                        &p.links
                            .iter()
                            .flat_map(|l| [l.from_id, l.to_id])
                            .collect::<Vec<_>>(),
                    )
                })
                .unwrap_or_default();
            Ok(QueryResult::Path {
                from: from.clone(),
                to: to.clone(),
                found,
                links: best
                    .map(|p| p.links.iter().map(|l| link_to_dto(&ctx, l)).collect())
                    .unwrap_or_default(),
                // Ph1 [D1] : rempli ssi coupure par négation (vide = vraiment aucun chemin).
                blocked_by: blocked.iter().map(|l| link_to_dto(&ctx, l)).collect(),
                path_confidence: best.map(|p| p.confidence).unwrap_or(0.0),
                // Chemins classés ssi TOPK > 1 (rang 1 = `links`).
                ranked_paths: if *topk > 1 {
                    paths
                        .iter()
                        .enumerate()
                        .map(|(i, p)| RankedPathDto {
                            rank: i + 1,
                            path_confidence: p.confidence,
                            links: p.links.iter().map(|l| link_to_dto(&ctx, l)).collect(),
                        })
                        .collect()
                } else {
                    vec![]
                },
                caveats,
            })
        }

        Query::Cycles => Ok(QueryResult::CycleList {
            count: ir.cycles.len(),
            cycles: ir.cycles.iter().map(cycle_to_dto(ir)).collect(),
        }),

        Query::Gaps => {
            let gap_edges: Vec<GapDto> = ir
                .edges
                .iter()
                .filter(|(_, _, e)| e.temporal_gap.is_some())
                .map(|(s, d, e)| GapDto {
                    from: ctx.node_label(*s),
                    to: ctx.node_label(*d),
                    relation: serde_name(&e.relation),
                })
                .collect();
            Ok(QueryResult::GapList {
                gap_edges,
                unresolved_nodes: ir.unresolved.len(),
            })
        }

        Query::Intervene { label, filter } => {
            let (target, result) = pearl::intervene_ctx(&ctx, label)
                .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
            // Ph7 : filtre résultat (severed + effets).
            let severed = filter_links_ctx(&ctx, result.severed, filter);
            let effects = filter_links_ctx(&ctx, result.effects, filter);
            // Ph7b : caveats de la cible + effets retenus.
            let mut ids = vec![
                pearl::find_best(ctx.ir, label)
                    .map(|n| n.id)
                    .unwrap_or(NodeId(0)),
            ];
            ids.extend(endpoint_ids(&effects));
            let caveats = caveats_for_ctx(&ctx, &ids);
            // Ph7d : cible irréversible → warning FIGÉ (effets calculés quand même).
            let warnings = match pearl::find_best(ctx.ir, label) {
                Some(n) if n.attributes.reversible == Some(false) => {
                    vec!["irreversible".to_string()]
                }
                _ => vec![],
            };
            Ok(QueryResult::Intervention {
                target,
                severed_count: severed.len(),
                severed: severed.iter().map(|l| link_to_dto(&ctx, l)).collect(),
                effects: effects.iter().map(|l| link_to_dto(&ctx, l)).collect(),
                // Ph1 [D1] : niées rencontrées en aval.
                blocked_by: result
                    .blocked
                    .iter()
                    .map(|l| link_to_dto(&ctx, l))
                    .collect(),
                caveats,
                warnings,
            })
        }

        Query::Counterfactual(label) => {
            let (target, result) = pearl::counterfactual_ctx(&ctx, label)
                .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
            // Ph7 : filtre résultat (pas de filtre requête sur CF : monde actuel brut).
            let actual = filter_links_ctx(&ctx, result.actual_effects, &LinkFilter::default());
            let gardes: std::collections::HashSet<String> =
                actual.iter().map(|l| l.to_label.clone()).collect();
            let unique_effects: Vec<String> = result
                .unique_effects
                .into_iter()
                .filter(|u| gardes.contains(u))
                .collect();
            // Ph7b : caveats de la cible + effets retenus.
            let mut ids = vec![
                pearl::find_best(ctx.ir, label)
                    .map(|n| n.id)
                    .unwrap_or(NodeId(0)),
            ];
            ids.extend(endpoint_ids(&actual));
            let caveats = caveats_for_ctx(&ctx, &ids);
            Ok(QueryResult::CounterfactualDiff {
                target,
                actual_effects: actual.iter().map(|l| link_to_dto(&ctx, l)).collect(),
                unique_effects,
                // Ph1 [D1] : niées rencontrées depuis la cible.
                blocked_by: result
                    .blocked
                    .iter()
                    .map(|l| link_to_dto(&ctx, l))
                    .collect(),
                caveats,
            })
        }

        Query::Analogy {
            from,
            to,
            weights,
            min_score,
            hops,
        } => {
            // Ph3 [D3] : poids/seuil paramétrables (défauts documentés).
            // Ph6b : HOPS = profondeur de contexte (1 = mono-saut).
            let matches =
                crate::analogy::find_analogies_hops(ir, from, to, *min_score, *weights, *hops);
            Ok(QueryResult::AnalogyReport {
                pattern_from: from.clone(),
                pattern_to: to.clone(),
                matches: matches
                    .into_iter()
                    .map(|m| AnalogyMatchDto {
                        from_label: m.from_label,
                        to_label: m.to_label,
                        relation: serde_name(&m.relation),
                        confidence: m.confidence,
                        similarity_score: m.similarity_score,
                    })
                    .collect(),
            })
        }

        Query::ZoomIn(label) => {
            let node = pearl::find_best(ctx.ir, label)
                .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
            let children: Vec<NodeSummaryDto> = ir
                .nodes
                .iter()
                .filter(|n| n.parent == Some(node.id))
                .map(|n| NodeSummaryDto {
                    label: n.label.clone(),
                    node_type: serde_name(&n.node_type),
                })
                .collect();
            Ok(QueryResult::HierarchyZoomIn {
                parent_label: node.label.clone(),
                children,
            })
        }

        Query::ZoomOut(label) => {
            let node = pearl::find_best(ctx.ir, label)
                .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
            let parent_label = node
                .parent
                .and_then(|pid| ir.nodes.iter().find(|n| n.id == pid))
                .map(|n| n.label.clone());
            Ok(QueryResult::HierarchyZoomOut {
                child_label: node.label.clone(),
                parent_label,
            })
        }

        Query::Aggregate(label) => {
            let node = pearl::find_best(ctx.ir, label)
                .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
            let child_ids: std::collections::HashSet<_> = ir
                .nodes
                .iter()
                .filter(|n| n.parent == Some(node.id))
                .map(|n| n.id)
                .collect();
            let child_count = child_ids.len();
            // M4 : index partagé (plus de build local).
            let nm = &ctx.nm;
            // Ph7e : les liens témoins (cause existentielle/spécifique) ne sont
            // jamais généralisés en agrégat — exclus ici.
            let outgoing: Vec<LinkDto> = ir
                .edges
                .iter()
                .filter(|(s, _, _)| child_ids.contains(s))
                .filter(|(_, d, _)| !child_ids.contains(d))
                .filter(|(s, _, _)| {
                    !nm.get(s).is_some_and(|n| {
                        matches!(
                            n.scope,
                            gcn_ir::Scope::Existential | gcn_ir::Scope::Specific
                        )
                    })
                })
                .map(|(s, d, e)| LinkDto {
                    from: nm.get(s).map(|n| n.label.clone()).unwrap_or_default(),
                    to: nm.get(d).map(|n| n.label.clone()).unwrap_or_default(),
                    relation: serde_name(&e.relation),
                    confidence: e.confidence,
                    negated: e.negated,
                    joint_group_id: e.joint_group_id.clone(),
                    // P1-5 : même résolution que link_to_dto, depuis CausalEdge.third.
                    third_node_label: e.third.as_ref().and_then(|t| {
                        ir.nodes
                            .iter()
                            .find(|n| n.id.0 as u64 == t.node)
                            .map(|n| n.label.clone())
                    }),
                    // Ph6b : dérivation D3 (même règle que link_to_dto).
                    derived_from: e.derivation.as_ref().map(|drv| {
                        drv.source_edges
                            .iter()
                            .map(|(s, d)| (ctx.node_label(*s), ctx.node_label(*d)))
                            .collect()
                    }),
                    motif: None,
                    cycle_effect: None,
                    origin: nm.get(s).map(|n| serde_name(&n.origin)),
                    explicit: e.explicit,
                    extraction_method: None,
                    model_version: None,
                    span: None,
                    kind: None,
                    scope: nm.get(s).map(|n| serde_name(&n.scope)),
                    agent: nm.get(s).and_then(|n| n.attributes.agent.clone()),
                    patient: nm.get(d).and_then(|n| n.attributes.patient.clone()),
                    agent_type: nm
                        .get(s)
                        .and_then(|n| n.attributes.agent_type)
                        .map(|t| serde_name(&t)),
                    witness_only: false,
                })
                .collect();
            let mean_confidence = if outgoing.is_empty() {
                0.0
            } else {
                outgoing.iter().map(|l| l.confidence).sum::<f32>() / outgoing.len() as f32
            };
            Ok(QueryResult::HierarchyAggregate {
                parent_label: node.label.clone(),
                child_count,
                outgoing_edges: outgoing,
                mean_confidence,
            })
        }

        Query::Centrality(label) => {
            let node = pearl::find_best(ctx.ir, label)
                .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
            let in_edges: Vec<f32> = ir
                .edges
                .iter()
                .filter(|(_, d, _)| *d == node.id)
                .map(|(_, _, e)| e.confidence)
                .collect();
            let out_edges: Vec<f32> = ir
                .edges
                .iter()
                .filter(|(s, _, _)| *s == node.id)
                .map(|(_, _, e)| e.confidence)
                .collect();
            let degree_in = in_edges.len();
            let degree_out = out_edges.len();
            let weighted_in: f32 = in_edges.iter().sum();
            let weighted_out: f32 = out_edges.iter().sum();
            let total_degree = degree_in + degree_out;
            let centrality_score = if total_degree > 0 {
                (weighted_in + weighted_out) / total_degree as f32
            } else {
                0.0
            };
            Ok(QueryResult::CentralityReport {
                label: node.label.clone(),
                degree_in,
                degree_out,
                weighted_in,
                weighted_out,
                centrality_score,
            })
        }

        Query::Spof => {
            // Garde : constante unique pearl::SPOF_MAX_NODES (pas de doublon local).
            if ir.nodes.len() > pearl::SPOF_MAX_NODES {
                return Err(BackendError::QueryParseError(format!(
                    "SPOF: graph too large ({} nodes > {}). Use CENTRALITY on specific nodes instead.",
                    ir.nodes.len(),
                    pearl::SPOF_MAX_NODES
                )));
            }
            let (total_pairs, scores) = pearl::spof_all_ctx(&ctx)?;
            let spof_nodes: Vec<SpofNodeDto> = scores
                .into_iter()
                .map(|s: SpofScore| {
                    let spof_score = if total_pairs > 0 {
                        s.score as f32 / total_pairs as f32
                    } else {
                        0.0
                    };
                    SpofNodeDto {
                        label: s.label,
                        paths_cut: s.score,
                        spof_score,
                    }
                })
                .collect();
            Ok(QueryResult::SpofReport {
                total_pairs,
                nodes: spof_nodes,
            })
        }

        Query::Doubts => {
            // Ph7b : liste brute des non-résolus (labels + champs + candidats).
            let mut doubts: Vec<Caveat> = ir
                .unresolved
                .iter()
                .map(|amb| {
                    let node = ctx.node_label(amb.node_id);
                    Caveat {
                        node,
                        field: serde_name(&amb.field),
                        candidates: amb.candidates.iter().map(|c| c.value.clone()).collect(),
                        hint: amb
                            .context_hint
                            .clone()
                            .unwrap_or_else(|| "levée d'ambiguïté requise".to_string()),
                    }
                })
                .collect();
            doubts.sort_by(|a, b| a.node.cmp(&b.node).then(a.field.cmp(&b.field)));
            Ok(QueryResult::Doubts {
                count: doubts.len(),
                doubts,
            })
        }

        Query::Provenance(label) => {
            // Ph7f : fiche complète des arêtes incidentes (qui, quand, comment, span).
            let node = pearl::find_best(ctx.ir, label)
                .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
            let mut entries: Vec<ProvenanceEntryDto> = ir
                .edges
                .iter()
                .filter(|(s, d, _)| *s == node.id || *d == node.id)
                .map(|(s, d, e)| {
                    let (method, version, doc, at) = e
                        .provenance
                        .as_ref()
                        .map(|p| {
                            (
                                Some(method_name(p.extraction_method).to_string()),
                                Some(p.model_version.clone()),
                                p.doc_ref.clone(),
                                Some(p.extracted_at.clone()),
                            )
                        })
                        .unwrap_or((None, None, None, None));
                    let (span, kind) = e
                        .provenance
                        .as_ref()
                        .map(|p| span_text_kind(ir, &p.span))
                        .unwrap_or((None, None));
                    ProvenanceEntryDto {
                        from: ctx.node_label(*s),
                        to: ctx.node_label(*d),
                        relation: serde_name(&e.relation),
                        explicit: e.explicit,
                        extraction_method: method,
                        model_version: version,
                        span,
                        kind,
                        doc_ref: doc,
                        extracted_at: at,
                    }
                })
                .collect();
            entries.sort_by(|a, b| {
                a.from
                    .cmp(&b.from)
                    .then(a.to.cmp(&b.to))
                    .then(a.relation.cmp(&b.relation))
            });
            Ok(QueryResult::ProvenanceReport {
                label: node.label.clone(),
                entries,
            })
        }

        Query::NormDiff(from, to) => {
            if pearl::find_best(ctx.ir, from).is_none() {
                return Err(BackendError::NodeNotFound(from.clone()));
            }
            if pearl::find_best(ctx.ir, to).is_none() {
                return Err(BackendError::NodeNotFound(to.clone()));
            }
            let links = pearl::chain_ctx(&ctx, from, to);
            let path_exists = links.is_some();
            let path_links_dto: Vec<LinkDto> = links
                .as_ref()
                .map(|ls| ls.iter().map(|l| link_to_dto(&ctx, l)).collect())
                .unwrap_or_default();
            let n_edges_on_path = path_links_dto.len();
            let observed_confidence = links
                .as_ref()
                .and_then(|ls| ls.iter().map(|l| l.confidence).reduce(f32::min))
                .unwrap_or(0.0);
            let gap = if path_exists {
                1.0 - observed_confidence
            } else {
                1.0
            };
            Ok(QueryResult::NormativeDiff {
                from: from.clone(),
                to: to.clone(),
                path_exists,
                n_edges_on_path,
                observed_confidence,
                gap,
                gap_rate_pct: gap * 100.0,
                path_links: path_links_dto,
            })
        }

        Query::Density(label_opt) => {
            let n_nodes = ir.nodes.len();
            let n_edges = ir.edges.len();
            let global_density = if n_nodes > 1 {
                n_edges as f32 / (n_nodes * (n_nodes - 1)) as f32
            } else {
                0.0
            };
            if let Some(label) = label_opt {
                let node = pearl::find_best(ctx.ir, label)
                    .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
                let local_degree = ir
                    .edges
                    .iter()
                    .filter(|(s, d, _)| *s == node.id || *d == node.id)
                    .count();
                let local_density = if n_nodes > 1 {
                    Some(local_degree as f32 / (2 * (n_nodes - 1)) as f32)
                } else {
                    None
                };
                Ok(QueryResult::DensityReport {
                    global_density,
                    n_nodes,
                    n_edges,
                    node_label: Some(node.label.clone()),
                    local_degree: Some(local_degree),
                    local_density,
                })
            } else {
                Ok(QueryResult::DensityReport {
                    global_density,
                    n_nodes,
                    n_edges,
                    node_label: None,
                    local_degree: None,
                    local_density: None,
                })
            }
        }

        Query::Coverage(label) => {
            let node = pearl::find_best(ctx.ir, label)
                .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
            let incident: Vec<_> = ir
                .edges
                .iter()
                .filter(|(s, d, _)| *s == node.id || *d == node.id)
                .collect();
            let degree = incident.len();
            let n_with_provenance = incident
                .iter()
                .filter(|(_, _, e)| e.provenance.is_some())
                .count();
            let n_edges = ir.edges.len();
            let coverage_score = if n_edges > 0 {
                degree as f32 / n_edges as f32
            } else {
                0.0
            };
            let provenance_ratio = if degree > 0 {
                n_with_provenance as f32 / degree as f32
            } else {
                1.0
            };
            Ok(QueryResult::CoverageReport {
                label: node.label.clone(),
                degree,
                n_with_provenance,
                coverage_score,
                provenance_ratio,
            })
        }

        Query::Reliability(label) => {
            let node = pearl::find_best(ctx.ir, label)
                .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
            let incident: Vec<_> = ir
                .edges
                .iter()
                .filter(|(s, d, _)| *s == node.id || *d == node.id)
                .map(|(_, _, e)| e)
                .collect();
            let degree = incident.len();
            let (min_confidence, mean_confidence) = if degree > 0 {
                let min = incident
                    .iter()
                    .map(|e| e.confidence)
                    .fold(f32::INFINITY, f32::min);
                let mean = incident.iter().map(|e| e.confidence).sum::<f32>() / degree as f32;
                (min, mean)
            } else {
                (0.0, 0.0)
            };
            let provenance_ratio = if degree > 0 {
                incident.iter().filter(|e| e.provenance.is_some()).count() as f32 / degree as f32
            } else {
                1.0
            };
            let reliability_score = mean_confidence * provenance_ratio;
            Ok(QueryResult::ReliabilityReport {
                label: node.label.clone(),
                min_confidence,
                mean_confidence,
                provenance_ratio,
                reliability_score,
            })
        }

        Query::Explain { effect, contrast } => {
            let node = pearl::find_best(ctx.ir, effect)
                .ok_or_else(|| BackendError::NodeNotFound(effect.clone()))?;
            let hypotheses = pearl::abduct_ctx(&ctx, node.id);
            // Ph6b : contraste vs alternative (causes de E hors F).
            let contrast_dto = match contrast {
                Some(alt) => {
                    let alt_node = pearl::find_best(ctx.ir, alt)
                        .ok_or_else(|| BackendError::NodeNotFound(alt.clone()))?;
                    let alt_labels: std::collections::HashSet<String> =
                        pearl::abduct_ctx(&ctx, alt_node.id)
                            .into_iter()
                            .map(|h| h.label)
                            .collect();
                    let mut distinctive = Vec::new();
                    let mut shared = Vec::new();
                    for h in &hypotheses {
                        if alt_labels.contains(&h.label) {
                            shared.push(h.label.clone());
                        } else {
                            distinctive.push(AbductionHypothesisDto {
                                label: h.label.clone(),
                                relation: serde_name(&h.relation),
                                edge_confidence: h.edge_confidence,
                                score: h.score,
                                depth: h.depth,
                            });
                        }
                    }
                    shared.sort();
                    shared.dedup();
                    Some(AbductionContrastDto {
                        alternative: alt_node.label.clone(),
                        distinctive,
                        shared,
                    })
                }
                None => None,
            };
            Ok(QueryResult::Abduction {
                effect: node.label.clone(),
                hypotheses: hypotheses
                    .iter()
                    .map(|h| AbductionHypothesisDto {
                        label: h.label.clone(),
                        relation: serde_name(&h.relation),
                        edge_confidence: h.edge_confidence,
                        score: h.score,
                        depth: h.depth,
                    })
                    .collect(),
                contrast: contrast_dto,
                by_agent: pearl::group_by_agent(ctx.ir, &hypotheses)
                    .into_iter()
                    .map(|(agent_type, hs)| AgentGroupDto {
                        agent_type,
                        hypotheses: hs
                            .into_iter()
                            .map(|h| AbductionHypothesisDto {
                                label: h.label,
                                relation: serde_name(&h.relation),
                                edge_confidence: h.edge_confidence,
                                score: h.score,
                                depth: h.depth,
                            })
                            .collect(),
                    })
                    .collect(),
            })
        }

        Query::Ask { label } => {
            // Ph6b : fiche nœud (1 saut, sans traversée ni propagation).
            let node = pearl::find_best(ctx.ir, label)
                .ok_or_else(|| BackendError::NodeNotFound(label.clone()))?;
            let (causes, effects) = pearl::direct_links_ctx(&ctx, node.id);
            Ok(QueryResult::AskReport {
                label: node.label.clone(),
                node_type: serde_name(&node.node_type),
                causes: causes.iter().map(|l| link_to_dto(&ctx, l)).collect(),
                effects: effects.iter().map(|l| link_to_dto(&ctx, l)).collect(),
            })
        }

        Query::WhyNot { from, to } => {
            // Ph6b : pourquoi aucun chemin causal (WHY-NOT = bloqueurs niés).
            let from_node = pearl::find_best(ctx.ir, from)
                .ok_or_else(|| BackendError::NodeNotFound(from.clone()))?;
            let to_node = pearl::find_best(ctx.ir, to)
                .ok_or_else(|| BackendError::NodeNotFound(to.clone()))?;
            let (paths, blocked) = pearl::chain_topk_ctx(&ctx, from, to, 1);
            if let Some(best) = paths.first() {
                // Un chemin pur existe : pas de « pourquoi pas ».
                return Ok(QueryResult::WhyNot {
                    from: from_node.label.clone(),
                    to: to_node.label.clone(),
                    verdict: "chemin_existe".to_string(),
                    links: best.links.iter().map(|l| link_to_dto(&ctx, l)).collect(),
                    blocked_by: vec![],
                    proches: vec![],
                });
            }
            if blocked.is_empty() {
                // Vraiment aucun lien : voisins pour rebondir.
                let mut proches_set: std::collections::HashSet<String> =
                    std::collections::HashSet::new();
                for (s, d, _) in &ir.edges {
                    if *s == from_node.id && *d != to_node.id {
                        proches_set.insert(ctx.node_label(*d));
                    }
                    if *d == to_node.id && *s != from_node.id {
                        proches_set.insert(ctx.node_label(*s));
                    }
                }
                let mut proches: Vec<String> = proches_set.into_iter().collect();
                proches.sort();
                proches.truncate(6);
                return Ok(QueryResult::WhyNot {
                    from: from_node.label.clone(),
                    to: to_node.label.clone(),
                    verdict: "aucun_lien".to_string(),
                    links: vec![],
                    blocked_by: vec![],
                    proches,
                });
            }
            Ok(QueryResult::WhyNot {
                from: from_node.label.clone(),
                to: to_node.label.clone(),
                verdict: "coupure_negation".to_string(),
                links: vec![],
                blocked_by: blocked.iter().map(|l| link_to_dto(&ctx, l)).collect(),
                proches: vec![],
            })
        }

        Query::ChainT(from, to) => {
            if pearl::find_best(ctx.ir, from).is_none() {
                return Err(BackendError::NodeNotFound(from.clone()));
            }
            if pearl::find_best(ctx.ir, to).is_none() {
                return Err(BackendError::NodeNotFound(to.clone()));
            }
            let result = pearl::chain_temporal_ctx(&ctx, from, to);
            let found = result.links.is_some();
            Ok(QueryResult::TemporalPath {
                from: from.clone(),
                to: to.clone(),
                found,
                temporally_ordered: result.temporally_ordered,
                links: result
                    .links
                    .unwrap_or_default()
                    .iter()
                    .map(temporal_link_to_dto)
                    .collect(),
            })
        }

        Query::Before(a, b) => {
            let node_a =
                pearl::find_best(ctx.ir, a).ok_or_else(|| BackendError::NodeNotFound(a.clone()))?;
            let node_b =
                pearl::find_best(ctx.ir, b).ok_or_else(|| BackendError::NodeNotFound(b.clone()))?;
            let a_index = node_a.temporal_index;
            let b_index = node_b.temporal_index;
            let path_exists = pearl::chain_ctx(&ctx, a, b).is_some();
            let a_before_b = match (a_index, b_index) {
                (Some(ai), Some(bi)) => ai < bi && path_exists,
                _ => path_exists,
            };
            Ok(QueryResult::TemporalOrder {
                a: node_a.label.clone(),
                b: node_b.label.clone(),
                a_before_b,
                a_index,
                b_index,
                path_exists,
            })
        }

        Query::Delay(from, to) => {
            if pearl::find_best(ctx.ir, from).is_none() {
                return Err(BackendError::NodeNotFound(from.clone()));
            }
            if pearl::find_best(ctx.ir, to).is_none() {
                return Err(BackendError::NodeNotFound(to.clone()));
            }
            let result = pearl::chain_temporal_ctx(&ctx, from, to);
            let (gap_min_sum, gap_max_sum, index_delta) = if let Some(ref links) = result.links {
                let min_sum: Option<i32> = links
                    .iter()
                    .map(|l| l.gap_min)
                    .try_fold(0i32, |acc, v| v.map(|x| acc + x));
                let max_sum: Option<i32> = links
                    .iter()
                    .map(|l| l.gap_max)
                    .try_fold(0i32, |acc, v| v.map(|x| acc + x));
                let idx_delta = links
                    .iter()
                    .filter_map(|l| l.index_delta)
                    .reduce(|a, b| a + b);
                (min_sum, max_sum, idx_delta)
            } else {
                (None, None, None)
            };
            Ok(QueryResult::TemporalDelay {
                from: from.clone(),
                to: to.clone(),
                found: result.links.is_some(),
                gap_min_sum,
                gap_max_sum,
                index_delta,
            })
        }
    }
}

/// Variante stricte de CHAIN : erreur `NoPath` si les nœuds existent mais sans
/// chemin causal (rend `BackendError::NoPath` atteignable).
///
/// Ph1 [D1] + tiers Éq.12 : comme CHAIN (niées jamais traversées) ET exige que
/// tout lien ternaire du chemin ait son `third_node` présent sur le chemin
/// (sinon `NoPath`). Les liens du tiers restent annotés via `third_node_label`.
pub fn chain_strict(ir: &CausalIR, from: &str, to: &str) -> Result<Vec<LinkDto>, BackendError> {
    // M4 : un seul build (contexte partagé comme execute).
    let ctx = pearl::GraphCtx::new(ir);
    if pearl::find_best(ctx.ir, from).is_none() {
        return Err(BackendError::NodeNotFound(from.to_string()));
    }
    if pearl::find_best(ctx.ir, to).is_none() {
        return Err(BackendError::NodeNotFound(to.to_string()));
    }
    let no_path = || BackendError::NoPath(from.to_string(), to.to_string());
    let (path, _) = pearl::chain_with_blocked_ctx(&ctx, from, to);
    let links = path.ok_or_else(no_path)?;
    let on_path: std::collections::HashSet<NodeId> =
        links.iter().flat_map(|l| [l.from_id, l.to_id]).collect();
    for l in &links {
        if let Some(third) = l.third_node
            && !on_path.contains(&third)
        {
            return Err(BackendError::NoPath(from.to_string(), to.to_string()));
        }
    }
    Ok(links.iter().map(|l| link_to_dto(&ctx, l)).collect())
}

fn temporal_link_to_dto(l: &pearl::TemporalLink) -> TemporalLinkDto {
    TemporalLinkDto {
        from: l.from_label.clone(),
        to: l.to_label.clone(),
        relation: serde_name(&l.relation),
        confidence: l.confidence,
        negated: l.negated,
        gap_min: l.gap_min,
        gap_max: l.gap_max,
        index_delta: l.index_delta,
    }
}

/// Nom serde snake_case d'une valeur (relation, type de nœud, type de cycle),
/// avec repli Debug minuscule. Évite les incohérences PascalCase/snake_case
/// entre DTOs (constat d'audit : Gaps/Analogy/Explain/Zoom/Temporal en Debug,
/// LinkDto en serde).
fn serde_name<T: serde::Serialize + std::fmt::Debug>(v: &T) -> String {
    serde_json::to_value(v)
        .ok()
        .and_then(|j| j.as_str().map(|s| s.to_string()))
        .unwrap_or_else(|| format!("{v:?}").to_lowercase())
}

/// Ph7f — vocabulaire court des méthodes (cohérent avec `METHOD=`).
pub fn method_name(m: gcn_ir::ExtractionMethod) -> &'static str {
    match m {
        gcn_ir::ExtractionMethod::SilverAnnotator => "silver",
        gcn_ir::ExtractionMethod::MlPython => "ml",
        gcn_ir::ExtractionMethod::SymbolicRust => "symbolic",
        gcn_ir::ExtractionMethod::Derived => "derived",
    }
}

/// Ph7f — span → (contenu, nature) : texte extrait si possible (bornes
/// caractères valides), sinon coordonnées ; `text`/`code`/`synthetic`.
pub fn span_text_kind(
    ir: &CausalIR,
    span: &gcn_ir::SourceSpan,
) -> (Option<String>, Option<String>) {
    match span {
        gcn_ir::SourceSpan::TokenSpan { start, end } => {
            let (s, e) = (*start as usize, *end as usize);
            let text = &ir.source_text;
            if s <= e
                && text.is_char_boundary(s)
                && text.is_char_boundary(e)
                && !text[s..e].is_empty()
            {
                (Some(text[s..e].to_string()), Some("text".to_string()))
            } else {
                (Some(format!("{s}-{e}")), Some("text".to_string()))
            }
        }
        gcn_ir::SourceSpan::CodeSpan {
            start_line,
            start_col,
            end_line,
            end_col,
        } => (
            Some(format!("L{start_line}:{start_col}-L{end_line}:{end_col}")),
            Some("code".to_string()),
        ),
        gcn_ir::SourceSpan::Synthetic => (None, Some("synthetic".to_string())),
    }
}

/// M4 — DTO sur contexte partagé : index nœuds/arêtes au lieu de scans
/// linéaires répétés (8 recherches nœuds + 1 arête par lien). Mêmes valeurs.
pub(crate) fn link_to_dto(ctx: &pearl::GraphCtx<'_>, l: &CausalLink) -> LinkDto {
    // C4 fix : sérialiser la relation en snake_case via serde (pas Debug).
    let relation = serde_name(&l.relation);
    let ir = ctx.ir;
    let nm = &ctx.nm;
    let em = &ctx.em;
    let src_node = nm.get(&l.from_id);
    let dst_node = nm.get(&l.to_id);
    // P1-5 : résoudre le tiers ternaire (médiateur/condition) en label.
    // Était `None` systématique alors que `CausalLink.third_node` est peuplé
    // depuis `CausalEdge.third` (pearl.rs) — perte ternaire de bout en bout.
    let third_node_label = l
        .third_node
        .and_then(|tid| nm.get(&tid).map(|n| n.label.clone()));
    // Une seule résolution d'arête (dérivation, cycle, provenance).
    // Même sémantique que le scan : première arête (src, dst) (cf. or_insert).
    let edge = em.get(&(l.from_id.0, l.to_id.0)).copied();
    // Ph6b : dérivations D3 systématiques (`Derivation.source_edges` → labels).
    let derived_from = edge.and_then(|e| e.derivation.as_ref()).map(|drv| {
        drv.source_edges
            .iter()
            .map(|(s, d)| (ctx.node_label(*s), ctx.node_label(*d)))
            .collect()
    });
    // Ph7a : effet de cycle (damped/amplifying) via in_cycle.
    let cycle_effect = edge
        .and_then(|e| e.in_cycle)
        .and_then(|cid| ir.cycles.iter().find(|c| c.id == cid))
        .and_then(|c| match c.cycle_type {
            gcn_ir::CycleType::FeedbackNegative => Some("damped".to_string()),
            gcn_ir::CycleType::FeedbackPositive => Some("amplifying".to_string()),
            gcn_ir::CycleType::Oscillation => None,
        });
    // Ph7f : provenance (méthode, modèle, span) + explicite.
    let (extraction_method, model_version) = edge
        .and_then(|e| e.provenance.as_ref())
        .map(|p| {
            (
                Some(method_name(p.extraction_method).to_string()),
                Some(p.model_version.clone()),
            )
        })
        .unwrap_or((None, None));
    let (span, kind) = edge
        .and_then(|e| e.provenance.as_ref())
        .map(|p| span_text_kind(ir, &p.span))
        .unwrap_or((None, None));
    LinkDto {
        from: l.from_label.clone(),
        to: l.to_label.clone(),
        relation,
        confidence: l.confidence,
        negated: l.negated,
        joint_group_id: l.joint_group_id.clone(),
        third_node_label,
        derived_from,
        // Ph7 : motif de coupure (None = traversé).
        motif: l.motif.map(|m| m.as_str().to_string()),
        cycle_effect,
        // Ph7b : origine du nœud source (cause) — ranking × poids.
        origin: src_node.map(|n| serde_name(&n.origin)),
        explicit: edge.is_some_and(|e| e.explicit),
        extraction_method,
        model_version,
        span,
        kind,
        // Ph7e : scope du nœud source (côté cause, cohérent avec origin/agent).
        scope: src_node.map(|n| serde_name(&n.scope)),
        // Ph7d : agent + type du nœud source (qui agit), patient du nœud cible (qui subit).
        agent: src_node.and_then(|n| n.attributes.agent.clone()),
        patient: dst_node.and_then(|n| n.attributes.patient.clone()),
        agent_type: src_node
            .and_then(|n| n.attributes.agent_type)
            .map(|t| serde_name(&t)),
        // Ph7e : lien témoin si la cause est existentielle/spécifique
        // (généralisation interdite — exclu des agrégats).
        witness_only: src_node.is_some_and(|n| {
            matches!(
                n.scope,
                gcn_ir::Scope::Existential | gcn_ir::Scope::Specific
            )
        }),
    }
}

fn cycle_to_dto(ir: &CausalIR) -> impl Fn(&CausalCycle) -> CycleDto + '_ {
    move |c| {
        // Ph7a : effet typé (feedback → damped/amplifying, oscillation → oscillating).
        let effect = match c.cycle_type {
            gcn_ir::CycleType::FeedbackNegative => Some("damped".to_string()),
            gcn_ir::CycleType::FeedbackPositive => Some("amplifying".to_string()),
            gcn_ir::CycleType::Oscillation => Some("oscillating".to_string()),
        };
        CycleDto {
            id: c.id.0,
            kind: serde_name(&c.cycle_type),
            nodes: c.path.iter().map(|&id| pearl::node_label(ir, id)).collect(),
            effect,
        }
    }
}
