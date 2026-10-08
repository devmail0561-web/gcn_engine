// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! Always-answer (Ph8 [R-ALWAYS]) : `respond()` enveloppe `execute()`.
//!
//! Jamais d'Err nue ni de None à la frontière interactive : tout point
//! d'entrée retourne `UserResponse` { phrase NL 1-3 phrases + contenu
//! structuré même partiel + statut + caveats + suggestions }.
//! `execute()` reste inchangé (Err internes, tests intacts).
//! Batch/CI : la CLI sort non-zéro sur degrade (contrat batch préservé).

use std::collections::{HashMap, HashSet};

use gcn_ir::CausalIR;
use gcn_verbalizer::fallback::{SpeechKind, Status, say};
use serde::{Deserialize, Serialize};

use crate::error::BackendError;
use crate::pearl::{self, SPOF_MAX_NODES};
use crate::query::{LinkDto, Query, QueryResult, execute};

/// Enveloppe always-answer (Ph8) : contrat d'échange versionné (pont M1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserResponse {
    /// Statut : ok | partiel | degrade (frontière interactive).
    pub status: Status,
    /// Phrase NL FR (1-3 phrases) : constat + miettes réelles + suggestions.
    /// Jamais vide, jamais mensongère ( gabarit de secours, voie noble =
    /// décodeur appris quand il fonctionne).
    pub phrase: String,
    /// Langue de la phrase (Ph8 : FR d'abord, autres langues à la demande ;
    /// préférera `cir.intent.lang` puis la langue source à l'avenir).
    pub lang: String,
    /// Contenu structuré, même partiel (None si rien d'exploitable).
    #[serde(default)]
    pub result: Option<QueryResult>,
    /// Avertissements (Ph7 : hypothétiques/ambiguïtés — vide en Ph8).
    #[serde(default)]
    pub caveats: Vec<String>,
    /// Prochaines requêtes possibles (labels réels, jamais inventés).
    #[serde(default)]
    pub suggestions: Vec<String>,
}

/// Langue des gabarits (Ph8 : FR uniquement, cf. `UserResponse::lang`).
fn phrase_lang() -> String {
    "fr".to_string()
}

/// Items plafonnés pour la phrase (déterministes : ordre du résultat).
const MAX_ITEMS: usize = 5;

fn dto_items(links: &[LinkDto]) -> (Vec<String>, usize) {
    let items: Vec<String> = links
        .iter()
        .take(MAX_ITEMS)
        .map(|l| format!("{} → {}", l.from, l.to))
        .collect();
    let extra = links.len().saturating_sub(items.len());
    (items, extra)
}

fn answer(
    status: Status,
    head: String,
    links: &[LinkDto],
    suggestions: Vec<String>,
) -> UserResponse {
    let (items, extra) = dto_items(links);
    answer_items(status, head, items, extra, suggestions)
}

fn answer_items(
    status: Status,
    head: String,
    items: Vec<String>,
    extra: usize,
    suggestions: Vec<String>,
) -> UserResponse {
    let phrase = say(status, "fr", &SpeechKind::Answer { head, items, extra });
    UserResponse {
        status,
        phrase,
        lang: phrase_lang(),
        result: None,
        caveats: vec![],
        suggestions,
    }
}

fn with_result(mut resp: UserResponse, result: QueryResult) -> UserResponse {
    resp.result = Some(result);
    resp
}

/// Distance d'édition (suggestion « vouliez-vous dire », déterministe).
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, &ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, &cb) in b.iter().enumerate() {
            cur.push(
                (prev[j] + usize::from(ca != cb))
                    .min(prev[j + 1] + 1)
                    .min(cur[j] + 1),
            );
        }
        prev = cur;
    }
    prev[b.len()]
}

const KNOWN_VERBS: &[&str] = &[
    "WHY",
    "WHAT",
    "CHAIN",
    "CYCLES",
    "GAPS",
    "DO",
    "COUNTERFACTUAL",
    "EXPLAIN",
    "ANALOGY",
    "CENTRALITY",
    "SPOF",
    "DIFF",
    "DENSITY",
    "COVERAGE",
    "RELIABILITY",
    "CHAIN_T",
    "BEFORE?",
    "DELAY",
    "ZOOM_IN",
    "ZOOM_OUT",
    "AGGREGATE",
];

/// « Vouliez-vous dire » sur le verbe (distance ≤ 2), sinon None.
fn did_you_mean(input: &str) -> Option<String> {
    let verb = input
        .trim()
        .trim_end_matches('?')
        .split_whitespace()
        .next()?
        .to_ascii_uppercase();
    if verb.is_empty() || KNOWN_VERBS.contains(&verb.as_str()) {
        return None;
    }
    KNOWN_VERBS
        .iter()
        .map(|v| (edit_distance(&verb, v), v))
        .filter(|(d, _)| *d <= 2 && *d > 0)
        .min_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(b.1)))
        .map(|(_, v)| format!("Vouliez-vous dire « {v} … » ?"))
}

/// Voisins directs (1 saut) de `from` (sortants) et `to` (entrants) — rebond NoPath.
fn proches(ir: &CausalIR, from: &str, to: &str) -> Vec<String> {
    let (Some(f), Some(t)) = (pearl::find_best(ir, from), pearl::find_best(ir, to)) else {
        return vec![];
    };
    let mut out: HashSet<String> = HashSet::new();
    for (s, d, _) in &ir.edges {
        if *s == f.id && *d != t.id {
            out.insert(pearl::node_label(ir, *d));
        }
        if *d == t.id && *s != f.id {
            out.insert(pearl::node_label(ir, *s));
        }
    }
    let mut v: Vec<String> = out.into_iter().collect();
    v.sort();
    v.truncate(6);
    v
}

/// Top-k nœuds par degré (entrant + sortant) — annonce trop-gros honnête.
fn top_connected(ir: &CausalIR, k: usize) -> Vec<(String, usize)> {
    let mut deg: HashMap<gcn_ir::NodeId, usize> = HashMap::new();
    for (s, d, _) in &ir.edges {
        *deg.entry(*s).or_default() += 1;
        *deg.entry(*d).or_default() += 1;
    }
    let mut v: Vec<(String, usize)> = deg
        .into_iter()
        .map(|(id, d)| (pearl::node_label(ir, id), d))
        .collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.truncate(k);
    v
}

fn bad_query(input: &str, err: &BackendError) -> UserResponse {
    let _ = err;
    let hint = did_you_mean(input);
    let suggestions = match &hint {
        Some(h) => vec![h.clone(), "Exemple : WHY <mot>.".to_string()],
        None => vec!["Exemple : WHY <mot>.".to_string()],
    };
    let phrase = say(Status::Degrade, "fr", &SpeechKind::BadQuery { hint });
    UserResponse {
        status: Status::Degrade,
        phrase,
        lang: phrase_lang(),
        result: None,
        caveats: vec![],
        suggestions,
    }
}

fn empty_ir() -> UserResponse {
    let phrase = say(Status::Degrade, "fr", &SpeechKind::EmptyIr);
    UserResponse {
        status: Status::Degrade,
        phrase,
        lang: phrase_lang(),
        result: None,
        caveats: vec![],
        suggestions: vec!["Chargez un graphe (CIR JSON) puis relancez la requête.".to_string()],
    }
}

fn too_big(ir: &CausalIR) -> UserResponse {
    let top = top_connected(ir, 5);
    let suggestions: Vec<String> = top.iter().map(|(l, _)| format!("WHY {l}")).collect();
    let phrase = say(
        Status::Partiel,
        "fr",
        &SpeechKind::TooBig {
            nodes: ir.nodes.len(),
            top,
        },
    );
    UserResponse {
        status: Status::Partiel,
        phrase,
        lang: phrase_lang(),
        result: None,
        caveats: vec![],
        suggestions,
    }
}

fn not_found(label: &str) -> UserResponse {
    let phrase = say(
        Status::Degrade,
        "fr",
        &SpeechKind::NotFound {
            label: label.to_string(),
        },
    );
    UserResponse {
        status: Status::Degrade,
        phrase,
        lang: phrase_lang(),
        result: None,
        caveats: vec![],
        suggestions: vec![
            "Essayez un mot plus court ou sans faute (recherche floue : préfixes, limite 5)."
                .to_string(),
        ],
    }
}

fn ambiguous(label: &str, candidates: Vec<String>) -> UserResponse {
    let shown: Vec<String> = candidates.iter().take(5).cloned().collect();
    let more = candidates.len().saturating_sub(shown.len());
    let mut suggestions: Vec<String> = candidates.iter().take(8).cloned().collect();
    suggestions.push("Précisez avec le label exact.".to_string());
    let phrase = say(
        Status::Partiel,
        "fr",
        &SpeechKind::Ambiguous {
            label: label.to_string(),
            candidates: shown,
            more,
        },
    );
    UserResponse {
        status: Status::Partiel,
        phrase,
        lang: phrase_lang(),
        result: None,
        caveats: vec![],
        suggestions,
    }
}

fn incident(detail: String) -> UserResponse {
    let phrase = say(Status::Degrade, "fr", &SpeechKind::Incident { detail });
    UserResponse {
        status: Status::Degrade,
        phrase,
        lang: phrase_lang(),
        result: None,
        caveats: vec![],
        suggestions: vec![
            "Réessayez.".to_string(),
            "Si ça persiste, signalez la requête.".to_string(),
        ],
    }
}

/// Enveloppe always-answer (Ph8) : ne retourne jamais d'Err.
///
/// Règle de statut (documentée) : Degrade = parse / introuvable / vide / incident ;
/// Partiel = ambigu / sans-chemin / flou / coupures niées / trop-gros / ordre
/// temporel rejeté ; Ok sinon.
pub fn respond(input: &str, ir: &CausalIR) -> UserResponse {
    let query = match Query::parse(input) {
        Ok(q) => q,
        Err(e) => return bad_query(input, &e),
    };
    if ir.nodes.is_empty() && ir.edges.is_empty() {
        return empty_ir();
    }
    // Garde SPOF : partiel + top-k au lieu d'Err (Ph8).
    if let Query::Spof = query
        && ir.nodes.len() > SPOF_MAX_NODES
    {
        return too_big(ir);
    }
    match execute(&query, ir) {
        Ok(result) => ok_result(&query, ir, result),
        Err(e) => err_result(&query, ir, e),
    }
}

fn err_result(query: &Query, ir: &CausalIR, err: BackendError) -> UserResponse {
    match err {
        BackendError::NodeNotFound(label) => not_found(&label),
        BackendError::Ambiguous { candidates } => {
            let label = match query {
                Query::Why { label, .. } | Query::What { label, .. } => label.clone(),
                _ => String::new(),
            };
            ambiguous(&label, candidates)
        }
        BackendError::NoPath(from, to) => {
            // Inatteignable via execute (found:false) — repli honnête.
            let (blocked, proches) = path_context(ir, &from, &to);
            no_path(&from, &to, blocked, proches, None)
        }
        other => incident(other.to_string()),
    }
}

/// Contexte d'un sans-chemin : niées de coupure + voisins (WHY-NOT : les bloqueurs).
fn path_context(ir: &CausalIR, from: &str, to: &str) -> (Vec<LinkDto>, Vec<String>) {
    let (_, blocked) = pearl::chain_topk(ir, from, to, 1);
    let blocked = blocked
        .iter()
        .map(|l| crate::query::link_to_dto(ir, l))
        .collect();
    (blocked, proches(ir, from, to))
}

fn no_path(
    from: &str,
    to: &str,
    blocked: Vec<LinkDto>,
    proches: Vec<String>,
    result: Option<QueryResult>,
) -> UserResponse {
    let mut suggestions = vec![format!("WHY {to}")];
    let (status, phrase) = if blocked.is_empty() {
        (
            Status::Partiel,
            say(
                Status::Partiel,
                "fr",
                &SpeechKind::NoPathEmpty {
                    from: from.to_string(),
                    to: to.to_string(),
                    proches,
                },
            ),
        )
    } else {
        let blockers: Vec<String> = blocked
            .iter()
            .take(5)
            .map(|l| format!("« {} → {} »", l.from, l.to))
            .collect();
        suggestions.push(format!("DO {}", blocked[0].to));
        (
            Status::Partiel,
            say(
                Status::Partiel,
                "fr",
                &SpeechKind::BlockedPath {
                    from: from.to_string(),
                    to: to.to_string(),
                    blockers,
                },
            ),
        )
    };
    UserResponse {
        status,
        phrase,
        lang: phrase_lang(),
        result,
        caveats: vec![],
        suggestions,
    }
}

fn ok_result(_query: &Query, ir: &CausalIR, result: QueryResult) -> UserResponse {
    match result {
        QueryResult::Causes {
            target,
            links,
            matched_targets,
            blocked_by,
            match_scores,
            fuzzy,
        } => {
            let mut head = format!("Causes de {target}");
            if !blocked_by.is_empty() {
                head.push_str(&format!(
                    " ({} liaison(s) niée(s) écartée(s))",
                    blocked_by.len()
                ));
            }
            let status = if fuzzy || !blocked_by.is_empty() {
                Status::Partiel
            } else {
                Status::Ok
            };
            let mut suggestions = vec![format!("WHAT {target}")];
            if fuzzy {
                suggestions.push("Match approximatif : précisez avec le label exact.".to_string());
            }
            let resp = answer(status, head, &links, suggestions);
            with_result(
                resp,
                QueryResult::Causes {
                    target,
                    links,
                    matched_targets,
                    blocked_by,
                    match_scores,
                    fuzzy,
                },
            )
        }
        QueryResult::Effects {
            source,
            links,
            matched_sources,
            blocked_by,
            match_scores,
            fuzzy,
        } => {
            let mut head = format!("Effets de {source}");
            if !blocked_by.is_empty() {
                head.push_str(&format!(
                    " ({} liaison(s) niée(s) écartée(s))",
                    blocked_by.len()
                ));
            }
            let status = if fuzzy || !blocked_by.is_empty() {
                Status::Partiel
            } else {
                Status::Ok
            };
            let resp = answer(status, head, &links, vec![format!("WHY {source}")]);
            with_result(
                resp,
                QueryResult::Effects {
                    source,
                    links,
                    matched_sources,
                    blocked_by,
                    match_scores,
                    fuzzy,
                },
            )
        }
        QueryResult::Path {
            from,
            to,
            found,
            links,
            blocked_by,
            path_confidence,
            ranked_paths,
        } => {
            if !found {
                let proches = proches(ir, &from, &to);
                let snapshot = QueryResult::Path {
                    from: from.clone(),
                    to: to.clone(),
                    found,
                    links,
                    blocked_by: blocked_by.clone(),
                    path_confidence,
                    ranked_paths,
                };
                return no_path(&from, &to, blocked_by, proches, Some(snapshot));
            }
            let head = format!("Chemin de {from} vers {to} (confiance {path_confidence:.2})");
            let resp = answer(Status::Ok, head, &links, vec![format!("WHY {to}")]);
            with_result(
                resp,
                QueryResult::Path {
                    from,
                    to,
                    found,
                    links,
                    blocked_by,
                    path_confidence,
                    ranked_paths,
                },
            )
        }
        QueryResult::TemporalPath {
            from,
            to,
            found,
            temporally_ordered,
            links,
        } => {
            if !found {
                let proches = proches(ir, &from, &to);
                let snapshot = QueryResult::TemporalPath {
                    from: from.clone(),
                    to: to.clone(),
                    found,
                    temporally_ordered,
                    links: vec![],
                };
                return no_path(&from, &to, vec![], proches, Some(snapshot));
            }
            let mut head = format!(
                "Chemin temporel de {from} vers {to} ({} saut(s))",
                links.len()
            );
            if !temporally_ordered {
                head.push_str(" — ordre temporel non respecté");
            }
            let status = if temporally_ordered {
                Status::Ok
            } else {
                Status::Partiel
            };
            let items: Vec<String> = links
                .iter()
                .take(MAX_ITEMS)
                .map(|l| format!("{} → {}", l.from, l.to))
                .collect();
            let extra = links.len().saturating_sub(items.len());
            let resp = answer_items(status, head, items, extra, vec![format!("WHY {to}")]);
            with_result(
                resp,
                QueryResult::TemporalPath {
                    from,
                    to,
                    found,
                    temporally_ordered,
                    links,
                },
            )
        }
        other => generic_ok(other),
    }
}

/// Réponse factuelle générique (requêtes non causales) : comptes + têtes.
fn generic_ok(result: QueryResult) -> UserResponse {
    match result {
        QueryResult::Intervention {
            target,
            severed_count,
            severed,
            effects,
            blocked_by,
        } => {
            let mut head =
                format!("Intervention sur {target} : {severed_count} cause(s) coupée(s)");
            if !blocked_by.is_empty() {
                head.push_str(&format!(
                    " ({} niée(s) en aval écartée(s))",
                    blocked_by.len()
                ));
            }
            let (items, extra) = dto_items(&effects);
            let suggestions = vec![format!("WHAT {target}"), format!("COUNTERFACTUAL {target}")];
            let status = if blocked_by.is_empty() {
                Status::Ok
            } else {
                Status::Partiel
            };
            let resp = answer_items(status, head, items, extra, suggestions);
            with_result(
                resp,
                QueryResult::Intervention {
                    target,
                    severed_count,
                    severed,
                    effects,
                    blocked_by,
                },
            )
        }
        QueryResult::CounterfactualDiff {
            target,
            actual_effects,
            unique_effects,
            blocked_by,
        } => {
            let head = if unique_effects.is_empty() {
                format!("Sans {target}, aucun effet ne disparaît")
            } else {
                format!("Sans {target}, disparaîtraient")
            };
            let items: Vec<String> = unique_effects.iter().take(MAX_ITEMS).cloned().collect();
            let extra = unique_effects.len().saturating_sub(items.len());
            let resp = answer_items(
                Status::Ok,
                head,
                items,
                extra,
                vec![format!("WHAT {target}")],
            );
            with_result(
                resp,
                QueryResult::CounterfactualDiff {
                    target,
                    actual_effects,
                    unique_effects,
                    blocked_by,
                },
            )
        }
        QueryResult::Abduction { effect, hypotheses } => {
            let head = format!("Explications possibles de {effect}");
            let items: Vec<String> = hypotheses
                .iter()
                .take(MAX_ITEMS)
                .map(|h| h.label.clone())
                .collect();
            let extra = hypotheses.len().saturating_sub(items.len());
            let resp = answer_items(
                Status::Ok,
                head,
                items,
                extra,
                vec![format!("WHY {effect}")],
            );
            with_result(resp, QueryResult::Abduction { effect, hypotheses })
        }
        QueryResult::AnalogyReport {
            pattern_from,
            pattern_to,
            matches,
        } => {
            let head = format!("Analogues de {pattern_from}→{pattern_to}");
            let items: Vec<String> = matches
                .iter()
                .take(MAX_ITEMS)
                .map(|m| format!("{}→{}", m.from_label, m.to_label))
                .collect();
            let extra = matches.len().saturating_sub(items.len());
            let suggestions = if matches.is_empty() {
                vec![format!(
                    "ANALOGY {pattern_from} -> {pattern_to} MIN_SCORE=0.1"
                )]
            } else {
                vec![]
            };
            let resp = answer_items(Status::Ok, head, items, extra, suggestions);
            with_result(
                resp,
                QueryResult::AnalogyReport {
                    pattern_from,
                    pattern_to,
                    matches,
                },
            )
        }
        QueryResult::CycleList { count, cycles } => {
            let head = format!("{count} cycle(s) détecté(s)");
            let items: Vec<String> = cycles.iter().take(3).map(|c| c.nodes.join(" → ")).collect();
            let extra = cycles.len().saturating_sub(items.len());
            let resp = answer_items(Status::Ok, head, items, extra, vec![]);
            with_result(resp, QueryResult::CycleList { count, cycles })
        }
        other => {
            let (head, n) = result_head_count(&other);
            let resp = answer_items(Status::Ok, head, vec![], n, vec![]);
            with_result(resp, other)
        }
    }
}

/// Têtes factuelles (comptes) pour les requêtes non causales.
fn result_head_count(result: &QueryResult) -> (String, usize) {
    match result {
        QueryResult::GapList {
            gap_edges,
            unresolved_nodes,
        } => (
            format!(
                "{} lacune(s) temporelle(s), {} nœud(s) non résolu(s)",
                gap_edges.len(),
                unresolved_nodes
            ),
            0,
        ),
        QueryResult::HierarchyZoomIn {
            parent_label,
            children,
        } => (
            format!("{} : {} enfant(s)", parent_label, children.len()),
            0,
        ),
        QueryResult::HierarchyZoomOut {
            child_label,
            parent_label,
        } => (
            match parent_label {
                Some(p) => format!("Parent de {child_label} : {p}"),
                None => format!("{child_label} est à la racine"),
            },
            0,
        ),
        QueryResult::HierarchyAggregate {
            parent_label,
            child_count,
            mean_confidence,
            ..
        } => (
            format!(
                "{parent_label} : {child_count} enfant(s), confiance moyenne {mean_confidence:.2}"
            ),
            0,
        ),
        QueryResult::CentralityReport {
            label,
            centrality_score,
            ..
        } => (format!("Centralité de {label} : {centrality_score:.2}"), 0),
        QueryResult::SpofReport { total_pairs, nodes } => (
            format!(
                "{} paire(s), {} nœud(s) critique(s)",
                total_pairs,
                nodes.len()
            ),
            0,
        ),
        QueryResult::NormativeDiff {
            from,
            to,
            gap_rate_pct,
            ..
        } => (
            format!("Écart normatif {from}→{to} : {gap_rate_pct:.1} %"),
            0,
        ),
        QueryResult::DensityReport {
            global_density,
            n_nodes,
            n_edges,
            ..
        } => (
            format!("Densité {global_density:.3} ({n_nodes} nœuds, {n_edges} arêtes)"),
            0,
        ),
        QueryResult::CoverageReport {
            label,
            coverage_score,
            ..
        } => (format!("Couverture de {label} : {coverage_score:.2}"), 0),
        QueryResult::ReliabilityReport {
            label,
            reliability_score,
            ..
        } => (format!("Fiabilité de {label} : {reliability_score:.2}"), 0),
        QueryResult::TemporalOrder {
            a, b, a_before_b, ..
        } => (format!("{a} avant {b} : {a_before_b}"), 0),
        QueryResult::TemporalDelay {
            from, to, found, ..
        } => (
            format!(
                "Délai {from}→{to} : {}",
                if *found { "estimé" } else { "introuvable" }
            ),
            0,
        ),
        _ => ("Résultat".to_string(), 0),
    }
}
