// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

//! Verbalizer de secours déterministe (Ph8 [R-ALWAYS]).
//!
//! Pur Rust, std uniquement (+ serde déjà dépendance de la crate, aucune
//! ajoutée), testé sans Python : phrase FR (autres langues à la demande)
//! qui ne MENT jamais — constat + miettes réelles + suggestions.
//! Voie noble = décodeur appris (`decode`) quand il fonctionne.

use std::fmt::Write as _;

/// Statut de l'enveloppe always-answer : ok | partiel | degrade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Ok,
    Partiel,
    Degrade,
}

/// Cas de phrase : données réelles portées par l'appelant (jamais inventées).
#[derive(Debug, Clone)]
pub enum SpeechKind {
    /// Réponse complète : tête factuelle + items réels (plafonnés par l'appelant).
    Answer {
        head: String,
        items: Vec<String>,
        extra: usize,
    },
    /// Label introuvable (NodeNotFound).
    NotFound { label: String },
    /// Requête ambiguë : candidats classés + question.
    Ambiguous {
        label: String,
        candidates: Vec<String>,
        more: usize,
    },
    /// Chemin existant mais coupé par négation (WHY-NOT : les bloqueurs).
    BlockedPath {
        from: String,
        to: String,
        blockers: Vec<String>,
    },
    /// Vraiment aucun chemin : voisins pour rebondir.
    NoPathEmpty {
        from: String,
        to: String,
        proches: Vec<String>,
    },
    /// Graphe trop gros (> 500 nœuds) : partiel + plus connectés.
    TooBig {
        nodes: usize,
        top: Vec<(String, usize)>,
    },
    /// Requête incompréhensible : aide + éventuel « vouliez-vous dire ».
    BadQuery { hint: Option<String> },
    /// Graphe vide.
    EmptyIr,
    /// Incident interne (erreur non mappable).
    Incident { detail: String },
}

fn join_items(items: &[String]) -> String {
    items.join(", ")
}

/// Verbalise un cas en français (1-3 phrases). `status` voyage dans
/// l'enveloppe ; la formulation exprime déjà la sévérité. `lang` sélectionnera
/// les gabarits quand ils existeront (Ph8 : FR d'abord, autres à la demande).
pub fn say(_status: Status, lang: &str, kind: &SpeechKind) -> String {
    let _ = lang;
    match kind {
        SpeechKind::Answer { head, items, extra } => {
            if items.is_empty() {
                return format!("{head} : aucun.");
            }
            let mut s = format!("{} : {}.", head, join_items(items));
            if *extra > 0 {
                let _ = write!(s, " Et {extra} autres.");
            }
            s
        }
        SpeechKind::NotFound { label } => format!(
            "« {label} » est introuvable dans le graphe. Essayez un mot plus court ou \
             sans faute : la recherche floue accepte les préfixes et approximations."
        ),
        SpeechKind::Ambiguous {
            label,
            candidates,
            more,
        } => {
            let mut s = format!(
                "« {label} » désigne plusieurs notions : {}. Laquelle ?",
                join_items(candidates)
            );
            if *more > 0 {
                let _ = write!(s, " Et {more} autres.");
            }
            s
        }
        SpeechKind::BlockedPath { from, to, blockers } => {
            let coupent = blockers.len() > 1;
            format!(
                "Aucun chemin de {from} vers {to} : {} {} la propagation \
                 (liaison niée). Essayez « WHY {to} » pour d'autres causes.",
                join_items(blockers),
                if coupent { "coupent" } else { "coupe" },
            )
        }
        SpeechKind::NoPathEmpty { from, to, proches } => {
            if proches.is_empty() {
                format!(
                    "Aucun chemin de {from} vers {to}, et aucun voisin direct. \
                     Essayez « WHY {to} »."
                )
            } else {
                format!(
                    "Aucun chemin de {from} vers {to}. Proches : {}. Essayez « WHY {to} ».",
                    join_items(proches)
                )
            }
        }
        SpeechKind::TooBig { nodes, top } => {
            let tops: Vec<String> = top.iter().map(|(l, d)| format!("{l} ({d})")).collect();
            format!(
                "Graphe trop gros ({nodes} nœuds) : réponse partielle. Les plus \
                 connectés : {}. Essayez « CENTRALITY <mot> » pour détailler.",
                join_items(&tops)
            )
        }
        SpeechKind::BadQuery { hint } => {
            let mut s = String::from("Requête incomprise.");
            if let Some(h) = hint {
                let _ = write!(s, " {h}");
            }
            s.push_str(" Requêtes valides : WHY <mot>, WHAT <mot>, CHAIN <a> -> <b>, DO <mot>.");
            s
        }
        SpeechKind::EmptyIr => String::from(
            "Aucune donnée chargée (graphe vide). Chargez un graphe puis relancez la requête.",
        ),
        SpeechKind::Incident { detail } => format!(
            "Incident interne : impossible de répondre ({detail}). Réessayez ; \
             si ça persiste, signalez la requête."
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_kinds() -> Vec<SpeechKind> {
        vec![
            SpeechKind::Answer {
                head: "Causes de Gamma".into(),
                items: vec!["Alpha → Beta".into()],
                extra: 2,
            },
            SpeechKind::Answer {
                head: "Effets de Alpha".into(),
                items: vec![],
                extra: 0,
            },
            SpeechKind::NotFound {
                label: "zzz".into(),
            },
            SpeechKind::Ambiguous {
                label: "vent".into(),
                candidates: vec!["ventes".into(), "ventilation".into()],
                more: 3,
            },
            SpeechKind::BlockedPath {
                from: "A".into(),
                to: "C".into(),
                blockers: vec!["« A → B »".into()],
            },
            SpeechKind::NoPathEmpty {
                from: "A".into(),
                to: "Z".into(),
                proches: vec![],
            },
            SpeechKind::NoPathEmpty {
                from: "A".into(),
                to: "Z".into(),
                proches: vec!["B".into()],
            },
            SpeechKind::TooBig {
                nodes: 501,
                top: vec![("A".into(), 12)],
            },
            SpeechKind::BadQuery { hint: None },
            SpeechKind::BadQuery {
                hint: Some("Vouliez-vous dire « WHY … » ?".into()),
            },
            SpeechKind::EmptyIr,
            SpeechKind::Incident { detail: "x".into() },
        ]
    }

    #[test]
    fn all_kinds_render_non_empty_fr() {
        for k in all_kinds() {
            for status in [Status::Ok, Status::Partiel, Status::Degrade] {
                let s = say(status, "fr", &k);
                assert!(!s.is_empty(), "phrase vide pour {k:?}");
                assert!(s.ends_with('.') || s.ends_with('?'), "ponctuation : {s}");
            }
        }
    }

    #[test]
    fn deterministic_and_total_on_adversarial_input() {
        let weird = "日本語 🎉 \0 ".repeat(50);
        let kinds = vec![
            SpeechKind::NotFound {
                label: weird.clone(),
            },
            SpeechKind::Ambiguous {
                label: weird.clone(),
                candidates: vec![weird.clone(); 100],
                more: 95,
            },
            SpeechKind::Incident {
                detail: weird.clone(),
            },
        ];
        for k in kinds {
            let a = say(Status::Degrade, "fr", &k);
            let b = say(Status::Degrade, "fr", &k);
            assert_eq!(a, b);
            assert!(!a.is_empty());
        }
    }

    #[test]
    fn answer_lists_items_and_remainder() {
        let s = say(
            Status::Ok,
            "fr",
            &SpeechKind::Answer {
                head: "Causes de Gamma".into(),
                items: vec!["Alpha".into(), "Beta".into()],
                extra: 0,
            },
        );
        assert!(s.contains("Alpha") && s.contains("Beta"), "{s}");
        let s = say(
            Status::Ok,
            "fr",
            &SpeechKind::Answer {
                head: "Causes de Gamma".into(),
                items: vec!["Alpha".into()],
                extra: 4,
            },
        );
        assert!(s.contains("Et 4 autres"), "{s}");
    }
}
