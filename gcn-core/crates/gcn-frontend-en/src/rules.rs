// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use gcn_ir::{AgentType, NodeType, RelationType, Scope};

// ---------------------------------------------------------------------------
// Type mappings: YAML string values → Rust enum variants
// ---------------------------------------------------------------------------

pub fn causal_direction_to_node_type(dir: &str) -> Option<NodeType> {
    match dir {
        "maintien"    => Some(NodeType::Etat),
        "production"  => Some(NodeType::Action),
        "rupture"     => Some(NodeType::Transition),
        "propagation" => Some(NodeType::Processus),
        _             => None,
    }
}

pub fn relation_type_str_to_enum(rt: &str) -> Option<RelationType> {
    match rt {
        "cause_directe" | "cause"              => Some(RelationType::Cause),
        "effet_direct"                          => Some(RelationType::Cause),
        "cause_conditionnelle" | "condition"   => Some(RelationType::Condition),
        "cause_attendue_non_réalisée" |
        "concession"                            => Some(RelationType::Concession),
        "séquence_temporelle" | "sequence"      => Some(RelationType::Sequence),
        "contraste_causal" | "opposition"       => Some(RelationType::Opposition),
        "enable"                                => Some(RelationType::Enable),
        "prevent"                               => Some(RelationType::Prevent),
        "motivation"                            => Some(RelationType::Motivation),
        _                                       => None,
    }
}

pub fn scope_str_to_enum(s: &str) -> Option<Scope> {
    match s {
        "specific"    => Some(Scope::Specific),
        "existential" => Some(Scope::Existential),
        "partial"     => Some(Scope::Partial),
        "null"        => Some(Scope::Null),
        "universal"   => Some(Scope::Universal),
        "unknown"     => Some(Scope::Unknown),
        _             => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerDir {
    Backward,
    Forward,
    GoalToAction,
}

pub fn direction_str_to_enum(s: &str) -> MarkerDir {
    match s {
        "backward"       => MarkerDir::Backward,
        "goal_to_action" => MarkerDir::GoalToAction,
        _                => MarkerDir::Forward,
    }
}

pub fn conjunction_class_to_direction(class_name: &str) -> MarkerDir {
    match class_name {
        "cause" => MarkerDir::Backward,
        _       => MarkerDir::Forward,
    }
}

pub fn pron_class_to_agent_type(class_name: &str, lemma: &str) -> AgentType {
    match class_name {
        "indefini" if lemma == "one" => AgentType::Collective,
        _ => AgentType::Human,
    }
}

pub fn noun_class_to_node_type(class_name: &str) -> Option<NodeType> {
    match class_name {
        "etat_systemique" => Some(NodeType::EtatSystemique),
        "processus"       => Some(NodeType::Processus),
        "agent"           => Some(NodeType::Entite),
        "patient"         => Some(NodeType::Entite),
        _                 => None,
    }
}

// ---------------------------------------------------------------------------
// English morphological heuristics
// ---------------------------------------------------------------------------

pub fn is_past_tense(form: &str) -> bool {
    let f = form.to_lowercase();
    f.ends_with("ed") && f.len() > 3
}

pub fn is_infinitive_or_base(form: &str) -> bool {
    // Most English base forms don't end in common inflection suffixes
    let f = form.to_lowercase();
    !f.ends_with("ing") && !f.ends_with("ed") && !f.ends_with("s")
}

/// Attempt to lemmatize an English verb form to its base form.
pub fn lemmatize_verb(form: &str) -> String {
    let f = form.to_lowercase();

    // Negation contraction: "doesn't" → "does"
    if let Some(stripped) = f.strip_suffix("n't") {
        return stripped.to_string();
    }

    // Past: -ied → -y ("carried" → "carry")
    if let Some(stripped) = f.strip_suffix("ied") {
        return format!("{}y", stripped);
    }

    // Past: -ed (double consonant: "stopped" → "stop")
    if let Some(stripped) = f.strip_suffix("ed")
        && stripped.len() > 2 {
        let chars: Vec<char> = stripped.chars().collect();
        let n = chars.len();
        if n >= 2 && chars[n-1] == chars[n-2] {
            return chars[..n-1].iter().collect();
        }
        return stripped.to_string();
    }

    // Progressive: -ing (double consonant: "running" → "run")
    if let Some(stripped) = f.strip_suffix("ing")
        && stripped.len() > 2 {
        let chars: Vec<char> = stripped.chars().collect();
        let n = chars.len();
        if n >= 2 && chars[n-1] == chars[n-2] {
            return chars[..n-1].iter().collect();
        }
        return stripped.to_string();
    }

    // 3rd person: -ies → -y ("carries" → "carry")
    if let Some(stripped) = f.strip_suffix("ies") {
        return format!("{}y", stripped);
    }

    // 3rd person: bare -s strip (must come before -es):
    // "causes" → "cause", "enables" → "enable", "prevents" → "prevent"
    // This is correct for verbs whose base form ends with a vowel (caus→e, enabl→e).
    if let Some(stripped) = f.strip_suffix('s')
        && stripped.len() > 2 {
        return stripped.to_string();
    }

    f.to_string()
}

/// Nominalize an English verb using the pre-loaded table.
pub fn nominalize_with_table<'a>(
    verb_lemma: &'a str,
    table: &'a std::collections::HashMap<String, String>,
) -> &'a str {
    table.get(verb_lemma).map(|s| s.as_str()).unwrap_or(verb_lemma)
}

/// Heuristic: does this token look like an English verb?
pub fn looks_like_verb_morphologically(form: &str) -> bool {
    let f = form.to_lowercase();
    f.ends_with("ing")
        || (f.ends_with("ed") && f.len() > 3)
        || f.ends_with("ize")
        || f.ends_with("ise")
        || f.ends_with("ify")
        || f.ends_with("ate")
}
