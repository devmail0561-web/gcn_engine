use gcn_ir::{NodeType, RelationType, Scope, AgentType};
use unicode_normalization::UnicodeNormalization;

// ---------------------------------------------------------------------------
// Type mappings: YAML string values → Rust enum variants
// These are the only things that may be hardcoded (per architecture rules).
// ---------------------------------------------------------------------------

/// Map a taxonomy `causal_direction` string (from verbes.yaml) → NodeType.
pub fn causal_direction_to_node_type(dir: &str) -> Option<NodeType> {
    match dir {
        "maintien"    => Some(NodeType::Etat),
        "production"  => Some(NodeType::Action),
        "rupture"     => Some(NodeType::Transition),
        "propagation" => Some(NodeType::Processus),
        _             => None,
    }
}

/// Map a taxonomy `relation_type` string → RelationType.
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

/// Map a taxonomy `scope` string → Scope.
pub fn scope_str_to_enum(s: &str) -> Option<Scope> {
    match s {
        "specific"     => Some(Scope::Specific),
        "existential"  => Some(Scope::Existential),
        "partial"      => Some(Scope::Partial),
        "null"         => Some(Scope::Null),
        "universal"    => Some(Scope::Universal),
        "unknown"      => Some(Scope::Unknown),
        _              => None,
    }
}

/// Derive edge direction from a taxonomy `direction` field or relation_type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerDir {
    /// Left clause is effect, right clause is cause (e.g. "parce que")
    Backward,
    /// Left clause is cause, right clause is effect (e.g. "donc", "si")
    Forward,
    /// Edge reversed: right (goal) → left (action) (e.g. "pour")
    GoalToAction,
}

pub fn direction_str_to_enum(s: &str) -> MarkerDir {
    match s {
        "backward" => MarkerDir::Backward,
        "goal_to_action" => MarkerDir::GoalToAction,
        _ => MarkerDir::Forward,
    }
}

/// For taxonomy class names in conjonctions.yaml, derive the marker direction.
/// This is a type-level mapping: class name → causal direction.
pub fn conjunction_class_to_direction(class_name: &str) -> MarkerDir {
    match class_name {
        // "parce que" etc: right clause is the cause → left is the effect
        "cause" => MarkerDir::Backward,
        // "bien que" etc: right clause is the concession → left is the surprising result
        "concession" => MarkerDir::Backward,
        _ => MarkerDir::Forward,
    }
}

/// Map pronoun taxonomy class → AgentType.
pub fn pron_class_to_agent_type(class_name: &str, lemma: &str) -> AgentType {
    match class_name {
        "indefini" if lemma == "on" => AgentType::Collective,
        "personnel" | "demonstratif" | "indefini" => AgentType::Human,
        _ => AgentType::Human,
    }
}

/// Map noun taxonomy class → NodeType (for named nouns in noun position).
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
// Morphological heuristics (no lexical data — purely form-based)
// ---------------------------------------------------------------------------

/// Detect imparfait tense from verb form endings.
pub fn is_imparfait(form: &str) -> bool {
    let f = form.to_lowercase();
    f.ends_with("ait") || f.ends_with("aient") || f.ends_with("ais")
}

/// Detect infinitive form.
pub fn is_infinitive(form: &str) -> bool {
    let f = form.to_lowercase();
    f.ends_with("er") || f.ends_with("ir") || f.ends_with("re") || f.ends_with("oir")
}

/// Attempt to lemmatize a verb form to its infinitive using morphological rules.
/// For regular verbs only; irregular forms should be in the taxonomy.
pub fn lemmatize_verb(form: &str) -> String {
    // Normaliser NFC pour que strip_suffix/ends_with fonctionnent sur les accents
    // composés (NFD e+U+0301 vs NFC U+00E9) quelle que soit la source du tokenizer.
    let f: String = form.to_lowercase().nfc().collect();

    // Strip reflexive clitic from "s'effondre" → "effondre"
    let f = f.strip_prefix("s'").unwrap_or(f.as_str());

    // Passé composé participle: -é/-ée → stem + er
    if let Some(stripped) = f.strip_suffix("ée") {
        return format!("{}er", stripped);
    }
    if f.ends_with("és") || f.ends_with("ées") {
        let end = if f.ends_with("ées") { 4 } else { 3 };
        return format!("{}er", &f[..f.len()-end]);
    }
    if let Some(stripped) = f.strip_suffix('é') {
        return format!("{}er", stripped);
    }

    // Passé composé 2nd group: -i → stem + ir
    if f.ends_with('i') && f.len() > 3
        && !["qui", "si", "merci", "aussi", "demi", "semi"].contains(&f)
    {
        let stem = &f[..f.len()-1];
        // Disambiguate: if already ends in -rir/-fir etc, just return
        if !stem.ends_with('r') {
            return format!("{}ir", stem);
        }
    }

    // Imparfait: -ait → stem + er; -aient → stem + er
    if let Some(stripped) = f.strip_suffix("aient") {
        return format!("{}er", stripped);
    }
    if let Some(stripped) = f.strip_suffix("ait") {
        return format!("{}er", stripped);
    }
    if let Some(stripped) = f.strip_suffix("ais") {
        return format!("{}er", stripped);
    }

    // Present 3rd plural -ent: baissent → baisser
    if f.ends_with("ent") && f.len() > 4 {
        let stem = &f[..f.len()-3];
        // Avoid matching "lent", "vent" etc.
        if stem.len() > 2 {
            return format!("{}er", stem);
        }
    }

    // Present 3rd singular -e: baisse → baisser, travaille → travailler
    if f.ends_with('e') && f.len() > 3 {
        // Avoid matching determiners and conjunctions (short tokens)
        let stem = &f[..f.len()-1];
        if stem.len() > 3 {
            return format!("{}er", stem);
        }
    }

    // Already an infinitive
    if is_infinitive(f) {
        return f.to_string();
    }

    f.to_string()
}

/// Nominalize a verb using a pre-loaded lookup table (from nominalizations.yaml).
/// Falls back to the lemma itself for regular or unknown verbs.
pub fn nominalize_with_table<'a>(verb_lemma: &'a str, table: &'a std::collections::HashMap<String, String>) -> &'a str {
    table.get(verb_lemma).map(|s| s.as_str()).unwrap_or(verb_lemma)
}

/// Heuristic: does this token look like a verb form?
/// Based purely on morphological patterns, not a word list.
pub fn looks_like_verb_morphologically(form: &str) -> bool {
    let f = form.to_lowercase();
    is_imparfait(&f)
        || f.ends_with("ent")   // 3rd plural present
        || f.ends_with("é")     // past participle -er verbs
        || is_infinitive(&f)
}

#[cfg(test)]
mod tests {
    use super::lemmatize_verb;

    #[test]
    fn lemmatize_non_verb_i_words_unchanged() {
        for w in ["merci", "aussi", "demi", "semi"] {
            assert_eq!(lemmatize_verb(w), w, "'{}' ne doit pas être lemmatisé", w);
        }
    }

    #[test]
    fn lemmatize_real_participes_ir() {
        assert_eq!(lemmatize_verb("fini"), "finir");
        assert_eq!(lemmatize_verb("parti"), "partir");
        assert_eq!(lemmatize_verb("choisi"), "choisir");
    }
}
