use crate::resources::LexicalResources;
use crate::rules::{is_imparfait, looks_like_verb_morphologically, lemmatize_verb};
use crate::tokenizer::Token;
use gcn_ir::Scope;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pos {
    Verb,
    Noun,
    Det,
    Pron,
    Conj,
    Prep,
    Adv,
    Punct,
    Other,
}

#[derive(Debug, Clone)]
pub struct TaggedToken {
    pub token: Token,
    pub pos: Pos,
    pub lemma: String,
    pub scope_hint: Option<Scope>,
    pub is_imparfait: bool,
    pub is_negation_particle: bool,
    pub is_negation_completer: bool,
}

pub fn tag(tokens: &[Token], res: &LexicalResources) -> Vec<TaggedToken> {
    // Two-pass: first classify all tokens, then apply positional heuristics.
    let mut tagged: Vec<TaggedToken> = tokens.iter().map(|t| classify_token(t, res)).collect();

    // Positional heuristic: if a non-verb content word immediately follows
    // a PRON (within 3 tokens, not blocked by another VERB), promote to VERB.
    for i in 0..tagged.len() {
        if matches!(tagged[i].pos, Pos::Noun | Pos::Other) {
            // Check for pronoun in the preceding 3 tokens
            let start = i.saturating_sub(3);
            let preceding = &tagged[start..i];
            let preceded_by_pron = preceding.iter().any(|t| t.pos == Pos::Pron);
            let no_verb_between = !preceding.iter().any(|t| t.pos == Pos::Verb);
            if preceded_by_pron && no_verb_between {
                // Don't promote known nouns (EtatSystemique etc.)
                let lower = tagged[i].token.lower.as_str();
                if !res.noun_node_types.contains_key(lower) {
                    let lemma = lemmatize_verb(lower);
                    tagged[i].pos = Pos::Verb;
                    tagged[i].lemma = lemma.clone();
                    tagged[i].is_imparfait = is_imparfait(&tagged[i].token.form);
                }
            }
        }
    }

    tagged
}

fn classify_token(t: &Token, res: &LexicalResources) -> TaggedToken {
    let lower = t.lower.as_str();
    let (pos, lemma) = classify(lower, res);
    let scope_hint = if pos == Pos::Det { res.det_scope.get(lower).copied() } else { None };
    let is_imp = pos == Pos::Verb && is_imparfait(&t.form);
    let is_neg_p = res.negation_particles.contains(lower);
    let is_neg_c = res.negation_completers.contains(lower);
    TaggedToken {
        token: t.clone(),
        pos,
        lemma,
        scope_hint,
        is_imparfait: is_imp,
        is_negation_particle: is_neg_p,
        is_negation_completer: is_neg_c,
    }
}

fn classify(lower: &str, res: &LexicalResources) -> (Pos, String) {
    // Punctuation
    if lower == "." || lower == "," || lower == ";" || lower == ":"
        || lower == "!" || lower == "?" {
        return (Pos::Punct, lower.to_string());
    }

    // Determiners first (before pronouns and verbs to avoid misclassification)
    if res.known_dets.contains(lower) {
        return (Pos::Det, lower.to_string());
    }

    // Pronouns
    if res.known_prons.contains(lower) {
        return (Pos::Pron, lower.to_string());
    }

    // Negation particles / completers (treated as ADV)
    if res.negation_particles.contains(lower) || res.negation_completers.contains(lower) {
        return (Pos::Adv, lower.to_string());
    }

    // Adverbs
    if res.known_advs.contains(lower) {
        return (Pos::Adv, lower.to_string());
    }

    // Conjunctions (single-word)
    if res.known_conjs.contains(lower) {
        return (Pos::Conj, lower.to_string());
    }

    // Prepositions (single-word)
    if res.known_preps.contains(lower) {
        return (Pos::Prep, lower.to_string());
    }

    // Try exact match in verb taxonomies (for infinitive forms already as lemmas)
    if res.verb_classes.contains_key(lower) || res.causal_verb_relations.contains_key(lower) {
        return (Pos::Verb, lower.to_string());
    }

    // Try lemmatizing and matching against verb taxonomies (handles inflected forms)
    let lemma = lemmatize_verb(lower);
    if lemma != lower {
        if res.verb_classes.contains_key(lemma.as_str())
            || res.causal_verb_relations.contains_key(lemma.as_str())
        {
            return (Pos::Verb, lemma);
        }
    }

    // Morphological verb detection (imparfait, past participle, present plural -ent)
    if looks_like_verb_morphologically(lower) {
        return (Pos::Verb, lemma);
    }

    // Known nouns from taxonomy
    if res.noun_node_types.contains_key(lower) {
        return (Pos::Noun, lower.to_string());
    }

    // Default: content word of sufficient length → Noun
    if lower.len() > 2 && !lower.ends_with('\'') {
        return (Pos::Noun, lower.to_string());
    }

    (Pos::Other, lower.to_string())
}
