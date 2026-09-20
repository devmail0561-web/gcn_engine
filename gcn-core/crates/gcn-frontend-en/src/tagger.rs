// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use crate::resources::LexicalResources;
use crate::rules::{is_past_tense, lemmatize_verb, looks_like_verb_morphologically};
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
    pub is_past: bool,
    pub is_negation_particle: bool,
    pub is_negation_completer: bool,
}

pub fn tag(tokens: &[Token], res: &LexicalResources) -> Vec<TaggedToken> {
    tokens.iter().map(|t| classify_token(t, res)).collect()
}

fn classify_token(t: &Token, res: &LexicalResources) -> TaggedToken {
    let lower = t.lower.as_str();
    let (pos, lemma) = classify(lower, res);
    let scope_hint = if pos == Pos::Det { res.det_scope.get(lower).copied() } else { None };
    let is_past = pos == Pos::Verb && is_past_tense(&t.form);
    let is_neg_p = res.negation_particles.contains(lower);
    let is_neg_c = res.negation_completers.contains(lower);
    TaggedToken {
        token: t.clone(),
        pos,
        lemma,
        scope_hint,
        is_past,
        is_negation_particle: is_neg_p,
        is_negation_completer: is_neg_c,
    }
}

fn classify(lower: &str, res: &LexicalResources) -> (Pos, String) {
    if matches!(lower, "." | "," | ";" | ":" | "!" | "?") {
        return (Pos::Punct, lower.to_string());
    }

    // Determiners first
    if res.known_dets.contains(lower) {
        return (Pos::Det, lower.to_string());
    }

    // Pronouns
    if res.known_prons.contains(lower) {
        return (Pos::Pron, lower.to_string());
    }

    // Negation particles / completers
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

    // Auxiliary forms
    if res.auxiliary_forms.contains(lower) {
        return (Pos::Verb, lower.to_string());
    }

    // Known verb base form
    if res.verb_classes.contains_key(lower) || res.causal_verb_relations.contains_key(lower) {
        return (Pos::Verb, lower.to_string());
    }

    // Lemmatize and retry
    let lemma = lemmatize_verb(lower);
    if lemma != lower
        && (res.verb_classes.contains_key(lemma.as_str())
            || res.causal_verb_relations.contains_key(lemma.as_str()))
    {
        return (Pos::Verb, lemma);
    }

    // Morphological verb detection
    if looks_like_verb_morphologically(lower) {
        return (Pos::Verb, lemma);
    }

    // Known nouns from taxonomy
    if res.noun_node_types.contains_key(lower) {
        return (Pos::Noun, lower.to_string());
    }

    // Default: content word
    if lower.len() > 2 && !lower.ends_with('\'') {
        return (Pos::Noun, lower.to_string());
    }

    (Pos::Other, lower.to_string())
}
