// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::path::Path;

use crate::KnowledgeError;
use crate::loader::load_all_taxonomies;
use crate::taxonomy::{LexicalEntry, Taxonomy, TaxonomyClass};

pub struct Lexicon {
    taxonomies: HashMap<String, Taxonomy>,
}

impl Lexicon {
    pub fn load_from_dir(path: &Path) -> Result<Self, KnowledgeError> {
        let taxonomies = load_all_taxonomies(path)?;
        Ok(Self { taxonomies })
    }

    pub fn lookup_verb(&self, lemma: &str) -> Option<&TaxonomyClass> {
        let taxonomy = self.taxonomies.get("verbes")?;
        taxonomy.classes.values().find(|class| {
            let in_fr = class
                .examples_fr
                .as_ref()
                .is_some_and(|ex| ex.iter().any(|e| e.lemma == lemma));
            let in_generic = class
                .examples
                .as_ref()
                .is_some_and(|ex| ex.iter().any(|e| e.lemma == lemma));
            in_fr || in_generic
        })
    }

    pub fn lookup_by_pos(&self, pos: &str, lemma: &str) -> Option<(&str, &LexicalEntry)> {
        let taxonomy_name = pos_to_taxonomy(pos)?;
        let taxonomy = self.taxonomies.get(taxonomy_name)?;
        for (class_name, class) in &taxonomy.classes {
            for examples in [&class.examples_fr, &class.examples].into_iter().flatten() {
                for entry in examples {
                    if entry.lemma == lemma {
                        return Some((class_name.as_str(), entry));
                    }
                }
            }
        }
        None
    }

    pub fn taxonomy(&self, name: &str) -> Option<&Taxonomy> {
        self.taxonomies.get(name)
    }

    pub fn taxonomies(&self) -> &HashMap<String, Taxonomy> {
        &self.taxonomies
    }

    #[cfg(test)]
    pub fn from_taxonomies_for_test(taxonomies: HashMap<String, Taxonomy>) -> Self {
        Self { taxonomies }
    }
}

fn pos_to_taxonomy(pos: &str) -> Option<&'static str> {
    match pos {
        "VERB" => Some("verbes"),
        "ADV" => Some("adverbes"),
        "ADJ" => Some("adjectifs"),
        "SCONJ" | "CCONJ" => Some("conjonctions"),
        "ADP" => Some("prepositions"),
        "NOUN" => Some("noms"),
        "PRON" => Some("pronoms"),
        "DET" => Some("determinants"),
        _ => None,
    }
}
