use std::collections::HashMap;
use std::path::Path;

use crate::loader::load_all_taxonomies;
use crate::taxonomy::{LexicalEntry, Taxonomy, TaxonomyClass};
use crate::KnowledgeError;

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
            class
                .examples_fr
                .as_ref()
                .is_some_and(|examples| examples.iter().any(|e| e.lemma == lemma))
        })
    }

    pub fn lookup_by_pos(&self, pos: &str, lemma: &str) -> Option<(&str, &LexicalEntry)> {
        let taxonomy_name = pos_to_taxonomy(pos)?;
        let taxonomy = self.taxonomies.get(taxonomy_name)?;
        for (class_name, class) in &taxonomy.classes {
            if let Some(examples) = &class.examples_fr {
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
