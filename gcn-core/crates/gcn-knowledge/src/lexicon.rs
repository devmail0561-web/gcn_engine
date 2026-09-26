// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::path::Path;

use gcn_ir::normalize_label;

use crate::KnowledgeError;
use crate::loader::load_all_taxonomies;
use crate::taxonomy::{LexicalEntry, Taxonomy, TaxonomyClass};

/// Table d'alias par domaine : variante normalisée → label canonique normalisé.
///
/// Utilisée par le SA avant ingestion pour fusionner les labels équivalents.
/// La table est définie dans la directive SA — le moteur fournit l'opérateur
/// de normalisation (`normalize_label`) mais ne gère pas les alias de domaine.
///
/// Exemple d'usage :
/// ```
/// use gcn_knowledge::AliasTable;
/// let mut t = AliasTable::new();
/// t.insert("auth", "authentification");
/// t.insert("login", "authentification");
/// assert_eq!(t.resolve("Login"), "authentification");
/// assert_eq!(t.resolve("inconnu"), "inconnu");
/// ```
#[derive(Debug, Default, Clone)]
pub struct AliasTable {
    aliases: HashMap<String, String>,
}

impl AliasTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enregistre `variant` comme alias de `canonical` (les deux sont normalisés à l'insertion).
    pub fn insert(&mut self, variant: &str, canonical: &str) {
        self.aliases
            .insert(normalize_label(variant), normalize_label(canonical));
    }

    /// Résout `label` vers son canonical normalisé, ou retourne `label` normalisé si inconnu.
    pub fn resolve(&self, label: &str) -> String {
        let norm = normalize_label(label);
        self.aliases.get(&norm).cloned().unwrap_or(norm)
    }

    /// Retourne vrai si `label` (après normalisation) est un alias connu.
    pub fn is_alias(&self, label: &str) -> bool {
        self.aliases.contains_key(&normalize_label(label))
    }

    pub fn len(&self) -> usize {
        self.aliases.len()
    }

    pub fn is_empty(&self) -> bool {
        self.aliases.is_empty()
    }
}

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

#[cfg(test)]
mod alias_tests {
    use super::*;

    #[test]
    fn alias_auth_login_authentification() {
        let mut t = AliasTable::new();
        t.insert("auth", "authentification");
        t.insert("login", "authentification");
        // Toutes les variantes résolvent vers le même canonical normalisé
        assert_eq!(t.resolve("auth"), "authentification");
        assert_eq!(t.resolve("Auth"), "authentification");
        assert_eq!(t.resolve("LOGIN"), "authentification");
        assert_eq!(t.resolve("login"), "authentification");
        // Label inconnu retourne lui-même normalisé
        assert_eq!(t.resolve("authentification"), "authentification");
        assert_eq!(t.resolve("Inconnu"), "inconnu");
    }

    #[test]
    fn alias_accented_variant() {
        let mut t = AliasTable::new();
        t.insert("économie", "economie");
        assert_eq!(t.resolve("Économie"), "economie");
        assert_eq!(t.resolve("economie"), "economie");
    }

    #[test]
    fn alias_empty_table() {
        let t = AliasTable::new();
        assert!(t.is_empty());
        assert_eq!(t.resolve("auth"), "auth");
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
