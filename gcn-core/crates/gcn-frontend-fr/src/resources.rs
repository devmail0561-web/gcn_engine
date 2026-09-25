// Copyright 2026 Michel Tendeng
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::path::Path;

const FR_INFINITIVE_MARKERS: &[&str] = &["pour"];

use gcn_ir::{AgentType, NodeType, RelationType, Scope};
use gcn_knowledge::{KnowledgeError, Lexicon};

use crate::rules::{
    MarkerDir, causal_direction_to_node_type, conjunction_class_to_direction,
    direction_str_to_enum, noun_class_to_node_type, pron_class_to_agent_type,
    relation_type_str_to_enum, scope_str_to_enum,
};

// ---------------------------------------------------------------------------
// CausalMarkerEntry: a causal connector with its semantic properties
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CausalMarkerEntry {
    /// Surface lemma, possibly multi-word (e.g. "parce que").
    pub lemma: String,
    /// Words in the lemma (pre-split for fast matching).
    pub words: Vec<String>,
    pub relation: RelationType,
    pub direction: MarkerDir,
    pub signals_gap: bool,
    /// Whether "pour" requires a following infinitive.
    pub requires_infinitive: bool,
}

// ---------------------------------------------------------------------------
// LexicalResources: all preloaded tables
// ---------------------------------------------------------------------------

pub struct LexicalResources {
    /// Causal markers sorted longest-first (multi-word then single-word).
    pub causal_markers: Vec<CausalMarkerEntry>,
    /// Causal verb connectors: lemma → RelationType.
    pub causal_verb_relations: HashMap<String, RelationType>,
    /// Verb classification: lemma → NodeType.
    pub verb_classes: HashMap<String, NodeType>,
    /// Determiner scope: lemma → Scope.
    pub det_scope: HashMap<String, Scope>,
    /// Noun classes: lemma → NodeType (for EtatSystemique etc.).
    pub noun_node_types: HashMap<String, NodeType>,
    /// Known POS sets (for tagging).
    pub known_dets: HashSet<String>,
    pub known_prons: HashSet<String>,
    pub known_advs: HashSet<String>,
    pub known_preps: HashSet<String>,
    pub known_conjs: HashSet<String>,
    /// Pronoun → AgentType.
    pub pron_agent_types: HashMap<String, AgentType>,
    /// Negation particles (subset of adverbs).
    pub negation_particles: HashSet<String>,
    /// Adverbs that signal negation when combined with "ne/n'".
    pub negation_completers: HashSet<String>,
    /// Verb lemma → nominalized form for label generation.
    pub nominalizations: HashMap<String, String>,
    /// Auxiliary verb forms — do not create CIR nodes.
    pub auxiliary_forms: HashSet<String>,
}

impl LexicalResources {
    pub fn load(data_dir: &Path) -> Result<Self, KnowledgeError> {
        let lexicon = Lexicon::load_from_dir(data_dir)?;

        let causal_markers = build_causal_markers(&lexicon);
        let causal_verb_relations = build_causal_verb_relations(&lexicon);
        let verb_classes = build_verb_classes(&lexicon);
        let det_scope = build_det_scope(&lexicon);
        let (noun_node_types, _known_nouns_all) = build_noun_tables(&lexicon);
        let (known_dets, det_scope_extra) = build_det_sets(&lexicon);
        let (known_prons, pron_agent_types) = build_pron_tables(&lexicon);
        let (known_advs, negation_particles, negation_completers) = build_adv_tables(&lexicon);
        let known_preps = build_prep_set(&lexicon);
        let known_conjs = build_conj_set(&lexicon);

        // Merge det_scope from taxonomy scan
        let mut final_det_scope = det_scope;
        final_det_scope.extend(det_scope_extra);

        let nominalizations = build_nominalizations(&lexicon);
        let auxiliary_forms = build_auxiliary_forms(&lexicon);

        Ok(LexicalResources {
            causal_markers,
            causal_verb_relations,
            verb_classes,
            det_scope: final_det_scope,
            noun_node_types,
            known_dets,
            known_prons,
            known_advs,
            known_preps,
            known_conjs,
            pron_agent_types,
            negation_particles,
            negation_completers,
            nominalizations,
            auxiliary_forms,
        })
    }
}

// ---------------------------------------------------------------------------
// Builder functions — all data comes from YAML via Lexicon
// ---------------------------------------------------------------------------

fn build_causal_markers(lexicon: &Lexicon) -> Vec<CausalMarkerEntry> {
    let mut markers: Vec<CausalMarkerEntry> = Vec::new();

    // --- From conjonctions.yaml ---
    if let Some(tax) = lexicon.taxonomy("conjonctions") {
        for (class_name, class) in &tax.classes {
            let relation = match &class.relation_type {
                Some(rt) => match relation_type_str_to_enum(rt) {
                    Some(r) => r,
                    None => continue,
                },
                None => continue,
            };
            let direction = conjunction_class_to_direction(class_name);
            let signals_gap = class.signals_gap.unwrap_or(false);

            if let Some(examples) = &class.examples_fr {
                for entry in examples {
                    let words: Vec<String> = entry
                        .lemma
                        .split_whitespace()
                        .map(|w| w.to_lowercase())
                        .collect();
                    markers.push(CausalMarkerEntry {
                        lemma: entry.lemma.clone(),
                        words,
                        relation,
                        direction,
                        signals_gap,
                        requires_infinitive: false,
                    });
                }
            }
        }
    }

    // --- From prepositions_causales.yaml ---
    if let Some(tax) = lexicon.taxonomy("prepositions_causales") {
        for (class_name, class) in &tax.classes {
            let relation = match &class.relation_type {
                Some(rt) => match relation_type_str_to_enum(rt) {
                    Some(r) => r,
                    None => continue,
                },
                None => continue,
            };
            // Direction comes from class-level field if present, else derive from class name
            let direction_str = class.causal_direction.as_deref().unwrap_or("");
            let direction = if direction_str.is_empty() {
                // Check if class has a custom direction field via relation_type field position
                // For "motivation" class in prepositions_causales.yaml, direction = goal_to_action
                match class_name.as_str() {
                    "motivation" => MarkerDir::GoalToAction,
                    _ => MarkerDir::Forward,
                }
            } else {
                direction_str_to_enum(direction_str)
            };

            let signals_gap = class.signals_gap.unwrap_or(false);

            if let Some(examples) = &class.examples_fr {
                for entry in examples {
                    let words: Vec<String> = entry
                        .lemma
                        .split_whitespace()
                        .map(|w| w.to_lowercase())
                        .collect();
                    // "pour" requires infinitive
                    let requires_infinitive = FR_INFINITIVE_MARKERS.contains(&entry.lemma.as_str());
                    markers.push(CausalMarkerEntry {
                        lemma: entry.lemma.clone(),
                        words,
                        relation,
                        direction,
                        signals_gap,
                        requires_infinitive,
                    });
                }
            }
        }
    }

    // Sort: longest (most words) first for greedy matching
    markers.sort_by(|a, b| {
        b.words
            .len()
            .cmp(&a.words.len())
            .then(a.lemma.cmp(&b.lemma))
    });
    markers
}

fn build_causal_verb_relations(lexicon: &Lexicon) -> HashMap<String, RelationType> {
    let mut map = HashMap::new();
    if let Some(tax) = lexicon.taxonomy("verbes_causaux") {
        for class in tax.classes.values() {
            let relation = match &class.relation_type {
                Some(rt) => match relation_type_str_to_enum(rt) {
                    Some(r) => r,
                    None => continue,
                },
                None => continue,
            };
            if let Some(examples) = &class.examples_fr {
                for entry in examples {
                    map.insert(entry.lemma.clone(), relation);
                }
            }
        }
    }
    map
}

fn build_verb_classes(lexicon: &Lexicon) -> HashMap<String, NodeType> {
    let mut map = HashMap::new();
    if let Some(tax) = lexicon.taxonomy("verbes") {
        for class in tax.classes.values() {
            let node_type = match &class.causal_direction {
                Some(dir) => match causal_direction_to_node_type(dir) {
                    Some(nt) => nt,
                    None => continue,
                },
                None => continue,
            };
            if let Some(examples) = &class.examples_fr {
                for entry in examples {
                    map.insert(entry.lemma.clone(), node_type);
                }
            }
        }
    }
    map
}

fn build_det_scope(lexicon: &Lexicon) -> HashMap<String, Scope> {
    let mut map = HashMap::new();
    if let Some(tax) = lexicon.taxonomy("determinants") {
        for class in tax.classes.values() {
            let class_scope = class.scope.as_deref().and_then(scope_str_to_enum);
            if let Some(examples) = &class.examples_fr {
                for entry in examples {
                    let scope = entry
                        .scope
                        .as_deref()
                        .and_then(scope_str_to_enum)
                        .or(class_scope);
                    if let Some(s) = scope {
                        map.insert(entry.lemma.to_lowercase(), s);
                    }
                }
            }
        }
    }
    map
}

fn build_det_sets(lexicon: &Lexicon) -> (HashSet<String>, HashMap<String, Scope>) {
    let mut set = HashSet::new();
    let mut scope_map = HashMap::new();
    if let Some(tax) = lexicon.taxonomy("determinants") {
        for class in tax.classes.values() {
            let class_scope = class.scope.as_deref().and_then(scope_str_to_enum);
            if let Some(examples) = &class.examples_fr {
                for entry in examples {
                    let lower = entry.lemma.to_lowercase();
                    set.insert(lower.clone());
                    let scope = entry
                        .scope
                        .as_deref()
                        .and_then(scope_str_to_enum)
                        .or(class_scope);
                    if let Some(s) = scope {
                        scope_map.insert(lower, s);
                    }
                }
            }
        }
    }
    (set, scope_map)
}

fn build_noun_tables(lexicon: &Lexicon) -> (HashMap<String, NodeType>, HashSet<String>) {
    let mut type_map = HashMap::new();
    let mut known = HashSet::new();
    if let Some(tax) = lexicon.taxonomy("noms") {
        for (class_name, class) in &tax.classes {
            let node_type = noun_class_to_node_type(class_name);
            if let Some(examples) = &class.examples_fr {
                for entry in examples {
                    let lower = entry.lemma.to_lowercase();
                    known.insert(lower.clone());
                    if let Some(nt) = node_type {
                        type_map.insert(lower, nt);
                    }
                }
            }
            // Also check subtypes for agent subtypes
            if let Some(subtypes) = &class.subtypes {
                for subtype in subtypes.values() {
                    if let Some(examples) = &subtype.examples_fr {
                        for lemma in examples {
                            known.insert(lemma.to_lowercase());
                        }
                    }
                }
            }
        }
    }
    (type_map, known)
}

fn build_pron_tables(lexicon: &Lexicon) -> (HashSet<String>, HashMap<String, AgentType>) {
    let mut set = HashSet::new();
    let mut agent_map = HashMap::new();
    if let Some(tax) = lexicon.taxonomy("pronoms") {
        for (class_name, class) in &tax.classes {
            if let Some(examples) = &class.examples_fr {
                for entry in examples {
                    let lower = entry.lemma.to_lowercase();
                    set.insert(lower.clone());
                    let at = pron_class_to_agent_type(class_name, &lower);
                    agent_map.insert(lower, at);
                }
            }
        }
    }
    (set, agent_map)
}

fn build_adv_tables(lexicon: &Lexicon) -> (HashSet<String>, HashSet<String>, HashSet<String>) {
    let mut all_advs = HashSet::new();
    let mut neg_particles = HashSet::new();
    let mut neg_completers = HashSet::new();
    if let Some(tax) = lexicon.taxonomy("adverbes") {
        for (class_name, class) in &tax.classes {
            if let Some(examples) = &class.examples_fr {
                for entry in examples {
                    let lower = entry.lemma.to_lowercase();
                    all_advs.insert(lower.clone());
                    if class_name == "negation" {
                        // "ne...pas" etc. — split into particles
                        // The lemma "ne...pas" → two tokens "ne" and "pas"
                        if lower.contains("...") {
                            let parts: Vec<&str> = lower.split("...").collect();
                            if let Some(p0) = parts.first() {
                                neg_particles.insert(p0.to_string());
                            }
                            if let Some(p1) = parts.get(1) {
                                neg_completers.insert(p1.to_string());
                            }
                        } else {
                            neg_completers.insert(lower.clone());
                        }
                    }
                }
            }
        }
    }
    // Always include "ne" and "n'" as negation starters (morphological apostrophe clitic)
    neg_particles.insert("ne".to_string());
    neg_particles.insert("n'".to_string());
    (all_advs, neg_particles, neg_completers)
}

fn build_prep_set(lexicon: &Lexicon) -> HashSet<String> {
    let mut set = HashSet::new();
    if let Some(tax) = lexicon.taxonomy("prepositions") {
        for class in tax.classes.values() {
            if let Some(examples) = &class.examples_fr {
                for entry in examples {
                    set.insert(entry.lemma.to_lowercase());
                }
            }
        }
    }
    // Also add causal prep lemmas
    if let Some(tax) = lexicon.taxonomy("prepositions_causales") {
        for class in tax.classes.values() {
            if let Some(examples) = &class.examples_fr {
                for entry in examples {
                    // Only single-word prepositions go in the set
                    let lower = entry.lemma.to_lowercase();
                    if !lower.contains(' ') {
                        set.insert(lower);
                    }
                }
            }
        }
    }
    set
}

fn build_conj_set(lexicon: &Lexicon) -> HashSet<String> {
    let mut set = HashSet::new();
    if let Some(tax) = lexicon.taxonomy("conjonctions") {
        for class in tax.classes.values() {
            if let Some(examples) = &class.examples_fr {
                for entry in examples {
                    // Only single-word conjunctions
                    let lower = entry.lemma.to_lowercase();
                    if !lower.contains(' ') {
                        set.insert(lower);
                    }
                }
            }
        }
    }
    set
}

fn build_nominalizations(lexicon: &Lexicon) -> HashMap<String, String> {
    let mut map = HashMap::new();
    if let Some(tax) = lexicon.taxonomy("nominalizations") {
        for class in tax.classes.values() {
            if let Some(examples) = &class.examples_fr {
                for entry in examples {
                    // The `note` field holds the nominalized form
                    if let Some(nom) = &entry.note {
                        map.insert(entry.lemma.to_lowercase(), nom.clone());
                    }
                }
            }
        }
    }
    map
}

fn build_auxiliary_forms(lexicon: &Lexicon) -> HashSet<String> {
    let mut set = HashSet::new();
    if let Some(tax) = lexicon.taxonomy("verbes")
        && let Some(aux_class) = tax.classes.get("auxiliaire")
        && let Some(examples) = &aux_class.examples_fr
    {
        for entry in examples {
            set.insert(entry.lemma.to_lowercase());
        }
    }
    set
}
