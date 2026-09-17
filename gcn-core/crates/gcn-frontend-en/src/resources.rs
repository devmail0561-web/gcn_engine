use std::collections::{HashMap, HashSet};
use std::path::Path;

const EN_INFINITIVE_MARKERS: &[&str] = &["to", "in order to", "so as to"];

use gcn_knowledge::{KnowledgeError, Lexicon};
use gcn_ir::{AgentType, NodeType, RelationType, Scope};

use crate::rules::{
    causal_direction_to_node_type, conjunction_class_to_direction, direction_str_to_enum,
    noun_class_to_node_type, pron_class_to_agent_type, relation_type_str_to_enum,
    scope_str_to_enum, MarkerDir,
};

#[derive(Debug, Clone)]
pub struct CausalMarkerEntry {
    pub lemma: String,
    pub words: Vec<String>,
    pub relation: RelationType,
    pub direction: MarkerDir,
    pub signals_gap: bool,
    pub requires_infinitive: bool,
}

pub struct LexicalResources {
    pub causal_markers: Vec<CausalMarkerEntry>,
    pub causal_verb_relations: HashMap<String, RelationType>,
    pub verb_classes: HashMap<String, NodeType>,
    pub det_scope: HashMap<String, Scope>,
    pub noun_node_types: HashMap<String, NodeType>,
    pub known_dets: HashSet<String>,
    pub known_prons: HashSet<String>,
    pub known_advs: HashSet<String>,
    pub known_preps: HashSet<String>,
    pub known_conjs: HashSet<String>,
    pub pron_agent_types: HashMap<String, AgentType>,
    pub negation_particles: HashSet<String>,
    pub negation_completers: HashSet<String>,
    pub nominalizations: HashMap<String, String>,
    pub auxiliary_forms: HashSet<String>,
}

impl LexicalResources {
    pub fn load(data_dir: &Path) -> Result<Self, KnowledgeError> {
        let lexicon = Lexicon::load_from_dir(data_dir)?;

        let causal_markers = build_causal_markers(&lexicon);
        let causal_verb_relations = build_causal_verb_relations(&lexicon);
        let verb_classes = build_verb_classes(&lexicon);
        let det_scope = build_det_scope(&lexicon);
        let (noun_node_types, _) = build_noun_tables(&lexicon);
        let (known_dets, det_scope_extra) = build_det_sets(&lexicon);
        let (known_prons, pron_agent_types) = build_pron_tables(&lexicon);
        let (known_advs, negation_particles, negation_completers) = build_adv_tables(&lexicon);
        let known_preps = build_prep_set(&lexicon);
        let known_conjs = build_conj_set(&lexicon);

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
// Helpers: read `examples` field (language-agnostic) from taxonomy classes
// ---------------------------------------------------------------------------

fn build_causal_markers(lexicon: &Lexicon) -> Vec<CausalMarkerEntry> {
    let mut markers: Vec<CausalMarkerEntry> = Vec::new();

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
            if let Some(examples) = &class.examples {
                for entry in examples {
                    let words = entry.lemma.split_whitespace()
                        .map(|w| w.to_lowercase()).collect();
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

    if let Some(tax) = lexicon.taxonomy("prepositions_causales") {
        for (class_name, class) in &tax.classes {
            let relation = match &class.relation_type {
                Some(rt) => match relation_type_str_to_enum(rt) {
                    Some(r) => r,
                    None => continue,
                },
                None => continue,
            };
            let direction_str = class.causal_direction.as_deref().unwrap_or("");
            let direction = if direction_str.is_empty() {
                match class_name.as_str() {
                    "motivation" => MarkerDir::GoalToAction,
                    _            => MarkerDir::Forward,
                }
            } else {
                direction_str_to_enum(direction_str)
            };
            let signals_gap = class.signals_gap.unwrap_or(false);
            if let Some(examples) = &class.examples {
                for entry in examples {
                    let words = entry.lemma.split_whitespace()
                        .map(|w| w.to_lowercase()).collect();
                    let requires_infinitive = EN_INFINITIVE_MARKERS.contains(&entry.lemma.as_str());
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

    markers.sort_by(|a, b| b.words.len().cmp(&a.words.len()).then(a.lemma.cmp(&b.lemma)));
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
            if let Some(examples) = &class.examples {
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
            if let Some(examples) = &class.examples {
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
            if let Some(examples) = &class.examples {
                for entry in examples {
                    let scope = entry.scope.as_deref()
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
            if let Some(examples) = &class.examples {
                for entry in examples {
                    let lower = entry.lemma.to_lowercase();
                    set.insert(lower.clone());
                    let scope = entry.scope.as_deref()
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
            if let Some(examples) = &class.examples {
                for entry in examples {
                    let lower = entry.lemma.to_lowercase();
                    known.insert(lower.clone());
                    if let Some(nt) = node_type {
                        type_map.insert(lower, nt);
                    }
                }
            }
            // Handle subtypes (stored as examples_fr in noms.yaml per existing pattern)
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
            if let Some(examples) = &class.examples {
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
            if let Some(examples) = &class.examples {
                for entry in examples {
                    let lower = entry.lemma.to_lowercase();
                    all_advs.insert(lower.clone());
                    if class_name == "negation" {
                        neg_completers.insert(lower.clone());
                    }
                }
            }
        }
    }
    // English negation particle is always "not" / "n't"
    neg_particles.insert("not".to_string());
    neg_particles.insert("n't".to_string());
    (all_advs, neg_particles, neg_completers)
}

fn build_prep_set(lexicon: &Lexicon) -> HashSet<String> {
    let mut set = HashSet::new();
    if let Some(tax) = lexicon.taxonomy("prepositions") {
        for class in tax.classes.values() {
            if let Some(examples) = &class.examples {
                for entry in examples {
                    set.insert(entry.lemma.to_lowercase());
                }
            }
        }
    }
    if let Some(tax) = lexicon.taxonomy("prepositions_causales") {
        for class in tax.classes.values() {
            if let Some(examples) = &class.examples {
                for entry in examples {
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
            if let Some(examples) = &class.examples {
                for entry in examples {
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
            if let Some(examples) = &class.examples {
                for entry in examples {
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
        && let Some(examples) = &aux_class.examples
    {
        for entry in examples {
            set.insert(entry.lemma.to_lowercase());
        }
    }
    set
}
