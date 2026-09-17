use std::collections::HashMap;
use std::path::Path;

const DELTA_STRONG_CAUSAL: f32 = 0.15;
const DELTA_CAUSAL: f32 = 0.10;
const DELTA_ADVERSATIVE: f32 = -0.10;

use gcn_ir::{
    ir::CausalIR,
    node::{NodeId, NodeOrigin, NodeType},
    edge::RelationType,
    scope::Scope,
};

use crate::lexicon::Lexicon;
use crate::KnowledgeError;

// ─────────────────────────────────────────────────────────────────────────────

/// Résultat d'une opération d'enrichissement sur un `CausalIR`.
#[derive(Debug, Clone, PartialEq)]
pub enum InferenceNote {
    /// Un type de nœud inféré a été corrigé par la taxonomie.
    NodeTypeResolved {
        node_id: NodeId,
        from: NodeType,
        to: NodeType,
        rule: String,
    },
    /// La confiance d'une arête implicite a été recalibrée.
    ConfidenceAdjusted {
        from_id: NodeId,
        to_id: NodeId,
        from: f32,
        to: f32,
        rule: String,
    },
    /// Une arête signale une lacune causale (Concession ou Opposition).
    CausalGapSignaled {
        from_id: NodeId,
        to_id: NodeId,
        relation: RelationType,
    },
}

// ─────────────────────────────────────────────────────────────────────────────

pub struct InferenceEngine {
    lexicon: Lexicon,
}

impl InferenceEngine {
    /// Construit l'engine depuis un `Lexicon` déjà chargé.
    pub fn new(lexicon: Lexicon) -> Self {
        Self { lexicon }
    }

    /// Construit l'engine en chargeant les taxonomies depuis un répertoire.
    pub fn from_dir(path: &Path) -> Result<Self, KnowledgeError> {
        let lexicon = Lexicon::load_from_dir(path)?;
        Ok(Self { lexicon })
    }

    /// Infère le `NodeType` d'un lemme depuis les taxonomies.
    ///
    /// Retourne `None` si le POS est hors périmètre (ex : `auxiliaire`)
    /// ou si le lemme est inconnu.
    pub fn infer_node_type(&self, lemma: &str, pos: &str) -> Option<NodeType> {
        let (class_name, _entry) = self.lexicon.lookup_by_pos(pos, lemma)?;
        class_name_to_node_type(class_name)
    }

    /// Infère le `Scope` d'un token DET ou PRON depuis les taxonomies.
    ///
    /// À appeler au moment du parse, quand le token est connu.
    /// L'annotation explicite `scope` de l'entrée lexicale a priorité
    /// sur le fallback par nom de classe.
    pub fn infer_scope(&self, lemma: &str, pos: &str) -> Option<Scope> {
        let (class_name, entry) = self.lexicon.lookup_by_pos(pos, lemma)?;
        if let Some(scope_str) = &entry.scope
            && let Some(scope) = parse_scope(scope_str)
        {
            return Some(scope);
        }
        class_name_to_scope(class_name, pos)
    }

    /// Score de confiance structurel basé sur la compatibilité des types.
    ///
    /// Base : `1.0` si arête explicite, `0.5` sinon.
    /// Un bonus/malus est appliqué selon la combinaison (from, to, relation).
    /// Résultat clampé dans `[0.1, 1.0]`.
    pub fn score_confidence(
        &self,
        from: NodeType,
        to: NodeType,
        rel: RelationType,
        explicit: bool,
    ) -> f32 {
        let base: f32 = if explicit { 1.0 } else { 0.5 };
        let delta: f32 = match (from, to, rel) {
            (NodeType::Action, NodeType::Etat, RelationType::Cause)
            | (NodeType::Action, NodeType::Transition, RelationType::Cause) => DELTA_STRONG_CAUSAL,
            (NodeType::Processus, NodeType::Etat, RelationType::Cause) => DELTA_CAUSAL,
            (NodeType::Transition, _, RelationType::Cause) => DELTA_CAUSAL,
            (NodeType::EtatSystemique, _, RelationType::Filter) => DELTA_STRONG_CAUSAL,
            (_, _, RelationType::Concession | RelationType::Opposition) => DELTA_ADVERSATIVE,
            _ => 0.0,
        };
        (base + delta).clamp(0.1, 1.0)
    }

    /// Enrichit un `CausalIR` en 3 passes borrow-safe et retourne les notes produites.
    ///
    /// - **Passe 1** : résolution des types de nœuds avec `origin == Inferred`
    ///   via `attributes.entity` (lookup NOUN dans les taxonomies).
    /// - **Passe 2** : snapshot `NodeId → NodeType` post-passe 1.
    /// - **Passe 3** : recalibrage de confiance des arêtes implicites ;
    ///   signalement des lacunes causales (Concession / Opposition).
    pub fn enrich(&self, ir: &mut CausalIR) -> Vec<InferenceNote> {
        let mut notes = Vec::new();

        // Passe 1 — types de nœuds inférés
        for node in &mut ir.nodes {
            if node.origin != NodeOrigin::Inferred {
                continue;
            }
            let entity = match node.attributes.entity.as_deref() {
                Some(e) if !e.is_empty() => e,
                _ => continue,
            };
            if let Some(inferred) = self.infer_node_type(entity, "NOUN")
                && inferred != node.node_type
            {
                notes.push(InferenceNote::NodeTypeResolved {
                    node_id: node.id,
                    from: node.node_type,
                    to: inferred,
                    rule: format!("taxonomy_lookup:noun:{entity}"),
                });
                node.node_type = inferred;
            }
        }

        // Passe 2 — snapshot post-passe 1 (évite le borrow mutable simultané)
        let node_types: HashMap<NodeId, NodeType> =
            ir.nodes.iter().map(|n| (n.id, n.node_type)).collect();

        // Passe 3 — arêtes
        for (src_id, dst_id, edge) in &mut ir.edges {
            if edge.relation.signals_causal_gap() {
                notes.push(InferenceNote::CausalGapSignaled {
                    from_id: *src_id,
                    to_id: *dst_id,
                    relation: edge.relation,
                });
            }

            if !edge.explicit {
                let from_nt = node_types.get(src_id).copied();
                let to_nt = node_types.get(dst_id).copied();
                if let (Some(f), Some(t)) = (from_nt, to_nt) {
                    let new_conf = self.score_confidence(f, t, edge.relation, false);
                    if (new_conf - edge.confidence).abs() > 0.05 {
                        notes.push(InferenceNote::ConfidenceAdjusted {
                            from_id: *src_id,
                            to_id: *dst_id,
                            from: edge.confidence,
                            to: new_conf,
                            rule: "structural_compatibility".to_string(),
                        });
                        edge.confidence = new_conf;
                    }
                }
            }
        }

        notes
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers internes

fn class_name_to_node_type(class_name: &str) -> Option<NodeType> {
    match class_name {
        "etat"       => Some(NodeType::Etat),
        "action"     => Some(NodeType::Action),
        "transition" => Some(NodeType::Transition),
        "processus"  => Some(NodeType::Processus),
        "agent" | "patient" | "abstrait" | "relation" => Some(NodeType::Entite),
        "etat_systemique" => Some(NodeType::EtatSystemique),
        _ => None,  // "auxiliaire" et autres POS non causaux
    }
}

fn class_name_to_scope(class_name: &str, pos: &str) -> Option<Scope> {
    match pos {
        "DET" => match class_name {
            "defini" | "demonstratif" | "possessif" => Some(Scope::Specific),
            "indefini"   => Some(Scope::Existential),
            "partitif"   => Some(Scope::Partial),
            "quantitatif" => Some(Scope::Universal),
            _ => Some(Scope::Unknown),
        },
        "PRON" => match class_name {
            "personnel" | "demonstratif" | "relatif" => Some(Scope::Specific),
            "indefini" => Some(Scope::Unknown),
            _ => None,
        },
        _ => None,
    }
}

fn parse_scope(s: &str) -> Option<Scope> {
    match s {
        "universal"   => Some(Scope::Universal),
        "existential" => Some(Scope::Existential),
        "partial"     => Some(Scope::Partial),
        "null"        => Some(Scope::Null),
        "specific"    => Some(Scope::Specific),
        "unknown"     => Some(Scope::Unknown),
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use gcn_ir::{
        CausalEdge, CausalIR, CausalNode, IrMetadata, NaturalLanguage, NodeAttributes, NodeId,
        NodeOrigin, NodeType, RelationType, Scope, SourceLanguage, SourceSpan, TemporalRef,
    };
    use smallvec::SmallVec;

    use crate::taxonomy::{LexicalEntry, Taxonomy, TaxonomyClass};

    // ── Helpers de construction synthétique ──────────────────────────────────

    fn make_entry(lemma: &str) -> LexicalEntry {
        LexicalEntry {
            lemma: lemma.to_string(),
            note: None, value: None, kind: None,
            anchor: None, scope: None, total: None,
        }
    }

    fn make_entry_with_scope(lemma: &str, scope: &str) -> LexicalEntry {
        LexicalEntry {
            lemma: lemma.to_string(),
            scope: Some(scope.to_string()),
            note: None, value: None, kind: None,
            anchor: None, total: None,
        }
    }

    fn make_class(examples: Vec<LexicalEntry>) -> TaxonomyClass {
        TaxonomyClass {
            description: None, role: None, causal_direction: None,
            causal_effect: None, dimension: None, relation_type: None,
            signals_gap: None, scope: None, formal: None, position: None,
            examples_fr: Some(examples), examples: None, subtypes: None,
        }
    }

    fn make_taxonomy(name: &str, classes: Vec<(&str, TaxonomyClass)>) -> Taxonomy {
        Taxonomy {
            taxonomy: name.to_string(),
            version: "test".to_string(),
            description: "test".to_string(),
            classes: classes.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
            compositional_rules: None,
        }
    }

    fn test_engine() -> InferenceEngine {
        let mut taxonomies: HashMap<String, Taxonomy> = HashMap::new();

        taxonomies.insert("verbes".to_string(), make_taxonomy("verbes", vec![
            ("etat",       make_class(vec![make_entry("être")])),
            ("action",     make_class(vec![make_entry("faire")])),
            ("transition", make_class(vec![make_entry("chuter")])),
            ("processus",  make_class(vec![make_entry("baisser")])),
            ("auxiliaire", make_class(vec![make_entry("avoir")])),
        ]));

        taxonomies.insert("noms".to_string(), make_taxonomy("noms", vec![
            ("agent",           make_class(vec![make_entry("agent")])),
            ("patient",         make_class(vec![make_entry("marché")])),
            ("processus",       make_class(vec![make_entry("croissance")])),
            ("etat_systemique", make_class(vec![make_entry("crise")])),
            ("abstrait",        make_class(vec![make_entry("idée")])),
            ("relation",        make_class(vec![make_entry("lien")])),
        ]));

        taxonomies.insert("determinants".to_string(), make_taxonomy("determinants", vec![
            ("defini",     make_class(vec![make_entry("le")])),
            ("indefini",   make_class(vec![make_entry("un")])),
            ("partitif",   make_class(vec![make_entry("du")])),
            ("quantitatif", make_class(vec![
                make_entry_with_scope("tous",  "universal"),
                make_entry_with_scope("aucun", "null"),
            ])),
        ]));

        taxonomies.insert("pronoms".to_string(), make_taxonomy("pronoms", vec![
            ("personnel",    make_class(vec![make_entry("il")])),
            ("demonstratif", make_class(vec![make_entry("cela")])),
            ("relatif",      make_class(vec![make_entry("qui")])),
            ("indefini",     make_class(vec![make_entry("on")])),
        ]));

        let lex = Lexicon::from_taxonomies_for_test(taxonomies);
        InferenceEngine::new(lex)
    }

    fn test_node(id: u32, nt: NodeType, origin: NodeOrigin) -> CausalNode {
        CausalNode {
            id: NodeId(id),
            node_type: nt,
            label: format!("node_{id}"),
            source_span: SourceSpan::Synthetic,
            scope: Scope::Unknown,
            modifiers: SmallVec::new(),
            temporal_ref: TemporalRef::Unresolved,
            temporal_index: None,
            origin,
            attributes: NodeAttributes::default(),
        }
    }

    fn test_node_with_entity(id: u32, nt: NodeType, origin: NodeOrigin, entity: &str) -> CausalNode {
        let mut n = test_node(id, nt, origin);
        n.attributes.entity = Some(entity.to_string());
        n
    }

    fn test_edge(
        src: u32, dst: u32,
        rel: RelationType,
        explicit: bool,
        confidence: f32,
    ) -> (NodeId, NodeId, CausalEdge) {
        (NodeId(src), NodeId(dst), CausalEdge {
            relation: rel,
            confidence,
            temporal_gap: None,
            explicit,
            negated: false,
            marker_token: None,
            in_cycle: None,
        })
    }

    fn make_ir(
        nodes: Vec<CausalNode>,
        edges: Vec<(NodeId, NodeId, CausalEdge)>,
    ) -> CausalIR {
        CausalIR {
            source_lang: SourceLanguage::Natural { lang: NaturalLanguage::French },
            source_text: String::new(),
            nodes,
            edges,
            cycles: vec![],
            unresolved: vec![],
            metadata: IrMetadata::default(),
        }
    }

    // ── infer_node_type ──────────────────────────────────────────────────────

    #[test]
    fn infer_verb_processus() {
        assert_eq!(test_engine().infer_node_type("baisser", "VERB"), Some(NodeType::Processus));
    }

    #[test]
    fn infer_verb_action() {
        assert_eq!(test_engine().infer_node_type("faire", "VERB"), Some(NodeType::Action));
    }

    #[test]
    fn infer_verb_etat() {
        assert_eq!(test_engine().infer_node_type("être", "VERB"), Some(NodeType::Etat));
    }

    #[test]
    fn infer_verb_transition() {
        assert_eq!(test_engine().infer_node_type("chuter", "VERB"), Some(NodeType::Transition));
    }

    #[test]
    fn infer_auxiliaire_returns_none() {
        // auxiliaire n'est pas un opérateur causal
        assert_eq!(test_engine().infer_node_type("avoir", "VERB"), None);
    }

    #[test]
    fn infer_noun_processus() {
        assert_eq!(test_engine().infer_node_type("croissance", "NOUN"), Some(NodeType::Processus));
    }

    #[test]
    fn infer_noun_agent_is_entite() {
        assert_eq!(test_engine().infer_node_type("agent", "NOUN"), Some(NodeType::Entite));
    }

    #[test]
    fn infer_noun_etat_systemique() {
        assert_eq!(test_engine().infer_node_type("crise", "NOUN"), Some(NodeType::EtatSystemique));
    }

    #[test]
    fn infer_unknown_lemma_returns_none() {
        assert_eq!(test_engine().infer_node_type("zzz_inconnu", "VERB"), None);
    }

    // ── infer_scope ──────────────────────────────────────────────────────────

    #[test]
    fn infer_scope_det_defini() {
        assert_eq!(test_engine().infer_scope("le", "DET"), Some(Scope::Specific));
    }

    #[test]
    fn infer_scope_det_indefini() {
        assert_eq!(test_engine().infer_scope("un", "DET"), Some(Scope::Existential));
    }

    #[test]
    fn infer_scope_det_quantitatif_universal() {
        // entry.scope == "universal" prend la priorité sur le fallback de classe
        assert_eq!(test_engine().infer_scope("tous", "DET"), Some(Scope::Universal));
    }

    #[test]
    fn infer_scope_det_quantitatif_null() {
        assert_eq!(test_engine().infer_scope("aucun", "DET"), Some(Scope::Null));
    }

    #[test]
    fn infer_scope_pron_personnel() {
        assert_eq!(test_engine().infer_scope("il", "PRON"), Some(Scope::Specific));
    }

    #[test]
    fn infer_scope_pron_indefini() {
        assert_eq!(test_engine().infer_scope("on", "PRON"), Some(Scope::Unknown));
    }

    // ── score_confidence ─────────────────────────────────────────────────────

    #[test]
    fn score_explicit_action_etat_cause_clamps_to_1() {
        // base 1.0 + bonus 0.15 = 1.15 → clamp → 1.0
        assert_eq!(
            test_engine().score_confidence(NodeType::Action, NodeType::Etat, RelationType::Cause, true),
            1.0,
        );
    }

    #[test]
    fn score_implicit_action_etat_cause() {
        // base 0.5 + bonus 0.15 = 0.65
        let s = test_engine().score_confidence(
            NodeType::Action, NodeType::Etat, RelationType::Cause, false,
        );
        assert!((s - 0.65).abs() < 1e-6);
    }

    #[test]
    fn score_implicit_concession_reduces_confidence() {
        // base 0.5 − 0.10 = 0.40
        let s = test_engine().score_confidence(
            NodeType::Etat, NodeType::Action, RelationType::Concession, false,
        );
        assert!((s - 0.40).abs() < 1e-6);
    }

    #[test]
    fn score_never_below_min() {
        let s = test_engine().score_confidence(
            NodeType::Entite, NodeType::Entite, RelationType::Opposition, false,
        );
        assert!(s >= 0.1);
    }

    // ── enrich ───────────────────────────────────────────────────────────────

    #[test]
    fn enrich_signals_causal_gap_on_concession() {
        let e = test_engine();
        let n1 = test_node(1, NodeType::Processus, NodeOrigin::Explicit);
        let n2 = test_node(2, NodeType::Etat, NodeOrigin::Explicit);
        let mut ir = make_ir(vec![n1, n2], vec![
            test_edge(1, 2, RelationType::Concession, true, 1.0),
        ]);
        let notes = e.enrich(&mut ir);
        assert!(notes.iter().any(|n| matches!(n,
            InferenceNote::CausalGapSignaled { relation: RelationType::Concession, .. }
        )));
    }

    #[test]
    fn enrich_resolves_inferred_node_type() {
        let e = test_engine();
        // Nœud Inferred de type Etat (incorrect), entity "croissance" → Processus
        let n1 = test_node_with_entity(1, NodeType::Etat, NodeOrigin::Inferred, "croissance");
        let n2 = test_node(2, NodeType::Action, NodeOrigin::Explicit);
        let mut ir = make_ir(vec![n1, n2], vec![
            test_edge(1, 2, RelationType::Cause, true, 1.0),
        ]);
        let notes = e.enrich(&mut ir);
        assert!(notes.iter().any(|n| matches!(n,
            InferenceNote::NodeTypeResolved { from: NodeType::Etat, to: NodeType::Processus, .. }
        )));
        assert_eq!(ir.nodes[0].node_type, NodeType::Processus);
    }

    #[test]
    fn enrich_adjusts_confidence_for_implicit_edge() {
        let e = test_engine();
        let n1 = test_node(1, NodeType::Action, NodeOrigin::Explicit);
        let n2 = test_node(2, NodeType::Etat, NodeOrigin::Explicit);
        // arête implicite Action→Etat Cause, confiance initiale 0.5 → score 0.65
        let mut ir = make_ir(vec![n1, n2], vec![
            test_edge(1, 2, RelationType::Cause, false, 0.5),
        ]);
        let notes = e.enrich(&mut ir);
        assert!(notes.iter().any(|n| matches!(n, InferenceNote::ConfidenceAdjusted { .. })));
        let (_, _, ref updated) = ir.edges[0];
        assert!((updated.confidence - 0.65).abs() < 1e-6);
    }

    #[test]
    fn enrich_does_not_modify_explicit_nodes() {
        let e = test_engine();
        // Explicit même si entity connue → pas de modification
        let n1 = test_node_with_entity(1, NodeType::Etat, NodeOrigin::Explicit, "croissance");
        let mut ir = make_ir(vec![n1], vec![]);
        let notes = e.enrich(&mut ir);
        assert!(!notes.iter().any(|n| matches!(n, InferenceNote::NodeTypeResolved { .. })));
    }
}
