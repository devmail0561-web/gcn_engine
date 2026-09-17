use crate::resources::{CausalMarkerEntry, LexicalResources};
use crate::rules::{MarkerDir, nominalize_with_table, is_infinitive, is_imparfait};
use crate::tagger::{TaggedToken, Pos};
use gcn_ir::{AgentType, NodeOrigin, NodeType, RelationType, Scope};

const FR_UNIVERSAL_SUBJECT_LEMMAS: &[&str] = &["on"];
const FR_DURATIVE_MARKERS: &[&str] = &["depuis"];

// ---------------------------------------------------------------------------
// Clause and edge annotations produced by the annotator
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ClauseAnnotation {
    pub span: (u32, u32),
    pub node_type: NodeType,
    pub label: String,
    pub scope: Scope,
    pub origin: NodeOrigin,
    pub agent: Option<String>,
    pub patient: Option<String>,
    pub agent_type: Option<AgentType>,
    pub entity: Option<String>,
    pub quality: Option<String>,
    pub has_depuis: bool,
    pub is_imparfait: bool,
    pub neg_on_node: bool,
}

#[derive(Debug, Clone)]
pub struct EdgeAnnotation {
    pub src_clause: usize,
    pub dst_clause: usize,
    pub relation: RelationType,
    pub confidence: f32,
    pub explicit: bool,
    pub negated: bool,
    pub marker_token_idx: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct SentenceAnnotation {
    pub clauses: Vec<ClauseAnnotation>,
    pub edges: Vec<EdgeAnnotation>,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn annotate(tagged: &[TaggedToken], res: &LexicalResources) -> SentenceAnnotation {
    let sentences = split_sentences(tagged);

    if sentences.len() == 1 {
        annotate_sentence(sentences[0], res)
    } else {
        let mut all_clauses: Vec<ClauseAnnotation> = Vec::new();
        let mut all_edges: Vec<EdgeAnnotation> = Vec::new();
        let mut sentence_clause_counts: Vec<usize> = Vec::with_capacity(sentences.len());

        for sent_tokens in &sentences {
            let offset = all_clauses.len();
            let mut ann = annotate_sentence(sent_tokens, res);
            sentence_clause_counts.push(ann.clauses.len());
            for e in &mut ann.edges {
                e.src_clause += offset;
                e.dst_clause += offset;
            }
            all_clauses.extend(ann.clauses);
            all_edges.extend(ann.edges);
        }

        // Implicit cause between last clause of sentence N and first clause of sentence N+1.
        // Skip hypothetical nodes (cause_cachée) — they are not real sentence subjects.
        let mut offset = 0usize;
        for i in 0..sentences.len().saturating_sub(1) {
            offset += sentence_clause_counts[i];
            if sentence_clause_counts[i] > 0 && sentence_clause_counts[i + 1] > 0 {
                let start = offset - sentence_clause_counts[i];
                let src_clause = (start..offset).rev()
                    .find(|&idx| all_clauses[idx].origin != NodeOrigin::Hypothetical)
                    .unwrap_or(offset - 1);
                all_edges.push(EdgeAnnotation {
                    src_clause,
                    dst_clause: offset,
                    relation: RelationType::Cause,
                    confidence: 0.5,
                    explicit: false,
                    negated: false,
                    marker_token_idx: None,
                });
            }
        }

        SentenceAnnotation { clauses: all_clauses, edges: all_edges }
    }
}

fn split_sentences(tokens: &[TaggedToken]) -> Vec<&[TaggedToken]> {
    let mut result = Vec::new();
    let mut start = 0;
    for (i, t) in tokens.iter().enumerate() {
        if t.token.lower == "." {
            if i > start {
                result.push(&tokens[start..i]);
            }
            start = i + 1;
        }
    }
    if start < tokens.len() {
        result.push(&tokens[start..]);
    }
    if result.is_empty() {
        result.push(tokens);
    }
    result
}

/// Strip leading/trailing commas and punctuation from a token slice.
fn strip_punct_ends(tokens: &[TaggedToken]) -> &[TaggedToken] {
    let start = tokens.iter().position(|t| t.token.lower != "," && t.token.lower != ".").unwrap_or(0);
    let end = tokens.iter().rposition(|t| t.token.lower != "," && t.token.lower != ".").map(|i| i + 1).unwrap_or(tokens.len());
    if start < end { &tokens[start..end] } else { tokens }
}

// ---------------------------------------------------------------------------
// Single-sentence annotation
// ---------------------------------------------------------------------------

fn annotate_sentence(tokens: &[TaggedToken], res: &LexicalResources) -> SentenceAnnotation {
    if tokens.is_empty() {
        return SentenceAnnotation { clauses: vec![], edges: vec![] };
    }

    let lower_seq: Vec<&str> = tokens.iter().map(|t| t.token.lower.as_str()).collect();

    // Normalize "qu'" → "que" for marker matching
    let norm_lower: Vec<String> = lower_seq.iter().map(|s| normalize_clitic(s)).collect();
    let norm_refs: Vec<&str> = norm_lower.iter().map(|s| s.as_str()).collect();

    if let Some(ann) = try_causal_verb(tokens, res) {
        return ann;
    }
    if let Some(ann) = try_causal_marker(tokens, &norm_refs, res) {
        return ann;
    }

    // Single clause
    let clause = build_clause(tokens, res);
    SentenceAnnotation { clauses: vec![clause], edges: vec![] }
}

/// Normalize apostrophe clitics for marker matching:
/// "qu'" → "que", "n'" → "ne", etc.
fn normalize_clitic(s: &str) -> String {
    if s.ends_with('\'') || s.ends_with('\u{2019}') {
        let base = s.trim_end_matches(['\'', '\u{2019}']);
        match base.to_lowercase().as_str() {
            "qu" => "que".to_string(),
            "n"  => "ne".to_string(),
            "l"  => "le".to_string(),
            "j"  => "je".to_string(),
            "m"  => "me".to_string(),
            "t"  => "te".to_string(),
            "s"  => "se".to_string(),
            "c"  => "ce".to_string(),
            "d"  => "de".to_string(),
            _    => s.to_lowercase(),
        }
    } else {
        s.to_lowercase()
    }
}

// ---------------------------------------------------------------------------
// Strategy 1: causal verb (e.g. entraîne, causé)
// ---------------------------------------------------------------------------

fn try_causal_verb(tokens: &[TaggedToken], res: &LexicalResources) -> Option<SentenceAnnotation> {
    for (i, t) in tokens.iter().enumerate() {
        if t.pos != Pos::Verb {
            continue;
        }
        // Check both the surface lemma and the lowercased form
        let candidate_lemmas = [t.lemma.as_str(), t.token.lower.as_str()];
        for lemma in candidate_lemmas {
            if let Some(&relation) = res.causal_verb_relations.get(lemma) {
                let negated = has_negation_around(tokens, i);

                // Everything before verb → left clause (source)
                // Everything after verb → right clause (target)
                if i == 0 || i + 1 >= tokens.len() {
                    break;
                }

                let left_clause = build_clause(&tokens[..i], res);
                let right_clause = build_clause(&tokens[i + 1..], res);

                let edge = EdgeAnnotation {
                    src_clause: 0,
                    dst_clause: 1,
                    relation,
                    confidence: 1.0,
                    explicit: true,
                    negated,
                    marker_token_idx: Some(t.token.index),
                };

                return Some(SentenceAnnotation {
                    clauses: vec![left_clause, right_clause],
                    edges: vec![edge],
                });
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Strategy 2: causal marker (connector)
// ---------------------------------------------------------------------------

fn try_causal_marker(
    tokens: &[TaggedToken],
    norm_lower: &[&str],
    res: &LexicalResources,
) -> Option<SentenceAnnotation> {
    for marker in &res.causal_markers {
        let mwords: Vec<&str> = marker.words.iter().map(|w| w.as_str()).collect();
        let mlen = mwords.len();

        'outer: for start in 0..norm_lower.len() {
            if start + mlen > norm_lower.len() {
                continue;
            }
            for (j, w) in mwords.iter().enumerate() {
                if norm_lower[start + j] != *w {
                    continue 'outer;
                }
            }

            let marker_tok_idx = tokens[start].token.index;
            let left_tokens = &tokens[..start];
            let right_tokens = &tokens[start + mlen..];

            // "pour" only as motivation before infinitive
            if marker.requires_infinitive && !right_contains_infinitive(right_tokens) {
                continue;
            }

            // Handle sentence-initial marker: left is empty, split right at comma
            if left_tokens.is_empty() {
                return handle_initial_marker(right_tokens, marker, marker_tok_idx, res);
            }

            if right_tokens.is_empty() {
                continue;
            }

            return Some(build_marker_annotation(left_tokens, right_tokens, marker, marker_tok_idx, res));
        }
    }
    None
}

/// Sentence-initial markers: "Si P, Q" or "Grâce à P, Q" or "Bien que P, Q"
/// Split `right_tokens` at the first comma to get P (subordinate) and Q (main).
fn handle_initial_marker(
    right_tokens: &[TaggedToken],
    marker: &CausalMarkerEntry,
    marker_tok_idx: u32,
    res: &LexicalResources,
) -> Option<SentenceAnnotation> {
    // Find comma position in right_tokens
    let comma_pos = right_tokens.iter().position(|t| t.token.lower == ",");

    let (sub_tokens, main_tokens) = {
        let cp = comma_pos?;
        (&right_tokens[..cp], &right_tokens[cp + 1..])
    };

    if sub_tokens.is_empty() || main_tokens.is_empty() {
        return None;
    }

    // For sentence-initial markers:
    // - subordinate clause (after marker, before comma) = first clause
    // - main clause (after comma) = second clause
    // Edge direction same as marker's direction but now:
    //   - if Forward: sub → main (sub is condition/concession source)
    //   - if Backward: sub is still the cause (unusual but keep)
    let sub_clause = build_clause(strip_punct_ends(sub_tokens), res);
    let main_clause = build_clause(strip_punct_ends(main_tokens), res);

    // Scope override for condition markers
    let (sub_clause_scoped, main_scoped) = if marker.relation == RelationType::Condition {
        let mut s = sub_clause;
        s.scope = Scope::Universal;
        let mut m = main_clause;
        m.scope = Scope::Universal;
        (s, m)
    } else {
        (sub_clause, main_clause)
    };

    // En position initiale, sub-clause (idx 0) précède toujours main-clause (idx 1)
    // quelle que soit la direction du marqueur.
    let (src_idx, dst_idx) = (0usize, 1usize);

    let mut clauses = vec![sub_clause_scoped, main_scoped];
    let mut edges = vec![EdgeAnnotation {
        src_clause: src_idx,
        dst_clause: dst_idx,
        relation: marker.relation,
        confidence: 1.0,
        explicit: true,
        negated: false,
        marker_token_idx: Some(marker_tok_idx),
    }];

    // Hypothetical node for concession
    if marker.signals_gap && marker.relation == RelationType::Concession {
        clauses.push(hypothetical_clause());
        edges.push(EdgeAnnotation {
            src_clause: 2,
            dst_clause: dst_idx,
            relation: RelationType::Cause,
            confidence: 0.3,
            explicit: false,
            negated: false,
            marker_token_idx: None,
        });
    }

    Some(SentenceAnnotation { clauses, edges })
}

fn right_contains_infinitive(tokens: &[TaggedToken]) -> bool {
    tokens.iter().any(|t| is_infinitive(&t.token.form))
}

fn build_marker_annotation(
    left: &[TaggedToken],
    right: &[TaggedToken],
    marker: &CausalMarkerEntry,
    marker_tok_idx: u32,
    res: &LexicalResources,
) -> SentenceAnnotation {
    let mut left_clause = build_clause(strip_punct_ends(left), res);
    let mut right_clause = build_clause(strip_punct_ends(right), res);

    if marker.relation == RelationType::Condition {
        left_clause.scope = Scope::Universal;
        right_clause.scope = Scope::Universal;
    }

    let (src_idx, dst_idx) = match marker.direction {
        MarkerDir::Forward  => (0, 1),
        MarkerDir::Backward => (1, 0),
        MarkerDir::GoalToAction => (1, 0),
    };

    let mut clauses = vec![left_clause, right_clause];
    let mut edges = vec![EdgeAnnotation {
        src_clause: src_idx,
        dst_clause: dst_idx,
        relation: marker.relation,
        confidence: 1.0,
        explicit: true,
        negated: false,
        marker_token_idx: Some(marker_tok_idx),
    }];

    // Hypothetical node for concession (paper-005, paper-012)
    if marker.signals_gap && marker.relation == RelationType::Concession {
        clauses.push(hypothetical_clause());
        edges.push(EdgeAnnotation {
            src_clause: 2,
            dst_clause: dst_idx,
            relation: RelationType::Cause,
            confidence: 0.3,
            explicit: false,
            negated: false,
            marker_token_idx: None,
        });
    }

    SentenceAnnotation { clauses, edges }
}

fn hypothetical_clause() -> ClauseAnnotation {
    ClauseAnnotation {
        span: (0, 0),
        node_type: NodeType::Condition,
        label: "cause_cachée(?)".to_string(),
        scope: Scope::Unknown,
        origin: NodeOrigin::Hypothetical,
        agent: None,
        patient: None,
        agent_type: None,
        entity: None,
        quality: None,
        has_depuis: false,
        is_imparfait: false,
        neg_on_node: false,
    }
}

// ---------------------------------------------------------------------------
// Clause builder
// ---------------------------------------------------------------------------

fn build_clause(tokens: &[TaggedToken], res: &LexicalResources) -> ClauseAnnotation {
    if tokens.is_empty() {
        return ClauseAnnotation {
            span: (0, 0),
            node_type: NodeType::Action,
            label: String::new(),
            scope: Scope::Specific,
            origin: NodeOrigin::Explicit,
            agent: None, patient: None, agent_type: None,
            entity: None, quality: None,
            has_depuis: false, is_imparfait: false, neg_on_node: false,
        };
    }

    let first_idx = tokens[0].token.index;
    let last_idx = tokens[tokens.len() - 1].token.index;

    // --- Scope from first DET in clause ---
    let mut scope = Scope::Specific;
    for t in tokens {
        if let Some(s) = t.scope_hint {
            scope = s;
            break;
        }
    }

    // --- Subject (first PRON or NOUN before main verb) ---
    let main_verb_idx = find_main_verb(tokens, res);
    let subject = extract_subject(tokens, main_verb_idx);

    // "on" as subject → universal scope
    if subject.as_deref().map_or(false, |s| FR_UNIVERSAL_SUBJECT_LEMMAS.contains(&s)) {
        scope = Scope::Universal;
    }

    // --- Negation in clause ---
    let neg_on_node = has_negation_in_clause(tokens);

    // --- depuis flag ---
    let has_depuis = tokens.iter().any(|t| FR_DURATIVE_MARKERS.contains(&t.token.lower.as_str()));

    // --- Main verb analysis ---
    let (node_type, verb_lemma, quality) = if let Some(vi) = main_verb_idx {
        let form_lower = tokens[vi].token.lower.clone();
        let lemma = tokens[vi].lemma.clone();

        // Use taxonomy lookup first, then fallback
        let mut nt = res.verb_classes.get(&lemma)
            .or_else(|| res.verb_classes.get(&form_lower))
            .copied()
            .unwrap_or(NodeType::Action);

        let is_imp = tokens[vi].is_imparfait || is_imparfait(&tokens[vi].token.form);
        let nom = nominalize_with_table(&lemma, &res.nominalizations).to_string();

        // Compositional: action/etat + depuis → processus
        if has_depuis && matches!(nt, NodeType::Action | NodeType::Etat) {
            nt = NodeType::Processus;
        }
        // Compositional: imparfait → processus (background state)
        if is_imp && matches!(nt, NodeType::Action | NodeType::Etat) {
            nt = NodeType::Processus;
        }

        (nt, lemma, Some(nom))
    } else {
        // Noun-headed clause
        let nt = classify_noun_clause(tokens, res);
        (nt, String::new(), None)
    };

    // --- Object / entity ---
    let (patient, entity_opt) = extract_object(tokens, main_verb_idx);

    // --- Resolve EtatSystemique from noun entity ---
    let mut resolved_type = node_type;
    if let Some(ent) = &entity_opt
        && let Some(&nt) = res.noun_node_types.get(ent.as_str())
    {
        resolved_type = nt;
    }
    // Also check if patient matches
    if (resolved_type == NodeType::Action || resolved_type == NodeType::Etat)
        && let Some(pat) = &patient
        && let Some(&nt) = res.noun_node_types.get(pat.as_str())
        && nt == NodeType::EtatSystemique
    {
        resolved_type = NodeType::EtatSystemique;
    }

    // --- Agent type ---
    let agent_type = subject.as_ref().map(|s| {
        res.pron_agent_types.get(s.as_str()).copied()
            .unwrap_or(gcn_ir::AgentType::Human)
    });

    // --- Build label ---
    let is_imp = main_verb_idx.map(|i| tokens[i].is_imparfait || is_imparfait(&tokens[i].token.form)).unwrap_or(false);
    let entity = entity_opt.clone().or_else(|| patient.clone());
    let label = build_label(&resolved_type, &verb_lemma, &entity, &subject, &res.nominalizations);

    ClauseAnnotation {
        span: (first_idx, last_idx),
        node_type: resolved_type,
        label,
        scope,
        origin: NodeOrigin::Explicit,
        agent: subject,
        patient,
        agent_type,
        entity,
        quality,
        has_depuis,
        is_imparfait: is_imp,
        neg_on_node,
    }
}

fn classify_noun_clause(tokens: &[TaggedToken], res: &LexicalResources) -> NodeType {
    for t in tokens {
        if let Some(&nt) = res.noun_node_types.get(t.token.lower.as_str()) {
            return nt;
        }
    }
    NodeType::Entite
}

fn find_main_verb(tokens: &[TaggedToken], res: &LexicalResources) -> Option<usize> {
    let mut last_lex_verb = None;
    let mut last_any_verb = None;
    for (i, t) in tokens.iter().enumerate() {
        if t.pos == Pos::Verb && t.token.lower != "," {
            last_any_verb = Some(i);
            if !res.auxiliary_forms.contains(t.token.lower.as_str()) {
                last_lex_verb = Some(i);
            }
        }
    }
    last_lex_verb.or(last_any_verb)
}

fn has_negation_in_clause(tokens: &[TaggedToken]) -> bool {
    let has_particle = tokens.iter().any(|t| t.is_negation_particle);
    let has_completer = tokens.iter().any(|t| t.is_negation_completer);
    has_particle && has_completer
}

fn has_negation_around(tokens: &[TaggedToken], verb_idx: usize) -> bool {
    let start = verb_idx.saturating_sub(4);
    let end = (verb_idx + 5).min(tokens.len());
    let window = &tokens[start..end];
    let has_particle = window.iter().any(|t| t.is_negation_particle);
    let has_completer = window.iter().any(|t| t.is_negation_completer);
    has_particle && has_completer
}

fn extract_subject(tokens: &[TaggedToken], verb_idx: Option<usize>) -> Option<String> {
    let vi = verb_idx.unwrap_or(tokens.len());
    for t in tokens.iter().take(vi) {
        if matches!(t.pos, Pos::Pron | Pos::Noun) && t.token.lower != "," {
            return Some(t.token.lower.clone());
        }
    }
    None
}

fn extract_object(
    tokens: &[TaggedToken],
    verb_idx: Option<usize>,
) -> (Option<String>, Option<String>) {
    let vi = match verb_idx {
        Some(i) => i,
        None => return (None, None),
    };
    let post_verb = &tokens[vi + 1..];
    let mut noun: Option<String> = None;
    for t in post_verb {
        if matches!(t.pos, Pos::Noun) {
            noun = Some(t.token.lower.clone());
            break;
        }
    }
    (noun.clone(), noun)
}

fn build_label(
    node_type: &NodeType,
    verb_lemma: &str,
    entity: &Option<String>,
    agent: &Option<String>,
    nominalizations: &std::collections::HashMap<String, String>,
) -> String {
    match node_type {
        NodeType::EtatSystemique | NodeType::Entite => {
            entity.clone().unwrap_or_else(|| verb_lemma.to_string())
        }
        NodeType::Condition => "cause_cachée(?)".to_string(),
        NodeType::Action => match agent {
            Some(a) => format!("{}({})", if verb_lemma.is_empty() { "?" } else { verb_lemma }, a),
            None => verb_lemma.to_string(),
        },
        NodeType::Etat | NodeType::Transition | NodeType::Processus => {
            let nom = nominalize_with_table(
                if verb_lemma.is_empty() { "?" } else { verb_lemma },
                nominalizations,
            );
            match entity {
                Some(e) => format!("{}({})", nom, e),
                None => match agent {
                    Some(a) => format!("{}({})", verb_lemma, a),
                    None => nom.to_string(),
                }
            }
        }
    }
}
