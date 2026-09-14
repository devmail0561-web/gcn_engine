use crate::resources::{CausalMarkerEntry, LexicalResources};
use crate::rules::{nominalize_with_table, looks_like_verb_morphologically, MarkerDir};
use crate::tagger::{Pos, TaggedToken};
use gcn_ir::{AgentType, NodeOrigin, NodeType, RelationType, Scope};

// ---------------------------------------------------------------------------
// Annotation types
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
    pub is_progressive: bool,
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

        for sent_tokens in &sentences {
            let mut ann = annotate_sentence(sent_tokens, res);
            let offset = all_clauses.len();
            for e in &mut ann.edges {
                e.src_clause += offset;
                e.dst_clause += offset;
            }
            all_clauses.extend(ann.clauses);
            all_edges.extend(ann.edges);
        }

        // Implicit cause between last node of sentence 0 and first of sentence 1
        if sentences.len() >= 2 {
            let s0_count = annotate_sentence(sentences[0], res).clauses.len();
            if s0_count > 0 {
                all_edges.push(EdgeAnnotation {
                    src_clause: s0_count - 1,
                    dst_clause: s0_count,
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

    // Multi-word markers take precedence over causal verbs to avoid misattribution
    // (e.g. "in order to prevent X" → Motivation, not Prevent)
    if let Some(ann) = try_causal_marker(tokens, &lower_seq, res) {
        return ann;
    }
    if let Some(ann) = try_causal_verb(tokens, res) {
        return ann;
    }

    let clause = build_clause(tokens, res);
    SentenceAnnotation { clauses: vec![clause], edges: vec![] }
}

// ---------------------------------------------------------------------------
// Strategy 1: causal verb ("causes", "led to", etc.)
// ---------------------------------------------------------------------------

fn try_causal_verb(tokens: &[TaggedToken], res: &LexicalResources) -> Option<SentenceAnnotation> {
    for (i, t) in tokens.iter().enumerate() {
        if t.pos != Pos::Verb {
            continue;
        }
        let candidates = [t.lemma.as_str(), t.token.lower.as_str()];
        for lemma in candidates {
            if let Some(&relation) = res.causal_verb_relations.get(lemma) {
                if i == 0 || i + 1 >= tokens.len() {
                    break;
                }
                let negated = has_negation_around(tokens, i);
                let left_clause = build_clause(&tokens[..i], res);
                let right_clause = build_clause(&tokens[i + 1..], res);
                return Some(SentenceAnnotation {
                    clauses: vec![left_clause, right_clause],
                    edges: vec![EdgeAnnotation {
                        src_clause: 0,
                        dst_clause: 1,
                        relation,
                        confidence: 1.0,
                        explicit: true,
                        negated,
                        marker_token_idx: Some(t.token.index),
                    }],
                });
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Strategy 2: causal marker connector
// ---------------------------------------------------------------------------

fn try_causal_marker(
    tokens: &[TaggedToken],
    lower_seq: &[&str],
    res: &LexicalResources,
) -> Option<SentenceAnnotation> {
    for marker in &res.causal_markers {
        let mwords: Vec<&str> = marker.words.iter().map(|w| w.as_str()).collect();
        let mlen = mwords.len();

        'outer: for start in 0..lower_seq.len() {
            if start + mlen > lower_seq.len() {
                continue;
            }
            for (j, w) in mwords.iter().enumerate() {
                if lower_seq[start + j] != *w {
                    continue 'outer;
                }
            }

            let marker_tok_idx = tokens[start].token.index;
            let left_tokens = &tokens[..start];
            let right_tokens = &tokens[start + mlen..];

            // Infinitive requirement: "to" / "in order to" must precede infinitive base form
            if marker.requires_infinitive && !right_has_infinitive(right_tokens) {
                continue;
            }

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

fn right_has_infinitive(tokens: &[TaggedToken]) -> bool {
    tokens.iter().any(|t| matches!(t.pos, Pos::Verb) && looks_like_verb_morphologically(&t.token.form))
        || tokens.iter().any(|t| matches!(t.pos, Pos::Verb | Pos::Noun))
}

fn handle_initial_marker(
    right_tokens: &[TaggedToken],
    marker: &CausalMarkerEntry,
    marker_tok_idx: u32,
    res: &LexicalResources,
) -> Option<SentenceAnnotation> {
    let comma_pos = right_tokens.iter().position(|t| t.token.lower == ",")?;
    let sub_tokens = &right_tokens[..comma_pos];
    let main_tokens = &right_tokens[comma_pos + 1..];

    if sub_tokens.is_empty() || main_tokens.is_empty() {
        return None;
    }

    let mut sub_clause = build_clause(strip_punct_ends(sub_tokens), res);
    let mut main_clause = build_clause(strip_punct_ends(main_tokens), res);

    if marker.relation == RelationType::Condition {
        sub_clause.scope = Scope::Universal;
        main_clause.scope = Scope::Universal;
    }

    let (src_idx, dst_idx) = match marker.direction {
        MarkerDir::Forward | MarkerDir::Backward => (0, 1),
        MarkerDir::GoalToAction => (1, 0),
    };

    let mut clauses = vec![sub_clause, main_clause];
    let mut edges = vec![EdgeAnnotation {
        src_clause: src_idx,
        dst_clause: dst_idx,
        relation: marker.relation,
        confidence: 1.0,
        explicit: true,
        negated: false,
        marker_token_idx: Some(marker_tok_idx),
    }];

    if marker.signals_gap && marker.relation == RelationType::Concession {
        clauses.push(hypothetical_clause());
        edges.push(EdgeAnnotation {
            src_clause: 2,
            dst_clause: 1,
            relation: RelationType::Cause,
            confidence: 0.3,
            explicit: false,
            negated: false,
            marker_token_idx: None,
        });
    }

    Some(SentenceAnnotation { clauses, edges })
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
        MarkerDir::Forward      => (0, 1),
        MarkerDir::Backward     => (1, 0),
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

    if marker.signals_gap && marker.relation == RelationType::Concession {
        clauses.push(hypothetical_clause());
        edges.push(EdgeAnnotation {
            src_clause: 2,
            dst_clause: 1,
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
        label: "hidden_cause(?)".to_string(),
        scope: Scope::Unknown,
        origin: NodeOrigin::Hypothetical,
        agent: None, patient: None, agent_type: None,
        entity: None, quality: None,
        is_progressive: false, neg_on_node: false,
    }
}

// ---------------------------------------------------------------------------
// Clause builder
// ---------------------------------------------------------------------------

fn build_clause(tokens: &[TaggedToken], res: &LexicalResources) -> ClauseAnnotation {
    if tokens.is_empty() {
        return ClauseAnnotation {
            span: (0, 0), node_type: NodeType::Action, label: String::new(),
            scope: Scope::Specific, origin: NodeOrigin::Explicit,
            agent: None, patient: None, agent_type: None,
            entity: None, quality: None, is_progressive: false, neg_on_node: false,
        };
    }

    let first_idx = tokens[0].token.index;
    let last_idx = tokens[tokens.len() - 1].token.index;

    let mut scope = Scope::Specific;
    for t in tokens {
        if let Some(s) = t.scope_hint {
            scope = s;
            break;
        }
    }

    let main_verb_idx = find_main_verb(tokens, res);
    let subject = extract_subject(tokens, main_verb_idx);

    if matches!(subject.as_deref(), Some("everyone") | Some("everything") | Some("all")) {
        scope = Scope::Universal;
    }

    let neg_on_node = has_negation_in_clause(tokens);
    let is_progressive = has_progressive(tokens);

    let (node_type, verb_lemma, quality) = if let Some(vi) = main_verb_idx {
        let lower = tokens[vi].token.lower.clone();
        let lemma = tokens[vi].lemma.clone();

        let mut nt = res.verb_classes.get(&lemma)
            .or_else(|| res.verb_classes.get(&lower))
            .copied()
            .unwrap_or(NodeType::Action);

        let nom = nominalize_with_table(&lemma, &res.nominalizations).to_string();

        // Progressive aspect: action/etat → processus
        if is_progressive && matches!(nt, NodeType::Action | NodeType::Etat) {
            nt = NodeType::Processus;
        }

        (nt, lemma, Some(nom))
    } else {
        let nt = classify_noun_clause(tokens, res);
        (nt, String::new(), None)
    };

    let (patient, entity_opt) = extract_object(tokens, main_verb_idx);

    let mut resolved_type = node_type;
    if let Some(ent) = &entity_opt
        && let Some(&nt) = res.noun_node_types.get(ent.as_str())
    {
        resolved_type = nt;
    }
    if (resolved_type == NodeType::Action || resolved_type == NodeType::Etat)
        && let Some(pat) = &patient
        && let Some(&nt) = res.noun_node_types.get(pat.as_str())
        && nt == NodeType::EtatSystemique
    {
        resolved_type = NodeType::EtatSystemique;
    }

    let agent_type = subject.as_ref().map(|s| {
        res.pron_agent_types.get(s.as_str()).copied().unwrap_or(AgentType::Human)
    });

    let entity = entity_opt.clone().or_else(|| patient.clone());
    let label = build_label(&resolved_type, &verb_lemma, &entity, &subject, &res.nominalizations);

    ClauseAnnotation {
        span: (first_idx, last_idx),
        node_type: resolved_type, label, scope,
        origin: NodeOrigin::Explicit, agent: subject,
        patient, agent_type, entity, quality,
        is_progressive, neg_on_node,
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
    let mut last_lex = None;
    let mut last_any = None;
    for (i, t) in tokens.iter().enumerate() {
        if t.pos == Pos::Verb {
            last_any = Some(i);
            if !res.auxiliary_forms.contains(t.token.lower.as_str()) {
                last_lex = Some(i);
            }
        }
    }
    last_lex.or(last_any)
}

fn has_negation_in_clause(tokens: &[TaggedToken]) -> bool {
    // English: "not" adjacent to a verb, or "n't" contraction
    tokens.iter().any(|t| t.is_negation_particle || t.is_negation_completer)
}

fn has_negation_around(tokens: &[TaggedToken], verb_idx: usize) -> bool {
    let start = verb_idx.saturating_sub(3);
    let end = (verb_idx + 4).min(tokens.len());
    tokens[start..end].iter().any(|t| t.is_negation_particle || t.is_negation_completer)
}

fn has_progressive(tokens: &[TaggedToken]) -> bool {
    tokens.iter().any(|t| t.token.lower.ends_with("ing") && t.pos == Pos::Verb)
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

fn extract_object(tokens: &[TaggedToken], verb_idx: Option<usize>) -> (Option<String>, Option<String>) {
    let vi = match verb_idx { Some(i) => i, None => return (None, None) };
    let post = &tokens[vi + 1..];
    let mut noun = None;
    for t in post {
        if t.pos == Pos::Noun {
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
        NodeType::Condition => "hidden_cause(?)".to_string(),
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
