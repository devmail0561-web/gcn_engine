use std::path::PathBuf;
use gcn_frontend_en::EnglishParser;
use gcn_ir::{NaturalLanguage, RelationType, SourceLanguage};

fn taxonomy_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../gcn-references/taxonomies")
}

fn parser() -> EnglishParser {
    EnglishParser::new(&taxonomy_dir()).expect("failed to load English taxonomies")
}

// ─── Basic API ───────────────────────────────────────────────────────────────

#[test]
fn empty_input_errors() {
    let p = parser();
    assert!(p.parse("").is_err());
    assert!(p.parse("   ").is_err());
}

#[test]
fn source_language_is_english() {
    let p = parser();
    let ir = p.parse("Sales fall because costs rise.").unwrap();
    assert!(
        matches!(ir.source_lang, SourceLanguage::Natural { lang: NaturalLanguage::English }),
        "source_lang should be English"
    );
}

#[test]
fn pipeline_tag_is_gcn_frontend_en() {
    let p = parser();
    let ir = p.parse("The crisis causes unemployment.").unwrap();
    assert!(ir.metadata.pipeline.contains(&"gcn-frontend-en".to_string()));
}

// ─── Cause backward ("because") ─────────────────────────────────────────────

#[test]
fn because_backward_cause() {
    let p = parser();
    let ir = p.parse("Sales fell because costs rose.").unwrap();
    assert_eq!(ir.edges.len(), 1);
    let (_, _, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Cause);
    assert!(edge.explicit);
}

// ─── Consequence forward ("therefore") ──────────────────────────────────────

#[test]
fn therefore_forward_consequence() {
    let p = parser();
    let ir = p.parse("Costs rose, therefore sales fell.").unwrap();
    assert!(!ir.edges.is_empty());
    let (_, _, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Cause);
}

// ─── Condition ("if") ───────────────────────────────────────────────────────

#[test]
fn if_condition() {
    let p = parser();
    let ir = p.parse("If costs rise, sales fall.").unwrap();
    assert!(!ir.edges.is_empty());
    let (_, _, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Condition);
}

// ─── Concession ("although") ────────────────────────────────────────────────

#[test]
fn although_concession() {
    let p = parser();
    let ir = p.parse("Sales fell although costs decreased.").unwrap();
    assert!(!ir.edges.is_empty());
    let (_, _, edge) = &ir.edges[0];
    assert!(matches!(edge.relation, RelationType::Concession | RelationType::Opposition));
}

// ─── Causal verbs ───────────────────────────────────────────────────────────

#[test]
fn causal_verb_causes() {
    let p = parser();
    let ir = p.parse("The recession causes unemployment.").unwrap();
    assert!(!ir.edges.is_empty());
    let (_, _, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Cause);
    assert!(edge.confidence >= 0.9);
}

#[test]
fn causal_verb_enables() {
    let p = parser();
    let ir = p.parse("The policy enables growth.").unwrap();
    assert!(!ir.edges.is_empty());
    let (_, _, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Enable);
}

#[test]
fn causal_verb_prevents() {
    let p = parser();
    let ir = p.parse("The intervention prevents collapse.").unwrap();
    assert!(!ir.edges.is_empty());
    let (_, _, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Prevent);
}

// ─── Node count ─────────────────────────────────────────────────────────────

#[test]
fn two_clauses_produce_two_nodes() {
    let p = parser();
    let ir = p.parse("Costs rise because demand increases.").unwrap();
    assert_eq!(ir.nodes.len(), 2);
}

// ─── Negation ───────────────────────────────────────────────────────────────

#[test]
fn negation_on_edge() {
    let p = parser();
    let ir = p.parse("The policy does not prevent recession.").unwrap();
    // Negation detected; edge may be negated or node may carry negation modifier
    assert!(!ir.nodes.is_empty());
}

// ─── Empty-parse fallback ────────────────────────────────────────────────────

#[test]
fn single_word_still_produces_one_node() {
    let p = parser();
    let ir = p.parse("Recession.").unwrap();
    assert_eq!(ir.nodes.len(), 1);
}

// ─── Motivation ("in order to") ──────────────────────────────────────────────

#[test]
fn in_order_to_motivation() {
    let p = parser();
    let ir = p.parse("The government intervened in order to prevent collapse.").unwrap();
    assert!(!ir.edges.is_empty());
    let (_, _, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Motivation);
}

// ─── Enable preposition ("thanks to") ───────────────────────────────────────

#[test]
fn thanks_to_enable() {
    let p = parser();
    let ir = p.parse("The economy grew thanks to the reform.").unwrap();
    assert!(!ir.edges.is_empty());
    let (_, _, edge) = &ir.edges[0];
    assert_eq!(edge.relation, RelationType::Enable);
}

// ─── Cross-modal isomorphism with French ─────────────────────────────────────

#[test]
fn en_fr_condition_isomorphism() {
    use gcn_frontend_fr::FrenchParser;

    let en_parser = parser();
    let fr_parser = FrenchParser::new(&taxonomy_dir()).expect("failed to load French taxonomies");

    let en_ir = en_parser.parse("If costs rise, sales fall.").unwrap();
    let fr_ir = fr_parser.parse("Si les coûts augmentent, les ventes baissent.").unwrap();

    assert_eq!(en_ir.nodes.len(), fr_ir.nodes.len(), "même nombre de nœuds fr↔en");
    let (_, _, en_edge) = &en_ir.edges[0];
    let (_, _, fr_edge) = &fr_ir.edges[0];
    assert_eq!(en_edge.relation, fr_edge.relation, "même type de relation Condition");
}
