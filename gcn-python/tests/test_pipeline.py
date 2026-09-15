from pathlib import Path
import numpy as np
from gcn_python.taxonomy.loader import TaxonomyIndex
from gcn_python.layer1.features import FeatureVocabulary
from gcn_python.layer1.representation import UDRepresentation
from gcn_python.layer2.reference import MLPEncoder
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.pipeline.cgnp import CGNPipeline


def make_rep(lang: str = "fr") -> UDRepresentation:
    return UDRepresentation(
        tokens=[{"lemma": "baisser", "pos": "VERB", "dep_rel": "root", "morph": {}}],
        root_lemma="baisser",
        root_pos="VERB",
        root_dep_rel="root",
        root_morph={"Tense": "Pres"},
        subject_pos="NOUN",
        has_object=False,
        has_advcl=False,
        has_temporal_obl=False,
        token_span=(1, 2),
        lang=lang,
    )


def make_pipeline(taxonomy_dir: Path, lang: str = "fr") -> CGNPipeline:
    tax = TaxonomyIndex.load(taxonomy_dir, lang)
    vocab = FeatureVocabulary.build(tax)
    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge)
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    return CGNPipeline(
        encoder=encoder, graph=graph,
        taxonomy_dir=taxonomy_dir, lang=lang, vocabulary=vocab,
    )


def test_forward_returns_cir(taxonomy_dir):
    pipeline = make_pipeline(taxonomy_dir)
    rep = make_rep()
    result = pipeline.forward([rep], "Si les ventes baissent, on réduit les coûts.")
    assert "source_lang" in result
    assert "nodes" in result
    assert "edges" in result
    assert result["source_text"] != ""


def test_forward_empty_reps(taxonomy_dir):
    pipeline = make_pipeline(taxonomy_dir)
    result = pipeline.forward([], "")
    assert result["nodes"] == []


def test_forward_single_rep(taxonomy_dir):
    pipeline = make_pipeline(taxonomy_dir)
    result = pipeline.forward([make_rep()], "Il court.")
    assert len(result["nodes"]) >= 1


def test_metadata_pipeline_field(taxonomy_dir):
    pipeline = make_pipeline(taxonomy_dir)
    result = pipeline.forward([make_rep()], "Il travaille.")
    assert "pipeline" in result["metadata"]
    assert any(
        "layer" in p.lower() or "cgnp" in p.lower()
        for p in result["metadata"]["pipeline"]
    )
    assert not any("spacy" in p.lower() for p in result["metadata"]["pipeline"])


def test_forward_two_reps(taxonomy_dir):
    pipeline = make_pipeline(taxonomy_dir)
    rep1, rep2 = make_rep(), make_rep()
    result = pipeline.forward([rep1, rep2], "Les ventes baissent puis on réduit.")
    assert len(result["nodes"]) == 2
    assert len(result["edges"]) >= 1


def test_forward_connector_slot_nonzero(taxonomy_dir):
    """M2 : avec un connector_rep, les features d_conn ne sont pas toutes à zéro."""
    from gcn_python.layer1.features import vectorize_edge, FeatureVocabulary
    from gcn_python.taxonomy.loader import TaxonomyIndex

    tax = TaxonomyIndex.load(taxonomy_dir, "fr")
    vocab = FeatureVocabulary.build(tax)

    rep1 = make_rep()
    rep2 = make_rep()
    connector = UDRepresentation(
        tokens=[{"lemma": "parce", "pos": "SCONJ", "dep_rel": "mark", "morph": {}}],
        root_lemma="parce", root_pos="SCONJ", root_dep_rel="mark",
        root_morph={}, subject_pos=None,
        has_object=False, has_advcl=False, has_temporal_obl=False,
        token_span=(3, 3), lang="fr",
    )

    vec_with = vectorize_edge(rep1, rep2, connector, 0, 1, 2, vocab, tax)
    vec_without = vectorize_edge(rep1, rep2, None, 0, 1, 2, vocab, tax)

    # La partie UPOS du connecteur (premières len(upos_tags) dims du slot d_conn)
    n_upos = len(vocab.upos_tags)
    d_conn = vocab.d_conn
    upos_with = vec_with[-d_conn: -d_conn + n_upos]
    upos_without = vec_without[-d_conn: -d_conn + n_upos]

    assert not np.all(upos_with == 0), "La UPOS du connecteur doit être encodée"
    assert np.all(upos_without == 0), "Sans connecteur, la UPOS doit être nulle"


def test_forward_position_features_noncontiguous(taxonomy_dir):
    """M4 : clause_positions corrige les features de distance pour clauses non-contiguës."""
    from gcn_python.layer1.features import vectorize_edge, FeatureVocabulary
    from gcn_python.taxonomy.loader import TaxonomyIndex

    tax = TaxonomyIndex.load(taxonomy_dir, "fr")
    vocab = FeatureVocabulary.build(tax)

    rep1 = make_rep()
    rep2 = make_rep()

    # Sans positions : src_i=0, dst_i=1, n=2 → distance = 1/2
    vec_filtered = vectorize_edge(rep1, rep2, None, 0, 1, 2, vocab, tax)
    # Avec positions originales : src=0, dst=2, n=3 → distance = 2/3
    vec_original = vectorize_edge(rep1, rep2, None, 0, 2, 3, vocab, tax)

    # Les deux dernières dimensions du connecteur encodent la position
    assert not np.allclose(vec_filtered[-2:], vec_original[-2:]), \
        "Les features de position doivent différer selon les positions originales"
