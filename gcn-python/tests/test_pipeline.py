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
