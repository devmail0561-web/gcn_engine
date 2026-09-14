import pytest
from pathlib import Path
from gcn_python.taxonomy.loader import TaxonomyIndex
from gcn_python.layer1.features import FeatureVocabulary
from gcn_python.layer2.reference import MLPEncoder
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.pipeline.cgnp import CGNPipeline

# Skip all pipeline tests that require spaCy models if not installed
def _spacy_model_available(lang: str) -> bool:
    try:
        from gcn_python.layer1.extractor import get_nlp
        get_nlp(lang)
        return True
    except OSError:
        return False

requires_fr_spacy = pytest.mark.skipif(
    not _spacy_model_available("fr"),
    reason="fr_core_news_sm not installed — run: python -m spacy download fr_core_news_sm",
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


@requires_fr_spacy
def test_forward_returns_cir(taxonomy_dir):
    pipeline = make_pipeline(taxonomy_dir)
    result = pipeline.forward("Si les ventes baissent, on réduit les coûts.")
    assert "source_lang" in result
    assert "nodes" in result
    assert "edges" in result
    assert result["source_text"] != ""


def test_forward_empty_text(taxonomy_dir):
    pipeline = make_pipeline(taxonomy_dir)
    result = pipeline.forward("")
    assert result["nodes"] == []


@requires_fr_spacy
def test_forward_single_clause(taxonomy_dir):
    pipeline = make_pipeline(taxonomy_dir)
    result = pipeline.forward("Il court.")
    assert len(result["nodes"]) >= 1


@requires_fr_spacy
def test_metadata_pipeline_field(taxonomy_dir):
    pipeline = make_pipeline(taxonomy_dir)
    result = pipeline.forward("Il travaille.")
    assert "pipeline" in result["metadata"]
    assert any(
        "layer" in p.lower() or "cgnp" in p.lower()
        for p in result["metadata"]["pipeline"]
    )
