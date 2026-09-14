import numpy as np
from gcn_python.layer2.reference import MLPEncoder
from gcn_python.taxonomy.loader import TaxonomyIndex
from gcn_python.layer1.features import FeatureVocabulary


def test_forward_node_shape(taxonomy_dir):
    tax = TaxonomyIndex.load(taxonomy_dir, "fr")
    vocab = FeatureVocabulary.build(tax)
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge)
    x = np.random.randn(vocab.d_clause).astype(np.float32)
    out = enc.forward_node(x)
    assert out.shape == (7,), f"Expected (7,), got {out.shape}"


def test_forward_edge_shape(taxonomy_dir):
    tax = TaxonomyIndex.load(taxonomy_dir, "fr")
    vocab = FeatureVocabulary.build(tax)
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge)
    x = np.random.randn(vocab.d_edge).astype(np.float32)
    out = enc.forward_edge(x)
    assert out.shape == (11,), f"Expected (11,), got {out.shape}"


def test_parameters_list(taxonomy_dir):
    tax = TaxonomyIndex.load(taxonomy_dir, "fr")
    vocab = FeatureVocabulary.build(tax)
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge)
    params = enc.parameters()
    assert len(params) > 0
    assert all(isinstance(p, np.ndarray) for p in params)
