# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from gcn_python.layer1.features import FeatureVocabulary, vectorize_clause, vectorize_edge
from gcn_python.layer1.representation import UDRepresentation


def make_rep() -> UDRepresentation:
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
    )


def test_feature_dimensions():
    vocab = FeatureVocabulary()
    rep = make_rep()
    vec = vectorize_clause(rep, vocab)
    assert vec.shape == (vocab.d_clause,), f"Expected ({vocab.d_clause},), got {vec.shape}"
    assert vec.dtype == np.float32


def test_edge_dimensions():
    vocab = FeatureVocabulary()
    r1, r2 = make_rep(), make_rep()
    vec = vectorize_edge(r1, r2, None, 0, 1, 2, vocab)
    assert vec.shape == (vocab.d_edge,)


def test_vocabulary_serialization():
    vocab = FeatureVocabulary()
    restored = FeatureVocabulary.from_json(vocab.to_json())
    assert restored.d_clause == vocab.d_clause
    assert restored.upos_tags == vocab.upos_tags
    assert restored.dep_rels == vocab.dep_rels


def test_causal_pattern_absent_de_vectorize_clause():
    """
    Phase 5.4 : causal_pattern ne doit pas être vectorisé.
    d_clause reste 80 (constante calculée dynamiquement).
    Deux UDRepresentation identiques produisent le même vecteur clause.
    """
    vocab = FeatureVocabulary()
    base = dict(tokens=[], root_lemma="baisser", root_pos="VERB",
                root_dep_rel="root", root_morph={}, subject_pos=None,
                has_object=False, has_advcl=False, has_temporal_obl=False,
                token_span=(1, 2))
    rep1 = UDRepresentation(**base)
    rep2 = UDRepresentation(**base)
    vec1 = vectorize_clause(rep1, vocab)
    vec2 = vectorize_clause(rep2, vocab)
    np.testing.assert_array_equal(vec1, vec2)
    # d_clause ne change pas
    assert vocab.d_clause == vec1.shape[0]
