# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests Phase B.5 — features.py : d_clause=106, Voice, PronType, bridge fix."""
from __future__ import annotations

import numpy as np

from gcn_python.layer1.features import FeatureVocabulary, vectorize_clause
from gcn_python.layer1.representation import UDRepresentation


class _MockEmb:
    d_emb = 64
    def lookup(self, lemma: str) -> np.ndarray:
        return np.zeros(self.d_emb, dtype=np.float32)


def make_rep(tokens=None, root_morph=None, has_advcl=False):
    return UDRepresentation(
        tokens=tokens or [{"lemma": "x", "pos": "VERB", "dep_rel": "root", "morph": {}}],
        root_lemma="x",
        root_pos="VERB",
        root_dep_rel="root",
        root_morph=root_morph or {},
        subject_pos=None,
        has_object=False,
        has_advcl=has_advcl,
        has_temporal_obl=False,
        token_span=(0, 1),
    )


def test_d_clause_equals_106():
    assert FeatureVocabulary().d_clause == 106


def test_voice_encoded_passive():
    from conftest import make_word_embedding
    we = make_word_embedding()
    vocab = FeatureVocabulary()
    rep = make_rep(root_morph={"Voice": "Pass"})
    vec = vectorize_clause(rep, vocab, we)
    assert vec.shape == (vocab.d_clause_effective(we.d_emb),)
    voice_offset = 18 + 38 + 5 + 5 + 4 + 5
    pass_idx = vocab.voice_values.index("Pass")
    assert vec[voice_offset + pass_idx] == 1.0


def test_prontype_int_encoded():
    from conftest import make_word_embedding
    we = make_word_embedding()
    vocab = FeatureVocabulary()
    tokens = [{"lemma": "qui", "pos": "PRON", "dep_rel": "advmod",
               "morph": {"PronType": "Int"}}]
    rep = make_rep(tokens=tokens)
    vec = vectorize_clause(rep, vocab, we)
    prontype_offset = 18 + 38 + 5 + 5 + 4 + 5 + 3
    int_idx = vocab.prontype_values.index("Int")
    assert vec[prontype_offset + int_idx] == 1.0


def test_has_advcl_condition_node_true():
    from gcn_python.frontend.bridge import NODE_TYPE_TO_DEP, NODE_TYPE_TO_POS
    assert NODE_TYPE_TO_POS.get("condition") == "SCONJ"
    assert NODE_TYPE_TO_DEP.get("condition") == "advcl"


def test_lemma_emb_shape_uses_d_emb():
    vocab = FeatureVocabulary()
    emb = _MockEmb()
    rep = make_rep()
    vec = vectorize_clause(rep, vocab, word_embedding=emb, subject_object_emb=True)
    expected = vocab.d_clause_effective(emb.d_emb, True)
    assert expected == 106 + 3 * 64  # 298
    assert vec.shape == (expected,)
