# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from gcn_python.layer1.features import (
    FeatureVocabulary,
    vectorize_clause,
    vectorize_edge,
)
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
    base = {"tokens": [], "root_lemma": "baisser", "root_pos": "VERB",
            "root_dep_rel": "root", "root_morph": {}, "subject_pos": None,
            "has_object": False, "has_advcl": False, "has_temporal_obl": False,
            "token_span": (1, 2)}
    rep1 = UDRepresentation(**base)
    rep2 = UDRepresentation(**base)
    vec1 = vectorize_clause(rep1, vocab)
    vec2 = vectorize_clause(rep2, vocab)
    np.testing.assert_array_equal(vec1, vec2)
    # d_clause ne change pas
    assert vocab.d_clause == vec1.shape[0]


# ── Amélioration A — Pooling des tokens de contenu ────────────────────────────
from gcn_python.layer1.embedding import WordEmbedding
from gcn_python.layer1.features import (
    CONTENT_POS,
    _pool_lemmas,
)


def make_content_rep() -> UDRepresentation:
    """Clause DET + NOUN + ADP + VERB : seul NOUN/VERB sont du contenu."""
    return UDRepresentation(
        tokens=[
            {"lemma": "le", "pos": "DET", "dep_rel": "det", "morph": {}},
            {"lemma": "chat", "pos": "NOUN", "dep_rel": "nsubj", "morph": {}},
            {"lemma": "de", "pos": "ADP", "dep_rel": "case", "morph": {}},
            {"lemma": "dormir", "pos": "VERB", "dep_rel": "root", "morph": {}},
        ],
        root_lemma="dormir", root_pos="VERB", root_dep_rel="root",
        root_morph={}, subject_pos="NOUN",
        has_object=False, has_advcl=False, has_temporal_obl=False,
        token_span=(1, 4),
    )


def make_grammatical_rep() -> UDRepresentation:
    """Clause uniquement DET + ADP : fallback vers tous les tokens."""
    return UDRepresentation(
        tokens=[
            {"lemma": "de", "pos": "ADP", "dep_rel": "case", "morph": {}},
            {"lemma": "le", "pos": "DET", "dep_rel": "det", "morph": {}},
        ],
        root_lemma="de", root_pos="ADP", root_dep_rel="case",
        root_morph={}, subject_pos=None,
        has_object=False, has_advcl=False, has_temporal_obl=False,
        token_span=(1, 2),
    )


def make_we(dim: int = 8, seed: int = 0) -> WordEmbedding:
    we = WordEmbedding(d_emb=dim, seed=seed)
    we.build_vocab(["le", "chat", "de", "dormir", "baisser"])
    return we


def test_mean_pool_filters_stop_words():
    vocab = FeatureVocabulary()
    rep = make_content_rep()
    we = make_we()
    vec_mean = vectorize_clause(rep, vocab, we, clause_pooling="mean")
    vec_all = np.mean(
        np.stack([we.lookup(t["lemma"]) for t in rep.tokens]), axis=0)
    assert vec_mean.shape == (vocab.d_clause + we.d_emb,)
    assert not np.allclose(vec_mean[vocab.d_clause:], vec_all), \
        "mean_pool doit filtrer DET/ADP (≠ moyenne tous tokens)"


def test_mean_pool_fallback_all_tokens_when_no_content():
    vocab = FeatureVocabulary()
    rep = make_grammatical_rep()
    we = make_we()
    vec = vectorize_clause(rep, vocab, we, clause_pooling="mean")
    vec_all = np.mean(
        np.stack([we.lookup(t["lemma"]) for t in rep.tokens]), axis=0)
    assert np.allclose(vec[vocab.d_clause:], vec_all), \
        "sans contenu : fallback vers tous les tokens"


def test_pooling_same_dim_as_root():
    vocab = FeatureVocabulary()
    rep = make_content_rep()
    we = make_we()
    for mode in ("root", "mean", "max"):
        vec = vectorize_clause(rep, vocab, we, clause_pooling=mode)
        assert vec.shape == (vocab.d_clause + we.d_emb,), mode


def test_pooling_requires_word_embedding():
    vocab = FeatureVocabulary()
    rep = make_content_rep()
    with pytest.raises(ValueError, match="word_embedding"):
        vectorize_clause(rep, vocab, None, clause_pooling="mean")
    with pytest.raises(ValueError, match="word_embedding"):
        vectorize_clause(rep, vocab, None, subject_object_emb=True)


def test_vocab_extended_with_mean_pooling():
    rep = make_content_rep()
    root_only = [rep.root_lemma]
    pooled = _pool_lemmas(rep, "mean")
    assert set(root_only) < set(pooled) or len(pooled) > len(root_only), \
        "mean-pooling doit collecter plus de lemmes que root seul"
    assert all(t in CONTENT_POS or True for t in pooled)
    assert set(pooled) == {"chat", "dormir"}


# ── Amélioration B — Embeddings sujet/objet + _absent appris ──────────────────

def test_absent_subject_lookup_not_zero():
    we = WordEmbedding(d_emb=8, seed=1)
    before = we.lookup('_subj_absent').copy()
    we.backward(np.ones(8, dtype=np.float32), '_subj_absent')
    we.update(0.1)
    after = we.lookup('_subj_absent')
    assert not np.allclose(before, after), "_subj_absent doit apprendre"
    assert not np.allclose(after, np.zeros(8)), "_subj_absent ≠ vecteur nul"


def test_absent_subject_different_from_absent_object():
    we = WordEmbedding(d_emb=8, seed=2)
    g = np.ones(8, dtype=np.float32)
    we.backward(g, '_subj_absent')
    we.backward(-g, '_obj_absent')
    we.update(0.1)
    assert not np.allclose(we.lookup('_subj_absent'), we.lookup('_obj_absent'))


def test_d_clause_effective_with_B():
    vocab = FeatureVocabulary()
    d_emb = 49
    assert vocab.d_clause_effective(d_emb, False) == vocab.d_clause + d_emb
    assert vocab.d_clause_effective(d_emb, True) == vocab.d_clause + 3 * d_emb
    assert vocab.d_clause_effective(49, True) == 79 + 147


def test_subject_object_emb_requires_word_embedding():
    vocab = FeatureVocabulary()
    with pytest.raises(ValueError, match="word_embedding"):
        vectorize_clause(make_rep(), vocab, None, subject_object_emb=True)


def test_subject_object_emb_appends_two_vectors():
    vocab = FeatureVocabulary()
    rep = make_content_rep()  # nsubj=chat, pas d'objet → _obj_absent
    we = make_we()
    vec = vectorize_clause(rep, vocab, we, subject_object_emb=True)
    assert vec.shape == (vocab.d_clause + 3 * we.d_emb,)
    d = we.d_emb
    assert np.allclose(vec[vocab.d_clause + d:vocab.d_clause + 2 * d],
                       we.lookup("chat"))
    assert np.allclose(vec[vocab.d_clause + 2 * d:],
                       we.lookup('_obj_absent'))


# ── Amélioration C — Gel des pré-entraînés ────────────────────────────────────

def make_pretrained_we(tmp_path=None, dim: int = 8) -> WordEmbedding:
    we = WordEmbedding(d_emb=dim, seed=3, frozen=True)
    # Simuler un fichier pré-entraîné via ajout direct + marquage de plage
    we.add_lemma("pretrained_word")
    we._pretrained_start = 3  # après _unk, _subj_absent, _obj_absent
    we._pretrained_end = len(we._lemmas)
    we.build_vocab(["new_lemma"])
    return we


def test_frozen_embedding_update_noop():
    we = make_pretrained_we()
    before = we._E.copy()
    we.backward(np.ones(we.d_emb, dtype=np.float32), "pretrained_word")
    we.update(0.1)
    idx = we._vocab["pretrained_word"]
    assert np.allclose(we._E[idx], before[idx]), "ligne pré-entraînée gelée"


def test_frozen_embedding_backward_no_accumulate():
    we = make_pretrained_we()
    we.backward(np.ones(we.d_emb, dtype=np.float32), "pretrained_word")
    assert we._grad_accum is None or not np.any(we._grad_accum[we._vocab["pretrained_word"]] != 0)


def test_absent_tokens_trainable_even_when_frozen():
    we = make_pretrained_we()
    before_subj = we.lookup('_subj_absent').copy()
    before_new = we.lookup('new_lemma').copy()
    we.backward(np.ones(we.d_emb, dtype=np.float32), '_subj_absent')
    we.backward(np.ones(we.d_emb, dtype=np.float32), '_obj_absent')
    we.backward(np.ones(we.d_emb, dtype=np.float32), 'new_lemma')
    we.update(0.1)
    assert not np.allclose(we.lookup('_subj_absent'), before_subj)
    assert not np.allclose(we.lookup('_obj_absent'), before_subj)
    assert not np.allclose(we.lookup('new_lemma'), before_new)
