# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Invalidation M1a côté gcn-transformers (supporté moteur).

Sans bump, CGNPipeline._model_tag est réutilisé et analyze() sert des CIR
pré-entraînement après mutation torch (staleness prouvée par sonde).
Requiert gcn-python>=4.0 (param_epoch) — skip sinon.
"""
import numpy as np
import pytest

pe = pytest.importorskip("gcn_python.param_epoch")

from gcn_transformers import XLMRobertaEncoder  # noqa: E402


def _encoder(dimensions):
    return XLMRobertaEncoder(d_clause=dimensions["d_clause"],
                             d_edge=dimensions["d_edge"])


def _node_grads(encoder):
    H = encoder.model.config.hidden_size
    return [(np.full((256, H), 0.01, dtype=np.float32),
             np.full((256,), 0.01, dtype=np.float32)),
            (np.full((8, 256), 0.01, dtype=np.float32),
             np.full((8,), 0.01, dtype=np.float32))]


def test_update_node_bumps_epoch(dimensions):
    import warnings
    enc = _encoder(dimensions)
    before = pe.current()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enc.update_node(_node_grads(enc), 1e-5)
    assert pe.current() == before + 1


def test_update_bumps_epoch(dimensions):
    enc = _encoder(dimensions)
    before = pe.current()
    enc.update([], 1e-5)
    assert pe.current() == before + 1


def test_load_parameters_bumps_epoch(dimensions):
    enc = _encoder(dimensions)
    params = [np.array(p, copy=True) for p in enc.parameters()]
    before = pe.current()
    enc.load_parameters(params)
    assert pe.current() == before + 1


def test_offline_missing_model_clear_error(monkeypatch):
    """Modèle absent du cache + hors-ligne → RuntimeError clair, pas de traceback OSError."""
    import gcn_transformers.base as _base
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="inaccessible"):
        _base.load_hf_model("inexistant/modele-xyz-123", revision=None)


def test_analyze_miss_after_update_node(dimensions, vocab):
    """Preuve bout-en-bout : update_node() -> analyze() recalcule (pas de hit stale)."""
    import warnings

    from gcn_python.layer1.embedding import WordEmbedding
    from gcn_python.layer1.representation import UDRepresentation
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline

    we = WordEmbedding(d_emb=8)
    d_eff = vocab.d_clause_effective(8, False)
    enc = XLMRobertaEncoder(d_clause=d_eff,
                            d_edge=vocab.d_edge_closed_loop(d_eff, 8, 8, False))
    graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=20)
    pipe = CGNPipeline(encoder=enc, graph=graph, vocabulary=vocab, word_embedding=we)

    class _P:
        def parse(self, text):
            r = UDRepresentation(
                tokens=[{"lemma": "pleuvoir", "pos": "VERB", "dep_rel": "root", "morph": {}}],
                root_lemma="pleuvoir", root_pos="VERB", root_dep_rel="root",
                root_morph={}, subject_pos=None, has_object=False, has_advcl=False,
                has_temporal_obl=False, token_span=(0, 1))
            return [r], [None]

    pipe.analyze("Il pleut.", text_parser=_P())
    assert pipe.cache_stats()["hits"] == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enc.update_node(_node_grads(enc), 1e-5)
    pipe.analyze("Il pleut.", text_parser=_P())
    assert pipe.cache_stats()["hits"] == 0  # recalcule, jamais de hit stale
    assert pipe.cache_stats()["misses"] == 2
