# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests Phase 10-M1 — ne pas recalculer les mots déjà vus.

M1a : cache analyze (clé = modèle + parseur + texte) — répétition = zéro
nouveau calcul (parse ET forward sautés, preuve par compteur du parseur) ;
invalidation automatique au changement de modèle (preuve par mutation de
poids) ; isolation par parseur ; restauration de l'état NLU/vecs.
M1b : dédup de lots (analyze_batch). M1c : pré-filtre avant tri (identique).
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "gcn-python/tests")
from conftest import make_test_pipeline

from gcn_python.layer1.representation import UDRepresentation


class _Parser:
    """Parseur stub déterministe + compteur d'appels (preuve de non-recalcul)."""

    def __init__(self, n_clauses: int = 2):
        self.calls = 0
        self.n_clauses = n_clauses

    def parse(self, text):
        self.calls += 1
        reps = []
        for i in range(self.n_clauses):
            lemma = f"w{i}"
            reps.append(UDRepresentation(
                tokens=[{"lemma": lemma, "pos": "VERB", "dep_rel": "root", "morph": {}}],
                root_lemma=lemma, root_pos="VERB", root_dep_rel="root",
                root_morph={}, subject_pos=None,
                has_object=False, has_advcl=False, has_temporal_obl=False,
                token_span=(i, i + 1),
            ))
        return reps, []


def _pipe():
    return make_test_pipeline(d_emb=8)


class TestAnalyzeCache:
    def test_repeat_is_free(self):
        pipe, parser = _pipe(), _Parser()
        c1 = pipe.analyze("phrase vue", text_parser=parser)
        c2 = pipe.analyze("phrase vue", text_parser=parser)
        assert parser.calls == 1  # parse sauté la 2e fois
        assert c1 == c2
        assert pipe.cache_stats() == {"hits": 1, "misses": 1, "size": 1, "max": 1024}

    def test_hit_restores_nlu_and_vec_state(self):
        pipe, parser = _pipe(), _Parser()
        pipe.analyze("phrase vue", text_parser=parser)
        vecs_miss = np.array(pipe.get_enriched_vectors(), copy=True)
        reps_miss = [r.root_lemma for r in pipe._cached_reps]
        pipe._cached_reps = None  # efface l'état : le hit doit le restaurer
        pipe._cached_enriched_vecs = None
        pipe.analyze("phrase vue", text_parser=parser)
        assert [r.root_lemma for r in pipe._cached_reps] == reps_miss
        assert np.array_equal(np.asarray(pipe.get_enriched_vectors()), vecs_miss)

    def test_hit_returns_independent_copies(self):
        pipe, parser = _pipe(), _Parser()
        c1 = pipe.analyze("phrase vue", text_parser=parser)
        c1["edges"] = []  # mutation appelant (cf. discuss : préfixe ids)
        c2 = pipe.analyze("phrase vue", text_parser=parser)
        assert c2["edges"] != []

    def test_model_change_invalidates(self):
        pipe, parser = _pipe(), _Parser()
        c1 = pipe.analyze("phrase vue", text_parser=parser)
        # Changement de modèle par voie publique : update dense de l'encodeur
        # (bump epoch → rehash → clé différente → recalcul obligatoire).
        grads = [np.ones_like(p) for p in pipe.encoder.parameters()]
        pipe.encoder.update(grads, 0.01)
        c2 = pipe.analyze("phrase vue", text_parser=parser)
        assert parser.calls == 2  # recalculé
        assert pipe.cache_stats()["misses"] == 2
        assert c2 != c1  # poids différents → sortie différente

    def test_parser_isolation(self):
        pipe = _pipe()
        p1, p2 = _Parser(), _Parser()
        pipe.analyze("phrase vue", text_parser=p1)
        pipe.analyze("phrase vue", text_parser=p2)
        assert (p1.calls, p2.calls) == (1, 1)
        assert pipe.cache_stats() == {"hits": 0, "misses": 2, "size": 2, "max": 1024}

    def test_different_text_misses(self):
        pipe, parser = _pipe(), _Parser()
        pipe.analyze("phrase une", text_parser=parser)
        pipe.analyze("phrase deux", text_parser=parser)
        assert parser.calls == 2

    def test_fifo_cap_and_clear(self):
        pipe, parser = _pipe(), _Parser()
        pipe.ANALYZE_CACHE_MAX = 2
        for t in ("a un", "b deux", "c trois"):
            pipe.analyze(t, text_parser=parser)
        assert pipe.cache_stats()["size"] == 2
        pipe.clear_cache()
        assert pipe.cache_stats() == {"hits": 0, "misses": 0, "size": 0, "max": 2}
        pipe.analyze("a un", text_parser=parser)  # évincé → recalcul
        assert parser.calls == 4

    def test_forward_bypasses_cache(self):
        # Train-safe : forward() direct ne consulte ni ne remplit le cache.
        pipe, parser = _pipe(), _Parser()
        reps, _ = parser.parse("phrase vue")
        pipe.forward(reps, "phrase vue")
        pipe.forward(reps, "phrase vue")
        assert pipe.cache_stats()["size"] == 0
        assert parser.calls == 1  # le parse manuel, pas le cache


class TestAnalyzeBatch:
    def test_dedup_preserves_order(self):
        from gcn_python.engine import GCNEngine
        pipe, parser = _pipe(), _Parser()
        engine = GCNEngine(pipe, text_parser=parser)
        out = engine.analyze_batch(["phrase une", "phrase deux", "phrase une", "  ", "phrase deux"])
        assert len(out) == 4
        assert parser.calls == 2  # chaque texte distinct une seule fois
        assert out[0] == out[2] and out[1] == out[3]
        out[0]["edges"] = []  # copies indépendantes
        assert out[2]["edges"] != []


class TestPredictLinksPrefilter:
    def test_threshold_identical_to_postfilter(self):
        from gcn_python.layer3.link_pred import LinkPredHead
        pipe = make_test_pipeline(d_emb=8)
        rng = np.random.default_rng(0)
        vecs = rng.normal(size=(6, 8)).astype(np.float32)
        head = LinkPredHead(d_in=8, src_aggregation="mean")
        pipe.link_predictor = head
        pairs = [(i, j) for i in range(6) for j in range(6) if i != j]
        full = pipe.predict_links(pairs, vecs)
        pre = pipe.predict_links(pairs, vecs, threshold=0.5)
        assert pre == [t for t in full if t[2] >= 0.5]
        k = 5
        top = pipe.predict_links(pairs, vecs, topk=k)
        assert top == full[:k]
