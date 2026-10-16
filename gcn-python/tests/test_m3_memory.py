# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests Phase 10-M3 — alléger la mémoire vive (garde-fou chiffré).

M3a : bulk O(n) bit-identique (RNG, écrasements, gel), subset `only=`,
mmap session roundtrip.
M3b : discipline float32 du forward (gate — float16 écarté, voir README).
M3c : documenté no-op (pic 481 Ko/50 forwards — zones pré-allouées injustifiées).
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "gcn-python/tests")
from conftest import make_test_pipeline
from gcn_python.layer1.embedding import WordEmbedding


def _we_seq(d_emb=8, seed=7):
    we = WordEmbedding(d_emb=d_emb, seed=seed)
    for w in ("alpha", "beta", "gamma"):
        we.add_lemma(w)
    return we


class TestBulkIdentical:
    def test_bulk_matches_sequential_rng(self):
        a, b = _we_seq(), WordEmbedding(d_emb=8, seed=7)
        b._add_lemmas_bulk(["alpha", "beta", "gamma"])
        assert a._lemmas == b._lemmas
        assert np.array_equal(a._E, b._E)  # tirage bit-identique

    def test_bulk_last_wins_and_overwrites_vocab(self):
        a = WordEmbedding(d_emb=4, seed=1)
        v1 = np.ones(4, dtype=np.float32)
        v2 = np.full(4, 2.0, dtype=np.float32)
        a._add_lemmas_bulk(["x", "x"], [v1, v2])
        assert np.array_equal(a._E[a._vocab["x"]], v2)  # dernier gagne
        a._add_lemmas_bulk(["x"], [v1])
        assert np.array_equal(a._E[a._vocab["x"]], v1)  # écrase l'existant

    def test_build_vocab_uses_bulk(self):
        a = WordEmbedding(d_emb=8, seed=3)
        a.build_vocab([f"w{i}" for i in range(50)])
        assert len(a._lemmas) == 3 + 50  # _unk + 2 spéciaux + 50
        assert a._E.shape == (53, 8)


class TestOnlySubset:
    def _vecfile(self, tmp_path):
        p = tmp_path / "vecs.txt"
        lines = ["3 4"]
        for w, base in (("alpha", 1.0), ("beta", 2.0), ("gamma", 3.0)):
            lines.append(f"{w} " + " ".join(f"{base + i * 0.1:.1f}" for i in range(4)))
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    def test_only_loads_subset_with_same_values(self, tmp_path):
        p = self._vecfile(tmp_path)
        full = WordEmbedding(d_emb=4, seed=0)
        assert full.load_from_file(str(p)) == 3
        sub = WordEmbedding(d_emb=4, seed=0)
        assert sub.load_from_file(str(p), only={"beta", "gamma"}) == 2
        assert np.array_equal(sub.lookup("beta"), full.lookup("beta"))
        assert np.array_equal(sub.lookup("gamma"), full.lookup("gamma"))
        # Plage gel cohérente (contiguë) + entraînement du reste
        assert sub._pretrained_start == 3 and sub._pretrained_end == 5

    def test_only_unknown_word_skipped_before_parse(self, tmp_path):
        p = tmp_path / "bad.txt"
        p.write_text("1 4\nzz 1.0 NON-FLOAT x y\n", encoding="utf-8")
        we = WordEmbedding(d_emb=4, seed=0)
        assert we.load_from_file(str(p), only={"absent"}) == 0


class TestMmapSessionRoundtrip:
    def test_vecs_survive_mmap_load(self, tmp_path):
        from gcn_python.cli.session import SessionStore
        from gcn_python.data.graph_vecs import stable_key
        d = tmp_path / "s"
        s = SessionStore(d)
        s.vecs[stable_key("phrase", 0)] = np.arange(8, dtype=np.float32)
        s.vecs_meta[stable_key("phrase", 0)] = {"node_id": "n"}
        s.save()
        s2 = SessionStore(d)
        s2.load()
        got = s2.lookup("phrase", 0)
        assert got is not None and np.array_equal(got, np.arange(8, dtype=np.float32))


class TestFloat32Discipline:
    def test_forward_stays_float32(self):
        # Gate M3b : aucune fuite float64 sur le chemin forward (poids + états).
        from gcn_python.layer1.representation import UDRepresentation
        pipe = make_test_pipeline(d_emb=16)
        reps = [UDRepresentation(
            tokens=[{"lemma": f"w{i}", "pos": "VERB", "dep_rel": "root", "morph": {}}],
            root_lemma=f"w{i}", root_pos="VERB", root_dep_rel="root",
            root_morph={}, subject_pos=None, has_object=False, has_advcl=False,
            has_temporal_obl=False, token_span=(i, i + 1)) for i in range(3)]
        pipe.forward(reps, "x")
        for arr in pipe.encoder.parameters():
            assert np.asarray(arr).dtype == np.float32
        for arr in pipe.word_embedding.parameters():
            assert np.asarray(arr).dtype == np.float32
        assert pipe._cached_clause_vecs.dtype == np.float32
        assert np.asarray(pipe.get_enriched_vectors()).dtype == np.float32
