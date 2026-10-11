# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests Phase 10-M2 — borner la session : compacter au lieu de refuser.

M2a : fusion doublons (nœuds/arêtes), plafonds + annonce `compacted:`,
résumé vieux tours (sujets gardés), cap vecs. Zéro refus.
M2b : photo périodique + journal des ajouts (roundtrip, cadence, destruction).
M2c : paraphrases (overlap trigrammes + seuil + repli exact).
"""
from __future__ import annotations

import pytest

from gcn_python.cli.session import SessionStore
from gcn_python.dialogue import DialogueSession, paraphrase_overlap
from gcn_python.verbalizer.instructions import CausalGraph, InstructionHandler


def _graph_dup():
    h = InstructionHandler()
    h.add_cir({
        "nodes": [{"id": "a", "label": "stress"}, {"id": "b", "label": "Stress"},
                  {"id": "c", "label": "insomnie"}],
        "edges": [
            ["a", "c", {"relation": "cause", "confidence": 0.9}],
            ["b", "c", {"relation": "cause", "confidence": 0.9}],
            ["a", "c", {"relation": "cause", "confidence": 0.9}],
        ],
        "source_text": "t",
    }, session_origin="document")
    return h


class TestMerge:
    def test_merge_duplicate_nodes(self):
        g = _graph_dup().graph
        assert len(g.nodes) == 3
        removed = g.merge_duplicate_nodes()
        assert removed == 1
        assert len(g.nodes) == 2
        # Arêtes recâblées vers le gardé (plus petit id)
        assert all(s != "b" for s, _, _, _ in g.edges)
        assert len(g.edges) == 3

    def test_merge_respects_origin_precedence(self):
        g = CausalGraph()
        g.add_cir({"nodes": [{"id": "q", "label": "doute"}],
                   "edges": []}, session_origin="question_utilisateur")
        g.add_cir({"nodes": [{"id": "d", "label": "doute"}],
                   "edges": []}, session_origin="document")
        assert g.merge_duplicate_nodes() == 1
        keeper = g.nodes["d"] if "d" in g.nodes else g.nodes["q"]
        assert keeper["session_origin"] == "document"  # jamais de downgrade

    def test_merge_duplicate_edges(self):
        g = _graph_dup().graph
        g.merge_duplicate_nodes()
        removed = g.merge_duplicate_edges()
        assert removed == 2  # 3 arêtes identiques → 1
        assert len(g.edges) == 1

    def test_merge_keeps_distinct_conf(self):
        g = CausalGraph()
        g.add_cir({"nodes": [{"id": "a", "label": "x"}, {"id": "b", "label": "y"}],
                   "edges": [["a", "b", {"relation": "cause", "confidence": 0.9}],
                             ["a", "b", {"relation": "cause", "confidence": 0.5}]],
                   "source_text": "t"}, session_origin="document")
        assert g.merge_duplicate_edges() == 0

    def test_merge_ignores_volatile_turn(self):
        g = CausalGraph()
        g.add_cir({"nodes": [{"id": "a", "label": "x"}, {"id": "b", "label": "y"}],
                   "edges": [["a", "b", {"relation": "cause", "confidence": 0.9}]],
                   "source_text": "t"}, session_origin="question_utilisateur")
        g.edges[0][2]["session_turn"] = 1
        g.add_cir({"nodes": [{"id": "a", "label": "x"}, {"id": "b", "label": "y"}],
                   "edges": [["a", "b", {"relation": "cause", "confidence": 0.9}]],
                   "source_text": "t"}, session_origin="question_utilisateur")
        g.edges[1][2]["session_turn"] = 5
        # Même fait (0.9×0.5=0.45 les deux), tours différents → fusionne quand même
        assert g.merge_duplicate_edges() == 1


class TestCapsAnnounced:
    def test_overflow_compacts_never_refuses(self):
        s = DialogueSession(max_nodes=2, max_edges=2)
        h = _graph_dup()  # 3 nœuds (2 stress), 3 arêtes (2 identiques)
        resp, info = s.handle("explain: insomnie", h)
        assert resp.startswith("  compacted: merged 1 node(s), merged 2 duplicate edge(s)\n")
        assert "stress" in resp  # réponse quand même : zéro refus
        assert len(h.graph.nodes) == 2 and len(h.graph.edges) == 1

    def test_under_caps_silent(self):
        s = DialogueSession()
        resp, _ = s.handle("explain: insomnie", _graph_dup())
        assert not resp.startswith("  compacted:")


class TestJournalSummary:
    def test_old_turns_summarized_not_dropped(self):
        s = DialogueSession(max_turns=3)
        h = _graph_dup()
        for i in range(5):
            s.handle(f"explain: insomnie", h)
        assert len(s.journal) == 3  # 1 résumé + 2 tours
        assert s.journal[0]["event"] == "summary"
        assert s.journal[0]["turns"] == [1, 2, 3]
        assert "insomnie" in s.journal[0]["subjects"]
        assert [e["n"] for e in s.journal[1:]] == [4, 5]

    def test_summary_persisted_via_record_summary(self, tmp_path):
        s, store = DialogueSession(max_turns=2), SessionStore(tmp_path / "s")
        h = _graph_dup()
        for _ in range(3):
            _, info = s.handle("explain: insomnie", h)
            if info.get("summarized") is not None:
                store.record_summary(info["summarized"])
        kinds = [e["event"] for e in store.history]
        assert "summary" in kinds
        store.graph = h.graph
        store.save()
        s2 = SessionStore(tmp_path / "s")
        s2.load()
        assert any(e["event"] == "summary" for e in s2.history)


class TestVecsCap:
    def test_oldest_vecs_evicted_on_collect(self):
        import numpy as np
        from types import SimpleNamespace

        store = SessionStore(None, max_vecs=3)
        for i in range(5):
            eng = SimpleNamespace(_pipeline=SimpleNamespace(
                get_enriched_vectors=lambda i=i: np.ones((1, 2), dtype=np.float32) * i,
                encoder=SimpleNamespace(parameters=lambda: iter([])),
            ))
            cir = {"nodes": [{"id": 0, "label": f"n{i}"}], "source_text": f"phrase {i}"}
            assert store.collect(eng, f"phrase {i}", cir) == 1
        assert len(store.vecs) == 3
        assert store.lookup("phrase 0", 0) is None  # évincé (FIFO)
        assert store.lookup("phrase 4", 0) is not None


class TestIncrementalSave:
    def _cir(self, tag, src="a", dst="b"):
        return {"nodes": [{"id": f"{tag}s", "label": src}, {"id": f"{tag}d", "label": dst}],
                "edges": [[f"{tag}s", f"{tag}d",
                           {"relation": "cause", "confidence": 0.9}]],
                "source_text": tag}

    def test_log_then_replay(self, tmp_path):
        d = tmp_path / "s"
        s = SessionStore(d, snapshot_every=10)
        s.graph.add_cir(self._cir("1"), session_origin="document")
        s.save()  # photo
        assert not (d / "graph_log.jsonl").exists()
        s.graph.add_cir(self._cir("2"), session_origin="document")
        s.save()  # journal
        assert (d / "graph_log.jsonl").exists()
        s2 = SessionStore(d)
        rep = s2.load()
        assert rep["graph_edges"] == 2
        assert rep["replayed"] == 1
        assert rep["resnapshotted"] is True
        assert not (d / "graph_log.jsonl").exists()  # convergé
        assert len(s2.graph.nodes) == 4

    def test_snapshot_cadence(self, tmp_path):
        d = tmp_path / "s"
        s = SessionStore(d, snapshot_every=2)
        s.graph.add_cir(self._cir("1"), session_origin="document")
        s.save()  # 1 : photo
        s.graph.add_cir(self._cir("2"), session_origin="document")
        s.save()  # 2 : journal
        assert (d / "graph_log.jsonl").exists()
        s.graph.add_cir(self._cir("3"), session_origin="document")
        s.save()  # 3 : photo ((3-1)%2==0)
        assert not (d / "graph_log.jsonl").exists()
        s2 = SessionStore(d)
        assert s2.load()["graph_edges"] == 3

    def test_destruction_forces_photo(self, tmp_path):
        d = tmp_path / "s"
        s = SessionStore(d, snapshot_every=100)
        s.graph.add_cir(self._cir("1"), session_origin="document")
        s.graph.add_cir(self._cir("2"), session_origin="document")
        s.save()  # photo (2 arêtes)
        s.graph.drop_edges(lambda _s, _d, _a: True)  # revoke tout
        assert len(s.graph.edges) == 0
        s.save()  # destruction → photo forcée, pas de journal menteur
        s2 = SessionStore(d)
        assert s2.load()["graph_edges"] == 0  # rien ressuscité

    def test_state_equals_full_save(self, tmp_path):
        d1, d2 = tmp_path / "s1", tmp_path / "s2"
        s1 = SessionStore(d1, snapshot_every=100)
        for t in ("1", "2", "3"):
            s1.graph.add_cir(self._cir(t), session_origin="document")
            s1.save()
        s2 = SessionStore(d2, snapshot_every=1)  # tout en photos
        for t in ("1", "2", "3"):
            s2.graph.add_cir(self._cir(t), session_origin="document")
            s2.save()
        r1 = SessionStore(d1)
        r2 = SessionStore(d2)
        r1.load()
        r2.load()
        assert r1.graph.nodes == r2.graph.nodes
        assert [(s, d, a, t) for s, d, a, t in r1.graph.edges] == [
            (s, d, a, t) for s, d, a, t in r2.graph.edges]


class TestParaphrase:
    def test_overlap_values(self):
        assert paraphrase_overlap("fatigue", "fatigues") >= 0.75
        assert paraphrase_overlap("eau", "bateau") < 0.75  # 0.67 via "au#"
        assert paraphrase_overlap("os", "osier") == 0.0  # trop court : bruit refusé
        assert paraphrase_overlap("stress", "insomnie") < 0.75

    def test_morphological_variant_resolves(self):
        s = DialogueSession()
        h = InstructionHandler()
        h.add_cir({"nodes": [{"id": "f", "label": "fatigue"}],
                   "edges": [], "source_text": "t"}, session_origin="document")
        # "fatigues" : ni exact ni word-boundary → paraphrase
        assert s.resolve("il a des fatigues", h.graph) == "fatigue"

    def test_exact_still_wins(self):
        s = DialogueSession()
        h = InstructionHandler()
        h.add_cir({"nodes": [{"id": "f", "label": "fatigue"},
                             {"id": "g", "label": "fatigues"}],
                   "edges": [], "source_text": "t"}, session_origin="document")
        assert s.resolve("fatigue", h.graph) == "fatigue"  # exact d'abord

    def test_multiple_paraphrases_fall_through(self):
        s = DialogueSession()
        h = InstructionHandler()
        h.add_cir({"nodes": [{"id": "a", "label": "stress"},
                             {"id": "b", "label": "stresse"}],
                   "edges": [], "source_text": "t"}, session_origin="document")
        # "astres" sous le seuil pour les 2 (0.5 < 0.75) → pas de clarify
        # sur du flou : repli mot (continuité vide, topics vides).
        assert s.resolve("astres", h.graph) == "astres"


class TestCorruptTolerance:
    def _cir(self, tag):
        return {"nodes": [{"id": f"{tag}s", "label": "a"}, {"id": f"{tag}d", "label": "b"}],
                "edges": [[f"{tag}s", f"{tag}d",
                           {"relation": "cause", "confidence": 0.9}]],
                "source_text": tag}

    def test_corrupt_history_line_keeps_tail(self, tmp_path):
        import json
        d = tmp_path / "s"
        s = SessionStore(d)
        s.history.append({"event": "turn", "n": 1})
        s.history.append({"event": "turn", "n": 2})
        s.save()
        with open(d / "history.jsonl", "a", encoding="utf-8") as f:
            f.write("{corrompu\n")
            f.write(json.dumps({"event": "turn", "n": 3}) + "\n")
        s2 = SessionStore(d)
        with pytest.warns(UserWarning):
            rep = s2.load()
        assert rep["history"] == 3  # queue préservée malgré la ligne pourrie
        assert rep["skipped_history"] == 1
        assert [e["n"] for e in s2.history if e.get("event") == "turn"] == [1, 2, 3]

    def test_corrupt_log_line_keeps_rest(self, tmp_path):
        d = tmp_path / "s"
        s = SessionStore(d, snapshot_every=100)
        s.graph.add_cir(self._cir("1"), session_origin="document")
        s.save()  # photo
        s.graph.add_cir(self._cir("2"), session_origin="document")
        s.save()  # journal : 1 entrée
        with open(d / "graph_log.jsonl", "a", encoding="utf-8") as f:
            f.write("{corrompu\n")
        s2 = SessionStore(d)
        with pytest.warns(UserWarning):
            rep = s2.load()
        assert rep["graph_edges"] == 2
        assert rep["replayed"] == 1
        assert rep["skipped"] == 1
