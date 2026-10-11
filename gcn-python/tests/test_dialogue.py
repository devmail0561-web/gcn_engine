# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests DialogueSession — Phases 9b/9c/9d/9e (moteur language-agnostic).

Aucun lexique FR/EN dans les tests non plus : les scénarios utilisent des
labels arbitraires ; seuls des mécanismes structurels sont assertés
(continuité, chiffres, répétition de nom, scripts Unicode, codes).
"""
from __future__ import annotations

import copy

from gcn_python.dialogue import (
    DialogueSession,
    content_words,
    format_response,
    is_question_surface,
    parse_digits,
    script_tag,
)
from gcn_python.verbalizer.instructions import CausalGraph, InstructionHandler


def _graph():
    h = InstructionHandler()
    h.add_cir({
        "nodes": [{"id": "s", "label": "stress"}, {"id": "i", "label": "insomnie"},
                  {"id": "f", "label": "fatigue"}, {"id": "b", "label": "bruit"}],
        "edges": [
            ["s", "i", {"relation": "cause", "confidence": 0.9}],
            ["i", "f", {"relation": "cause", "confidence": 0.8}],
            ["b", "f", {"relation": "cause", "confidence": 0.7}],
        ],
        "source_text": "t",
    }, session_origin="document")
    return h


class _FakeEngine:
    """Moteur stub : CIR fixe, pas de pipeline (routage NLU surface)."""

    def __init__(self, cir):
        self._cir = cir
        self._pipeline = None

    def analyze(self, text):
        return copy.deepcopy(self._cir)


_QUESTION_CIR = {
    "nodes": [{"id": 0, "label": "cafe"}, {"id": 1, "label": "insomnie"}],
    "edges": [[0, 1, {"relation": "cause", "confidence": 0.9}]],
    "source_text": "q",
}


# ── 9b sujets ─────────────────────────────────────────────────────────────

class TestTopics:
    def test_topic_announced(self):
        s = DialogueSession()
        resp, info = s.handle("explain: insomnie", _graph())
        assert resp.startswith("  topic: insomnie\n")
        assert info["subject"] == "insomnie"
        assert "stress" in resp

    def test_continuity_without_recognized_concept(self):
        s = DialogueSession()
        h = _graph()
        s.handle("explain: insomnie", h)
        # "ses"/"effets" ne matchent aucun label → continuité sur insomnie
        resp, info = s.handle("et ses effets ?", h)
        assert info["subject"] == "insomnie"
        assert "fatigue" in resp  # effets de l'insomnie, pas du vide

    def test_digit_selects_last_candidates(self):
        s = DialogueSession()
        h = _graph()
        s.handle("explain: insomnie", h)
        resp, info = s.handle("1", h)
        assert info["subject"] == "insomnie"
        assert resp.startswith("  topic: insomnie\n")

    def test_reprise_two_turns_back_via_journal(self):
        s = DialogueSession()
        h = _graph()
        s.handle("explain: insomnie", h)       # tour 1 : sujet insomnie
        s.handle("stress insomnie", h)         # tour 2 : clarification (pas de sujet)
        resp, info = s.handle("et ensuite ?", h)  # tour 3 : continuité
        assert info["subject"] == "insomnie"  # sujet vieux de 2 tours, via journal
        assert len(s.journal) == 3

    def test_topics_capped_at_three(self):
        s = DialogueSession(max_topics=3)
        h = _graph()
        for q in ("explain: insomnie", "explain: fatigue", "explain: stress",
                  "explain: bruit"):
            s.handle(q, h)
        assert s.topics == ["insomnie", "fatigue", "stress"] or len(s.topics) == 3


# ── 9c clarification ──────────────────────────────────────────────────────

class TestClarify:
    def test_ask_on_two_exact_labels(self):
        s = DialogueSession()
        resp, info = s.handle("stress insomnie", _graph())
        assert resp == "  clarify: insomnie | stress"
        assert info["status"] == "clarification"
        assert s.pending is not None
        assert s.topics == []  # pas de sujet résolu → pile intacte

    def test_answer_by_number(self):
        s = DialogueSession()
        h = _graph()
        s.handle("stress insomnie", h)
        resp, info = s.handle("1", h)
        assert info["subject"] == "insomnie"
        assert info["status"] == "clarified"
        assert s.pending is None

    def test_answer_by_name(self):
        s = DialogueSession()
        h = _graph()
        s.handle("stress insomnie", h)
        resp, info = s.handle("stress !", h)
        assert info["subject"] == "stress"
        assert s.pending is None

    def test_ambiguous_answer_keeps_pending(self):
        s = DialogueSession()
        h = _graph()
        s.handle("stress insomnie", h)
        resp, info = s.handle("stress insomnie", h)
        assert s.pending is not None  # jamais de file, jamais 2 : rappel
        assert resp == "  clarify: insomnie | stress"
        assert info["status"] == "clarification"

    def test_never_two_pending(self):
        s = DialogueSession()
        h = _graph()
        s.handle("stress insomnie", h)   # pending A
        # Bascule nette vers une autre ambiguïté : abandon ANNONCÉ, pas de file
        resp, _ = s.handle("bruit fatigue", h)
        assert resp == ("  abandoned: insomnie | stress\n"
                        "  clarify: bruit | fatigue")
        assert s.pending["options"] == ["bruit", "fatigue"]

    def test_abandoned_after_three_turns(self):
        s = DialogueSession(abandon_after=3)
        h = _graph()
        s.handle("stress insomnie", h)          # tour 1 : demande
        s.handle("blah blah", h)                # tour 2 : vague (survit)
        assert s.pending is not None
        s.handle("blah blah", h)                # tour 3 : vague (survit)
        assert s.pending is not None
        s.handle("blah blah", h)                # tour 4 : âge 3, survit encore
        assert s.pending is not None
        resp, _ = s.handle("blah blah", h)      # tour 5 : âge 4 > 3 → abandon
        assert resp.startswith("  abandoned: insomnie | stress\n")
        assert s.pending is None

    def test_topic_switch_abandons(self):
        s = DialogueSession()
        h = _graph()
        s.handle("stress insomnie", h)
        resp, info = s.handle("explain: fatigue", h)
        assert resp.startswith("  abandoned: insomnie | stress\n")
        assert "topic: fatigue" in resp
        assert s.pending is None


# ── 9d langue ─────────────────────────────────────────────────────────────

class TestLang:
    def test_script_tags(self):
        assert script_tag("pourquoi l'insomnie ?") == "latin"
        assert script_tag("الأرق يسبب التعب") == "arabic"
        assert script_tag("123 ?!") is None

    def test_default_und_then_thread(self):
        s = DialogueSession()
        assert s.effective_lang() == "und"
        _, info = s.handle("explain: insomnie", _graph())
        assert info["lang"] == "latin"
        assert s.lang_thread == "latin"

    def test_explicit_instruction_persists(self):
        s = DialogueSession()
        h = _graph()
        resp, _ = s.handle("lang: wo", h)
        assert resp == "  lang: wo"
        _, info = s.handle("explain: insomnie", h)  # message latin : consigne > message
        assert info["lang"] == "wo"

    def test_lang_query_and_clear(self):
        s = DialogueSession()
        h = _graph()
        s.handle("lang: wo", h)
        resp, _ = s.handle("lang:", h)
        assert resp == "  lang: wo"
        resp, _ = s.handle("lang: auto", h)
        assert resp == "  lang: latin"  # retour au fil
        assert s.lang_override is None

    def test_counter_order(self):
        s = DialogueSession()
        h = _graph()
        s.handle("lang: wo", h)
        s.handle("lang: es", h)
        _, info = s.handle("explain: insomnie", h)
        assert info["lang"] == "es"


# ── 9e journal + révocation ────────────────────────────────────────────────

class TestJournalRevoke:
    def test_journal_shape(self):
        s = DialogueSession()
        h = _graph()
        s.handle("explain: insomnie", h)
        s.handle("zzz zzz", h)  # continuité → insomnie, répondu
        assert [e["n"] for e in s.journal] == [1, 2]
        for e in s.journal:
            assert set(e) >= {"n", "event", "question", "subject", "summary",
                              "status", "lang", "added"}
            assert e["event"] == "turn"
        assert s.journal[0]["status"] == "ok"
        assert s.journal[1]["subject"] == "insomnie"  # continuité enregistrée
        s2 = DialogueSession()  # session vierge : vraiment sans réponse
        _, info = s2.handle("zzz zzz", _graph())
        assert info["status"] == "unanswered"

    def test_from_history(self):
        s = DialogueSession()
        h = _graph()
        s.handle("explain: insomnie", h)
        s.handle("explain: fatigue", h)
        s2 = DialogueSession()
        s2.from_history(s.journal)
        assert s2.turn == 2
        assert s2.topics[-1] == "fatigue"
        assert s2.lang_thread == "latin"

    def test_revoke_removes_only_user_edges(self):
        s = DialogueSession()
        h = _graph()
        eng = _FakeEngine(_QUESTION_CIR)
        resp, info = s.handle("le cafe cause insomnie ?", h, engine=eng)
        assert info["added"] == 1
        stored = [e for e in h.graph.edges if e[2].get("session_turn") == 1]
        assert len(stored) == 1
        assert stored[0][2]["confidence"] == 0.9 * 0.5  # poids 9a à l'écriture
        resp, info = s.handle("revoke:", h)
        assert resp == "  revoked: turn 1 (1 edge(s))"
        assert info["status"] == "corrected"
        assert len(h.graph.edges) == 3  # corpus intact
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            causes = h.graph.find_causes("insomnie")
        assert len(causes) == 1 and h.graph._label(causes[0][0]) == "stress"

    def test_revoke_never_touches_corpus(self):
        s = DialogueSession()
        h = _graph()
        resp, _ = s.handle("revoke:", h)  # aucun apport utilisateur
        assert resp == "  revoked: nothing (no user-added turn)"
        assert len(h.graph.edges) == 3

    def test_revoke_explicit_turn(self):
        s = DialogueSession()
        h = _graph()
        eng = _FakeEngine(_QUESTION_CIR)
        s.handle("le cafe cause insomnie ?", h, engine=eng)
        s.handle("explain: fatigue", h)
        resp, _ = s.handle("revoke: 1", h)
        assert resp == "  revoked: turn 1 (1 edge(s))"
        assert len(h.graph.edges) == 3


# ── 9a enrichissement ─────────────────────────────────────────────────────

class TestEnrichment:
    def test_declarative_not_stored(self):
        s = DialogueSession()
        h = _graph()
        _, info = s.handle("le cafe cause insomnie", h, engine=_FakeEngine(_QUESTION_CIR))
        assert info["added"] == 0
        assert len(h.graph.edges) == 3

    def test_formal_command_not_stored(self):
        s = DialogueSession()
        h = _graph()
        _, info = s.handle("explain: insomnie ?", h, engine=_FakeEngine(_QUESTION_CIR))
        assert info["added"] == 0

    def test_engine_error_still_answers(self):
        class _Boom:
            _pipeline = None

            def analyze(self, text):
                raise RuntimeError("boom")

        s = DialogueSession()
        resp, info = s.handle("explain: insomnie ?", _graph(), engine=_Boom())
        assert info["added"] == 0
        assert "stress" in resp


# ── helpers ───────────────────────────────────────────────────────────────

class TestHelpers:
    def test_content_words_and_digits(self):
        # Pas de traitement d'élision (lexique) : "l'insomnie" reste tel quel
        assert "l'insomnie" in content_words("Pourquoi l'insomnie ?")
        assert parse_digits("le 2 et 10") == [2, 10]
        assert parse_digits("٢") == [2]  # chiffres arabes (Unicode)

    def test_is_question_surface(self):
        assert is_question_surface("pourquoi ?")
        assert is_question_surface("لماذا؟")
        assert not is_question_surface("le stress cause tout.")

    def test_format_response_subject_and_legacy(self):
        h = _graph()
        out = format_response(h, "et ses effets ?", subject="insomnie")
        assert "fatigue" in out
        out2 = format_response(h, "et ses effets fatigue")
        assert "fatigue" in out2  # repli dernier mot


def test_giant_input_echo_capped():
    """20k d'entrée → réponse bornée (pas d'amplification 2x)."""
    from gcn_python.dialogue import ECHO_SUBJECT_MAX
    h = _graph()
    d = DialogueSession()
    r, _ = d.handle("x" * 20000, h)
    assert len(r) < 3 * ECHO_SUBJECT_MAX
    assert ("x" * (ECHO_SUBJECT_MAX + 1)) not in r  # jamais de plage non tronquée


def test_nlu_failure_counted_not_silent(monkeypatch):
    """NLU cassé → repli intact + compteur (plus de silence)."""
    import gcn_python.pipeline.nlu_routing as _nr
    from gcn_python.dialogue import nlu_fallback_counts
    from types import SimpleNamespace
    monkeypatch.setattr(_nr, "nlu_route", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    before = nlu_fallback_counts()["nlu_route"]
    h = _graph()
    eng = SimpleNamespace(analyze=lambda t: {"edges": []},
                          _pipeline=SimpleNamespace(_cached_reps=None,
                                                    _cached_enriched_vecs=None))
    d = DialogueSession()
    r, info = d.handle("parle-moi d'insomnie", h, engine=eng)
    assert "fatigue" in r  # repli format_response intact
    assert nlu_fallback_counts()["nlu_route"] == before + 1
    assert info["nlu_fallbacks"]["nlu_route"] == before + 1
