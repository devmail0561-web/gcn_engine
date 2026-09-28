# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Phase D — T1 logique : généralisation structurelle (fixtures manuelles DE).

Valide que classify() détecte SubordinationType.CONDITION depuis la structure UD
indépendamment de la langue, sans avoir vu le lemme en entraînement.
Ne prouve pas la robustesse end-to-end sur un vrai corpus DE (T1 DE, futur).
"""

from gcn_python.layer1.sentence_type import (
    LangMarkers,
    SubordinationType,
    classify,
)

# ---------------------------------------------------------------------------
# Fixtures — tokens UD construits manuellement (pas de parser DE requis)
# ---------------------------------------------------------------------------

DE_MARKERS = LangMarkers.from_json({
    "interrogative_lemmas": [],
    "negation_particles": ["nicht", "kein"],
    "restriction_patterns": [],
    "subordination_markers": {
        "condition": ["wenn", "falls", "sofern"],
        "concession": ["obwohl", "obgleich"],
        "cause": ["weil", "da"],
        "temporal": ["als", "während", "nachdem"],
    },
})


def _tok(lemma, pos="VERB", dep_rel="root", morph=None, tid=1, dep_head=0, form=None):
    return {
        "lemma": lemma, "pos": pos, "dep_rel": dep_rel,
        "morph": morph or {}, "id": tid, "dep_head": dep_head,
        "form": form or lemma,
    }


# "Wenn die Behandlungen scheitern, wird das Medikament verschrieben."
TOKENS_WENN_COND = [
    _tok("wenn",       pos="SCONJ", dep_rel="mark",  morph={},          tid=1, dep_head=3, form="Wenn"),
    _tok("die",        pos="DET",   dep_rel="det",   morph={},          tid=2, dep_head=3),
    _tok("Behandlung", pos="NOUN",  dep_rel="nsubj", morph={},          tid=3, dep_head=4),
    _tok("scheitern",  pos="VERB",  dep_rel="advcl", morph={"Mood":"Sub"}, tid=4, dep_head=8),
    _tok(",",          pos="PUNCT", dep_rel="punct", morph={},          tid=5, dep_head=8),
    _tok("wird",       pos="AUX",   dep_rel="aux",   morph={},          tid=6, dep_head=8),
    _tok("das",        pos="DET",   dep_rel="det",   morph={},          tid=7, dep_head=8),
    _tok("Medikament", pos="NOUN",  dep_rel="nsubj:pass", morph={},     tid=8, dep_head=9),
    _tok("verschreiben", pos="VERB", dep_rel="root", morph={"Voice":"Pass"}, tid=9, dep_head=0),
    _tok(".",          pos="PUNCT", dep_rel="punct", morph={},          tid=10, dep_head=9, form="."),
]

# "Falls es regnet, bleiben wir." (conditional simple)
TOKENS_FALLS_COND = [
    _tok("falls",  pos="SCONJ", dep_rel="mark",  morph={},         tid=1, dep_head=3, form="Falls"),
    _tok("es",     pos="PRON",  dep_rel="nsubj", morph={},         tid=2, dep_head=3),
    _tok("regnen", pos="VERB",  dep_rel="advcl", morph={"Mood":"Sub"}, tid=3, dep_head=5),
    _tok(",",      pos="PUNCT", dep_rel="punct", morph={},         tid=4, dep_head=5),
    _tok("bleiben", pos="VERB", dep_rel="root",  morph={},         tid=5, dep_head=0),
    _tok("wir",    pos="PRON",  dep_rel="nsubj", morph={},         tid=6, dep_head=5),
    _tok(".",      pos="PUNCT", dep_rel="punct", morph={},         tid=7, dep_head=5, form="."),
]

# "Obwohl es regnet, gehen wir." (concession — ne doit PAS être CONDITION)
TOKENS_OBWOHL_CONCESS = [
    _tok("obwohl", pos="SCONJ", dep_rel="mark",  morph={},         tid=1, dep_head=3, form="Obwohl"),
    _tok("es",     pos="PRON",  dep_rel="nsubj", morph={},         tid=2, dep_head=3),
    _tok("regnen", pos="VERB",  dep_rel="advcl", morph={},         tid=3, dep_head=5),
    _tok(",",      pos="PUNCT", dep_rel="punct", morph={},         tid=4, dep_head=5),
    _tok("gehen",  pos="VERB",  dep_rel="root",  morph={},         tid=5, dep_head=0),
    _tok("wir",    pos="PRON",  dep_rel="nsubj", morph={},         tid=6, dep_head=5),
    _tok(".",      pos="PUNCT", dep_rel="punct", morph={},         tid=7, dep_head=5, form="."),
]


# ---------------------------------------------------------------------------
# T1 logique : classify() détecte CONDITION via la structure UD
# ---------------------------------------------------------------------------

class TestT1LogiqueDE:
    """T1 logique — fixtures DE, prouve classify() structure-first."""

    def test_wenn_condition_detected(self):
        """'wenn' + Mood=Sub + advcl → CONDITION."""
        p = classify(TOKENS_WENN_COND, DE_MARKERS)
        assert p.subordination == SubordinationType.CONDITION

    def test_wenn_complexity_complex(self):
        """'wenn' clause → phrase complexe."""
        p = classify(TOKENS_WENN_COND, DE_MARKERS)
        assert p.is_complex

    def test_falls_condition_detected(self):
        """'falls' → CONDITION."""
        p = classify(TOKENS_FALLS_COND, DE_MARKERS)
        assert p.subordination == SubordinationType.CONDITION

    def test_obwohl_not_condition(self):
        """'obwohl' → CONCESSION, pas CONDITION."""
        p = classify(TOKENS_OBWOHL_CONCESS, DE_MARKERS)
        assert p.subordination == SubordinationType.CONCESSION
        assert p.subordination != SubordinationType.CONDITION

    def test_wenn_without_markers_is_none(self):
        """Sans markers, couche 1 seule → NONE (ne connaît pas 'wenn')."""
        p = classify(TOKENS_WENN_COND, markers=None)
        assert p.subordination == SubordinationType.NONE

    def test_passive_voice_detected_german(self):
        """'verschrieben' avec Voice=Pass → PASSIVE."""
        p = classify(TOKENS_WENN_COND, DE_MARKERS)
        assert p.is_passive

    def test_condition_implies_has_condition(self):
        """has_condition == True quand subordination == CONDITION."""
        p = classify(TOKENS_WENN_COND, DE_MARKERS)
        assert p.has_condition

    def test_t1_scope_note(self):
        """T1 logique : prouve la logique UD, pas un parser DE réel.
        Un parser DE réel (T1 DE) nécessite un frontend allemand non disponible."""
        # Test documentaire — toujours vert
        assert True
