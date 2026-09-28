# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Classifieur de type et forme de phrase depuis les features UD (D1).

Couche 1 (UD structurel pur — zéro lemme) + couche 2 optionnelle (LangMarkers).
Entrée : liste de tokens UD [{lemma, pos, dep_rel, morph, id, dep_head, form}]
Sortie : SentenceProfile (7 champs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SentenceType(str, Enum):
    DECLARATIVE = "declarative"
    INTERROGATIVE = "interrogative"
    IMPERATIVE = "imperative"
    EXCLAMATIVE = "exclamative"


class Polarity(str, Enum):
    AFFIRMATIVE = "affirmative"
    NEGATIVE = "negative"


class Voice(str, Enum):
    ACTIVE = "active"
    PASSIVE = "passive"


class Modality(str, Enum):
    INDICATIVE = "indicative"
    SUBJUNCTIVE = "subjunctive"
    CONDITIONAL = "conditional"
    IMPERATIVE = "imperative"


class Complexity(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class SubordinationType(str, Enum):
    NONE = "none"
    CONDITION = "condition"
    CAUSE = "cause"
    CONCESSION = "concession"
    TEMPORAL = "temporal"
    MOTIVATION = "motivation"
    FILTER = "filter"
    SEQUENCE = "sequence"
    OPPOSITION = "opposition"
    ENABLE = "enable"


@dataclass(frozen=True)
class LangMarkers:
    interrogative_lemmas: frozenset[str] = frozenset()
    negation_particles: frozenset[str] = frozenset()
    restriction_patterns: tuple = ()
    subordination_markers: dict = field(default_factory=dict)
    discourse_connectors: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict) -> "LangMarkers":
        return cls(
            interrogative_lemmas=frozenset(
                str(x).lower() for x in data.get("interrogative_lemmas", [])
            ),
            negation_particles=frozenset(
                str(x).lower() for x in data.get("negation_particles", [])
            ),
            restriction_patterns=tuple(data.get("restriction_patterns", [])),
            subordination_markers=dict(data.get("subordination_markers", {})),
            discourse_connectors=dict(data.get("discourse_connectors", {})),
        )

    @classmethod
    def load(cls, path) -> "LangMarkers":
        import json

        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class SentenceProfile:
    sentence_type: SentenceType
    polarity: Polarity
    voice: Voice
    modality: Modality
    complexity: Complexity = Complexity.SIMPLE
    subordination: SubordinationType = SubordinationType.NONE
    has_restriction: bool = False

    @property
    def expects_response(self) -> bool:
        return self.sentence_type == SentenceType.INTERROGATIVE

    @property
    def expects_action(self) -> bool:
        return self.sentence_type == SentenceType.IMPERATIVE

    @property
    def asserts_fact(self) -> bool:
        return self.sentence_type == SentenceType.DECLARATIVE

    @property
    def is_negated(self) -> bool:
        return self.polarity == Polarity.NEGATIVE

    @property
    def is_passive(self) -> bool:
        return self.voice == Voice.PASSIVE

    @property
    def is_complex(self) -> bool:
        return self.complexity == Complexity.COMPLEX

    @property
    def has_condition(self) -> bool:
        return self.subordination == SubordinationType.CONDITION


def classify(tokens: list[dict], markers: LangMarkers | None = None) -> SentenceProfile:
    """Classifie le type et la forme d'une phrase depuis ses tokens UD."""
    if not tokens:
        return SentenceProfile(
            SentenceType.DECLARATIVE, Polarity.AFFIRMATIVE,
            Voice.ACTIVE, Modality.INDICATIVE,
        )

    final_punct = _final_punctuation(tokens)
    root = _find_root(tokens)
    root_morph = (root or {}).get("morph", {}) if root else {}
    if not isinstance(root_morph, dict):
        root_morph = {}

    stype = _classify_type(tokens, root, root_morph, final_punct, markers)
    polarity = _classify_polarity(tokens, root_morph, markers)
    voice = _classify_voice(tokens, root_morph)
    modality = _classify_modality(root_morph, stype)
    complexity = _classify_complexity(tokens)
    subordination = _classify_subordination(tokens, markers)
    restriction = _classify_restriction(tokens, markers)

    return SentenceProfile(stype, polarity, voice, modality, complexity,
                           subordination, restriction)


def _final_punctuation(tokens: list[dict]) -> str:
    for t in reversed(tokens):
        form = t.get("form", "")
        if form in (".", "?", "!", "...", "?!", "!?"):
            return form
        if t.get("pos") != "PUNCT":
            break
    return "."


def _find_root(tokens: list[dict]) -> dict | None:
    for t in tokens:
        if str(t.get("dep_rel", "")).upper() == "ROOT":
            return t
        try:
            if int(t.get("dep_head", -1)) == 0:
                return t
        except (TypeError, ValueError):
            pass
    return tokens[0] if tokens else None


def _classify_type(tokens: list[dict], root: dict | None,
                   root_morph: dict, final_punct: str,
                   markers: LangMarkers | None = None) -> SentenceType:
    if final_punct in ("?", "?!"):
        return SentenceType.INTERROGATIVE

    # Couche 1 : PronType=Int structurel
    for t in tokens:
        morph = t.get("morph", {}) or {}
        if morph.get("PronType") == "Int" and t.get("dep_rel") in (
            "advmod", "det", "nsubj", "obl", "mark", "obj", "root",
        ):
            return SentenceType.INTERROGATIVE

    # Couche 2 : fallback lemmes via markers
    if markers is not None and markers.interrogative_lemmas:
        for t in tokens:
            if str(t.get("lemma", "")).lower() in markers.interrogative_lemmas:
                return SentenceType.INTERROGATIVE

    if final_punct == "!" and root_morph.get("Mood") != "Imp":
        return SentenceType.EXCLAMATIVE

    if root_morph.get("Mood") == "Imp":
        return SentenceType.IMPERATIVE
    # VerbForm=Inf sans sujet → impératif
    if root_morph.get("VerbForm") == "Inf":
        if not any(t.get("dep_rel") in ("nsubj", "nsubj:pass", "expl:subj", "csubj")
                   for t in tokens):
            return SentenceType.IMPERATIVE
    if root and root.get("pos") == "VERB":
        has_subject = any(
            t.get("dep_rel") in ("nsubj", "nsubj:pass", "expl:subj", "csubj")
            for t in tokens
        )
        if not has_subject:
            person = root_morph.get("Person", "")
            if person in ("1", "2") and root_morph.get("Mood") != "Ind":
                return SentenceType.IMPERATIVE

    # Inversion sujet-verbe : nsubj.id > root.id
    if root:
        root_idx = root.get("id", -1)
        if isinstance(root_idx, int) and root.get("pos") == "VERB":
            for t in tokens:
                if t.get("dep_rel") == "nsubj":
                    subj_idx = t.get("id", -2)
                    if isinstance(subj_idx, int) and subj_idx > root_idx:
                        return SentenceType.INTERROGATIVE

    return SentenceType.DECLARATIVE


def _classify_polarity(tokens: list[dict], root_morph: dict,
                       markers: LangMarkers | None = None) -> Polarity:
    if root_morph.get("Polarity") == "Neg":
        return Polarity.NEGATIVE
    for t in tokens:
        if (t.get("morph", {}) or {}).get("Polarity") == "Neg":
            return Polarity.NEGATIVE
    # Couche 2 : particules via markers + advmod
    if markers is not None and markers.negation_particles:
        for t in tokens:
            if (t.get("dep_rel") == "advmod"
                    and str(t.get("lemma", "")).lower() in markers.negation_particles):
                return Polarity.NEGATIVE
    return Polarity.AFFIRMATIVE


def _classify_voice(tokens: list[dict], root_morph: dict) -> Voice:
    if root_morph.get("Voice") == "Pass":
        return Voice.PASSIVE
    for t in tokens:
        if t.get("dep_rel") in ("aux:pass", "nsubj:pass"):
            return Voice.PASSIVE
    return Voice.ACTIVE


def _classify_modality(root_morph: dict, stype: SentenceType) -> Modality:
    mood = root_morph.get("Mood", "")
    if mood == "Sub":
        return Modality.SUBJUNCTIVE
    if mood == "Cnd":
        return Modality.CONDITIONAL
    if mood == "Imp" or stype == SentenceType.IMPERATIVE:
        return Modality.IMPERATIVE
    return Modality.INDICATIVE


def _classify_complexity(tokens: list[dict]) -> Complexity:
    for t in tokens:
        if t.get("dep_rel") in ("advcl", "ccomp", "xcomp", "acl"):
            morph = t.get("morph", {}) or {}
            if morph.get("VerbForm", "Fin") == "Fin":
                return Complexity.COMPLEX
    # SCONJ + mark explicite → complexe même sans advcl
    for t in tokens:
        if t.get("dep_rel") == "mark" and t.get("pos") == "SCONJ":
            return Complexity.COMPLEX
    return Complexity.SIMPLE


def _classify_subordination(tokens: list[dict],
                            markers: LangMarkers | None) -> SubordinationType:
    # Couche 1 : NONE (structure seule ne distingue pas condition/cause/concession)
    if markers is None or not markers.subordination_markers:
        return SubordinationType.NONE
    lemmas = {str(t.get("lemma", "")).lower()
              for t in tokens if t.get("dep_rel") == "mark" or t.get("pos") == "SCONJ"}
    for sub_type, marker_list in markers.subordination_markers.items():
        try:
            st = SubordinationType(sub_type)
        except ValueError:
            continue
        for m in (marker_list or []):
            if str(m).lower() in lemmas:
                return st
    return SubordinationType.NONE


def _classify_restriction(tokens: list[dict],
                          markers: LangMarkers | None) -> bool:
    # Couche 1 : False
    if markers is None:
        return False
    patterns = list(markers.restriction_patterns or [])
    if not patterns:
        return False
    lemmas = [str(t.get("lemma", "")).lower() for t in tokens]
    for pat in patterns:
        if isinstance(pat, (list, tuple)) and len(pat) >= 2:
            wanted = [str(x).lower() for x in pat]
            for i in range(len(lemmas) - len(wanted) + 1):
                if lemmas[i:i + len(wanted)] == wanted:
                    return True
        elif isinstance(pat, str) and pat.lower() in lemmas:
            return True
    return False
