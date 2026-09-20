# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Scoring de phrases par type de relation causale (FR+EN)."""
from __future__ import annotations
import re

RELATION_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "cause": {
        "fr": [
            "parce que", "car", "puisque", "donc", "ainsi", "c'est pourquoi",
            "entraîne", "provoque", "engendre", "à cause de", "en raison de",
            "résulte", "conduit à", "mène à",
        ],
        "en": [
            "because", "since", "therefore", "thus", "hence", "causes",
            "leads to", "results in", "due to", "as a result", "consequently",
            "triggers", "produces", "generates",
        ],
    },
    "enable": {
        "fr": [
            "permet", "permet de", "rend possible", "facilite", "autorise",
            "grâce à", "en permettant", "donne accès", "ouvre la voie",
        ],
        "en": [
            "enables", "allows", "makes possible", "facilitates", "permits",
            "thanks to", "gives access", "unlocks", "makes it possible",
        ],
    },
    "prevent": {
        "fr": [
            "empêche", "prévient", "évite", "bloque", "interdit", "inhibe",
            "sans lequel", "sinon", "à défaut", "freine", "entrave",
        ],
        "en": [
            "prevents", "avoids", "blocks", "inhibits", "stops", "hinders",
            "without which", "otherwise", "unless", "precludes",
        ],
    },
    "condition": {
        "fr": [
            "si", "à condition que", "en cas de", "lorsque", "quand",
            "sous réserve", "pourvu que", "dès lors que", "dès que",
            "dans le cas où", "à moins que", "sous condition",
        ],
        "en": [
            "if", "provided that", "in case", "when", "as long as",
            "unless", "given that", "conditional on", "assuming", "whenever",
            "on condition that",
        ],
    },
    "concession": {
        "fr": [
            "bien que", "même si", "quoique", "malgré", "en dépit de",
            "pourtant", "cependant", "néanmoins", "toutefois", "certes",
            "quand bien même",
        ],
        "en": [
            "although", "even though", "despite", "however", "nevertheless",
            "nonetheless", "yet", "while", "albeit", "even if", "granted",
        ],
    },
    "sequence": {
        "fr": [
            "d'abord", "puis", "ensuite", "enfin", "après", "avant",
            "premièrement", "deuxièmement", "finalement", "en premier lieu",
            "suite à", "consécutivement", "ultérieurement",
        ],
        "en": [
            "first", "then", "next", "finally", "after", "before",
            "subsequently", "following", "prior to", "once", "step",
            "in sequence", "thereafter",
        ],
    },
    "motivation": {
        "fr": [
            "afin de", "pour", "dans le but de", "en vue de", "vise à",
            "cherche à", "a pour objectif", "dans l'intention de",
            "de façon à", "de manière à",
        ],
        "en": [
            "in order to", "so as to", "aims to", "seeks to", "intended to",
            "with the goal of", "to achieve", "motivated by", "driven by",
        ],
    },
    "filter": {
        "fr": [
            "sauf", "sauf si", "excepté", "hormis", "à l'exception de",
            "sous condition de", "sous réserve que", "filtré par",
            "limité à", "restreint à",
        ],
        "en": [
            "except", "excluding", "apart from", "filtered by",
            "limited to", "restricted to", "only if", "but not",
        ],
    },
    "opposition": {
        "fr": [
            "mais", "en revanche", "au contraire", "à l'opposé",
            "tandis que", "alors que", "contrairement à", "par opposition",
            "au lieu de",
        ],
        "en": [
            "but", "on the contrary", "whereas",
            "contrary to", "as opposed to", "instead of", "in contrast",
        ],
    },
    "data_dependency": {
        "fr": [
            "dépend de", "repose sur", "utilise les données de",
            "prend en entrée", "lit depuis", "accède à", "requiert",
            "basé sur les données",
        ],
        "en": [
            "depends on", "relies on", "reads from", "uses data from",
            "takes as input", "accesses", "requires data", "based on data",
            "queries", "fetches from",
        ],
    },
    "control_dependency": {
        "fr": [
            "appelle", "déclenche", "invoque", "contrôle", "orchestre",
            "gère l'exécution", "pilote", "dirige",
        ],
        "en": [
            "calls", "invokes", "triggers", "controls", "orchestrates",
            "manages", "dispatches", "delegates to", "spawns", "fires",
        ],
    },
}

# Pré-compilation des patterns regex pour la performance
_COMPILED: dict[str, dict[str, list[re.Pattern]]] = {}
for _rel, _langs in RELATION_KEYWORDS.items():
    _COMPILED[_rel] = {}
    for _lang, _kws in _langs.items():
        _COMPILED[_rel][_lang] = [
            re.compile(r'(?<!\w)' + re.escape(kw) + r'(?!\w)', re.IGNORECASE)
            for kw in _kws
        ]

_FR_MARKERS = {"le", "la", "les", "de", "du", "des", "et", "est", "une", "un", "en", "à", "que", "pas", "plus"}
_EN_MARKERS = {"the", "is", "are", "was", "were", "and", "of", "to", "in", "that", "it", "with", "for"}


class CausalScorer:
    """Détecte les types de relations causales dans une phrase."""

    def detect_lang(self, text: str) -> str:
        """Détecte 'fr' ou 'en' par stopwords."""
        words = set(text.lower().split())
        return "fr" if len(words & _FR_MARKERS) >= len(words & _EN_MARKERS) else "en"

    def score(self, text: str, lang: str = "auto") -> tuple[float, list[str]]:
        """
        Retourne (score, hints).
        score = nombre de types couverts / 11, hints = types détectés.
        """
        if lang == "auto":
            lang = self.detect_lang(text)
        detected = []
        for rel, lang_patterns in _COMPILED.items():
            patterns = lang_patterns.get(lang, [])
            if not patterns:
                # fallback : essayer les deux langues
                patterns = lang_patterns.get("fr", []) + lang_patterns.get("en", [])
            for pat in patterns:
                if pat.search(text):
                    detected.append(rel)
                    break
        score = len(detected) / len(RELATION_KEYWORDS)
        return score, detected
