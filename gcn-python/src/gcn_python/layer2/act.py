# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Actes de langage — 3 actes + détection règles (refacto tête act, T4).

Références : ``plan-remodelage-tout-appris.md`` ([T4] 3 actes, [R-LANG]),
``plan-remodelage-inference.md`` §[D] (``route_intent``), ``bascule-intents.md``
(option (b) couches FIGÉE, exclamative → assert saillant FIGÉ),
[R9] (règles gold : ``!`` ≠ request, Si+Pres → assert, question sans ``?`` → assert).

3 actes : ``assert`` → stocker, ``question`` → répondre, ``request`` → ack +
file-action. ``cir.intent`` porte l'acte + ``saillant`` (exclamatif) + ``lang``
COMPRISE ([R-LANG] : défaut ``fr`` ; consigne explicite → bascule ; arabe →
``ar`` ; CJK indéterminé → ``und`` — pas de détecteur statistique ici, les
différences fines s'apprennent sur DONNÉES, D2).

Règles v0 (revues, apprises ensuite sur gold-3000) :
- ``?/؟/؟`` final (ou ``¿…?``) → ``question`` ([T6] familles de ponctuation) ;
- ``!`` (ou ``¡…!``) → ``assert`` + ``saillant=True`` ([R9] ``!`` ≠ request) ;
- consigne explicite (« réponds en X » fr/en/es/de) → ``request`` + bascule ;
- marqueurs de politesse (listes closes documentées) → ``request`` ;
- impératif nu sans marqueur → ``assert`` par défaut (écart assumé, corrigé
  par l'apprentissage — aucun lexique verbal inventé) ;
- question sans ``?`` → ``assert`` assumé ([R9]) ; Si+conditionnel → ``assert``.
"""

from __future__ import annotations

import re
import unicodedata

#: Les 3 actes (T4) — ordre = indices de la tête act (s3 ``act_types``).
ACT_ASSERT = "assert"
ACT_QUESTION = "question"
ACT_REQUEST = "request"
ACT_TYPES: tuple[str, str, str] = (ACT_ASSERT, ACT_QUESTION, ACT_REQUEST)

#: Familles de ponctuation interrogative ([T6]).
_QUESTION_MARKS = ("?", "؟", "？")
#: Familles de ponctuation exclamative.
_EXCLAMATION_MARKS = ("!", "‼", "❗")

#: Marqueurs de requête polie v0 (listes closes, revues — apprises ensuite).
_REQUEST_MARKERS = (
    # fr
    "s'il vous plaît", "s'il te plaît", "s il vous plait", "s il te plait",
    "merci de", "prière de", "veuillez", "auriez-vous l'amabilité",
    # en
    "please", "could you", "would you", "kindly",
    # es
    "por favor",
    # de
    "bitte",
)

#: Consignes de langue explicites → (motif, code) ([R-LANG] bascule).
_CONSIGNE_PATTERNS = (
    (re.compile(r"réponds?\s+(?:moi\s+)?en\s+(\w+)", re.IGNORECASE), "fr"),
    (re.compile(r"answer\s+in\s+(\w+)", re.IGNORECASE), "en"),
    (re.compile(r"responde\s+en\s+(\w+)", re.IGNORECASE), "es"),
    (re.compile(r"antworte\s+(?:mir\s+)?auf\s+(\w+)", re.IGNORECASE), "de"),
)

#: Noms de langues → codes (consigne explicite uniquement).
_LANG_NAMES = {
    "wolof": "wo", "français": "fr", "francais": "fr", "french": "fr",
    "frances": "fr", "französisch": "fr", "franzosisch": "fr",
    "anglais": "en", "english": "en", "ingles": "en", "englisch": "en",
    "espagnol": "es", "spanish": "es", "español": "es", "espanol": "es",
    "spanisch": "es", "allemand": "de", "german": "de", "alemán": "de",
    "aleman": "de", "deutsch": "de", "portugais": "pt", "portuguese": "pt",
    "portugués": "pt", "portugues": "pt", "portugiesisch": "pt",
    "italien": "it", "italian": "it", "italiano": "it", "italienisch": "it",
    "arabe": "ar", "arabic": "ar", "árabe": "ar",
    "arabisch": "ar", "chinois": "zh", "chinese": "zh", "chino": "zh",
    "chinesisch": "zh",
}

_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_CJK_RE = re.compile(
    r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\u0E00-\u0E7F\u1000-\u109F\u1780-\u17FF]"
)


def _letters(text: str) -> str:
    return "".join(c for c in text if unicodedata.category(c).startswith("L"))


def detect_lang(text: str) -> str:
    """Langue COMPRISE ([R-LANG]) : consigne explicite > script > défaut ``fr``.

    Arabe dominant → ``ar`` ; CJK dominant sans consigne → ``und``
    (indéterminé honnête : chinois/japonais indiscernables sans données) ;
    latin → ``fr`` par défaut ([R-LANG]).
    """
    if not isinstance(text, str):
        raise TypeError(f"text doit être str, reçu {type(text).__name__}")
    for motif, _ in _CONSIGNE_PATTERNS:
        m = motif.search(text)
        if m:
            code = _LANG_NAMES.get(m.group(1).lower(), "")
            if code:
                return code
    lettres = _letters(text)
    if not lettres:
        return "fr"
    n_ar = len(_ARABIC_RE.findall(lettres))
    n_cjk = len(_CJK_RE.findall(lettres))
    if n_ar * 2 >= len(lettres):
        return "ar"
    if n_cjk * 2 >= len(lettres):
        return "und"
    return "fr"


def detect_act(text: str) -> dict:
    """Acte + saillance + langue → dict ``cir.intent`` v0 (règles [R9]/[T6]).

    Retourne ``{"act", "saillant", "lang"}``. Ne lève que ``TypeError``
    (non-``str``) ; texte vide/blanc → ``assert`` non saillant, ``fr``.
    """
    if not isinstance(text, str):
        raise TypeError(f"text doit être str, reçu {type(text).__name__}")
    nettoye = text.strip()
    lang = detect_lang(text)
    if not nettoye:
        return {"act": ACT_ASSERT, "saillant": False, "lang": lang}
    # Consigne explicite = requête (demande), avec bascule éventuelle.
    for motif, _ in _CONSIGNE_PATTERNS:
        if motif.search(text):
            return {"act": ACT_REQUEST, "saillant": False, "lang": lang}
    dernier = nettoye[-1]
    premier = nettoye[0]
    if dernier in _QUESTION_MARKS or (premier == "¿" and "?" in nettoye):
        return {"act": ACT_QUESTION, "saillant": False, "lang": lang}
    saillant = dernier in _EXCLAMATION_MARKS or (premier == "¡" and "!" in nettoye)
    if saillant:
        # [R9] : ``!`` exclamatif ≠ request — assert saillant.
        return {"act": ACT_ASSERT, "saillant": True, "lang": lang}
    bas = nettoye.lower()
    if any(m in bas for m in _REQUEST_MARKERS):
        return {"act": ACT_REQUEST, "saillant": False, "lang": lang}
    # Défaut : assert assumé ([R9] question sans ``?``, Si+Pres, impératif nu).
    return {"act": ACT_ASSERT, "saillant": False, "lang": lang}


__all__ = [
    "ACT_ASSERT",
    "ACT_QUESTION",
    "ACT_REQUEST",
    "ACT_TYPES",
    "detect_act",
    "detect_lang",
]
