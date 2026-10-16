# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""NLU sans UD — B + actes + concepts (refacto étape 6, [T4]/[T6]).

Références : ``plan-remodelage-tout-appris.md`` (étape 6, [T4] NLU consomme
``cir.intent``), ``plan-remodelage-inference.md`` §[D] (P1 22 → P2 expects_*
→ P3 ``?``, concepts depuis chunks, lazy + no-cycle), ``spec-moteur-B.md``
(B sert le silver ET la NLU), [T6] (silver intent par ponctuation).

``infer_nlu(text, lang)`` : B segmente (clauses + source) → acte par clause
(``detect_act``, ponctuation [T6]) + langue du MESSAGE ([R-LANG]) + concepts
(mots porteurs sans UD) + attentes P2 (``expects_response``/``expects_action``
dérivées de l'acte — pas du ``SentenceProfile`` UD legacy).
``silver_annotate`` : même chose + traçabilité annotation (§6 : source,
phrase-hash, outil+version) = branchement silver-B dans le pipeline.

Chaîne P1/P2/P3 ([D]) : P1 = 21 intents legacy SOUS ``question`` (via
``legacy_router`` injecté dans ``route_intent``) ; P2 = attentes ci-dessous ;
P3 = ``?`` surface (déjà consommé par ``detect_act``). Lazy : aucun import
``pipeline/``/``training`` ici (no-cycle testé) ; spaCy reste lazy dans B.
"""

from __future__ import annotations

import hashlib
import unicodedata

from ..layer0.engine_b import SOURCE_FALLBACK, SOURCE_SILVER_B, analyze, describe
from ..layer2.act import detect_act, detect_lang

#: Actes qui attendent une réponse / une action (P2, dérivés sans UD).
EXPECTS = {
    "assert": {"response": False, "action": False},
    "question": {"response": True, "action": False},
    "request": {"response": False, "action": True},
}


def content_words(text: str) -> list[str]:
    """Mots porteurs sans UD : minuscules, > 2 lettres, ponctuation retirée."""
    mots = []
    for brut in text.split():
        mot = brut.strip("?.,!()«»:;\"'¿¡…؟？").lower()
        if len(mot) > 2 and any(
            unicodedata.category(c).startswith("L") for c in mot
        ):
            mots.append(mot)
    return mots


def infer_nlu(text: str, lang: str = "fr", *, force_fallback: bool = False) -> dict:
    """Infère par clause : intent + concepts + attentes (sans UD).

    Retourne ``{"clauses": [...], "source", "lang"}`` où chaque clause porte
    ``{texte, span, intent{act,saillant,lang}, concepts, expects{response,
    action}}``. ``lang`` = langue du MESSAGE ([R-LANG], une seule par appel).
    Ne lève que ``TypeError``/``ValueError`` (args invalides, budget B).
    """
    if not isinstance(text, str):
        raise TypeError(f"text doit être str, reçu {type(text).__name__}")
    langue = detect_lang(text)
    sortie = analyze(text, lang, force_fallback=force_fallback)
    clauses = []
    for clause in sortie.clauses:
        acte = detect_act(clause["texte"])
        acte["lang"] = langue
        clauses.append({
            "texte": clause["texte"],
            "span": list(clause["span"]),
            "intent": acte,
            "concepts": content_words(clause["texte"]),
            "expects": dict(EXPECTS[acte["act"]]),
        })
    source = sortie.source
    assert source in (SOURCE_SILVER_B, SOURCE_FALLBACK)
    return {"clauses": clauses, "source": source, "lang": langue}


def silver_annotate(text: str, lang: str = "fr", *, campagne: str = "socle") -> dict:
    """Annotation silver : ``infer_nlu`` + traçabilité (§6 : source,
    phrase-hash, campagne, outil+version). Loss masquée en aval si
    ``source == "fallback"`` ([T1] — le consommateur applique le masque)."""
    inference = infer_nlu(text, lang)
    return {
        "text": text,
        "lang": inference["lang"],
        "source": inference["source"],
        "phrase_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "campagne": campagne,
        "outil": {"moteur-b": describe(lang), "actes": "regles-v0"},
        "clauses": inference["clauses"],
    }


__all__ = ["EXPECTS", "content_words", "infer_nlu", "silver_annotate"]
