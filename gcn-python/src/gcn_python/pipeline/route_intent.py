# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Routage des 3 actes — session (refacto tête act, [D]/[E]).

Références : ``plan-remodelage-inference.md`` §[D] (``route_intent`` consomme
``cir.intent``) et §[E] (``request→ack+file-action``, ``question→execute``,
``assert→add_cir``), ``bascule-intents.md`` (option (b) couches FIGÉE :
les 21 intents survivent SOUS ``question`` via ``nlu_route`` legacy).

``route_intent(intent, text, cir=None, legacy_router=None)`` :
- ``assert`` → ``{"act", "store": cir}`` (stockage = ``add_cir`` côté session) ;
- ``question`` → ``{"act", "query", "dsl"}`` (``dsl`` via ``legacy_router`` si
  fourni — typiquement ``nlu_route`` — sinon ``None``, rempli à la bascule) ;
- ``request`` → ``{"act", "ack", "filed"}`` (ack factuel + requête filée pour
  la session/clarification 9c — 1 en attente max, gérée côté session).
``cir.intent`` est prioritaire (prio 0, [T4]) ; ce module ne devine jamais
l'acte (le produire = ``act.py`` règles ou ``act_head.py`` apprise).
"""

from __future__ import annotations

from ..layer2.act import ACT_ASSERT, ACT_QUESTION, ACT_REQUEST, ACT_TYPES


def _ack_request(text: str, lang: str) -> str:
    """Ack factuel : reprise + langue éventuelle (jamais de promesse d'action)."""
    court = text.strip()
    if len(court) > 120:
        court = court[:117] + "…"
    phrase = f"Requête enregistrée : « {court} »."
    if lang != "fr":
        phrase += f" (langue : {lang})."
    return phrase


def route_intent(intent: dict, text: str, cir=None, legacy_router=None) -> dict:
    """Route un ``cir.intent`` vers l'action session (3 actes, couches (b)).

    ``intent`` : ``{"act", "saillant", "lang"}`` (``detect_act`` ou tête apprise).
    ``legacy_router`` : appelable ``(text) -> dsl|None`` (ex. ``nlu_route``),
    injecté (pas d'import, pas de cycle) — utilisé SOUS ``question`` seulement.
    Ne lève que ``TypeError``/``ValueError`` (intent/text invalides).
    """
    if not isinstance(text, str):
        raise TypeError(f"text doit être str, reçu {type(text).__name__}")
    if not isinstance(intent, dict):
        raise TypeError(f"intent doit être dict, reçu {type(intent).__name__}")
    act = intent.get("act")
    if act not in ACT_TYPES:
        raise ValueError(f"act ∈ {list(ACT_TYPES)} (reçu {act!r}).")
    lang = intent.get("lang", "fr")
    saillant = bool(intent.get("saillant", False))
    if act == ACT_ASSERT:
        return {"act": act, "saillant": saillant, "lang": lang, "store": cir}
    if act == ACT_QUESTION:
        dsl = legacy_router(text) if legacy_router is not None else None
        return {"act": act, "saillant": saillant, "lang": lang, "query": text, "dsl": dsl}
    assert act == ACT_REQUEST  # validé plus haut (3 actes clos)
    filed = {"text": text, "intent": {"act": act, "saillant": saillant, "lang": lang}}
    return {
        "act": act,
        "saillant": saillant,
        "lang": lang,
        "ack": _ack_request(text, lang),
        "filed": filed,
    }


__all__ = ["route_intent"]
