# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Spans gold → spans caractères pour le chemin tok (supervision pure).

Le gold annote en ids de tokens UD (``token_span: [8, 11]``) ; le chemin
tok aligne en caractères (``align_spans``). Ce module convertit l'un en
l'autre par matching ordonné des formes sur le texte — DÉTERMINISTE,
validé, rejet explicite. Même méthode que ``scripts/audit_spans_tok.py``
(l'oracle rejouable) ; dupliquée ici car ``src/`` n'importe jamais
``scripts/`` (tout écart entre les deux = bug — test de parité exigé
dans ``tests/test_gold_spans.py``).

Règle : pliage apostrophes 1:1 (offsets conservés, longueur vérifiée) ;
échec = ``SpanError`` avec phrase + motif, jamais de deviné.
"""

from __future__ import annotations

APOS = {
    "\u2019": "'",
    "\u2018": "'",
    "\u201b": "'",
    "\u02bc": "'",
    "\u0060": "'",
    "\u00b4": "'",
}


class SpanError(ValueError):
    """Reconstruction impossible (phrase + motif — rejet explicite)."""


def plier(s: str) -> str:
    """Pliage apostrophes typographiques → droite (1:1, offsets conservés)."""
    for a, b in APOS.items():
        s = s.replace(a, b)
    return s


def reconstruire_offsets(texte: str, tokens: list) -> dict:
    """Formes UD ordonnées → ``{token_id: (début, fin)}`` caractères.

    Lève ``SpanError`` (forme vide / introuvable / pliage non 1:1).
    """
    if not isinstance(texte, str):
        raise TypeError(f"texte doit être str, reçu {type(texte).__name__}")
    fond = plier(texte)
    if len(fond) != len(texte):
        raise SpanError("pliage non 1:1 — offsets non transférables.")
    offs: dict = {}
    pos = 0
    for t in tokens:
        forme = plier(t.get("form", ""))
        if not forme:
            raise SpanError(f"forme vide (token id={t.get('id')!r}).")
        i = fond.find(forme, pos)
        if i < 0:
            raise SpanError(f"forme introuvable : {forme[:24]!r} (token id={t.get('id')!r}).")
        offs[t["id"]] = (i, i + len(forme))
        pos = i + len(forme)
    return offs


def span_clause(offs: dict, debut_id: int, fin_id: int, ntoks: int) -> tuple[int, int]:
    """Span token UD [début, fin] (inclusif, 1-based) → span caractères.

    Lève ``SpanError`` sur bornes invalides (jamais de clamp silencieux —
    le chargeur legacy clampe (v5.7) ; ici on refuse : la supervision
    tok ne devine pas).
    """
    if not (1 <= debut_id <= fin_id <= ntoks):
        raise SpanError(f"bornes invalides : [{debut_id}, {fin_id}] pour {ntoks} tokens.")
    try:
        debuts = [offs[k][0] for k in range(debut_id, fin_id + 1)]
        fins = [offs[k][1] for k in range(debut_id, fin_id + 1)]
    except KeyError as exc:
        raise SpanError(f"token id manquant dans les offsets : {exc}.") from exc
    return (min(debuts), max(fins))
