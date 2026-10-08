# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Fallback regex du moteur B (socle 0a, [T1]) — 100 % stdlib, jamais spaCy.

Références : ``docs/refactory/spec-moteur-B.md`` (§Fallback), ``docs/refactory/DECISIONS.md``
(D-SPACY), ``docs/refactory/plan-remodelage-tout-appris.md`` ([T1], [R4] format s0).

Rôle : chemin dégradé du moteur B (``engine_b.py``). Segmentation en phrases par
ponctuation forte + détection de connecteurs via les tables EXISTANTES — aucun
lexique nouveau (sans audit) :

- ``gcn-tools/gcn-scraper/.../filters/causal_scorer.py`` : ``RELATION_KEYWORDS`` (fr/en) ;
- ``gcn-tools/gcn-annotate/.../balanced_auto.py`` : ``CONN`` (fr/en/es/de).

Résolution des tables (``TABLES_ORIGIN``) : fichiers du dépôt d'abord
(sources citées par la spec), paquets installés en complément, sinon repli
sur ``_MINIMAL_SNAPSHOT`` (sous-ensemble VERBATIM des mêmes fichiers — pas
de nouveau lexique — rappelé comme dégradé).

Écart documenté à la spec : pt/it n'ont AUCUNE table auditée existante
(``causal_scorer`` = fr/en, ``balanced_auto`` = fr/en/es/de) → segmentation
seule, ``connecteurs == []``. Ajouter pt/it = audit lexique dédié (pas ici).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

FALLBACK_SOURCE = "fallback"

#: Langues budgétées (T1/[R-LANG]) : le moteur B accepte exactement celles-ci.
SUPPORTED_LANGS = ("fr", "en", "es", "de", "pt", "it")

#: Langues sans table auditée existante → segmentation seule (voir écart en docstring).
LANGS_WITHOUT_TABLE = ("pt", "it")

# ---------------------------------------------------------------------------
# Segmentation : ponctuation forte + familles ?/؟/？/¿…?/!
# ---------------------------------------------------------------------------

#: Points d'arrêt de phrase : `.` `!` `?` `…` + `؟` (arabe) + `？` (pleine chasse).
_ENDERS = ".!?…؟？"
_SENT_RE = re.compile(r"[^.!?…؟？]+(?:[.!?…؟؟]+|$)")
#: Marques ouvrantes (espagnol ¿…? / ¡…!) : jamais une frontière, retirées en tête.
_OPENERS = "¿¡«»\"'‘’“”([{\u00a0 \t"

_CLAUSE_ID = "c{:03d}"


def segment_sentences(text: str) -> list[dict]:
    """Découpe ``text`` en phrases (ponctuation forte + familles ?/؟/？/¿…?/!).

    Retourne une liste de ``{"texte", "span": [debut, fin]}`` (offsets absolus
    sur ``text``, texte débarrassé des blancs et des marques ouvrantes).
    """
    clauses: list[dict] = []
    for m in _SENT_RE.finditer(text):
        brut = m.group(0)
        debut = m.start() + (len(brut) - len(brut.lstrip(" \t\r\n" + _OPENERS)))
        fin = m.end() - (len(brut) - len(brut.rstrip(" \t\r\n")))
        morceau = text[debut:fin].strip(" \t\r\n" + _OPENERS)
        debut += (text[debut:fin].find(morceau) if morceau else 0)
        fin = debut + len(morceau)
        if morceau:
            clauses.append({"texte": morceau, "span": [debut, fin]})
    for i, clause in enumerate(clauses, 1):
        clause["id"] = _CLAUSE_ID.format(i)
    return clauses


# ---------------------------------------------------------------------------
# Tables de connecteurs EXISTANTES (aucun lexique nouveau)
# ---------------------------------------------------------------------------

#: Origine effective des tables (chargée à l'import : fichiers du dépôt →
#: paquets → snapshot ; voir ``_load_tables``).
TABLES_ORIGIN = "non-chargées"

_TABLES: dict[str, list[tuple[str, str | None, str | None]]] = {}
_COMPILED: dict[str, list[tuple[re.Pattern, str, str | None, str | None]]] = {}


def _merge(lang: str, connecteur: str, relation: str | None,
           direction: str | None, seen: set) -> None:
    conn = " ".join(connecteur.split())
    if not conn or (conn.lower(), relation) in seen:
        return
    seen.add((conn.lower(), relation))
    _TABLES.setdefault(lang, []).append((conn, relation, direction))


def _load_from_module(mod, attr: str, kind: str) -> bool:
    """Injecte une table depuis un module déjà importable. Retourne le succès."""
    table = getattr(mod, attr, None)
    if not table:
        return False
    seen: set = set()
    for lang in SUPPORTED_LANGS:
        if lang in LANGS_WITHOUT_TABLE:
            continue
        if kind == "keywords":  # RELATION_KEYWORDS : {relation: {lang: [mots]}}
            for relation, par_langue in table.items():
                for mot in par_langue.get(lang, []):
                    _merge(lang, mot, relation, None, seen)
        else:  # CONN : {lang: [(connecteur, relation, direction)]}
            for conn, relation, direction in table.get(lang, []):
                _merge(lang, conn.strip(), relation, direction, seen)
    return True


def _load_file(path: Path, module_name: str, attr: str, kind: str) -> bool:
    if not path.is_file():
        return False
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return False
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    # Fichier tiers arbitraire : tout échec = source ignorée (pas de lexique neuf).
    except Exception:  # noqa: BLE001 — périmètre borné au chargement de tables
        return False
    return _load_from_module(mod, attr, kind)


def _load_tables() -> str:
    """Charge les tables existantes (fichiers du dépôt → paquets → snapshot).

    Chaque source (``causal_scorer``, ``balanced_auto``) est résolue
    indépendamment : les fichiers du dépôt (cités par la spec) priment, les
    paquets installés ne servent qu'en complément. Retourne l'origine
    effective (ex. ``"fichiers-repo:causal_scorer+balanced_auto"``).
    """
    _TABLES.clear()
    sources: list[str] = []
    # 1) fichiers du même dépôt (aucune installation requise).
    racine = Path(__file__).resolve().parents[4]  # …/gcn-python/src/gcn_python/layer0 → dépôt
    if _load_file(
        racine / "gcn-tools/gcn-scraper/src/gcn_scraper/filters/causal_scorer.py",
        "_b_causal_scorer", "RELATION_KEYWORDS", "keywords",
    ):
        sources.append("causal_scorer")
    if _load_file(
        racine / "gcn-tools/gcn-annotate/src/gcn_annotate/balanced_auto.py",
        "_b_balanced_auto", "CONN", "conn",
    ):
        sources.append("balanced_auto")
    # 2) paquets installés / sur le chemin, en complément seulement.
    # Module absent = source ignorée (pas d'échec : le snapshot prend le relais).
    if "causal_scorer" not in sources:
        try:
            from gcn_scraper.filters import causal_scorer  # type: ignore
        except ImportError:
            causal_scorer = None
        if causal_scorer is not None and _load_from_module(
            causal_scorer, "RELATION_KEYWORDS", "keywords"
        ):
            sources.append("causal_scorer(paquet)")
    if "balanced_auto" not in sources:
        try:
            from gcn_annotate import balanced_auto  # type: ignore
        except ImportError:
            balanced_auto = None
        if balanced_auto is not None and _load_from_module(balanced_auto, "CONN", "conn"):
            sources.append("balanced_auto(paquet)")
    if sources and any(_TABLES.values()):
        return "fichiers-repo:" + "+".join(sources)
    # 3) dernier repli : sous-ensemble VERBATIM (dégradé, documenté).
    _TABLES.clear()
    seen: set = set()
    for lang, entries in _MINIMAL_SNAPSHOT.items():
        for conn, relation, direction in entries:
            _merge(lang, conn, relation, direction, seen)
    return "snapshot-minimal-integre"


#: Repli ultime : marqueurs forts VERBATIM issus des deux fichiers sources
#: (aucun mot inventé ; toute extension exige un audit — spec §Fallback).
_MINIMAL_SNAPSHOT: dict[str, list[tuple[str, str, str | None]]] = {
    "fr": [("parce que", "cause", "bwd"), ("car", "cause", "bwd"),
           ("donc", "cause", "fwd"), ("mais", "opposition", "bwd"),
           ("si", "condition", "bwd"), ("malgré", "concession", "bwd")],
    "en": [("because", "cause", "bwd"), ("therefore", "cause", "fwd"),
           ("but", "opposition", "bwd"), ("if", "condition", "bwd"),
           ("despite", "concession", "bwd")],
    "es": [("porque", "cause", "bwd"), ("pero", "opposition", "bwd"),
           ("si", "condition", "bwd"), ("aunque", "concession", "bwd")],
    "de": [("weil", "cause", "bwd"), ("aber", "opposition", "bwd"),
           ("wenn", "condition", "bwd"), ("obwohl", "concession", "bwd")],
}


def _compiled(lang: str) -> list[tuple[re.Pattern, str, str | None, str | None]]:
    if lang in _COMPILED:
        return _COMPILED[lang]
    if not _TABLES:
        _load_tables()
    motifs = []
    for conn, relation, direction in sorted(_TABLES.get(lang, []), key=lambda e: -len(e[0])):
        motifs.append((
            re.compile(r"(?<!\w)" + re.escape(conn) + r"(?!\w)", re.IGNORECASE),
            conn, relation, direction,
        ))
    _COMPILED[lang] = motifs
    return motifs


def find_connectors(text: str, lang: str) -> list[dict]:
    """Détecte les connecteurs de ``text`` via les tables existantes (``lang``).

    Retourne ``[{"texte", "relation", "direction", "span": [debut, fin]}]``
    triés par position, sans recouvrement (le plus long gagne). ``[]`` pour
    pt/it (aucune table auditée — écart documenté en module).
    """
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"langue hors budget fr/en/es/de/pt/it : {lang!r}")
    trouves: list[dict] = []
    for motif, _conn, relation, direction in _compiled(lang):
        m = motif.search(text)
        if m:
            trouves.append({"texte": m.group(0), "relation": relation,
                            "direction": direction, "span": [m.start(), m.end()]})
    trouves.sort(key=lambda c: (c["span"][0], -(c["span"][1] - c["span"][0])))
    filtres: list[dict] = []
    occupe_jusqua = -1
    for c in trouves:
        if c["span"][0] >= occupe_jusqua:
            filtres.append(c)
            occupe_jusqua = c["span"][1]
    return filtres


def analyze_fallback(text: str, lang: str = "fr") -> tuple[list[dict], list[dict], str]:
    """Chemin dégradé : ``(clauses, connecteurs, "fallback")`` (``source`` systématique)."""
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"langue hors budget fr/en/es/de/pt/it : {lang!r}")
    clauses = segment_sentences(text)
    for clause in clauses:
        clause["lang"] = lang
    connecteurs = []
    for conn in find_connectors(text, lang):
        rattache = next((c["id"] for c in clauses
                         if c["span"][0] <= conn["span"][0] < c["span"][1]), None)
        connecteurs.append({**conn, "clause_id": rattache})
    return clauses, connecteurs, FALLBACK_SOURCE


__all__ = [
    "FALLBACK_SOURCE",
    "LANGS_WITHOUT_TABLE",
    "SUPPORTED_LANGS",
    "TABLES_ORIGIN",
    "analyze_fallback",
    "find_connectors",
    "segment_sentences",
]

#: Chargement eager à l'import (stdlib, fichiers locaux, sans réseau) :
#: aucune réassignation ensuite (pas de ``global``).
TABLES_ORIGIN = _load_tables()
