# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Moteur B — socle spaCy + fallback regex (socle 0a, [T1], format s0).

Références : ``docs/refactory/spec-moteur-B.md``, ``docs/refactory/DECISIONS.md``
(D-SPACY : import spaCy UNIQUEMENT lazy et optionnel — déjà amendé dans
``layer0/interface.py``), ``docs/refactory/plan-remodelage-tout-appris.md``
([T1] pin + lazy + timeout + fallback + test sans-spacy, [R4] format s0).

Cadre STRICT : spaCy n'entre JAMAIS dans le moteur appris — ce module n'est
importé par aucun code de ``pipeline/``, ``training/`` ni ``engine.py``
(garde testée dans ``gcn-python/tests/test_engine_b.py``). Import spaCy
LAZY (dans la fonction, jamais au top-level), timeout 30 s, repli regex
si absent/lent/HS (``source="fallback"`` systématique + compteur exposé).

Interface : entrée ``text: str`` (+ ``lang`` budgétée fr/en/es/de/pt/it),
sortie ``(clauses, connecteurs, source)`` avec
``source ∈ {"silver-B", "fallback"}`` obligatoire.

Traçabilité : ``provenance()`` / ``to_record()`` portent
``{outil, version, source}`` (version = spacy + modèle) ; tout enregistrement
B porte ``format_refacto=20261007`` (socle s0 — [R4]). s1/s2/s3
(``tok_vocab``, ``d_model``, ``act_types``) viendront après et ne sont PAS
anticipés ici (garde testée : ces clés sont absentes du socle s0).

Seuil d'alerte fallback : ``FALLBACK_ALERT_RATE = 0.05`` (5 %, PROPOSÉ —
à valider en phase 1 ; voir ``fallback_stats()`` / ``fallback_rate()``).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, NamedTuple

from .fallback_regex import SUPPORTED_LANGS, analyze_fallback

#: Étiquette portée jusqu'au CIR quand spaCy a produit la segmentation.
SOURCE_SILVER_B = "silver-B"
#: Étiquette portée jusqu'au CIR sur le chemin dégradé (loss masquée en aval).
SOURCE_FALLBACK = "fallback"

#: Nom d'outil tracé dans chaque sortie B.
OUTIL_B = "moteur-B"
#: Pin reproductible exigé par [T1] (``spacy==3.8`` ; env constaté : 3.8.16).
SPACY_PINNED = "3.8"
#: Modèle spaCy candidat par langue budgétée (seul fr garanti installé).
MODEL_FOR_LANG = {
    "fr": "fr_core_news_sm",
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "de": "de_core_news_sm",
    "pt": "pt_core_news_sm",
    "it": "it_core_news_sm",
}
#: Timeout FIGÉ 2026-10-08 (spec §Interface, même ordre que l'ancien pont).
ENGINE_TIMEOUT_S = 30.0
#: Socle s0 ([R4]) : seule clé de format écrite par B.
FORMAT_REFACTO_S0 = 20261007
#: Seuil d'alerte sur le taux de fallback : 5 % PROPOSÉ (à valider phase 1).
FALLBACK_ALERT_RATE = 0.05


class EngineBOutput(NamedTuple):
    """Sortie du moteur B : ``(clauses, connecteurs, source)``."""

    clauses: list[dict]
    connecteurs: list[dict]
    source: str


_VERROU = threading.Lock()
_NLP_CACHE: dict[str, Any] = {}
#: Executor unique (un par appel = churn de threads + workers orphelins).
_EXECUTEUR = ThreadPoolExecutor(max_workers=1)
#: Compteur d'appels (dict mutable : pas de ``global``).
_ETAT = {"total": 0, "fallbacks": 0}


def reset_stats() -> None:
    """Remet à zéro le compteur d'appels (tests / supervision)."""
    with _VERROU:
        _ETAT["total"] = 0
        _ETAT["fallbacks"] = 0


def _reset_cache() -> None:
    """Vide le cache de modèles spaCy (tests simulant l'absence)."""
    with _VERROU:
        _NLP_CACHE.clear()


def fallback_stats() -> dict:
    """Compteur fallback exposé : ``{"total", "fallbacks", "taux", "seuil_alerte"}``."""
    with _VERROU:
        total, fallbacks = _ETAT["total"], _ETAT["fallbacks"]
    taux = (fallbacks / total) if total else 0.0
    return {"total": total, "fallbacks": fallbacks, "taux": taux,
            "seuil_alerte": FALLBACK_ALERT_RATE, "alerte": taux > FALLBACK_ALERT_RATE}


def fallback_rate() -> float:
    """Taux de fallback courant (0.0 si aucun appel)."""
    return fallback_stats()["taux"]


def _compte(total_1: bool, fallback_1: bool) -> None:
    with _VERROU:
        _ETAT["total"] += total_1
        _ETAT["fallbacks"] += fallback_1


def _with_timeout(fn, timeout_s: float):
    """Exécute ``fn()`` avec ``timeout_s`` (le thread orphelin est abandonné)."""
    if timeout_s <= 0:
        raise TimeoutError(f"moteur B : délai dépassé ({timeout_s}s)")
    futur = _EXECUTEUR.submit(fn)
    try:
        return futur.result(timeout=timeout_s)
    except FuturesTimeoutError:
        futur.cancel()
        raise TimeoutError(f"moteur B : délai dépassé ({timeout_s}s)") from None


def _import_spacy():
    """Import LAZY de spaCy (jamais au top-level — D-SPACY)."""
    import spacy

    return spacy


def _exige_pin(spacy) -> None:
    """Lève si spaCy ne respecte pas le pin 3.8 (extraite de analyze)."""
    if _version_spacy(spacy) is None:
        raise RuntimeError(f"spaCy hors pin {SPACY_PINNED}.x")


def _version_spacy(spacy) -> str | None:
    version = getattr(spacy, "__version__", None)
    if not isinstance(version, str):
        return None
    noyau = ".".join(version.split(".")[:2])
    return version if noyau == SPACY_PINNED else None


def _version_modele(nom_modele: str) -> str | None:
    try:
        module = __import__(nom_modele)
    except ImportError:
        return None
    version = getattr(module, "__version__", None)
    return version if isinstance(version, str) else "inconnue"


def _charge_modele(spacy, lang: str, timeout_s: float):
    with _VERROU:
        if lang in _NLP_CACHE:
            return _NLP_CACHE[lang]
    nlp = _with_timeout(lambda: spacy.load(MODEL_FOR_LANG[lang]), timeout_s)
    with _VERROU:
        _NLP_CACHE.setdefault(lang, nlp)
        return _NLP_CACHE[lang]


def _parse_spacy(nlp, text: str):
    """Unique point d'appel spaCy sur texte (patchable pour le test timeout)."""
    return nlp(text)


def _clauses_spacy(doc, lang: str) -> list[dict]:
    phrases = [p for p in doc.sents if p.text.strip()]
    if not phrases:
        raise ValueError("spaCy : aucune phrase segmentée")
    clauses = []
    for i, phrase in enumerate(phrases, 1):
        brut, texte = phrase.text, phrase.text.strip()
        debut = phrase.start_char + (len(brut) - len(brut.lstrip()))
        clauses.append({"id": f"c{i:03d}", "texte": texte, "lang": lang,
                        "span": [debut, debut + len(texte)]})
    return clauses


def _connecteurs_rattaches(text: str, lang: str, clauses: list[dict]) -> list[dict]:
    from .fallback_regex import find_connectors  # stdlib, sans spaCy

    connecteurs = []
    for conn in find_connectors(text, lang):
        rattache = next((c["id"] for c in clauses
                         if c["span"][0] <= conn["span"][0] < c["span"][1]), None)
        connecteurs.append({**conn, "clause_id": rattache})
    return connecteurs


def _valide_args(text: str, lang: str) -> None:
    """Lève TypeError/ValueError sur args invalides (extraite de analyze)."""
    if not isinstance(text, str):
        raise TypeError(f"text doit être str, reçu {type(text).__name__}")
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"langue hors budget fr/en/es/de/pt/it : {lang!r}")


def analyze(text: str, lang: str = "fr", *, timeout: float = ENGINE_TIMEOUT_S,
            force_fallback: bool = False) -> EngineBOutput:
    """Analyse ``text`` → ``(clauses, connecteurs, source)`` (``lang`` budgétée).

    Chemin silver-B (segmentation spaCy + connecteurs des tables existantes)
    si spaCy 3.8 + modèle de ``lang`` disponibles dans ``timeout`` secondes,
    sinon chemin ``fallback`` (regex, ``source="fallback"`` systématique,
    compteur incrémenté). Ne lève que ``ValueError`` (``lang`` hors budget)
    et ``TypeError`` (``text`` non-``str``) — tout incident spaCy dégrade.
    """
    _valide_args(text, lang)
    if not text.strip() or force_fallback:
        _compte(True, True)
        clauses, connecteurs, source = analyze_fallback(text, lang)
        return EngineBOutput(clauses, connecteurs, source)
    try:
        spacy = _import_spacy()  # absent → ImportError → fallback
        _exige_pin(spacy)  # pin 3.8 non respecté → RuntimeError → fallback
        nlp = _charge_modele(spacy, lang, timeout)  # modèle absent → OSError → fallback
        doc = _with_timeout(lambda: _parse_spacy(nlp, text), timeout)  # lent → fallback
        clauses = _clauses_spacy(doc, lang)
        connecteurs = _connecteurs_rattaches(text, lang, clauses)
        _compte(True, False)
        return EngineBOutput(clauses, connecteurs, SOURCE_SILVER_B)
    # Tout incident spaCy (absent, lent, HS, pin non respecté) → fallback.
    except Exception:  # noqa: BLE001 — attrape-tout volontaire ([T1] : dégradation)
        _compte(True, True)
        clauses, connecteurs, source = analyze_fallback(text, lang)
        return EngineBOutput(clauses, connecteurs, source)


def describe(lang: str = "fr") -> dict:
    """Versions live ``{"spacy", "modele"}`` (``None`` si indisponible)."""
    try:
        spacy = _import_spacy()
        version_spacy: str | None = getattr(spacy, "__version__", None)
    except ImportError:
        version_spacy = None
    modele = MODEL_FOR_LANG.get(lang, MODEL_FOR_LANG["fr"])
    return {"spacy": version_spacy,
            "modele": f"{modele}=={_version_modele(modele)}"
                      if _version_modele(modele) else None}


def provenance(source: str, lang: str = "fr") -> dict:
    """``{outil, version, source}`` porté par chaque sortie B (spec §Traçabilité)."""
    if source not in (SOURCE_SILVER_B, SOURCE_FALLBACK):
        raise ValueError(f"source B invalide : {source!r}")
    return {"outil": OUTIL_B, "version": describe(lang), "source": source}


def to_record(output: EngineBOutput, lang: str = "fr") -> dict:
    """Enregistrement B : sortie + traçabilité + ``format_refacto`` (socle s0).

    Clé de format EXPLICITE ``format_refacto=20261007`` ([R4]) ; aucune clé
    s1/s2/s3 (``tok_vocab``, ``d_model``, ``act_types``) n'est anticipée.
    """
    clauses, connecteurs, source = output
    return {"clauses": clauses, "connecteurs": connecteurs, "source": source,
            "meta": provenance(source, lang), "format_refacto": FORMAT_REFACTO_S0}


def save_record(path: str, output: EngineBOutput, lang: str = "fr") -> str:
    """Sauvegarde un enregistrement B (JSON, socle s0). Retourne ``path``."""
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_record(output, lang), f, ensure_ascii=False)
    return path


__all__ = [
    "ENGINE_TIMEOUT_S",
    "FALLBACK_ALERT_RATE",
    "FORMAT_REFACTO_S0",
    "MODEL_FOR_LANG",
    "OUTIL_B",
    "SOURCE_FALLBACK",
    "SOURCE_SILVER_B",
    "SPACY_PINNED",
    "EngineBOutput",
    "analyze",
    "describe",
    "fallback_rate",
    "fallback_stats",
    "provenance",
    "reset_stats",
    "save_record",
    "to_record",
]
