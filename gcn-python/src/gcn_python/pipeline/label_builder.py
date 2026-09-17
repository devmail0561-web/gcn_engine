from __future__ import annotations
import json
from pathlib import Path

from ..layer1.representation import UDRepresentation
from ..constants import NODE_TYPES

_NT_CONDITION  = NODE_TYPES[4]   # "condition"
_NT_ENTITE     = NODE_TYPES[5]   # "entite"
_NT_ETAT_SYS   = NODE_TYPES[6]   # "etat_systemique"
_NT_ACTION     = NODE_TYPES[1]   # "action"
_NT_TRANSITION = NODE_TYPES[2]   # "transition"

_nom_cache: dict[str, dict[str, str]] = {}   # max ~100 répertoires en pratique
_NOM_CACHE_MAXSIZE = 128


def build_label(
    rep: UDRepresentation,
    node_type: str,
    taxonomies_dir: Path | None = None,
) -> tuple[str, dict]:
    """
    (UDRepresentation, node_type, TaxonomyIndex) → (str label CIR, dict attributes).

    Format selon node_type :
      action               → "{verb_lemma}({subject_lemma})"
      etat/transition/     → "{nominalization}({entity})" ou "{verb}({subject})"
      processus
      entite/etat_system.  → "{entity_lemma}"
      condition            → "cause_cachée(?)"
    """
    subject = _find_subject_lemma(rep)
    entity = _find_entity_lemma(rep)
    nom = _nominalize(rep.root_lemma, taxonomies_dir)

    if node_type == _NT_CONDITION:
        label = "hidden_cause(?)"
    elif node_type in (_NT_ENTITE, _NT_ETAT_SYS):
        label = entity or rep.root_lemma
    elif node_type == _NT_ACTION:
        label = f"{rep.root_lemma}({subject})" if subject else rep.root_lemma
    else:
        # etat, transition, processus
        if entity:
            label = f"{nom}({entity})"
        elif subject:
            label = f"{rep.root_lemma}({subject})"
        else:
            label = nom

    attributes = {
        "entity": entity,
        "agent": subject if node_type in (_NT_ACTION, _NT_TRANSITION) else None,
        "patient": _find_patient_lemma(rep),
        "quality": None,
        "agent_type": None,
        "reversible": None,
    }
    return label, attributes


def _find_patient_lemma(rep: UDRepresentation) -> str | None:
    """Premier token avec dep_rel obj/iobj — patient syntaxique de la clause."""
    for t in rep.tokens:
        # L4 : retirer "nobj" (relation UD invalide)
        if t.get("dep_rel") in {"obj", "iobj"}:
            return t["lemma"]
    return None


def _find_subject_lemma(rep: UDRepresentation) -> str | None:
    for t in rep.tokens:
        if t.get("dep_rel") in {"nsubj", "nsubj:pass"}:
            return t["lemma"]
    return None


def _find_entity_lemma(rep: UDRepresentation) -> str | None:
    for t in rep.tokens:
        if t.get("dep_rel") in {"nsubj", "nsubj:pass"} and t.get("pos") in {"NOUN", "PROPN"}:
            return t["lemma"]
    for t in rep.tokens:
        if t.get("pos") in {"NOUN", "PROPN"} and t.get("dep_rel") != "punct":
            return t["lemma"]
    return None


def _nominalize(lemma: str, taxonomies_dir: Path | None) -> str:
    if taxonomies_dir is None:
        return lemma
    cache_key = str(taxonomies_dir)
    if cache_key not in _nom_cache:
        if len(_nom_cache) >= _NOM_CACHE_MAXSIZE:
            _nom_cache.pop(next(iter(_nom_cache)))  # FIFO eviction
        _nom_cache[cache_key] = _load_nominalizations(taxonomies_dir)
    return _nom_cache[cache_key].get(lemma, lemma)


def _load_nominalizations(taxonomies_dir: Path) -> dict[str, str]:
    """Charge et fusionne les nominalizations de TOUS les sous-répertoires disponibles."""
    table: dict[str, str] = {}
    dirs_to_scan = [taxonomies_dir]
    if taxonomies_dir.is_dir():
        dirs_to_scan += sorted(d for d in taxonomies_dir.iterdir() if d.is_dir())
    for search_dir in dirs_to_scan:
        nom_path = search_dir / "nominalizations.json"
        if not nom_path.exists():
            continue
        with open(nom_path, encoding="utf-8") as f:
            doc = json.load(f)
        if not isinstance(doc, dict):
            continue
        for _cls, cls_data in (doc.get("classes") or {}).items():
            for key in ("examples_fr", "examples"):  # essayer les deux clés
                for entry in (cls_data or {}).get(key) or []:
                    if isinstance(entry, dict) and "lemma" in entry and "note" in entry:
                        table.setdefault(entry["lemma"].lower(), entry["note"])
    return table
