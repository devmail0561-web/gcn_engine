from __future__ import annotations
from pathlib import Path
import yaml

from ..layer1.representation import UDRepresentation

_nom_cache: dict[tuple[str, str], dict[str, str]] = {}


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
    nom = _nominalize(rep.root_lemma, rep.lang, taxonomies_dir)

    if node_type == "condition":
        label = "hidden_cause(?)" if rep.lang == "en" else "cause_cachée(?)"
    elif node_type in ("entite", "etat_systemique"):
        label = entity or rep.root_lemma
    elif node_type == "action":
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
        "agent": subject if node_type in ("action", "transition") else None,
        "patient": _find_patient_lemma(rep),
        "quality": None,
        "agent_type": None,
        "reversible": None,
    }
    return label, attributes


def _find_patient_lemma(rep: UDRepresentation) -> str | None:
    """Premier token avec dep_rel obj/iobj/nobj — patient syntaxique de la clause."""
    for t in rep.tokens:
        if t.get("dep_rel") in {"obj", "iobj", "nobj"}:
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


def _nominalize(lemma: str, lang: str, taxonomies_dir: Path | None) -> str:
    if taxonomies_dir is None:
        return lemma
    cache_key = (str(taxonomies_dir), lang)
    if cache_key not in _nom_cache:
        _nom_cache[cache_key] = _load_nominalizations(taxonomies_dir, lang)
    return _nom_cache[cache_key].get(lemma, lemma)


def _load_nominalizations(taxonomies_dir: Path, lang: str) -> dict[str, str]:
    table: dict[str, str] = {}
    for search_dir in [taxonomies_dir / lang, taxonomies_dir]:
        nom_path = search_dir / "nominalizations.yaml"
        if nom_path.exists():
            doc = yaml.safe_load(nom_path.read_text(encoding="utf-8"))
            if isinstance(doc, dict):
                examples_key = "examples_fr" if lang == "fr" else "examples"
                for _cls, cls_data in (doc.get("classes") or {}).items():
                    for entry in (cls_data or {}).get(examples_key) or []:
                        if isinstance(entry, dict) and "lemma" in entry and "note" in entry:
                            table[entry["lemma"].lower()] = entry["note"]
            break
    return table
