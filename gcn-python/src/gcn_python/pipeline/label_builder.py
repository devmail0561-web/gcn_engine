# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from ..layer1.representation import UDRepresentation

# Accès par nom D5 — robuste au réordre de NODE_TYPES
_NT_CONDITION   = "condition"
_NT_ENTITE      = "entite"
_NT_CONCEPT     = "concept"
_NT_ETAT_LOCAL  = "etat_local"
_NT_ETAT_GLOBAL = "etat_global"
_NT_EVENEMENT   = "evenement"
_NT_PROCESSUS   = "processus"
_NT_CONTRAINTE  = "contrainte"
# Aliases v2 pour lecture de CIR legacy
_NT_ETAT_SYS   = "etat_global"   # était etat_systemique
_NT_ACTION     = "processus"     # était action (fusionné D5)
_NT_TRANSITION = "processus"     # était transition (fusionné D5)


def build_label(
    rep: UDRepresentation,
    node_type: str,
) -> tuple[str, dict]:
    """
    (UDRepresentation, node_type) → (str label CIR, dict attributes).

    Zéro chargement externe : le label dérive des lemmes observés
    (sujet/entité/patient syntaxiques + racine). Aucun YAML, aucune
    table de nominalisation — le moteur apprend, il ne charge pas de mots.

    Format selon node_type :
      action               → "{verb_lemma}({subject_lemma})"
      etat/transition/     → "{nominalization}({entity})" ou "{verb}({subject})"
      processus
      entite/etat_system.  → "{entity_lemma}"
      condition            → "hidden_cause(?)" (neutre — T-1 : jamais "cause_cachée(?)"
                            côté moteur ; le FR vit dans gcn-frontend-fr)
    """
    subject = _find_subject_lemma(rep)
    entity = _find_entity_lemma(rep)
    nom = rep.root_lemma

    if node_type == _NT_CONDITION:
        label = "hidden_cause(?)"
    elif node_type in (_NT_ENTITE, _NT_CONCEPT, _NT_ETAT_GLOBAL, "etat_systemique"):
        label = entity or rep.root_lemma
    elif node_type == _NT_EVENEMENT:
        label = f"{entity or rep.root_lemma}"
    # processus, etat_local, contrainte (et compat v2 : action, transition, etat)
    elif entity:
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
