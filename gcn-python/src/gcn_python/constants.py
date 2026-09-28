# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
# Sync avec gcn-core/crates/gcn-ir/src/{ir,node,edge}.rs
# D5 ETUDE — 8 types canoniques (renommage + fusion + ajout)
NODE_TYPES = [
    "processus",   # absorbe action + transition (D5 ETUDE)
    "etat_local",  # était "etat"
    "etat_global", # était "etat_systemique"
    "entite",
    "condition",
    "concept",     # nouveau D5
    "evenement",   # nouveau D5
    "contrainte",
]
# Aliases lecture pour données v2 (migration transparente)
NODE_TYPE_ALIASES: dict[str, str] = {
    "etat":            "etat_local",
    "etat_local":      "etat_local",
    "etat_global":     "etat_global",
    "etat_systemique": "etat_global",
    "action":          "processus",
    "transition":      "processus",
}
RELATION_TYPES = [
    # 11 relations directes (existantes)
    "cause", "enable", "prevent", "condition", "concession", "sequence",
    "motivation", "filter", "opposition", "data_dependency", "control_dependency",
    # 8 nouvelles relations v3.0 (Analogy, Counterfactual, ternaires)
    "analogy", "counterfactual",
    "conditional_cause", "mediated_cause", "joint_cause",
    "conditional_prevent", "mediated_prevent", "joint_prevent",
]  # 19 total

# 11 relations partagées v2↔v3 — pour K2 (macro-F1 sur classes comparables)
RELATION_TYPES_V2: list[str] = [
    "cause", "enable", "prevent", "condition", "concession", "sequence",
    "motivation", "filter", "opposition", "data_dependency", "control_dependency",
]

# Types inverses pour message passing bidirectionnel (v3.0)
# Indices 0-18 = forward, indices 19-37 = backward (r_inv = r + 19)
RELATION_TYPES_INV = [r + "_inv" for r in RELATION_TYPES]
ALL_RELATION_TYPES = RELATION_TYPES + RELATION_TYPES_INV  # 38 types
SCOPE_VALUES = ["universal", "existential", "partial", "null", "specific", "unknown"]

NODE_ORIGIN_VALUES = ["explicit", "inferred", "hypothetical"]
TEMPORAL_REF_DEFAULT = "unresolved"
AGENT_TYPE_VALUES = ["human", "collective", "institutional", "natural"]  # RÉSERVÉ — non utilisé v2.0

# Universal Dependencies constants
UPOS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
             "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "_unk"]  # 18 (_unk inclus)

UD_DEP_RELS = ["acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp", "clf",
               "compound", "conj", "cop", "csubj", "dep", "det", "discourse", "dislocated",
               "expl", "fixed", "flat", "goeswith", "iobj", "list", "mark", "nmod", "nsubj",
               "nummod", "obj", "obl", "orphan", "parataxis", "punct", "reparandum", "root",
               "vocative", "xcomp", "_unk"]  # 37 + _unk = 38

UD_TENSE_VALUES = ["Pres", "Past", "Fut", "Imp", "_absent"]      # 5
UD_ASPECT_VALUES = ["Perf", "Imp", "Prog", "_absent"]             # 4
UD_MOOD_VALUES = ["Ind", "Sub", "Cnd", "Imp", "_absent"]          # 5 (S-3 : UD utilise Cnd, pas Cond)
UD_POLARITY_VALUES = ["Neg"]                                       # 1 binaire
UD_VOICE_VALUES    = ["Act", "Pass", "_absent"]                    # 3 — v3.0
UD_PRONTYPE_VALUES = ["Int", "Rel", "Prs", "Dem", "Ind", "Art", "_absent"]  # 7 — ETUDE §7.2

# d_clause (v3.0) = 18(UPOS) + 38(dep_rel) + 5(subject_pos) + 5(tense) + 4(aspect)
#                 + 5(mood) + 3(voice) + 7(prontype) + 1(polarity) + 3(flags)
#                 + 12(positionnels Éq.9) + 5(ternaires) = 106
# d_conn = 18(marker_upos) + N_taxonomy + 2(position)

# Subject POS categories
SUBJECT_POS_CATS = ["PRON", "NOUN", "PROPN", "_other", "_absent"]  # 5

# D4/§15 ETUDE — groupes coarse→fine (niveau 1 : position+POS sans Mood/Tense)
COARSE_RELATION_GROUPS: dict[str, list[str]] = {
    "causale":       ["cause", "enable", "condition", "motivation",
                      "conditional_cause", "mediated_cause", "joint_cause"],
    "restrictive":   ["filter", "concession",
                      "conditional_prevent", "mediated_prevent", "joint_prevent"],
    "negative":      ["prevent", "opposition"],
    "sequentielle":  ["sequence"],
    "systemique":    ["data_dependency", "control_dependency"],
    # analogy, counterfactual : hors §15, pas de groupe coarse → restent fine
}
FINE_TO_COARSE_RELATION: dict[str, str] = {
    fine: coarse
    for coarse, fines in COARSE_RELATION_GROUPS.items()
    for fine in fines
}
COARSE_RELATION_TYPES: list[str] = list(COARSE_RELATION_GROUPS)

COARSE_NODE_GROUPS: dict[str, list[str]] = {
    "action_coarse":    ["processus"],
    "nominal_coarse":   ["entite", "concept"],
    "evenement_coarse": ["etat_local", "etat_global", "evenement"],
    "condition_coarse": ["condition", "contrainte"],
}
FINE_TO_COARSE_NODE: dict[str, str] = {
    fine: coarse
    for coarse, fines in COARSE_NODE_GROUPS.items()
    for fine in fines
}
COARSE_NODE_TYPES: list[str] = list(COARSE_NODE_GROUPS)


def coarse_relation(rel: str) -> str:
    """Mappe un type fin vers son groupe coarse. Types hors §15 → inchangés."""
    return FINE_TO_COARSE_RELATION.get(rel, rel)


def coarse_node(nt: str) -> str:
    """Mappe un type nœud fin vers son groupe coarse."""
    return FINE_TO_COARSE_NODE.get(nt, nt)


# §9.4 ETUDE — algèbre de négation : mapping cause → prevent (R1)
NEGATION_PREVENT_MAP: dict[str, str] = {
    "cause":             "prevent",
    "conditional_cause": "conditional_prevent",
    "mediated_cause":    "mediated_prevent",
    "joint_cause":       "joint_prevent",
}

# Éq.11 ETUDE — seuil d'ambiguïté (calibré sur validation, prior 0.65)
THETA_AMBIGUITY_DEFAULT: float = 0.65

# Éq.6 ETUDE — types d'intention (tête MLP apprise, n_intent_types=0 = désactivé)
INTENT_TYPES: list[str] = [
    # Causal
    "explain", "effects", "abduct", "counterfactual",
    # Path
    "chain", "chain_t", "before", "delay",
    # Structure
    "spof", "centrality", "analogy",
    # Méta-qualité
    "summarize", "density", "coverage", "reliability", "diff",
    # Navigation
    "zoom_in", "zoom_out", "aggregate",
    # Génération
    "verbalize",
    # Extraction (déclaratif → CIR)
    "none",
]
