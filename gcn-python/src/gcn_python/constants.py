# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
# Sync avec gcn-core/crates/gcn-ir/src/{ir,node,edge}.rs
NODE_TYPES = [
    "etat", "action", "transition", "processus", "condition",
    "entite", "etat_systemique", "contrainte",               # 8 types — v3.0
]
RELATION_TYPES = [
    # 11 relations directes (existantes)
    "cause", "enable", "prevent", "condition", "concession", "sequence",
    "motivation", "filter", "opposition", "data_dependency", "control_dependency",
    # 8 nouvelles relations v3.0 (Analogy, Counterfactual, ternaires)
    "analogy", "counterfactual",
    "conditional_cause", "mediated_cause", "joint_cause",
    "conditional_prevent", "mediated_prevent", "joint_prevent",
]  # 19 total

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
