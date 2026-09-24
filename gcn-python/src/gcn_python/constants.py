# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
# Sync avec gcn-core/crates/gcn-ir/src/{ir,node,edge}.rs
NODE_TYPES = ["etat", "action", "transition", "processus", "condition", "entite", "etat_systemique"]
RELATION_TYPES = ["cause", "enable", "prevent", "condition", "concession", "sequence",
                  "motivation", "filter", "opposition", "data_dependency", "control_dependency"]

# Types inverses pour message passing bidirectionnel (Phase 2b)
# Indices 0-10 = forward, indices 11-21 = backward (r_inv = r + 11)
RELATION_TYPES_INV = [r + "_inv" for r in RELATION_TYPES]
ALL_RELATION_TYPES = RELATION_TYPES + RELATION_TYPES_INV  # 22 types
SCOPE_VALUES = ["universal", "existential", "partial", "null", "specific", "unknown"]

NODE_ORIGIN_VALUES = ["explicit", "inferred", "hypothetical"]
TEMPORAL_REF_DEFAULT = "unresolved"
AGENT_TYPE_VALUES = ["human", "collective", "institutional", "natural"]  # RÉSERVÉ — non utilisé v2.0

# Universal Dependencies constants
UPOS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
             "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "_unk"]  # 18 + _unk = 19

UD_DEP_RELS = ["acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp", "clf",
               "compound", "conj", "cop", "csubj", "dep", "det", "discourse", "dislocated",
               "expl", "fixed", "flat", "goeswith", "iobj", "list", "mark", "nmod", "nsubj",
               "nummod", "obj", "obl", "orphan", "parataxis", "punct", "reparandum", "root",
               "vocative", "xcomp", "_unk"]  # 37 + _unk = 38

UD_TENSE_VALUES = ["Pres", "Past", "Fut", "Imp", "_absent"]      # 5
UD_ASPECT_VALUES = ["Perf", "Imp", "Prog", "_absent"]             # 4
UD_MOOD_VALUES = ["Ind", "Sub", "Cnd", "Imp", "_absent"]          # 5 (S-3 : UD utilise Cnd, pas Cond)
UD_POLARITY_VALUES = ["Neg"]                                       # 1 binaire

# D_clause = 19 + 38 + 5(subject_pos) + 5 + 4 + 5 + 1 + 3(flags) + N_taxonomy
# D_conn = 19(marker_upos) + N_taxonomy + 2(position)

# Subject POS categories
SUBJECT_POS_CATS = ["PRON", "NOUN", "PROPN", "_other", "_absent"]  # 5
