# Sync avec gcn-core/crates/gcn-ir/src/{ir,node,edge}.rs
NODE_TYPES = ["etat", "action", "transition", "processus", "condition", "entite", "etat_systemique"]
RELATION_TYPES = ["cause", "enable", "prevent", "condition", "concession", "sequence",
                  "motivation", "filter", "opposition", "data_dependency", "control_dependency"]
SCOPE_VALUES = ["universal", "existential", "partial", "null", "specific", "unknown"]
NODE_ORIGIN_VALUES = ["explicit", "inferred", "hypothetical"]
AGENT_TYPE_VALUES = ["human", "collective", "institutional", "natural"]

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
UD_MOOD_VALUES = ["Ind", "Sub", "Cond", "Imp", "_absent"]         # 5
UD_POLARITY_VALUES = ["Neg"]                                       # 1 binaire

# D_clause = 19 + 38 + 5(subject_pos) + 5 + 4 + 5 + 1 + 3(flags) + N_taxonomy
# D_conn = 19(marker_upos) + N_taxonomy + 2(position)

# spaCy models by language code
SPACY_MODELS: dict[str, str] = {
    "fr": "fr_core_news_sm",
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "es": "es_core_news_sm",
    "it": "it_core_news_sm",
    "pt": "pt_core_news_sm",
    "nl": "nl_core_news_sm",
    "zh": "zh_core_web_sm",
    "ja": "ja_core_news_sm",
}

# Subject POS categories
SUBJECT_POS_CATS = ["PRON", "NOUN", "PROPN", "_other", "_absent"]  # 5
