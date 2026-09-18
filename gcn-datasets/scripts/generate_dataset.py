#!/usr/bin/env python3
"""
Générateur de dataset d'entraînement pour GCN-Core.

Génère ~1000 phrases françaises avec annotations causales complètes :
- Tokens (id, form, lemma, pos, dep_rel, dep_head, morph, gcn)
- CIR (nodes avec node_type, token_span, attributes + edges avec relation)
- Compatible avec load_sentences() de json_reader.py

Usage :
    python gcn-datasets/generate_dataset.py
    # → gcn-datasets/corpus/generated_1000.json
    # → gcn-datasets/splits/{train,val,test}.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Lexique
# ---------------------------------------------------------------------------

# Noms (sujets ou objets selon contexte)
NOUNS = [
    "marché", "ventes", "pluie", "chômage", "qualité", "température",
    "demande", "production", "coût", "prix", "pollution", "croissance",
    "innovation", "stress", "consommation", "énergie", "eau", "forêt",
    "biodiversité", "salaire", "revenu", "emploi", "santé", "sécurité",
    "éducation", "recherche", "transport", "bâtiment", "climat", "société",
    "entreprise", "usine", "ferme", "école", "hôpital", "route", "réseau",
    "système", "projet", "programme", "plan", "mesure", "règle", "loi",
    "technologie", "donnée", "information", "connaissance", "compétence",
]

# Verbes (infinitif) — catégorisés par rôle causal
VERBS_ACTION = [
    "réduire", "augmenter", "limiter", "soutenir", "développer",
    "améliorer", "corriger", "renforcer", "accélérer", "freiner",
    "contrôler", "maintenir", "optimiser", "adapter", "transformer",
    "construire", "détruire", "créer", "supprimer", "modifier",
]

VERBS_PROCESS = [
    "baisser", "croître", "diminuer", "augmenter", "stagner",
    "fluctuer", "stabiliser", "progresser", "reculer", "varier",
    "souffrir", "décliner", "persister", "évoluer", "changer",
]

VERBS_TRANSITION = [
    "chuter", "s'effondrer", "s'améliorer", "se détériorer", "se stabiliser",
    "exploser", "s'écrouler", "bondir", "plonger", "rebondir",
    "diminuer", "augmenter", "tomber", "monter", "descendre",
]

# Marqueurs causaux
MARKERS = {
    "condition": [("si", "SCONJ"), ("à condition que", "SCONJ"), ("seulement si", "SCONJ"),
                  ("pourvu que", "SCONJ"), ("si jamais", "SCONJ"), ("dans le cas où", "SCONJ")],
    "cause": [("parce que", "SCONJ"), ("car", "SCONJ"), ("puisque", "SCONJ"),
              ("comme", "SCONJ"), ("étant donné que", "SCONJ"), ("en raison de", "ADP")],
    "concession": [("bien que", "SCONJ"), ("malgré", "ADP"), ("quoique", "SCONJ"),
                   ("même si", "SCONJ"), ("bien que", "SCONJ"), ("encore que", "SCONJ")],
    "motivation": [("pour", "ADP"), ("afin de", "ADP"), ("pour que", "SCONJ"),
                   ("dans le but de", "ADP"), ("en vue de", "ADP"), ("afin que", "SCONJ")],
    "enable": [("permet à", "VERB"), ("facilite", "VERB"), ("autorise", "VERB"),
               ("ouvre la voie à", "VERB"), ("donne accès à", "VERB"), ("rend possible", "VERB")],
    "prevent": [("empêche", "VERB"), ("bloque", "VERB"), ("interdit", "VERB"),
                ("empêche de", "VERB"), ("fait obstacle à", "VERB"), ("nuit à", "VERB")],
    "sequence": [("d'abord", "ADV"), ("ensuite", "ADV"), ("puis", "ADV"),
                 ("après", "ADP"), ("avant de", "ADP"), ("une fois que", "SCONJ")],
    "filter": [("sépare", "VERB"), ("filtre", "VERB"), ("trier", "VERB"),
               ("sélectionner", "VERB"), ("distinguer", "VERB"), ("classer", "VERB")],
    "opposition": [("au lieu de", "ADP"), ("plutôt que", "SCONJ"), ("pas", "ADV"),
                   ("ne pas", "ADV"), ("jamais", "ADV"), ("ni", "CCONJ")],
    "data_dependency": [("lit", "VERB"), ("consulte", "VERB"), ("exploite", "VERB"),
                        ("utilise les données de", "VERB"), ("requiert", "VERB"), ("nécessite", "VERB")],
    "control_dependency": [("appelle", "VERB"), ("déclenche", "VERB"), ("exécute", "VERB"),
                           ("invoque", "VERB"), ("lançait", "VERB"), ("ordonne", "VERB")],
}

# Adjectifs
ADJECTIFS = [
    "important", "grave", "sévère", "léger", "significatif",
    "notable", "critique", "majeur", "modeste", "considérable",
    "profond", "faible", "élevé", "stable", "permanent",
]

# Determinants
DET_MASC = ["le", "un", "ce", "chaque", "tout"]
DET_FEM = ["la", "une", "cette", "chaque", "toute"]
DET_PLUR = ["les", "des", "ces", "tous", "toutes"]

# Pronoms
PRONOUNS = ["il", "elle", "on", "nous", "ils", "elles"]

# ---------------------------------------------------------------------------
# Templates : (relation, structure_template, description)
# Structure : liste de (type_token, contenu)
# Types : N=nom, V=verbe_action, P=verbe_process, T=verbe_transition,
#         M=marqueur, D=determinant, A=adjectif, PR=pronom, AUX=auxiliaire,
#         PUNCT=punctuation
# ---------------------------------------------------------------------------

def _n(det: str, noun: str, adj: str | None = None) -> list[tuple[str, str]]:
    """Construit un groupe nominal : DET NOUN [ADJ]."""
    tokens = [("DET", det), ("NOUN", noun)]
    if adj:
        tokens.append(("ADJ", adj))
    return tokens

def _v(verb: str, tense: str = "Pres", mood: str = "Ind", person: str = "3s") -> tuple[str, str, str, str]:
    """Retourne (form, lemma, tense, mood)."""
    return (verb, verb, tense, mood)

# Templates par relation
# Chaque template est une liste de segments. Chaque segment est un tuple (type, contenu).
# Les segments sont assemblés dans l'ordre pour former la phrase.

RELATION_TEMPLATES: dict[str, list[tuple[str, list]]] = {
    "cause": [
        # Sujet VERB_cause OBJET.
        ("cause_simple", [
            ("NP_subj", None), ("VERB_c", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        # NP_subj VERB_cause parce que NP_subj2 VERB_effet.
        ("cause_parce_que", [
            ("NP_subj", None), ("VERB_c", None), ("parce que", None),
            ("NP_subj2", None), ("VERB_e", None), ("PUNCT", ".")
        ]),
        # NP_subj, qui VERB_cause, VERB_effet OBJET.
        ("cause_relative", [
            ("NP_subj", None), (",", None), ("qui", None), ("VERB_c", None),
            (",", None), ("VERB_e", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        # C'est NP_subj qui VERB_cause OBJET.
        ("cause_c_est", [
            ("C'est", None), ("NP_subj", None), ("qui", None), ("VERB_c", None),
            ("NP_obj", None), ("PUNCT", ".")
        ]),
        # NP_subj VERB_effet à cause de NP_subj2.
        ("cause_a_cause_de", [
            ("NP_subj", None), ("VERB_e", None), ("à cause de", None),
            ("NP_subj2", None), ("PUNCT", ".")
        ]),
        # Étant donné NP_subj, NP_subj2 VERB_effet.
        ("cause_etant_donne", [
            ("Étant donné", None), ("NP_subj", None), (",", None),
            ("NP_subj2", None), ("VERB_e", None), ("PUNCT", ".")
        ]),
    ],
    "enable": [
        ("enable_simple", [
            ("NP_subj", None), ("permet à", None), ("NP_obj", None),
            ("de", None), ("VERB_e", None), ("PUNCT", ".")
        ]),
        ("enable_grace", [
            ("Grâce à", None), ("NP_subj", None), (",", None),
            ("NP_subj2", None), ("VERB_e", None), ("PUNCT", ".")
        ]),
        ("enable_facilite", [
            ("NP_subj", None), ("facilite", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("enable_rend_possible", [
            ("NP_subj", None), ("rend possible", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("enable_ouvre", [
            ("NP_subj", None), ("ouvre la voie à", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("enable_autorise", [
            ("NP_subj", None), ("autorise", None), ("NP_obj", None),
            ("à", None), ("VERB_e", None), ("PUNCT", ".")
        ]),
    ],
    "prevent": [
        ("prevent_simple", [
            ("NP_subj", None), ("empêche", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("prevent_bloque", [
            ("NP_subj", None), ("bloque", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("prevent_interdit", [
            ("NP_subj", None), ("interdit", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("prevent_nuit", [
            ("NP_subj", None), ("nuis à", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("prevent_obstacle", [
            ("NP_subj", None), ("fait obstacle à", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("prevent_empede_de", [
            ("NP_subj", None), ("empêche", None), ("NP_obj", None),
            ("de", None), ("VERB_e", None), ("PUNCT", ".")
        ]),
    ],
    "condition": [
        ("cond_si", [
            ("Si", None), ("NP_subj", None), ("VERB_e", None), (",", None),
            ("NP_subj2", None), ("VERB_c", None), ("PUNCT", ".")
        ]),
        ("cond_a_condition", [
            ("NP_subj", None), ("VERB_c", None), ("à condition que", None),
            ("NP_subj2", None), ("VERB_e", None), ("PUNCT", ".")
        ]),
        ("cond_seulement_si", [
            ("NP_subj", None), ("VERB_c", None), ("seulement si", None),
            ("NP_subj2", None), ("VERB_e", None), ("PUNCT", ".")
        ]),
        ("cond_pourvu_que", [
            ("NP_subj", None), ("VERB_c", None), ("pourvu que", None),
            ("NP_subj2", None), ("VERB_e", None), ("PUNCT", ".")
        ]),
        ("cond_dans_cas", [
            ("Dans le cas où", None), ("NP_subj", None), ("VERB_e", None), (",", None),
            ("NP_subj2", None), ("VERB_c", None), ("PUNCT", ".")
        ]),
        ("cond_si_jamais", [
            ("Si jamais", None), ("NP_subj", None), ("VERB_e", None), (",", None),
            ("NP_subj2", None), ("VERB_c", None), ("PUNCT", ".")
        ]),
    ],
    "concession": [
        ("conc_bien_que", [
            ("Bien que", None), ("NP_subj", None), ("VERB_e", None), (",", None),
            ("NP_subj2", None), ("VERB_c", None), ("PUNCT", ".")
        ]),
        ("conc_malgre", [
            ("Malgré", None), ("NP_subj", None), (",", None),
            ("NP_subj2", None), ("VERB_c", None), ("PUNCT", ".")
        ]),
        ("conc_meme_si", [
            ("Même si", None), ("NP_subj", None), ("VERB_e", None), (",", None),
            ("NP_subj2", None), ("VERB_c", None), ("PUNCT", ".")
        ]),
        ("conc_quoique", [
            ("Quoique", None), ("NP_subj", None), ("VERB_e", None), (",", None),
            ("NP_subj2", None), ("VERB_c", None), ("PUNCT", ".")
        ]),
        ("conc_encore_que", [
            ("Encore que", None), ("NP_subj", None), ("VERB_e", None), (",", None),
            ("NP_subj2", None), ("VERB_c", None), ("PUNCT", ".")
        ]),
        ("conc_bien_que2", [
            ("NP_subj", None), ("VERB_c", None), ("bien que", None),
            ("NP_subj2", None), ("VERB_e", None), ("PUNCT", ".")
        ]),
    ],
    "motivation": [
        ("mot_pour", [
            ("NP_subj", None), ("VERB_c", None), ("pour", None), ("VERB_e", None),
            ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("mot_afin_de", [
            ("NP_subj", None), ("VERB_c", None), ("afin de", None), ("VERB_e", None),
            ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("mot_dans_le_but", [
            ("NP_subj", None), ("VERB_c", None), ("dans le but de", None),
            ("VERB_e", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("mot_en_vue_de", [
            ("NP_subj", None), ("VERB_c", None), ("en vue de", None),
            ("VERB_e", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("mot_pour_que", [
            ("NP_subj", None), ("VERB_c", None), ("pour que", None),
            ("NP_subj2", None), ("VERB_e", None), ("PUNCT", ".")
        ]),
        ("mot_afin_que", [
            ("NP_subj", None), ("VERB_c", None), ("afin que", None),
            ("NP_subj2", None), ("VERB_e", None), ("PUNCT", ".")
        ]),
    ],
    "sequence": [
        ("seq_dabord", [
            ("D'abord", None), ("NP_subj", None), ("VERB_e", None), (",", None),
            ("ensuite", None), ("NP_subj2", None), ("VERB_c", None), ("PUNCT", ".")
        ]),
        ("seq_puis", [
            ("NP_subj", None), ("VERB_e", None), (",", None),
            ("puis", None), ("NP_subj2", None), ("VERB_c", None), ("PUNCT", ".")
        ]),
        ("seq_apres", [
            ("Après", None), ("avoir", None), ("VERB_e", None), (",", None),
            ("NP_subj", None), ("VERB_c", None), ("PUNCT", ".")
        ]),
        ("seq_avant_de", [
            ("Avant de", None), ("VERB_e", None), (",", None),
            ("NP_subj", None), ("VERB_c", None), ("PUNCT", ".")
        ]),
        ("seq_une_fois_que", [
            ("Une fois que", None), ("NP_subj", None), ("VERB_e", None), (",", None),
            ("NP_subj2", None), ("VERB_c", None), ("PUNCT", ".")
        ]),
        ("seq_a_mesure_que", [
            ("À mesure que", None), ("NP_subj", None), ("VERB_e", None), (",", None),
            ("NP_subj2", None), ("VERB_c", None), ("PUNCT", ".")
        ]),
    ],
    "filter": [
        ("filt_simple", [
            ("NP_subj", None), ("sépare", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("filt_filtre", [
            ("NP_subj", None), ("filtre", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("filt_classe", [
            ("NP_subj", None), ("classe", None), ("NP_obj", None),
            ("selon", None), ("NP_crit", None), ("PUNCT", ".")
        ]),
        ("filt_selectionne", [
            ("NP_subj", None), ("sélectionne", None), ("NP_obj", None),
            ("parmi", None), ("NP_crit", None), ("PUNCT", ".")
        ]),
        ("filt_distingue", [
            ("NP_subj", None), ("distingue", None), ("NP_obj", None),
            ("de", None), ("NP_crit", None), ("PUNCT", ".")
        ]),
        ("filt_trie", [
            ("NP_subj", None), ("trie", None), ("NP_obj", None),
            ("en", None), ("NP_crit", None), ("PUNCT", ".")
        ]),
    ],
    "opposition": [
        ("opp_au_lieu_de", [
            ("NP_subj", None), ("VERB_e", None), ("au lieu de", None),
            ("VERB_c", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("opp_plutot_que", [
            ("NP_subj", None), ("VERB_c", None), ("plutôt que", None),
            ("VERB_e", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("opp_pas", [
            ("NP_subj", None), ("ne", None), ("VERB_e", None), ("pas", None),
            ("mais", None), ("VERB_c", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("opp_ni", [
            ("NP_subj", None), ("ne", None), ("VERB_e", None), ("ni", None),
            ("VERB_c", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("opp_jamais", [
            ("NP_subj", None), ("ne", None), ("VERB_e", None), ("jamais", None),
            ("mais", None), ("VERB_c", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("opp_en_revanche", [
            ("NP_subj", None), ("VERB_e", None), (",", None),
            ("en revanche", None), ("NP_subj2", None), ("VERB_c", None), ("PUNCT", ".")
        ]),
    ],
    "data_dependency": [
        ("data_lit", [
            ("NP_subj", None), ("lit", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("data_consulte", [
            ("NP_subj", None), ("consulte", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("data_exploite", [
            ("NP_subj", None), ("exploite", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("data_requiert", [
            ("NP_subj", None), ("requiert", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("data_necessite", [
            ("NP_subj", None), ("nécessite", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("data_utilise", [
            ("NP_subj", None), ("utilise les données de", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
    ],
    "control_dependency": [
        ("ctrl_appelle", [
            ("NP_subj", None), ("appelle", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("ctrl_declenche", [
            ("NP_subj", None), ("déclenche", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("ctrl_execute", [
            ("NP_subj", None), ("exécute", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("ctrl_invoque", [
            ("NP_subj", None), ("invoque", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("ctrl_lance", [
            ("NP_subj", None), ("lance", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
        ("ctrl_ordonne", [
            ("NP_subj", None), ("ordonne", None), ("NP_obj", None), ("PUNCT", ".")
        ]),
    ],
}

# ---------------------------------------------------------------------------
# Conjugaison simplifiée
# ---------------------------------------------------------------------------

CONJ = {
    # verbe, personne(1s,2s,3s,1p,2p,3p), temps
    "réduire": {"3s_pres": "réduit", "3s_past": "a réduit", "3s_fut": "réduira"},
    "augmenter": {"3s_pres": "augmente", "3s_past": "a augmenté", "3s_fut": "augmentera"},
    "limiter": {"3s_pres": "limite", "3s_past": "a limité", "3s_fut": "limitera"},
    "soutenir": {"3s_pres": "soutient", "3s_past": "a soutenu", "3s_fut": "soutiendra"},
    "développer": {"3s_pres": "développe", "3s_past": "a développé", "3s_fut": "développera"},
    "améliorer": {"3s_pres": "améliore", "3s_past": "amélioré", "3s_fut": "améliorera"},
    "corriger": {"3s_pres": "corrige", "3s_past": "corrigé", "3s_fut": "corrigera"},
    "renforcer": {"3s_pres": "renforce", "3s_past": "renforcé", "3s_fut": "renforcera"},
    "accélérer": {"3s_pres": "accélère", "3s_past": "accéléré", "3s_fut": "accélérera"},
    "freiner": {"3s_pres": "freine", "3s_past": "freiné", "3s_fut": "freinera"},
    "contrôler": {"3s_pres": "contrôle", "3s_past": "contrôlé", "3s_fut": "contrôlera"},
    "maintenir": {"3s_pres": "maintient", "3s_past": "maintenu", "3s_fut": "maintiendra"},
    "optimiser": {"3s_pres": "optimise", "3s_past": "optimisé", "3s_fut": "optimisera"},
    "adapter": {"3s_pres": "adapte", "3s_past": "adapté", "3s_fut": "adaptera"},
    "transformer": {"3s_pres": "transforme", "3s_past": "transformé", "3s_fut": "transformera"},
    "construire": {"3s_pres": "construit", "3s_past": "construit", "3s_fut": "construira"},
    "détruire": {"3s_pres": "détruit", "3s_past": "détruit", "3s_fut": "détruira"},
    "créer": {"3s_pres": "crée", "3s_past": "créé", "3s_fut": "créera"},
    "supprimer": {"3s_pres": "supprime", "3s_past": "supprimé", "3s_fut": "supprimera"},
    "modifier": {"3s_pres": "modifie", "3s_past": "modifié", "3s_fut": "modifiera"},
    "baisser": {"3s_pres": "baisse", "3s_past": "a baissé", "3s_fut": "baissera"},
    "croître": {"3s_pres": "croît", "3s_past": "a crû", "3s_fut": "croîtra"},
    "diminuer": {"3s_pres": "diminue", "3s_past": "diminué", "3s_fut": "diminuera"},
    "stagner": {"3s_pres": "stagne", "3s_past": "stagné", "3s_fut": "stagnera"},
    "fluctuer": {"3s_pres": "fluctue", "3s_past": "fluctué", "3s_fut": "fluctuera"},
    "stabiliser": {"3s_pres": "stabilise", "3s_past": "stabilisé", "3s_fut": "stabilisera"},
    "progresser": {"3s_pres": "progresse", "3s_past": "progressé", "3s_fut": "progressera"},
    "reculer": {"3s_pres": "recule", "3s_past": "reculé", "3s_fut": "reculera"},
    "varier": {"3s_pres": "varie", "3s_past": "varié", "3s_fut": "variera"},
    "souffrir": {"3s_pres": "souffre", "3s_past": "souffert", "3s_fut": "souffrira"},
    "décliner": {"3s_pres": "décline", "3s_past": "décliné", "3s_fut": "déclinera"},
    "persister": {"3s_pres": "persiste", "3s_past": "persisté", "3s_fut": "persistera"},
    "évoluer": {"3s_pres": "évolue", "3s_past": "évolué", "3s_fut": "évoluera"},
    "changer": {"3s_pres": "change", "3s_past": "changé", "3s_fut": "changera"},
    "chuter": {"3s_pres": "chute", "3s_past": "chuté", "3s_fut": "chutera"},
    "s'effondrer": {"3s_pres": "s'effondre", "3s_past": "s'est effondré", "3s_fut": "s'effondrera"},
    "s'améliorer": {"3s_pres": "s'améliore", "3s_past": "s'est amélioré", "3s_fut": "s'améliorera"},
    "se détériorer": {"3s_pres": "se détériore", "3s_past": "s'est détérioré", "3s_fut": "se détériorera"},
    "se stabiliser": {"3s_pres": "se stabilise", "3s_past": "s'est stabilisé", "3s_fut": "se stabilisera"},
    "exploser": {"3s_pres": "explose", "3s_past": "explosé", "3s_fut": "explosera"},
    "s'écrouler": {"3s_pres": "s'écroule", "3s_past": "s'est écroulé", "3s_fut": "s'écroulera"},
    "bondir": {"3s_pres": "bondit", "3s_past": "bondi", "3s_fut": "bondira"},
    "plonger": {"3s_pres": "plonge", "3s_past": "plongé", "3s_fut": "plongera"},
    "rebondir": {"3s_pres": "rebondit", "3s_past": "rebondi", "3s_fut": "rebondira"},
    "tomber": {"3s_pres": "tombe", "3s_past": "tombé", "3s_fut": "tombera"},
    "monter": {"3s_pres": "monte", "3s_past": "monté", "3s_fut": "montera"},
    "descendre": {"3s_pres": "descend", "3s_past": "descendu", "3s_fut": "descendra"},
    "permettre": {"3s_pres": "permet", "3s_past": "permis", "3s_fut": "permettra"},
    "faciliter": {"3s_pres": "facilite", "3s_past": "facilité", "3s_fut": "facilitera"},
    "empêcher": {"3s_pres": "empêche", "3s_past": "empêché", "3s_fut": "empêchera"},
    "bloquer": {"3s_pres": "bloque", "3s_past": "bloqué", "3s_fut": "bloquera"},
    "interdire": {"3s_pres": "interdit", "3s_past": "interdit", "3s_fut": "interdira"},
    "nuire": {"3s_pres": "nuit", "3s_past": "nui", "3s_fut": "nuira"},
    "séparer": {"3s_pres": "sépare", "3s_past": "séparé", "3s_fut": "séparera"},
    "filtrer": {"3s_pres": "filtre", "3s_past": "filtré", "3s_fut": "filtrera"},
    "trier": {"3s_pres": "trie", "3s_past": "trié", "3s_fut": "triera"},
    "sélectionner": {"3s_pres": "sélectionne", "3s_past": "sélectionné", "3s_fut": "sélectionnera"},
    "distinguer": {"3s_pres": "distingue", "3s_past": "distingué", "3s_fut": "distinguera"},
    "classer": {"3s_pres": "classe", "3s_past": "classé", "3s_fut": "classera"},
    "lire": {"3s_pres": "lit", "3s_past": "lu", "3s_fut": "lira"},
    "consulter": {"3s_pres": "consulte", "3s_past": "consulté", "3s_fut": "consultera"},
    "exploiter": {"3s_pres": "exploite", "3s_past": "exploité", "3s_fut": "exploitera"},
    "requérir": {"3s_pres": "requiert", "3s_past": "requis", "3s_fut": "requerra"},
    "nécessiter": {"3s_pres": "nécessite", "3s_past": "nécessité", "3s_fut": "nécessitera"},
    "utiliser": {"3s_pres": "utilise", "3s_past": "utilisé", "3s_fut": "utilisera"},
    "appeler": {"3s_pres": "appelle", "3s_past": "appelé", "3s_fut": "appellera"},
    "déclencher": {"3s_pres": "déclenche", "3s_past": "déclenché", "3s_fut": "déclenchera"},
    "exécuter": {"3s_pres": "exécute", "3s_past": "exécuté", "3s_fut": "exécutera"},
    "invoquer": {"3s_pres": "invoque", "3s_past": "invoqué", "3s_fut": "invoquera"},
    "lancer": {"3s_pres": "lance", "3s_past": "lancé", "3s_fut": "lancera"},
    "ordonner": {"3s_pres": "ordonne", "3s_past": "ordonné", "3s_fut": "ordonnera"},
    "travailler": {"3s_pres": "travaille", "3s_past": "travaillé", "3s_fut": "travaillera"},
    "réussir": {"3s_pres": "réussit", "3s_past": "réussi", "3s_fut": "réussira"},
}

def conjugue(verb: str, form_key: str = "3s_pres") -> str:
    """Conjuge un verbe. Retourne le verbe brut si non trouvé."""
    if verb in CONJ:
        return CONJ[verb].get(form_key, verb)
    # fallback : ajouter -e ou -s
    return verb


# ---------------------------------------------------------------------------
# Déterminants
# ---------------------------------------------------------------------------

def get_det(noun: str) -> str:
    """Retourne un déterminant adapté au nom."""
    # heuristic simple
    if noun.endswith(("s", "x")):
        return random.choice(DET_PLUR)
    if noun.endswith(("e", "ion", "té", "ure", "ence", "ance")):
        return random.choice(DET_FEM)
    return random.choice(DET_MASC)


# ---------------------------------------------------------------------------
# Générateur de tokens
# ---------------------------------------------------------------------------

# Node type mapping : verbe -> node_type causal
VERB_TO_NODE_TYPE = {
    # action verbs
    "réduire": "action", "augmenter": "action", "limiter": "action",
    "soutenir": "action", "développer": "action", "améliorer": "action",
    "corriger": "action", "renforcer": "action", "accélérer": "action",
    "freiner": "action", "contrôler": "action", "maintenir": "action",
    "optimiser": "action", "adapter": "action", "transformer": "action",
    "construire": "action", "détruire": "action", "créer": "action",
    "supprimer": "action", "modifier": "action",
    # process verbs
    "baisser": "processus", "croître": "processus", "diminuer": "processus",
    "stagner": "processus", "fluctuer": "processus", "stabiliser": "processus",
    "progresser": "processus", "reculer": "processus", "varier": "processus",
    "souffrir": "processus", "décliner": "processus", "persister": "processus",
    "évoluer": "processus", "changer": "processus",
    # transition verbs
    "chuter": "transition", "s'effondrer": "transition", "s'améliorer": "transition",
    "se détériorer": "transition", "se stabiliser": "transition",
    "exploser": "transition", "s'écrouler": "transition",
    "bondir": "transition", "plonger": "transition", "rebondir": "transition",
    "tomber": "transition", "monter": "transition", "descendre": "transition",
    # enable/prevent as action
    "permettre": "action", "faciliter": "action", "empêcher": "action",
    "bloquer": "action", "interdire": "action", "nuire": "action",
    # filter
    "séparer": "action", "filtrer": "action", "trier": "action",
    "sélectionner": "action", "distinguer": "action", "classer": "action",
    # data/control
    "lire": "action", "consulter": "action", "exploiter": "action",
    "requérir": "action", "nécessiter": "action", "utiliser": "action",
    "appeler": "action", "déclencher": "action", "exécuter": "action",
    "invoquer": "action", "lancer": "action", "ordonner": "action",
    "travailler": "action", "réussir": "transition",
}

NOUN_TO_NODE_TYPE = {
    "marché": "etat", "ventes": "processus", "pluie": "processus",
    "chômage": "etat", "qualité": "etat", "température": "processus",
    "demande": "processus", "production": "processus", "coût": "entite",
    "prix": "entite", "pollution": "processus", "croissance": "processus",
    "innovation": "processus", "stress": "etat", "consommation": "processus",
    "énergie": "entite", "eau": "entite", "forêt": "entite",
    "biodiversité": "processus", "salaire": "entite", "revenu": "entite",
    "emploi": "processus", "santé": "etat", "sécurité": "etat",
    "éducation": "processus", "recherche": "processus", "transport": "processus",
    "bâtiment": "entite", "climat": "etat_systemique", "société": "etat_systemique",
    "entreprise": "entite", "usine": "entite", "ferme": "entite",
    "école": "entite", "hôpital": "entite", "route": "entite",
    "réseau": "entite", "système": "etat_systemique", "projet": "processus",
    "programme": "processus", "plan": "processus", "mesure": "processus",
    "règle": "entite", "loi": "entite", "technologie": "entite",
    "donnée": "entite", "information": "entite", "connaissance": "entite",
    "compétence": "entite",
}


def make_token(token_id: int, form: str, pos: str, dep_rel: str, dep_head: int,
               lemma: str | None = None, morph: dict | None = None,
               gcn_type: str | None = None, gcn_class: str | None = None) -> dict:
    """Construit un token dict."""
    tok = {
        "id": token_id,
        "form": form,
        "lemma": lemma or form.lower(),
        "pos": pos,
        "dep_rel": dep_rel,
        "dep_head": dep_head,
        "morph": morph or {},
    }
    if gcn_type:
        tok["gcn"] = {"causal_type": gcn_type, "causal_class": gcn_class or ""}
    return tok


# ---------------------------------------------------------------------------
# Générateur de phrases
# ---------------------------------------------------------------------------

class SentenceGenerator:
    """Génère une phrase avec tokens et CIR."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.sentence_counter = 0

    def _pick(self, lst):
        return self.rng.choice(lst)

    def _pick_n(self, lst, n):
        return self.rng.sample(lst, min(n, len(lst)))

    def generate_sentence(self, relation: str) -> dict:
        """Génère une phrase complète pour une relation donnée."""
        self.sentence_counter += 1
        templates = RELATION_TEMPLATES[relation]
        template_name, template = self._pick(templates)

        # Choisir les mots
        noun1 = self._pick(NOUNS)
        noun2 = self._pick([n for n in NOUNS if n != noun1])
        noun3 = self._pick([n for n in NOUNS if n not in (noun1, noun2)])

        verb_c = self._pick(VERBS_ACTION)
        verb_e = self._pick(VERBS_PROCESS + VERBS_TRANSITION + VERBS_ACTION)

        det1 = get_det(noun1)
        det2 = get_det(noun2)
        det3 = get_det(noun3)

        adj1 = self._pick(ADJECTIFS) if self.rng.random() > 0.5 else None
        adj2 = self._pick(ADJECTIFS) if self.rng.random() > 0.5 else None

        # Construire la phrase
        tokens = []
        token_id = 1
        text_parts = []
        node_spans = {}  # node_id -> (start_token, end_token, noun, verb, node_type)
        edge_list = []

        # Mapper les noms aux node types
        n1_type = NOUN_TO_NODE_TYPE.get(noun1, "entite")
        n2_type = NOUN_TO_NODE_TYPE.get(noun2, "entite")
        vc_type = VERB_TO_NODE_TYPE.get(verb_c, "action")
        ve_type = VERB_TO_NODE_TYPE.get(verb_e, "processus")

        # Construire les nœuds
        # n001 = noun1 (cause/source), n002 = noun2 (effet/cible)
        n001_id = "n001"
        n002_id = "n002"

        # Pour chaque segment du template
        NP_subj_tokens_start = None
        NP_subj_tokens_end = None
        NP_subj2_tokens_start = None
        NP_subj2_tokens_end = None
        NP_obj_tokens_start = None
        NP_obj_tokens_end = None
        NP_crit_tokens_start = None
        NP_crit_tokens_end = None
        VERB_c_token_id = None
        VERB_e_token_id = None
        connector_token_id = None

        for seg_type, seg_content in template:
            if seg_type == "NP_subj":
                NP_subj_tokens_start = token_id
                # DET
                tokens.append(make_token(token_id, det1, "DET", "det", token_id + 1,
                                          lemma=det1.lower()))
                text_parts.append(det1)
                token_id += 1
                # NOUN
                tokens.append(make_token(token_id, noun1, "NOUN", "nsubj", 0,
                                          lemma=noun1, gcn_type="nom", gcn_class=n1_type))
                text_parts.append(noun1)
                NP_subj_tokens_end = token_id
                token_id += 1
                # ADJ (optionnel)
                if adj1:
                    tokens.append(make_token(token_id, adj1, "ADJ", "amod", token_id - 1,
                                              lemma=adj1))
                    text_parts.append(adj1)
                    NP_subj_tokens_end = token_id
                    token_id += 1

            elif seg_type == "NP_subj2":
                NP_subj2_tokens_start = token_id
                tokens.append(make_token(token_id, det2, "DET", "det", token_id + 1,
                                          lemma=det2.lower()))
                text_parts.append(det2)
                token_id += 1
                tokens.append(make_token(token_id, noun2, "NOUN", "nsubj", 0,
                                          lemma=noun2, gcn_type="nom", gcn_class=n2_type))
                text_parts.append(noun2)
                NP_subj2_tokens_end = token_id
                token_id += 1
                if adj2:
                    tokens.append(make_token(token_id, adj2, "ADJ", "amod", token_id - 1,
                                              lemma=adj2))
                    text_parts.append(adj2)
                    NP_subj2_tokens_end = token_id
                    token_id += 1

            elif seg_type == "NP_obj":
                NP_obj_tokens_start = token_id
                tokens.append(make_token(token_id, det2, "DET", "det", token_id + 1,
                                          lemma=det2.lower()))
                text_parts.append(det2)
                token_id += 1
                tokens.append(make_token(token_id, noun2, "NOUN", "obj", 0,
                                          lemma=noun2, gcn_type="nom", gcn_class=n2_type))
                text_parts.append(noun2)
                NP_obj_tokens_end = token_id
                token_id += 1

            elif seg_type == "NP_crit":
                NP_crit_tokens_start = token_id
                tokens.append(make_token(token_id, det3, "DET", "det", token_id + 1,
                                          lemma=det3.lower()))
                text_parts.append(det3)
                token_id += 1
                tokens.append(make_token(token_id, noun3, "NOUN", "nmod", 0,
                                          lemma=noun3, gcn_type="nom", gcn_class="entite"))
                text_parts.append(noun3)
                NP_crit_tokens_end = token_id
                token_id += 1

            elif seg_type.startswith("VERB_c"):
                form = conjugue(verb_c)
                VERB_c_token_id = token_id
                tokens.append(make_token(token_id, form, "VERB", "root", 0,
                                          lemma=verb_c, morph={"Tense": "Pres", "Mood": "Ind"},
                                          gcn_type="verbe", gcn_class=vc_type))
                text_parts.append(form)
                token_id += 1

            elif seg_type.startswith("VERB_e"):
                form = conjugue(verb_e)
                VERB_e_token_id = token_id
                tokens.append(make_token(token_id, form, "VERB", "advcl", 0,
                                          lemma=verb_e, morph={"Tense": "Pres", "Mood": "Ind"},
                                          gcn_type="verbe", gcn_class=ve_type))
                text_parts.append(form)
                token_id += 1

            elif seg_type == "PUNCT":
                tokens.append(make_token(token_id, ".", "PUNCT", "punct", 0))
                text_parts.append(".")
                token_id += 1

            elif seg_type in (",",):
                tokens.append(make_token(token_id, ",", "PUNCT", "punct", 0))
                text_parts.append(",")
                token_id += 1

            elif seg_type == "C'est":
                tokens.append(make_token(token_id, "C'est", "PRON", "expl", token_id + 2,
                                          lemma="ce"))
                text_parts.append("C'est")
                token_id += 1

            elif seg_type == "qui":
                tokens.append(make_token(token_id, "qui", "PRON", "nsubj", 0, lemma="qui"))
                text_parts.append("qui")
                token_id += 1

            elif seg_type == "de":
                tokens.append(make_token(token_id, "de", "ADP", "mark", 0, lemma="de"))
                text_parts.append("de")
                token_id += 1

            elif seg_type == "pour":
                tokens.append(make_token(token_id, "pour", "ADP", "mark", 0, lemma="pour"))
                text_parts.append("pour")
                token_id += 1

            elif seg_type == "parce que":
                connector_token_id = token_id
                tokens.append(make_token(token_id, "parce", "SCONJ", "mark", 0, lemma="parce"))
                text_parts.append("parce")
                token_id += 1
                tokens.append(make_token(token_id, "que", "SCONJ", "fixed", token_id - 1, lemma="que"))
                text_parts.append("que")
                token_id += 1

            elif seg_type == "bien que":
                connector_token_id = token_id
                tokens.append(make_token(token_id, "bien", "SCONJ", "mark", 0, lemma="bien"))
                text_parts.append("bien")
                token_id += 1
                tokens.append(make_token(token_id, "que", "SCONJ", "fixed", token_id - 1, lemma="que"))
                text_parts.append("que")
                token_id += 1

            elif seg_type in ("Si", "Bien que", "Même si", "Quoique", "Encore que",
                              "Étant donné", "Grâce à", "D'abord", "Après", "Avant de",
                              "Une fois que", "À mesure que", "Dans le cas où", "Si jamais",
                              "à condition que", "seulement si", "pourvu que"):
                # Marqueurs multi-mots
                words = seg_type.split()
                connector_token_id = token_id
                for wi, w in enumerate(words):
                    tokens.append(make_token(token_id, w, "SCONJ" if wi > 0 else "SCONJ",
                                              "mark" if wi > 0 else "mark", 0, lemma=w.lower()))
                    text_parts.append(w)
                    token_id += 1

            elif seg_type == "au lieu de":
                connector_token_id = token_id
                for w in ["au", "lieu", "de"]:
                    tokens.append(make_token(token_id, w, "ADP" if w == "de" else "NOUN",
                                              "case" if w == "de" else "obl", 0, lemma=w.lower()))
                    text_parts.append(w)
                    token_id += 1

            elif seg_type == "plutôt que":
                connector_token_id = token_id
                tokens.append(make_token(token_id, "plutôt", "ADV", "mark", 0, lemma="plutôt"))
                text_parts.append("plutôt")
                token_id += 1
                tokens.append(make_token(token_id, "que", "SCONJ", "fixed", token_id - 1, lemma="que"))
                text_parts.append("que")
                token_id += 1

            elif seg_type in ("ensuite", "puis", "alors", "mais", "en revanche"):
                tokens.append(make_token(token_id, seg_type, "ADV", "advmod", 0, lemma=seg_type.lower()))
                text_parts.append(seg_type)
                token_id += 1

            elif seg_type in ("pas", "jamais"):
                tokens.append(make_token(token_id, seg_type, "ADV", "advmod", 0, lemma=seg_type))
                text_parts.append(seg_type)
                token_id += 1

            elif seg_type in ("ni",):
                tokens.append(make_token(token_id, seg_type, "CCONJ", "cc", 0, lemma=seg_type))
                text_parts.append(seg_type)
                token_id += 1

            elif seg_type in ("ne",):
                tokens.append(make_token(token_id, seg_type, "ADV", "advmod", 0, lemma=seg_type))
                text_parts.append(seg_type)
                token_id += 1

            elif seg_type in ("à cause de", "à condition que", "dans le but de",
                              "en vue de", "pour que", "afin que", "afin de",
                              "seulement si", "pourvu que", "dans le cas où", "si jamais"):
                words = seg_type.split()
                connector_token_id = token_id
                for w in words:
                    tokens.append(make_token(token_id, w, "ADP" if w in ("de", "à", "en") else "SCONJ",
                                              "mark", 0, lemma=w.lower()))
                    text_parts.append(w)
                    token_id += 1

            elif seg_type in ("permet à", "facilite", "autorise", "ouvre la voie à",
                              "rend possible", "empêche", "bloque", "interdit",
                              "nuis à", "fait obstacle à", "sépare", "filtre",
                              "classe", "sélectionne", "distingue", "trie",
                              "lit", "consulte", "exploite", "requiert",
                              "nécessite", "utilise les données de",
                              "appelle", "déclenche", "exécute", "invoque",
                              "lance", "ordonne"):
                words = seg_type.split()
                connector_token_id = token_id
                for wi, w in enumerate(words):
                    dep = "root" if wi == 0 else ("case" if w in ("à", "de", "la") else "fixed")
                    tokens.append(make_token(token_id, w, "VERB" if wi == 0 else ("ADP" if w in ("à", "de") else "NOUN"),
                                              dep, 0, lemma=w.lower(),
                                              gcn_type="verbe" if wi == 0 else None,
                                              gcn_class=vc_type if wi == 0 else None))
                    text_parts.append(w)
                    token_id += 1

            elif seg_type == "utilise":
                tokens.append(make_token(token_id, "utilise", "VERB", "root", 0,
                                          lemma="utiliser", gcn_type="verbe", gcn_class="action"))
                text_parts.append("utilise")
                token_id += 1

            elif seg_type == "avoir":
                tokens.append(make_token(token_id, "avoir", "AUX", "aux", 0, lemma="avoir"))
                text_parts.append("avoir")
                token_id += 1

            else:
                # Fallback : traiter comme mot simple
                tokens.append(make_token(token_id, seg_type, "X", "dep", 0, lemma=seg_type.lower()))
                text_parts.append(seg_type)
                token_id += 1

        # Fixer dep_head des VERB vers root
        for tok in tokens:
            if tok["pos"] == "VERB" and tok["dep_rel"] == "advcl":
                # Chercher le root
                for t2 in tokens:
                    if t2["dep_rel"] == "root":
                        tok["dep_head"] = t2["id"]
                        break

        # Construire le texte
        text = " ".join(text_parts)
        # Nettoyer la ponctuation
        text = text.replace(" ,", ",").replace(" .", ".").replace("  ", " ")

        # Construire les nœuds CIR
        nodes = []
        edges = []

        # n001 : cause/source (noun1)
        n001_span = (NP_subj_tokens_start or 1, NP_subj_tokens_end or 1)
        nodes.append({
            "id": n001_id,
            "type": n1_type,
            "label": f"{noun1}",
            "token_span": list(n001_span),
            "origin": "explicit",
            "scope": "specific",
            "temporal_index": 0,
            "modifiers": [],
            "attributes": {"entity": noun1},
        })

        # n002 : effet/cible (noun2)
        n002_span_start = NP_obj_tokens_start or NP_subj2_tokens_start or 1
        n002_span_end = NP_obj_tokens_end or NP_subj2_tokens_end or n002_span_start
        nodes.append({
            "id": n002_id,
            "type": n2_type,
            "label": f"{noun2}",
            "token_span": [n002_span_start, n002_span_end],
            "origin": "explicit",
            "scope": "specific",
            "temporal_index": 1,
            "modifiers": [],
            "attributes": {"entity": noun2},
        })

        # n003 (optionnel) : verbe action comme nœud si le token est localisé
        if vc_type == "action" and VERB_c_token_id is not None and self.rng.random() > 0.6:
            nodes.append({
                "id": "n003",
                "type": "action",
                "label": f"{verb_c}({noun1})",
                "token_span": [VERB_c_token_id, VERB_c_token_id],
                "origin": "explicit",
                "scope": "specific",
                "temporal_index": 0,
                "modifiers": [],
                "attributes": {"agent": noun1, "patient": noun2},
            })
            # Edge de n003 -> n002
            edge_list.append(("n003", n002_id, relation, 1.0, False, connector_token_id))
        else:
            # Edge de n001 -> n002
            edge_list.append((n001_id, n002_id, relation, 1.0, False, connector_token_id))

        # Construire les edges CIR
        for src, tgt, rel, conf, neg, marker in edge_list:
            edges.append({
                "source": src,
                "target": tgt,
                "relation": rel,
                "attributes": {
                    "confidence": conf,
                    "explicit": marker is not None,
                    "negated": neg,
                    "marker_token": marker,
                },
            })

        return {
            "id": f"s{self.sentence_counter:04d}",
            "text": text,
            "causal_pattern": relation,
            "tokens": tokens,
            "cir": {
                "nodes": nodes,
                "edges": edges,
            },
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

NODE_TYPES = {"etat", "action", "transition", "processus", "condition", "entite", "etat_systemique"}
RELATION_TYPES = {"cause", "enable", "prevent", "condition", "concession", "sequence",
                  "motivation", "filter", "opposition", "data_dependency", "control_dependency"}


def validate_sentence(sent: dict) -> list[str]:
    """Valide une phrase. Retourne la liste des erreurs (vide = OK)."""
    errors = []
    cir = sent.get("cir", {})
    nodes = cir.get("nodes", [])
    edges = cir.get("edges", [])
    tokens = sent.get("tokens", [])

    if not nodes:
        errors.append("pas de nœuds")
    if not edges:
        errors.append("pas d'arêtes")
    if not tokens:
        errors.append("pas de tokens")

    node_ids = {n["id"] for n in nodes}
    for node in nodes:
        if node.get("type") not in NODE_TYPES:
            errors.append(f"nœud {node['id']}: type inconnu '{node.get('type')}'")
        span = node.get("token_span", [])
        if len(span) != 2:
            errors.append(f"nœud {node['id']}: token_span invalide {span}")
        elif span[0] > span[1]:
            errors.append(f"nœud {node['id']}: token_span inversé {span}")
        elif span[0] < 1 or span[1] > len(tokens):
            errors.append(f"nœud {node['id']}: token_span hors bornes {span}")

    for edge in edges:
        if edge.get("relation") not in RELATION_TYPES:
            errors.append(f"arête: relation inconnue '{edge.get('relation')}'")
        if edge.get("source") not in node_ids:
            errors.append(f"arête: source inexistante '{edge.get('source')}'")
        if edge.get("target") not in node_ids:
            errors.append(f"arête: target inexistante '{edge.get('target')}'")

    return errors


# ---------------------------------------------------------------------------
# Split train/val/test
# ---------------------------------------------------------------------------

def split_dataset(sentences: list[dict], train_ratio=0.7, val_ratio=0.15, seed=42) -> tuple:
    """Split en train/val/test."""
    rng = random.Random(seed)
    indices = list(range(len(sentences)))
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = sorted(indices[:n_train])
    val_idx = sorted(indices[n_train:n_train + n_val])
    test_idx = sorted(indices[n_train + n_val:])

    train = [sentences[i] for i in train_idx]
    val = [sentences[i] for i in val_idx]
    test = [sentences[i] for i in test_idx]

    return train, val, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    seed = 42
    gen = SentenceGenerator(seed=seed)
    sentences = []

    # Générer ~90 phrases par relation (11 relations × 90 = 990)
    target_per_relation = 90
    relation_counts = {}

    for relation in RELATION_TEMPLATES:
        for i in range(target_per_relation):
            sent = gen.generate_sentence(relation)
            sentences.append(sent)
            relation_counts[relation] = relation_counts.get(relation, 0) + 1

    # Validation
    all_errors = []
    for sent in sentences:
        errors = validate_sentence(sent)
        if errors:
            all_errors.append((sent["id"], errors))

    if all_errors:
        print(f"ERREURS DE VALIDATION ({len(all_errors)} phrases) :")
        for sid, errs in all_errors[:10]:
            print(f"  {sid}: {errs}")
        if len(all_errors) > 10:
            print(f"  ... et {len(all_errors) - 10} autres")
        sys.exit(1)

    # Statistiques
    print(f"Dataset généré : {len(sentences)} phrases")
    print(f"\nDistribution des relations :")
    for rel, count in sorted(relation_counts.items()):
        print(f"  {rel:25s}: {count}")

    # Compter les types de nœuds
    node_type_counts = {}
    for sent in sentences:
        for node in sent["cir"]["nodes"]:
            nt = node["type"]
            node_type_counts[nt] = node_type_counts.get(nt, 0) + 1
    print(f"\nDistribution des node types :")
    for nt, count in sorted(node_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {nt:25s}: {count} ({count/sum(node_type_counts.values())*100:.1f}%)")

    total_nodes = sum(node_type_counts.values())
    total_edges = sum(len(s["cir"]["edges"]) for s in sentences)
    print(f"\nTotal : {total_nodes} nœuds, {total_edges} arêtes")

    # Écriture
    output_dir = Path("gcn-datasets/corpus")
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = {
        "schema_version": "1.0",
        "description": "Dataset d'entraînement GCN-Core généré automatiquement. "
                       "11 relations causales × 90 variations = ~990 phrases françaises.",
        "document": {
            "id": "doc-generated-001",
            "lang": "fr",
            "sentences": sentences,
        },
    }

    output_path = output_dir / "generated_1000.json"
    output_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nÉcrit : {output_path}")

    # Split
    train, val, test = split_dataset(sentences, seed=seed)
    print(f"\nSplit : train={len(train)}, val={len(val)}, test={len(test)}")

    splits_dir = Path("gcn-datasets/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        split_doc = {
            "schema_version": "1.0",
            "document": {
                "id": f"doc-{name}",
                "lang": "fr",
                "sentences": split,
            },
        }
        split_path = splits_dir / f"{name}.json"
        split_path.write_text(json.dumps(split_doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Écrit : {split_path} ({len(split)} phrases)")


if __name__ == "__main__":
    main()
