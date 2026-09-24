# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: MIT
"""Annotateur auto v2 — FR/EN/ES/DE, sans spaCy (tokenizer regex + typage lexical).

Produit du SILVER : CIR 2 noeuds / 1 arete avec spans ancres sur tokens regex,
direction causale explicite (marqueurs avant/arriere), confiance calibree.
Le GOLD manuel (219 phrases) sert a mesurer sa qualite par relation.

Usage :
    python balanced_auto.py balanced_fr.jsonl out_fr.jsonl
    python balanced_auto.py --all /tmp/opencode/annot /tmp/opencode/annot/silver
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

TOK = re.compile(r"[A-Za-zÀ-ÿ0-9]+(?:'[a-zàâäéèêëîïôöùûüçñ]+)?|[^\s\w]", re.UNICODE)

# connecteur -> (relation, direction) ; direction = sens de l'arete
#   "fwd" : clause_avant -> clause_apres   (ex : X donc Y, X entraîne Y)
#   "bwd" : clause_apres -> clause_avant   (ex : X parce que Y, X because Y)
# Marqueurs multi-mots d'abord (tri par longueur décroissante à la compilation).
CONN: dict[str, list[tuple[str, str, str]]] = {
 "fr": [
  ("parce que", "cause", "bwd"), ("parce qu", "cause", "bwd"), ("puisque", "cause", "bwd"),
  ("c'est pourquoi", "cause", "fwd"), ("par conséquent", "cause", "fwd"),
  ("en raison de", "cause", "bwd"), ("à cause de", "cause", "bwd"), ("grâce à", "enable", "bwd"),
  ("du fait que", "cause", "bwd"), ("étant donné", "cause", "bwd"), ("vu que", "cause", "bwd"),
  ("entraîne", "cause", "fwd"), ("provoque", "cause", "fwd"), ("engendre", "cause", "fwd"),
  ("conduit à", "cause", "fwd"), ("mène à", "cause", "fwd"), ("résulte", "cause", "fwd"),
  ("cause ", "cause", "fwd"),
  ("donc", "cause", "fwd"), ("ainsi", "cause", "fwd"), ("car ", "cause", "bwd"),
  ("permet", "enable", "fwd"), ("rend possible", "enable", "fwd"), ("facilite", "enable", "fwd"),
  ("favorise", "enable", "fwd"), ("contribue à", "enable", "fwd"), ("autorise", "control_dependency", "fwd"),
  ("empêche", "prevent", "fwd"), ("prévient", "prevent", "fwd"), ("évite", "prevent", "fwd"),
  ("bloque", "prevent", "fwd"), ("interdit", "prevent", "fwd"), ("inhibe", "prevent", "fwd"),
  ("protège", "prevent", "fwd"), ("lutte contre", "prevent", "fwd"), ("sans ", "prevent", "fwd"),
  ("contre ", "prevent", "fwd"),
  (" à condition que", "condition", "bwd"), (" en cas de", "condition", "bwd"),
  ("lorsque", "condition", "bwd"), ("quand ", "condition", "bwd"), (" si ", "condition", "bwd"),
  ("dès que", "condition", "bwd"), ("devant ", "condition", "bwd"), ("dans le cas où", "condition", "bwd"),
  ("bien que", "concession", "bwd"), ("même si", "concession", "bwd"), ("quoique", "concession", "bwd"),
  ("malgré", "concession", "bwd"), ("en dépit de", "concession", "bwd"),
  ("pourtant", "concession", "bwd"), ("cependant", "concession", "bwd"),
  ("néanmoins", "concession", "bwd"), ("toutefois", "concession", "bwd"),
  ("d'abord", "sequence", "fwd"), ("finit par", "sequence", "fwd"), ("finissent par", "sequence", "fwd"), ("puis ", "sequence", "fwd"), ("ensuite", "sequence", "fwd"),
  ("enfin", "sequence", "fwd"), ("après ", "sequence", "fwd"), ("avant ", "sequence", "bwd"),
  ("préalablement", "sequence", "bwd"), ("tout d'abord", "sequence", "fwd"),
  ("afin de", "motivation", "fwd"), ("pour que", "motivation", "bwd"),
  ("dans le but de", "motivation", "fwd"), ("en vue de", "motivation", "fwd"),
  ("vise à", "motivation", "fwd"), ("a pour objectif", "motivation", "fwd"),
  ("sauf", "filter", "fwd"), ("excepté", "filter", "fwd"), ("hormis", "filter", "fwd"),
  ("à l'exception", "filter", "fwd"), ("uniquement", "filter", "fwd"), ("seuls", "filter", "fwd"),
  ("seules", "filter", "fwd"), ("réservé", "filter", "fwd"),
  ("mais ", "opposition", "bwd"), ("en revanche", "opposition", "bwd"),
  ("au contraire", "opposition", "bwd"), ("tandis que", "opposition", "bwd"),
  ("alors que", "opposition", "bwd"), ("au lieu de", "opposition", "bwd"),
  ("contrairement à", "opposition", "bwd"),
  ("dépend de", "data_dependency", "fwd"), ("repose sur", "data_dependency", "fwd"),
  ("nécessite", "data_dependency", "fwd"), ("nécessaire", "data_dependency", "fwd"),
  ("exige", "data_dependency", "fwd"), ("requiert", "data_dependency", "fwd"),
  ("basé sur", "data_dependency", "bwd"), ("basée sur", "data_dependency", "bwd"),
  ("fondé sur", "data_dependency", "bwd"),
  ("déclenche", "control_dependency", "fwd"), ("active ", "control_dependency", "fwd"),
  ("pilote", "control_dependency", "fwd"), ("ordonne", "control_dependency", "fwd"),
  ("gère", "control_dependency", "fwd"), ("orchestre", "control_dependency", "fwd"),
  ("coordonne", "control_dependency", "fwd"), ("contrôle", "control_dependency", "fwd"),
  ("détermine", "control_dependency", "fwd"), ("appelle", "cause", "fwd"),
 ],
 "en": [
  ("because", "cause", "bwd"), ("since ", "cause", "bwd"), ("therefore", "cause", "fwd"),
  ("thus", "cause", "fwd"), ("hence", "cause", "fwd"), ("consequently", "cause", "fwd"),
  ("as a result", "cause", "fwd"), ("due to", "cause", "bwd"), ("thanks to", "enable", "bwd"),
  ("caused by", "cause", "bwd"), ("causes", "cause", "fwd"), ("leads to", "cause", "fwd"),
  ("results in", "cause", "fwd"), ("triggers", "cause", "fwd"), ("produces", "cause", "fwd"),
  ("enables", "enable", "fwd"), ("allows", "enable", "fwd"), ("facilitates", "enable", "fwd"),
  ("makes possible", "enable", "fwd"), ("contributes to", "enable", "fwd"),
  ("prevents", "prevent", "fwd"), ("avoids", "prevent", "fwd"), ("blocks", "prevent", "fwd"),
  ("inhibits", "prevent", "fwd"), ("stops ", "prevent", "fwd"), ("protects", "prevent", "fwd"),
  ("against ", "prevent", "fwd"), ("without ", "prevent", "fwd"),
  ("provided that", "condition", "bwd"), ("as long as", "condition", "bwd"),
  (" if ", "condition", "bwd"), ("unless", "condition", "bwd"), ("when ", "condition", "bwd"),
  ("whenever", "condition", "bwd"), ("given that", "condition", "bwd"),
  ("in case", "condition", "bwd"),
  ("although", "concession", "bwd"), ("even though", "concession", "bwd"),
  ("despite", "concession", "bwd"), ("however", "concession", "bwd"),
  ("nevertheless", "concession", "bwd"), ("nonetheless", "concession", "bwd"),
  ("albeit", "concession", "bwd"),
  ("first", "sequence", "fwd"), ("ends up", "sequence", "fwd"), ("end up", "sequence", "fwd"), ("then ", "sequence", "fwd"), ("next ", "sequence", "fwd"),
  ("finally", "sequence", "fwd"), ("after ", "sequence", "fwd"), ("before ", "sequence", "bwd"),
  ("subsequently", "sequence", "fwd"), ("prior to", "sequence", "bwd"), ("thereafter", "sequence", "fwd"),
  ("in order to", "motivation", "fwd"), ("so as to", "motivation", "fwd"),
  ("aims to", "motivation", "fwd"), ("seeks to", "motivation", "fwd"),
  ("intended to", "motivation", "fwd"), ("motivated by", "motivation", "bwd"),
  ("except", "filter", "fwd"), ("excluding", "filter", "fwd"), ("apart from", "filter", "fwd"),
  ("limited to", "filter", "fwd"), ("restricted to", "filter", "fwd"),
  ("only if", "filter", "fwd"), ("solely", "filter", "fwd"), ("exclusively", "filter", "fwd"),
  ("but ", "opposition", "bwd"), ("whereas", "opposition", "bwd"),
  ("on the contrary", "opposition", "bwd"), ("instead of", "opposition", "bwd"),
  ("in contrast", "opposition", "bwd"), ("contrary to", "opposition", "bwd"),
  ("depends on", "data_dependency", "fwd"), ("relies on", "data_dependency", "fwd"),
  ("requires", "data_dependency", "fwd"), ("needs ", "data_dependency", "fwd"),
  ("based on", "data_dependency", "bwd"),
  ("calls ", "control_dependency", "fwd"), ("invokes", "control_dependency", "fwd"),
  ("triggers", "control_dependency", "fwd"), ("controls", "control_dependency", "fwd"),
  ("orchestrates", "control_dependency", "fwd"), ("dispatches", "control_dependency", "fwd"),
  ("manages ", "control_dependency", "fwd"), ("activates", "control_dependency", "fwd"),
 ],
 "es": [
  ("porque", "cause", "bwd"), ("ya que", "cause", "bwd"), ("puesto que", "cause", "bwd"),
  ("debido a", "cause", "bwd"), ("por lo tanto", "cause", "fwd"), ("por tanto", "cause", "fwd"),
  ("por consiguiente", "cause", "fwd"), ("provoca", "cause", "fwd"), ("genera", "cause", "fwd"),
  ("conduce a", "cause", "fwd"), ("causa", "cause", "fwd"), ("determina", "cause", "fwd"),
  ("permite", "enable", "fwd"), ("facilita", "enable", "fwd"), ("gracias a", "enable", "bwd"),
  ("hace posible", "enable", "fwd"), ("contribuye a", "enable", "fwd"),
  ("impide", "prevent", "fwd"), ("evita", "prevent", "fwd"), ("bloquea", "prevent", "fwd"),
  ("previene", "prevent", "fwd"), ("prohíbe", "prevent", "fwd"), ("protege", "prevent", "fwd"),
  ("sin ", "prevent", "fwd"), ("contra ", "prevent", "fwd"),
  (" en caso de", "condition", "bwd"), ("cuando ", "condition", "bwd"),
  (" si ", "condition", "bwd"), ("siempre que", "condition", "bwd"),
  ("menos que", "condition", "bwd"), ("ante ", "condition", "bwd"),
  ("para que", "motivation", "bwd"), ("aunque", "concession", "bwd"),
  ("pesar de", "concession", "bwd"), ("sin embargo", "concession", "bwd"),
  ("no obstante", "concession", "bwd"), ("aun cuando", "concession", "bwd"),
  ("primero", "sequence", "fwd"), ("luego", "sequence", "fwd"), ("después", "sequence", "fwd"),
  ("entonces", "sequence", "fwd"), ("antes de", "sequence", "bwd"),
  ("finalmente", "sequence", "fwd"), ("previamente", "sequence", "bwd"),
  ("fin de", "motivation", "fwd"), ("con el objetivo", "motivation", "fwd"),
  ("con el fin", "motivation", "fwd"), ("motivado por", "motivation", "bwd"),
  ("busca ", "motivation", "fwd"),
  ("salvo", "filter", "fwd"), ("excepto", "filter", "fwd"), ("únicamente", "filter", "fwd"),
  ("solamente", "filter", "fwd"), ("reservado", "filter", "fwd"), ("limitado a", "filter", "fwd"),
  ("sólo ", "filter", "fwd"),
  ("pero ", "opposition", "bwd"), ("en cambio", "opposition", "bwd"),
  ("contrario", "opposition", "bwd"), ("mientras que", "opposition", "bwd"),
  ("lugar de", "opposition", "bwd"),
  ("depende de", "data_dependency", "fwd"), ("se basa en", "data_dependency", "bwd"),
  ("requiere", "data_dependency", "fwd"), ("necesita", "data_dependency", "fwd"),
  ("necesario", "data_dependency", "fwd"), ("basado en", "data_dependency", "bwd"),
  ("desencadena", "control_dependency", "fwd"), ("activa ", "control_dependency", "fwd"),
  ("controla", "control_dependency", "fwd"), ("ordena", "control_dependency", "fwd"),
  ("autoriza", "control_dependency", "fwd"), ("gestiona", "control_dependency", "fwd"),
  ("coordina", "control_dependency", "fwd"), ("llama", "cause", "fwd"),
 ],
 "de": [
  ("weil", "cause", "bwd"), ("denn", "cause", "bwd"), ("deshalb", "cause", "fwd"),
  ("daher", "cause", "fwd"), ("aufgrund", "cause", "bwd"), ("führt zu", "cause", "fwd"),
  ("verursacht", "cause", "fwd"), ("bewirkt", "cause", "fwd"), ("löst aus", "cause", "fwd"),
  ("ermöglicht", "enable", "fwd"), ("erleichtert", "enable", "fwd"),
  ("dank ", "enable", "bwd"), ("macht möglich", "enable", "fwd"), ("erlaubt", "enable", "fwd"),
  ("verhindert", "prevent", "fwd"), ("blockiert", "prevent", "fwd"),
  ("stoppt", "prevent", "fwd"), ("vermeidet", "prevent", "fwd"), ("verbietet", "prevent", "fwd"),
  ("schützt", "prevent", "fwd"), ("gegen ", "prevent", "fwd"), ("ohne ", "prevent", "fwd"),
  ("wenn ", "condition", "bwd"), ("falls ", "condition", "bwd"), ("sofern ", "condition", "bwd"),
  ("Bedingung", "condition", "bwd"), ("vorausgesetzt", "condition", "bwd"),
  ("obwohl", "concession", "bwd"), ("trotz", "concession", "bwd"), ("dennoch", "concession", "bwd"),
  ("jedoch", "concession", "bwd"), ("allerdings", "concession", "bwd"),
  ("zuerst", "sequence", "fwd"), ("dann ", "sequence", "fwd"), ("anschließend", "sequence", "fwd"),
  ("schließlich", "sequence", "fwd"), ("nach ", "sequence", "fwd"), ("vor ", "sequence", "bwd"),
  ("zuvor", "sequence", "bwd"), ("im Voraus", "sequence", "bwd"),
  ("um zu", "motivation", "fwd"), ("damit ", "motivation", "fwd"),
  ("mit dem Ziel", "motivation", "fwd"), ("strebt an", "motivation", "fwd"),
  ("außer", "filter", "fwd"), ("ausgenommen", "filter", "fwd"), ("nur wenn", "filter", "fwd"),
  ("einzig", "filter", "fwd"), ("vorbehalten", "filter", "fwd"),
  ("aber ", "opposition", "bwd"), ("hingegen", "opposition", "bwd"),
  ("Gegenteil", "opposition", "bwd"), ("während ", "opposition", "bwd"),
  ("stattdessen", "opposition", "bwd"),
  ("hängt ab von", "data_dependency", "fwd"), ("abhängig von", "data_dependency", "fwd"),
  ("beruht auf", "data_dependency", "bwd"), ("benötigt", "data_dependency", "fwd"),
  ("erfordert", "data_dependency", "fwd"), ("basiert auf", "data_dependency", "bwd"),
  ("löst aus", "control_dependency", "fwd"), ("steuert", "control_dependency", "fwd"),
  ("orchestriert", "control_dependency", "fwd"), ("aktiviert", "control_dependency", "fwd"),
  ("befiehlt", "control_dependency", "fwd"), ("reguliert", "control_dependency", "fwd"),
 ],
}

# Confiance par marqueur fort (1.0) vs standard (0.9) vs discours faible (0.8)
CONF1 = {"parce que", "because", "weil", "porque", "c'est pourquoi", "par conséquent",
         "therefore", "deshalb", "por lo tanto", "bien que", "although", "obwohl", "aunque",
         "si ", " if ", "wenn ", "malgré", "despite", "trotz", "empêche", "prevents",
         "verhindert", "impide", "entraîne", "causes", "provoca", "verursacht"}
CONF08 = {"donc", "ainsi", "car ", "puis ", "then ", "ensuite", "luego", "dann ",
          "cependant", "however", "jedoch", "enfin", "finalement", "finally",
          "mais ", "but ", "aber ", "alors que", "tandis que", "si ", " if "}

# Lexiques de typage (minuscules, lemmes approximatifs)
STATE_V = {"fr": {"être", "avoir", "rester", "demeurer", "sembler", "paraître", "devenir", "souffrir"},
           "en": {"be", "remain", "stay", "seem", "appear", "suffer"},
           "es": {"ser", "estar", "quedar", "permanecer", "parecer", "sufrir"},
           "de": {"sein", "bleiben", "scheinen", "leiden"}}
TRANS_V = {"fr": {"devenir", "changer", "évoluer", "transformer", "modifier", "passer"},
           "en": {"become", "change", "evolve", "transform", "shift", "turn"},
           "es": {"convertir", "transformar", "cambiar", "evolucionar", "pasar"},
           "de": {"werden", "ändern", "wandeln", "entwickeln", "übergehen"}}
ACT_V = {"fr": {"faire", "créer", "produire", "construire", "prendre", "réaliser",
                "effectuer", "appliquer", "lancer", "décider", "choisir", "écrire"},
         "en": {"make", "create", "build", "produce", "apply", "launch", "decide",
                "execute", "run", "write", "take"},
         "es": {"hacer", "crear", "producir", "construir", "tomar", "realizar",
                "aplicar", "lanzar", "decidir", "elegir", "escribir"},
         "de": {"machen", "erstellen", "produzieren", "bauen", "nehmen", "entscheiden",
                "starten", "schreiben", "wählen"}}
SYS_N = {"fr": {"loi", "règle", "norme", "politique", "système", "protocole", "procédure",
                "mécanisme", "régulation", "standard", "principe"},
         "en": {"law", "rule", "norm", "policy", "system", "protocol", "regulation",
                "mechanism", "standard", "principle"},
         "es": {"ley", "regla", "norma", "política", "sistema", "protocolo", "mecanismo",
                "principio"},
         "de": {"gesetz", "regel", "norm", "politik", "system", "protokoll", "mechanismus",
                "prinzip"}}
COND_M = {"fr": {"si ", "quand ", "lorsque"}, "en": {" if ", "when ", "unless"},
          "es": {" si ", "cuando "}, "de": {"wenn ", "falls "}}
STOP = {"fr": {"le", "la", "les", "de", "du", "des", "une", "un", "et", "est", "dans",
               "pour", "avec", "sur", "qui", "que", "par", "pas", "plus", "tout", "cette",
               "ces", "entre", "mais", "donc", "alors", "car", "il", "elle", "ils", "nous",
               "vous", "son", "sa", "ses", "leur", "leurs", "aux", "au", "en", "au", "ce",
               "cette", "cet"},
        "en": {"the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
               "have", "has", "had", "will", "would", "can", "could", "not", "but", "are",
               "was", "were", "been", "has", "have", "had", "will", "would", "should",
               "they", "their", "them", "he", "she", "his", "her", "its", "our", "your",
               "which", "who", "what", "when", "where", "there", "here", "all", "any"},
        "es": {"el", "la", "los", "las", "de", "del", "una", "uno", "y", "es", "en",
               "para", "con", "por", "que", "se", "no", "los", "las", "este", "esta",
               "estos", "estas", "como", "más", "pero", "sus", "les", "al", "su"},
        "de": {"der", "die", "das", "den", "dem", "des", "und", "ist", "sind", "in",
               "für", "mit", "von", "zu", "dass", "die", "der", "das", "nicht", "eine",
               "einer", "einem", "einen", "als", "auch", "sich", "sie", "er", "es", "wir"}}

COMP = {l: sorted([(c, r, d) for c, r, d in lst], key=lambda x: -len(x[0]))
        for l, lst in CONN.items()}
PAT = {l: [(c, r, d, re.compile(r"(?<!\w)" + re.escape(c) + r"(?!\w)", re.IGNORECASE))
           for c, r, d in lst] for l, lst in COMP.items()}


def tokenize(text: str) -> list[str]:
    return TOK.findall(text)


def content_words(clause: str, lang: str) -> list[str]:
    stop = STOP.get(lang, set())
    return [w for w in re.findall(r"[A-Za-zÀ-ÿ]+", clause.lower())
            if len(w) >= 4 and w not in stop]


AUX = {
 "fr": {"est", "sont", "était", "étaient", "sera", "seront", "ont", "a", "ont",
        "peut", "peuvent", "doit", "doivent", "faut", "va", "vont", "été"},
 "en": {"is", "are", "was", "were", "be", "been", "has", "have", "had", "will",
        "would", "can", "could", "may", "might", "must", "shall", "do", "does", "did"},
 "es": {"es", "son", "era", "eran", "será", "han", "ha", "tiene", "tienen",
        "puede", "pueden", "debe", "hay", "está", "están", "fue", "fueron"},
 "de": {"ist", "sind", "war", "waren", "wird", "werden", "hat", "haben",
        "kann", "können", "muss", "soll", "darf"},
}
VSUF = {
 "fr": ("er", "ir", "re", "é", "ée", "és", "ées", "ant", "ent", "ait", "aient",
        "era", "ont", "ez", "ons", "ît"),
 "en": ("ed", "ing",),
 "es": ("ar", "er", "ir", "ando", "iendo", "ado", "ido", "an", "en", "amos"),
 "de": ("en", "ern", "eln", "ieren",),
}


def has_verb(clause: str, lang: str) -> bool:
    words = re.findall(r"[A-Za-zÀ-ÿ]+", clause.lower())
    aux = AUX.get(lang, set())
    if any(w in aux for w in words):
        return True
    lex = (STATE_V.get(lang, set()) | TRANS_V.get(lang, set()) | ACT_V.get(lang, set()))
    if any(w in lex for w in words):
        return True
    for w in words:
        if len(w) >= 6 and w.endswith(VSUF.get(lang, ())):
            return True
    return False


def classify(clause: str, lang: str) -> str:
    low = clause.lower()
    if any(m in low for m in COND_M.get(lang, ())):
        return "condition"
    words = set(re.findall(r"[A-Za-zÀ-ÿ]+", low))
    if not words:
        return "entite"
    if words & SYS_N.get(lang, set()):
        return "etat_systemique"
    if not has_verb(clause, lang):
        return "entite"
    if words & TRANS_V.get(lang, set()):
        return "transition"
    if words & STATE_V.get(lang, set()):
        return "etat"
    if words & ACT_V.get(lang, set()):
        return "action"
    return "processus"


# Priorité des marqueurs : P0 condition structurelle > P1 verbes/lexique causal >
# P2 discours (adverbes). On essaie tous les marqueurs présents, par priorité.
P0 = {" si ", " à condition que", " en cas de", "lorsque", "quand ", "dès que",
      " if ", "provided that", "as long as", "unless", "when ", "whenever",
      " si ", "cuando ", "siempre que", "wenn ", "falls ", "sofern "}
P2 = {"cependant", "pourtant", "néanmoins", "toutefois", "donc", "ainsi",
      "car ", "puis ", "ensuite", "enfin", "mais ", "alors que", "tandis que",
      "however", "nevertheless", "nonetheless", "therefore", "thus", "hence",
      "then ", "next ", "finally", "but ", "(Le système)", "dennoch", "jedoch",
      "allerdings", "dann ", "daher", "deshalb", "aber ", "sin embargo",
      "no obstante", "luego", "entonces", "después", "pero ", "en cambio",
      "appelle", "llama"}
LEAD_DISCOURSE = {"donc", "ainsi", "cependant", "pourtant", "néanmoins",
                  "toutefois", "however", "nevertheless", "also", "dennoch",
                  "sin", "embargo", "en", "revanche", "par", "ailleurs"}
STOP_EXTRA = {"notamment", "toutefois", "cependant", "néanmoins", "pourtant",
              "généralement", "également", "souvent", "toujours", "jamais",
              "however", "also", "often", "always", "never", "generally",
              "notamment"}
for _s in STOP_EXTRA:
    for _l in STOP:
        STOP[_l].add(_s)

# Verbes de transformation "A -> B" (règle séquence mono-proposition)
TRANS_SPLIT = [
 ("fr", re.compile(r"(.*?)\best (?:transformé|converti)\s+en\b\s*(.*)", re.I)),
 ("fr", re.compile(r"(.*?)\bdevient\b\s*(.*)", re.I)),
 ("en", re.compile(r"(.*?)\bis (?:transformed|converted)\s+into\b\s*(.*)", re.I)),
 ("en", re.compile(r"(.*?)\bbecomes?\b\s*(.*)", re.I)),
 ("es", re.compile(r"(.*?)\bse (?:convierte|transforma)\s+en\b\s*(.*)", re.I)),
 ("de", re.compile(r"(.*?)\bwird\s+zu\b\s*(.*)", re.I)),
]


def _prio(conn: str) -> int:
    c = conn.lower().strip()
    if any(p.strip() == c or p in f" {c} " for p in P0):
        return 0
    if c in P2 or f" {c} " in {f" {p} " for p in P2} or c in P2:
        return 2
    return 1


def find_markers(text: str, lang: str):
    out = []
    for conn, rel, direction, pat in PAT.get(lang, []):
        m = pat.search(text)
        if m:
            out.append((conn, rel, direction, m))
    out.sort(key=lambda x: (_prio(x[0]), -len(x[0])))
    return out


def find_marker(text: str, lang: str):
    ms = find_markers(text, lang)
    return ms[0] if ms else None


def _strip_lead(text: str) -> str:
    parts = text.split(None, 2)
    if len(parts) >= 2 and parts[0].strip(",;:").lower() in LEAD_DISCOURSE:
        rest = text.split(None, 1)[1] if len(text.split(None, 1)) > 1 else text
        rest = rest.lstrip(",;: ")
        if rest != text:
            return _strip_lead(rest)
    return text


def _try_split(text: str, lang: str, conn: str, rel: str, direction: str, m) -> dict | None:
    before, after = text[:m.start()].strip(" ,;:"), text[m.end():].strip(" ,;:")
    # PP causale en tête ("En raison de X, Y") : split à la première virgule
    if not before and "," in after:
        head, tail = after.split(",", 1)
        before, after = head.strip(), tail.strip()
        if direction == "bwd":
            direction = "fwd"  # "En raison de X, Y" : X -> Y
    if not before or not after:
        return None
    cb, ca = content_words(before, lang), content_words(after, lang)
    if len(cb) < 1 or len(ca) < 1:
        return None
    if len(cb) + len(ca) < 4:
        return None
    toks = tokenize(text)
    T = len(toks)
    # spans : tokens avant/après le marqueur (indices 1-based)
    conn_toks = tokenize(conn)
    # localiser le marqueur dans la séquence de tokens (insensible à la casse)
    low = [t.lower() for t in toks]
    first = conn_toks[0].lower()
    midx = next((i for i, t in enumerate(low) if t == first), None)
    if midx is None:
        cut = len(tokenize(before))
    else:
        cut = midx
    if direction == "fwd":
        src_txt, dst_txt, s_span, d_span = before, after, [1, max(1, cut)], [min(T, cut + 2), T]
        t_src, t_dst = 0, 1
    else:
        src_txt, dst_txt, s_span, d_span = after, before, [min(T, cut + 2), T], [1, max(1, cut)]
        t_src, t_dst = 0, 1
    type_src, type_dst = classify(src_txt, lang), classify(dst_txt, lang)
    cw_s = content_words(src_txt, lang) or ["x"]
    cw_d = content_words(dst_txt, lang) or ["y"]
    conf = 1.0 if conn.lower().strip() in CONF1 else (0.8 if conn.lower().strip() in CONF08 else 0.9)
    marker_id = cut + 1 if cut < T else T
    return {
        "causal_pattern": rel,
        "nodes": [
            {"id": "n001", "type": type_src, "label": cw_s[0], "token_span": s_span,
             "origin": "explicit", "scope": "specific", "temporal_index": t_src,
             "modifiers": [], "attributes": {}},
            {"id": "n002", "type": type_dst, "label": cw_d[0], "token_span": d_span,
             "origin": "explicit", "scope": "specific", "temporal_index": t_dst,
             "modifiers": [], "attributes": {}},
        ],
        "edges": [
            {"source": "n001", "target": "n002", "relation": rel,
             "attributes": {"confidence": conf, "explicit": True, "negated": False,
                            "marker_token": marker_id}},
        ],
        "_silver": {"connector": conn, "direction": direction, "method": "auto-v2"},
    }


def _trans_split(text: str, lang: str) -> dict | None:
    """Transformation mono-proposition 'A -> B' (sequence)."""
    for lg, pat in TRANS_SPLIT:
        if lg != lang:
            continue
        m = pat.search(text)
        if not m:
            continue
        before, after = m.group(1).strip(" ,;:"), m.group(2).strip(" ,;:")
        cb, ca = content_words(before, lang), content_words(after, lang)
        if len(cb) < 1 or len(ca) < 1 or len(cb) + len(ca) < 4:
            continue
        toks = tokenize(text)
        T = len(toks)
        cut = len(tokenize(before))
        s_span, d_span = [1, max(1, cut)], [min(T, cut + 2), T]
        return {
            "causal_pattern": "sequence",
            "nodes": [
                {"id": "n001", "type": "entite", "label": cb[0], "token_span": s_span,
                 "origin": "explicit", "scope": "specific", "temporal_index": 0,
                 "modifiers": [], "attributes": {}},
                {"id": "n002", "type": "transition", "label": ca[0], "token_span": d_span,
                 "origin": "explicit", "scope": "specific", "temporal_index": 1,
                 "modifiers": [], "attributes": {}},
            ],
            "edges": [
                {"source": "n001", "target": "n002", "relation": "sequence",
                 "attributes": {"confidence": 0.9, "explicit": True, "negated": False,
                                "marker_token": min(T, cut + 1)}},
            ],
            "_silver": {"connector": "transformation", "direction": "fwd",
                        "method": "auto-v2"},
        }
    return None


def annotate(text: str, lang: str) -> dict | None:
    """Retourne CIR silver ou None (structure insuffisante)."""
    text = _strip_lead(text.strip())
    for conn, rel, direction, m in find_markers(text, lang):
        r = _try_split(text, lang, conn, rel, direction, m)
        if r:
            return r
    return _trans_split(text, lang)


def run_file(inp: Path, outp: Path) -> dict:
    from collections import Counter
    rel_c, type_c, n_in, n_out = Counter(), Counter(), 0, 0
    with open(inp, encoding="utf-8") as f, open(outp, "w", encoding="utf-8") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            n_in += 1
            r = annotate(o["text"], o.get("lang", "fr"))
            if not r:
                continue
            n_out += 1
            rel_c[r["causal_pattern"]] += 1
            for nd in r["nodes"]:
                type_c[nd["type"]] += 1
            toks = tokenize(o["text"])
            g.write(json.dumps({"text": o["text"], "lang": o.get("lang"),
                                "source": o.get("source", ""), "cir": r,
                                "_methode": "silver-auto-v2",
                                "tokens_provisional": toks}, ensure_ascii=False) + "\n")
    return {"in": n_in, "out": n_out, "relations": dict(rel_c), "types": dict(type_c)}


def _run_lang(dstdir: Path, fin: Path, fout: Path, model: str, tag: str) -> dict:
    # substitue la langue du modèle (ex : code -> en) via fichier tmp dédié
    tmp = dstdir / f"_tmp_{tag}.jsonl"
    with open(fin, encoding="utf-8") as f, open(tmp, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            o["lang"] = model
            g.write(json.dumps(o, ensure_ascii=False) + "\n")
    rep = run_file(tmp, fout)
    tmp.unlink(missing_ok=True)
    return rep


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--all":
        srcdir, dstdir = Path(sys.argv[2]), Path(sys.argv[3])
        dstdir.mkdir(parents=True, exist_ok=True)
        _langs = (("fr", "fr"), ("en", "en"), ("es", "es"),
                  ("de", "de"), ("code", "en"))
        for lang, model in _langs:
            rep = _run_lang(dstdir, srcdir / f"balanced_{lang}.jsonl",
                            dstdir / f"silver_{lang}.jsonl", model, lang)
            print(f"{lang}: {rep['out']}/{rep['in']} annotées")
        print("  rel:", {k: rep["relations"].get(k, 0) for k in
                         ["cause", "enable", "prevent", "condition", "concession",
                          "sequence", "motivation", "filter", "opposition",
                          "data_dependency", "control_dependency"]})
        print("  types:", rep["types"])
    elif len(sys.argv) == 3:
        print(run_file(Path(sys.argv[1]), Path(sys.argv[2])))
    else:
        print("usage: balanced_auto.py [--all SRCDIR DSTDIR | IN.jsonl OUT.jsonl]")
