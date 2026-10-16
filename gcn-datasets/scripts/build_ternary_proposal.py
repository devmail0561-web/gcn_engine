# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Génère la proposition silver ternaire (20 phrases, poids 0.0).

Données codées en dur : (texte, clauses, nœuds, arêtes). Les tokens sont
une segmentation whitespace (formes retrouvées en ordre — même contrat que
``gold_spans.reconstruire_offsets``) et les ``token_span`` sont CALCULÉS
depuis les spans caractères (jamais indexés à la main).

Statut : SILVER non validé — exclu du train (poids 0.0, hors DATA/train).
Promotion en gold : validation humaine + gate IC95 (``GATES:15``).
"""
from __future__ import annotations

import argparse
import hashlib
import json

# (texte, [(clause, type_noeud)], [(src, dst, relation, third)]) — indices clauses.
PROPOSITIONS: list = [
    ("Si le pare-feu est actif, l'attaque échoue.",
     [("le pare-feu est actif", "condition"), ("l'attaque échoue", "evenement")],
     [(0, 1, "conditional_cause", {"role": "Condition", "clause": 0})]),
    ("Le traitement guérit si le diagnostic est précoce.",
     [("Le traitement guérit", "processus"), ("le diagnostic est précoce", "condition")],
     [(0, 1, "conditional_cause", {"role": "Condition", "clause": 1})]),
    ("Sans sauvegarde, la panne efface les données.",
     [("la panne efface les données", "evenement"), ("Sans sauvegarde", "condition")],
     [(0, 1, "conditional_prevent", {"role": "Condition", "clause": 1})]),
    ("Le vaccin protège, à condition d'être conservé au froid.",
     [("Le vaccin protège", "processus"), ("conservé au froid", "condition")],
     [(0, 1, "conditional_cause", {"role": "Condition", "clause": 1})]),
    ("La pluie provoque des inondations via la saturation des sols.",
     [("La pluie", "La pluie", "evenement"), ("la saturation des sols", "sols saturés", "etat_local"), ("inondations", "inondations", "evenement")],
     [(0, 2, "mediated_cause", {"role": "Mediator", "clause": 1})]),
    ("Le stress cause l'insomnie par l'hypervigilance.",
     [("Le stress", "Le stress", "etat_local"), ("l'hypervigilance", "hypervigilance", "etat_local"), ("l'insomnie", "insomnie", "etat_local")],
     [(0, 2, "mediated_cause", {"role": "Mediator", "clause": 1})]),
    ("Le couvre-feu a stoppé l'émeute grâce à l'intervention policière.",
     [("Le couvre-feu", "Le couvre-feu", "evenement"), ("l'intervention policière", "intervention policière", "evenement"), ("l'émeute", "émeute stoppée", "etat_local")],
     [(0, 2, "mediated_prevent", {"role": "Mediator", "clause": 1})]),
    ("L'antibiotique élimine l'infection en détruisant la paroi bactérienne.",
     [("L'antibiotique", "L'antibiotique", "entite"), ("la paroi bactérienne", "paroi détruite", "etat_local"), ("l'infection", "infection éliminée", "etat_local")],
     [(0, 2, "mediated_cause", {"role": "Mediator", "clause": 1})]),
    ("La sécheresse et la canicule ont provoqué l'incendie.",
     [("La sécheresse", "La sécheresse", "etat_global"), ("la canicule", "la canicule", "etat_global"), ("l'incendie", "incendie", "evenement")],
     [([0, 1], 2, "joint_cause", None)]),
    ("Le masque et la distance réduisent la contagion.",
     [("Le masque", "Le masque", "entite"), ("la distance", "la distance", "concept"), ("la contagion", "contagion réduite", "etat_local")],
     [([0, 1], 2, "joint_prevent", None)]),
    ("La formation et l'outillage augmentent la productivité.",
     [("La formation", "processus"), ("l'outillage", "entite"), ("productivité", "concept")],
     [([0, 1], 2, "joint_cause", None)]),
    ("Le mot de passe et la biométrie bloquent l'intrusion.",
     [("Le mot de passe", "Le mot de passe", "entite"), ("la biométrie", "la biométrie", "entite"), ("l'intrusion", "intrusion bloquée", "evenement")],
     [([0, 1], 2, "joint_prevent", None)]),
    ("Comme le tabac cause le cancer, l'amiante cause le mésothéliome.",
     [("le tabac cause le cancer", "tabac→cancer", "concept"), ("l'amiante cause le mésothéliome", "amiante→mésothéliome", "concept")],
     [(0, 1, "analogy", None)]),
    ("De même que la digue retient l'eau, le pare-feu retient les attaques.",
     [("la digue retient l'eau", "digue→eau", "concept"), ("le pare-feu retient les attaques", "pare-feu→attaques", "concept")],
     [(0, 1, "analogy", None)]),
    ("Tel le vaccin contre la grippe, ce patch protège le système.",
     [("le vaccin contre la grippe", "vaccin→grippe", "concept"), ("ce patch protège le système", "patch→système", "concept")],
     [(0, 1, "analogy", None)]),
    ("Comme la rouille ronge le fer, l'inflation ronge l'épargne.",
     [("la rouille ronge le fer", "rouille→fer", "concept"), ("l'inflation ronge l'épargne", "inflation→épargne", "concept")],
     [(0, 1, "analogy", None)]),
    ("Sans la ceinture, le choc aurait été mortel.",
     [("Sans la ceinture", "Sans la ceinture", "condition"), ("le choc aurait été mortel", "choc mortel hypothétique", "evenement")],
     [(0, 1, "counterfactual", None)]),
    ("Si l'alerte avait sonné, l'évacuation aurait réussi.",
     [("l'alerte avait sonné", "alerte non sonnée (irréel)", "evenement"), ("l'évacuation aurait réussi", "évacuation ratée (irréel)", "evenement")],
     [(0, 1, "counterfactual", None)]),
    ("Avec un dépistage précoce, le patient aurait survécu.",
     [("Avec un dépistage précoce", "dépistage non réalisé (irréel)", "evenement"), ("le patient aurait survécu", "survie hypothétique", "evenement")],
     [(0, 1, "counterfactual", None)]),
    ("Sans la grève, les trains seraient arrivés à l'heure.",
     [("Sans la grève", "Sans la grève", "evenement"), ("arrivés à l'heure", "retard (irréel)", "evenement")],
     [(0, 1, "counterfactual", None)]),
]


def _tokens(text: str) -> list[dict]:
    toks = []
    pos = 0
    for k, forme in enumerate(text.split(), 1):
        i = text.find(forme, pos)
        assert i >= 0, forme
        toks.append({"id": k, "form": forme, "lemma": forme.lower(), "pos": "other",
                     "dep_rel": "other", "dep_head": -1, "morph": {}})
        pos = i + len(forme)
    return toks


def _span_tokens(text: str, toks: list[dict], clause: str) -> list[int]:
    s = text.find(clause)
    assert s >= 0, clause
    e = s + len(clause)
    # Token retenu ssi son début est dans [s, e) — même règle que le loader.
    out = []
    bounds = []
    p = 0
    for t in toks:
        i = text.find(t["form"], p)
        bounds.append((i, i + len(t["form"])))
        p = i + len(t["form"])
    for k, t in enumerate(toks, 1):
        a, _b = bounds[k - 1]
        if s <= a < e:
            out.append(k)
    assert out, clause
    return [out[0], out[-1]]


def construire() -> dict:
    sentences = []
    for n, (text, clauses, arretes) in enumerate(PROPOSITIONS, 1):
        toks = _tokens(text)
        nodes = []
        for k, spec in enumerate(clauses):
            # (span, label, type) ou (clause, type) avec span == label.
            span, label, typ = spec if len(spec) == 3 else (spec[0], spec[0], spec[1])
            a, b = _span_tokens(text, toks, span)
            nodes.append({"id": f"n{k:03d}", "type": typ,
                          "label": label, "token_span": [a, b]})
        edges = []
        for src, dst, rel, third in arretes:
            e = {"relation": rel, "confidence": 0.0, "explicit": True, "negated": False}
            if isinstance(src, list):
                e["sources"] = [f"n{k:03d}" for k in src]
            else:
                e["source"] = f"n{src:03d}"
            e["target"] = f"n{dst:03d}"
            if third is not None:
                e["third"] = {"role": third["role"],
                              "node": f"n{third['clause']:03d}"}
            else:
                e["third"] = None
            edges.append(e)
        sentences.append({"id": f"t{n:03d}", "text": text, "tokens": toks,
                          "cir": {"nodes": nodes, "edges": edges}})
    return {"schema_version": "4.0",
            "document": {"id": "silver-ternary-proposal-20", "lang": "fr",
                         "weight": 0.0, "sentences": sentences}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Génère la proposition silver ternaire.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    doc = construire()
    blob = json.dumps(doc, ensure_ascii=False, indent=1)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(blob)
    print(json.dumps({"phrases": len(doc["document"]["sentences"]),
                      "sha256": hashlib.sha256(blob.encode()).hexdigest()[:16],
                      "poids": 0.0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
