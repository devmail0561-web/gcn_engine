# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Silver joints-b : 12 joint_prevent de plus (22 → 34, gold-3000).

Même contrat que la vague J (registre pastoral, spans calculés, poids 0.0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re

J: list = []
_P = "processus"
_G = "etat_global"
_L = "etat_local"
_E = "evenement"


def _j(text, c0, l0, t0, c1, l1, t1, c2, l2, t2):
    J.append((text, [(c0, l0, t0), (c1, l1, t1), (c2, l2, t2)], ([0, 1], 2, "joint_prevent", None)))


_j("La grêle hache les avoines et le gel fend les betteraves : le bétail jeûnera.",
   "La grêle hache les avoines", "grêle qui hache", _E,
   "le gel fend les betteraves", "gel qui fend", _G,
   "le bétail jeûnera", "bétail qui jeûne", _L)
_j("Le percepteur prend le blé et l'huissier compte les meules : le grenier sera vide à Pâques.",
   "Le percepteur prend le blé", "percepteur qui prend", _P,
   "l'huissier compte les meules", "huissier qui compte", _P,
   "le grenier sera vide à Pâques", "grenier vide", _L)
_j("La taupe mine le pré et les mulots rongent les racines : la faux ne trouvera rien.",
   "La taupe mine le pré", "taupe qui mine", _P,
   "les mulots rongent les racines", "mulots qui rongent", _P,
   "la faux ne trouvera rien", "faux sans ouvrage", _L)
_j("L'ordonnance ferme les cabarets et le couvre-feu vide les rues : la fête n'aura pas lieu.",
   "L'ordonnance ferme les cabarets", "ordonnance qui ferme", _P,
   "le couvre-feu vide les rues", "couvre-feu qui vide", _G,
   "la fête n'aura pas lieu", "fête annulée", _L)
_j("Le verglas bloque les charrettes et le brouillard égare les piétons : le marché sera désert.",
   "Le verglas bloque les charrettes", "verglas qui bloque", _G,
   "le brouillard égare les piétons", "brouillard qui égare", _G,
   "le marché sera désert", "marché désert", _L)
_j("La rouille mange les socs et le gel casse les mancherons : la charrue restera au hangar.",
   "La rouille mange les socs", "rouille qui mange", _P,
   "le gel casse les mancherons", "gel qui casse", _G,
   "la charrue restera au hangar", "charrue au hangar", _L)
_j("Le choléra vide les maisons et la peur cloître les vivants : l'école restera fermée.",
   "Le choléra vide les maisons", "choléra qui vide", _E,
   "la peur cloître les vivants", "peur qui cloître", _G,
   "l'école restera fermée", "école fermée", _L)
_j("Le torrent coupe le sentier et l'éboulement barre le tunnel : les secours n'arriveront pas.",
   "Le torrent coupe le sentier", "torrent qui coupe", _E,
   "l'éboulement barre le tunnel", "éboulement qui barre", _E,
   "les secours n'arriveront pas", "secours bloqués", _L)
_j("La censure raye les affiches et la police surveille les carrefours : le mot d'ordre ne circulera pas.",
   "La censure raye les affiches", "censure qui raye", _P,
   "la police surveille les carrefours", "police qui surveille", _P,
   "le mot d'ordre ne circulera pas", "mot d'ordre étouffé", _L)
_j("Le mildiou noircit les feuilles et l'oïdium poudre les grappes : la vendange est compromise.",
   "Le mildiou noircit les feuilles", "mildiou qui noircit", _P,
   "l'oïdium poudre les grappes", "oïdium qui poudre", _P,
   "la vendange est compromise", "vendange compromise", _L)
_j("La faillite ferme l'atelier et le créancier vend les outils : les ouvriers resteront sans ouvrage.",
   "La faillite ferme l'atelier", "faillite qui ferme", _E,
   "le créancier vend les outils", "créancier qui vend", _P,
   "les ouvriers resteront sans ouvrage", "ouvriers sans ouvrage", _L)
_j("La sécheresse assèche les mares et la canicule écrase les bêtes : l'abreuvoir restera à sec.",
   "La sécheresse assèche les mares", "sécheresse qui assèche", _G,
   "la canicule écrase les bêtes", "canicule qui écrase", _G,
   "l'abreuvoir restera à sec", "abreuvoir à sec", _L)


def _tokens(text: str) -> list[dict]:
    toks, cur = [], 0
    for mot in text.split(" "):
        i = text.find(mot, cur)
        toks.append({"id": len(toks) + 1, "form": mot})
        cur = i + len(mot)
    return toks


def _span_tokens(text: str, toks: list[dict], clause: str) -> list[int]:
    bounds, p = [], 0
    for t in toks:
        i = text.find(t["form"], p)
        bounds.append((i, i + len(t["form"])))
        p = i + len(t["form"])
    s = text.find(clause)
    assert s >= 0, clause
    e = s + len(clause)
    out = [k for k, t in enumerate(toks, 1) if s <= bounds[k - 1][0] < e]
    assert out, clause
    while len(out) > 1 and re.fullmatch(r"[.,;:!?]+", toks[out[-1] - 1]["form"]):
        out.pop()
    return [out[0], out[-1]]


def construire() -> dict:
    assert len(J) == 12, len(J)
    sentences = []
    for n, (text, clauses, (src, dst, rel, third)) in enumerate(J, 1):
        toks = _tokens(text)
        nodes = []
        for k, (span, label, typ) in enumerate(clauses):
            a, b = _span_tokens(text, toks, span)
            nodes.append({"id": f"n{k:03d}", "type": typ, "label": label,
                          "token_span": [a, b], "scope": "specific",
                          "temporal_index": k, "origin": "propose",
                          "span_source": "silver-joints-b"})
        e = {"relation": rel, "confidence": 0.0, "explicit": True, "negated": False}
        e["sources"] = [f"n{k:03d}" for k in src]
        e["target"] = f"n{dst:03d}"
        e["third"] = None
        sentences.append({"id": f"jb{n:03d}", "text": text, "tokens": toks,
                          "sentence_type": "declarative", "registre": "pastoral",
                          "cir": {"nodes": nodes, "edges": [e]}})
    return {"schema_version": "4.0",
            "document": {"id": "silver-joints-b", "lang": "fr", "weight": 0.0,
                         "sentences": sentences}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Génère silver joints-b.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    doc = construire()
    blob = json.dumps(doc, ensure_ascii=False, indent=1)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(blob)
    from collections import Counter
    rels = Counter(e["relation"]
                   for s in doc["document"]["sentences"] for e in s["cir"]["edges"])
    print(json.dumps({"phrases": len(doc["document"]["sentences"]),
                      "relations": dict(sorted(rels.items())),
                      "sha256": hashlib.sha256(blob.encode()).hexdigest()[:16],
                      "poids": 0.0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
