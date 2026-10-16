# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Silver spécial : 20 analogy + 20 counterfactual (vague S, gold-3000).

Même contrat que mc3 (registre pastoral, spans calculés, poids 0.0).
Analogy : « Tout comme X, Y » / « De même que X, Y » (concepts).
Counterfactual : irréel (« Sans X, Y aurait... », « Si X avait..., Y aurait... »).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re

S: list = []


def _a(text, c0, l0, c1, l1):
    S.append((text, [(c0, l0, "concept"), (c1, l1, "concept")], (0, 1, "analogy", None)))


def _c(text, c0, l0, t0, c1, l1, t1):
    S.append((text, [(c0, l0, t0), (c1, l1, t1)], (0, 1, "counterfactual", None)))


# --- analogy × 20 ---
_a("Tout comme le berger compte ses brebis, le maître compte ses écoliers.",
   "le berger compte ses brebis", "berger→brebis",
   "le maître compte ses écoliers", "maître→écoliers")
_a("De même que la lavandière bat son linge, la fermière bat le beurre.",
   "la lavandière bat son linge", "lavandière→linge",
   "la fermière bat le beurre", "fermière→beurre")
_a("Tout comme le forgeron trempe la lame, le meunier mouille sa meule.",
   "le forgeron trempe la lame", "forgeron→lame",
   "le meunier mouille sa meule", "meunier→meule")
_a("De même que le jardinier arrache les mauvaises herbes, le maître arrache l'ignorance.",
   "le jardinier arrache les mauvaises herbes", "jardinier→herbes",
   "le maître arrache l'ignorance", "maître→ignorance")
_a("Tout comme la chouette guette la nuit, le garde guette les braconniers.",
   "la chouette guette la nuit", "chouette→nuit",
   "le garde guette les braconniers", "garde→braconniers")
_a("De même que le tisserand noue ses fils, le greffier noue ses dossiers.",
   "le tisserand noue ses fils", "tisserand→fils",
   "le greffier noue ses dossiers", "greffier→dossiers")
_a("Tout comme le laboureur suit son sillon, le marin suit son étoile.",
   "le laboureur suit son sillon", "laboureur→sillon",
   "le marin suit son étoile", "marin→étoile")
_a("De même que la ruche bourdonne au printemps, le marché bourdonne les jours de foire.",
   "la ruche bourdonne au printemps", "ruche→printemps",
   "le marché bourdonne les jours de foire", "marché→foire")
_a("Tout comme le chien garde la bergerie, le douanier garde la frontière.",
   "le chien garde la bergerie", "chien→bergerie",
   "le douanier garde la frontière", "douanier→frontière")
_a("De même que le moissonneur lie ses gerbes, le notaire lie ses actes.",
   "le moissonneur lie ses gerbes", "moissonneur→gerbes",
   "le notaire lie ses actes", "notaire→actes")
_a("Tout comme la pie amasse les brillants, l'usurier amasse les écus.",
   "la pie amasse les brillants", "pie→brillants",
   "l'usurier amasse les écus", "usurier→écus")
_a("De même que le fumier engraisse le champ, la lecture engraisse l'esprit.",
   "le fumier engraisse le champ", "fumier→champ",
   "la lecture engraisse l'esprit", "lecture→esprit")
_a("Tout comme le pivert tambourine l'écorce, le crieur tambourine la nouvelle.",
   "le pivert tambourine l'écorce", "pivert→écorce",
   "le crieur tambourine la nouvelle", "crieur→nouvelle")
_a("De même que la baratte tourne sans fin, la médisance tourne dans le village.",
   "la baratte tourne sans fin", "baratte→sans-fin",
   "la médisance tourne dans le village", "médisance→village")
_a("Tout comme le renard flaire le vent, le maquignon flaire les affaires.",
   "le renard flaire le vent", "renard→vent",
   "le maquignon flaire les affaires", "maquignon→affaires")
_a("De même que le pressoir écrase le raisin, le fisc écrase le contribuable.",
   "le pressoir écrase le raisin", "pressoir→raisin",
   "le fisc écrase le contribuable", "fisc→contribuable")
_a("Tout comme l'hirondelle annonce le printemps, la primevère annonce les beaux jours.",
   "l'hirondelle annonce le printemps", "hirondelle→printemps",
   "la primevère annonce les beaux jours", "primevère→beaux-jours")
_a("De même que le puits rend l'eau claire, le temps rend les choses claires.",
   "le puits rend l'eau claire", "puits→eau-claire",
   "le temps rend les choses claires", "temps→choses-claires")
_a("Tout comme le coq réveille la basse-cour, la cloche réveille le village.",
   "le coq réveille la basse-cour", "coq→basse-cour",
   "la cloche réveille le village", "cloche→village")
_a("De même que la chèvre broute les ronces, le critique broute les réputations.",
   "la chèvre broute les ronces", "chèvre→ronces",
   "le critique broute les réputations", "critique→réputations")

# --- counterfactual × 20 ---
_C_DATA = [
("Sans la digue, le pré aurait été inondé.",
 "Sans la digue", "digue absente (irréel)", "condition",
 "le pré aurait été inondé", "pré inondé (irréel)", "etat_local"),
("Si le garde avait veillé, les maraudeurs ne seraient point passés.",
 "Si le garde avait veillé", "garde qui veille (irréel)", "evenement",
 "les maraudeurs ne seraient point passés", "maraudeurs bloqués (irréel)", "etat_local"),
("Sans le forgeron, la lame serait restée molle.",
 "Sans le forgeron", "forgeron absent (irréel)", "condition",
 "la lame serait restée molle", "lame molle (irréel)", "etat_local"),
("Si la cloche avait sonné faux, les fidèles auraient levé la tête.",
 "Si la cloche avait sonné faux", "cloche fausse (irréel)", "evenement",
 "les fidèles auraient levé la tête", "fidèles attentifs (irréel)", "etat_local"),
("Sans la sage-femme, l'enfant n'aurait point vécu.",
 "Sans la sage-femme", "sage-femme absente (irréel)", "condition",
 "l'enfant n'aurait point vécu", "enfant mort (irréel)", "etat_local"),
("Si le meunier avait dormi, la meule se serait arrêtée avant l'aube.",
 "Si le meunier avait dormi", "meunier endormi (irréel)", "evenement",
 "la meule se serait arrêtée avant l'aube", "meule arrêtée (irréel)", "etat_local"),
("Sans le berger, le troupeau se serait dispersé.",
 "Sans le berger", "berger absent (irréel)", "condition",
 "le troupeau se serait dispersé", "troupeau dispersé (irréel)", "etat_local"),
("Si la rivière avait baissé, le passeur aurait gardé sa barque au sec.",
 "Si la rivière avait baissé", "rivière basse (irréel)", "evenement",
 "le passeur aurait gardé sa barque au sec", "barque au sec (irréel)", "etat_local"),
("Sans le coq, la basse-cour dormirait encore.",
 "Sans le coq", "coq absent (irréel)", "condition",
 "la basse-cour dormirait encore", "basse-cour endormie (irréel)", "etat_local"),
("Si l'orage avait éclaté, le lait aurait tourné.",
 "Si l'orage avait éclaté", "orage éclaté (irréel)", "evenement",
 "le lait aurait tourné", "lait tourné (irréel)", "etat_local"),
("Sans la haie, le potager aurait gelé.",
 "Sans la haie", "haie absente (irréel)", "condition",
 "le potager aurait gelé", "potager gelé (irréel)", "etat_local"),
("Si le chien avait aboyé, le renard aurait contourné le poulailler.",
 "Si le chien avait aboyé", "chien aboyant (irréel)", "evenement",
 "le renard aurait contourné le poulailler", "renard détourné (irréel)", "etat_local"),
("Sans l'instituteur, les enfants seraient restés ignorants.",
 "Sans l'instituteur", "instituteur absent (irréel)", "condition",
 "les enfants seraient restés ignorants", "enfants ignorants (irréel)", "etat_local"),
("Si le soleil avait chauffé la vigne, les bourgeons auraient éclaté.",
 "Si le soleil avait chauffé la vigne", "soleil chauffant (irréel)", "evenement",
 "les bourgeons auraient éclaté", "bourgeons éclatés (irréel)", "etat_local"),
("Sans le cantonnier, le chemin serait resté impraticable.",
 "Sans le cantonnier", "cantonnier absent (irréel)", "condition",
 "le chemin serait resté impraticable", "chemin impraticable (irréel)", "etat_local"),
("Si la barrière avait cédé, les vaches auraient pâturé le jardin du voisin.",
 "Si la barrière avait cédé", "barrière cédée (irréel)", "evenement",
 "les vaches auraient pâturé le jardin du voisin", "jardin pâturé (irréel)", "etat_local"),
("Sans la mère, la pâte n'aurait point levé.",
 "Sans la mère", "mère absente (irréel)", "condition",
 "la pâte n'aurait point levé", "pâte plate (irréel)", "etat_local"),
("Si les taupes n'avaient point labouré, les prés ne seraient point bosselés.",
 "Si les taupes n'avaient point labouré", "taupes inactives (irréel)", "evenement",
 "les prés ne seraient point bosselés", "prés plans (irréel)", "etat_local"),
("Sans le sonneur, la procession ne se serait point ébranlée.",
 "Sans le sonneur", "sonneur absent (irréel)", "condition",
 "la procession ne se serait point ébranlée", "procession immobile (irréel)", "etat_local"),
("Si le beurre avait manqué de glace, il aurait ranci.",
 "Si le beurre avait manqué de glace", "glace manquante (irréel)", "evenement",
 "il aurait ranci", "beurre rance (irréel)", "etat_local"),
]
for _t, _c0, _l0, _t0, _c1, _l1, _t1 in _C_DATA:
    _c(_t, _c0, _l0, _t0, _c1, _l1, _t1)


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
    assert len(S) == 40, len(S)
    sentences = []
    for n, (text, clauses, (src, dst, rel, third)) in enumerate(S, 1):
        toks = _tokens(text)
        nodes = []
        for k, (span, label, typ) in enumerate(clauses):
            a, b = _span_tokens(text, toks, span)
            nodes.append({"id": f"n{k:03d}", "type": typ, "label": label,
                          "token_span": [a, b], "scope": "specific",
                          "temporal_index": k, "origin": "propose",
                          "span_source": "silver-special"})
        e = {"relation": rel, "confidence": 0.0, "explicit": True, "negated": False,
             "sources": [f"n{src:03d}"], "target": f"n{dst:03d}", "third": None}
        sentences.append({"id": f"sp{n:03d}", "text": text, "tokens": toks,
                          "sentence_type": "declarative", "registre": "pastoral",
                          "cir": {"nodes": nodes, "edges": [e]}})
    return {"schema_version": "4.0",
            "document": {"id": "silver-special", "lang": "fr", "weight": 0.0,
                         "sentences": sentences}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Génère silver spécial (vague S).")
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
