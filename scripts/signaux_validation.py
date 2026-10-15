#!/usr/bin/env python3
"""Signaux PV-2/ PV-3 protocole-validation-machine.md sur lots propose (minage).

Entrées : propose.json + silver.jsonl + visées {sid: classe}.
Sortie : <out>.json [{id, relation, visee, S1..S4, red:[R1..], auto: oui-auto|manuel}].
Règle auto-oui : S1 ∧ (S2 ∨ S3) ∧ S4 ∧ ¬Red. R5 (silver rédigé main) : n/a ici (minage).
"""
import argparse
import json
import re
import sys

AP = argparse.ArgumentParser()
AP.add_argument("--propose", required=True)
AP.add_argument("--silver", required=True)
AP.add_argument("--visees", required=True)
AP.add_argument("--out", required=True)
A = AP.parse_args()

LEX = {
 "prevent": ["empêch", "évit", "stopp", "bloqu", "interdi", "prévien", "neutralis",
             "écarte", "détourne", "coup", "retient", "retiennent"],
 "filter": ["exception", "sauf", "hormis", "excepté", "seuls", "seul", "seule",
            "uniquement", "seulement", "hors", "part"],
 "control_dependency": ["déclench", "contrôle", "direction", "suspend", "ordonn",
                        "limit", "restrein", "censur", "confisqu", "expuls",
                        "interromp", "rejet", "gel", "ferm", "exclu"],
 "data_dependency": ["dépend", "repose", "découle", "résulte", "traduit"],
 "cause": ["caus", "provoqu", "engendre", "entraîn", "génère", "déclench"],
 "enable": ["permet", "facilit", "rend possible", "donne lieu"],
 "condition": ["si", "lorsque", "quand", "condition"],
 "concession": ["bien que", "quoique", "toutefois", "cependant", "malgré", "même si"],
 "opposition": ["mais", "alors que", "tandis", "contre", "revanche"],
 "sequence": ["ensuite", "puis", "après", "dès", "lors"],
 "motivation": ["pour", "afin", "car", "parce"],
}
AMBIGUS = ["pour", "sous", "protège", "bloque", "contre"]
# types cibles acceptés par relation (S4) ; source contrainte si précisée
TYPES = {
 "prevent": ({"entite", "processus", "contrainte", "etat_global"},
             {"evenement", "processus", "etat_local", "etat_global", "entite"}),
 "filter": ({"entite", "concept", "processus", "evenement"},
            {"entite", "concept", "evenement", "processus", "etat_local"}),
 "control_dependency": ({"entite"}, None),
 "data_dependency": (None, {"evenement", "concept", "processus", "etat_local", "etat_global"}),
 "cause": (None, {"evenement", "processus", "etat_local", "etat_global", "entite"}),
 "enable": (None, None),
 "condition": (None, None),
 "concession": ({"processus", "evenement", "etat_local", "etat_global", "condition"},
                {"processus", "evenement", "etat_local", "etat_global", "condition"}),
 "opposition": ({"processus", "evenement", "etat_local", "etat_global", "condition"},
                {"processus", "evenement", "etat_local", "etat_global", "condition"}),
 "sequence": ({"processus", "evenement"}, {"processus", "evenement"}),
 "motivation": (None, {"processus", "evenement"}),
}
# R1 (span>15) = flag informatif (précédent lots 01-07), PAS bloquant (PV-3 v2).
R1_INFO = True

silver = {}
for line in open(A.silver, encoding="utf-8"):
    r = json.loads(line)
    silver[r["sid"]] = r
visees = json.load(open(A.visees, encoding="utf-8"))
lot = json.load(open(A.propose, encoding="utf-8"))["document"]["sentences"]

res = []
for s in lot:
    sid = s["id"]
    e = s["cir"]["edges"][0]
    rel = e["relation"]
    ns = {n["id"]: n for n in s["cir"]["nodes"]}
    toks = [t["form"] for t in s["tokens"]]
    src = e["sources"][0]
    m = toks[e["marker_token"] - 1].lower() if e.get("marker_token") else ""
    s0 = ns[src]["token_span"][0]
    t0 = ns[e["target"]]["token_span"][0]
    S1 = (rel == visees.get(sid))
    lexhits = [r for r, lst in LEX.items()
               if any(m == k or m.startswith(k) or k.startswith(m) for k in lst)]
    S2 = (rel in lexhits) and not (m in AMBIGUS)
    S3 = s0 < t0
    tsrc, ttgt = ns[src]["type"], ns[e["target"]]["type"]
    tcomp = TYPES.get(rel, (None, None))
    S4 = ((tcomp[0] is None or tsrc in tcomp[0])
          and (tcomp[1] is None or ttgt in tcomp[1]))
    red = []
    r1 = False
    if any(n["token_span"][1] - n["token_span"][0] > 15 for n in ns.values()):
        r1 = True  # informatif seul (PV-3 v2) : noté, pas bloquant
    conc = set(c.get("relation") for c in silver.get(sid, {}).get("connecteurs", [])
               if isinstance(c, dict) and c.get("relation"))
    if len(conc - {"motivation"} if rel == "motivation" else conc) > 1:
        red.append("R2")
    if m in AMBIGUS:
        red.append("R3")
    # R5 : négation ±4 tok autour du marqueur (sens nié).
    NEG = ("ne", "n'", "pas", "jamais", "plus", "guère", "aucun", "nulle", "nul")
    if e.get("marker_token"):
        mid = e["marker_token"]
        fen = [t.lower().strip(",.;:!?«»\"'") for t in toks[max(0, mid - 5):mid + 4]]
        if any(w in NEG for w in fen):
            red.append("R5")
    # R6 : span dégénéré (≤2 tok) sur phrase >12 tok.
    if len(toks) > 12 and any(n["token_span"][1] - n["token_span"][0] + 1 <= 1
                             for n in ns.values()):
        red.append("R6")
    auto = "oui-auto" if (S1 and (S2 or S3) and S4 and not red) else "manuel"
    if r1 and auto == "oui-auto":
        auto = "oui-auto-r1"
    res.append({"id": sid, "relation": rel, "visee": visees.get(sid),
                "S": [S1, S2, S3, S4], "red": red, "r1": r1, "auto": auto,
                "marqueur": m, "types": [tsrc, ttgt]})
json.dump(res, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
from collections import Counter
print(f"{A.propose}: n={len(res)} auto-oui={sum(1 for r in res if r['auto']=='oui-auto')} "
      f"manuel={sum(1 for r in res if r['auto']!='oui-auto')}")
print("motifs manuel:", Counter(r["red"] and tuple(r["red"]) or ("pas-de-signal",)
      for r in res if r["auto"] != "oui-auto").most_common(6))
