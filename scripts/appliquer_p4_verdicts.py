#!/usr/bin/env python3
"""Applique les verdicts P4 (v4/p4/P4_propositions.json) aux lots. REFUSE si un verdict
est manquant (None) : pas d'application partielle silencieuse.

Verdicts : garder | inverser | quarantaine | retyper:<REL11>.
- inverser : swap sources<->target (+ third.node si concerné) ; multi-sources => refusé, rapporté.
- retyper : REL validée ∈ 11 relations directes.
- quarantaine : tombstone motif P4 (disque intact).
Par défaut dry-run. --appliquer écrit (avec sauvegardes /tmp).
"""
import argparse
import copy
import glob
import hashlib
import json
import os
import shutil
import sys
from datetime import date

REL11 = {"cause", "concession", "condition", "control_dependency", "data_dependency",
         "enable", "filter", "motivation", "opposition", "prevent", "sequence"}

AP = argparse.ArgumentParser()
AP.add_argument("--annotated", required=True)
AP.add_argument("--propositions", required=True)
AP.add_argument("--appliquer", action="store_true")
A = AP.parse_args()

props = json.load(open(A.propositions, encoding="utf-8"))
sans_verdict = [p for p in props if not p.get("verdict")]
if sans_verdict:
    print(f"REFUS: {len(sans_verdict)} cas sans verdict (ex: "
          f"{sans_verdict[0]['lot']}#{sans_verdict[0]['id']}). Renseigner d'abord.")
    sys.exit(2)

by_lot = {}
for fp in sorted(glob.glob(os.path.join(A.annotated, "lot*.json"))):
    d = json.load(open(fp, encoding="utf-8"))
    by_lot[os.path.basename(fp)] = (fp, {s["id"]: s for s in d["document"]["sentences"]})

quar, refuses, stats = [], [], {"garder": 0, "inverser": 0, "quarantaine": 0, "retyper": 0}
touched = {}
for p in props:
    v = p["verdict"]
    lot, sid = p["lot"], p["id"]
    fp, idx = by_lot[lot]
    s = idx.get(sid)
    if s is None:
        refuses.append({**p, "motif": "phrase introuvable"})
        continue
    e = next((x for x in (s.get("cir") or {}).get("edges", [])
              if x.get("relation") == p["relation"]), None)
    if e is None:
        refuses.append({**p, "motif": "arête introuvable"})
        continue
    if v == "garder":
        stats["garder"] += 1
    elif v == "inverser":
        srcs, tgt = list(e.get("sources", [])), e.get("target")
        if len(srcs) != 1 or p["src"] not in srcs or p["tgt"] != tgt:
            refuses.append({**p, "motif": "structure inattendue (multi-sources ou dérive)"})
            continue
        e["sources"], e["target"] = [tgt], srcs[0]
        th = e.get("third")
        if isinstance(th, dict) and th.get("node") in (srcs[0], tgt):
            th["node"] = tgt if th["node"] == srcs[0] else srcs[0]
        stats["inverser"] += 1
        touched.setdefault(lot, []).append(f"{sid}: {srcs[0]}->{tgt} => {tgt}->{srcs[0]}")
    elif v == "quarantaine":
        quar.append({"id": sid, "text": s.get("text", ""),
                     "motif": f"P4 {p['relation']} ({p.get('regle', '')})",
                     "lot": lot, "tombstone": True})
        stats["quarantaine"] += 1
    elif v.startswith("retyper:"):
        rel = v.split(":", 1)[1]
        if rel not in REL11:
            refuses.append({**p, "motif": f"relation invalide {rel}"})
            continue
        e["relation"] = rel
        stats["retyper"] += 1
        touched.setdefault(lot, []).append(f"{sid}: {p['relation']} => {rel}")
    else:
        refuses.append({**p, "motif": f"verdict inconnu {v}"})

print(f"verdicts: {stats} | refusés: {len(refuses)}")
for r in refuses:
    print(f"  REFUSE {r['lot']}#{r['id']} [{r.get('motif')}]")
for lot, ops in sorted(touched.items()):
    for op in ops:
        print(f"  {lot}#{op}")
print(f"quarantaine P4: {len(quar)}")
if not A.appliquer:
    print("dry-run : rien écrit (relancer avec --appliquer)")
    sys.exit(0 if not refuses else 3)

bak = f"/tmp/opencode/p4_backup_{date.today():%Y%m%d}"
os.makedirs(bak, exist_ok=True)
for lot in touched:
    fp, _ = by_lot[lot]
    shutil.copy(fp, os.path.join(bak, lot))
    d = json.load(open(fp, encoding="utf-8"))
    # Les objets ont été mutés en mémoire : réécriture depuis by_lot.
    sents = [by_lot[lot][1][s["id"]] for s in d["document"]["sentences"]]
    d["document"]["sentences"] = sents
    json.dump(d, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sha = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
    print(f"écrit {lot} sha={sha} (backup {bak}/{lot})")
if quar:
    qf = os.path.join(A.annotated, f"quarantaine_p4_{date.today():%Y%m%d}.json")
    json.dump(quar, open(qf, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"écrit {qf} ({len(quar)} tombstones)")
