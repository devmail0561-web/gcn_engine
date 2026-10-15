#!/usr/bin/env python3
"""P2-allégé plan-remediation-donnees.md : correction mécanique des coquilles de labels
univoques (éter->être, décriver->décrire). Spans/types/relations intacts.

Gardes : lots 01-04 intouchables (REGLE_EQUILIBRE) ; phrases tombstonees ignorées ;
verbes ambigus (sociéter, hypothesi, antarctique) + sujets hors-span : rapportés, non touchés.
Sortie : diff console + patch log JSON. Re-jouer detect_bruit.py après pour vérifier.
"""
import argparse
import glob
import json
import os
import re
import sys

AP = argparse.ArgumentParser()
AP.add_argument("--annotated", required=True)
AP.add_argument("--rapport", required=True, help="bruit_*.json (liste label_coquille)")
AP.add_argument("--dry-run", action="store_true")
A = AP.parse_args()

FIX = {"éter": "être", "décriver": "décrire"}
INTOUCHABLES = {"lot01_13.json", "lot02_12.json", "lot03_12.json", "lot04_18.json"}

tomb = set()
for fp in sorted(glob.glob(os.path.join(A.annotated, "quarantaine*.json"))):
    for q in json.load(open(fp, encoding="utf-8")):
        if isinstance(q, dict) and q.get("id"):
            tomb.add(q["id"])

rep = json.load(open(A.rapport, encoding="utf-8"))
cibles = [(lot, f) for lot, l in rep["lots"].items() for f in l["flags"]
          if f["cat"] == "label_coquille"]
patch, reportes, vus = [], [], set()
for lot, f in cibles:
    key = (lot, f["id"], f["detail"])
    if key in vus:
        continue
    vus.add(key)
    nid = f["detail"].split(" ", 1)[0]
    if lot in INTOUCHABLES:
        reportes.append({"lot": lot, "id": f["id"], "motif": "lot 01-04 intouchable",
                         "detail": f["detail"]})
        continue
    if f["id"] in tomb:
        reportes.append({"lot": lot, "id": f["id"], "motif": "phrase tombstonee",
                         "detail": f["detail"]})
        continue
    m = re.match(r"^[^(]+\(([^)]+)\)$", f["detail"].split(" ", 1)[-1].split(" | ")[0])
    verbe = f["detail"].split(" ", 1)[-1].split("(", 1)[0]
    if verbe not in FIX or not m:
        reportes.append({"lot": lot, "id": f["id"], "motif": f"verbe ambigu ({verbe})",
                         "detail": f["detail"]})
        continue
    patch.append({"lot": lot, "id": f["id"], "nid": nid,
                  "avant": f"{verbe}({m.group(1)})",
                  "apres": f"{FIX[verbe]}({m.group(1)})"})

print(f"correctifs mécaniques: {len(patch)} | reportés: {len(reportes)}")
for r in reportes:
    print(f"  REPORTE {r['lot']}#{r['id']} [{r['motif']}] {r['detail'][:70]}")
if A.dry_run:
    for p in patch:
        print(f"  {p['lot']}#{p['id']}/{p['nid']}: {p['avant']} -> {p['apres']}")
    sys.exit(0)

by_lot = {}
for p in patch:
    by_lot.setdefault(p["lot"], []).append(p)
for lot, ps in by_lot.items():
    fp = os.path.join(A.annotated, lot)
    d = json.load(open(fp, encoding="utf-8"))
    idx = {s["id"]: s for s in d["document"]["sentences"]}
    for p in ps:
        s = idx[p["id"]]
        n = next(n for n in s["cir"]["nodes"] if n.get("id") == p["nid"])
        assert n.get("label") == p["avant"], f"derive: {lot}#{p['id']} {n.get('label')}"
        n["label"] = p["apres"]
        print(f"  {lot}#{p['id']}/{p['nid']}: {p['avant']} -> {p['apres']}")
    json.dump(d, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"lots réécrits: {len(by_lot)} (labels seuls, spans/types/relations intacts)")
