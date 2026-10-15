#!/usr/bin/env python3
"""Promeut les phrases P1b VALIDÉES vers le pool (annotated/lot26_p1b.json, format lot v4).

Entrée : verdicts JSON {id: "oui"|"non"|{"corriger": ...}} pour les 129 p1bNNN.
REFUSE si un verdict manque. Seuls les "oui" sont promus (origin="valide-humain",
confidence inchangée 0.0 = poids silver, jamais 1.0 auto). Les "non" restent silver.
Dédup vs pool existant (texte normalisé) : doublon => refusé, rapporté.
Par défaut dry-run. --appliquer écrit.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

AP = argparse.ArgumentParser()
AP.add_argument("--silver", required=True, help="silver_rares_p1b.json")
AP.add_argument("--verdicts", required=True, help='JSON {id: "oui"|"non"}')
AP.add_argument("--annotated", required=True)
AP.add_argument("--appliquer", action="store_true")
A = AP.parse_args()

silver = json.load(open(A.silver, encoding="utf-8"))
ss = silver["document"]["sentences"]
verdicts = json.load(open(A.verdicts, encoding="utf-8"))
manquants = [s["id"] for s in ss if s["id"] not in verdicts]
if manquants:
    print(f"REFUS: {len(manquants)} sans verdict (ex: {manquants[0]}). Renseigner d'abord.")
    sys.exit(2)
bad = {k: v for k, v in verdicts.items() if v not in ("oui", "non")}
if bad:
    print(f"REFUS: verdicts invalides: {list(bad)[:5]} (oui|non seuls dans v1)")
    sys.exit(2)

pool = set()
for fp in sorted(glob.glob(os.path.join(A.annotated, "lot*.json"))):
    d = json.load(open(fp, encoding="utf-8"))
    for s in d["document"]["sentences"]:
        pool.add(re.sub(r"\s+", " ", (s.get("text") or "").strip().lower()))

promus, rejetes, doublons = [], [], []
for s in ss:
    v = verdicts[s["id"]]
    key = re.sub(r"\s+", " ", s["text"].strip().lower())
    if key in pool:
        doublons.append(s["id"])
        continue
    if v == "oui":
        s = dict(s)
        s["cir"] = dict(s["cir"])
        s["cir"]["nodes"] = [{**n, "origin": "valide-humain"} for n in s["cir"]["nodes"]]
        promus.append(s)
    else:
        rejetes.append(s["id"])
print(f"promus: {len(promus)} | non: {len(rejetes)} | doublons: {len(doublons)} {doublons}")
if not A.appliquer:
    print("dry-run : rien écrit (relancer avec --appliquer)")
    sys.exit(0)

out = {"schema_version": "4.0",
       "document": {"id": "lot26_p1b", "sentences": promus}}
fp = os.path.join(A.annotated, "lot26_p1b.json")
blob = json.dumps(out, ensure_ascii=False, indent=1)
open(fp, "w", encoding="utf-8").write(blob)
print(f"écrit {fp} sha256={hashlib.sha256(blob.encode()).hexdigest()[:16]}")
