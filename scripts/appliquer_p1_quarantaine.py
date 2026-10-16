#!/usr/bin/env python3
"""P1+P3 plan-remediation-donnees.md : quarantaine junk + split EN (tombstones, disque intact).

Entree : rapport scripts/detect_bruit.py. Ne modifie AUCUN lot existant.
Ecrit :
  annotated/quarantaine_nettoyage_AAAAMMJJ.json : [{id, text, motif, lot, cat, tombstone:true}]
    (format compatible make_split.py:57-62 ; EN et latex, motifs distincts)
  ../split_en/split_en.json + manifest.json : phrases EN preservees (format lot v4).
Sortie console : manifests sha256.
"""
import argparse
import glob
import hashlib
import json
import os
from datetime import date

AP = argparse.ArgumentParser()
AP.add_argument("--rapport", required=True)
AP.add_argument("--annotated", required=True)
AP.add_argument("--dry-run", action="store_true")
A = AP.parse_args()
tag = date.today().strftime("%Y%m%d")

rep = json.load(open(A.rapport, encoding="utf-8"))
by_lot = {}
for fp in sorted(glob.glob(os.path.join(A.annotated, "lot*.json"))):
    name = os.path.basename(fp)
    d = json.load(open(fp, encoding="utf-8"))
    by_lot[name] = {s["id"]: s for s in d["document"]["sentences"]}

def entries(keys, motif, cat):
    out = []
    for k in sorted(set(keys)):
        lot, sid = k.split("#", 1)
        s = by_lot.get(lot, {}).get(sid)
        if s is None:
            print(f"ATTENTION introuvable: {k}")
            continue
        out.append({"id": sid, "text": s.get("text", ""), "motif": motif,
                    "lot": lot, "cat": cat, "tombstone": True})
    return out

junk = entries(rep.get("quarantaine_junk", []), "P1 latex (junk certain)", "latex")
en = entries(rep.get("split_en", []), "P1 EN (split_en, exclu FR)", "en_texte")
print(f"junk: {len(junk)} | EN: {len(en)}")

def sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

qpath = os.path.join(A.annotated, f"quarantaine_nettoyage_{tag}.json")
en_dir = os.path.normpath(os.path.join(A.annotated, "..", "split_en"))
en_path = os.path.join(en_dir, "split_en.json")
en_man = os.path.join(en_dir, "manifest.json")
en_doc = {"schema_version": "4.0",
          "document": {"id": "split_en",
                       "sentences": [by_lot[k.split('#', 1)[0]][k.split('#', 1)[1]]
                                     for k in sorted(set(rep.get("split_en", [])))
                                     if k.split('#', 1)[1] in by_lot.get(k.split('#', 1)[0], {})]}}
manifest = {"date": tag, "rapport": os.path.basename(A.rapport),
            "quarantaine": {"fichier": os.path.basename(qpath), "n": len(junk) + len(en),
                            "dont_junk": len(junk), "dont_en": len(en)},
            "split_en": {"fichier": "split_en/split_en.json", "n": len(en_doc["document"]["sentences"])}}
if not A.dry_run:
    json.dump(junk + en, open(qpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.makedirs(en_dir, exist_ok=True)
    json.dump(en_doc, open(en_path, "w", encoding="utf-8"), ensure_ascii=False)
    manifest["quarantaine"]["sha256"] = sha(junk + en)
    manifest["split_en"]["sha256"] = sha(en_doc)
    json.dump(manifest, open(en_man, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"ecrit: {qpath}\n  sha256={manifest['quarantaine']['sha256']}")
    print(f"ecrit: {en_path} (n={len(en_doc['document']['sentences'])})\n"
          f"  sha256={manifest['split_en']['sha256']}\nmanifest: {en_man}")
else:
    print("dry-run : rien ecrit")
