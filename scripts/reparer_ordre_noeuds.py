#!/usr/bin/env python3
"""Répare l'ordre des nœuds après inversions P4 : le loader (tok_loader.py:64-66)
exclut les arêtes « arrières » (rang source >= rang cible, rangs = ordre liste).
Pour chaque arête à sens inversé, remonte la/les source(s) juste avant la cible
(ordre relatif sinon préservé, ids intacts). Dry-run par défaut.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

AP = argparse.ArgumentParser()
AP.add_argument("--annotated", required=True)
AP.add_argument("--appliquer", action="store_true")
A = AP.parse_args()

touches = 0
for fp in sorted(glob.glob(os.path.join(A.annotated, "lot*.json"))):
    d = json.load(open(fp, encoding="utf-8"))
    changed = False
    for s in d["document"]["sentences"]:
        nodes = (s.get("cir") or {}).get("nodes", [])
        if len(nodes) < 2:
            continue
        for e in (s.get("cir") or {}).get("edges", []):
            srcs = list(e.get("sources", []))
            tgt = e.get("target")
            ids = [n.get("id") for n in nodes]
            if tgt not in ids or any(x not in ids for x in srcs):
                continue
            if max(ids.index(x) for x in srcs) < ids.index(tgt):
                continue
            # Remonte les sources juste avant la cible.
            src_nodes = [n for n in nodes if n.get("id") in srcs]
            rest = [n for n in nodes if n.get("id") not in srcs]
            ti = next(k for k, n in enumerate(rest) if n.get("id") == tgt)
            s["cir"]["nodes"] = rest[:ti] + src_nodes + rest[ti:]
            changed = True
    if changed:
        touches += 1
        if A.appliquer:
            blob = json.dumps(d, ensure_ascii=False, indent=1)
            open(fp, "w", encoding="utf-8").write(blob)
            print(f"réordonné {os.path.basename(fp)} "
                  f"sha={hashlib.sha256(blob.encode()).hexdigest()[:16]}")
        else:
            print(f"à réordonner (dry-run) : {os.path.basename(fp)}")
print(f"lots touchés: {touches} {'(écrit)' if A.appliquer else '(dry-run)'}")
