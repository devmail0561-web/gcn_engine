#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Audit de schéma des lots supervisés — phase 0 (P0-5, D2).

Classe chaque fichier .jsonl/.json de supervision par version de schéma :
types de noeuds (8 cibles vs aliases v2), relations (11 vs 19), présence
d'intent/acte, méthode (_methode), confiance. Lecture seule, stdlib uniquement.

Usage :
    python3 scripts/audit_gold_schema.py [RACINE] [--max-per-file N]
Sortie : tableau + verdict par répertoire (v2 / v4 / mixte / silver-only).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V4_TYPES = {"processus", "etat_local", "etat_global", "entite",
            "condition", "concept", "evenement", "contrainte"}
V2_ALIASES = {"etat", "etat_systemique", "action", "transition"}
V4_RELS = {"cause", "enable", "prevent", "condition", "concession", "sequence",
           "motivation", "filter", "opposition", "data_dependency",
           "control_dependency", "analogy", "counterfactual",
           "conditional_cause", "mediated_cause", "joint_cause",
           "conditional_prevent", "mediated_prevent", "joint_prevent"}
V2_RELS = {"cause", "enable", "prevent", "condition", "concession", "sequence",
           "motivation", "filter", "opposition", "data_dependency",
           "control_dependency"}


def _iter_records(path: Path, cap: int):
    try:
        if path.suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= cap:
                        break
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        elif path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                yield from data[:cap]
            elif isinstance(data, dict):
                sents = data.get("document", {}).get("sentences", data.get("sentences", []))
                yield from (sents[:cap] if isinstance(sents, list) else [])
    except Exception as e:  # fichier illisible : signaler, pas crasher
        yield {"_audit_error": str(e)}


def _cir_of(rec: dict):
    if not isinstance(rec, dict):
        return None
    cir = rec.get("cir")
    if isinstance(cir, dict) and ("nodes" in cir or "edges" in cir):
        return cir
    return None


def audit_file(path: Path, cap: int) -> dict:
    types, rels, methodes, statuts, confs = set(), set(), set(), set(), set()
    n = n_cir = n_intent = 0
    for rec in _iter_records(path, cap):
        n += 1
        if "_audit_error" in rec:
            return {"fichier": str(path), "erreur": rec["_audit_error"]}
        cir = _cir_of(rec)
        if cir is None:
            continue
        n_cir += 1
        if "intent" in cir:
            n_intent += 1
        for nd in cir.get("nodes", []) or []:
            if isinstance(nd, dict) and nd.get("type"):
                types.add(nd["type"])
        for e in cir.get("edges", []) or []:
            rel = e.get("relation") if isinstance(e, dict) else None
            if rel:
                rels.add(rel)
            attrs = e.get("attributes", {}) if isinstance(e, dict) else {}
            if "confidence" in attrs:
                c = attrs["confidence"]
                confs.add("null" if c is None else ("1.0" if c == 1.0 else "varied"))
        if rec.get("_methode"):
            methodes.add(rec["_methode"])
        if rec.get("statut"):
            statuts.add(rec["statut"])
    if types <= V4_TYPES and rels <= V4_RELS and not (types & V2_ALIASES):
        verdict = "v4"
    elif types & V2_ALIASES or not (rels <= V4_RELS):
        verdict = "v2-legacy/mixte"
    elif n_cir == 0:
        verdict = "sans-cir"
    else:
        verdict = "v4-partiel"
    return {"fichier": str(path), "n": n, "n_cir": n_cir, "types": sorted(types),
            "relations": sorted(rels), "intent": n_intent, "conf": sorted(confs),
            "methodes": sorted(methodes), "statuts": sorted(statuts), "verdict": verdict}


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("gcn-datasets/DATA/supervision")
    cap = int(sys.argv[sys.argv.index("--max-per-file") + 1]) if "--max-per-file" in sys.argv else 2000
    files = sorted(root.rglob("*.jsonl")) + sorted(root.rglob("*.json"))
    print(f"# Audit schéma — {len(files)} fichiers sous {root} (cap {cap}/fichier)")
    agg: dict[str, dict[str, int]] = {}
    for p in files:
        r = audit_file(p, cap)
        if "erreur" in r:
            print(f"ERREUR {r['fichier']}: {r['erreur']}")
            continue
        d = agg.setdefault(str(p.parent), {"fichiers": 0, "n": 0, "n_cir": 0})
        d["fichiers"] += 1
        d["n"] += r["n"]
        d["n_cir"] += r["n_cir"]
        print(f"{r['verdict']:>14} | n={r['n']:>5} cir={r['n_cir']:>5} "
              f"types={len(r['types'])} rels={len(r['relations'])} intent={r['intent']} "
              f"conf={r['conf']} meth={r['methodes']} | {p.name}")
    print("\n# Agrégats par répertoire")
    for d, v in sorted(agg.items()):
        print(f"{v['fichiers']:>3} fich. n={v['n']:>6} cir={v['n_cir']:>6} | {d}")


if __name__ == "__main__":
    main()
