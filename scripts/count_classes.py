#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Comptes par classe SANS cap — effectifs réels pour R8/R10 (lot A).

Parcourt toute la supervision (351 Mo) : relations, types de noeuds,
sentence_types, registres, verdicts schéma. Sortie :
    gcn-datasets/DATA/supervision/comptes_classes.json + tableau stdout.
Lecture seule (aucune écriture hors le rapport).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

V4_TYPES = {"processus", "etat_local", "etat_global", "entite",
            "condition", "concept", "evenement", "contrainte"}
V2_ALIASES = {"etat", "etat_systemique", "action", "transition"}
V4_RELS = {"cause", "enable", "prevent", "condition", "concession", "sequence",
           "motivation", "filter", "opposition", "data_dependency",
           "control_dependency", "analogy", "counterfactual",
           "conditional_cause", "mediated_cause", "joint_cause",
           "conditional_prevent", "mediated_prevent", "joint_prevent"}


def iter_records(path: Path):
    try:
        if path.suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        elif path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                yield from data
            elif isinstance(data, dict):
                sents = data.get("document", {}).get("sentences", data.get("sentences", []))
                if isinstance(sents, list):
                    yield from sents
    except (OSError, ValueError, UnicodeDecodeError) as e:
        yield {"_audit_error": str(e)}


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("gcn-datasets/DATA/supervision")
    rels, types, stypes, regs, verdicts = Counter(), Counter(), Counter(), Counter(), Counter()
    n_phr = n_cir = n_files = n_err = 0
    for p in sorted(root.rglob("*.jsonl")) + sorted(root.rglob("*.json")):
        n_files += 1
        for rec in iter_records(p):
            if "_audit_error" in rec:
                n_err += 1
                break
            if not isinstance(rec, dict):
                continue
            n_phr += 1
            cir = rec.get("cir")
            if not (isinstance(cir, dict) and ("nodes" in cir or "edges" in cir)):
                verdicts["sans-cir"] += 1
                continue
            n_cir += 1
            t = {nd.get("type") for nd in cir.get("nodes", []) or []
                 if isinstance(nd, dict) and nd.get("type")}
            r = {e.get("relation") for e in cir.get("edges", []) or []
                 if isinstance(e, dict) and e.get("relation")}
            types.update(t)
            rels.update(r)
            stypes[rec.get("sentence_type", "?")] += 1
            regs[rec.get("registre", "?")] += 1
            if t <= V4_TYPES and r <= V4_RELS and not (t & V2_ALIASES):
                verdicts["v4"] += 1
            elif (t & V2_ALIASES) or not (r <= V4_RELS):
                verdicts["v2-legacy/mixte"] += 1
            else:
                verdicts["v4-partiel"] += 1
    report = {"fichiers": n_files, "phrases": n_phr, "avec_cir": n_cir,
              "erreurs_lecture": n_err, "verdicts": dict(verdicts),
              "relations": dict(sorted(rels.items(), key=lambda kv: -kv[1])),
              "types_noeuds": dict(sorted(types.items(), key=lambda kv: -kv[1])),
              "sentence_types": dict(stypes), "registres": dict(regs)}
    out = root / "comptes_classes.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"fichiers={n_files} phrases={n_phr} avec_cir={n_cir} erreurs={n_err}")
    print(f"verdicts={dict(verdicts)}")
    print("relations (effectifs) :")
    for k, v in sorted(rels.items(), key=lambda kv: -kv[1]):
        flag = " <30 !" if v < 30 else ""
        print(f"  {k:>20} {v:>7}{flag}")
    print("types noeuds :", dict(sorted(types.items(), key=lambda kv: -kv[1])))
    print("sentence_types :", dict(stypes))
    print(f"rapport : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
