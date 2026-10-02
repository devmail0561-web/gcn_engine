#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Gate §3 : accuracy du proposeur (balanced_auto) contre le gold, par relation.

--gold219 (défaut, OPPOSABLE) : gold manuel indépendant, 219 phrases FR/EN/ES.
   Seule mesure recevable : ces phrases n'ont jamais été proposées par le moteur.
Sans --gold219 : v4 (~1345, lots 01-19) — INFORMATIF SEULEMENT, auto-circulaire
   (lots 05+ proposés par le même moteur ; accord 0.90+ attendu, pas une preuve).

Pas de lattice ici : on juge le proposeur pur (règle métier), pas l'alignement.
Edgeless (_none) : le proposeur doit se taire — compté à part (faux positifs
silver), jamais dans le verdict (pas une relation).

Seuil : relation < 0.50 = REJETÉE du silver (quotas rebaissés en conséquence).
Sortie : tableau + gcn-datasets/DATA/supervision/v5/gates/accuracy[_219].json.
Exit 1 + liste des rejetées si gate non tenu.

Usage :
    python3 scripts/gate_accuracy.py [--gold219] [--seuil 0.5]
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gcn-tools" / "gcn-annotate" / "src"))
from gcn_annotate.balanced_auto import annotate  # noqa: E402

V4 = ROOT / "gcn-datasets" / "DATA" / "supervision" / "v4"
GATES = ROOT / "gcn-datasets" / "DATA" / "supervision" / "v5" / "gates"

GOLD219 = [
    ("gcn-datasets/DATA/data_annotees/fr/B_batch_fr_001.json", "fr"),
    ("gcn-datasets/DATA/data_annotees/en/B_batch_en_001_partial.json", "en"),
    ("gcn-datasets/DATA/data_annotees/es/B_batch_es_001_partial.json", "es"),
]


def gold_sentences():
    if "--gold219" not in sys.argv:
        out = []
        for f in sorted(V4.glob("annotated/lot*.json")):
            for s in json.load(open(f))["document"]["sentences"]:
                out.append((s, None))
        return out, "v4"
    out = []
    for rel, lang in GOLD219:
        for s in json.load(open(ROOT / rel)).get("document", {}).get("sentences", []):
            out.append((s, lang))
    return out, "gold219"


def main() -> None:
    seuil = float(sys.argv[sys.argv.index("--seuil") + 1]) if "--seuil" in sys.argv else 0.5
    sentences, tag = gold_sentences()
    ok_rel: Counter = Counter()
    tot_rel: Counter = Counter()
    n_missed, n_spurious = 0, 0
    for s, fixed_lang in sentences:
        es = s.get("cir", {}).get("edges", [])
        gold = es[0].get("relation", "_none") if es else "_none"
        langs = [fixed_lang] if fixed_lang else ["fr", "en"]
        p = None
        for lang in langs:
            p = annotate(s["text"], lang)
            if p and p.get("edges"):
                break
        if gold == "_none":
            if p and p.get("edges"):
                n_spurious += 1  # proposeur bruyant sur edgeless (silver douteux)
            continue
        tot_rel[gold] += 1
        if not p or not p.get("edges"):
            n_missed += 1
            continue
        if p["edges"][0].get("relation") == gold:
            ok_rel[gold] += 1
    GATES.mkdir(parents=True, exist_ok=True)
    table = {r: {"acc": round(ok_rel[r] / n, 3), "n": n}
             for r, n in sorted(tot_rel.items())}
    json.dump(table, open(GATES / f"accuracy_{tag}.json", "w"), ensure_ascii=False, indent=1)
    print(f"GOLD: {tag} | seuil: {seuil}")
    print(f"{'relation':22s} {'acc':>6s} {'n':>5s}  gate")
    rejected = []
    for r, v in sorted(table.items()):
        flag = "OK" if v["acc"] >= seuil else "REJET"
        if v["acc"] < seuil:
            rejected.append(r)
        print(f"{r:22s} {v['acc']:6.3f} {v['n']:5d}  {flag}")
    tot_ok, tot_n = sum(ok_rel.values()), sum(tot_rel.values())
    print(f"TOTAL: {tot_ok}/{tot_n} = {tot_ok / max(1, tot_n):.3f} | "
          f"ratées: {n_missed} | spurious-edgeless: {n_spurious}")
    if rejected:
        print("REJETÉES (< 0.50) :", ", ".join(rejected))
        sys.exit(1)
    print("GATE TENU : toutes les relations >= 0.50")


main()
