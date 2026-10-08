#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Grid F1 des seuils sur jouet (étape 5, [R10]) — démo en attendant gold-3000.

Entraîne le jouet slice (4 phrases, 300 steps, seed 42), collecte les logits
arêtes SANS oracle sur les 81 autres phrases gold_mini, puis grid
(edge_threshold × theta_ambiguity × temperature) maximisant le micro-F1.

Gate ([R8]/[R10]) : effectifs < 30/classe sur le gold → GEL (sortie 2, aucun
seuil adopté). ``--force`` exécute quand même le grid (DÉMO : seuils non
adoptés sous gel — le vrai grid attend gold-3000).

Sortie : JSON {"verdict": "GEL"|"GO", ...} sur stdout.
Usage :
    PYTHONPATH=gcn-python/src python3 scripts/grid_f1_thresholds.py [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gcn-python" / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import slice_minus1_gate as S
from gcn_python.constants import RELATION_TYPES
from gcn_python.data.loader import GCNDataLoader, reps_from_sentence
from gcn_python.evaluation.grid_f1 import REF_THRESHOLDS, run_grid
from gcn_python.training.refacto_loss import GelError, require_min_counts

SLICE_IDS = ["s0555", "s0801", "f0005", "s2028"]


def charger_val():
    """81 phrases gold_mini hors slice (même conversion que le slice)."""
    import ast

    src = REPO / "gcn-datasets" / "DATA" / "supervision" / "gold_mini" / "gold_mini.jsonl"
    sents = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o["id"] in SLICE_IDS:
                continue
            cir = o["cir"]
            if isinstance(cir, str):
                try:
                    cir = json.loads(cir)
                except json.JSONDecodeError:
                    cir = ast.literal_eval(cir)
            nodes = [{"id": n["id"], "type": n["type"], "label": n.get("label", ""),
                      "token_span": n.get("token_span")} for n in cir["nodes"]]
            edges = [{"sources": (e.get("sources") or [e.get("source")]), "target": e.get("target"),
                      "relation": e.get("relation"), "confidence": e.get("confidence", 1.0),
                      "explicit": True, "negated": False} for e in cir["edges"]]
            sents.append({"id": o["id"], "text": o["text"], "tokens": o["tokens"],
                          "cir": {"nodes": nodes, "edges": edges}})
    td = tempfile.mkdtemp(prefix="grid_f1_val_")
    Path(td, "train.json").write_text(json.dumps({"document": {"lang": "fr", "sentences": sents}},
                                                 ensure_ascii=False), encoding="utf-8")
    return list(GCNDataLoader(Path(td), shuffle=False))


def collecter_logits(pipe, samples):
    """Logits arêtes SANS oracle + gold (paires gold uniquement)."""
    import numpy as _np

    tous_l, tous_g = [], []
    for s in samples:
        reps, valid, conn = reps_from_sentence(s.sentence)
        pipe.forward(reps, s.sentence.text, clause_positions=valid,
                     n_total_clauses=len(s.sentence.clauses), connector_reps=conn)
        el = _np.asarray(pipe._cached_edge_logits)
        pairs = [(valid[i], valid[j]) for i in range(len(valid)) for j in range(i + 1, len(valid))]
        gf = _np.array([s.edge_map.get(p, -1) for p in pairs], dtype=_np.int64)
        m = gf >= 0
        if m.any():
            tous_l.append(el[m])
            tous_g.append(gf[m])
    return _np.concatenate(tous_l), _np.concatenate(tous_g)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Grid F1 des seuils (jouet, R10).")
    ap.add_argument("--force", action="store_true",
                    help="exécute le grid malgré le gel (DÉMO : seuils non adoptés)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    train = S.load_slice_samples()
    val = charger_val()
    comptes = [0] * len(RELATION_TYPES)
    for s in val:
        for r in s.edge_map.values():
            comptes[int(r)] += 1
    try:
        require_min_counts([c for c in comptes if c > 0], nom="gold-val")
        gele = None
    except GelError as exc:
        gele = str(exc)
    if gele is not None and not args.force:
        print(json.dumps({"verdict": "GEL", "motif": gele,
                          "effectifs_relations": {RELATION_TYPES[i]: c
                                                  for i, c in enumerate(comptes) if c > 0}},
                         ensure_ascii=False, indent=1))
        return 2
    pipe, _ = S.build_pipeline(args.seed, train)
    S.train_gold.lr = 0.05
    rng = np.random.default_rng(args.seed)
    ordre = list(range(len(train)))
    for step in range(args.steps):
        if step % len(train) == 0:
            ordre = list(rng.permutation(len(train)))
        S.train_gold(pipe, train[ordre[step % len(train)]])
    logits, gold = collecter_logits(pipe, val)
    best = run_grid(logits, gold)
    print(json.dumps({"verdict": "GO-DÉMO" + (" (sous gel, non adopté)" if gele else ""),
                      "gele": gele,
                      "n_arêtes": int(gold.shape[0]),
                      "meilleur": best,
                      "reference_interdite": list(REF_THRESHOLDS)},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
