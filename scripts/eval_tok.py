# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Éval best-modèle sur test/holdout (protocole §4, §7) + calibration gatée.

Charge un `.npz` s1+ best-restore, évalue F1 sur test, collecte les logits
arêtes pour ECE + `require_production_sample(N>=100)` : N=99 → NO-GO
documenté, pas de chiffre publié comme calibré.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from gcn_python.constants import (
    COARSE_NODE_TYPES,
    COARSE_RELATION_TYPES,
    FINE_TO_COARSE_NODE,
    FINE_TO_COARSE_RELATION,
    NODE_TYPES,
    RELATION_TYPES,
)
from gcn_python.evaluation.calibration import (
    ece_score,
    require_production_sample,
    softmax,
)
from gcn_python.pipeline.forward_ids import TokForwardEngine
from gcn_python.training.refacto_loss import build_class_mask
from gcn_python.training.train_tok import charger_split, evaluer


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Éval test + calibration gatée.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True, help="train.json (masques R8)")
    ap.add_argument("--test", required=True)
    ap.add_argument("--espace", choices=["fin", "coarse"], default="coarse")
    args = ap.parse_args(argv)

    eng = TokForwardEngine.load_npz(args.model)
    noms_n = list(COARSE_NODE_TYPES) if args.espace == "coarse" else list(NODE_TYPES)
    noms_r = list(COARSE_RELATION_TYPES) if args.espace == "coarse" else list(RELATION_TYPES)
    tr, _ = charger_split(args.train)
    te, st = charger_split(args.test)

    def _remap(samples):
        if args.espace != "coarse":
            return
        for s in samples:
            s["node_gold"] = [
                noms_n.index(FINE_TO_COARSE_NODE[NODE_TYPES[g]]) for g in s["node_gold"]
            ]
            s["edge_gold"] = {
                p: noms_r.index(FINE_TO_COARSE_RELATION[RELATION_TYPES[g]])
                for p, g in s["edge_gold"].items()
            }

    _remap(tr)
    _remap(te)
    nc = np.zeros(len(noms_n), dtype=np.int64)
    ec = np.zeros(len(noms_r), dtype=np.int64)
    for s in tr:
        for g in s["node_gold"]:
            nc[g] += 1
        for g in s["edge_gold"].values():
            ec[g] += 1
    met = evaluer(
        eng, te, build_class_mask(nc), build_class_mask(ec),
        node_types=noms_n, relation_types=noms_r,
    )
    print(json.dumps({"test_" + k: v for k, v in met.items()}, ensure_ascii=False))

    # Calibration : logits arêtes collectés, gate N>=100.
    all_logits, all_gold = [], []
    for s in te:
        parts = eng.forward_supervise(
            s["text"], s["ids"], s["offsets"], s["clause_spans"], s["clause_textes"]
        )
        for k, p in enumerate(parts["paires"]):
            if p in s["edge_gold"]:
                all_logits.append(np.asarray(parts["edge_logits"][k], dtype=float))
                all_gold.append(int(s["edge_gold"][p]))
    if all_logits:
        mat = np.stack(all_logits)
        lab = np.array(all_gold, dtype=int)
        try:
            require_production_sample(len(lab))
            print(json.dumps({"ece": ece_score(softmax(mat), lab), "n": len(lab)}))
        except ValueError as exc:
            print(json.dumps({"ece": "NO-GO", "motif": str(exc), "n": len(lab)}))
    else:
        print(json.dumps({"ece": "NO-GO", "motif": "aucune arête supervisée", "n": 0}))
    print(json.dumps({"rejets_test": st["rejetees"], "n_test": len(te)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
