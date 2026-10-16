# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""O1 — pré-entraînement brut sans labels ni lexique (CPU-only).

Skip-gram sur ids stables : contexte = moyenne des voisins (±2, hors cible),
prédit l'id masqué via logits = E @ ctx. Zéro liste de langue, zéro UD/gold.
Sortie : init E_tok `.npz` s1 chargeable par `train_tok --init-npz`.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from gcn_python.layer0.tokenizer import split as tok_split
from gcn_python.layer1.tok_embedding import TokEmbedding


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x -= x.max()
    e = np.exp(x)
    return e / e.sum()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="O1 skip-gram brut (CPU, sans lexique).")
    ap.add_argument("--train", required=True, help="train.json (document.sentences[].text)")
    ap.add_argument("--out", required=True, help="init .npz s1 de sortie")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window", type=int, default=2)
    args = ap.parse_args(argv)

    with open(args.train, encoding="utf-8") as fh:
        doc = json.load(fh)["document"]
    texts = [s.get("text", "") for s in doc.get("sentences", []) if s.get("text")]
    if not texts:
        print("O1 : aucun texte — STOP (données).", flush=True)
        return 1

    ids_list = [tok_split(t)[0] for t in texts]
    emb = TokEmbedding(d_model=args.d_model, seed=args.seed)
    for ids in ids_list:
        emb.build_vocab(ids)
    V = emb.vocab_size
    print(f"O1 : {len(texts)} phrases, vocab {V}, d={args.d_model}.", flush=True)

    rng = np.random.default_rng(args.seed)
    for ep in range(int(args.epochs)):
        ordre = rng.permutation(len(ids_list))
        perte = 0.0
        n = 0
        for k in ordre:
            ids = ids_list[int(k)]
            if len(ids) < 3:
                continue
            E = emb._E
            for pos in range(len(ids)):
                tgt = emb.row_index(ids[pos])
                if tgt is None:
                    continue
                ctx_idx = []
                for d in range(-args.window, args.window + 1):
                    if d == 0:
                        continue
                    q = pos + d
                    if 0 <= q < len(ids):
                        j = emb.row_index(ids[q])
                        if j is not None:
                            ctx_idx.append(j)
                if not ctx_idx:
                    continue
                ctx = E[ctx_idx].mean(axis=0)
                logits = E @ ctx
                probs = _softmax(logits)
                perte += float(-np.log(max(probs[tgt], 1e-12)))
                n += 1
                grad_logits = probs
                grad_logits[tgt] -= 1.0
                # dE[tgt] via ctx, dE[ctx] via logits (poids partagés, SGD direct)
                E[tgt] -= args.lr * float(grad_logits[tgt]) * ctx
                g_ctx = E.T @ grad_logits / len(ctx_idx)
                for j in ctx_idx:
                    E[j] -= args.lr * g_ctx
        print(f"O1 ep{ep} : perte_moy={perte / max(1, n):.4f} sur {n} cibles.", flush=True)
    emb.save_npz(args.out)
    print(f"O1 : init écrite → {args.out} (V={V}).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
