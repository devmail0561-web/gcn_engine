# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""O1-MLM — pré-entraînement span-masqué sur brut (CPU-only, sans lexique).

Masque des spans (15%), encode le contexte (E_tok + ContextEncoder), prédit
les ids masqués par logits = H @ E.T (partage de poids — 0 param de tête).
Zéro liste de langue, zéro UD/gold. Sortie : init moteur `.npz`
(E + encodeur) chargeable pour l'aval.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from gcn_python.layer0.tokenizer import split as tok_split
from gcn_python.layer1.context import ContextEncoder
from gcn_python.layer1.tok_embedding import TokEmbedding
from gcn_python.pretrain.masking import span_mask

FORMAT_O1 = 20261014


def _collect_texts(pools: list[str], max_phrases: int) -> list[str]:
    texts: list[str] = []
    for pool in pools:
        with open(pool, encoding="utf-8") as fh:
            doc = json.load(fh)["document"]
        for s in doc.get("sentences", []):
            if s.get("text"):
                texts.append(s["text"])
            if len(texts) >= max_phrases:
                return texts
    return texts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="O1-MLM brut (CPU, sans lexique).")
    ap.add_argument("--pools", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-phrases", type=int, default=2500)
    args = ap.parse_args(argv)

    texts = _collect_texts(args.pools, args.max_phrases)
    if not texts:
        print("O1-MLM : aucun texte — STOP (données).", flush=True)
        return 1
    ids_list = [tok_split(t)[0] for t in texts]
    emb = TokEmbedding(d_model=args.d_model, seed=args.seed)
    for ids in ids_list:
        emb.build_vocab(ids)
    enc = ContextEncoder(d_model=args.d_model, seed=args.seed + 10)
    V = emb.vocab_size
    print(f"O1-MLM : {len(texts)} phrases, vocab {V}, ctx {enc.n_params} params.", flush=True)

    rng = np.random.default_rng(args.seed)
    for ep in range(int(args.epochs)):
        perte, n = 0.0, 0
        for k in rng.permutation(len(ids_list)):
            ids = ids_list[int(k)]
            if len(ids) < 4:
                continue
            masked, pos = span_mask(ids, seed=int(rng.integers(0, 2**31)))
            if not pos:
                continue
            H = enc.forward(emb.lookup_batch(masked))
            E = emb._E
            logits = H[pos] @ E.T
            m = logits.max(axis=1, keepdims=True)
            e = np.exp(logits - m)
            probs = e / e.sum(axis=1, keepdims=True)
            tgt = np.array([emb.row_index(ids[p]) for p in pos])
            perte += float(-np.log(np.maximum(probs[np.arange(len(pos)), tgt], 1e-12)).sum())
            n += len(pos)
            dlog = probs
            dlog[np.arange(len(pos)), tgt] -= 1.0
            dlog /= len(pos)
            # Partage de poids : dH[pos_r,:] = dlog[r,:] @ E ;
            # dE[t,:] += Σ_{r: tgt_r=t} dlog[r,t] · H[pos_r,:].
            dH = np.zeros_like(H)
            dH[pos] = dlog @ E
            Hpos = H[pos]
            for r_idx, t in enumerate(tgt.tolist()):
                E[int(t)] -= args.lr * (float(dlog[r_idx, int(t)]) * Hpos[r_idx])
            dR, grads = enc.backward(dH)
            enc.update(grads, args.lr)
            emb.backward(dR, masked)
            emb.update(args.lr)
        print(f"O1-MLM ep{ep} : ppl={np.exp(perte / max(1, n)):.1f} sur {n} cibles.", flush=True)
    np.savez(
        args.out,
        E=emb._E,
        tok_vocab=np.asarray(emb.tok_vocab, dtype=np.uint64),
        d_model=np.asarray(args.d_model, dtype=np.int64),
        format_o1=np.asarray(FORMAT_O1, dtype=np.int64),
        ctx_W1=enc.bloc1.W,
        ctx_b1=enc.bloc1.b,
        ctx_W2=enc.bloc2.W,
        ctx_b2=enc.bloc2.b,
    )
    print(f"O1-MLM : init écrite → {args.out}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
