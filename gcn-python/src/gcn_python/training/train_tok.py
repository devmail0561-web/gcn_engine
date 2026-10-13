# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Entraînement tok (E_tok + têtes) sur paires gold — module neuf, legacy intact.

Supervision gold pure, SANS socle B au train : formes UD → char-spans
(``data/gold_spans.py``, validés, rejet explicite) → split → align → pool
→ têtes → ``refacto_loss`` (masques R8, ``GelError``) → backward chaîné
→ SGD → éval best-unique + CSV provenance (protocole §1-9).

Contrat (même que le legacy) : forward phrase → loss → backward → update
(caches par tête — backward AVANT tout autre forward). Paires gold hors
fenêtre k=2 ou arrières : COMPTÉES + rapportées, jamais supervisées
silencieusement (R2 : assumée perdue ; support arrière = étape ultérieure).
Silver exclu (weight 0.0 — R9, non négociable à ce pas).
"""

from __future__ import annotations

import csv
import hashlib
import json
import warnings

import numpy as np

from ..constants import NODE_TYPES, RELATION_TYPES
from ..data.gold_spans import SpanError, reconstruire_offsets, span_clause
from ..evaluation.metrics import edge_macro_f1, node_macro_f1
from ..layer0.tokenizer import split as tok_split
from ..pipeline.forward_ids import OOV_WARN_RATE, WINDOW_K, TokForwardEngine
from ..training.refacto_loss import GelError, build_class_mask, ce_loss, require_min_counts

#: Seuil STOP d'alignement gold (revue training-tok) : au-delà = données, pas acharnement.
SEUIL_REJET = 0.05


def _ce_backward(
    logits: np.ndarray,
    gold: np.ndarray,
    class_mask: np.ndarray | None = None,
    weight: float = 1.0,
) -> np.ndarray:
    """Gradient de ``ce_loss`` (mêmes masques, même normalisation).

    Cohérence exigée : ``ce_loss`` moyenne sur les lignes gardées × weight —
    ici ``(p - onehot) / n_gardées × weight``, zéros hors garde. Lignes
    gardées vides → zéros (``ce_loss`` retourne 0.0). Parité testée par
    finite-diff (gate §6[G]).
    """
    mat = np.asarray(logits, dtype=np.float64)
    refs = np.asarray(gold, dtype=np.int64).ravel()
    if mat.ndim != 2 or mat.shape[0] != refs.shape[0]:
        raise ValueError(f"logits {mat.shape} vs gold {refs.shape}.")
    if ((refs < 0) | (refs >= mat.shape[1])).any():
        raise ValueError("labels gold hors vocabulaire.")
    garde = np.ones(mat.shape[0], dtype=bool)
    if class_mask is not None:
        masque = np.asarray(class_mask, dtype=bool).ravel()
        if masque.shape[0] != mat.shape[1]:
            raise ValueError("class_mask de mauvaise taille.")
        garde &= masque[refs]
    d = np.zeros_like(mat)
    if not garde.any():
        return d.astype(np.float32)
    m = mat[garde] - mat[garde].max(axis=1, keepdims=True)
    e = np.exp(m)
    p = e / e.sum(axis=1, keepdims=True)
    g = np.zeros_like(m)
    g[np.arange(m.shape[0]), refs[garde]] = 1.0
    d[garde] = (p - g) / garde.sum() * weight
    return d.astype(np.float32)


def _softmax_backward(probs: np.ndarray, d_probs: np.ndarray) -> np.ndarray:
    """Jacobien softmax transposé : ``d_z = (diag(p) - p pᵀ) d_p`` par ligne.

    Accepte 1-D (un vecteur) ou 2-D (lignes) — même sortie que l'entrée.
    """
    uni = np.asarray(probs).ndim == 1
    p = np.atleast_2d(np.asarray(probs, dtype=np.float64))
    dp = np.atleast_2d(np.asarray(d_probs, dtype=np.float64))
    if p.shape != dp.shape:
        raise ValueError(f"probs {p.shape} vs d_probs {dp.shape}.")
    s = (dp * p).sum(axis=1, keepdims=True)
    out = (p * (dp - s)).astype(np.float32)
    return out[0] if uni else out


def charger_split(fichier: str) -> tuple[list, dict]:
    """Split JSON gold → samples tok + stats d'alignement.

    Sample : ``{text, ids, offsets, clause_spans, clause_textes,
    node_gold: [idx], edge_gold: {(i,j): rel_idx}}``. Phrases non
    alignables : rejetées (comptées ``rejetees``). Paires gold hors
    fenêtre ou arrières : exclues (comptées ``hors_fenetre`` /
    ``arrieres``) — rapportées, jamais silencieuses.
    """
    with open(fichier, encoding="utf-8") as fh:
        doc = json.load(fh)
    samples = []
    stats = {"phrases": 0, "rejetees": [], "hors_fenetre": 0, "arrieres": 0}
    for s in doc["document"]["sentences"]:
        stats["phrases"] += 1
        ident = s.get("id", f"#{stats['phrases']}")
        try:
            offs = reconstruire_offsets(s["text"], s["tokens"])
        except SpanError as exc:
            stats["rejetees"].append((ident, str(exc)))
            continue
        ntoks = len(s["tokens"])
        try:
            rangs = {n["id"]: k for k, n in enumerate(s["cir"]["nodes"])}
            cspans, ctextes, ngold = [], [], []
            for n in s["cir"]["nodes"]:
                a, b = n["token_span"]
                cspans.append(list(span_clause(offs, a, b, ntoks)))
                ctextes.append(s["text"][cspans[-1][0] : cspans[-1][1]])
                try:
                    ngold.append(NODE_TYPES.index(n["type"]))
                except ValueError:
                    raise SpanError(f"type de nœud inconnu : {n['type']!r}.") from None
            egold = {}
            for e in s["cir"]["edges"]:
                srcs = e.get("sources") or ([e["source"]] if "source" in e else [])
                for src in srcs:
                    i, j = rangs[src], rangs[e["target"]]
                    try:
                        r = RELATION_TYPES.index(e["relation"])
                    except ValueError:
                        raise SpanError(f"relation inconnue : {e['relation']!r}.") from None
                    if i >= j:
                        stats["arrieres"] += 1
                        continue
                    if j - i > WINDOW_K:
                        stats["hors_fenetre"] += 1
                        continue
                    egold[(i, j)] = r
        except SpanError as exc:
            stats["rejetees"].append((ident, str(exc)))
            continue
        ids, offsets, _ = tok_split(s["text"])
        samples.append(
            {
                "id": ident,
                "text": s["text"],
                "ids": ids,
                "offsets": offsets,
                "clause_spans": cspans,
                "clause_textes": ctextes,
                "node_gold": ngold,
                "edge_gold": egold,
            }
        )
    return samples, stats


def backward_tok(
    eng: TokForwardEngine, parts: dict, node_gold: list[int], edge_gold: dict, lr: float
) -> dict:
    """Un pas SGD phrase (caches chauds de ``forward_supervise``).

    Chaîne : d_edge → MLP.backward_edge_dx → split (base→pooled,
    enriched→clauses, probs→softmax⁻¹→nœuds) → d_node total →
    MLP.backward_node_dx → RGCN.backward → pool-mean⁻¹ → E_tok.backward
    → updates (bump epoch M1a partout). Paires sans gold : aucun gradient
    (exclues, pas masquées à 0 — le 0 biaiserait la CE).
    Retourne ``{"node": loss, "edge": loss, "n_edge_sup": int}``.
    """
    if not (np.isfinite(lr) and lr > 0):
        raise ValueError(f"lr doit être > 0 et fini (reçu {lr!r}).")
    paires = parts["paires"]
    sup_idx = [k for k, p in enumerate(paires) if p in edge_gold]
    sup_gold = np.array([edge_gold[paires[k]] for k in sup_idx], dtype=np.int64)
    edge_logits = parts["edge_logits"]
    mat_edge = np.stack(edge_logits, axis=0) if edge_logits else np.zeros((0, len(RELATION_TYPES)))
    d_edge_full = np.zeros_like(mat_edge, dtype=np.float32)
    loss_edge = 0.0
    if sup_idx:
        sub = mat_edge[sup_idx]
        loss_edge = ce_loss(sub, sup_gold, class_mask=parts.get("edge_mask"))
        d_edge_full[sup_idx] = _ce_backward(sub, sup_gold, parts.get("edge_mask"))
    node_logits = np.asarray(parts["node_logits"], dtype=np.float32)
    node_gold_a = np.asarray(node_gold, dtype=np.int64)
    loss_node = ce_loss(node_logits, node_gold_a, class_mask=parts.get("node_mask"))
    d_node = _ce_backward(node_logits, node_gold_a, parts.get("node_mask"))

    n = len(node_gold)
    d_base = parts["d_base"]
    d_pool = parts["d_pooled"]
    d_enriched = np.zeros((n, d_pool), dtype=np.float32)
    d_pooled = np.zeros((n, d_pool), dtype=np.float32)
    d_node_extra = np.zeros_like(node_logits, dtype=np.float64)
    grads_edge = None
    # Arêtes : re-run déterministe (dropout 0 = cache exact) + backward.
    for k, (i, j) in enumerate(paires):
        eng.encoder.forward_edge(parts["edge_vecs"][k])
        row = d_edge_full[k]
        if not np.abs(row).any():
            continue
        grads_e, dx = eng.encoder.backward_edge_dx(row)
        grads_edge = (
            grads_e
            if grads_edge is None
            else [(a[0] + b[0], a[1] + b[1]) for a, b in zip(grads_edge, grads_e, strict=True)]
        )
        d_pooled[i] += dx[:d_pool]
        d_pooled[j] += dx[d_pool : 2 * d_pool]
        d_enriched[i] += dx[d_base : d_base + d_pool]
        d_enriched[j] += dx[d_base + d_pool : d_base + 2 * d_pool]
        nn = parts["n_node_types"]
        d_node_extra[i] += _softmax_backward(
            parts["node_probs"][i], dx[d_base + 2 * d_pool : d_base + 2 * d_pool + nn]
        )
        d_node_extra[j] += _softmax_backward(parts["node_probs"][j], dx[d_base + 2 * d_pool + nn :])
    # Nœuds : gradient total (CE + reflux probs).
    d_node_total = (d_node + d_node_extra).astype(np.float32)
    # Garde non-finis (miroir legacy) : warn + pas de mise à jour.
    for nom, arr in (("noeuds", d_node_total), ("aretes", d_edge_full)):
        if not np.all(np.isfinite(arr)):
            warnings.warn(
                f"backward_tok : gradients {nom} non finis — mise à jour annulée.",
                UserWarning,
                stacklevel=2,
            )
            return {"node": loss_node, "edge": loss_edge, "n_edge_sup": len(sup_idx)}
    grads_node = None
    for i in range(n):
        eng.encoder.forward_node(parts["pooled"][i])
        grads_n, dx = eng.encoder.backward_node_dx(d_node_total[i])
        grads_node = (
            grads_n
            if grads_node is None
            else [(a[0] + b[0], a[1] + b[1]) for a, b in zip(grads_node, grads_n, strict=True)]
        )
        d_pooled[i] += dx
    # R-GCN (sauté à 1 clause — jamais appelé au forward).
    if n > 1 and paires:
        d_rgcn, grads_r = eng.rgcn.backward_message_pass(d_enriched.astype(np.float32))
        d_pooled += d_rgcn
    else:
        grads_r = None
    # Pool-mean⁻¹ via les alignements + E_tok.
    d_tok = np.zeros((len(parts["ids"]), d_pool), dtype=np.float32)
    for c, membres in enumerate(parts["alignes"]):
        if membres:
            d_tok[membres] += (d_pooled[c] / len(membres)).astype(np.float32)
    eng.emb.backward(d_tok, parts["ids"])
    # Updates (bump epoch M1a dans chaque update — sens sûr).
    if grads_node is not None:
        eng.encoder.update_node(grads_node, lr)
    if grads_edge is not None:
        eng.encoder.update_edge(grads_edge, lr)
    if grads_r is not None:
        eng.rgcn.update(grads_r, lr)
    eng.emb.update(lr)
    return {"node": loss_node, "edge": loss_edge, "n_edge_sup": len(sup_idx)}


def _git_sha() -> str:
    """SHA court du workspace (provenance CSV — même pattern que train_cmd)."""
    try:
        import shutil as _shutil
        import subprocess as _sp

        binaire = _shutil.which("git")
        if binaire is None:
            return "?"
        out = _sp.run(
            [binaire, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or "?"
    except Exception:  # noqa: BLE001 — provenance best-effort, jamais bloquante
        return "?"


def _hash_fichiers(fichiers: list[str]) -> str:
    h = hashlib.sha256()
    for f in fichiers:
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def _parametres(eng: TokForwardEngine) -> dict:
    """Snapshot copié (E + têtes MLP + R-GCN) pour best-restore."""
    snap = {"E": eng.emb._E.copy(), "vocab": list(eng.emb.tok_vocab)}
    snap["node"] = [(c.W.copy(), c.b.copy()) for c in eng.encoder._node_layers]
    snap["edge"] = [(c.W.copy(), c.b.copy()) for c in eng.encoder._edge_layers]
    snap["rgcn"] = [eng.rgcn.W_r.copy(), eng.rgcn.W_0.copy()]
    return snap


def _restaurer(eng: TokForwardEngine, snap: dict) -> None:
    """Restaure un snapshot (formes validées — fail-fast)."""
    if eng.emb._E.shape != snap["E"].shape:
        raise ValueError("snapshot incompatible (E).")
    eng.emb._E[:] = snap["E"]
    for (w, b), couche in zip(snap["node"], eng.encoder._node_layers, strict=True):
        if couche.W.shape != w.shape:
            raise ValueError("snapshot incompatible (nœuds).")
        couche.W[:] = w
        couche.b[:] = b
    for (w, b), couche in zip(snap["edge"], eng.encoder._edge_layers, strict=True):
        if couche.W.shape != w.shape:
            raise ValueError("snapshot incompatible (arêtes).")
        couche.W[:] = w
        couche.b[:] = b
    if eng.rgcn.W_r.shape != snap["rgcn"][0].shape:
        raise ValueError("snapshot incompatible (R-GCN).")
    eng.rgcn.W_r[:] = snap["rgcn"][0]
    eng.rgcn.W_0[:] = snap["rgcn"][1]


def evaluer(eng: TokForwardEngine, samples: list, node_mask=None, edge_mask=None) -> dict:
    """Passe éval (sans grad) : argmax → F1 macro (NaN si vide, jamais 0.0).

    Arêtes : paires SUPERVISÉES seulement (ce qu'on a demandé au modèle).
    """
    pred_n, gold_n, pred_e, gold_e = [], [], [], []
    perte = 0.0
    for s in samples:
        parts = eng.forward_supervise(
            s["text"], s["ids"], s["offsets"], s["clause_spans"], s["clause_textes"]
        )
        node_gold = np.asarray(s["node_gold"], dtype=np.int64)
        perte += ce_loss(np.asarray(parts["node_logits"]), node_gold, class_mask=node_mask)
        pred_n += [NODE_TYPES[int(i)] for i in np.argmax(parts["node_logits"], axis=1)]
        gold_n += [NODE_TYPES[int(i)] for i in node_gold]
        mat = np.stack(parts["edge_logits"], axis=0) if parts["edge_logits"] else None
        for k, p in enumerate(parts["paires"]):
            if p not in s["edge_gold"]:
                continue
            g = s["edge_gold"][p]
            perte += ce_loss(mat[k : k + 1], np.asarray([g]), class_mask=edge_mask)
            pred_e.append(RELATION_TYPES[int(np.argmax(mat[k]))])
            gold_e.append(RELATION_TYPES[int(g)])
    return {
        "loss": perte / max(1, len(samples)),
        "node_f1": node_macro_f1(pred_n, gold_n),
        "edge_f1": edge_macro_f1(pred_e, gold_e),
        "n_node": len(gold_n),
        "n_edge": len(gold_e),
    }


def entrainer(
    train_path: str,
    val_path: str,
    *,
    epochs: int = 20,
    lr: float = 0.001,
    seed: int = 42,
    patience: int = 10,
    d_model: int = 128,
    csv_out: str | None = None,
) -> dict:
    """Boucle SGD tok (ordre fixe, seedée — reproductible, protocole §5).

    1. Charge splits (rejet > 5 % → ValueError STOP).
    2. Vocabulaire sur train (+sanity OOV train = 0) ; OOV val publié + warn
       (écart documenté à R5 : la val mesure, elle n'avorte pas).
    3. Masques R8 + ``require_min_counts`` (``GelError`` fail-fast).
    4. Epochs : forward_supervise → backward_tok → update par phrase ;
       éval val ; best sur ``val_edge_macro_f1`` (NaN jamais best) ;
       patience puis restore best (early-stopping unifié edge, §6).
    Retourne le résumé (le moteur porte les poids best).
    """
    if epochs < 1:
        raise ValueError(f"epochs doit être ≥ 1 (reçu {epochs!r}).")
    train, st_tr = charger_split(train_path)
    val, st_val = charger_split(val_path)
    for nom, stats in (("train", st_tr), ("val", st_val)):
        taux = len(stats["rejetees"]) / max(1, stats["phrases"])
        if taux > SEUIL_REJET:
            raise ValueError(f"{nom} : rejet alignement {taux:.1%} > 5 % — STOP (données).")
    eng = TokForwardEngine(d_model=d_model, seed=seed)
    eng.build_vocab([s["text"] for s in train])
    oov_tr = sum(1 for s in train for i in s["ids"] if eng.emb.row_index(i) is None)
    tot_tr = sum(len(s["ids"]) for s in train)
    if tot_tr and oov_tr / tot_tr > OOV_WARN_RATE:
        raise ValueError(f"OOV train {oov_tr / tot_tr:.1%} > 5 % — vocabulaire défectueux.")
    oov_val = sum(1 for s in val for i in s["ids"] if eng.emb.row_index(i) is None)
    tot_val = sum(len(s["ids"]) for s in val) or 1
    taux_oov_val = oov_val / tot_val
    if taux_oov_val > OOV_WARN_RATE:
        warnings.warn(
            f"train_tok : OOV val {taux_oov_val:.1%} > 5 % — publié, pas avorté "
            "(la val mesure ; écart documenté à R5).",
            UserWarning,
            stacklevel=2,
        )
    # Masques + gate R8 (fail-fast — GelError = sortie écrite, pas acharnement).
    n_counts = np.zeros(len(NODE_TYPES), dtype=np.int64)
    e_counts = np.zeros(len(RELATION_TYPES), dtype=np.int64)
    for s in train:
        for g in s["node_gold"]:
            n_counts[g] += 1
        for g in s["edge_gold"].values():
            e_counts[g] += 1
    require_min_counts(n_counts, nom="nœuds")
    require_min_counts(e_counts, nom="relations")
    node_mask = build_class_mask(n_counts)
    edge_mask = build_class_mask(e_counts)
    sha_donnees = _hash_fichiers([train_path, val_path])
    lignes = []
    best_f1 = float("-inf")
    best_snap = None
    best_ep = -1
    attente = 0
    for ep in range(epochs):
        perte_tr = 0.0
        for s in train:
            parts = eng.forward_supervise(
                s["text"], s["ids"], s["offsets"], s["clause_spans"], s["clause_textes"]
            )
            parts["node_mask"] = node_mask
            parts["edge_mask"] = edge_mask
            res = backward_tok(eng, parts, s["node_gold"], s["edge_gold"], lr)
            perte_tr += res["node"] + res["edge"]
        met = evaluer(eng, val, node_mask, edge_mask)
        f1 = met["edge_f1"]
        if np.isfinite(f1) and f1 > best_f1:
            best_f1 = float(f1)
            best_snap = _parametres(eng)
            best_ep = ep
            attente = 0
        else:
            attente += 1
        lignes.append(
            {
                "epoch": ep,
                "train_loss": perte_tr / max(1, len(train)),
                **{f"val_{k}": v for k, v in met.items()},
            }
        )
        if attente >= patience:
            break
    if best_snap is not None:
        _restaurer(eng, best_snap)
    resume = {
        "epochs": len(lignes),
        "best_epoch": best_ep,
        "best_val_edge_f1": best_f1,
        "oov_val": taux_oov_val,
        "n_train": len(train),
        "n_val": len(val),
        "hors_fenetre": st_tr["hors_fenetre"] + st_val["hors_fenetre"],
        "arrieres": st_tr["arrieres"] + st_val["arrieres"],
        "data_hash": sha_donnees,
        "git_sha": _git_sha(),
        "seed": seed,
    }
    if csv_out:
        cols = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_node_f1",
            "val_edge_f1",
            "val_n_node",
            "val_n_edge",
        ]
        with open(csv_out, "w", newline="", encoding="utf-8") as fh:
            for k, v in resume.items():
                fh.write(f"# {k}={v}\n")
            wr = csv.DictWriter(fh, fieldnames=cols)
            wr.writeheader()
            for lgn in lignes:
                wr.writerow({c: lgn.get(c, "") for c in cols})
    return resume


def main(argv: list[str] | None = None) -> int:
    """CLI minimale (les hyperparamètres vivent en argv, jamais en dur)."""
    import argparse

    ap = argparse.ArgumentParser(description="Entraînement tok (E_tok + têtes).")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--csv-out", default=None)
    args = ap.parse_args(argv)
    try:
        resume = entrainer(
            args.train,
            args.val,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            patience=args.patience,
            d_model=args.d_model,
            csv_out=args.csv_out,
        )
    except GelError as exc:
        print(f"GELÉ (R8) : {exc}", flush=True)
        return 2
    print(
        json.dumps(
            {
                k: (v if not isinstance(v, float) or np.isfinite(v) else "NaN")
                for k, v in resume.items()
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
