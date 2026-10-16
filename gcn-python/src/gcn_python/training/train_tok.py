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

from ..constants import (
    COARSE_NODE_TYPES,
    COARSE_RELATION_TYPES,
    FINE_TO_COARSE_NODE,
    FINE_TO_COARSE_RELATION,
    NODE_TYPES,
    RELATION_TYPES,
)
from ..data.tok_loader import couverture_vocab
from ..evaluation.metrics import edge_macro_f1, node_macro_f1
from ..pipeline.forward_ids import OOV_WARN_RATE, TokForwardEngine
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
    """Split JSON gold → samples tok + stats (délègue à ``data/tok_loader``).

    Conservé pour compatibilité (tests + appelants) : UNE seule
    implémentation d'alignement, ici réexportée.
    """
    from ..data.tok_loader import charger_split as _charger

    return _charger(fichier)


def _bce_loss(logits: np.ndarray, gold: np.ndarray) -> float:
    """BCE-with-logits stable (voie A, tête seule)."""
    lg = np.asarray(logits, dtype=np.float64).ravel()
    y = np.asarray(gold, dtype=np.float64).ravel()
    if lg.shape != y.shape or lg.size == 0:
        raise ValueError(f"BCE : formes {lg.shape} vs {y.shape} (non vides).")
    if ((y != 0.0) & (y != 1.0)).any():
        raise ValueError("BCE : gold non binaire.")
    return float(np.mean(np.maximum(lg, 0.0) - lg * y + np.log1p(np.exp(-np.abs(lg)))))


def _bce_backward(logits: np.ndarray, gold: np.ndarray) -> np.ndarray:
    """Gradient de ``_bce_loss`` : (sigmoid - y) / N (1-D, comme ``_ce_backward``)."""
    lg = np.asarray(logits, dtype=np.float64).ravel()
    y = np.asarray(gold, dtype=np.float64).ravel()
    p = 1.0 / (1.0 + np.exp(-lg))
    return ((p - y) / lg.size).astype(np.float32)


def backward_tok(
    eng: TokForwardEngine,
    parts: dict,
    node_gold: list[int],
    edge_gold: dict,
    lr: float,
    edge_gold_fin: dict | None = None,
    factor_primary: bool = False,
    factor_masks: tuple | None = None,
    exist_gold: dict | None = None,
) -> dict:
    """Un pas SGD phrase (caches chauds de ``forward_supervise``).

    Chaîne : d_edge → MLP.backward_edge_dx → split (base→pooled,
    enriched→clauses, probs→softmax⁻¹→nœuds) → d_node total →
    MLP.backward_node_dx → RGCN.backward → pool-mean⁻¹ → E_tok.backward
    → updates (bump epoch M1a partout). Paires sans gold : aucun gradient
    (exclues, pas masquées à 0 — le 0 biaiserait la CE).

    Auxiliaire factorisé (R2, opt-in via ``eng.factor_aux_enabled``) :
    les mêmes paires supervisées entraînent aussi 3 petites têtes
    (base 11 × ternary 4 × special 3) depuis ``edge_gold_fin`` (indices
    fins, jamais remappés). Pas de gradient vers le tronc — signal
    partagé futur, mesure immédiate. Zéro lexique.
    Retourne ``{"node": loss, "edge": loss, "factor": loss, "span": loss,
    "exist": loss, "n_edge_sup": int, "n_exist_sup": int}`` (``span`` = 0.0
    sans ``span_aux``). ``exist_gold`` : {(i,j): 0/1} — paires à existence
    connue ; paires hors ``exist_gold`` : aucun gradient (exclues, jamais
    devinées — même doctrine que les paires sans gold). Tête existence seule
    mise à jour (précédent R2 : pas de reflux tronc).
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
    dx_primaire: dict = {}
    if factor_primary and sup_idx:
        # Primaire factorisée (espace factor, D-R8F) : REMPLACE la loss
        # plate. ``edge_gold`` porte des indices fins (jamais remappés dans
        # cet espace). Têtes partagées base/ternary/special + reflux tronc.
        from ..layer2.edge_factor import (
            BASE_RELATIONS as _BB,
        )
        from ..layer2.edge_factor import (
            SPECIAL as _SS,
        )
        from ..layer2.edge_factor import (
            TERNARY as _TT,
        )
        from ..layer2.edge_factor import (
            FactorizedEdgeHead as _FEH,
        )
        from ..layer2.edge_factor import (
            decompose as _decomp,
        )

        _vecs = parts.get("edge_vecs") or []
        if _vecs:
            if getattr(eng, "factor_primary", None) is None:
                eng.factor_primary = _FEH(
                    d_edge=len(np.asarray(_vecs[0]).ravel()),
                    seed=int(getattr(eng, "seed", 42)) + 5,
                )
            _X, _gb, _gt, _gs, _ks = [], [], [], [], []
            for k in sup_idx:
                _p = paires[k]
                try:
                    _b, _t, _s = _decomp(RELATION_TYPES[int(edge_gold[_p])])
                except (ValueError, IndexError):
                    continue
                _X.append(np.asarray(_vecs[k], dtype=np.float32).ravel())
                _gb.append(_BB.index(_b))
                _gt.append(_TT.index(_t))
                _gs.append(_SS.index(_s))
                _ks.append(k)
            if _X:
                loss_edge, _dX = eng.factor_primary.loss_update_dx(
                    np.stack(_X),
                    np.asarray(_gb, dtype=np.int64),
                    np.asarray(_gt, dtype=np.int64),
                    np.asarray(_gs, dtype=np.int64),
                    factor_masks,
                    lr,
                )
                dx_primaire = {paires[_k]: _dX[_r] for _r, _k in enumerate(_ks)}
    elif sup_idx:
        sub = mat_edge[sup_idx]
        loss_edge = ce_loss(sub, sup_gold, class_mask=parts.get("edge_mask"))
        d_edge_full[sup_idx] = _ce_backward(sub, sup_gold, parts.get("edge_mask"))
    # Auxiliaire factorisé R2 : 3 CEs (base/ternary/special) sur les mêmes
    # paires, labels fins décomposés. Têtes seules mises à jour.
    loss_factor = 0.0
    if getattr(eng, "factor_aux_enabled", False) and edge_gold_fin and sup_idx:
        from ..layer2.edge_factor import (
            BASE_RELATIONS,
            SPECIAL,
            TERNARY,
            FactorizedEdgeHead,
            decompose,
        )

        vecs = parts.get("edge_vecs") or []
        if vecs:
            if getattr(eng, "factor_aux", None) is None:
                eng.factor_aux = FactorizedEdgeHead(
                    d_edge=len(np.asarray(vecs[0]).ravel()),
                    seed=int(getattr(eng, "seed", 42)),
                )
            X_list, gb, gt, gs = [], [], [], []
            for k in sup_idx:
                p = paires[k]
                if p not in edge_gold_fin:
                    continue
                try:
                    b, t, s_ = decompose(RELATION_TYPES[int(edge_gold_fin[p])])
                except (ValueError, IndexError):
                    continue
                X_list.append(np.asarray(vecs[k], dtype=np.float32).ravel())
                gb.append(BASE_RELATIONS.index(b))
                gt.append(TERNARY.index(t))
                gs.append(SPECIAL.index(s_))
            if X_list:
                loss_factor = eng.factor_aux.loss_and_update(
                    np.stack(X_list),
                    np.asarray(gb, dtype=np.int64),
                    np.asarray(gt, dtype=np.int64),
                    np.asarray(gs, dtype=np.int64),
                    lr,
                )
    # Auxiliaire spans R3 : bornes start/end sur reps token (token_reps),
    # gold = frontières des alignements clauses (même signal que les noeuds
    # actuels — la granularité sub-clause viendra du gold-3000). Tête seule
    # + reflux partagé (2e update ctx/phrase si contexte — documenté).
    loss_span = 0.0
    if getattr(eng, "span_aux_enabled", False):
        from ..layer1.span_head import SpanHead

        if getattr(eng, "span_head", None) is None:
            _d_span = int(np.asarray(parts["token_reps"]).shape[1])
            eng.span_head = SpanHead(d_model=_d_span, seed=int(getattr(eng, "seed", 42)) + 4)
        _H = np.asarray(parts["token_reps"], dtype=np.float32)
        _n_tok = len(parts["ids"])
        _ys = np.zeros(_n_tok)
        _ye = np.zeros(_n_tok)
        for _m in parts["alignes"]:
            if _m:
                _ys[_m[0]] = 1.0
                _ye[_m[-1]] = 1.0
        loss_span = eng.span_head.loss_and_update(_H, _ys, _ye, lr)
        _dH = eng.span_head.grad_input(_H, _ys, _ye)
        if getattr(eng, "ctx", None) is not None:
            _d_in_s, _g_s = eng.ctx.backward(_dH.astype(np.float32))
            eng.ctx.update(_g_s, lr)
            _dH = _d_in_s
        eng.emb.backward(_dH, parts["ids"])
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
    # Voie primaire factorisée : dx direct, pas de tête plate (grads_edge None).
    for k, (i, j) in enumerate(paires):
        if dx_primaire:
            row = dx_primaire.get((i, j))
            if row is None or not np.abs(row).any():
                continue
            dx = np.asarray(row, dtype=np.float32)
        else:
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
    # Existence (voie A) : BCE sur paires à existence connue, re-run
    # déterministe par paire (même pattern que les arêtes). Tête mise à jour
    # + REFLUX TRONC (2026-10-16 : tête seule prouvée insuffisante — collapse
    # total, F1=0 — le signal de détection doit façonner les représentations
    # partagées, même split que la dx arêtes). Sans tête ou sans exist_gold :
    # loss 0.0 (P10 : paires non jugées = exclues).
    loss_exist = 0.0
    n_exist_sup = 0
    _xg = exist_gold or {}
    _vecs_ex = parts.get("edge_vecs") or []
    if _xg and getattr(eng.encoder, "_exist_layers", None) is not None and _vecs_ex:
        _xl, _yl, _kk = [], [], []
        for k, p in enumerate(paires):
            if p in _xg and k < len(_vecs_ex):
                _xl.append(
                    float(
                        eng.encoder.forward_exist(
                            np.asarray(_vecs_ex[k], dtype=np.float32)
                        ).ravel()[0]
                    )
                )
                _yl.append(float(_xg[p]))
                _kk.append(k)
        if _xl:
            _la = np.asarray(_xl, dtype=np.float64)
            _ya = np.asarray(_yl, dtype=np.float64)
            loss_exist = _bce_loss(_la, _ya)
            _d = _bce_backward(_la, _ya)
            grads_exist = None
            for r in range(len(_kk)):
                eng.encoder.forward_exist(np.asarray(_vecs_ex[_kk[r]], dtype=np.float32))
                _g, _dx = eng.encoder.backward_exist(_d[r : r + 1])
                grads_exist = (
                    _g
                    if grads_exist is None
                    else [(a[0] + b[0], a[1] + b[1]) for a, b in zip(grads_exist, _g, strict=True)]
                )
                # Reflux tronc (même split que la dx arêtes).
                _i, _j = paires[_kk[r]]
                _dx = np.asarray(_dx, dtype=np.float32).ravel()
                d_pooled[_i] += _dx[:d_pool]
                d_pooled[_j] += _dx[d_pool : 2 * d_pool]
                d_enriched[_i] += _dx[d_base : d_base + d_pool]
                d_enriched[_j] += _dx[d_base + d_pool : d_base + 2 * d_pool]
            eng.encoder.update_exist(grads_exist, lr)
            n_exist_sup = len(_kk)
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
            return {
                "node": loss_node,
                "edge": loss_edge,
                "factor": loss_factor,
                "exist": 0.0,
                "n_edge_sup": len(sup_idx),
                "n_exist_sup": 0,
            }
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
    # Pool-mean⁻¹ via les alignements + E_tok (via contexte O1 si actif).
    d_tok = np.zeros((len(parts["ids"]), d_pool), dtype=np.float32)
    for c, membres in enumerate(parts["alignes"]):
        if membres:
            d_tok[membres] += (d_pooled[c] / len(membres)).astype(np.float32)
    if getattr(eng, "ctx", None) is not None:
        d_in, grads_c = eng.ctx.backward(d_tok.astype(np.float32))
        eng.ctx.update(grads_c, lr)
        d_tok = d_in
    eng.emb.backward(d_tok, parts["ids"])
    # Updates (bump epoch M1a dans chaque update — sens sûr).
    if grads_node is not None:
        eng.encoder.update_node(grads_node, lr)
    if grads_edge is not None:
        eng.encoder.update_edge(grads_edge, lr)
    if grads_r is not None:
        eng.rgcn.update(grads_r, lr)
    eng.emb.update(lr)
    return {
        "node": loss_node,
        "edge": loss_edge,
        "factor": loss_factor,
        "span": loss_span,
        "exist": loss_exist,
        "n_edge_sup": len(sup_idx),
        "n_exist_sup": n_exist_sup,
    }


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
    if eng.emb.oov_mode == "hash":
        snap["oov_B"] = eng.emb._B.copy()
    snap["node"] = [(c.W.copy(), c.b.copy()) for c in eng.encoder._node_layers]
    snap["edge"] = [(c.W.copy(), c.b.copy()) for c in eng.encoder._edge_layers]
    if getattr(eng.encoder, "_exist_layers", None) is not None:
        snap["exist"] = [(c.W.copy(), c.b.copy()) for c in eng.encoder._exist_layers]
    snap["rgcn"] = [eng.rgcn.W_r.copy(), eng.rgcn.W_0.copy()]
    if getattr(eng, "ctx", None) is not None:
        snap["ctx"] = [a.copy() for a in eng.ctx.parameters()]
    if getattr(eng, "factor_aux", None) is not None:
        snap["factor"] = [a.copy() for a in eng.factor_aux.parameters()]
    if getattr(eng, "factor_primary", None) is not None:
        snap["factor_primary"] = [a.copy() for a in eng.factor_primary.parameters()]
    if getattr(eng, "span_head", None) is not None:
        snap["span"] = [np.asarray(a).copy() for a in eng.span_head.parameters()]
        snap["span_d"] = int(eng.span_head.w_s.shape[0])
    return snap


def _restaurer(eng: TokForwardEngine, snap: dict) -> None:
    """Restaure un snapshot (formes validées — fail-fast)."""
    if eng.emb._E.shape != snap["E"].shape:
        raise ValueError("snapshot incompatible (E).")
    eng.emb._E[:] = snap["E"]
    if "oov_B" in snap:
        if eng.emb.oov_mode != "hash":
            raise ValueError("snapshot avec table OOV mais moteur sans.")
        if eng.emb._B.shape != snap["oov_B"].shape:
            raise ValueError("snapshot incompatible (table OOV).")
        eng.emb._B[:] = snap["oov_B"]
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
    if "exist" in snap:
        _ex = getattr(eng.encoder, "_exist_layers", None)
        if _ex is None:
            raise ValueError("snapshot avec existence mais moteur sans.")
        for (w, b), couche in zip(snap["exist"], _ex, strict=True):
            if couche.W.shape != w.shape:
                raise ValueError("snapshot incompatible (existence).")
            couche.W[:] = w
            couche.b[:] = b
    if eng.rgcn.W_r.shape != snap["rgcn"][0].shape:
        raise ValueError("snapshot incompatible (R-GCN).")
    eng.rgcn.W_r[:] = snap["rgcn"][0]
    eng.rgcn.W_0[:] = snap["rgcn"][1]
    if "ctx" in snap:
        if getattr(eng, "ctx", None) is None:
            raise ValueError("snapshot avec contexte mais moteur sans.")
        for dst, src in zip(eng.ctx.parameters(), snap["ctx"], strict=True):
            if dst.shape != src.shape:
                raise ValueError("snapshot incompatible (contexte).")
            dst[:] = src
    if "factor" in snap:
        if getattr(eng, "factor_aux", None) is None:
            raise ValueError("snapshot avec facteur mais tête absente.")
        for dst, src in zip(eng.factor_aux.parameters(), snap["factor"], strict=True):
            if dst.shape != src.shape:
                raise ValueError("snapshot incompatible (facteur).")
            dst[:] = src
    if "factor_primary" in snap:
        from ..layer2.edge_factor import FactorizedEdgeHead as _FEHp

        if getattr(eng, "factor_primary", None) is None:
            _d0 = int(np.asarray(snap["factor_primary"][0]).shape[1])
            eng.factor_primary = _FEHp(d_edge=_d0, seed=int(getattr(eng, "seed", 42)) + 5)
        for dst, src in zip(eng.factor_primary.parameters(), snap["factor_primary"], strict=True):
            if dst.shape != src.shape:
                raise ValueError("snapshot incompatible (facteur primaire).")
            dst[:] = np.asarray(src)
    if "span" in snap:
        from ..layer1.span_head import SpanHead as _SpanHead

        if getattr(eng, "span_head", None) is None:
            eng.span_head = _SpanHead(
                d_model=int(snap["span_d"]), seed=int(getattr(eng, "seed", 42)) + 4
            )
        for dst, src in zip(eng.span_head.parameters(), snap["span"], strict=True):
            if dst.shape != src.shape:
                raise ValueError("snapshot incompatible (spans).")
            np.copyto(dst, np.asarray(src))


def evaluer(
    eng: TokForwardEngine,
    samples: list,
    node_mask=None,
    edge_mask=None,
    node_types: list[str] | None = None,
    relation_types: list[str] | None = None,
    use_factor: bool = False,
    factor_masks: tuple | None = None,
) -> dict:
    """Passe éval (sans grad) : argmax → F1 macro (NaN si vide, jamais 0.0).

    Arêtes : paires SUPERVISÉES seulement (ce qu'on a demandé au modèle).
    ``node_types`` : espace de labels du moteur (défaut canonique) — la F1
    est calculée dans cet espace et étiquetée comme telle (jamais comparée
    entre espaces).
    ``use_factor`` (espace factor D-R8F) : prédiction via têtes primaires +
    ``compose()``, argmax restreint aux classes observées (masques) —
    jamais d'hallucination ternaire sans signal.
    """
    noms_noeuds = list(node_types) if node_types else list(NODE_TYPES)
    noms_rels = list(relation_types) if relation_types else list(RELATION_TYPES)
    pred_n, gold_n, pred_e, gold_e = [], [], [], []
    perte = 0.0
    for s in samples:
        parts = eng.forward_supervise(
            s["text"], s["ids"], s["offsets"], s["clause_spans"], s["clause_textes"]
        )
        node_gold = np.asarray(s["node_gold"], dtype=np.int64)
        perte += ce_loss(np.asarray(parts["node_logits"]), node_gold, class_mask=node_mask)
        pred_n += [noms_noeuds[int(i)] for i in np.argmax(parts["node_logits"], axis=1)]
        gold_n += [noms_noeuds[int(i)] for i in node_gold]
        if use_factor:
            from ..layer2.edge_factor import (
                BASE_RELATIONS as _EB,
            )
            from ..layer2.edge_factor import (
                SPECIAL as _ES,
            )
            from ..layer2.edge_factor import (
                TERNARY as _ET,
            )
            from ..layer2.edge_factor import (
                compose as _comp,
            )
            from ..layer2.edge_factor import (
                decompose as _dec2,
            )

            _head = getattr(eng, "factor_primary", None)
            if _head is None:
                raise ValueError("éval factor sans tête primaire (pas entraînée).")
            _mb, _mt, _ms = factor_masks if factor_masks is not None else (None, None, None)
            for k, p in enumerate(parts["paires"]):
                if p not in s["edge_gold"]:
                    continue
                g = s["edge_gold"][p]
                _gb, _gt, _gs = _dec2(noms_rels[int(g)])
                _lb, _lt, _ls = _head.forward(parts["edge_vecs"][k])
                if _mb is not None:
                    _lb = np.where(np.asarray(_mb, dtype=bool), _lb, -np.inf)
                    _lt = np.where(np.asarray(_mt, dtype=bool), _lt, -np.inf)
                    _ls = np.where(np.asarray(_ms, dtype=bool), _ls, -np.inf)
                pred_e.append(
                    _comp(
                        _EB[int(np.argmax(_lb))], _ET[int(np.argmax(_lt))], _ES[int(np.argmax(_ls))]
                    )
                )
                gold_e.append(noms_rels[int(g)])
                perte += ce_loss(_lb.reshape(1, -1), np.asarray([_EB.index(_gb)]))
            continue
        mat = np.stack(parts["edge_logits"], axis=0) if parts["edge_logits"] else None
        for k, p in enumerate(parts["paires"]):
            if p not in s["edge_gold"]:
                continue
            g = s["edge_gold"][p]
            perte += ce_loss(mat[k : k + 1], np.asarray([g]), class_mask=edge_mask)
            pred_e.append(noms_rels[int(np.argmax(mat[k]))])
            gold_e.append(noms_rels[int(g)])
    return {
        "loss": perte / max(1, len(samples)),
        "node_f1": node_macro_f1(pred_n, gold_n, class_subset=noms_noeuds),
        "edge_f1": edge_macro_f1(pred_e, gold_e, class_subset=noms_rels),
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
    espace: str = "fin",
    init_npz: str | None = None,
    factor_aux: bool = False,
    context: bool = False,
    init_o1: str | None = None,
    oov_hash: bool = False,
    model_out: str | None = None,
    span_aux: bool = False,
) -> dict:
    """Boucle SGD tok (ordre fixe, seedée — reproductible, protocole §5).

    1. Charge splits (rejet > 5 % → ValueError STOP).
    2. Vocabulaire sur train (+sanity OOV train = 0) ; OOV val publié + warn
       (écart documenté à R5 : la val mesure, elle n'avorte pas).
    3. Espace : ``fin`` (8 nœuds + 19 relations — GEL R8 actuel),
    ``coarse`` D4 (4 + 5 — R8 franchi, gates prod intacts) ou ``factor``
    (nœuds fins + arêtes primaires factorisées base 11 × ternary 4 ×
    special 3 — doctrine D-R8F : ``require`` sur les 11 bases apprises,
    masques seuls pour nœuds/ternary/special présents ; seuils et
    calibration restent gelés par ``grid_f1``/``require`` aval) ; remap
    gold fin→coarse sauf factor (fins intacts), F1 étiquetée dans l'espace
    mesuré (jamais comparée entre espaces).
    4. Masques R8 + ``require_min_counts`` (``GelError`` fail-fast).
    5. Epochs : forward_supervise → backward_tok → update par phrase ;
       éval val ; best sur ``val_edge_macro_f1`` (NaN jamais best) ;
       patience puis restore best (early-stopping unifié edge, §6).
    Retourne le résumé (le moteur porte les poids best).
    """
    if epochs < 1:
        raise ValueError(f"epochs doit être ≥ 1 (reçu {epochs!r}).")
    if espace not in ("fin", "coarse", "factor"):
        raise ValueError(f"espace doit être 'fin', 'coarse' ou 'factor' (reçu {espace!r}).")
    noms_noeuds = list(COARSE_NODE_TYPES) if espace == "coarse" else list(NODE_TYPES)
    noms_rels = list(COARSE_RELATION_TYPES) if espace == "coarse" else list(RELATION_TYPES)
    train, st_tr = charger_split(train_path)
    val, st_val = charger_split(val_path)
    for nom, stats in (("train", st_tr), ("val", st_val)):
        taux = len(stats["rejetees"]) / max(1, stats["phrases"])
        if taux > SEUIL_REJET:
            raise ValueError(f"{nom} : rejet alignement {taux:.1%} > 5 % — STOP (données).")
    if espace == "coarse":
        for s in train + val:
            if s.get("espace_natif") == "coarse":
                # Production : labels coarse natifs (jamais remappés) —
                # valide les bornes au lieu de convertir.
                if any(g >= len(noms_noeuds) for g in s["node_gold"]):
                    raise ValueError(f"node coarse hors espace : {s.get('id')} — données.")
                if any(g >= len(noms_rels) for g in s["edge_gold"].values()):
                    raise ValueError(f"edge coarse hors espace : {s.get('id')} — données.")
                # Pas de labels fins → auxiliaire factorisé désactivé sur
                # ces samples (edge_gold_fin vide = skip, jamais deviné).
                s["edge_gold_fin"] = {}
                continue
            s["edge_gold_fin"] = dict(s["edge_gold"])
            try:
                s["node_gold"] = [
                    noms_noeuds.index(FINE_TO_COARSE_NODE[NODE_TYPES[g]]) for g in s["node_gold"]
                ]
                s["edge_gold"] = {
                    p: noms_rels.index(FINE_TO_COARSE_RELATION[RELATION_TYPES[g]])
                    for p, g in s["edge_gold"].items()
                }
            except KeyError as exc:
                raise ValueError(f"label fin hors table coarse : {exc} — données.") from exc
    else:
        for s in train + val:
            s["edge_gold_fin"] = s["edge_gold"]
    eng = TokForwardEngine(
        d_model=d_model,
        seed=seed,
        node_types=noms_noeuds,
        relation_types=noms_rels,
        use_context=bool(context),
        use_oov_hash=bool(oov_hash),
        exist_head=True,
    )
    eng.build_vocab([s["text"] for s in train])
    eng.factor_aux_enabled = bool(factor_aux)
    eng.factor_aux = None
    eng.span_aux_enabled = bool(span_aux)
    eng.span_head = None
    if init_npz is not None:
        # E2 : init O1 pré-entraînée (même vocab exigé, sinon STOP — pas de
        # mélange silencieux entre vocabulaires).
        from ..layer1.tok_embedding import TokEmbedding as _TokEmbInit

        _init = _TokEmbInit.load_npz(init_npz)
        if _init.d_model != eng.emb.d_model:
            raise ValueError(
                f"init-npz d_model={_init.d_model} ≠ d_model={eng.emb.d_model} — STOP."
            )
        if set(_init.tok_vocab) != set(eng.emb.tok_vocab):
            raise ValueError(
                f"init-npz vocab {len(_init.tok_vocab)} ≠ train vocab "
                f"{len(eng.emb.tok_vocab)} (ensembles) — régénérer l'init "
                "sur le même train."
            )
        # Réindexe les lignes par id stable (l'ordre interne dépend de l'ordre
        # d'insertion, pas du contenu — les ensembles sont la vérité).
        _pos = {v: i for i, v in enumerate(_init.tok_vocab)}
        _E_init = np.asarray(_init._E, dtype=np.float32)
        _E_cible = np.stack([_E_init[_pos[v]] for v in eng.emb.tok_vocab]).astype(np.float32)
        eng.emb._E = _E_cible.copy()
    if init_o1 is not None:
        # E2-MLM : init complète E + contexte O1 (mêmes ensembles exigés,
        # --context obligatoire — pas de chargement partiel silencieux).
        if getattr(eng, "ctx", None) is None:
            raise ValueError("init-o1 exige --context (moteur sans contexte).")
        with np.load(init_o1, allow_pickle=False) as _z:
            if int(_z["format_o1"]) != 20261014:
                raise ValueError("init-o1 : format inconnu — régénérer.")
            if int(_z["d_model"]) != eng.emb.d_model:
                raise ValueError("init-o1 : d_model divergent — STOP.")
            _v = [int(v) for v in _z["tok_vocab"].tolist()]
            if set(_v) != set(eng.emb.tok_vocab):
                raise ValueError("init-o1 : vocabulaire divergent — régénérer sur le même train.")
            _pv = {v: i for i, v in enumerate(_v)}
            _Eo = np.asarray(_z["E"], dtype=np.float32)
            eng.emb._E = np.stack([_Eo[_pv[v]] for v in eng.emb.tok_vocab]).astype(np.float32)
            for _nom, _bloc, _attr in (
                ("ctx_W1", eng.ctx.bloc1, "W"),
                ("ctx_b1", eng.ctx.bloc1, "b"),
                ("ctx_W2", eng.ctx.bloc2, "W"),
                ("ctx_b2", eng.ctx.bloc2, "b"),
            ):
                _w = np.asarray(_z[_nom], dtype=np.float32)
                _cible = getattr(_bloc, _attr)
                if _cible.shape != _w.shape:
                    raise ValueError(f"init-o1 : {_nom} forme incompatible.")
                _cible[:] = _w

    def _dans_vocab(i) -> bool:
        return eng.emb.row_index(i) is not None

    oov_tr, tot_tr = couverture_vocab(_dans_vocab, train)
    if tot_tr and oov_tr / tot_tr > OOV_WARN_RATE:
        raise ValueError(f"OOV train {oov_tr / tot_tr:.1%} > 5 % — vocabulaire défectueux.")
    oov_val, tot_val = couverture_vocab(_dans_vocab, val)
    taux_oov_val = oov_val / (tot_val or 1)
    if taux_oov_val > OOV_WARN_RATE:
        warnings.warn(
            f"train_tok : OOV val {taux_oov_val:.1%} > 5 % — publié, pas avorté "
            "(la val mesure ; écart documenté à R5).",
            UserWarning,
            stacklevel=2,
        )
    # Masques + gate R8 (fail-fast — GelError = sortie écrite, pas acharnement).
    # Espace factor (D-R8F) : require sur les 11 bases APPRISES (tête
    # primaire), masques seuls pour nœuds présents + ternary/special
    # observés. Seuils/calibration restent gelés en aval (grid_f1).
    n_counts = np.zeros(len(noms_noeuds), dtype=np.int64)
    e_counts = np.zeros(len(noms_rels), dtype=np.int64)
    for s in train:
        for g in s["node_gold"]:
            n_counts[g] += 1
        for g in s["edge_gold"].values():
            e_counts[g] += 1
    factor_masks = None
    use_factor_primary = espace == "factor"
    if use_factor_primary:
        from ..layer2.edge_factor import (
            BASE_RELATIONS as _FB,
        )
        from ..layer2.edge_factor import (
            SPECIAL as _FS,
        )
        from ..layer2.edge_factor import (
            TERNARY as _FT,
        )
        from ..layer2.edge_factor import (
            decompose as _dec,
        )

        _b_cnt = np.zeros(len(_FB), dtype=np.int64)
        _t_cnt = np.zeros(len(_FT), dtype=np.int64)
        _s_cnt = np.zeros(len(_FS), dtype=np.int64)
        for s in train:
            for g in s["edge_gold"].values():
                _b, _t, _s = _dec(RELATION_TYPES[int(g)])
                _b_cnt[_FB.index(_b)] += 1
                _t_cnt[_FT.index(_t)] += 1
                _s_cnt[_FS.index(_s)] += 1
        require_min_counts(_b_cnt, nom="bases factorisées")
        node_mask = build_class_mask(n_counts)
        edge_mask = build_class_mask(e_counts)
        factor_masks = (
            build_class_mask(_b_cnt),
            _t_cnt > 0,
            _s_cnt > 0,
        )
    else:
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
        perte_ex_tr = 0.0
        for s in train:
            parts = eng.forward_supervise(
                s["text"], s["ids"], s["offsets"], s["clause_spans"], s["clause_textes"]
            )
            parts["node_mask"] = node_mask
            parts["edge_mask"] = edge_mask
            res = backward_tok(
                eng,
                parts,
                s["node_gold"],
                s["edge_gold"],
                lr,
                edge_gold_fin=s.get("edge_gold_fin"),
                factor_primary=use_factor_primary,
                factor_masks=factor_masks,
                exist_gold=s.get("exist_gold"),
            )
            perte_tr += res["node"] + res["edge"] + res.get("factor", 0.0) + res.get("span", 0.0)
            perte_ex_tr += res.get("exist", 0.0)
        met = evaluer(
            eng,
            val,
            node_mask,
            edge_mask,
            node_types=noms_noeuds,
            relation_types=noms_rels,
            use_factor=use_factor_primary,
            factor_masks=factor_masks,
        )
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
                "train_exist": perte_ex_tr / max(1, len(train)),
                **{f"val_{k}": v for k, v in met.items()},
            }
        )
        if attente >= patience:
            break
    if best_snap is not None:
        _restaurer(eng, best_snap)
    if model_out is not None:
        # Sauvegarde best-restore s1+ (E + têtes + R-GCN + ctx + facteur) —
        # le seul artefact évaluable sur test/holdout (protocole §4).
        eng.save_npz(model_out)
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
        "espace": espace,
        "factor_aux": bool(factor_aux),
        "span_aux": bool(span_aux),
        "context": bool(context),
        "oov_hash": bool(oov_hash),
        "init_npz": bool(init_npz is not None),
        "init_o1": bool(init_o1 is not None),
        "node_types": noms_noeuds,
        "relation_types": noms_rels,
    }
    if csv_out:
        cols = [
            "epoch",
            "train_loss",
            "train_exist",
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
    ap.add_argument(
        "--espace",
        choices=["fin", "coarse", "factor"],
        default="fin",
        help="Espace de labels : fin (8+19, GEL R8 actuel), coarse "
        "D4 (4+5, R8 franchi), factor (fins + têtes primaires factorisées, "
        "D-R8F : require sur les 11 bases). F1 étiquetée par espace.",
    )
    ap.add_argument("--csv-out", default=None)
    ap.add_argument(
        "--init-npz",
        default=None,
        help="Init E_tok O1 pré-entraînée (même train exigé, sinon STOP).",
    )
    ap.add_argument(
        "--factor-aux",
        action="store_true",
        help="Auxiliaire factorisé R2 (base×ternary×special) sur labels fins.",
    )
    ap.add_argument(
        "--span-aux",
        action="store_true",
        help="Auxiliaire bornes start/end R3 sur reps token (même signal clauses).",
    )
    ap.add_argument(
        "--context",
        action="store_true",
        default=True,
        help="Encodeur contextuel O1 entre lookup et pooling "
        "(défaut ON depuis E8-révisé : +0.21 prouvé E4/E8, généralisé 3 splits ; "
        " calibration des confiances reste indépendante, Ph5).",
    )
    ap.add_argument(
        "--no-context",
        action="store_false",
        dest="context",
        help="Désactive l'encodeur contextuel (sacs seuls, 0.107 — mesure seule).",
    )
    ap.add_argument(
        "--oov-hash",
        action="store_true",
        help="OOV → seaux hashés appris au lieu de zéros (E7, défaut zéros).",
    )
    ap.add_argument(
        "--model-out",
        default=None,
        help="Sauvegarde best-restore s1+ évaluable sur test/holdout.",
    )
    ap.add_argument(
        "--init-o1",
        default=None,
        help="Init complète O1-MLM (E+ctx, exige --context, même train sinon STOP).",
    )
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
            espace=args.espace,
            init_npz=args.init_npz,
            factor_aux=args.factor_aux,
            span_aux=args.span_aux,
            context=args.context,
            init_o1=args.init_o1,
            oov_hash=args.oov_hash,
            model_out=args.model_out,
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
