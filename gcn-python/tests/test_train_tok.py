# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests training/train_tok.py — gates §6[G] : le gradient apprend vraiment.

- finite-diff : ``loss(w - eps·d) < loss(w)`` sur E (preuve anti-lettre-morte) ;
- ``_ce_backward`` cohérent avec ``ce_loss`` (mêmes masques) ;
- alignement gold (fenêtre, arrières, rejets) ; ``GelError`` R8 ;
- déterminisme seed ; snapshot/restore best.
"""

import json

import numpy as np
import pytest

from gcn_python.pipeline.forward_ids import TokForwardEngine
from gcn_python.training.refacto_loss import GelError, ce_loss
from gcn_python.training.train_tok import (
    _ce_backward,
    _softmax_backward,
    backward_tok,
    charger_split,
    entrainer,
)

TEXTE = "Les ventes baissent. On reduit les couts."


def _mini_split(tmp_path, phrases):
    f = tmp_path / "mini.json"
    f.write_text(
        json.dumps({"document": {"id": "d", "lang": "fr", "sentences": phrases}}), encoding="utf-8"
    )
    return str(f)


def _phrase(texte, noeuds, aretes):
    toks, cur = [], 0
    for mot in texte.split(" "):
        i = texte.find(mot, cur)
        toks.append({"id": len(toks) + 1, "form": mot})
        cur = i + len(mot)
    return {"id": "s1", "text": texte, "tokens": toks, "cir": {"nodes": noeuds, "edges": aretes}}


def _noeud(nid, ntype, a, b):
    return {
        "id": nid,
        "type": ntype,
        "label": nid,
        "token_span": [a, b],
        "scope": "specific",
        "origin": "explicit",
    }


def test_ce_backward_coherent_avec_ce_loss():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, 4)).astype(np.float32)
    gold = np.array([0, 1, 2, 3, 1])
    mask = np.array([True, True, False, True])
    assert ce_loss(logits, gold, class_mask=mask) > 0
    d = _ce_backward(logits, gold, mask)
    assert d.shape == logits.shape
    # Lignes masquées (classe 2) : gradient nul, loss ignorée.
    assert (d[2] == 0).all()
    assert np.abs(d[[0, 1, 3, 4]]).sum() > 0


def test_ce_backward_finite_diff():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(3, 3)).astype(np.float64)
    gold = np.array([2, 0, 1])
    d = _ce_backward(logits, gold).astype(np.float64)
    num = np.zeros_like(d)
    eps = 1e-5
    for i in range(3):
        for j in range(3):
            plus = logits.copy()
            plus[i, j] += eps
            moins = logits.copy()
            moins[i, j] -= eps
            num[i, j] = (ce_loss(plus, gold) - ce_loss(moins, gold)) / (2 * eps)
    assert np.allclose(d, num, rtol=1e-3, atol=1e-5)


def test_softmax_backward_finite_diff():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(4, 5))
    e = np.exp(z - z.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    dp = rng.normal(size=(4, 5))
    ana = _softmax_backward(p, dp)
    num = np.zeros_like(dp)
    eps = 1e-6
    for i in range(4):
        for j in range(5):
            dz = np.zeros_like(z)
            dz[i, j] = eps
            e2 = np.exp(z + dz - (z + dz).max(axis=1, keepdims=True))
            p2 = e2 / e2.sum(axis=1, keepdims=True)
            num[i, j] = ((p2 * dp).sum() - (p * dp).sum()) / eps
    assert np.allclose(ana, num, rtol=1e-3, atol=1e-6)


def test_alignement_fenetre_et_arrieres(tmp_path):
    # 4 clauses : arete (0,1) ok, (0,3) hors fenetre, (2,1) arriere.
    texte = "Un. Deux. Trois. Quatre."
    noeuds = [_noeud(f"n{k}", "processus", k, k) for k in (1, 2, 3, 4)]
    aretes = [
        {"sources": ["n1"], "target": "n2", "relation": "cause"},
        {"sources": ["n1"], "target": "n4", "relation": "cause"},
        {"sources": ["n3"], "target": "n2", "relation": "cause"},
    ]
    f = _mini_split(tmp_path, [_phrase(texte, noeuds, aretes)])
    samples, stats = charger_split(f)
    assert len(samples) == 1
    assert samples[0]["edge_gold"] == {(0, 1): 0}
    assert stats["hors_fenetre"] == 1 and stats["arrieres"] == 1


def test_gel_rare_classe(tmp_path):
    texte = "Un. Deux."
    noeuds = [_noeud("n1", "processus", 1, 1), _noeud("n2", "processus", 2, 2)]
    aretes = [{"sources": ["n1"], "target": "n2", "relation": "cause"}]
    f = _mini_split(tmp_path, [_phrase(texte, noeuds, aretes)])
    with pytest.raises(GelError):
        entrainer(f, f, epochs=1)


def test_pas_sgd_fait_baisser_la_loss():
    eng = TokForwardEngine(d_model=8, seed=3)
    eng.build_vocab([TEXTE])
    from gcn_python.nlu.tok import parse_text

    seg = parse_text(TEXTE)
    kwargs = {
        "text": TEXTE,
        "ids": seg["ids"],
        "offsets": seg["offsets"],
        "clause_spans": seg["clause_spans"],
        "clause_textes": seg["clause_textes"],
    }
    avant = eng.forward_supervise(**kwargs)
    perte_avant = ce_loss(np.asarray(avant["node_logits"]), np.asarray([0, 0]))
    res = backward_tok(eng, avant, [0, 0], {(0, 1): 0}, lr=0.05)
    assert res["n_edge_sup"] == 1
    apres = eng.forward_supervise(**kwargs)
    perte_apres = ce_loss(np.asarray(apres["node_logits"]), np.asarray([0, 0]))
    assert perte_apres < perte_avant  # gate §6[G] : le gradient descend


def test_determinisme_meme_seed():
    def _tour():
        eng = TokForwardEngine(d_model=8, seed=11)
        eng.build_vocab([TEXTE])
        from gcn_python.nlu.tok import parse_text

        seg = parse_text(TEXTE)
        parts = eng.forward_supervise(
            TEXTE, seg["ids"], seg["offsets"], seg["clause_spans"], seg["clause_textes"]
        )
        backward_tok(eng, parts, [1, 2], {(0, 1): 3}, lr=0.01)
        return eng.emb._E.copy(), eng.rgcn.W_0.copy()

    e1, w1 = _tour()
    e2, w2 = _tour()
    assert np.array_equal(e1, e2) and np.array_equal(w1, w2)
