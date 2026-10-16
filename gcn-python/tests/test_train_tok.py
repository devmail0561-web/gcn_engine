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


def test_pas_sgd_une_clause_sans_arete():  # E=0 : boucle arêtes + R-GCN sautés, nœuds + E_tok mis à jour, fini.
    import numpy as np

    eng = TokForwardEngine(d_model=8, seed=1)
    eng.build_vocab(["Les ventes baissent."])
    from gcn_python.nlu.tok import parse_text

    seg = parse_text("Les ventes baissent.")
    parts = eng.forward_supervise(
        "Les ventes baissent.",
        seg["ids"],
        seg["offsets"],
        seg["clause_spans"],
        seg["clause_textes"],
    )
    res = backward_tok(eng, parts, [2], {}, lr=0.01)
    assert res["n_edge_sup"] == 0
    assert np.isfinite(res["node"])


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


def _corpus_equilibre(tmp_path, repetitions=8):
    """160 phrases 2-clauses : 4 fins x 5 groupes coarse-relations x 8.

    Groupes couverts (un représentant fin chacun) : causale=cause,
    restrictive=filter, negative=prevent, sequentielle=sequence,
    systemique=data_dependency → 32 occurrences/groupe (≥ N_min 30).
    Nœuds coarse : 40/groupe (≥ 30). GERME déterministe (pas de RNG).
    """
    fins = ["processus", "entite", "etat_local", "condition"]
    rels = ["cause", "filter", "prevent", "sequence", "data_dependency"]
    phrases = []
    k = 0
    for _ in range(repetitions):
        for rel in rels:
            for nt in fins:
                texte = f"Phrase{k} sujet. Suite{k} objet."
                toks = [{"id": i + 1, "form": m} for i, m in enumerate(texte.split(" "))]
                phrases.append(
                    {
                        "id": f"s{k}",
                        "text": texte,
                        "tokens": toks,
                        "cir": {
                            "nodes": [
                                {
                                    "id": "n1",
                                    "type": nt,
                                    "label": "n1",
                                    "token_span": [1, 2],
                                    "scope": "s",
                                    "origin": "e",
                                },
                                {
                                    "id": "n2",
                                    "type": "entite",
                                    "label": "n2",
                                    "token_span": [3, 4],
                                    "scope": "s",
                                    "origin": "e",
                                },
                            ],
                            "edges": [{"sources": ["n1"], "target": "n2", "relation": rel}],
                        },
                    }
                )
                k += 1
    f = tmp_path / "equilibre.json"
    f.write_text(
        json.dumps({"document": {"id": "d", "lang": "fr", "sentences": phrases}}), encoding="utf-8"
    )
    return str(f)


def test_coarse_debloque_sans_toucher_r8(tmp_path):
    f = _corpus_equilibre(tmp_path)
    resume = entrainer(f, f, epochs=2, lr=0.01, seed=7, patience=5, d_model=8, espace="coarse")
    assert resume["espace"] == "coarse"
    assert resume["node_types"] == [
        "action_coarse",
        "nominal_coarse",
        "evenement_coarse",
        "condition_coarse",
    ]
    assert np.isfinite(resume["best_val_edge_f1"])
    assert resume["epochs"] >= 1


def test_fin_gel_meme_donnees(tmp_path):
    # Espace fin : 8 relations + 4 types fins à 0 → GelError (gate intacte).
    f = _corpus_equilibre(tmp_path)
    with pytest.raises(GelError):
        entrainer(f, f, epochs=1, espace="fin")


def test_espace_invalide_refuse():
    with pytest.raises(ValueError):
        entrainer("x", "y", espace="moyen")


def test_moteur_espaces_custom_et_persistance(tmp_path):
    from gcn_python.constants import COARSE_NODE_TYPES, COARSE_RELATION_TYPES
    from gcn_python.pipeline.forward_ids import TokForwardEngine as _TE

    eng = _TE(
        d_model=8,
        seed=2,
        node_types=list(COARSE_NODE_TYPES),
        relation_types=list(COARSE_RELATION_TYPES),
    )
    assert eng.encoder is not None and eng.rgcn.n_relations == 6  # 5 + no-edge
    with pytest.raises(ValueError):
        _TE(node_types=[])
    with pytest.raises(ValueError):
        _TE(node_types=["a", "a"])
    eng.build_vocab([TEXTE])
    chemin = str(tmp_path / "coarse.npz")
    eng.save_npz(chemin)
    charge = _TE.load_npz(chemin)
    assert charge.node_types == list(COARSE_NODE_TYPES)
    assert charge.relation_types == list(COARSE_RELATION_TYPES)


def test_bce_coherent_finite_diff():
    # Gate §6[G] : _bce_backward descend _bce_loss.
    from gcn_python.training.train_tok import _bce_backward, _bce_loss

    rng = np.random.default_rng(4)
    lg = rng.normal(size=6).astype(np.float64)
    y = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    avant = _bce_loss(lg, y)
    d = _bce_backward(lg, y)
    apres = _bce_loss(lg - 0.5 * d, y)
    assert apres < avant
    with pytest.raises(ValueError):
        _bce_loss(np.array([0.5]), np.array([2.0]))


def test_exist_head_bout_en_bout():
    # Tête seule : backward met à jour les poids exist, loss descend.
    eng = TokForwardEngine(
        d_model=8, seed=5, exist_head=True, node_types=["a", "b"], relation_types=["r0", "r1"]
    )
    eng.build_vocab([TEXTE])
    from gcn_python.nlu.tok import parse_text

    seg = parse_text(TEXTE)
    kw = {
        "text": TEXTE,
        "ids": seg["ids"],
        "offsets": seg["offsets"],
        "clause_spans": seg["clause_spans"],
        "clause_textes": seg["clause_textes"],
    }
    parts = eng.forward_supervise(**kw)
    w_avant = [c.W.copy() for c in eng.encoder._exist_layers]
    res = backward_tok(eng, parts, [0, 1], {(0, 1): 0}, lr=0.05, exist_gold={(0, 1): 1})
    assert res["n_exist_sup"] == 1 and np.isfinite(res["exist"])
    assert any(
        not np.array_equal(a, b.W) for a, b in zip(w_avant, eng.encoder._exist_layers, strict=True)
    )
    # Sans exist_gold : tête intacte, loss 0.
    w_apres = [c.W.copy() for c in eng.encoder._exist_layers]
    res2 = backward_tok(eng, parts, [0, 1], {(0, 1): 0}, lr=0.05)
    assert res2["n_exist_sup"] == 0 and res2["exist"] == 0.0
    assert all(
        np.array_equal(a, b.W) for a, b in zip(w_apres, eng.encoder._exist_layers, strict=True)
    )


def test_loader_coarse_natif_et_exist_gold():
    # Production : labels coarse natifs + exist_gold, jamais remappés.
    from gcn_python.data.gold_spans import SpanError
    from gcn_python.data.tok_loader import convertir_phrase

    s = _phrase(
        "Il reste parce que la pluie tombe.",
        [_noeud("n0", "action_coarse", 1, 2), _noeud("n1", "action_coarse", 3, 7)],
        [{"sources": ["n0"], "target": "n1", "relation": "causale"}],
    )
    s["espace_natif"] = "coarse"
    s["exist_gold"] = {"0,1": 1}
    echantillon = convertir_phrase(s)
    assert echantillon["espace_natif"] == "coarse"
    assert echantillon["edge_gold"] == {(0, 1): 0}  # causale = idx 0 coarse
    assert echantillon["node_gold"] == [0, 0]  # action_coarse = idx 0
    assert echantillon["exist_gold"] == {(0, 1): 1}
    s2 = dict(s)
    s2["cir"] = {
        "nodes": s["cir"]["nodes"],
        "edges": [{"sources": ["n0"], "target": "n1", "relation": "cause"}],
    }  # fin refusé en natif coarse
    with pytest.raises(SpanError):
        convertir_phrase(s2)
    s3 = dict(s)
    s3["exist_gold"] = {"0,1": 7}
    with pytest.raises(SpanError):
        convertir_phrase(s3)


def test_save_load_exist_roundtrip(tmp_path):
    eng = TokForwardEngine(
        d_model=8, seed=6, exist_head=True, node_types=["a", "b"], relation_types=["r0", "r1"]
    )
    eng.build_vocab([TEXTE])
    p = str(tmp_path / "ex.npz")
    eng.save_npz(p)
    e2 = TokForwardEngine.load_npz(p)
    assert getattr(e2.encoder, "exist_head", False) is True
    for a, b in zip(eng.encoder._exist_layers, e2.encoder._exist_layers, strict=True):
        assert np.array_equal(a.W, b.W)
    # Vieux checkpoint (sans exist) : chargé sans tête, comportement intact.
    e3 = TokForwardEngine(d_model=8, seed=6, node_types=["a", "b"], relation_types=["r0", "r1"])
    e3.build_vocab([TEXTE])
    p2 = str(tmp_path / "ancien.npz")
    e3.save_npz(p2)
    e4 = TokForwardEngine.load_npz(p2)
    assert getattr(e4.encoder, "exist_head", False) is False
