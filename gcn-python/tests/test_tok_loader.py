# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests data/tok_loader.py — chargeur générique (UNE implémentation).

Parité avec l'ancien ``train_tok.charger_split`` (délégation) : mêmes
samples, mêmes stats — tout écart = bug.
"""

import json

import pytest

from gcn_python.data.tok_loader import (
    charger_split,
    convertir_phrase,
    couverture_vocab,
    iter_phrases,
)
from gcn_python.training.train_tok import charger_split as charger_legacy


def _phrase(texte="Un. Deux.", ntypes=("processus", "processus"), rel="cause"):
    toks = [{"id": k + 1, "form": mot} for k, mot in enumerate(texte.split(" "))]
    noeuds = [
        {
            "id": f"n{k + 1}",
            "type": t,
            "label": f"n{k + 1}",
            "token_span": [k + 1, k + 1],
            "scope": "specific",
            "origin": "explicit",
        }
        for k, t in enumerate(ntypes)
    ]
    return {
        "id": "s1",
        "text": texte,
        "tokens": toks,
        "cir": {"nodes": noeuds, "edges": [{"sources": ["n1"], "target": "n2", "relation": rel}]},
    }


def _fichier(tmp_path, phrases):
    f = tmp_path / "split.json"
    f.write_text(
        json.dumps({"document": {"id": "d", "lang": "fr", "sentences": phrases}}), encoding="utf-8"
    )
    return str(f)


def test_iter_phrases(tmp_path):
    f = _fichier(tmp_path, [_phrase(), _phrase("Allo. Monde.")])
    phrases = list(iter_phrases(f))
    assert [p[0] for p in phrases] == ["s1", "s1"]
    assert phrases[0][1]["text"].startswith("Un.")


def test_convertir_ok_et_compteurs():
    s = _phrase()
    echantillon = convertir_phrase(s)
    assert echantillon["node_gold"] == [0, 0]
    assert echantillon["edge_gold"] == {(0, 1): 0}
    assert echantillon["hors_fenetre"] == 0 and echantillon["arrieres"] == 0
    assert len(echantillon["ids"]) == len(echantillon["offsets"]) > 0


def test_convertir_type_inconnu_refuse():
    from gcn_python.data.gold_spans import SpanError

    s = _phrase(ntypes=("processus", "bidule"))
    with pytest.raises(SpanError):
        convertir_phrase(s)


def test_convertir_fenetre_parametrable():
    toks = [{"id": k + 1, "form": f"w{k}"} for k in range(4)]
    noeuds = [
        {
            "id": f"n{k + 1}",
            "type": "processus",
            "label": "x",
            "token_span": [k + 1, k + 1],
            "scope": "s",
            "origin": "e",
        }
        for k in range(4)
    ]
    s = {
        "id": "s",
        "text": "w0 w1 w2 w3",
        "tokens": toks,
        "cir": {
            "nodes": noeuds,
            "edges": [{"sources": ["n1"], "target": "n4", "relation": "cause"}],
        },
    }
    echantillon = convertir_phrase(s, window_k=3)
    assert echantillon["edge_gold"] == {(0, 3): 0}
    echantillon = convertir_phrase(s, window_k=2)
    assert echantillon["edge_gold"] == {} and echantillon["hors_fenetre"] == 1
    with pytest.raises(ValueError):
        convertir_phrase(s, window_k=0)


def test_charger_parite_avec_legacy(tmp_path):
    f = _fichier(tmp_path, [_phrase(), _phrase("Allo. Monde.")])
    samples, stats = charger_split(f)
    samples_l, stats_l = charger_legacy(f)
    assert samples == samples_l
    assert stats == stats_l
    assert stats["phrases"] == 2 and stats["rejetees"] == []


def test_charger_rejet_compte(tmp_path):
    s = _phrase()
    s["tokens"] = [{"id": 1, "form": "zzz-introuvable-ici"}]
    f = _fichier(tmp_path, [s])
    samples, stats = charger_split(f)
    assert samples == [] and len(stats["rejetees"]) == 1


def test_couverture_vocab():
    samples = [{"ids": [1, 2, 3]}, {"ids": [2, 4]}]
    oov, total = couverture_vocab(lambda i: i in {1, 2, 3}, samples)
    assert (oov, total) == (1, 5)
    assert couverture_vocab(lambda i: True, []) == (0, 0)


def test_loader_passe_third_et_joints():
    from gcn_python.data.tok_loader import charger_split as _charger

    samples, stats = _charger("gcn-datasets/silver_ternary_proposal_20.json")
    assert stats["phrases"] == 20 and stats["rejetees"] == []
    assert sum(len(s["edge_third"]) for s in samples) == 8
    assert {v for s in samples for v in s["edge_joint"].values()} != set()
    roles = {v["role"] for s in samples for v in s["edge_third"].values()}
    assert roles <= {"Condition", "Mediator"}


def test_loader_rejette_third_inconnu():
    from gcn_python.data.gold_spans import SpanError
    from gcn_python.data.tok_loader import convertir_phrase

    base = {
        "id": "x", "text": "Il pleut donc je reste.",
        "tokens": [
            {"id": 1, "form": "Il"}, {"id": 2, "form": "pleut"},
            {"id": 3, "form": "donc"}, {"id": 4, "form": "je"},
            {"id": 5, "form": "reste."},
        ],
        "cir": {
            "nodes": [
                {"id": "n1", "type": "evenement", "token_span": [1, 2]},
                {"id": "n2", "type": "evenement", "token_span": [4, 5]},
            ],
            "edges": [
                {"source": "n1", "target": "n2", "relation": "cause",
                 "third": {"role": "Telepati", "node": "n1"}},
            ],
        },
    }
    try:
        convertir_phrase(base)
    except SpanError:
        return
    raise AssertionError("aurait dû lever")
