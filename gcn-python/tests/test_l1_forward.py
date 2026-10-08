# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests forward L1 — forward_ids + pooling fixe + miroir (refacto L1b)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gcn_python.layer0.tokenizer import split
from gcn_python.layer1.l1_forward import PoolMap, align_spans, forward_ids, pool_clauses
from gcn_python.layer1.tok_embedding import TokEmbedding

#: Données locales (DATA hors git) : test ignoré en CI, actif en local.
GOLD_MINI = Path("gcn-datasets/DATA/supervision/gold_mini/gold_mini.jsonl")


def test_forward_ids_forme_masque_copies():
    e = TokEmbedding(d_model=4, seed=1)
    e.build_vocab([1, 2, 3])
    out = forward_ids(e, [1, 2, 3], mask=[1, 0, 1])
    assert out.shape == (3, 4)
    assert np.array_equal(out[1], np.zeros(4, dtype=np.float32))
    assert np.array_equal(out[0], e.lookup(1))
    out[0, 0] = -999.0
    assert e.lookup(1)[0] != -999.0  # E préservé (R1)
    with pytest.raises(ValueError, match="mask"):
        forward_ids(e, [1, 2], mask=[1])


def test_align_spans_tokenise_et_invalide():
    offsets = [(0, 2), (3, 5), (6, 8)]
    assert align_spans(offsets, [(0, 5), (6, 8)]) == [[0, 1], [2]]
    assert align_spans(offsets, [(0, 8)]) == [[0, 1, 2]]
    assert align_spans(offsets, [(4, 4)]) == [[]]  # span vide : clause vide
    with pytest.raises(ValueError, match="invalide"):
        align_spans(offsets, [(5, 3)])


def test_pool_moyenne_miroir_et_vides():
    E = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    reps, miroir = pool_clauses(E, [[0, 2], [2], []])
    assert reps.shape == (3, 2)
    assert np.allclose(reps[0], [1.0, 0.5])  # moyenne des lignes 0 et 2
    assert np.allclose(reps[1], [1.0, 1.0])
    assert np.array_equal(reps[2], np.zeros(2, dtype=np.float32))
    assert isinstance(miroir, PoolMap) and len(miroir) == 3 and miroir.vides == 1
    assert miroir.rows == [[0, 2], [2], []]
    with pytest.raises(ValueError, match=r"\(N, d\)"):
        pool_clauses(np.zeros(4, dtype=np.float32), [[0]])


def test_poolmap_copies_independantes():
    lignes = [[0, 1]]
    miroir = PoolMap(lignes)
    lignes[0].append(99)
    assert miroir.rows == [[0, 1]]


@pytest.mark.skipif(not GOLD_MINI.exists(), reason="DATA hors git (local uniquement)")
def test_smoke_l0_l1_phrase_gold():
    # Bout-en-bout L0→L1 sur phrase réelle : split → vocab → embed → pool → grad.
    with open(GOLD_MINI, encoding="utf-8") as f:
        rec = json.loads(f.readline())
    ids, offsets, _ = split(rec["text"])
    assert len(ids) > 0
    e = TokEmbedding(d_model=16, seed=42)
    e.build_vocab(ids)
    E = forward_ids(e, ids)
    assert E.shape == (len(ids), 16) and np.isfinite(E).all()
    assert e.oov_rate == 0.0  # vocab construit sur ces ids : zéro OOV
    moitie = len(offsets) // 2
    coupure = offsets[moitie][0] if moitie else len(rec["text"])
    reps, miroir = pool_clauses(E, align_spans(offsets, [(0, coupure), (coupure, 10**9)]))
    assert reps.shape == (2, 16) and np.isfinite(reps).all()
    assert sum(len(r) for r in miroir.rows) == len(ids)  # couverture totale
    # Le gradient ne touche que les membres : OOV sans effet, membres appris.
    e.reset_oov()
    avant = e._E.copy()
    e.backward(np.ones((len(ids), 16), dtype=np.float32), ids)
    e.update(0.01)
    assert not np.array_equal(e._E, avant)
    assert np.isfinite(e._E).all()


def test_determinisme():
    e = TokEmbedding(d_model=8, seed=5)
    e.build_vocab([1, 2, 3])
    E = forward_ids(e, [3, 1, 2, 2])
    r1, m1 = pool_clauses(E, [[0, 2], [2, 3]])
    r2, m2 = pool_clauses(E, [[0, 2], [2, 3]])
    assert np.array_equal(r1, r2) and m1.rows == m2.rows
    with pytest.raises(ValueError, match="hors E"):
        pool_clauses(E, [[0, 9]])
