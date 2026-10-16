# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests E_tok — vocab trié + format s1/s2 (refacto L1a)."""

from __future__ import annotations

import numpy as np
import pytest

from gcn_python.layer1.tok_embedding import FORMAT_REFACTO, TokEmbedding


def test_vocab_trie_independant_ordre_insertion():
    a = TokEmbedding(d_model=8, seed=7)
    a.build_vocab([30, 10, 20, 10])
    b = TokEmbedding(d_model=8, seed=7)
    b.build_vocab([10, 20, 30])
    assert a.tok_vocab == [10, 20, 30] == b.tok_vocab
    assert np.array_equal(a._E, b._E)
    assert a.vocab_size == 3


def test_lookup_oov_zero_et_compteur():
    e = TokEmbedding(d_model=4, seed=1)
    e.build_vocab([5])
    assert np.array_equal(e.lookup(999), np.zeros(4, dtype=np.float32))
    assert e.row_index(999) is None
    assert e.row_index(5) == 0
    assert e.oov_rate == 1.0  # 1 lookup, 1 OOV (build non compté)
    e.reset_oov()
    assert e.oov_rate == 0.0


def test_lookup_batch_et_copies():
    e = TokEmbedding(d_model=4, seed=1)
    e.build_vocab([5, 6])
    out = e.lookup_batch([5, 999, 6])
    assert out.shape == (3, 4)
    assert np.array_equal(out[1], np.zeros(4, dtype=np.float32))
    out[0, 0] = 12345.0
    assert e.lookup(5)[0] != 12345.0  # copies défensives (R1)


def test_backward_update_fait_descendre_le_cout():
    rng = np.random.default_rng(0)
    e = TokEmbedding(d_model=6, seed=3)
    e.build_vocab([11, 22])
    cible = rng.normal(0, 1, (4, 6)).astype(np.float32)
    ids = [11, 22, 11, 99]  # 99 = OOV (zéros, jamais appris)
    def cout():
        return float(((e.lookup_batch(ids) - cible) ** 2).mean())
    avant = cout()
    for _ in range(50):
        pred = e.lookup_batch(ids)
        e.backward(2.0 * (pred - cible) / pred.size, ids)
        e.update(0.5)
    assert cout() < avant
    assert np.isfinite(e._E).all()


def test_backward_ignore_oov():
    e = TokEmbedding(d_model=3, seed=1)
    e.build_vocab([1])
    avant = e._E.copy()
    e.backward(np.ones((2, 3), dtype=np.float32), [1, 999])
    e.update(1.0)
    assert not np.array_equal(e._E, avant)  # membre appris
    assert e._E.shape == (1, 3)  # OOV : rien créé, rien absorbé


def test_seed_reproductible():
    def fabrique(seed):
        e = TokEmbedding(d_model=5, seed=seed)
        e.build_vocab([9, 3, 7])
        return e._E

    assert np.array_equal(fabrique(9), fabrique(9))
    assert not np.array_equal(fabrique(9), fabrique(10))


def test_json_npz_roundtrip(tmp_path):
    e = TokEmbedding(d_model=8, seed=2)
    e.build_vocab([4, 8, 15, 16, 23, 42])
    r = TokEmbedding.from_json(e.to_json())
    assert r.tok_vocab == e.tok_vocab and np.array_equal(r._E, e._E)
    d = e.to_dict()
    assert d["format_refacto"] == FORMAT_REFACTO == 20261007
    assert d["d_model"] == 8 and "tokenizer_hash" in d and "rules" in d
    p = str(tmp_path / "tok.npz")
    r2 = TokEmbedding.load_npz(e.save_npz(p))
    assert r2.tok_vocab == e.tok_vocab and np.array_equal(r2._E, e._E)


def test_from_dict_rejette():
    e = TokEmbedding(d_model=4, seed=1)
    e.build_vocab([7])
    d = e.to_dict()
    with pytest.raises(ValueError, match="format_refacto"):
        TokEmbedding.from_dict({**d, "format_refacto": 1})
    with pytest.raises(ValueError, match="d_model"):
        TokEmbedding.from_dict({**d, "d_model": 0})
    with pytest.raises(ValueError, match="tokenizer_hash"):
        TokEmbedding.from_dict({**d, "tok_vocab": [7, 8]})
    with pytest.raises(ValueError, match="forme"):
        TokEmbedding.from_dict({**d, "E": [[0.0] * 4, [0.0] * 4]})
    with pytest.raises(ValueError):
        TokEmbedding(d_model=0)


def test_oov_hash_deterministe_et_appris():
    e = TokEmbedding(d_model=4, seed=1, oov_mode="hash")
    e.build_vocab([5])
    a = e.lookup(999)
    assert not np.array_equal(a, np.zeros(4, dtype=np.float32))
    assert np.array_equal(a, e.lookup(999))  # déterministe
    assert e.oov_rate == 1.0  # couverture comptée comme OOV
    avant = e._B.copy()
    e.backward(np.ones((1, 4), dtype=np.float32), [999])
    e.update(1.0)
    assert not np.array_equal(e._B, avant)  # seau appris
    assert e._E.shape == (1, 4)  # vocab intact


def test_oov_hash_rejet_mode():
    try:
        TokEmbedding(d_model=4, oov_mode="tfidf")
    except ValueError:
        return
    raise AssertionError("aurait dû lever")
