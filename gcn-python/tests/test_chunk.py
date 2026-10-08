# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests chunks + manifest — hash + compatibilité ([C], [R5])."""

from __future__ import annotations

import pytest

from gcn_python.layer1.chunk import (
    assert_compatible,
    chunk_hash,
    read_manifest,
    write_manifest,
)


def test_chunk_hash_deterministe_sensible():
    h1 = chunk_hash("tokhash", [10, 20, 30], 0, 12)
    assert h1 == chunk_hash("tokhash", [10, 20, 30], 0, 12) and len(h1) == 64
    assert chunk_hash("tokhash", [10, 20, 31], 0, 12) != h1  # ids
    assert chunk_hash("tokhash", [30, 20, 10], 0, 12) != h1  # ordre
    assert chunk_hash("tokhash", [10, 20, 30], 0, 13) != h1  # bornes
    assert chunk_hash("autre", [10, 20, 30], 0, 12) != h1  # tokenizer
    assert chunk_hash("tokhash", [10, 20, 30]) == chunk_hash("tokhash", [10, 20, 30], 0, 3)
    with pytest.raises(ValueError, match="tokenizer_hash"):
        chunk_hash("", [1])


def test_manifest_roundtrip_et_gardes(tmp_path):
    p = str(tmp_path / "manifest.json")
    write_manifest(p, "tokhash", 128)
    m = read_manifest(p)
    assert m == {"tokenizer_hash": "tokhash", "d_model": 128}
    assert_compatible(m, 128, "tokhash")
    with pytest.raises(ValueError, match="vec_dim"):
        assert_compatible(m, 64, "tokhash")
    with pytest.raises(ValueError, match="tokenizer_hash"):
        assert_compatible(m, 128, "autre")
    with pytest.raises(ValueError, match="d_model"):
        write_manifest(str(tmp_path / "x.json"), "tokhash", 0)
    with pytest.raises(ValueError, match="tokenizer_hash"):
        write_manifest(str(tmp_path / "x.json"), "", 128)
    mauvais = tmp_path / "mauvais.json"
    mauvais.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizer_hash"):
        read_manifest(str(mauvais))
