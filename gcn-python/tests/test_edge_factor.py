# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests têtes factorisées : roundtrip 19/19 + formes + totalité compose()."""
import numpy as np
import pytest

from gcn_python.layer2.edge_factor import (
    BASE_RELATIONS,
    RELATION_19,
    SPECIAL,
    TERNARY,
    FactorizedEdgeHead,
    compose,
    decompose,
)


def test_roundtrip_19():
    assert len(RELATION_19) == 19
    for rel in RELATION_19:
        assert compose(*decompose(rel)) == rel


def test_compose_total():
    # Tout triplet valide rend une des 19 (jamais d'exception inattendue).
    vus = set()
    for b in BASE_RELATIONS:
        for t in TERNARY:
            for s in SPECIAL:
                vus.add(compose(b, t, s))
    assert vus <= set(RELATION_19)


def test_decompose_rejects_unknown():
    with pytest.raises(ValueError):
        decompose("telepathy")


def test_head_shapes_and_predict():
    head = FactorizedEdgeHead(d_edge=64)
    lb, lt, ls = head.forward(np.zeros(64, dtype=np.float32))
    assert lb.shape == (11,) and lt.shape == (4,) and ls.shape == (3,)
    assert head.predict(np.zeros(64, dtype=np.float32)) in RELATION_19


def test_head_rejects_bad_dim():
    with pytest.raises(ValueError):
        FactorizedEdgeHead(d_edge=0)
