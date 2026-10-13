# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests data/gold_spans.py — reconstruction validée + parité avec l'oracle.

Tout écart avec ``scripts/audit_spans_tok.py`` (même méthode) = bug :
le test de parité l'exige sur les trois splits v4/final2.
"""

import json
import subprocess
import sys

import pytest

from gcn_python.data.gold_spans import SpanError, plier, reconstruire_offsets, span_clause

SPLITS = [
    "gcn-datasets/DATA/supervision/v4/final2/train/train.json",
    "gcn-datasets/DATA/supervision/v4/final2/val/val.json",
    "gcn-datasets/DATA/supervision/v4/final2/test/test.json",
]


def test_pliage_conserve_longueur():
    assert len(plier("l’empire d’aujourd’hui")) == len("l’empire d’aujourd’hui")
    assert plier("l’empire") == "l'empire"


def test_reconstruction_unitaire():
    texte = "Les ventes baissent."
    tokens = [
        {"id": 1, "form": "Les"},
        {"id": 2, "form": "ventes"},
        {"id": 3, "form": "baissent"},
        {"id": 4, "form": "."},
    ]
    offs = reconstruire_offsets(texte, tokens)
    assert offs == {1: (0, 3), 2: (4, 10), 3: (11, 19), 4: (19, 20)}
    assert span_clause(offs, 2, 3, 4) == (4, 19)


def test_forme_introuvable_leve():
    with pytest.raises(SpanError):
        reconstruire_offsets("Bonjour.", [{"id": 1, "form": "zzz"}])


def test_bornes_invalides_levent():
    offs = {1: (0, 3)}
    with pytest.raises(SpanError):
        span_clause(offs, 2, 1, 1)
    with pytest.raises(SpanError):
        span_clause(offs, 1, 5, 1)


def test_texte_non_str():
    with pytest.raises(TypeError):
        reconstruire_offsets(42, [])


@pytest.mark.parametrize("fichier", SPLITS)
def test_taux_rejet_nul_sur_gold(fichier):
    with open(fichier, encoding="utf-8") as fh:
        doc = json.load(fh)
    rejet = 0
    for s in doc["document"]["sentences"]:
        try:
            offs = reconstruire_offsets(s["text"], s["tokens"])
            for n in s["cir"]["nodes"]:
                a, b = n["token_span"]
                span_clause(offs, a, b, len(s["tokens"]))
        except SpanError:
            rejet += 1
    assert rejet == 0  # seuil STOP 5 % : tout rejet = enquête, pas silencieux


def test_parite_avec_oracle_audit():
    proc = subprocess.run(
        [sys.executable, "scripts/audit_spans_tok.py", *SPLITS],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0
    assert proc.stdout.count("[OK]") == 3
