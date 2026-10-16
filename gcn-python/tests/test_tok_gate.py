# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests gate d'entrée anti-dégénéré (Ph5, 2026-10-16).

71/71 abstentions non-gold, 0 gold perdu, 13/14 surconfiants interceptés.
"""
import pytest

from gcn_python.nlu.tok import clause_degeneree, parse_text


def test_clause_degeneree_motifs():
    assert clause_degeneree("") == "vide"
    assert clause_degeneree("   ") == "vide"
    assert clause_degeneree("»") == "court(1tok)"
    assert clause_degeneree("DÉCRYPTAGE") == "court(1tok)"
    assert clause_degeneree("« » ...") == "ponctuation-pure"
    assert clause_degeneree("Les ventes baissent nettement") is None
    assert clause_degeneree("Pour prolonger la ligne un du métro") is None


def test_parse_text_filtre_et_trace():
    seg = parse_text("DÉCRYPTAGE - Les ventes baissent nettement.")
    rejetees = dict(seg["clauses_rejetees"])
    assert "DÉCRYPTAGE - Les ventes baissent nettement." not in rejetees
    assert len(seg["clause_spans"]) == len(seg["clause_textes"]) >= 1
    # Tout rejeté => forward vide mais jamais d'exception.
    seg2 = parse_text("DÉCRYPTAGE")
    assert seg2["clause_spans"] == []
    assert seg2["clauses_rejetees"] != []


def test_parse_text_vide_et_types():
    seg = parse_text("   ")
    assert seg["clause_spans"] == [] and seg["clauses_rejetees"] == []
    with pytest.raises(TypeError):
        parse_text(42)
