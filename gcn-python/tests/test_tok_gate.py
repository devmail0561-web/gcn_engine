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
    # Prédication minimale conservée (S+V) :
    assert clause_degeneree("Il reste") is None
    assert clause_degeneree("Université Ibno") is None


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


def test_subdivision_connecteurs():
    from gcn_python.nlu.tok import subdiviser_connecteurs

    t = "Il reste parce que la pluie tombe."
    s, x = subdiviser_connecteurs(t, [[0, len(t)]], [t], "fr")
    assert len(x) == 2 and x[1].startswith("parce que")
    assert s[0][0] == 0 and s[-1][1] == len(t)
    t2 = "Les ventes baissent. On reduit les couts."
    _, x2 = subdiviser_connecteurs(
        t2, [[0, 19], [20, len(t2)]], ["Les ventes baissent.", "On reduit les couts."], "fr"
    )
    assert x2 == ["Les ventes baissent.", "On reduit les couts."]


def test_subdivision_connecteur_en_tete():
    # B-c1 : protase en tête -> coupe à la première virgule (ponctuation
    # seule, aucun mot-matching ajouté).
    from gcn_python.nlu.tok import subdiviser_connecteurs

    t = "Parce qu'il a fourni une représentation simple, l'article a eu une influence considérable."
    _, x = subdiviser_connecteurs(t, [[0, len(t)]], [t], "fr")
    assert len(x) == 2 and x[0].startswith("Parce qu")
