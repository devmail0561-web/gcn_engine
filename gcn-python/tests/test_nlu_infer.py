# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests NLU sans UD — B + actes + concepts +expects (étape 6)."""

from __future__ import annotations

import pytest

from gcn_python.nlu.infer import EXPECTS, content_words, infer_nlu, silver_annotate
from gcn_python.pipeline.route_intent import route_intent


def test_infer_clauses_intents_concepts():
    r = infer_nlu("Il pleut. Tu viens ?")
    assert r["source"] in ("silver-B", "fallback")
    assert r["lang"] == "fr"
    assert len(r["clauses"]) == 2
    actes = [c["intent"]["act"] for c in r["clauses"]]
    assert actes == ["assert", "question"]
    assert r["clauses"][1]["expects"] == {"response": True, "action": False}
    assert r["clauses"][0]["expects"] == {"response": False, "action": False}
    assert "pleut" in r["clauses"][0]["concepts"]
    assert all(set(c) >= {"texte", "span", "intent", "concepts", "expects"}
               for c in r["clauses"])


def test_infer_expects_request():
    r = infer_nlu("Réponds en wolof.")
    assert r["clauses"][0]["intent"]["act"] == "request"
    assert r["clauses"][0]["expects"] == {"response": False, "action": True}
    assert r["lang"] == "wo"
    assert EXPECTS["request"] == {"response": False, "action": True}


def test_infer_fallback_force():
    r = infer_nlu("Il pleut car il reste.", force_fallback=True)
    assert r["source"] == "fallback"
    assert len(r["clauses"]) >= 1


def test_infer_invalide():
    with pytest.raises(TypeError):
        infer_nlu(42)
    with pytest.raises(ValueError, match="budget"):
        infer_nlu("x", "xx")


def test_content_words_sans_ud():
    assert content_words("Pourquoi la pluie tombe-t-elle ?") == ["pourquoi", "pluie", "tombe-t-elle"]
    assert content_words("... !?") == []


def test_silver_annotate_tracabilite():
    r = silver_annotate("Il reste parce qu'il pleut.", campagne="test-socle")
    assert r["source"] in ("silver-B", "fallback")
    assert len(r["phrase_hash"]) == 64
    assert r["campagne"] == "test-socle"
    assert set(r["outil"]) == {"moteur-b", "actes"}
    assert r["clauses"] and "intent" in r["clauses"][0]


def test_e2e_infer_route_sans_ud():
    # Question → route avec routeur legacy injecté (P1 sous question).
    r = infer_nlu("Pourquoi pleut-il ?")
    clause = r["clauses"][0]
    assert clause["intent"]["act"] == "question"
    action = route_intent(clause["intent"], clause["texte"],
                          legacy_router=lambda t: "WHY pluie")
    assert action["act"] == "question" and action["dsl"] == "WHY pluie"
    # Assert → stockage.
    r = infer_nlu("Il pleut.")
    clause = r["clauses"][0]
    action = route_intent(clause["intent"], clause["texte"], cir={"nodes": []})
    assert action["act"] == "assert" and action["store"] == {"nodes": []}


def test_pas_de_cycle_ni_ud():
    # Ni pipeline/ ni training importés (lazy, [D]) ; aucun module UD/lattice.
    import ast
    from pathlib import Path

    racine = Path("gcn-python/src/gcn_python")
    for nom in ("nlu/infer.py", "nlu/__init__.py"):
        arbre = ast.parse((racine / nom).read_text(encoding="utf-8"))
        imports = set()
        for n in ast.walk(arbre):
            if isinstance(n, ast.Import):
                imports.update(a.name.split(".")[0] for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                mod = (n.module or "").split(".")
                imports.add(mod[0])
                if mod[0] == "gcn_python" and len(mod) > 1:
                    imports.add(mod[1])
        assert "pipeline" not in imports and "training" not in imports, (nom, imports)
