# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests actes + routage — règles [R9]/[T6] et couches (b)."""

from __future__ import annotations

import pytest

from gcn_python.layer2.act import ACT_TYPES, detect_act, detect_lang
from gcn_python.pipeline.route_intent import route_intent


def test_trois_actes_ordonnes():
    assert ACT_TYPES == ("assert", "question", "request")


def test_question_familles_ponctuation():
    assert detect_act("Tu viens ?")["act"] == "question"
    assert detect_act("¿Vienes?")["act"] == "question"
    assert detect_act("تأتي؟")["act"] == "question"


def test_exclamatif_assert_saillant_pas_request():
    r = detect_act("Quelle horreur !")
    assert r == {"act": "assert", "saillant": True, "lang": "fr"}


def test_question_sans_point_est_assert():
    r = detect_act("Tu viens")
    assert r["act"] == "assert" and not r["saillant"]


def test_si_conditionnel_assert():
    assert detect_act("Si tu viens, on part.")["act"] == "assert"


def test_consigne_request_et_bascule():
    r = detect_act("Réponds en wolof.")
    assert r["act"] == "request" and r["lang"] == "wo"
    assert detect_act("Answer in english.")["lang"] == "en"


def test_politesse_request():
    assert detect_act("Merci de fermer la porte.")["act"] == "request"
    assert detect_act("Please, sit down.")["act"] == "request"


def test_imperatif_nu_assert_par_defaut():
    # Écart assumé et documenté (pas de lexique verbal inventé).
    assert detect_act("Ferme la porte.")["act"] == "assert"


def test_vide_et_type():
    assert detect_act("   ")["act"] == "assert"
    with pytest.raises(TypeError):
        detect_act(42)
    with pytest.raises(TypeError):
        detect_lang(None)


def test_lang_defaut_script():
    assert detect_lang("Bonjour le monde.") == "fr"
    assert detect_lang("هل تأتي اليوم معنا؟") == "ar"
    assert detect_lang("今日は良い天気ですね。") == "und"
    assert detect_lang("...") == "fr"


def test_route_assert_stocke():
    cir = {"nodes": []}
    r = route_intent({"act": "assert", "saillant": False, "lang": "fr"}, "Il pleut.", cir)
    assert r == {"act": "assert", "saillant": False, "lang": "fr", "store": cir}


def test_route_question_dsl_injecte():
    r = route_intent(
        {"act": "question", "saillant": False, "lang": "fr"},
        "Pourquoi pleut-il ?",
        legacy_router=lambda t: "WHY pluie",
    )
    assert r["dsl"] == "WHY pluie" and r["query"].startswith("Pourquoi")
    r2 = route_intent({"act": "question", "saillant": False, "lang": "fr"}, "Quoi ?")
    assert r2["dsl"] is None  # rempli à la bascule


def test_route_request_ack_et_file():
    r = route_intent(
        {"act": "request", "saillant": False, "lang": "wo"}, "Réponds en wolof."
    )
    assert "enregistrée" in r["ack"] and "(langue : wo)" in r["ack"]
    assert r["filed"]["text"] == "Réponds en wolof."
    assert r["filed"]["intent"]["lang"] == "wo"


def test_route_invalide():
    with pytest.raises(ValueError, match="act"):
        route_intent({"act": "danser"}, "x")
    with pytest.raises(TypeError):
        route_intent("assert", "x")
    with pytest.raises(TypeError):
        route_intent({"act": "assert"}, 42)
