# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests du moteur B — socle 0a ([T1] : pin + lazy + timeout + fallback + sans-spacy).

Exécution : ``PYTHONPATH=gcn-python/src pytest gcn-python/tests/test_engine_b.py -q``
(``import gcn_python`` seul résout vers un autre checkout — ne pas omettre PYTHONPATH).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from gcn_python.layer0 import engine_b, fallback_regex
from gcn_python.layer0.fallback_regex import analyze_fallback

TEXTE_FR = "Il pleut. Il reste car il est malade !"
TEXTE_CAUSE_FR = "Il reste parce que la pluie tombe."
TEXTE_CAUSE_EN = "She stayed because it rained."
RACINE_DEPOT = Path(__file__).resolve().parents[2]  # …/tests → gcn-python → dépôt
SRC = RACINE_DEPOT / "gcn-python" / "src" / "gcn_python"


@pytest.fixture(autouse=True)
def _compteurs_propres():
    engine_b.reset_stats()
    yield
    engine_b.reset_stats()


def test_interface_tuple_et_source_fallback_force():
    clauses, _connecteurs, source = engine_b.analyze(TEXTE_FR, force_fallback=True)
    assert source == "fallback"
    assert len(clauses) >= 2
    assert all(set(c) >= {"id", "texte", "span", "lang"} for c in clauses)


def test_fallback_familles_ponctuation():
    clauses, _, source = analyze_fallback("¿Vienes? ¡Vamos! C'est vrai… vraiment؟ حقا？", "es")
    assert source == "fallback"
    assert len(clauses) >= 4  # ? + ! + … + ؟ (+ ？)


def test_fallback_connecteurs_tables_existantes():
    # Marqueur hors snapshot minimal → prouve l'usage des tables du dépôt.
    assert "causal_scorer" in fallback_regex.TABLES_ORIGIN
    assert "balanced_auto" in fallback_regex.TABLES_ORIGIN
    _, conns, _ = analyze_fallback("Il sort en dépit de la pluie.", "fr")
    assert any(c["relation"] == "concession" for c in conns)
    _, conns, _ = analyze_fallback(TEXTE_CAUSE_FR, "fr")
    assert any(c["relation"] == "cause" and c["texte"].lower() == "parce que" for c in conns)
    _, conns, _ = analyze_fallback(TEXTE_CAUSE_EN, "en")
    assert any(c["relation"] == "cause" for c in conns)
    _, conns, _ = analyze_fallback("Er bleibt, weil er krank ist.", "de")
    assert any(c["relation"] == "cause" for c in conns)


def test_silver_fr_si_spacy_disponible():
    pytest.importorskip("spacy")
    try:
        __import__("fr_core_news_sm")
    except ImportError:
        pytest.skip("fr_core_news_sm absent")
    engine_b._reset_cache()
    clauses, connecteurs, source = engine_b.analyze(TEXTE_FR, lang="fr")
    assert source == "silver-B"
    assert len(clauses) >= 2
    assert any(c["relation"] == "cause" for c in connecteurs)
    meta = engine_b.provenance(source, "fr")
    assert meta["outil"] == "moteur-B" and meta["source"] == "silver-B"
    assert (meta["version"]["spacy"] or "").startswith("3.8")


def test_sans_spacy_vert(monkeypatch):
    """Simule l'absence de spaCy : fallback systématique + compteur."""
    engine_b._reset_cache()
    monkeypatch.setitem(sys.modules, "spacy", None)
    clauses, _, source = engine_b.analyze(TEXTE_FR, lang="fr")
    assert source == "fallback" and len(clauses) >= 1
    stats = engine_b.fallback_stats()
    assert stats["fallbacks"] == 1 and stats["total"] == 1


def test_timeout_spacy_lent_declenche_fallback(monkeypatch):
    pytest.importorskip("spacy")
    engine_b._reset_cache()

    def _lent(nlp, text):
        import time

        time.sleep(2)
        return nlp(text)

    monkeypatch.setattr(engine_b, "_parse_spacy", _lent)
    try:
        __import__("fr_core_news_sm")
    except ImportError:
        pytest.skip("fr_core_news_sm absent")
    _, _, source = engine_b.analyze(TEXTE_FR, lang="fr", timeout=0.2)
    assert source == "fallback"


def test_langs_budgetees_pt_it_sans_table():
    # Écart spec documenté : pt/it = segmentation seule, pas de lexique inventé.
    for lang in ("pt", "it"):
        clauses, connecteurs, source = engine_b.analyze("Chove. Fico em casa.", lang=lang,
                                                        force_fallback=True)
        assert source == "fallback" and len(clauses) >= 1 and connecteurs == []
    with pytest.raises(ValueError):
        engine_b.analyze("hello", lang="xx")
    with pytest.raises(TypeError):
        engine_b.analyze(42)


def test_compteur_fallback_et_seuil_alerte():
    assert engine_b.FALLBACK_ALERT_RATE == 0.05  # 5 % proposé, à valider phase 1
    engine_b.analyze(TEXTE_FR, force_fallback=True)
    engine_b.analyze(TEXTE_FR, force_fallback=True)
    stats = engine_b.fallback_stats()
    assert (stats["total"], stats["fallbacks"], stats["taux"]) == (2, 2, 1.0)
    assert stats["seuil_alerte"] == 0.05 and stats["alerte"] is True


def test_format_s0_explicite_sans_anticipation_s1_s3():
    sortie = engine_b.analyze(TEXTE_CAUSE_FR, force_fallback=True)
    enregistrement = engine_b.to_record(sortie, lang="fr")
    assert enregistrement["format_refacto"] == 20261007
    assert set(enregistrement["meta"]) == {"outil", "version", "source"}
    assert set(enregistrement["meta"]["version"]) == {"spacy", "modele"}
    for cle_interdite in ("tok_vocab", "d_model", "act_types"):
        assert cle_interdite not in enregistrement


def test_spacy_jamais_importe_au_top_level():
    # 1) AST : aucun import spaCy au niveau module (lazy-only, D-SPACY).
    for nom in ("engine_b.py", "fallback_regex.py"):
        arbre = ast.parse((SRC / "layer0" / nom).read_text(encoding="utf-8"))
        imports_top = [n for n in arbre.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        assert not any("spacy" in (getattr(n, "module", "") or "")
                       or any(a.name == "spacy" for a in getattr(n, "names", []))
                       for n in imports_top), nom
    # 2) runtime : importer nos modules ne charge pas spaCy (sous-processus frais).
    code = ("import sys; from gcn_python.layer0 import engine_b, fallback_regex;"
            "from gcn_python.layer0.fallback_regex import analyze_fallback;"
            "analyze_fallback('Il pleut car il reste.', 'fr');"
            "sys.exit(1 if 'spacy' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, check=False,
                           cwd=RACINE_DEPOT).returncode == 0


def test_runtime_appris_n_importe_jamais_moteur_b():
    racines = [SRC / "pipeline", SRC / "training", SRC / "engine.py"]
    fautifs = []
    for racine in racines:
        fichiers = [racine] if racine.is_file() else sorted(racine.rglob("*.py")) if racine.is_dir() else []
        for f in fichiers:
            texte = f.read_text(encoding="utf-8")
            if "engine_b" in texte or "fallback_regex" in texte:
                fautifs.append(str(f))
    assert fautifs == []


def test_fusion_sans_ponctuation_forte():
    from gcn_python.layer0.engine_b import fusionner_clauses
    t = "Respect for Private and Family Life under article"
    c1, c2, c3 = "Respect for Private and Family", "Life", "under article"
    cl = [{"id": "c001", "texte": c1, "span": [0, len(c1)], "lang": "fr"},
          {"id": "c002", "texte": c2,
           "span": [t.index(c2), t.index(c2) + len(c2)], "lang": "fr"},
          {"id": "c003", "texte": c3,
           "span": [t.index(c3), t.index(c3) + len(c3)], "lang": "fr"}]
    f = fusionner_clauses(cl, t, "fr")
    assert len(f) == 1 and f[0]["texte"] == t and f[0]["id"] == "c001"
    # Frontière forte préservée + idempotence.
    t2 = "Il pleut. Il reste."
    cl2 = [{"id": "c001", "texte": "Il pleut.", "span": [0, 9], "lang": "fr"},
           {"id": "c002", "texte": "Il reste.", "span": [10, 19], "lang": "fr"}]
    assert len(fusionner_clauses(cl2, t2, "fr")) == 2
    assert fusionner_clauses(f, t, "fr") == f
