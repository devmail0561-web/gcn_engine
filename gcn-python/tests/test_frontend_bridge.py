# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests pour frontend/bridge.py — pont lattice (tokens réels, zéro dictionnaire)."""
import json
import shutil
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gcn_python.frontend.bridge import (
    GCNBridgeError,
    GCNLatticeParser,
    _call_gcn_lattice,
    _lattice_token_to_ud,
    _reps_from_lattice,
)

# ── Fixture lattice de référence ──────────────────────────────────────────────

_LATTICE_TWO_CLAUSES = {
    "source_text": "Il venait si tu venais.",
    "tokens": [
        {"index": 1, "form": "Il", "pos": "other", "dep_rel": "other",
         "dep_head": -1, "lemma": "il", "clause": 0, "flags": []},
        {"index": 2, "form": "venait", "pos": "verb", "dep_rel": "root",
         "dep_head": 0, "lemma": "venir", "clause": 0, "flags": ["imparfait"]},
        {"index": 3, "form": "si", "pos": "other", "dep_rel": "mark",
         "dep_head": 5, "lemma": "si", "clause": 1, "flags": []},
        {"index": 4, "form": "tu", "pos": "other", "dep_rel": "other",
         "dep_head": -1, "lemma": "tu", "clause": 1, "flags": []},
        {"index": 5, "form": "venais", "pos": "verb", "dep_rel": "advcl",
         "dep_head": 2, "lemma": "venir", "clause": 1, "flags": []},
        {"index": 6, "form": ".", "pos": "punct", "dep_rel": "punct",
         "dep_head": -1, "lemma": ".", "clause": 1, "flags": []},
    ],
    "clauses": [[1, 2], [3, 6]],
}


# ── Tests mappings lattice → UD ───────────────────────────────────────────────


def test_lattice_pos_mapping_all_cases():
    """Tous les LatticePos ont un UPOS (fonction positionnelle, pas de lemme)."""
    assert _lattice_token_to_ud({"pos": "verb", "dep_rel": "root"})["pos"] == "VERB"
    assert _lattice_token_to_ud({"pos": "noun", "dep_rel": "nsubj"})["pos"] == "NOUN"
    assert _lattice_token_to_ud({"pos": "adv", "dep_rel": "advmod"})["pos"] == "ADV"
    assert _lattice_token_to_ud({"pos": "punct", "dep_rel": "punct"})["pos"] == "PUNCT"
    assert _lattice_token_to_ud({"pos": "other", "dep_rel": "other"})["pos"] == "X"


def test_lattice_function_overrides_pos():
    """mark→SCONJ, det→DET, advmod→ADV : la fonction positionnelle prime."""
    assert _lattice_token_to_ud({"pos": "other", "dep_rel": "mark"})["pos"] == "SCONJ"
    assert _lattice_token_to_ud({"pos": "other", "dep_rel": "det"})["pos"] == "DET"
    assert _lattice_token_to_ud({"pos": "other", "dep_rel": "advmod"})["pos"] == "ADV"


def test_lattice_morph_from_shape_flags():
    """imparfait→Tense=Past, infinitive/participle→VerbForm. Rien d'autre."""
    tok = _lattice_token_to_ud({"pos": "verb", "dep_rel": "root", "flags": ["imparfait"]})
    assert tok["morph"] == {"Tense": "Past"}
    tok = _lattice_token_to_ud({"pos": "verb", "dep_rel": "root", "flags": ["infinitive"]})
    assert tok["morph"] == {"VerbForm": "Inf"}
    tok = _lattice_token_to_ud({"pos": "verb", "dep_rel": "root", "flags": []})
    assert tok["morph"] == {}


def test_lattice_token_ids_and_form_kept():
    """Index, tête, lemme, forme observés — jamais inventés."""
    tok = _lattice_token_to_ud(_LATTICE_TWO_CLAUSES["tokens"][2])
    assert tok["id"] == 3 and tok["dep_head"] == 5
    assert tok["lemma"] == "si" and tok["form"] == "si"
    assert tok["dep_rel"] == "mark"


# ── Tests _reps_from_lattice ──────────────────────────────────────────────────


def test_reps_from_lattice_two_clauses():
    """2 clauses → 2 reps + 1 connecteur (token mark réel)."""
    reps, connectors = _reps_from_lattice(_LATTICE_TWO_CLAUSES)
    assert len(reps) == 2
    assert len(connectors) == 1
    assert connectors[0] is not None
    assert connectors[0].root_lemma == "si"
    assert connectors[0].root_pos == "SCONJ"
    assert connectors[0].root_dep_rel == "mark"


def test_reps_from_lattice_root_lemmas():
    """Racines = verbes observés (venir), pas de lemme inventé."""
    reps, _ = _reps_from_lattice(_LATTICE_TWO_CLAUSES)
    assert reps[0].root_lemma == "venir"
    assert reps[1].root_lemma == "venir"


def test_reps_from_lattice_empty():
    """Lattice vide → ([], []) sans exception."""
    assert _reps_from_lattice({"tokens": [], "clauses": []}) == ([], [])
    assert _reps_from_lattice({}) == ([], [])


# ── Tests _call_gcn_lattice (subprocess mocké) ────────────────────────────────


def _make_mock_result(lattice_dict, returncode=0, stderr=""):
    mock = MagicMock()
    mock.returncode = returncode
    mock.stdout = json.dumps(lattice_dict)
    mock.stderr = stderr
    return mock


@patch("gcn_python.frontend.bridge._resolve_gcn_bin", return_value="/usr/bin/gcn")
@patch("gcn_python.frontend.bridge.subprocess.run")
def test_parse_mocked_success(mock_run, _resolve):
    """Workflow complet mocké → 2 reps avec racines observées."""
    mock_run.return_value = _make_mock_result(_LATTICE_TWO_CLAUSES)
    reps, _ = GCNLatticeParser(gcn_bin="/usr/bin/gcn").parse("Il venait si tu venais.")
    assert len(reps) == 2
    assert reps[0].root_lemma == "venir"


@patch("gcn_python.frontend.bridge._resolve_gcn_bin", return_value="/usr/bin/gcn")
@patch("gcn_python.frontend.bridge.subprocess.run")
def test_parse_cmd_has_no_data_dir(mock_run, _resolve):
    """Doctrine : aucun --data-dir transmis (zéro YAML au runtime)."""
    mock_run.return_value = _make_mock_result(_LATTICE_TWO_CLAUSES)
    GCNLatticeParser(gcn_bin="/usr/bin/gcn").parse("x")
    cmd = mock_run.call_args[0][0]
    assert "--data-dir" not in cmd, f"--data-dir interdit : {cmd}"
    assert cmd[1] == "analyze" and "--" in cmd


def test_parse_gcn_not_found():
    """gcn absent du PATH → GCNBridgeError avec 'introuvable'."""
    with (patch("shutil.which", return_value=None),
          pytest.raises(GCNBridgeError, match="introuvable")):
        GCNLatticeParser(gcn_bin="gcn").parse("test")


@patch("gcn_python.frontend.bridge._resolve_gcn_bin", return_value="/usr/bin/gcn")
@patch("gcn_python.frontend.bridge.subprocess.run")
def test_parse_timeout(mock_run, _resolve):
    """TimeoutExpired → GCNBridgeError avec 'Timeout'."""
    import subprocess
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="gcn", timeout=30)
    with pytest.raises(GCNBridgeError, match="Timeout"):
        GCNLatticeParser(gcn_bin="/usr/bin/gcn").parse("test")


@patch("gcn_python.frontend.bridge._resolve_gcn_bin", return_value="/usr/bin/gcn")
@patch("gcn_python.frontend.bridge.subprocess.run")
def test_parse_nonzero_return(mock_run, _resolve):
    """returncode=1 → GCNBridgeError avec 'échoué'."""
    mock_run.return_value = _make_mock_result({}, returncode=1, stderr="erreur")
    with pytest.raises(GCNBridgeError, match="échoué"):
        GCNLatticeParser(gcn_bin="/usr/bin/gcn").parse("test")


@patch("gcn_python.frontend.bridge._resolve_gcn_bin", return_value="/usr/bin/gcn")
@patch("gcn_python.frontend.bridge.subprocess.run")
def test_parse_invalid_json(mock_run, _resolve):
    """Sortie non-JSON → GCNBridgeError avec 'JSON'."""
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = "not valid json"
    mock_run.return_value = mock
    with pytest.raises(GCNBridgeError, match="JSON"):
        GCNLatticeParser(gcn_bin="/usr/bin/gcn").parse("test")


def _find_gcn_bin():
    """Chemin du binaire gcn (PATH, sinon artefact du workspace Rust)."""
    found = shutil.which("gcn")
    if found:
        return found
    root = Path(__file__).resolve().parents[2]
    for profile in ("release", "debug"):
        candidate = root / "gcn-core" / "target" / profile / "gcn"
        if candidate.is_file():
            return str(candidate)
    return None


GCN_BIN = _find_gcn_bin()


@pytest.mark.skipif(
    GCN_BIN is None,
    reason="Binaire gcn-cli absent (cargo build --workspace)",
)
def test_parse_integration_real_binary():
    """Intégration vrai binaire : tokens réels, spans valides."""
    reps, _ = GCNLatticeParser(gcn_bin=GCN_BIN).parse("Le médicament réduit la douleur.")
    assert len(reps) >= 1
    for rep in reps:
        assert rep.root_lemma, "root_lemma vide"
        start, end = rep.token_span
        assert 0 <= start <= end, f"token_span invalide : {rep.token_span}"


# ── Tests CGNPipeline.analyze() et analyze_or_skip() ─────────────────────────


def _make_pipeline():
    """Helper : pipeline ML minimal pour tests."""
    from conftest import make_test_pipeline
    return make_test_pipeline()


@patch("gcn_python.frontend.bridge._resolve_gcn_bin", return_value="/usr/bin/gcn")
@patch("gcn_python.frontend.bridge.subprocess.run")
def test_pipeline_analyze_returns_cir(mock_run, _resolve):
    """pipeline.analyze() retourne un CausalIR dict avec les champs attendus."""
    mock_run.return_value = _make_mock_result(_LATTICE_TWO_CLAUSES)
    pipeline = _make_pipeline()
    cir = pipeline.analyze("Il venait si tu venais.")
    assert "nodes" in cir
    assert "edges" in cir
    assert len(cir["nodes"]) == 2


def test_pipeline_analyze_or_skip_none_without_binary():
    """analyze_or_skip() retourne None si le binaire gcn est absent."""
    pipeline = _make_pipeline()
    result = pipeline.analyze_or_skip("test", gcn_bin="nonexistent-gcn-binary-xyz")
    assert result is None


def test_pipeline_analyze_or_skip_no_double_warning():
    """analyze_or_skip() n'émet aucun UserWarning (dégradation gracieuse silencieuse)."""
    pipeline = _make_pipeline()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = pipeline.analyze_or_skip("test", gcn_bin="nonexistent-gcn-binary-xyz")
    assert result is None
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    assert len(user_warnings) == 0, f"Warning inattendu : {[str(x.message) for x in user_warnings]}"


# ── Subcommand libre, aucune langue en dur ────────────────────────────────────

@patch("gcn_python.frontend.bridge._resolve_gcn_bin", return_value="/usr/bin/gcn")
@patch("gcn_python.frontend.bridge.subprocess.run")
def test_call_gcn_lattice_routes_subcommand(mock_run, _resolve):
    """La sous-commande est routée telle quelle (défaut analyze, ex. analyze-en)."""
    mock_run.return_value = _make_mock_result(_LATTICE_TWO_CLAUSES)
    _call_gcn_lattice("x", "gcn", "analyze")
    assert mock_run.call_args[0][0][1] == "analyze"
    _call_gcn_lattice("x", "gcn", "analyze-en")
    assert mock_run.call_args[0][0][1] == "analyze-en"


def test_call_gcn_lattice_empty_subcommand_rejected():
    """Sous-commande vide → GCNBridgeError (pas d'appel)."""
    with pytest.raises(GCNBridgeError, match="subcommand"):
        _call_gcn_lattice("x", "gcn", "  ")


def test_lattice_parser_default_subcommand():
    """Défaut analyze, pas de littéral de langue stocké."""
    assert GCNLatticeParser().subcommand == "analyze"
    assert GCNLatticeParser(subcommand="analyze-en").subcommand == "analyze-en"
