# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests convertisseur legacy-UD → refacto + refus en eval + ban final/ (R4)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

RACINE = Path(__file__).resolve().parents[2]
SCRIPT = RACINE / "scripts" / "convert_legacy_ud.py"

from gcn_python.layer0.tokenizer import normalize_token, stable_id, token_family
from gcn_python.layer1.tok_embedding import TokEmbedding
from gcn_python.training.checkpoint import (
    LegacyUDError,
    checkpoint_format_kind,
    load_checkpoint,
)


def legacy_npz(path: Path) -> Path:
    """Mini legacy-UD : 6 lemmes (spéciaux + collision café/cafe) × dim 3."""
    lemmes = ["_unk", "ventes", "café", "cafe", "_subj_absent", "pluie"]
    rng = np.random.default_rng(11)
    E = rng.normal(0, 1, (len(lemmes), 3)).astype(np.float32)
    vocab_json = json.dumps({"lemmas": lemmes}, ensure_ascii=False)
    np.savez(path, word_emb_E=E,
             _word_emb_vocab_json=np.array([vocab_json], dtype=object))
    return path


def lance(*args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"}
    if env_extra:
        env.update(env_extra)
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True,
                          text=True, check=False, cwd=RACINE, env=env)


def test_kind_legacy_refacto_unknown(tmp_path):
    leg = legacy_npz(tmp_path / "leg.npz")
    assert checkpoint_format_kind(leg) == "legacy-ud"
    np.savez(tmp_path / "autre.npz", x=np.zeros(2))
    assert checkpoint_format_kind(tmp_path / "autre.npz") == "unknown"
    with pytest.raises(FileNotFoundError):
        checkpoint_format_kind(tmp_path / "absent.npz")


def test_dry_run_necrit_rien(tmp_path):
    leg = legacy_npz(tmp_path / "leg.npz")
    sortie = tmp_path / "conv.npz"
    avant = (os.stat(leg).st_mtime_ns,
             hashlib.sha256(leg.read_bytes()).hexdigest())
    proc = lance(str(leg), "--output", str(sortie), "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert not sortie.exists(), "dry-run : rien d'écrit"
    apres = (os.stat(leg).st_mtime_ns,
             hashlib.sha256(leg.read_bytes()).hexdigest())
    assert apres == avant, "dry-run : entrée intacte (mtime + hash)"
    rapport = json.loads(proc.stdout)
    assert rapport["dry_run"] is True and rapport["entree_inchangee"] is True
    assert rapport["n_collisions"] == 1  # café/cafe → même id
    assert rapport["n_abandonnes"] == 2  # _unk + _subj_absent
    assert rapport["n_mappees"] == 3


def test_reel_deterministe_et_equivalent(tmp_path):
    leg = legacy_npz(tmp_path / "leg.npz")
    s1 = tmp_path / "c1.npz"
    s2 = tmp_path / "c2.npz"
    assert lance(str(leg), "--output", str(s1)).returncode == 0
    assert lance(str(leg), "--output", str(s2)).returncode == 0
    assert hashlib.sha256(s1.read_bytes()).hexdigest() == \
        hashlib.sha256(s2.read_bytes()).hexdigest(), "déterminisme (gold hash)"
    assert checkpoint_format_kind(s1) == "refacto"
    conv = TokEmbedding.load_npz(str(s1))
    assert conv.d_model == 3
    # Équivalence forward : lookup legacy == lookup converti par lemme.
    E_leg = dict(zip(
        ["_unk", "ventes", "café", "cafe", "_subj_absent", "pluie"],
        np.load(leg, allow_pickle=True)["word_emb_E"].tolist(),
        strict=True,
    ))
    for lemme, attendu in (("ventes", "ventes"), ("cafe", "cafe"), ("pluie", "pluie")):
        tid = stable_id(normalize_token(lemme, token_family(lemme)))
        assert np.allclose(conv.lookup(tid), E_leg[attendu]), lemme
    # Collision : premier en ordre trié gagne (cafe < café).
    assert np.allclose(
        conv.lookup(stable_id(normalize_token("cafe", token_family("cafe")))),
        E_leg["cafe"],
    )


def test_refacto_refuse_reconversion(tmp_path):
    leg = legacy_npz(tmp_path / "leg.npz")
    s1 = tmp_path / "c1.npz"
    assert lance(str(leg), "--output", str(s1)).returncode == 0
    proc = lance(str(s1), "--output", str(tmp_path / "c2.npz"))
    assert proc.returncode != 0
    assert "pas un legacy-UD" in proc.stdout + proc.stderr


def test_d_model_incompatible(tmp_path):
    leg = legacy_npz(tmp_path / "leg.npz")
    proc = lance(str(leg), "--output", str(tmp_path / "c.npz"), "--d-model", "128")
    assert proc.returncode != 0


def test_refus_eval_legacy(tmp_path):
    from gcn_python.evaluation.eval_runner import run_eval

    leg = legacy_npz(tmp_path / "leg.npz")
    donnees = tmp_path / "val_ok"
    donnees.mkdir()
    with pytest.raises(LegacyUDError, match="legacy-UD : convertir ou ré-entraîner"):
        run_eval(donnees, leg)


def test_load_checkpoint_gate_directe(tmp_path):
    leg = legacy_npz(tmp_path / "leg.npz")
    with pytest.raises(LegacyUDError, match="legacy-UD"):
        load_checkpoint(None, leg, trusted=True, allow_legacy=False)


def test_ban_final_en_eval(tmp_path):
    from gcn_python.evaluation.eval_runner import run_eval

    leg = legacy_npz(tmp_path / "leg.npz")
    final = tmp_path / "final"
    final.mkdir()
    with pytest.raises(ValueError, match="interdit"):
        run_eval(final, leg)
