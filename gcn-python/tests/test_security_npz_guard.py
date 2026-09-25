# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Sûreté des checkpoints .npz — garde anti-RCE avant désérialisation.

Le point de vigilance est l'ordre : un .npz piégé doit être refusé SANS que
son pickle ait jamais été exécuté. Le marqueur ci-dessous est posé par le
« payload » si — et seulement si — le pickle malveillant est désérialisé.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gcn_python.security import (
    UnsafeCheckpointError,
    assert_npz_pickle_safe,
    guarded_np_load,
)

_MARKER: dict[str, bool] = {"executed": False}


def _pwn(*_args, **_kwargs):
    """Cible du payload : exécutée uniquement si le pickle est désérialisé."""
    _MARKER["executed"] = True
    return "PWNED"


class _Evil:
    def __reduce__(self):
        return (_pwn, ("via __reduce__",))


@pytest.fixture(autouse=True)
def _reset_marker():
    _MARKER["executed"] = False
    yield
    _MARKER["executed"] = False


def _write_malicious_npz(path: Path) -> Path:
    """Un .npz dont _arch_json contient un objet à __reduce__ exécutable."""
    np.savez(path, _arch_json=np.array([_Evil()], dtype=object))
    return path


def _write_benign_npz(path: Path) -> Path:
    np.savez(
        path,
        _arch_json=np.array([json.dumps({"d_eff": 4, "graph_class": "RGCNLayer"})],
                            dtype=object),
        encoder_0=np.zeros(3, dtype=np.float32),
    )
    return path


def test_guard_refuses_malicious_object_member(tmp_path: Path):
    """Le pickle malveillant est rejeté AVANT toute exécution."""
    ckpt = _write_malicious_npz(tmp_path / "evil.npz")
    with pytest.raises(UnsafeCheckpointError, match="pickle interdit"):
        assert_npz_pickle_safe(ckpt)
    assert _MARKER["executed"] is False, (
        "le payload __reduce__ s'est exécuté — la garde est placée APRÈS la "
        "désérialisation (régression de l'ordre de vérification)"
    )


def test_guarded_np_load_refuses_before_opening(tmp_path: Path):
    ckpt = _write_malicious_npz(tmp_path / "evil.npz")
    with pytest.raises(UnsafeCheckpointError):
        guarded_np_load(ckpt)
    assert _MARKER["executed"] is False


def test_load_checkpoint_guard_applies_even_when_trusted(tmp_path: Path):
    """trusted=True n'est pas un passe-droit : l'audit pickle reste appliqué."""
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import load_checkpoint

    vocab = FeatureVocabulary()
    pipe = CGNPipeline(
        encoder=MLPEncoder(d_clause=vocab.d_clause,
                           d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7)),
        graph=RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause),
        vocabulary=vocab,
    )
    ckpt = _write_malicious_npz(tmp_path / "evil.npz")
    with pytest.raises(UnsafeCheckpointError):
        load_checkpoint(pipe, ckpt, trusted=True)
    assert _MARKER["executed"] is False


def test_run_eval_refuses_malicious_checkpoint_before_building(tmp_path: Path):
    from gcn_python.evaluation.eval_runner import run_eval

    ckpt = _write_malicious_npz(tmp_path / "evil.npz")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with pytest.raises(UnsafeCheckpointError):
        run_eval(data_dir, ckpt)
    assert _MARKER["executed"] is False


def test_run_eval_refuses_checkpoint_without_arch_json(tmp_path: Path):
    """Refus immédiat : avant la construction du pipeline (pas après)."""
    from gcn_python.evaluation.eval_runner import run_eval

    ckpt = tmp_path / "no_arch.npz"
    np.savez(ckpt, encoder_0=np.zeros(2, dtype=np.float32))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with pytest.raises(ValueError, match="ne contient pas _arch_json"):
        run_eval(data_dir, ckpt)


def test_guard_accepts_legitimate_checkpoint(tmp_path: Path):
    """Un .npz conforme (métadonnées JSON + poids float) passe l'audit."""
    ckpt = _write_benign_npz(tmp_path / "ok.npz")
    assert_npz_pickle_safe(ckpt)
    with guarded_np_load(ckpt) as raw:
        assert "_arch_json" in raw.files
        assert json.loads(str(raw["_arch_json"][0]))["d_eff"] == 4
        np.testing.assert_array_equal(raw["encoder_0"], np.zeros(3, dtype=np.float32))


def test_guard_accepts_checkpoint_produced_by_save_checkpoint(tmp_path: Path):
    """Round-trip réel : save_checkpoint → guarded_np_load sans erreur."""
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import load_checkpoint, save_checkpoint

    vocab = FeatureVocabulary()
    pipe = CGNPipeline(
        encoder=MLPEncoder(d_clause=vocab.d_clause,
                           d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7)),
        graph=RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause),
        vocabulary=vocab,
    )
    ckpt = tmp_path / "real.npz"
    save_checkpoint(pipe, ckpt)
    assert_npz_pickle_safe(ckpt)
    load_checkpoint(pipe, ckpt, trusted=True)


def test_guard_rejects_non_archive(tmp_path: Path):
    bogus = tmp_path / "not_an_npz.npz"
    bogus.write_text("pas une archive", encoding="utf-8")
    with pytest.raises(ValueError, match="pas une archive"):
        assert_npz_pickle_safe(bogus)
