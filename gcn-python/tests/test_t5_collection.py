# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Régression B2 (audit) : `collect_val_logits_for_calibration` collecte
réellement des logits alignés sur le gold (non-régression du bug T5-min
silencieuse §2.7 — le `pipeline.forward(str)` ne peuplait jamais le cache)."""
import json

import numpy as np

from conftest import make_test_pipeline
from gcn_python.constants import RELATION_TYPES
from gcn_python.data.loader import GCNDataLoader
from gcn_python.evaluation.calibration import optimize_temperature
from gcn_python.training.train import collect_val_logits_for_calibration


def _sentence(sid: str, relation: str | None):
    toks = [
        {"id": 1, "form": "A", "lemma": "pluie", "pos": "NOUN",
         "dep_rel": "nsubj", "dep_head": 2, "morph": {}},
        {"id": 2, "form": "cause", "lemma": "causer", "pos": "VERB",
         "dep_rel": "root", "dep_head": 0, "morph": {}},
        {"id": 3, "form": "B", "lemma": "recolte", "pos": "NOUN",
         "dep_rel": "obj", "dep_head": 2, "morph": {}},
    ]
    nodes = [
        {"id": "n001", "type": "processus", "label": "A",
         "token_span": [1, 1], "scope": "specific",
         "temporal_index": 0, "origin": "explicit"},
        {"id": "n002", "type": "processus", "label": "B",
         "token_span": [2, 3], "scope": "specific",
         "temporal_index": 1, "origin": "explicit"},
    ]
    edges = ([{"source": "n001", "target": "n002", "relation": relation,
               "confidence": None, "explicit": True, "negated": None}]
             if relation else [])
    return {"id": sid, "text": f"A {relation or 'seul'} B.",
            "tokens": toks, "cir": {"nodes": nodes, "edges": edges}}


def _val_dir(tmp_path, relations):
    d = tmp_path / "val_t5"
    d.mkdir()
    doc = {"document": {"sentences": [
        _sentence(f"s{i}", r) for i, r in enumerate(relations)]}}
    (d / "val.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return d


def test_collecte_logtis_alignes_gold(tmp_path):
    """2 phrases avec arêtes + 1 sans : 2 blocs de logits, labels exacts, 0 skipped."""
    loader = GCNDataLoader(_val_dir(tmp_path, ["cause", "enable", None]),
                           all_pairs=True, shuffle=False, silver_weight=1.0)
    pipeline = make_test_pipeline(all_pairs=True)
    logits, labels, skipped = collect_val_logits_for_calibration(loader, pipeline, True)
    assert skipped == 0
    assert len(logits) == 2 and len(labels) == 2
    assert [RELATION_TYPES[i] for i in labels] == ["cause", "enable"]


def test_collecte_alimente_optimize_temperature(tmp_path):
    """12 phrases → gate N>=10 satisfaite → température optimisée finie."""
    rels = ["cause", "enable", "prevent"] * 4
    loader = GCNDataLoader(_val_dir(tmp_path, rels),
                           all_pairs=True, shuffle=False, silver_weight=1.0)
    pipeline = make_test_pipeline(all_pairs=True)
    logits, labels, _ = collect_val_logits_for_calibration(loader, pipeline, True)
    assert len(labels) == 12
    all_logits = np.vstack(logits)[:len(labels)]
    best_t, best_ece = optimize_temperature(
        all_logits, np.array(labels[:len(all_logits)], dtype=np.int64))
    assert np.isfinite(best_t) and best_t > 0
    assert np.isfinite(best_ece)
