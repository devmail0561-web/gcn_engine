# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests de régression pour les correctifs du Lot 1 (audit 2026-09-21).

V1 : eval_runner.py — training=False positionné avant l'évaluation.
V2 : eval_runner.py — vocab chargé depuis le checkpoint avant l'encodeur.
V3 : cgnp.py — gradient word_embedding utilise d_curr (post-RGCN) pas d_enriched.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from gcn_python.layer1.features import FeatureVocabulary
from gcn_python.layer1.representation import UDRepresentation
from gcn_python.layer2.reference import MLPEncoder
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.pipeline.cgnp import CGNPipeline
from gcn_python.training.checkpoint import save_checkpoint


# ---------------------------------------------------------------------------
# Helpers partagés
# ---------------------------------------------------------------------------

def _make_rep(lemma: str = "baisser") -> UDRepresentation:
    return UDRepresentation(
        tokens=[{"lemma": lemma, "pos": "VERB", "dep_rel": "root", "morph": {}}],
        root_lemma=lemma,
        root_pos="VERB",
        root_dep_rel="root",
        root_morph={"Tense": "Pres"},
        subject_pos="NOUN",
        has_object=False,
        has_advcl=False,
        has_temporal_obl=False,
        token_span=(1, 2),
    )


def _make_pipeline(vocab=None, **kwargs) -> CGNPipeline:
    if vocab is None:
        vocab = FeatureVocabulary()
    d_edge_cl = vocab.d_edge_closed_loop(vocab.d_clause, 7)
    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=d_edge_cl)
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    return CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab, **kwargs)


def _minimal_dataset_json(text: str = "test phrase.") -> dict:
    """Génère un document gcn-nl minimal avec une seule phrase sans arêtes."""
    return {
        "document": {
            "sentences": [
                {
                    "id": "s001",
                    "text": text,
                    "tokens": [
                        {
                            "id": 1, "form": "test", "lemma": "test",
                            "pos": "NOUN", "dep_rel": "root", "dep_head": 0, "morph": {},
                        }
                    ],
                    "cir": {
                        "nodes": [
                            {
                                "id": "n001", "type": "action", "label": "test",
                                "token_span": [1, 1], "scope": "specific",
                                "temporal_index": 0, "origin": "explicit",
                            }
                        ],
                        "edges": [],
                    },
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# V1 — eval_runner : training=False positionné lors de l'évaluation
# ---------------------------------------------------------------------------

def test_run_eval_training_false(tmp_path: Path):
    """V1 : run_eval doit évaluer le pipeline en mode inference (training=False).

    Avant le correctif, MLPEncoder.training restait True → edge_dropout actif →
    métriques val non-déterministes.
    """
    from gcn_python.evaluation.eval_runner import run_eval

    pipeline = _make_pipeline()
    ckpt = tmp_path / "model.npz"
    save_checkpoint(pipeline, ckpt)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.json").write_text(
        json.dumps(_minimal_dataset_json()), encoding="utf-8"
    )

    training_states: list[bool] = []

    orig_forward = CGNPipeline.forward
    def capture_forward(self, *args, **kwargs):
        training_states.append(bool(self.encoder.training))
        return orig_forward(self, *args, **kwargs)

    with patch.object(CGNPipeline, "forward", capture_forward):
        run_eval(data_dir, ckpt)

    assert training_states, "forward() n'a pas été appelé — dataset vide?"
    assert not any(training_states), (
        f"encoder.training={training_states} pendant run_eval — "
        "dropout actif → métriques non-déterministes (correctif V1 manquant)."
    )


# ---------------------------------------------------------------------------
# V2 — eval_runner : vocab chargé depuis le checkpoint avant l'encodeur
# ---------------------------------------------------------------------------

def test_run_eval_vocab_restored_from_checkpoint(tmp_path: Path):
    """V2 : run_eval doit charger _vocab_json avant de construire l'encodeur.

    Avant le correctif, un modèle entraîné avec connector_lemmas provoquait
    un crash ValueError sur shape mismatch dans load_checkpoint.
    """
    from gcn_python.evaluation.eval_runner import run_eval

    # Modèle entraîné avec connector_lemmas : d_edge ≠ vocab vide
    vocab_with_lemmas = FeatureVocabulary(connector_lemmas=["parce", "car", "because"])
    d_eff = vocab_with_lemmas.d_clause
    d_edge = vocab_with_lemmas.d_edge_closed_loop(d_eff, 7)
    encoder = MLPEncoder(d_clause=d_eff, d_edge=d_edge)
    graph = RGCNLayer(d_in=d_eff, d_out=d_eff)
    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab_with_lemmas)

    ckpt = tmp_path / "model_with_lemmas.npz"
    save_checkpoint(pipeline, ckpt)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.json").write_text(
        json.dumps(_minimal_dataset_json()), encoding="utf-8"
    )

    # Avant correctif : ValueError sur encoder_0 shape mismatch
    # Après correctif : doit passer sans exception
    result = run_eval(data_dir, ckpt)
    assert "n_samples" in result
    assert "node_accuracy" in result


# ---------------------------------------------------------------------------
# V3 — cgnp.py backward : d_curr utilisé pour le gradient word_embedding
# ---------------------------------------------------------------------------

def test_word_embedding_gradient_received_after_backward():
    """V3 smoke : word_embedding.backward() doit être appelé après backward().

    Vérifie que le code path est actif (le gradient atteint les embeddings).
    """
    from gcn_python.layer1.embedding import WordEmbedding

    d_emb = 4
    vocab = FeatureVocabulary()
    d_eff = vocab.d_clause + d_emb
    d_edge_cl = vocab.d_edge_closed_loop(d_eff, 7, d_emb)
    we = WordEmbedding(d_emb=d_emb, seed=0)
    we.add_lemma("baisser")
    we.add_lemma("hausser")

    encoder = MLPEncoder(d_clause=d_eff, d_edge=d_edge_cl, seed=0)
    graph = RGCNLayer(d_in=d_eff, d_out=d_eff, seed=0)
    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab,
                           word_embedding=we)

    reps = [_make_rep("baisser"), _make_rep("hausser")]
    pipeline.forward(reps, "test")
    node_logits = pipeline._cached_node_logits
    d_node = np.ones_like(node_logits) / node_logits.size
    d_edge = np.zeros((0, 11), dtype=np.float32)

    # Capturer les appels à word_embedding.backward
    calls: list[tuple] = []
    original_bwd = we.backward
    def patched_bwd(d, lemma):
        calls.append((d.copy(), lemma))
        original_bwd(d, lemma)
    we.backward = patched_bwd

    pipeline.backward(d_node, d_edge, lr=1e-9)

    assert len(calls) == 2, (
        f"word_embedding.backward appelé {len(calls)} fois pour 2 reps — "
        "gradient n'atteint pas les embeddings (correctif V3 manquant)."
    )
    for grad, lemma in calls:
        assert grad.shape == (d_emb,)
        assert lemma in ("baisser", "hausser")


def test_word_embedding_gradient_1clause_no_unboundlocalerror():
    """V3 hotfix régression : 1 clause → _cached_edge_index=None → d_curr non initialisé
    à l'intérieur du if → UnboundLocalError avant le hotfix.
    """
    from gcn_python.layer1.embedding import WordEmbedding

    d_emb = 4
    vocab = FeatureVocabulary()
    d_eff = vocab.d_clause + d_emb
    d_edge_cl = vocab.d_edge_closed_loop(d_eff, 7, d_emb)
    we = WordEmbedding(d_emb=d_emb, seed=0)
    we.add_lemma("solo")

    encoder = MLPEncoder(d_clause=d_eff, d_edge=d_edge_cl, seed=0)
    graph = RGCNLayer(d_in=d_eff, d_out=d_eff, seed=0)
    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab,
                           word_embedding=we)

    # 1 seul rep → pas d'arête → _cached_edge_index reste None
    pipeline.forward([_make_rep("solo")], "test")
    assert pipeline._cached_edge_index is None or pipeline._cached_edge_index.shape[1] == 0, \
        "Précondition : pas d'arête pour 1 clause"

    node_logits = pipeline._cached_node_logits
    d_node = np.ones_like(node_logits) / node_logits.size
    d_edge = np.zeros((0, 11), dtype=np.float32)

    # Ne doit pas lever UnboundLocalError
    calls: list = []
    orig_bwd = we.backward
    def patched_bwd(d, lemma):
        calls.append(d.copy())
        orig_bwd(d, lemma)
    we.backward = patched_bwd

    pipeline.backward(d_node, d_edge, lr=1e-9)
    assert len(calls) == 1, "word_embedding.backward doit être appelé pour 1 clause"


def test_word_embedding_gradient_accumulate_1clause_no_unboundlocalerror():
    """V3 hotfix : backward_accumulate avec 1 clause ne doit pas lever UnboundLocalError."""
    from gcn_python.layer1.embedding import WordEmbedding

    d_emb = 4
    vocab = FeatureVocabulary()
    d_eff = vocab.d_clause + d_emb
    d_edge_cl = vocab.d_edge_closed_loop(d_eff, 7, d_emb)
    we = WordEmbedding(d_emb=d_emb, seed=0)
    we.add_lemma("solo")

    encoder = MLPEncoder(d_clause=d_eff, d_edge=d_edge_cl, seed=0)
    graph = RGCNLayer(d_in=d_eff, d_out=d_eff, seed=0)
    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab,
                           word_embedding=we)

    pipeline.forward([_make_rep("solo")], "test")
    node_logits = pipeline._cached_node_logits
    d_node = np.ones_like(node_logits) / node_logits.size
    d_edge = np.zeros((0, 11), dtype=np.float32)

    calls: list = []
    orig_bwd = we.backward
    def patched_bwd(d, lemma):
        calls.append(d.copy())
        orig_bwd(d, lemma)
    we.backward = patched_bwd

    # Ne doit pas lever UnboundLocalError
    pipeline.backward_accumulate(d_node, d_edge)
    assert len(calls) == 1, "word_embedding.backward doit être appelé en accumulate aussi"


def test_word_embedding_gradient_uses_dcurr_not_denriched():
    """V3 régression : le gradient embedding doit venir de d_curr (post-RGCN).

    Vérifie que le gradient reçu par word_embedding correspond au slice de d_curr
    (qui traverse le RGCN backward) et non à d_enriched (qui ne le traverse pas).
    Avec un RGCN dont W_0 ≠ identité, d_curr[:, d_clause:] ≠ d_enriched[:, d_clause:].
    """
    from gcn_python.layer1.embedding import WordEmbedding

    d_emb = 4
    vocab = FeatureVocabulary()
    d_eff = vocab.d_clause + d_emb
    d_edge_cl = vocab.d_edge_closed_loop(d_eff, 7, d_emb)
    we = WordEmbedding(d_emb=d_emb, seed=7)
    we.add_lemma("alpha")
    we.add_lemma("beta")

    encoder = MLPEncoder(d_clause=d_eff, d_edge=d_edge_cl, seed=7)
    graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=11, seed=7)
    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab,
                           word_embedding=we)

    reps = [_make_rep("alpha"), _make_rep("beta")]
    pipeline.forward(reps, "test")

    node_logits = pipeline._cached_node_logits
    d_node = np.ones_like(node_logits) / node_logits.size
    d_edge = np.zeros((0, 11), dtype=np.float32)

    # Capturer d_enriched (pré-RGCN) et d_curr (post-RGCN) pendant backward
    captured: dict = {}
    orig_bmp = graph.backward_message_pass
    def patched_bmp(d_out):
        captured["d_enriched_before"] = d_out.copy()
        d_in, grads = orig_bmp(d_out)
        captured["d_curr_after"] = d_in.copy()
        return d_in, grads
    graph.backward_message_pass = patched_bmp

    grads_received: list = []
    orig_we_bwd = we.backward
    def patched_we_bwd(d, lemma):
        grads_received.append(d.copy())
        orig_we_bwd(d, lemma)
    we.backward = patched_we_bwd

    pipeline.backward(d_node, d_edge, lr=1e-9)

    assert "d_curr_after" in captured, "backward_message_pass n'a pas été appelé"
    assert len(grads_received) == 2

    d_clause = vocab.d_clause
    # Le gradient embedding reçu doit correspondre à d_curr[:, d_clause:] (post-RGCN)
    # et PAS à d_enriched[:, d_clause:] (pré-RGCN)
    d_curr_emb_slice = captured["d_curr_after"][:, d_clause:]
    d_enriched_emb_slice = captured["d_enriched_before"][:, d_clause:]

    # Vérifier que d_curr ≠ d_enriched (sinon le test ne distingue rien)
    if np.allclose(d_curr_emb_slice, d_enriched_emb_slice):
        pytest.skip("d_curr == d_enriched pour ce seed — test non discriminant")

    # Le gradient reçu par les embeddings doit correspondre à d_curr, pas d_enriched
    received_sum = sum(g for g in grads_received)
    expected_from_dcurr = d_curr_emb_slice.sum(axis=0)
    expected_from_denriched = d_enriched_emb_slice.sum(axis=0)

    err_dcurr = float(np.linalg.norm(received_sum - expected_from_dcurr))
    err_denriched = float(np.linalg.norm(received_sum - expected_from_denriched))

    assert err_dcurr < err_denriched, (
        f"Le gradient embedding est plus proche de d_enriched ({err_denriched:.4f}) "
        f"que de d_curr ({err_dcurr:.4f}) — correctif V3 manquant ou incorrect."
    )


# ---------------------------------------------------------------------------
# Sécu+held-out — edge_threshold/drop_morph dans _arch_json, --test-dir
# ---------------------------------------------------------------------------

def test_run_eval_restores_edge_threshold_from_arch(tmp_path: Path):
    """edge_threshold restauré depuis _arch_json dans run_eval.

    Si le modèle a été entraîné avec edge_threshold=0.5 et que run_eval ne
    restaure pas la valeur, le pipeline évalue avec seuil 0.0 → métriques
    optimistes (toutes les arêtes émises, y compris les low-confidence).
    """
    from gcn_python.evaluation.eval_runner import run_eval

    pipeline = _make_pipeline()
    pipeline.edge_threshold = 0.5
    ckpt = tmp_path / "model_thr.npz"
    save_checkpoint(pipeline, ckpt)

    # Vérifier que l'arch JSON contient bien edge_threshold
    import json as _json
    raw = np.load(ckpt, allow_pickle=True)
    arch = _json.loads(str(raw["_arch_json"][0]))
    assert arch["edge_threshold"] == pytest.approx(0.5), "arch doit stocker edge_threshold"

    data_dir = tmp_path / "data2"
    data_dir.mkdir()
    (data_dir / "s.json").write_text(
        json.dumps(_minimal_dataset_json()), encoding="utf-8"
    )

    # Capturer le seuil utilisé lors du forward
    captured_thr: list[float] = []
    orig_forward = CGNPipeline.forward
    def capture_forward(self, *args, **kwargs):
        captured_thr.append(self.edge_threshold)
        return orig_forward(self, *args, **kwargs)

    with patch.object(CGNPipeline, "forward", capture_forward):
        run_eval(data_dir, ckpt)

    assert captured_thr, "forward() n'a pas été appelé"
    assert all(abs(t - 0.5) < 1e-6 for t in captured_thr), (
        f"edge_threshold={captured_thr} attendu 0.5 — run_eval ne restaure pas "
        "la valeur depuis _arch_json."
    )


def test_run_eval_edge_threshold_override(tmp_path: Path):
    """edge_threshold_override surcharge la valeur du checkpoint."""
    from gcn_python.evaluation.eval_runner import run_eval

    pipeline = _make_pipeline()
    pipeline.edge_threshold = 0.3
    ckpt = tmp_path / "model_ov.npz"
    save_checkpoint(pipeline, ckpt)

    data_dir = tmp_path / "data3"
    data_dir.mkdir()
    (data_dir / "s.json").write_text(
        json.dumps(_minimal_dataset_json()), encoding="utf-8"
    )

    captured_thr: list[float] = []
    orig_forward = CGNPipeline.forward
    def capture_forward(self, *args, **kwargs):
        captured_thr.append(self.edge_threshold)
        return orig_forward(self, *args, **kwargs)

    with patch.object(CGNPipeline, "forward", capture_forward):
        run_eval(data_dir, ckpt, edge_threshold_override=0.7)

    assert all(abs(t - 0.7) < 1e-6 for t in captured_thr), (
        f"edge_threshold={captured_thr} attendu 0.7 (override) — "
        "edge_threshold_override ignoré."
    )


def test_eval_cmd_test_dir_adds_test_keys(tmp_path: Path):
    """--test-dir ajoute les métriques held-out sous la clé 'test' dans le rapport."""
    from click.testing import CliRunner
    from gcn_python.evaluation.eval_runner import eval_cmd

    pipeline = _make_pipeline()
    ckpt = tmp_path / "model_td.npz"
    save_checkpoint(pipeline, ckpt)

    def _make_data_dir(name: str) -> Path:
        d = tmp_path / name
        d.mkdir()
        (d / "s.json").write_text(json.dumps(_minimal_dataset_json()), encoding="utf-8")
        return d

    val_dir = _make_data_dir("val")
    test_dir = _make_data_dir("test")

    runner = CliRunner()
    result = runner.invoke(eval_cmd, [
        "--data-dir", str(val_dir),
        "--model-path", str(ckpt),
        "--test-dir", str(test_dir),
    ])
    assert result.exit_code == 0, f"gcn-eval a échoué : {result.output}"
    report = json.loads(result.output)
    assert "test" in report, "La clé 'test' doit être présente quand --test-dir est passé"
    assert "node_macro_f1" in report["test"], "test.node_macro_f1 manquant"
    assert "edge_macro_f1" in report["test"], "test.edge_macro_f1 manquant"
    assert "n_samples" in report["test"], "test.n_samples manquant"


# ---------------------------------------------------------------------------
# 5ᵉ audit — éval all_pairs propagé au GCNDataLoader
# ---------------------------------------------------------------------------

def test_run_eval_all_pairs_passed_to_loader(tmp_path: Path):
    """all_pairs depuis _arch_json doit être propagé à GCNDataLoader.

    Avant le correctif, GCNDataLoader(data_dir) ignorait all_pairs=True →
    les arêtes gold avec gap > 1 étaient droppées de edge_map silencieusement →
    edge_macro_f1 structurellement sous-estimé sans aucun signal.
    """
    from gcn_python.evaluation.eval_runner import run_eval
    from gcn_python.data.loader import GCNDataLoader

    pipeline = _make_pipeline(all_pairs=True)
    ckpt = tmp_path / "model_ap.npz"
    save_checkpoint(pipeline, ckpt)

    data_dir = tmp_path / "data_ap"
    data_dir.mkdir()
    (data_dir / "s.json").write_text(json.dumps(_minimal_dataset_json()), encoding="utf-8")

    captured_all_pairs: list[bool] = []
    orig_init = GCNDataLoader.__init__

    def patched_init(self, data_dir_arg, *,
                     repeat=False, all_pairs=False, shuffle=False, seed=42):
        captured_all_pairs.append(all_pairs)
        orig_init(self, data_dir_arg, repeat=repeat, all_pairs=all_pairs,
                  shuffle=shuffle, seed=seed)

    with patch.object(GCNDataLoader, "__init__", patched_init):
        run_eval(data_dir, ckpt)

    assert captured_all_pairs, "GCNDataLoader.__init__ n'a pas été appelé"
    assert all(ap is True for ap in captured_all_pairs), (
        f"GCNDataLoader instancié avec all_pairs={captured_all_pairs} — "
        "devrait être True pour un modèle entraîné avec all_pairs=True "
        "(correctif éval all_pairs manquant)."
    )


def test_run_eval_all_pairs_gap2_edge_in_metrics(tmp_path: Path):
    """Preuve fonctionnelle : arête gap>1 dans edge_map quand all_pairs=True.

    Audit 7 fix : edge_macro_f1 retourne 0.0 (pas None) même quand all_edge_gold
    est vide — l'assertion is not None était tautologique.

    Preuve en deux étapes :
    1. GCNDataLoader direct (sans mock) : (0,2) ∈ edge_map avec all_pairs=True,
       absent avec all_pairs=False — discriminant réel sur loader.py:128-129.
    2. Intégration run_eval (patch transparent _to_sample) : assert len(edge_map) > 0
       pour au moins une sentence — prouve que l'arête traverse loader→run_eval.
    """
    from gcn_python.evaluation.eval_runner import run_eval
    from gcn_python.data.loader import GCNDataLoader

    pipeline = _make_pipeline(all_pairs=True)
    ckpt = tmp_path / "model_gap2.npz"
    save_checkpoint(pipeline, ckpt)

    # 3 clauses, 1 arête gap=2 (n001→n003, indices 0→2)
    doc = {
        "document": {
            "sentences": [{
                "id": "s001",
                "text": "A cause C via B.",
                "tokens": [
                    {"id": 1, "form": "A", "lemma": "A", "pos": "NOUN",
                     "dep_rel": "nsubj", "dep_head": 2, "morph": {}},
                    {"id": 2, "form": "cause", "lemma": "causer", "pos": "VERB",
                     "dep_rel": "root", "dep_head": 0, "morph": {}},
                    {"id": 3, "form": "C", "lemma": "C", "pos": "NOUN",
                     "dep_rel": "obj", "dep_head": 2, "morph": {}},
                ],
                "cir": {
                    "nodes": [
                        {"id": "n001", "type": "action", "label": "A",
                         "token_span": [1, 1], "scope": "specific",
                         "temporal_index": 0, "origin": "explicit"},
                        {"id": "n002", "type": "action", "label": "B",
                         "token_span": [2, 2], "scope": "specific",
                         "temporal_index": 1, "origin": "explicit"},
                        {"id": "n003", "type": "action", "label": "C",
                         "token_span": [3, 3], "scope": "specific",
                         "temporal_index": 2, "origin": "explicit"},
                    ],
                    "edges": [
                        {"source": "n001", "target": "n003", "relation": "cause",
                         "confidence": None, "explicit": True, "negated": None},
                    ],
                },
            }]
        }
    }
    data_dir = tmp_path / "data_gap2_func"
    data_dir.mkdir()
    (data_dir / "data.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    # Étape 1 : preuve structurelle directe via GCNDataLoader
    samples_true = list(GCNDataLoader(data_dir, all_pairs=True))
    assert samples_true and (0, 2) in samples_true[0].edge_map, (
        "GCNDataLoader(all_pairs=True) doit conserver l'arête gap=2 dans edge_map."
    )
    samples_false = list(GCNDataLoader(data_dir, all_pairs=False))
    assert samples_false and (0, 2) not in samples_false[0].edge_map, (
        "GCNDataLoader(all_pairs=False) doit dropper l'arête gap=2."
    )

    # Étape 2 : intégration — run_eval utilise all_pairs=True depuis l'arch ;
    # capturer all_edge_gold via forward pour prouver la traversée complète
    gold_edges_seen: list[int] = []
    orig_to_sample = GCNDataLoader._to_sample
    def patched_to_sample(self, rec):
        sample = orig_to_sample(self, rec)
        gold_edges_seen.append(len(sample.edge_map))
        return sample
    with patch.object(GCNDataLoader, "_to_sample", patched_to_sample):
        run_eval(data_dir, ckpt)
    assert gold_edges_seen and any(n > 0 for n in gold_edges_seen), (
        f"edge_map vide pour toutes les sentences ({gold_edges_seen}) — "
        "all_pairs non propagé au GCNDataLoader dans run_eval."
    )


def _make_dataset_with_edge(tmp_path: Path, name: str) -> Path:
    """Dataset minimal à 2 nœuds + 1 arête cause (pour tester le filtrage par seuil)."""
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    doc = {
        "document": {
            "sentences": [
                {
                    "id": "s001",
                    "text": "A cause B.",
                    "tokens": [
                        {"id": 1, "form": "A", "lemma": "A", "pos": "NOUN",
                         "dep_rel": "nsubj", "dep_head": 2, "morph": {}},
                        {"id": 2, "form": "cause", "lemma": "causer", "pos": "VERB",
                         "dep_rel": "root", "dep_head": 0, "morph": {}},
                        {"id": 3, "form": "B", "lemma": "B", "pos": "NOUN",
                         "dep_rel": "obj", "dep_head": 2, "morph": {}},
                    ],
                    "cir": {
                        "nodes": [
                            {"id": "n001", "type": "action", "label": "A",
                             "token_span": [1, 1], "scope": "specific",
                             "temporal_index": 0, "origin": "explicit"},
                            {"id": "n002", "type": "action", "label": "B",
                             "token_span": [3, 3], "scope": "specific",
                             "temporal_index": 1, "origin": "explicit"},
                        ],
                        "edges": [
                            {"source": "n001", "target": "n002", "relation": "cause",
                             "confidence": None, "explicit": True, "negated": None},
                        ],
                    },
                }
            ]
        }
    }
    (d / "data.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return d


def test_edge_threshold_filters_edges_in_forward(tmp_path: Path):
    """Preuve différentielle : seuil élevé réduit les arêtes émises vs seuil 0.

    Sans mock masquant — vérifie le comportement fonctionnel du filtrage,
    pas seulement la propagation d'attribut.
    """
    vocab = FeatureVocabulary()
    d_edge_cl = vocab.d_edge_closed_loop(vocab.d_clause, 7)
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=d_edge_cl, seed=42)
    gr = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause, n_relations=11, seed=42)

    # Pipeline seuil 0 : émet toutes les arêtes (argmax conf ≥ 0)
    pipe_low = CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab,
                           edge_threshold=0.0)
    # Pipeline seuil haut : émet seulement les arêtes très confiantes
    pipe_high = CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab,
                            edge_threshold=0.99)

    reps = [_make_rep("alpha"), _make_rep("beta")]
    cir_low  = pipe_low.forward(reps, "alpha cause beta")
    cir_high = pipe_high.forward(reps, "alpha cause beta")

    edges_low  = len(cir_low.get("edges", []))  if cir_low  else 0
    edges_high = len(cir_high.get("edges", [])) if cir_high else 0

    assert edges_low >= edges_high, (
        f"seuil=0.0 ({edges_low} arêtes) devrait émettre ≥ seuil=0.99 ({edges_high} arêtes)"
    )
    # Avec un seuil 0.99 et des logits aléatoires, softmax max ≈ 0.27 (11 classes) → 0 arêtes
    assert edges_high == 0, (
        f"Seuil=0.99 : attendu 0 arêtes (conf softmax max ≈ 1/11), obtenu {edges_high}"
    )
