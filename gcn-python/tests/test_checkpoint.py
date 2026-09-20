# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import pytest
import numpy as np


def test_checkpoint_pytorch_rgcn(tmp_path: Path):
    """H5 : load_checkpoint restaure correctement les poids de RGCNLayerPT."""
    pytest.importorskip("torch")  # skip si PyTorch absent

    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.pytorch_rgcn import RGCNLayerPT
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import save_checkpoint, load_checkpoint

    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr = RGCNLayerPT(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipeline = CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab)

    # Sauvegarder poids originaux
    ckpt = tmp_path / "model.npz"
    save_checkpoint(pipeline, ckpt)
    W_r_before = gr.W_r.detach().cpu().numpy().copy()
    W_0_before = gr.W_0.detach().cpu().numpy().copy()

    # Créer nouveau pipeline et charger
    enc2 = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr2 = RGCNLayerPT(d_in=vocab.d_clause, d_out=vocab.d_clause)
    p2 = CGNPipeline(encoder=enc2, graph=gr2, vocabulary=FeatureVocabulary())
    load_checkpoint(p2, ckpt, trusted=True)

    # Vérifier que les poids PyTorch ont bien été restaurés
    W_r_after = gr2.W_r.detach().cpu().numpy()
    W_0_after = gr2.W_0.detach().cpu().numpy()
    assert np.allclose(W_r_before, W_r_after), "W_r restauré correctement"
    assert np.allclose(W_0_before, W_0_after), "W_0 restauré correctement"


def test_checkpoint_untrusted_refused(tmp_path: Path):
    """Sécurité : load_checkpoint refuse par défaut (pickle non fiable)."""
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import save_checkpoint, load_checkpoint

    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipeline = CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab)
    ckpt = tmp_path / "model.npz"
    save_checkpoint(pipeline, ckpt)

    enc2 = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr2 = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    p2 = CGNPipeline(encoder=enc2, graph=gr2, vocabulary=FeatureVocabulary())
    with pytest.raises(RuntimeError, match="non fiable"):
        load_checkpoint(p2, ckpt)
    # Opt-in explicite : charge normalement.
    load_checkpoint(p2, ckpt, trusted=True)


def test_checkpoint_allpairs_roundtrip(tmp_path: Path):
    """Item 6 : all_pairs + bidirectional persistés et restaurés via from_pretrained."""
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import save_checkpoint
    from gcn_python.engine import GCNEngine

    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipeline = CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab,
                           all_pairs=True, bidirectional=True)
    ckpt = tmp_path / "model.npz"
    save_checkpoint(pipeline, ckpt)

    engine = GCNEngine.from_pretrained(ckpt, trusted=True)
    assert engine._pipeline.all_pairs is True
    assert engine._pipeline.bidirectional is True


def test_backward_slice_mismatch_raises():
    """Item 6 : backward refuse les gradients désalignés (pas de min() silencieux)."""
    import numpy as np
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer1.representation import UDRepresentation
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline

    def _rep(lemma: str) -> UDRepresentation:
        return UDRepresentation(
            tokens=[{"lemma": lemma, "pos": "VERB", "dep_rel": "root", "morph": {}}],
            root_lemma=lemma, root_pos="VERB", root_dep_rel="root",
            root_morph={"Tense": "Pres"}, subject_pos="NOUN", has_object=False,
            has_advcl=False, has_temporal_obl=False, token_span=(1, 2),
        )

    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipeline = CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab)
    pipeline.forward([_rep("baisser"), _rep("réduire")], "Les ventes baissent puis on réduit.")
    d_edge = np.zeros((0, 11), dtype=np.float32)
    with pytest.raises(ValueError, match="gradients nœuds"):
        pipeline.backward(np.zeros((5, 7), dtype=np.float32), d_edge)
    with pytest.raises(ValueError, match="gradients nœuds"):
        pipeline.backward_accumulate(np.zeros((5, 7), dtype=np.float32), d_edge)


def test_e2e_train_save_reload_inference(tmp_path: Path):
    """Prod gate : forward→loss→backward→save→from_pretrained→forward identique."""
    import numpy as np
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer1.representation import UDRepresentation
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import save_checkpoint
    from gcn_python.engine import GCNEngine

    def _rep(lemma: str, dep: str = "root") -> UDRepresentation:
        return UDRepresentation(
            tokens=[{"lemma": lemma, "pos": "VERB", "dep_rel": dep, "morph": {}}],
            root_lemma=lemma, root_pos="VERB", root_dep_rel=dep,
            root_morph={"Tense": "Pres"}, subject_pos="NOUN", has_object=False,
            has_advcl=False, has_temporal_obl=False, token_span=(1, 2),
        )

    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipe = CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab,
                       all_pairs=True, bidirectional=True)
    reps = [_rep("baisser"), _rep("augmenter", "advcl")]
    txt = "Les ventes baissent parce que les coûts augmentent."
    pipe.forward(reps, txt)
    nl, el = pipe._cached_node_logits, pipe._cached_edge_logits
    gn = np.zeros(len(nl), dtype=np.int64)
    ge = np.zeros(len(el), dtype=np.int64) if el is not None else None
    loss, d_n, d_e = pipe.loss(nl, el, gn, ge)
    assert np.isfinite(loss)
    pipe.backward(d_n, d_e, lr=0.01)
    pipe.forward(reps, txt)
    nl_ref = pipe._cached_node_logits.copy()

    ckpt = tmp_path / "e2e.npz"
    save_checkpoint(pipe, ckpt)
    eng = GCNEngine.from_pretrained(ckpt, trusted=True)
    eng._pipeline.forward(reps, txt)
    assert np.allclose(nl_ref, eng._pipeline._cached_node_logits)
