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
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge)
    gr = RGCNLayerPT(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipeline = CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab)

    # Sauvegarder poids originaux
    ckpt = tmp_path / "model.npz"
    save_checkpoint(pipeline, ckpt)
    W_r_before = gr.W_r.detach().cpu().numpy().copy()
    W_0_before = gr.W_0.detach().cpu().numpy().copy()

    # Créer nouveau pipeline et charger
    enc2 = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge)
    gr2 = RGCNLayerPT(d_in=vocab.d_clause, d_out=vocab.d_clause)
    p2 = CGNPipeline(encoder=enc2, graph=gr2, vocabulary=FeatureVocabulary())
    load_checkpoint(p2, ckpt)

    # Vérifier que les poids PyTorch ont bien été restaurés
    W_r_after = gr2.W_r.detach().cpu().numpy()
    W_0_after = gr2.W_0.detach().cpu().numpy()
    assert np.allclose(W_r_before, W_r_after), "W_r restauré correctement"
    assert np.allclose(W_0_before, W_0_after), "W_0 restauré correctement"
