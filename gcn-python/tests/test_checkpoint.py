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


def test_arch_json_stores_inference_hyperparams(tmp_path: Path):
    """edge_threshold, drop_morph et temperature persistés dans _arch_json."""
    import json, numpy as np
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import save_checkpoint

    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipe = CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab,
                       edge_threshold=0.35, drop_morph=True, temperature=2.0)
    ckpt = tmp_path / "model.npz"
    save_checkpoint(pipe, ckpt)

    raw = np.load(ckpt, allow_pickle=True)
    arch = json.loads(str(raw["_arch_json"][0]))
    assert arch["edge_threshold"] == pytest.approx(0.35)
    assert arch["drop_morph"] is True
    assert arch["temperature"] == pytest.approx(2.0)


def test_from_pretrained_restores_inference_hyperparams(tmp_path: Path):
    """from_pretrained restaure edge_threshold, drop_morph, temperature dans le pipeline.

    Sans ce correctif, analyze() utilisait les défauts (seuil=0.0, morph actif,
    temp=1.0) même si le modèle avait été entraîné avec d'autres valeurs.
    """
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import save_checkpoint
    from gcn_python.engine import GCNEngine

    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipe = CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab,
                       edge_threshold=0.42, drop_morph=True, temperature=1.5)
    ckpt = tmp_path / "model.npz"
    save_checkpoint(pipe, ckpt)

    engine = GCNEngine.from_pretrained(ckpt, trusted=True)
    assert engine._pipeline.edge_threshold == pytest.approx(0.42), (
        "edge_threshold non restauré depuis _arch_json dans from_pretrained"
    )
    assert engine._pipeline.drop_morph is True, (
        "drop_morph non restauré depuis _arch_json dans from_pretrained"
    )
    assert engine._pipeline.temperature == pytest.approx(1.5), (
        "temperature non restaurée depuis _arch_json dans from_pretrained"
    )


def test_check_path_safe_refuses_symlink(tmp_path: Path):
    """_check_path_safe refuse un symlink leaf."""
    import os
    from gcn_python.training.checkpoint import _check_path_safe

    real = tmp_path / "real.npz"
    real.write_bytes(b"")
    link = tmp_path / "link.npz"
    os.symlink(real, link)

    with pytest.raises(RuntimeError, match="symlink"):
        _check_path_safe(link, allow_symlink=False)
    # allow_symlink=True ne lève pas
    _check_path_safe(link, allow_symlink=True)


def test_check_path_safe_refuses_hardlink(tmp_path: Path):
    """_check_path_safe refuse un hardlink (nlink > 1)."""
    import os
    from gcn_python.training.checkpoint import _check_path_safe

    real = tmp_path / "real.npz"
    real.write_bytes(b"")
    hard = tmp_path / "hard.npz"
    os.link(real, hard)  # crée un hardlink

    with pytest.raises(RuntimeError, match="liens durs"):
        _check_path_safe(hard, allow_symlink=False)


def test_check_path_safe_refuses_dir_symlink(tmp_path: Path):
    """_check_path_safe refuse un chemin dont un répertoire parent est un symlink."""
    import os
    from gcn_python.training.checkpoint import _check_path_safe

    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    link_dir = tmp_path / "link_dir"
    os.symlink(real_dir, link_dir)
    target = link_dir / "model.npz"

    with pytest.raises(RuntimeError, match="symlink"):
        _check_path_safe(target, allow_symlink=False)


def test_load_checkpoint_validates_n_rgcn_layers(tmp_path: Path):
    """load_checkpoint lève ValueError si n_rgcn_layers du checkpoint ≠ pipeline.

    Sans ce correctif, un pipeline 2 couches chargerait silencieusement un
    checkpoint 1 couche — la couche extra reste aléatoire sans avertissement.
    """
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import save_checkpoint, load_checkpoint

    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause,
                     d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipe1 = CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab, n_rgcn_layers=1)
    ckpt = tmp_path / "model_1layer.npz"
    save_checkpoint(pipe1, ckpt)

    enc2 = MLPEncoder(d_clause=vocab.d_clause,
                      d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr2 = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipe2 = CGNPipeline(encoder=enc2, graph=gr2, vocabulary=FeatureVocabulary(),
                        n_rgcn_layers=2)
    with pytest.raises(ValueError, match="n_rgcn_layers"):
        load_checkpoint(pipe2, ckpt, trusted=True)


def test_load_checkpoint_validates_n_rgcn_layers_reverse(tmp_path: Path):
    """load_checkpoint lève ValueError si le checkpoint a plus de couches que le pipeline.

    Direction 2→1 : checkpoint 2 couches chargé dans pipeline 1 couche.
    Sans ce check, la couche extra du checkpoint serait silencieusement ignorée
    (clé graph_extra_1_* whitelistée dans _VALID_PREFIXES mais jamais restaurée).
    """
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import save_checkpoint, load_checkpoint

    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause,
                     d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipe2 = CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab, n_rgcn_layers=2)
    ckpt = tmp_path / "model_2layers.npz"
    save_checkpoint(pipe2, ckpt)

    enc1 = MLPEncoder(d_clause=vocab.d_clause,
                      d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr1 = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipe1 = CGNPipeline(encoder=enc1, graph=gr1, vocabulary=FeatureVocabulary(),
                        n_rgcn_layers=1)
    with pytest.raises(ValueError, match="n_rgcn_layers"):
        load_checkpoint(pipe1, ckpt, trusted=True)


def test_load_checkpoint_n_rgcn_layers_matching_no_error(tmp_path: Path):
    """load_checkpoint ne lève pas d'erreur quand n_rgcn_layers correspond (cas passant 2→2).

    Vérifie que la validation est non-bloquante sur les cas corrects et que
    les poids de la couche extra sont effectivement restaurés.
    """
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import save_checkpoint, load_checkpoint

    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause,
                     d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipe_src = CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab, n_rgcn_layers=2)
    ckpt = tmp_path / "model_2to2.npz"
    save_checkpoint(pipe_src, ckpt)
    W_extra_before = pipe_src._graph_layers[1].W_0.copy()

    enc2 = MLPEncoder(d_clause=vocab.d_clause,
                      d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr2 = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipe_dst = CGNPipeline(encoder=enc2, graph=gr2, vocabulary=FeatureVocabulary(),
                           n_rgcn_layers=2)
    W_extra_r_before = pipe_src._graph_layers[1].W_r.copy()
    load_checkpoint(pipe_dst, ckpt, trusted=True)
    W_extra_after = pipe_dst._graph_layers[1].W_0.copy()
    W_extra_r_after = pipe_dst._graph_layers[1].W_r.copy()
    assert np.allclose(W_extra_before, W_extra_after), (
        "W_0 couche extra (graph_extra_1) non restauré après load_checkpoint 2→2."
    )
    assert np.allclose(W_extra_r_before, W_extra_r_after), (
        "W_r couche extra (graph_extra_1) non restauré après load_checkpoint 2→2."
    )


def test_load_checkpoint_n_rgcn_layers_absent_warns_for_multilayer(tmp_path: Path):
    """load_checkpoint warn si n_rgcn_layers absent en arch et pipeline > 1 couche.

    Vieux checkpoint pré-2.1.0 sans n_rgcn_layers : si chargé dans un pipeline
    2 couches, la validation est désactivée → couche extra reste aléatoire.
    Le warn signale ce danger au lieu de le laisser silencieux.
    """
    import json as _json, warnings as _w
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import save_checkpoint, load_checkpoint

    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause,
                     d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipe_old = CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab, n_rgcn_layers=1)
    ckpt = tmp_path / "old_checkpoint.npz"
    save_checkpoint(pipe_old, ckpt)

    # Simuler un vieux checkpoint en retirant n_rgcn_layers de l'arch
    raw = np.load(ckpt, allow_pickle=True)
    arch = _json.loads(str(raw["_arch_json"][0]))
    del arch["n_rgcn_layers"]
    arrays = dict(raw)
    arrays["_arch_json"] = np.array([_json.dumps(arch)], dtype=object)
    np.savez_compressed(ckpt, **arrays)

    enc2 = MLPEncoder(d_clause=vocab.d_clause,
                      d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr2 = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipe2 = CGNPipeline(encoder=enc2, graph=gr2, vocabulary=FeatureVocabulary(),
                        n_rgcn_layers=2)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        load_checkpoint(pipe2, ckpt, trusted=True)
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("n_rgcn_layers" in m for m in messages), (
        f"Aucun UserWarning sur n_rgcn_layers absent — silence silencieux. "
        f"Warnings : {messages}"
    )


def test_load_checkpoint_warns_when_arch_json_absent(tmp_path: Path):
    """load_checkpoint émet un UserWarning si _arch_json est absent du checkpoint.

    Un checkpoint sans _arch_json désactive la validation de shapes — l'appelant
    doit en être informé, pas obtenir un silence silencieux.
    """
    import json, warnings as _w
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import load_checkpoint

    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause,
                     d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipe = CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab)

    # Fabriquer un .npz sans _arch_json ni _vocab_json
    ckpt = tmp_path / "no_arch.npz"
    import numpy as np
    params = list(pipe.encoder.parameters())
    arrays = {f"encoder_{i}": p for i, p in enumerate(params)}
    for li, layer in enumerate(pipe._graph_layers):
        prefix = "graph" if li == 0 else f"graph_extra_{li}"
        for j, p in enumerate(layer.parameters()):
            arrays[f"{prefix}_{j}"] = p
    np.savez_compressed(ckpt, **arrays)

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        load_checkpoint(pipe, ckpt, trusted=True)

    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("sans _arch_json" in m for m in messages), (
        f"Aucun UserWarning 'sans _arch_json' — silence silencieux. "
        f"Warnings reçus : {messages}"
    )


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
