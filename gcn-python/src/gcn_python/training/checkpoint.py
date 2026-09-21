# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from ..pipeline.cgnp import CGNPipeline
from ..layer1.features import FeatureVocabulary


def save_checkpoint(pipeline: CGNPipeline, path: Path) -> None:
    """Sérialise tous les poids du pipeline dans un fichier .npz."""
    arrays: dict[str, np.ndarray] = {}

    # Métadonnées d'architecture — lues par GCNEngine.from_pretrained()
    # pour reconstruire le pipeline sans hardcoder les indices de shapes.
    we = getattr(pipeline, 'word_embedding', None)
    d_emb = we.d_emb if we is not None else 0
    d_eff = pipeline.vocabulary.d_clause + d_emb
    graph0 = pipeline._graph_layers[0]
    n_rel = getattr(graph0, 'n_relations', len(pipeline.relation_types))
    d_hidden = getattr(graph0, 'd_out', d_eff)
    try:
        vocab_size = len(pipeline.vocabulary)
    except TypeError:
        vocab_size = getattr(pipeline.vocabulary, 'vocab_size', 0)
    arch = {
        "d_eff":        d_eff,
        "d_emb":        d_emb,
        "d_hidden":     int(d_hidden),
        "vocab_size":   int(vocab_size),
        "n_relations":  n_rel,
        "bidirectional": pipeline.bidirectional,
        "bidi_flag":    bool(pipeline.bidirectional),
        "all_pairs": pipeline.all_pairs,
        "n_rgcn_layers": pipeline.n_rgcn_layers,
        "graph_class":  type(graph0).__name__,
    }
    arrays["_arch_json"] = np.array([json.dumps(arch)], dtype=object)

    encoder_params = pipeline.encoder.parameters()
    for i, p in enumerate(encoder_params):
        arrays[f"encoder_{i}"] = p

    # S5 : sauvegarder toutes les couches R-GCN (_graph_layers)
    for layer_i, layer in enumerate(getattr(pipeline, '_graph_layers', [pipeline.graph])):
        prefix = "graph" if layer_i == 0 else f"graph_extra_{layer_i}"
        for j, p in enumerate(layer.parameters()):
            arrays[f"{prefix}_{j}"] = p

    vocab_json = pipeline.vocabulary.to_json()
    arrays["_vocab_json"] = np.array([vocab_json], dtype=object)

    if pipeline.decoder is not None and hasattr(pipeline.decoder, 'parameters'):
        decoder_params = pipeline.decoder.parameters()
        for i, p in enumerate(decoder_params):
            arrays[f"decoder_{i}"] = p
        if hasattr(pipeline.decoder, 'to_json'):
            arrays["_decoder_meta_json"] = np.array([pipeline.decoder.to_json()], dtype=object)

    # LinkPredHead : clés optionnelles (warn si absentes au chargement, pas ValueError)
    link_pred = getattr(pipeline, 'link_predictor', None)
    if link_pred is not None and hasattr(link_pred, 'parameters'):
        for i, p in enumerate(link_pred.parameters()):
            arrays[f"link_pred_{i}"] = np.asarray(p)
        if hasattr(link_pred, 'to_json'):
            arrays["_link_pred_meta_json"] = np.array([link_pred.to_json()], dtype=object)

    # S9 : sauvegarder word_embedding si présent
    if getattr(pipeline, 'word_embedding', None) is not None:
        we = pipeline.word_embedding
        arrays["_word_emb_vocab_json"] = np.array([we.to_json()], dtype=object)
        arrays["word_emb_E"] = we._E

    # Sauvegarde atomique : tmp + replace POSIX
    path = Path(path)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def load_checkpoint(
    pipeline: CGNPipeline, path: Path,
    *, trusted: bool = False, allow_symlink: bool = False,
) -> None:
    """Restaure les poids depuis un fichier .npz produit par save_checkpoint.

    Atomique : toutes les shapes sont validées avant toute mutation du pipeline.
    Un ValueError laisse le pipeline intact (vocabulary, encoder et graph inchangés).

    Sécurité : le .npz contient des tableaux `object` (JSON d'architecture/vocab)
    qui exigent `allow_pickle=True`, soit une exécution de pickle au chargement.
    `trusted=False` par défaut refuse le chargement ; passez `trusted=True`
    uniquement pour un checkpoint local de confiance (produit par votre
    `gcn-train`). Les CLI (`gcn-train --encoder-checkpoint`, `gcn-eval`,
    `gcn-discuss`, `gcn-index`) le passent explicitement — l'invocation vaut opt-in.

    Symlinks : refusés par défaut (R4 — vecteur de substitution silencieuse).
    Passez `allow_symlink=True` si vous êtes sûr de la cible.
    """
    if not trusted:
        raise RuntimeError(
            f"Refus de charger {Path(path).name} : checkpoint .npz non fiable par défaut "
            "(allow_pickle requis → exécution de pickle). Relancez avec trusted=True "
            "pour un fichier local de confiance."
        )
    if Path(path).is_symlink():
        if not allow_symlink:
            raise RuntimeError(
                f"Refus de charger {Path(path).name} : le chemin est un symlink. "
                "Un symlink peut pointer vers un checkpoint malveillant (vecteur de substitution). "
                "Passez allow_symlink=True si vous avez vérifié la cible."
            )
        import warnings as _w
        _w.warn(
            f"Checkpoint {Path(path).name} est un symlink — allow_symlink=True "
            "passé explicitement.",
            UserWarning,
            stacklevel=2,
        )
    data = np.load(path, allow_pickle=True)

    # Clés attendues : inconnues -> warn+ignore (forward-compat v2.5 dans code v2.0)
    _VALID_PREFIXES = ("encoder_", "graph_", "graph_extra_", "decoder_",
                       "link_pred_", "hyperedge_")
    _VALID_EXACT = {"_vocab_json", "_decoder_meta_json", "_word_emb_vocab_json", "word_emb_E",
                    "_arch_json", "_link_pred_meta_json"}
    unexpected = set(data.files) - _VALID_EXACT
    unexpected = {k for k in unexpected if not any(k.startswith(p) for p in _VALID_PREFIXES)}
    if unexpected:
        import warnings as _w2
        _w2.warn(
            f"Checkpoint {Path(path).name} contient des clés inconnues ignorées : "
            f"{sorted(unexpected)}.",
            UserWarning,
            stacklevel=2,
        )

    # Validation sémantique du triplet (d_eff, d_hidden, |V|, n_relations, bidi)
    if "_arch_json" in data:
        import json as _json
        try:
            _arch = _json.loads(str(data["_arch_json"][0]))
        except Exception:
            _arch = {}
        if isinstance(_arch, dict):
            _we = getattr(pipeline, 'word_embedding', None)
            _d_emb = _we.d_emb if _we is not None else 0
            _d_eff = pipeline.vocabulary.d_clause + _d_emb
            _g0 = pipeline._graph_layers[0] if getattr(pipeline, '_graph_layers', None) else pipeline.graph
            _d_hid = getattr(_g0, 'd_out', _d_eff)
            _n_rel = getattr(_g0, 'n_relations', len(pipeline.relation_types))
            for _k, _exp, _found in (
                ("d_eff", _d_eff, _arch.get("d_eff")),
                ("d_hidden", int(_d_hid), _arch.get("d_hidden")),
                ("n_relations", _n_rel, _arch.get("n_relations")),
                ("bidirectional", bool(pipeline.bidirectional),
                 _arch.get("bidirectional", _arch.get("bidi_flag"))),
                ("all_pairs", bool(pipeline.all_pairs), _arch.get("all_pairs")),
            ):
                if _found is not None and _found != _exp:
                    raise ValueError(
                        f"Incompatibilité de dimension pour {_k} : "
                        f"checkpoint={_found!r} ≠ pipeline={_exp!r}."
                    )

    # Double-absent policy : decoder
    import logging as _log3
    _ckpt_log = _log3.getLogger(__name__)
    _has_ckpt_decoder = any(k.startswith("decoder_") for k in data.files)
    if pipeline.decoder is None and not _has_ckpt_decoder:
        _ckpt_log.debug("Checkpoint sans decoder_* et pipeline sans decoder — usage normal.")
    elif pipeline.decoder is not None and not _has_ckpt_decoder:
        import warnings as _w3
        _w3.warn("Pipeline avec decoder mais checkpoint sans decoder_* "
                 "— poids non restaurés (init aléatoire).", UserWarning, stacklevel=2)

    # Double-absent policy : link_predictor (optionnel, jamais bloquant)
    _has_ckpt_lp = any(k.startswith("link_pred_") for k in data.files)
    _has_pipe_lp = getattr(pipeline, 'link_predictor', None) is not None
    if _has_pipe_lp and not _has_ckpt_lp:
        import warnings as _w3lp
        _w3lp.warn("Pipeline avec link_predictor mais checkpoint sans link_pred_* "
                   "— tête non restaurée (init aléatoire).", UserWarning, stacklevel=2)
    elif not _has_pipe_lp and not _has_ckpt_lp:
        _ckpt_log.debug("Checkpoint sans link_pred_* et pipeline sans link_predictor — usage normal.")

    new_vocab = None
    if "_vocab_json" in data:
        new_vocab = FeatureVocabulary.from_json(str(data["_vocab_json"][0]))

    # Validation de toutes les formes avant toute mutation
    encoder_params = pipeline.encoder.parameters()
    for i, p in enumerate(encoder_params):
        key = f"encoder_{i}"
        if key in data and data[key].shape != p.shape:
            raise ValueError(
                f"Incompatibilité de dimension pour encoder_{i} : "
                f"checkpoint={data[key].shape} ≠ pipeline={p.shape}. "
                f"Reconstruisez le pipeline avec la même taxonomie que le checkpoint."
            )

    # S5 : valider toutes les couches R-GCN
    for layer_i, layer in enumerate(getattr(pipeline, '_graph_layers', [pipeline.graph])):
        prefix = "graph" if layer_i == 0 else f"graph_extra_{layer_i}"
        for j, p in enumerate(layer.parameters()):
            key = f"{prefix}_{j}"
            if key in data and data[key].shape != p.shape:
                raise ValueError(
                    f"Incompatibilité de dimension pour {key} : "
                    f"checkpoint={data[key].shape} ≠ pipeline={p.shape}."
                )

    # Toutes les formes validées — mutation sûre
    if new_vocab is not None:
        pipeline.vocabulary = new_vocab

    for i, p in enumerate(encoder_params):
        key = f"encoder_{i}"
        if key in data:
            p[:] = data[key]

    # S5 : restaurer toutes les couches R-GCN
    for layer_i, layer in enumerate(getattr(pipeline, '_graph_layers', [pipeline.graph])):
        prefix = "graph" if layer_i == 0 else f"graph_extra_{layer_i}"
        layer_params = layer.parameters()
        # H5 correction : utiliser load_state() pour PyTorch RGCNLayerPT
        if hasattr(layer, 'load_state'):
            ckpt_arrays = [data[f"{prefix}_{j}"] for j in range(len(layer_params))
                           if f"{prefix}_{j}" in data]
            if ckpt_arrays:
                layer.load_state(ckpt_arrays)
        else:
            for j, p in enumerate(layer_params):
                key = f"{prefix}_{j}"
                if key in data:
                    p[:] = data[key]

    # S9 : restaurer word_embedding si présent dans le checkpoint
    if "_word_emb_vocab_json" in data:
        from ..layer1.embedding import WordEmbedding
        d_emb = int(data["word_emb_E"].shape[1]) if "word_emb_E" in data else 50
        we = WordEmbedding.from_json(str(data["_word_emb_vocab_json"][0]), d_emb=d_emb)
        if "word_emb_E" in data:
            we._E = data["word_emb_E"].astype(np.float32)
        pipeline.word_embedding = we

    if "_decoder_meta_json" in data:
        from ..verbalizer.trainable import TrainableDecoder
        decoder = TrainableDecoder.from_json(str(data["_decoder_meta_json"][0]))
        decoder_params = decoder.parameters()
        # B7 : pré-validation complète de TOUTES les shapes avant toute mutation
        # (identique au pattern encodeur/RGCN lignes 183-202).
        for i, p in enumerate(decoder_params):
            key = f"decoder_{i}"
            if key in data and data[key].shape != p.shape:
                raise ValueError(
                    f"Incompatibilité de dimension pour decoder_{i} : "
                    f"checkpoint={data[key].shape} ≠ decoder={p.shape}."
                )
        # Toutes les shapes validées — mutation sûre
        for i, p in enumerate(decoder_params):
            key = f"decoder_{i}"
            if key in data:
                p[:] = data[key]
        pipeline.decoder = decoder

    # Restaurer LinkPredHead si présent dans le checkpoint
    if "_link_pred_meta_json" in data:
        from ..layer3.link_pred import LinkPredHead
        head = LinkPredHead.from_json(str(data["_link_pred_meta_json"][0]))
        # B7 parity : pré-validation de TOUTES les shapes avant toute mutation
        for i, p in enumerate(head.parameters()):
            key = f"link_pred_{i}"
            if key in data and data[key].shape != p.shape:
                raise ValueError(
                    f"Incompatibilité de dimension pour link_pred_{i} : "
                    f"checkpoint={data[key].shape} ≠ head={p.shape}."
                )
        # Toutes les shapes validées — mutation sûre
        for i, p in enumerate(head.parameters()):
            key = f"link_pred_{i}"
            if key in data:
                if p.ndim == 0:
                    p[()] = data[key].item() if hasattr(data[key], "item") else data[key]
                else:
                    p[:] = data[key]
        pipeline.link_predictor = head
