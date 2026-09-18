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
    arch = {
        "d_eff":        d_eff,
        "d_emb":        d_emb,
        "n_relations":  n_rel,
        "bidirectional": pipeline.bidirectional,
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

    # S9 : sauvegarder word_embedding si présent
    if getattr(pipeline, 'word_embedding', None) is not None:
        we = pipeline.word_embedding
        arrays["_word_emb_vocab_json"] = np.array([we.to_json()], dtype=object)
        arrays["word_emb_E"] = we._E

    np.savez(path, **arrays)


def load_checkpoint(pipeline: CGNPipeline, path: Path, *, trusted: bool = False) -> None:
    """Restaure les poids depuis un fichier .npz produit par save_checkpoint.

    Atomique : toutes les shapes sont validées avant toute mutation du pipeline.
    Un ValueError laisse le pipeline intact (vocabulary, encoder et graph inchangés).

    Sécurité : le .npz contient des tableaux `object` (JSON d'architecture/vocab)
    qui exigent `allow_pickle=True`, soit une exécution de pickle au chargement.
    `trusted=False` par défaut refuse le chargement ; passez `trusted=True`
    uniquement pour un checkpoint local de confiance (produit par votre
    `gcn-train`). Les CLI (`gcn-train --encoder-checkpoint`, `gcn-eval`,
    `gcn-discuss`, `gcn-index`) le passent explicitement — l'invocation vaut
    opt-in.
    """
    if not trusted:
        raise RuntimeError(
            f"Refus de charger {Path(path).name} : checkpoint .npz non fiable par défaut "
            "(allow_pickle requis → exécution de pickle). Relancez avec trusted=True "
            "pour un fichier local de confiance."
        )
    data = np.load(path, allow_pickle=True)

    # Valider que seules les clés attendues sont présentes (détection de corruption)
    _VALID_PREFIXES = ("encoder_", "graph_", "graph_extra_", "decoder_")
    _VALID_EXACT = {"_vocab_json", "_decoder_meta_json", "_word_emb_vocab_json", "word_emb_E", "_arch_json"}
    unexpected = set(data.files) - _VALID_EXACT
    unexpected = {k for k in unexpected if not any(k.startswith(p) for p in _VALID_PREFIXES)}
    if unexpected:
        raise ValueError(
            f"Checkpoint {path.name} contient des clés inattendues : {sorted(unexpected)}. "
            f"Fichier potentiellement corrompu ou incompatible."
        )

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
        for i, p in enumerate(decoder_params):
            key = f"decoder_{i}"
            if key in data:
                if data[key].shape != p.shape:
                    raise ValueError(
                        f"Incompatibilité de dimension pour decoder_{i} : "
                        f"checkpoint={data[key].shape} ≠ decoder={p.shape}."
                    )
                p[:] = data[key]
        pipeline.decoder = decoder
