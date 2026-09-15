from __future__ import annotations
from pathlib import Path
import numpy as np

from ..pipeline.cgnp import CGNPipeline
from ..layer1.features import FeatureVocabulary


def save_checkpoint(pipeline: CGNPipeline, path: Path) -> None:
    """Sérialise tous les poids du pipeline dans un fichier .npz."""
    arrays: dict[str, np.ndarray] = {}

    encoder_params = pipeline.encoder.parameters()
    for i, p in enumerate(encoder_params):
        arrays[f"encoder_{i}"] = p

    graph_params = pipeline.graph.parameters()
    for i, p in enumerate(graph_params):
        arrays[f"graph_{i}"] = p

    vocab_json = pipeline.vocabulary.to_json()
    arrays["_vocab_json"] = np.array([vocab_json], dtype=object)

    np.savez(path, **arrays)


def load_checkpoint(pipeline: CGNPipeline, path: Path) -> None:
    """Restaure les poids depuis un fichier .npz produit par save_checkpoint.

    Atomique : toutes les shapes sont validées avant toute mutation du pipeline.
    Un ValueError laisse le pipeline intact (vocabulary, encoder et graph inchangés).
    """
    data = np.load(path, allow_pickle=True)

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

    graph_params = pipeline.graph.parameters()
    for i, p in enumerate(graph_params):
        key = f"graph_{i}"
        if key in data and data[key].shape != p.shape:
            raise ValueError(
                f"Incompatibilité de dimension pour graph_{i} : "
                f"checkpoint={data[key].shape} ≠ pipeline={p.shape}."
            )

    # Toutes les formes validées — mutation sûre
    if new_vocab is not None:
        pipeline.vocabulary = new_vocab

    for i, p in enumerate(encoder_params):
        key = f"encoder_{i}"
        if key in data:
            p[:] = data[key]

    for i, p in enumerate(graph_params):
        key = f"graph_{i}"
        if key in data:
            p[:] = data[key]
