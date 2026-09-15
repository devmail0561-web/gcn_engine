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
    """Restaure les poids depuis un fichier .npz produit par save_checkpoint."""
    data = np.load(path, allow_pickle=True)

    # 1. Vocabulaire d'abord — les dimensions des poids en dépendent
    if "_vocab_json" in data:
        pipeline.vocabulary = FeatureVocabulary.from_json(str(data["_vocab_json"][0]))

    # 2. Validation des formes + chargement poids encodeur
    encoder_params = pipeline.encoder.parameters()
    for i, p in enumerate(encoder_params):
        key = f"encoder_{i}"
        if key in data:
            if data[key].shape != p.shape:
                raise ValueError(
                    f"Incompatibilité de dimension pour encoder_{i} : "
                    f"checkpoint={data[key].shape} ≠ pipeline={p.shape}. "
                    f"Reconstruisez le pipeline avec la même taxonomie que le checkpoint."
                )
            p[:] = data[key]

    # 3. Validation des formes + chargement poids graph
    graph_params = pipeline.graph.parameters()
    for i, p in enumerate(graph_params):
        key = f"graph_{i}"
        if key in data:
            if data[key].shape != p.shape:
                raise ValueError(
                    f"Incompatibilité de dimension pour graph_{i} : "
                    f"checkpoint={data[key].shape} ≠ pipeline={p.shape}."
                )
            p[:] = data[key]
