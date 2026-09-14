from __future__ import annotations
from pathlib import Path
import numpy as np

from ..pipeline.cgnp import CGNPipeline


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

    encoder_params = pipeline.encoder.parameters()
    for i, p in enumerate(encoder_params):
        key = f"encoder_{i}"
        if key in data:
            p[:] = data[key]

    graph_params = pipeline.graph.parameters()
    for i, p in enumerate(graph_params):
        key = f"graph_{i}"
        if key in data:
            p[:] = data[key]
