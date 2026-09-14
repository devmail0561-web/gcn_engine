from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .yaml_reader import load_all_sentences
from .schema import SentenceRecord
from ..constants import NODE_TYPES, RELATION_TYPES


@dataclass
class TrainingSample:
    sentence: SentenceRecord
    gold_node_labels: np.ndarray  # (N,) int — indices dans NODE_TYPES
    gold_edge_labels: np.ndarray  # (E,) int — indices dans RELATION_TYPES


class GCNDataLoader:
    """Itère sur les sentences YAML d'un répertoire et produit des TrainingSample."""

    def __init__(self, data_dir: Path, lang: str = "fr", repeat: bool = False):
        self.data_dir = data_dir
        self.lang = lang
        self.repeat = repeat
        self._records = load_all_sentences(data_dir, lang)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        while True:
            for rec in self._records:
                yield self._to_sample(rec)
            if not self.repeat:
                break

    def _to_sample(self, rec: SentenceRecord) -> TrainingSample:
        node_labels = np.array(
            [NODE_TYPES.index(c.node_type) if c.node_type in NODE_TYPES else 0
             for c in rec.clauses],
            dtype=np.int64,
        )
        edge_labels = np.array(
            [RELATION_TYPES.index(e.relation) if e.relation in RELATION_TYPES else 0
             for e in rec.edges],
            dtype=np.int64,
        )
        return TrainingSample(rec, node_labels, edge_labels)
