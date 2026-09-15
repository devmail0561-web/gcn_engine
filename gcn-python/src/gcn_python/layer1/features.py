from __future__ import annotations
from dataclasses import dataclass, field
import json
import numpy as np

from ..constants import (
    UPOS_TAGS, UD_DEP_RELS, UD_TENSE_VALUES, UD_ASPECT_VALUES,
    UD_MOOD_VALUES, SUBJECT_POS_CATS,
)
from .representation import UDRepresentation


@dataclass
class FeatureVocabulary:
    """
    Schéma ordonné des features de la Couche 1.
    Sérialisable JSON — partagé entre entraînement et inférence.
    Le data scientist sérialise cette instance avec son checkpoint.
    Features purement syntaxiques : UPOS, DEP_REL, tense/aspect/mood, polarity, flags structurels.
    """
    upos_tags: list[str] = field(default_factory=lambda: list(UPOS_TAGS))
    dep_rels: list[str] = field(default_factory=lambda: list(UD_DEP_RELS))
    tense_values: list[str] = field(default_factory=lambda: list(UD_TENSE_VALUES))
    aspect_values: list[str] = field(default_factory=lambda: list(UD_ASPECT_VALUES))
    mood_values: list[str] = field(default_factory=lambda: list(UD_MOOD_VALUES))
    subject_pos_cats: list[str] = field(default_factory=lambda: list(SUBJECT_POS_CATS))

    @property
    def d_clause(self) -> int:
        return (
            len(self.upos_tags)
            + len(self.dep_rels)
            + len(self.subject_pos_cats)
            + len(self.tense_values)
            + len(self.aspect_values)
            + len(self.mood_values)
            + 1   # Polarity
            + 3   # structural flags
        )

    @property
    def d_conn(self) -> int:
        return len(self.upos_tags) + 2

    @property
    def d_edge(self) -> int:
        return 2 * self.d_clause + self.d_conn

    def to_json(self) -> str:
        return json.dumps({
            "upos_tags": self.upos_tags,
            "dep_rels": self.dep_rels,
            "tense_values": self.tense_values,
            "aspect_values": self.aspect_values,
            "mood_values": self.mood_values,
            "subject_pos_cats": self.subject_pos_cats,
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "FeatureVocabulary":
        return cls(**json.loads(s))


def _one_hot(value: str, vocab: list[str]) -> np.ndarray:
    v = np.zeros(len(vocab), dtype=np.float32)
    if value in vocab:
        v[vocab.index(value)] = 1.0
    elif "_unk" in vocab:
        v[vocab.index("_unk")] = 1.0
    return v


def vectorize_clause(
    rep: UDRepresentation,
    vocab: FeatureVocabulary,
) -> np.ndarray:
    """UDRepresentation → np.ndarray[d_clause]"""
    parts = [
        _one_hot(rep.root_pos, vocab.upos_tags),
        _one_hot(rep.root_dep_rel, vocab.dep_rels),
        _one_hot(rep.subject_pos or "_absent", vocab.subject_pos_cats),
        _one_hot(rep.tense, vocab.tense_values),
        _one_hot(rep.aspect, vocab.aspect_values),
        _one_hot(rep.mood, vocab.mood_values),
        np.array([1.0 if rep.is_negative else 0.0], dtype=np.float32),
        np.array([float(rep.has_object), float(rep.has_advcl), float(rep.has_temporal_obl)],
                 dtype=np.float32),
    ]
    return np.concatenate(parts)


def vectorize_connector(
    marker_rep: UDRepresentation | None,
    src_idx: int,
    dst_idx: int,
    n_clauses: int,
    vocab: FeatureVocabulary,
) -> np.ndarray:
    """Connector features between two clauses → np.ndarray[d_conn]"""
    if marker_rep is not None:
        upos_vec = _one_hot(marker_rep.root_pos, vocab.upos_tags)
    else:
        upos_vec = np.zeros(len(vocab.upos_tags), dtype=np.float32)

    pos_vec = np.array(
        [float(src_idx < dst_idx), abs(dst_idx - src_idx) / max(n_clauses, 1)],
        dtype=np.float32,
    )
    return np.concatenate([upos_vec, pos_vec])


def vectorize_edge(
    src: UDRepresentation,
    dst: UDRepresentation,
    connector: UDRepresentation | None,
    src_idx: int,
    dst_idx: int,
    n_clauses: int,
    vocab: FeatureVocabulary,
) -> np.ndarray:
    """Two clauses + connector → np.ndarray[d_edge]"""
    return np.concatenate([
        vectorize_clause(src, vocab),
        vectorize_clause(dst, vocab),
        vectorize_connector(connector, src_idx, dst_idx, n_clauses, vocab),
    ])
