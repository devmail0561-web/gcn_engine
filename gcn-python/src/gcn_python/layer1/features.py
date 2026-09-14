from __future__ import annotations
from dataclasses import dataclass, field
import json
import numpy as np

from ..constants import (
    UPOS_TAGS, UD_DEP_RELS, UD_TENSE_VALUES, UD_ASPECT_VALUES,
    UD_MOOD_VALUES, SUBJECT_POS_CATS,
)
from ..taxonomy.loader import TaxonomyIndex
from .representation import UDRepresentation


@dataclass
class FeatureVocabulary:
    """
    Schéma ordonné des features de la Couche 1.
    Sérialisable JSON — partagé entre entraînement et inférence.
    Le data scientist sérialise cette instance avec son checkpoint.
    """
    upos_tags: list[str] = field(default_factory=lambda: list(UPOS_TAGS))
    dep_rels: list[str] = field(default_factory=lambda: list(UD_DEP_RELS))
    tense_values: list[str] = field(default_factory=lambda: list(UD_TENSE_VALUES))
    aspect_values: list[str] = field(default_factory=lambda: list(UD_ASPECT_VALUES))
    mood_values: list[str] = field(default_factory=lambda: list(UD_MOOD_VALUES))
    subject_pos_cats: list[str] = field(default_factory=lambda: list(SUBJECT_POS_CATS))
    taxonomy_keys: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, tax_index: TaxonomyIndex) -> "FeatureVocabulary":
        return cls(taxonomy_keys=tax_index.keys())

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
            + len(self.taxonomy_keys)
        )

    @property
    def d_conn(self) -> int:
        return len(self.upos_tags) + len(self.taxonomy_keys) + 2

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
            "taxonomy_keys": self.taxonomy_keys,
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
    tax: TaxonomyIndex,
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

    # Taxonomy membership — check all lemmas in the clause
    all_lemmas = {t["lemma"] for t in rep.tokens}
    all_lemmas.add(rep.root_lemma)
    tax_vec = np.zeros(len(vocab.taxonomy_keys), dtype=np.float32)
    for i, key in enumerate(vocab.taxonomy_keys):
        lemmas_in_class = tax.data.get(key, frozenset())
        if any(lemma in lemmas_in_class for lemma in all_lemmas):
            tax_vec[i] = 1.0
    parts.append(tax_vec)

    return np.concatenate(parts)


def vectorize_connector(
    marker_rep: UDRepresentation | None,
    src_idx: int,
    dst_idx: int,
    n_clauses: int,
    vocab: FeatureVocabulary,
    tax: TaxonomyIndex,
) -> np.ndarray:
    """Connector features between two clauses → np.ndarray[d_conn]"""
    if marker_rep is not None:
        upos_vec = _one_hot(marker_rep.root_pos, vocab.upos_tags)
        all_lemmas = {t["lemma"] for t in marker_rep.tokens}
        tax_vec = np.zeros(len(vocab.taxonomy_keys), dtype=np.float32)
        for i, key in enumerate(vocab.taxonomy_keys):
            if any(lemma in tax.data.get(key, frozenset()) for lemma in all_lemmas):
                tax_vec[i] = 1.0
    else:
        upos_vec = np.zeros(len(vocab.upos_tags), dtype=np.float32)
        tax_vec = np.zeros(len(vocab.taxonomy_keys), dtype=np.float32)

    pos_vec = np.array(
        [float(src_idx < dst_idx), abs(dst_idx - src_idx) / max(n_clauses, 1)],
        dtype=np.float32,
    )
    return np.concatenate([upos_vec, tax_vec, pos_vec])


def vectorize_edge(
    src: UDRepresentation,
    dst: UDRepresentation,
    connector: UDRepresentation | None,
    src_idx: int,
    dst_idx: int,
    n_clauses: int,
    vocab: FeatureVocabulary,
    tax: TaxonomyIndex,
) -> np.ndarray:
    """Two clauses + connector → np.ndarray[d_edge]"""
    return np.concatenate([
        vectorize_clause(src, vocab, tax),
        vectorize_clause(dst, vocab, tax),
        vectorize_connector(connector, src_idx, dst_idx, n_clauses, vocab, tax),
    ])
