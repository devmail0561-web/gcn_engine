from __future__ import annotations
from dataclasses import dataclass, field
import json
import numpy as np

from ..constants import (
    UPOS_TAGS, UD_DEP_RELS, UD_TENSE_VALUES, UD_ASPECT_VALUES,
    UD_MOOD_VALUES, SUBJECT_POS_CATS,
)
from .representation import UDRepresentation

# Lemmes de connecteurs causaux les plus fréquents en français
_CONNECTOR_LEMMAS_RAW = [
    "parce", "car", "puisque", "comme", "si", "bien", "quoique",
    "quoique", "malgré", "pour", "afin", "donc", "alors", "ensuite",
    "puis", "mais", "or", "pourtant", "cependant", "néanmoins",
    "cependant", "pourvu", "seulement", "lorsque", "dès", "tant",
    "si", "non", "jamais", "ni", "plutôt", "au lieu",
    "en revanche", "en raison", "grâce", "sous", "condition",
    "contrairement", "selon", "à cause", "devant", "chez",
    "vers", "après", "avant", "depuis", "pendant", "durant",
    "chez", "entre", "parmi", "hors", "outre", "faute",
]
# Déduplication programmatique — préserve l'ordre, élimine les doublons
CONNECTOR_LEMMAS = list(dict.fromkeys(_CONNECTOR_LEMMAS_RAW))

# Relations de dépendance UD typiques des connecteurs
CONNECTOR_DEP_RELS = [
    "mark", "case", "fixed", "cc", "advmod", "obl", "nmod",
    "conj", "ccomp", "xcomp", "_unk",
]


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
        return (
            len(self.upos_tags)       # 19 : UPOS du connecteur
            + len(CONNECTOR_LEMMAS)   # 54 : lemma du connecteur
            + len(CONNECTOR_DEP_RELS) # 11 : dep_rel du connecteur
            + 2                       #  2 : direction + distance
        )

    N_INTERACTION_FEATURES = 4  # shared_pos, shared_subject, clause_distance, obj_xor

    @property
    def d_edge(self) -> int:
        return 2 * self.d_clause + self.d_conn + self.N_INTERACTION_FEATURES

    def d_edge_closed_loop(self, d_effective: int, n_node_types: int, d_emb: int = 0) -> int:
        """Dimension du edge MLP en closed-loop (avec enriched vectors + node probs).

        d_effective : dimension effective des clause vectors (d_clause + d_emb).
        d_emb : dimension des word embeddings (0 si pas d'embeddings).
        """
        return self.d_edge + 2 * d_emb + 2 * d_effective + 2 * n_node_types

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
    word_embedding=None,
) -> np.ndarray:
    """UDRepresentation → np.ndarray[d_clause (+ d_emb si word_embedding fourni)]

    word_embedding : WordEmbedding optionnel (S1/S2). Si fourni, le vecteur
    d'embedding du root_lemma est concaténé à la fin des features structurelles.
    Quand None (défaut), comportement identique à l'original — rétrocompatible.
    """
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
    if word_embedding is not None:
        parts.append(word_embedding.lookup(rep.root_lemma))
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
        lemma_vec = _one_hot(marker_rep.root_lemma, CONNECTOR_LEMMAS)
        dep_vec = _one_hot(marker_rep.root_dep_rel, CONNECTOR_DEP_RELS)
    else:
        upos_vec = np.zeros(len(vocab.upos_tags), dtype=np.float32)
        lemma_vec = np.zeros(len(CONNECTOR_LEMMAS), dtype=np.float32)
        dep_vec = np.zeros(len(CONNECTOR_DEP_RELS), dtype=np.float32)

    pos_vec = np.array(
        [float(src_idx < dst_idx), abs(dst_idx - src_idx) / max(n_clauses, 1)],
        dtype=np.float32,
    )
    return np.concatenate([upos_vec, lemma_vec, dep_vec, pos_vec])


def vectorize_edge(
    src: UDRepresentation,
    dst: UDRepresentation,
    connector: UDRepresentation | None,
    src_idx: int,
    dst_idx: int,
    n_clauses: int,
    vocab: FeatureVocabulary,
    word_embedding=None,
) -> np.ndarray:
    """Two clauses + connector + interaction features → np.ndarray[d_edge (+ 2*d_emb)]"""
    interaction = _interaction_features(src, dst, src_idx, dst_idx, n_clauses)
    return np.concatenate([
        vectorize_clause(src, vocab, word_embedding),
        vectorize_clause(dst, vocab, word_embedding),
        vectorize_connector(connector, src_idx, dst_idx, n_clauses, vocab),
        interaction,
    ])


def _interaction_features(
    src: UDRepresentation,
    dst: UDRepresentation,
    src_idx: int,
    dst_idx: int,
    n_clauses: int,
) -> np.ndarray:
    """Features d'interaction entre deux clauses (4 dims)."""
    shared_pos = float(src.root_pos == dst.root_pos)
    shared_subject = float(
        src.subject_pos is not None
        and dst.subject_pos is not None
        and src.subject_pos == dst.subject_pos
    )
    clause_dist = abs(dst_idx - src_idx) / max(n_clauses, 1)
    obj_xor = float(src.has_object != dst.has_object)
    return np.array([shared_pos, shared_subject, clause_dist, obj_xor],
                    dtype=np.float32)
