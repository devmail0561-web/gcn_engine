# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
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
# Relations de dépendance UD typiques des connecteurs
CONNECTOR_DEP_RELS = [
    "mark", "case", "fixed", "cc", "advmod", "obl", "nmod",
    "conj", "ccomp", "xcomp", "_unk",
]


@dataclass
class FeatureVocabulary:
    """
    Schéma ordonné des features de la Couche 1 — language-agnostic.

    Features universelles UD (UPOS, dep_rel, morph) : applicables à toute langue.
    Lexique de connecteurs : vide par défaut — l'utilisateur fournit
    les lemmes propres à sa langue via connector_lemmas.

    Exemple (lemmes définis par l'utilisateur, langue au choix) :
        vocab = FeatureVocabulary(connector_lemmas=["parce", "car", "because", "since"])

    Sans connector_lemmas, d_conn est minimal (UPOS + dep_rel + position)
    et fonctionne pour toute langue sans modification.
    """
    # Features universelles UD
    upos_tags: list[str] = field(default_factory=lambda: list(UPOS_TAGS))
    dep_rels: list[str] = field(default_factory=lambda: list(UD_DEP_RELS))
    tense_values: list[str] = field(default_factory=lambda: list(UD_TENSE_VALUES))
    aspect_values: list[str] = field(default_factory=lambda: list(UD_ASPECT_VALUES))
    mood_values: list[str] = field(default_factory=lambda: list(UD_MOOD_VALUES))
    subject_pos_cats: list[str] = field(default_factory=lambda: list(SUBJECT_POS_CATS))
    # Lexique de connecteurs — vide par défaut (language-agnostic)
    connector_lemmas: list[str] = field(default_factory=list)
    connector_dep_rels: list[str] = field(default_factory=lambda: list(CONNECTOR_DEP_RELS))

    @property
    def d_clause(self) -> int:
        return (
            len(self.upos_tags)          # UPOS universel
            + len(self.dep_rels)         # dep_rel universel
            + len(self.subject_pos_cats) # POS du sujet
            + len(self.tense_values)     # morphologie
            + len(self.aspect_values)
            + len(self.mood_values)
            + 1   # Polarity
            + 3   # flags structurels : has_object, has_advcl, has_temporal_obl
        )

    @property
    def d_conn(self) -> int:
        return (
            len(self.upos_tags)              # UPOS du connecteur
            + len(self.connector_lemmas)     # lemme (0 si pas de lexique fourni)
            + len(self.connector_dep_rels)   # dep_rel du connecteur
            + 2                              # direction + distance
        )

    N_INTERACTION_FEATURES = 4  # shared_pos, shared_subject, clause_distance, obj_xor

    @property
    def d_edge(self) -> int:
        return 2 * self.d_clause + self.d_conn + self.N_INTERACTION_FEATURES

    def d_edge_closed_loop(self, d_effective: int, n_node_types: int, d_emb: int = 0) -> int:
        """Dimension du edge MLP en closed-loop."""
        return self.d_edge + 2 * d_emb + 2 * d_effective + 2 * n_node_types

    def to_json(self) -> str:
        return json.dumps({
            "upos_tags": self.upos_tags,
            "dep_rels": self.dep_rels,
            "tense_values": self.tense_values,
            "aspect_values": self.aspect_values,
            "mood_values": self.mood_values,
            "subject_pos_cats": self.subject_pos_cats,
            "connector_lemmas": self.connector_lemmas,
            "connector_dep_rels": self.connector_dep_rels,
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
    drop_morph: bool = False,
) -> np.ndarray:
    """UDRepresentation → np.ndarray[d_clause (+ d_emb si word_embedding fourni)]

    word_embedding : WordEmbedding optionnel (S1/S2). Si fourni, le vecteur
    d'embedding du root_lemma est concaténé à la fin des features structurelles.
    Quand None (défaut), comportement identique à l'original — rétrocompatible.

    drop_morph : si True, zérote les features Tense/Aspect/Mood/Polarity (→ _absent/0.0).
    Utilisé avec --drop-morph pour simuler le bridge heuristique à l'entraînement
    (parité train/inférence quand root_morph={} dans les UDRepresentation bridge).
    """
    if drop_morph:
        tense_vec  = _one_hot("_absent", vocab.tense_values)
        aspect_vec = _one_hot("_absent", vocab.aspect_values)
        mood_vec   = _one_hot("_absent", vocab.mood_values)
        polarity   = np.zeros(1, dtype=np.float32)
    else:
        tense_vec  = _one_hot(rep.tense,  vocab.tense_values)
        aspect_vec = _one_hot(rep.aspect, vocab.aspect_values)
        mood_vec   = _one_hot(rep.mood,   vocab.mood_values)
        polarity   = np.array([1.0 if rep.is_negative else 0.0], dtype=np.float32)
    parts = [
        _one_hot(rep.root_pos, vocab.upos_tags),
        _one_hot(rep.root_dep_rel, vocab.dep_rels),
        _one_hot(rep.subject_pos or "_absent", vocab.subject_pos_cats),
        tense_vec,
        aspect_vec,
        mood_vec,
        polarity,
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
        upos_vec  = _one_hot(marker_rep.root_pos, vocab.upos_tags)
        lemma_vec = _one_hot(marker_rep.root_lemma, vocab.connector_lemmas) \
                    if vocab.connector_lemmas else np.zeros(0, dtype=np.float32)
        dep_vec   = _one_hot(marker_rep.root_dep_rel, vocab.connector_dep_rels)
    else:
        upos_vec  = np.zeros(len(vocab.upos_tags), dtype=np.float32)
        lemma_vec = np.zeros(len(vocab.connector_lemmas), dtype=np.float32)
        dep_vec   = np.zeros(len(vocab.connector_dep_rels), dtype=np.float32)

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
    drop_morph: bool = False,
) -> np.ndarray:
    """Two clauses + connector + interaction features → np.ndarray[d_edge (+ 2*d_emb)]"""
    interaction = _interaction_features(src, dst, src_idx, dst_idx, n_clauses)
    return np.concatenate([
        vectorize_clause(src, vocab, word_embedding, drop_morph=drop_morph),
        vectorize_clause(dst, vocab, word_embedding, drop_morph=drop_morph),
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
