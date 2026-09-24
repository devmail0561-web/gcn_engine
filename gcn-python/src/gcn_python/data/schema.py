# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenRecord:
    id: int           # 1-based
    form: str
    lemma: str
    pos: str          # UPOS
    dep_rel: str      # UD dep relation
    dep_head: int     # 0 = root
    morph: dict[str, str] = field(default_factory=dict)
    gcn_causal_type: Optional[str] = None
    gcn_causal_class: Optional[str] = None


@dataclass
class ClauseRecord:
    node_id: str           # "n001"
    node_type: str         # NodeType snake_case
    label: str
    token_span: tuple[int, int]
    scope: str
    temporal_index: int
    origin: str
    attributes: dict[str, object] = field(default_factory=dict)
    modifiers: list[dict] = field(default_factory=list)


@dataclass
class EdgeRecord:
    source: str = ""           # legacy mono-source (from_legacy / shim)
    target: str = ""           # "n002"
    relation: str = ""         # RelationType snake_case
    confidence: Optional[float] = None   # None = annotation absente (≠ 0.0)
    explicit: Optional[bool] = None      # None = non renseigné ; True = connecteur présent
    negated: Optional[bool] = None       # None = non renseigné (détection pipeline)
    marker_token: Optional[int] = None
    sources: Optional[list[str]] = None  # champ canonique v2 (remplace source)

    def __post_init__(self):
        # Shim rétrocompat : sources est canonique, source reste lisible.
        if self.sources is None:
            self.sources = [self.source] if self.source else []
        if not self.source and self.sources:
            self.source = self.sources[0]

    @classmethod
    def from_legacy(cls, source: str, target: str = "", relation: str = "",
                    confidence: Optional[float] = None,
                    explicit: Optional[bool] = None,
                    negated: Optional[bool] = None,
                    marker_token: Optional[int] = None) -> "EdgeRecord":
        return cls(source=source, target=target, relation=relation,
                   confidence=confidence, explicit=explicit, negated=negated,
                   marker_token=marker_token, sources=[source] if source else [])


@dataclass
class SentenceRecord:
    id: str
    text: str
    tokens: list[TokenRecord]
    clauses: list[ClauseRecord]
    edges: list[EdgeRecord]
    lang: str = ""  # conservé comme métadonnée, non utilisé en calcul
    causal_pattern: str = ""  # métadonnée pour split stratifié uniquement, pas propagé aux features
    weight: float = 1.0  # Amélioration F : pondération gold=1.0 / silver=silver_weight
