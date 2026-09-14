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
    source: str            # "n001"
    target: str            # "n002"
    relation: str          # RelationType snake_case
    confidence: float
    explicit: bool
    negated: bool
    marker_token: Optional[int]


@dataclass
class SentenceRecord:
    id: str
    text: str
    lang: str
    tokens: list[TokenRecord]
    clauses: list[ClauseRecord]
    edges: list[EdgeRecord]
