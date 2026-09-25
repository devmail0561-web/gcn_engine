# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants import NODE_TYPES, RELATION_TYPES
from ..verbalizer.trainable import SurfaceVocabulary
from .json_reader import load_all_sentences


def _node_type_embeddings(nodes: list[dict]) -> np.ndarray:
    """Build one-hot node_type embeddings from a list of CausalIR node dicts."""
    if not nodes:
        return np.zeros((1, len(NODE_TYPES)), dtype=np.float32)
    embs = []
    for node in nodes:
        nt = node.get("node_type", "")
        if nt not in NODE_TYPES:
            warnings.warn(
                f"Type de nœud inconnu '{nt}' — mappé à index 0 ('{NODE_TYPES[0]}').",
                UserWarning, stacklevel=3,
            )
        idx = NODE_TYPES.index(nt) if nt in NODE_TYPES else 0
        onehot = np.zeros(len(NODE_TYPES), dtype=np.float32)
        onehot[idx] = 1.0
        embs.append(onehot)
    return np.stack(embs)  # (N, 7)


def _extract_clause_texts(nodes: list[dict], sentence) -> list[str] | None:
    """G1 : reconstruit le texte de chaque clause depuis sentence.tokens + token_span.

    Le nœud verbalize d'indice i (id int) correspond à la clause i du dataset
    source (même ordre). Retourne None si la phrase source est introuvable
    ou sans tokens.
    """
    if sentence is None or not getattr(sentence, 'tokens', None):
        return None
    if not getattr(sentence, 'clauses', None):
        return None
    texts: list[str] = []
    for i, _node in enumerate(nodes):
        if i >= len(sentence.clauses):
            return None
        span_start, span_end = sentence.clauses[i].token_span
        clause_tokens = [t for t in sentence.tokens if span_start <= t.id <= span_end]
        if not clause_tokens:
            return None
        texts.append(" ".join(t.form for t in clause_tokens))
    return texts


def _parse_edge_triples(edges: list) -> list[tuple[int, int, int]]:
    """G2 : normalise les arêtes verbalize en [(src, dst, rel_idx)].

    Formats supportés : [src, dst, {relation}] (listes) et
    {source/target ou sources, relation} (dicts). Relations inconnues ignorées.
    """
    triples: list[tuple[int, int, int]] = []
    for e in edges:
        try:
            if isinstance(e, (list, tuple)) and len(e) >= 3:
                src, dst = int(e[0]), int(e[1])
                rel = (e[2] or {}).get("relation", "") if isinstance(e[2], dict) else str(e[2])
            elif isinstance(e, dict):
                _srcs = e.get("sources") or ([e.get("source")] if e.get("source") is not None else [])
                if not _srcs:
                    continue
                src, dst = int(_srcs[0]), int(e.get("target"))
                _attrs = e.get("attributes") or {}
                rel = e.get("relation") or e.get("relation_type") or _attrs.get("relation") or ""
            else:
                continue
            if rel not in RELATION_TYPES:
                continue
            triples.append((src, dst, RELATION_TYPES.index(rel)))
        except (TypeError, ValueError, IndexError, KeyError):
            continue
    return triples


def _extract_connector_gold(
    edge_triples: list[tuple[int, int, int]],
    sentence,
    connector_vocab: list[str] | None,
) -> list[int | None] | None:
    """G2 : connecteur gold par arête = tokens entre les deux clauses.

    end_src = fin du span source, start_dst = début du span destination ;
    le texte inter-clauses est matché contre le vocabulaire.
    None par arête si aucun match ; None global si vocab/phrase absents.
    """
    if not edge_triples or connector_vocab is None:
        return None
    if sentence is None or not getattr(sentence, 'clauses', None):
        return None
    gold: list[int | None] = []
    for (src, dst, _rel) in edge_triples:
        if src >= len(sentence.clauses) or dst >= len(sentence.clauses):
            gold.append(None)
            continue
        end_src = sentence.clauses[src].token_span[1]
        start_dst = sentence.clauses[dst].token_span[0]
        connector_toks = [t for t in sentence.tokens if end_src < t.id < start_dst]
        connector_text = " ".join(t.form.lower() for t in connector_toks).strip()
        if not connector_text:
            gold.append(None)
            continue
        gold.append(next(
            (i for i, c in enumerate(connector_vocab)
             if c in connector_text or connector_text in c),
            None,
        ))
    return gold


@dataclass
class VerbalizeSample:
    ir_json: str                      # CausalIR JSON (for inference)
    node_type_embeddings: np.ndarray  # (N, 7) one-hot — conservé pour rétrocompat
    gold_tokens: np.ndarray           # (T,) int indices in SurfaceVocabulary
    source_text: str                  # used to match encoding dataset samples
    node_labels: list[str] | None = None  # labels depuis causal_ir.nodes[].label
    subgraph: dict | None = None                   # NOUVEAU v2.0 (rétrocompat)
    source_sentences: list[str] = None             # NOUVEAU v2.0 (rétrocompat)
    clause_texts: list[str] | None = None          # G1 : texte réel des clauses (spans dataset source)
    edge_triples: list[tuple[int, int, int]] | None = None  # G2 : (src, dst, rel_idx)
    connector_gold_idx: list[int | None] | None = None      # G2 : connecteur gold par arête (None si aucun match)

    def __post_init__(self):
        if self.source_sentences is None:
            self.source_sentences = []


class VerbalizerDataLoader:
    """Loads verbalize JSON files and produces VerbalizeSample per (CausalIR, surface) pair.

    Reads files matching verbalize_*.json in data_dir.
    Each cross-modal example with N surfaces produces N VerbalizeSamples.
    Only gold and silver quality surfaces are used for training.
    gcn-verbalize.schema.yaml is documentation for annotators — never read here.
    """

    def __init__(
        self,
        data_dir: Path,
        vocab: SurfaceVocabulary | None = None,
        source_json_dir: Path | None = None,
        connector_vocab: list[str] | None = None,
    ) -> None:
        """Charge les paires verbalize.

        source_json_dir (G1) : répertoire du dataset source (train.json) pour
        extraire les clause_texts depuis les tokens + token_span (via ex['id']).
        connector_vocab (G2) : vocabulaire de connecteurs pour extraire
        connector_gold_idx par arête (None si vocab absent ou aucun match).
        """
        raw = self._load_raw(data_dir)
        # G1 : map id phrase → SentenceRecord source
        source_map: dict[str, object] = {}
        if source_json_dir is not None:
            for _rec in load_all_sentences(Path(source_json_dir)):
                source_map[_rec.id] = _rec

        if vocab is None:
            vocab = SurfaceVocabulary()
            surfaces = [
                surf["text"]
                for ex in raw
                for surf in ex.get("surfaces", [])
                if surf.get("quality") in ("gold", "silver")
            ]
            vocab.build(surfaces)
        self.vocab = vocab

        self._samples: list[VerbalizeSample] = []
        for ex in raw:
            ir = ex["causal_ir"]
            ir_json = json.dumps(ir)
            source_text: str = ir.get("source_text", "")
            nodes: list[dict] = ir.get("nodes", [])
            node_embs = _node_type_embeddings(nodes)
            node_labels = [n.get("label", n.get("node_type", "")) for n in nodes]
            subgraph = ex.get("subgraph")
            source_sentences = list(ex.get("source_sentences", []) or [])
            # G1/G2 : extraction lexicale depuis le dataset source (si fourni)
            _sent = source_map.get(ex.get("id", ""))
            clause_texts = _extract_clause_texts(nodes, _sent)
            edge_triples = _parse_edge_triples(ir.get("edges", []))
            connector_gold_idx = _extract_connector_gold(
                edge_triples, _sent, connector_vocab)
            # Trier surfaces par qualité (gold < silver) puis ordre insertion
            _rank = {"gold": 0, "silver": 1}
            surfaces = sorted(
                [s for s in ex.get("surfaces", []) if s.get("quality") in ("gold", "silver")],
                key=lambda s: _rank.get(s.get("quality"), 9),
            )
            for surf in surfaces:
                gold_tokens = np.array(vocab.encode(surf["text"]), dtype=np.int64)
                self._samples.append(
                    VerbalizeSample(ir_json, node_embs, gold_tokens, source_text,
                                    node_labels, subgraph, source_sentences,
                                    clause_texts, edge_triples, connector_gold_idx)
                )

    @staticmethod
    def _load_raw(data_dir: Path) -> list[dict]:
        examples: list[dict] = []
        loaded: set[str] = set()
        for p in sorted(data_dir.glob("verbalize_*.json")):
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
            if "schema_version" not in data:
                warnings.warn(
                    f"VerbalizerDataLoader: {p.name} sans 'schema_version' "
                    "(attendu '2.0' — format legacy accepté).",
                    UserWarning,
                    stacklevel=3,
                )
            examples.extend(data.get("examples", []))
            loaded.add(p.name)
        ignored = sorted(p.name for p in data_dir.glob("*.json") if p.name not in loaded)
        if ignored:
            warnings.warn(
                f"VerbalizerDataLoader: {len(ignored)} fichier(s) JSON ignorés "
                f"(ne commencent pas par 'verbalize_') : {ignored}",
                UserWarning,
                stacklevel=3,
            )
        return examples

    def source_text_map(self) -> dict[str, list[np.ndarray]]:
        """Returns {source_text: [gold_tokens, ...]} for joint training lookup."""
        result: dict[str, list[np.ndarray]] = {}
        for s in self._samples:
            result.setdefault(s.source_text, []).append(s.gold_tokens)
        return result

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        yield from self._samples
