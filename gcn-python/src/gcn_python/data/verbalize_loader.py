# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..verbalizer.trainable import SurfaceVocabulary
from ..constants import NODE_TYPES


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


@dataclass
class VerbalizeSample:
    ir_json: str                      # CausalIR JSON (for inference)
    node_type_embeddings: np.ndarray  # (N, 7) one-hot — conservé pour rétrocompat
    gold_tokens: np.ndarray           # (T,) int indices in SurfaceVocabulary
    source_text: str                  # used to match encoding dataset samples
    node_labels: list[str] = None     # NOUVEAU — labels depuis causal_ir.nodes[].label


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
    ) -> None:
        raw = self._load_raw(data_dir)

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
            for surf in ex.get("surfaces", []):
                if surf.get("quality") not in ("gold", "silver"):
                    continue
                gold_tokens = np.array(vocab.encode(surf["text"]), dtype=np.int64)
                self._samples.append(
                    VerbalizeSample(ir_json, node_embs, gold_tokens, source_text, node_labels)
                )

    @staticmethod
    def _load_raw(data_dir: Path) -> list[dict]:
        examples: list[dict] = []
        loaded: set[str] = set()
        for p in sorted(data_dir.glob("verbalize_*.json")):
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
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
