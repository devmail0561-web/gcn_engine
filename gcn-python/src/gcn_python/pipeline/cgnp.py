from __future__ import annotations
from pathlib import Path
import numpy as np

from ..layer1.extractor import extract
from ..layer1.features import FeatureVocabulary, vectorize_clause, vectorize_edge
from ..layer2.interface import CausalEncoder
from ..layer3.interface import CausalGraph
from ..taxonomy.loader import TaxonomyIndex
from ..constants import NODE_TYPES, RELATION_TYPES
from .label_builder import build_label
from .ir_emitter import emit


class CGNPipeline:
    """
    Pipeline complet des couches 1-3 de l'architecture CGNP.

    Le data scientist instancie ce pipeline avec ses implémentations de
    CausalEncoder (Couche 2) et CausalGraph (Couche 3).

    - forward(text)         → CausalIR dict (prêt pour serde_json Rust)
    - loss(pred, gold)      → float
    - backward(loss)        → point d'entrée rétropropagation (no-op de référence)
    """

    def __init__(
        self,
        encoder: CausalEncoder,
        graph: CausalGraph,
        taxonomy_dir: Path,
        lang: str,
        vocabulary: FeatureVocabulary,
    ):
        self.encoder = encoder
        self.graph = graph
        self.taxonomy_dir = taxonomy_dir
        self.lang = lang
        self.vocabulary = vocabulary
        self._tax = TaxonomyIndex.load(taxonomy_dir, lang)

    def forward(self, text: str) -> dict:
        """text → CausalIR dict (JSON-serializable, conforme schéma serde Rust)"""
        reps = extract(text, self.lang)
        if not reps:
            return emit(text, self.lang, [], [], [], [], [])

        # Couche 1 — vectorisation
        clause_vecs = np.stack([
            vectorize_clause(r, self.vocabulary, self._tax) for r in reps
        ])  # (N, D_clause)

        # Couche 2 — prédiction des types de nœuds
        node_logits = np.stack([self.encoder.forward_node(v) for v in clause_vecs])
        node_type_idxs = np.argmax(node_logits, axis=1)
        node_types = [NODE_TYPES[i] for i in node_type_idxs]

        # Prédiction des arêtes entre clauses adjacentes
        edge_triples: list[tuple[int, int, str, float, bool, int | None]] = []
        if len(reps) >= 2:
            for src_i in range(len(reps) - 1):
                dst_i = src_i + 1
                edge_vec = vectorize_edge(
                    reps[src_i], reps[dst_i], None,
                    src_i, dst_i, len(reps),
                    self.vocabulary, self._tax,
                )
                edge_logits = self.encoder.forward_edge(edge_vec)
                rel_idx = int(np.argmax(edge_logits))
                rel_conf = float(_softmax(edge_logits.reshape(1, -1))[0, rel_idx])
                edge_triples.append((src_i, dst_i, RELATION_TYPES[rel_idx], rel_conf, False, None))

        # Couche 3 — R-GCN message passing
        if len(reps) > 1 and edge_triples:
            edge_index = np.array(
                [[e[0] for e in edge_triples], [e[1] for e in edge_triples]], dtype=np.int64
            )
            edge_type_idxs = np.array(
                [RELATION_TYPES.index(e[2]) if e[2] in RELATION_TYPES else 0
                 for e in edge_triples],
                dtype=np.int64,
            )
            enriched = self.graph.message_pass(clause_vecs, edge_index, edge_type_idxs)
            # Re-predict if R-GCN output matches D_clause
            if enriched.shape[1] == self.vocabulary.d_clause:
                node_logits2 = np.stack([self.encoder.forward_node(v) for v in enriched])
                node_type_idxs = np.argmax(node_logits2, axis=1)
                node_types = [NODE_TYPES[i] for i in node_type_idxs]

        node_labels = [
            build_label(r, nt, self._tax, self.taxonomy_dir)
            for r, nt in zip(reps, node_types)
        ]
        token_spans = [r.token_span for r in reps]
        scopes = ["specific"] * len(reps)

        return emit(text, self.lang, node_types, node_labels, token_spans,
                    scopes, edge_triples)

    def loss(self, pred: dict, gold: dict) -> float:
        """
        Loss de référence (0/1 sur types de nœuds).
        Le DS remplace par sa propre loss avec son framework.
        """
        pred_nodes = pred.get("nodes", [])
        gold_nodes = gold.get("nodes", [])
        if not gold_nodes:
            return 0.0
        errors = sum(
            float(p.get("node_type") != g.get("node_type"))
            for p, g in zip(pred_nodes, gold_nodes)
        )
        return errors / len(gold_nodes)

    def backward(self, loss: float) -> None:
        """
        Point d'entrée rétropropagation — no-op de référence.
        Le DS appelle encoder.backward_node / backward_edge directement.
        """
        pass


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
