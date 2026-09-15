from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np

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

    - forward(text)                           → CausalIR dict (prêt pour serde_json Rust)
    - loss(node_logits, edge_logits, ...)     → (float, d_node, d_edge)
    - backward(d_node, d_edge, lr)            → rétropropagation + mise à jour SGD
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

        # Cache rempli par forward() — utilisé par loss() et backward()
        self._cached_clause_vecs: np.ndarray | None = None
        self._cached_enriched_vecs: np.ndarray | None = None
        self._cached_edge_vecs: np.ndarray | None = None
        self._cached_node_logits: np.ndarray | None = None
        self._cached_edge_logits: np.ndarray | None = None
        self._cached_edge_index: np.ndarray | None = None
        self._cached_edge_type_idxs: np.ndarray | None = None
        # Snapshots des activations MLP par nœud/arête — évitent de re-exécuter
        # forward_node au backward (pas de re-run, pas de fragilitié de cache)
        self._cached_node_snapshots: list | None = None
        self._cached_edge_snapshots: list | None = None

    def forward(
        self,
        reps: list,
        text: str = "",
        clause_positions: list[int] | None = None,
        n_total_clauses: int | None = None,
        connector_reps: list | None = None,
    ) -> dict:
        """UDRepresentation list → CausalIR dict (JSON-serializable, conforme schéma serde Rust).

        clause_positions : indices originaux des reps dans la phrase complète — utilisés
          pour calculer les features de position dans vectorize_edge.
        n_total_clauses : nombre total de clauses dans la phrase (dénominateur de la distance).
        connector_reps : UDRepresentation|None par paire consécutive (len = len(reps)-1).
        """
        return self._forward_from_reps(reps, text, clause_positions, n_total_clauses, connector_reps)

    def _forward_from_reps(
        self,
        reps: list,
        text: str,
        clause_positions: list[int] | None = None,
        n_total_clauses: int | None = None,
        connector_reps: list | None = None,
    ) -> dict:

        # Réinitialiser le cache
        self._cached_clause_vecs = None
        self._cached_enriched_vecs = None
        self._cached_edge_vecs = None
        self._cached_node_logits = None
        self._cached_edge_logits = None
        self._cached_edge_index = None
        self._cached_edge_type_idxs = None
        self._cached_node_snapshots = None
        self._cached_edge_snapshots = None
        _snap = hasattr(self.encoder, 'snapshot_node_cache')

        if not reps:
            return emit(text, self.lang, [], [], [], [], [])

        # Couche 1 — vectorisation
        clause_vecs = np.stack([
            vectorize_clause(r, self.vocabulary, self._tax) for r in reps
        ])  # (N, D_clause)
        self._cached_clause_vecs = clause_vecs

        # Couche 2 — prédiction des types de nœuds (snapshot par nœud pour backward)
        node_logits_list: list[np.ndarray] = []
        node_snapshots: list = []
        for v in clause_vecs:
            node_logits_list.append(self.encoder.forward_node(v))
            if _snap:
                node_snapshots.append(self.encoder.snapshot_node_cache())
        node_logits = np.stack(node_logits_list)
        node_type_idxs = np.argmax(node_logits, axis=1)
        node_types = [NODE_TYPES[i] for i in node_type_idxs]

        # Prédiction des arêtes entre clauses adjacentes
        edge_triples: list[tuple[int, int, str, float, bool, int | None]] = []
        edge_vecs: list[np.ndarray] = []
        all_edge_logits: list[np.ndarray] = []
        edge_snapshots: list = []
        if len(reps) >= 2:
            real_n = n_total_clauses if n_total_clauses else len(reps)
            for src_i in range(len(reps) - 1):
                dst_i = src_i + 1
                real_src = clause_positions[src_i] if clause_positions else src_i
                real_dst = clause_positions[dst_i] if clause_positions else dst_i
                connector = (connector_reps[src_i]
                             if connector_reps and src_i < len(connector_reps) else None)
                edge_vec = vectorize_edge(
                    reps[src_i], reps[dst_i], connector,
                    real_src, real_dst, real_n,
                    self.vocabulary, self._tax,
                )
                edge_vecs.append(edge_vec)
                edge_logit = self.encoder.forward_edge(edge_vec)
                all_edge_logits.append(edge_logit)
                if _snap:
                    edge_snapshots.append(self.encoder.snapshot_edge_cache())
                rel_idx = int(np.argmax(edge_logit))
                rel_conf = float(_softmax(edge_logit.reshape(1, -1))[0, rel_idx])
                edge_triples.append((src_i, dst_i, RELATION_TYPES[rel_idx], rel_conf, False, None))

        if edge_vecs:
            self._cached_edge_vecs = np.stack(edge_vecs)
            self._cached_edge_logits = np.stack(all_edge_logits)
            if _snap:
                self._cached_edge_snapshots = edge_snapshots

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
            self._cached_edge_index = edge_index
            self._cached_edge_type_idxs = edge_type_idxs
            enriched = self.graph.message_pass(clause_vecs, edge_index, edge_type_idxs)
            # Re-predict from enriched features ; snapshots overridés par ce pass
            if enriched.shape[1] != self.vocabulary.d_clause:
                warnings.warn(
                    f"RGCNLayer.d_out={enriched.shape[1]} ≠ vocabulary.d_clause="
                    f"{self.vocabulary.d_clause} : les logits ne seront pas recalculés "
                    f"après enrichissement R-GCN. Instanciez RGCNLayer avec d_out=d_clause.",
                    UserWarning, stacklevel=3,
                )
            if enriched.shape[1] == self.vocabulary.d_clause:
                node_logits2_list: list[np.ndarray] = []
                node_snapshots = []
                for v in enriched:
                    node_logits2_list.append(self.encoder.forward_node(v))
                    if _snap:
                        node_snapshots.append(self.encoder.snapshot_node_cache())
                node_logits2 = np.stack(node_logits2_list)
                node_type_idxs = np.argmax(node_logits2, axis=1)
                node_types = [NODE_TYPES[i] for i in node_type_idxs]
                node_logits = node_logits2
                self._cached_enriched_vecs = enriched

        self._cached_node_logits = node_logits
        if _snap:
            self._cached_node_snapshots = node_snapshots

        node_labels = [
            build_label(r, nt, self._tax, self.taxonomy_dir)
            for r, nt in zip(reps, node_types)
        ]
        token_spans = [r.token_span for r in reps]
        scopes = ["specific"] * len(reps)

        return emit(text, self.lang, node_types, node_labels, token_spans,
                    scopes, edge_triples)

    def filter_edge_cache(self, valid_idxs: np.ndarray) -> None:
        """Filtre les caches MLP d'arêtes aux seuls indices valides.

        Appelé après forward() pour aligner edge_logits ↔ gold_edge avant loss/backward.
        _cached_edge_index/_cached_edge_type_idxs ne sont PAS filtrés : le R-GCN
        a utilisé toutes les arêtes dans son forward et a besoin de toutes pour backward.
        """
        if self._cached_edge_vecs is not None:
            self._cached_edge_vecs = self._cached_edge_vecs[valid_idxs]
        if self._cached_edge_logits is not None:
            self._cached_edge_logits = self._cached_edge_logits[valid_idxs]
        if self._cached_edge_snapshots is not None:
            self._cached_edge_snapshots = [self._cached_edge_snapshots[i] for i in valid_idxs]

    def loss(
        self,
        node_logits: np.ndarray,    # (N, 7)  — logits nœuds du forward
        edge_logits: np.ndarray | None,  # (E, 11) — logits arêtes du forward, ou None
        gold_node: np.ndarray,      # (N,) int — indices dans NODE_TYPES
        gold_edge: np.ndarray | None = None,  # (E,) int — indices dans RELATION_TYPES
        edge_loss_weight: float = 1.0,  # pondération relative edge_loss / node_loss
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Cross-entropie NumPy sur nœuds + arêtes.

        Retourne (total_loss, d_node_logits, d_edge_logits).
        Gradients normalisés par le nombre d'exemples.
        edge_loss_weight permet d'équilibrer la contribution des arêtes dans la loss totale.
        """
        if len(node_logits) != len(gold_node):
            raise ValueError(
                f"Désalignement node_logits/gold_node : {len(node_logits)} logits vs {len(gold_node)} labels"
            )
        node_loss, d_node = _cross_entropy(node_logits, gold_node)

        if edge_logits is not None and gold_edge is not None and len(edge_logits) > 0:
            if len(edge_logits) != len(gold_edge):
                raise ValueError(
                    f"Désalignement edge_logits/gold_edge : {len(edge_logits)} logits vs {len(gold_edge)} labels"
                )
            edge_loss, d_edge = _cross_entropy(edge_logits, gold_edge)
        else:
            edge_loss = 0.0
            d_edge = np.zeros((0, len(RELATION_TYPES)), dtype=np.float32)

        return node_loss + edge_loss_weight * edge_loss, d_node, d_edge

    def backward(
        self,
        d_node_logits: np.ndarray,  # (N, 7)
        d_edge_logits: np.ndarray,  # (E, 11)
        lr: float = 0.01,
    ) -> None:
        """
        Rétropropagation + SGD sur l'implémentation de référence NumPy.

        Opère sur MLPEncoder (backward_node_dx / backward_edge) et
        RGCNLayer (backward_message_pass). Le DS PyTorch override cette méthode.
        """
        if not hasattr(self.encoder, 'backward_node_dx'):
            return  # implémentation non-référence, le DS gère son propre backward

        vecs = (self._cached_enriched_vecs
                if self._cached_enriched_vecs is not None
                else self._cached_clause_vecs)
        if vecs is None or len(vecs) == 0:
            return

        n = min(len(d_node_logits), len(vecs))
        _has_node_snap = (
            hasattr(self.encoder, 'restore_node_cache')
            and self._cached_node_snapshots is not None
            and len(self._cached_node_snapshots) >= n
        )
        _has_edge_snap = (
            hasattr(self.encoder, 'restore_edge_cache')
            and self._cached_edge_snapshots is not None
        )

        # --- Rétropropagation nœuds ---
        all_node_grads: list[tuple[np.ndarray, np.ndarray]] | None = None
        d_enriched = np.zeros((n, vecs.shape[1]), dtype=np.float32)

        for i in range(n):
            if _has_node_snap:
                self.encoder.restore_node_cache(self._cached_node_snapshots[i])
            else:
                self.encoder.forward_node(vecs[i])  # fallback sans snapshot
            grads_i, dx_i = self.encoder.backward_node_dx(d_node_logits[i])
            d_enriched[i] = dx_i
            if all_node_grads is None:
                all_node_grads = [(dW.copy(), db.copy()) for dW, db in grads_i]
            else:
                for j, (dW_i, db_i) in enumerate(grads_i):
                    all_node_grads[j] = (
                        all_node_grads[j][0] + dW_i,
                        all_node_grads[j][1] + db_i,
                    )

        # _cross_entropy normalise déjà par N — pas de renormalisation ici

        # --- Rétropropagation arêtes ---
        all_edge_grads: list[tuple[np.ndarray, np.ndarray]] | None = None
        if (d_edge_logits is not None and len(d_edge_logits) > 0
                and self._cached_edge_vecs is not None):
            e = min(len(d_edge_logits), len(self._cached_edge_vecs))
            for i in range(e):
                if _has_edge_snap and i < len(self._cached_edge_snapshots):
                    self.encoder.restore_edge_cache(self._cached_edge_snapshots[i])
                else:
                    self.encoder.forward_edge(self._cached_edge_vecs[i])
                grads_i = self.encoder.backward_edge(d_edge_logits[i])
                if all_edge_grads is None:
                    all_edge_grads = [(dW.copy(), db.copy()) for dW, db in grads_i]
                else:
                    for j, (dW_i, db_i) in enumerate(grads_i):
                        all_edge_grads[j] = (
                            all_edge_grads[j][0] + dW_i,
                            all_edge_grads[j][1] + db_i,
                        )
            # _cross_entropy normalise déjà par E — pas de renormalisation ici

        # --- Mise à jour encodeur ---
        if all_node_grads is not None:
            self.encoder.update_node(all_node_grads, lr)
        if all_edge_grads is not None:
            self.encoder.update_edge(all_edge_grads, lr)

        # --- Rétropropagation R-GCN ---
        if (self._cached_edge_index is not None
                and self._cached_enriched_vecs is not None
                and hasattr(self.graph, 'backward_message_pass')):
            N_full = len(self._cached_enriched_vecs)
            if d_enriched.shape[0] < N_full:
                pad = np.zeros((N_full - d_enriched.shape[0], d_enriched.shape[1]), dtype=np.float32)
                d_enriched_full = np.concatenate([d_enriched, pad], axis=0)
            else:
                d_enriched_full = d_enriched
            _, graph_grads = self.graph.backward_message_pass(d_enriched_full)
            self.graph.update(graph_grads, lr)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _cross_entropy(
    logits: np.ndarray,   # (N, C)
    labels: np.ndarray,   # (N,) int
) -> tuple[float, np.ndarray]:
    """Cross-entropie NumPy. Retourne (loss, d_logits) normalisés par N."""
    if len(logits) == 0:
        return 0.0, np.zeros_like(logits)
    if len(labels) > 0 and int(labels.max()) >= logits.shape[1]:
        raise ValueError(
            f"Label hors-bornes dans _cross_entropy : "
            f"max={labels.max()} >= n_classes={logits.shape[1]}"
        )
    N = len(logits)
    probs = _softmax(logits)                               # (N, C)
    loss = float(-np.log(probs[np.arange(N), labels] + 1e-9).mean())
    d_logits = probs.copy()
    d_logits[np.arange(N), labels] -= 1.0
    d_logits /= N
    return loss, d_logits
