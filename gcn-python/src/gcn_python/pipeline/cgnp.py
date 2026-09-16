from __future__ import annotations
import numpy as np

from ..layer1.features import FeatureVocabulary, vectorize_clause, vectorize_edge
from ..layer2.interface import CausalEncoder
from ..layer3.interface import CausalGraph
from ..constants import NODE_TYPES, RELATION_TYPES
from .label_builder import build_label
from .ir_emitter import emit


class CGNPipeline:
    """
    Pipeline complet des couches 1-3 de l'architecture CGNP.

    Le data scientist instancie ce pipeline avec ses implémentations de
    CausalEncoder (Couche 2) et CausalGraph (Couche 3).

    - forward(reps, text, ...)                → CausalIR dict (prêt pour serde_json Rust)
    - loss(node_logits, edge_logits, ...)     → (float, d_node, d_edge)
    - backward(d_node, d_edge, lr)            → rétropropagation + mise à jour SGD

    Précondition à la construction : si graph expose d_out, il doit être égal à
    vocabulary.d_clause — vérifié immédiatement, ValueError sinon.

    decoder (optionnel) : TrainableDecoder ou tout objet implémentant
    forward_decode / loss_decode / backward_decode / update. Si None, le pipeline
    se comporte exactement comme avant (rétro-compatible).

    taxonomies_dir (optionnel) : Path vers le répertoire des taxonomies pour
    la nominalisation des labels de nœuds.
    """

    def __init__(
        self,
        encoder: CausalEncoder,
        graph: CausalGraph,
        lang: str,
        vocabulary: FeatureVocabulary,
        *,
        decoder=None,
        taxonomies_dir=None,
    ):
        if hasattr(graph, 'd_out') and graph.d_out != vocabulary.d_clause:
            raise ValueError(
                f"RGCNLayer.d_out={graph.d_out} ≠ vocabulary.d_clause="
                f"{vocabulary.d_clause} : instanciez RGCNLayer avec "
                f"d_out=vocabulary.d_clause pour alimenter le MLP nœud "
                f"depuis les représentations enrichies."
            )
        self.encoder = encoder
        self.graph = graph
        self.lang = lang
        self.vocabulary = vocabulary
        self.decoder = decoder
        self.taxonomies_dir = taxonomies_dir

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
        # Cache décodeur — rempli par loss() quand gold_surface est fourni
        self._cached_decode_gradient: np.ndarray | None = None

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
          Doit avoir exactement len(reps) éléments si fourni, ValueError sinon.
        n_total_clauses : nombre total de clauses dans la phrase (dénominateur de la distance).
          Comparé via `is not None` — la valeur 0 est traitée comme zéro clause, pas comme absent.
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
        if clause_positions is not None and len(clause_positions) != len(reps):
            raise ValueError(
                f"clause_positions a {len(clause_positions)} éléments pour {len(reps)} reps."
            )

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
        self._cached_decode_gradient = None
        _snap = hasattr(self.encoder, 'snapshot_node_cache')

        if not reps:
            return emit(text, self.lang, [], [], [], [], [])

        # Couche 1 — vectorisation
        clause_vecs = np.stack([
            vectorize_clause(r, self.vocabulary) for r in reps
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
            real_n = n_total_clauses if n_total_clauses is not None else len(reps)
            for src_i in range(len(reps) - 1):
                dst_i = src_i + 1
                real_src = clause_positions[src_i] if clause_positions else src_i
                real_dst = clause_positions[dst_i] if clause_positions else dst_i
                connector = (connector_reps[src_i]
                             if connector_reps and src_i < len(connector_reps) else None)
                edge_vec = vectorize_edge(
                    reps[src_i], reps[dst_i], connector,
                    real_src, real_dst, real_n,
                    self.vocabulary,
                )
                edge_vecs.append(edge_vec)
                edge_logit = self.encoder.forward_edge(edge_vec)
                all_edge_logits.append(edge_logit)
                if _snap:
                    edge_snapshots.append(self.encoder.snapshot_edge_cache())
                rel_idx = int(np.argmax(edge_logit))
                rel_conf = float(_softmax(edge_logit.reshape(1, -1))[0, rel_idx])
                marker_tok_id = connector.token_span[0] if connector is not None else None
                negated = _detect_negation(reps[src_i], reps[dst_i], connector)
                edge_triples.append((src_i, dst_i, RELATION_TYPES[rel_idx], rel_conf, negated, marker_tok_id))

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

        node_labels_attrs = [
            build_label(r, nt, self.taxonomies_dir)
            for r, nt in zip(reps, node_types)
        ]
        node_labels = [la[0] for la in node_labels_attrs]
        node_attributes = [la[1] for la in node_labels_attrs]

        token_spans = [r.token_span for r in reps]
        scopes = [_infer_scope(r) for r in reps]
        node_origins = [
            _infer_origin(
                nt,
                (connector_reps[i] if connector_reps and i < len(connector_reps) else None)
                or (connector_reps[i - 1] if connector_reps and i > 0 else None),
            )
            for i, nt in enumerate(node_types)
        ]

        return emit(text, self.lang, node_types, node_labels, token_spans,
                    scopes, edge_triples, node_origins=node_origins,
                    node_attributes=node_attributes)

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
        gold_surface: np.ndarray | None = None,  # (T,) int — tokens gold pour le décodeur
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Cross-entropie NumPy sur nœuds + arêtes + décodeur (optionnel).

        Retourne (total_loss, d_node_logits, d_edge_logits).
        Gradients normalisés par le nombre d'exemples.
        edge_loss_weight permet d'équilibrer la contribution des arêtes dans la loss totale.
        Si gold_surface est fourni et que le décodeur a produit des logits (forward()),
        la loss décodeur est ajoutée au total et son gradient est caché pour backward().
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

        total_loss = node_loss + edge_loss_weight * edge_loss

        # Decoder loss (optionnel — teacher forcing si gold_surface fourni)
        self._cached_decode_gradient = None
        if (self.decoder is not None
                and gold_surface is not None
                and len(gold_surface) > 0):
            _vecs = (self._cached_enriched_vecs
                     if self._cached_enriched_vecs is not None
                     else self._cached_clause_vecs)
            if _vecs is not None and len(_vecs) > 0:
                dec_logits = self.decoder.forward_decode(_vecs, gold_surface)
                dec_loss, d_dec = self.decoder.loss_decode(dec_logits, gold_surface)
                total_loss += dec_loss
                self._cached_decode_gradient = d_dec

        return total_loss, d_node, d_edge

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
        Les appels à update_node/update_edge sont gardés par hasattr — un encodeur
        tiers sans ces méthodes est silencieusement ignoré (ses poids ne sont pas
        mis à jour par ce backward).

        Contrainte d'interface : si l'encodeur n'implémente pas backward_node_dx,
        les poids R-GCN ne peuvent pas être mis à jour (d_enriched provient du
        backward de l'encodeur). Un UserWarning est émis dans ce cas.
        """
        if not hasattr(self.encoder, 'backward_node_dx'):
            import warnings
            warnings.warn(
                "CGNPipeline.backward() : encodeur sans backward_node_dx — "
                "les poids R-GCN ne sont pas mis à jour par ce backward. "
                "Implémenter backward_node_dx ou appeler graph.update() manuellement.",
                UserWarning,
                stacklevel=2,
            )
            return

        vecs = (self._cached_enriched_vecs
                if self._cached_enriched_vecs is not None
                else self._cached_clause_vecs)
        if vecs is None or len(vecs) == 0:
            return

        N = len(vecs)
        n = min(len(d_node_logits), N)
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
        d_enriched = np.zeros((N, vecs.shape[1]), dtype=np.float32)

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
        if all_node_grads is not None and hasattr(self.encoder, 'update_node'):
            self.encoder.update_node(all_node_grads, lr)
        if all_edge_grads is not None and hasattr(self.encoder, 'update_edge'):
            self.encoder.update_edge(all_edge_grads, lr)

        # --- Décodeur backward + update (sans couplage vers d_enriched — P3e) ---
        if (self.decoder is not None
                and self._cached_decode_gradient is not None
                and hasattr(self.decoder, 'backward_decode')):
            # P2d: backward_decode retourne (d_node_embs, dec_grads, d_attn_vec)
            d_node_embs, dec_grads, d_attn_vec = self.decoder.backward_decode(
                self._cached_decode_gradient
            )
            self.decoder.update(dec_grads, d_attn_vec, lr)
            # d_node_embs (N, D_in) ignoré (stop_gradient=True — voir P3e)

        # --- Rétropropagation R-GCN ---
        if (self._cached_edge_index is not None
                and self._cached_enriched_vecs is not None
                and hasattr(self.graph, 'backward_message_pass')):
            _, graph_grads = self.graph.backward_message_pass(d_enriched)
            self.graph.update(graph_grads, lr)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _cross_entropy(
    logits: np.ndarray,   # (N, C)
    labels: np.ndarray,   # (N,) int
) -> tuple[float, np.ndarray]:
    """Cross-entropie NumPy. Retourne (loss, d_logits) normalisés par N.

    Lève ValueError si labels contient des valeurs négatives (sentinelle -1 non filtrée)
    ou hors-bornes (>= n_classes).
    """
    if len(logits) == 0:
        return 0.0, np.zeros_like(logits)
    if len(labels) > 0 and int(labels.min()) < 0:
        raise ValueError(
            f"Label négatif dans _cross_entropy : min={labels.min()} "
            f"(sentinelle -1 non filtrée ?)"
        )
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


def _detect_negation(src_rep, dst_rep, connector_rep) -> bool:
    """Détecte la négation depuis is_negative des représentations UD.

    is_negative repose sur Polarity=Neg (morphologie UD). Les négations
    analytiques (ne...pas) dont 'pas' n'est pas le root ne sont pas
    détectées. Voir phase 10 pour la couverture complète.
    """
    if getattr(src_rep, 'is_negative', False):
        return True
    if getattr(dst_rep, 'is_negative', False):
        return True
    if connector_rep is not None and getattr(connector_rep, 'is_negative', False):
        return True
    return False


_SCOPE_HINTS: dict[str, str] = {
    "tous": "universal", "toutes": "universal", "chaque": "universal",
    "tout": "universal", "aucun": "null", "aucune": "null",
    "certains": "existential", "certaines": "existential",
    "quelques": "partial",
}


def _infer_scope(rep) -> str:
    """Dérive le scope depuis les déterminants/pronoms du span (français uniquement).

    Support multilingue à ajouter en phase 10.
    """
    for tok in rep.tokens:
        if tok.get("dep_rel") in {"det", "nsubj"} and tok.get("pos") in {"DET", "PRON"}:
            hint = _SCOPE_HINTS.get(tok["lemma"].lower())
            if hint:
                return hint
    return "specific"


def _infer_origin(node_type: str, connector_rep) -> str:
    """Un nœud condition sans connecteur explicite dans le texte est inféré."""
    if node_type == "condition" and connector_rep is None:
        return "inferred"
    return "explicit"
