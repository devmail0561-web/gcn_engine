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
        vocabulary: FeatureVocabulary,
        *,
        decoder=None,
        taxonomies_dir=None,
        temperature: float = 1.0,
        node_types: list[str] | None = None,
        relation_types: list[str] | None = None,
        n_rgcn_layers: int = 1,
        all_pairs: bool = False,
        word_embedding=None,
        bidirectional: bool = False,
    ):
        # S1 : dimension effective = features structurelles + embedding si actif
        _d_eff = vocabulary.d_clause + (word_embedding.d_emb if word_embedding is not None else 0)
        if hasattr(graph, 'd_out') and graph.d_out != _d_eff:
            raise ValueError(
                f"RGCNLayer.d_out={graph.d_out} ≠ d_effective={_d_eff} "
                f"(vocabulary.d_clause={vocabulary.d_clause}"
                + (f" + word_embedding.d_emb={word_embedding.d_emb}" if word_embedding is not None else "")
                + ") : instanciez RGCNLayer avec d_out=d_effective."
            )
        self.encoder = encoder
        self.graph = graph
        self.vocabulary = vocabulary
        self.decoder = decoder
        self.taxonomies_dir = taxonomies_dir
        self.temperature = float(temperature)
        self.node_types = list(node_types) if node_types is not None else list(NODE_TYPES)
        self.relation_types = list(relation_types) if relation_types is not None else list(RELATION_TYPES)
        self.all_pairs = all_pairs
        self.word_embedding = word_embedding
        self.bidirectional = bidirectional

        # S5 : liste des couches R-GCN (≥1). Couche 0 = graph passé en paramètre.
        self.n_rgcn_layers = n_rgcn_layers
        self._graph_layers: list = [graph]
        if n_rgcn_layers > 1:
            if hasattr(graph, 'd_in') and hasattr(graph, 'd_out') and hasattr(graph, 'n_relations'):
                from ..layer3.reference import RGCNLayer
                for extra_i in range(1, n_rgcn_layers):
                    self._graph_layers.append(
                        RGCNLayer(
                            d_in=graph.d_in, d_out=graph.d_out,
                            n_relations=graph.n_relations,
                            seed=extra_i * 100 + 42,
                        )
                    )
            else:
                import warnings as _w
                _w.warn(
                    f"n_rgcn_layers={n_rgcn_layers} requiert un graph avec d_in/d_out/n_relations. "
                    "Repli sur n_rgcn_layers=1.",
                    UserWarning, stacklevel=2,
                )
                self.n_rgcn_layers = 1

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
        # Cache reps — utilisé par backward pour word_embedding.backward()
        self._cached_reps: list | None = None
        # S10 : accumulateurs de gradients pour mini-batch
        self._accum_node_grads = None
        self._accum_edge_grads = None
        self._accum_rgcn_grads = None
        self._accum_dec_grads = None
        self._accum_dec_attn = None

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
        self._cached_reps = None
        _snap = hasattr(self.encoder, 'snapshot_node_cache')

        if not reps:
            return emit(text, [], [], [], [], [])

        self._cached_reps = reps

        # Couche 1 — vectorisation (S1 : word_embedding optionnel)
        clause_vecs = np.stack([
            vectorize_clause(r, self.vocabulary, self.word_embedding) for r in reps
        ])  # (N, D_effective)
        self._cached_clause_vecs = clause_vecs

        # S3 : chemin batch si forward_batch disponible et pas de snapshots requis
        _has_batch = hasattr(self.encoder, 'forward_batch')

        # Couche 2 — prédiction des types de nœuds (premier passage)
        node_snapshots: list = []
        if _has_batch and not _snap:
            node_logits = self.encoder.forward_batch(clause_vecs)
        else:
            node_logits_list: list[np.ndarray] = []
            for v in clause_vecs:
                node_logits_list.append(self.encoder.forward_node(v))
                if _snap:
                    node_snapshots.append(self.encoder.snapshot_node_cache())
            node_logits = np.stack(node_logits_list)

        # Couche 3 — R-GCN message passing (AVANT edge classification)
        enriched = clause_vecs
        edge_triples: list[tuple[int, int, str, float, bool, int | None]] = []
        if len(reps) > 1:
            # Paires d'arêtes pour le R-GCN (toutes les paires si all_pairs, sinon adjacentes)
            _edge_pairs_rgcn = (
                [(i, j) for i in range(len(reps)) for j in range(i + 1, len(reps))]
                if self.all_pairs
                else [(i, i + 1) for i in range(len(reps) - 1)]
            )
            # Pour le R-GCN, on a besoin d'arêtes même sans gold — utiliser la prédiction courante
            edge_index_rgcn = np.array(
                [[p[0] for p in _edge_pairs_rgcn], [p[1] for p in _edge_pairs_rgcn]],
                dtype=np.int64,
            ) if _edge_pairs_rgcn else np.zeros((2, 0), dtype=np.int64)
            # Types d'arêtes : type 0 uniforme (proxy neutre — les logits nœuds
            # n'indexent pas les relations, argmax en ferait un bug sémantique)
            edge_type_idxs_rgcn = np.zeros(len(_edge_pairs_rgcn), dtype=np.int64)

            self._cached_edge_index = edge_index_rgcn
            self._cached_edge_type_idxs = edge_type_idxs_rgcn

            # Message passing bidirectionnel
            if self.bidirectional and edge_index_rgcn.shape[1] > 0:
                rev_index = edge_index_rgcn[[1, 0], :]
                rev_types = edge_type_idxs_rgcn + len(self.relation_types)
                edge_index_mp = np.concatenate([edge_index_rgcn, rev_index], axis=1)
                edge_types_mp = np.concatenate([edge_type_idxs_rgcn, rev_types])
            else:
                edge_index_mp = edge_index_rgcn
                edge_types_mp = edge_type_idxs_rgcn

            for _layer in self._graph_layers:
                enriched = _layer.message_pass(enriched, edge_index_mp, edge_types_mp)

        # Deuxième passage nœuds sur les vecteurs enrichis
        if len(reps) > 1:
            node_snapshots = []
            if _has_batch and not _snap:
                node_logits2 = self.encoder.forward_batch(enriched)
            else:
                node_logits2_list: list[np.ndarray] = []
                for v in enriched:
                    node_logits2_list.append(self.encoder.forward_node(v))
                    if _snap:
                        node_snapshots.append(self.encoder.snapshot_node_cache())
                node_logits2 = np.stack(node_logits2_list)
            node_logits = node_logits2

        node_type_idxs = np.argmax(node_logits, axis=1)
        node_types = [self.node_types[i] for i in node_type_idxs]
        self._cached_enriched_vecs = enriched

        # --- Edge classification CLOSED-LOOP : utilise enriched_vecs + node_logits ---
        edge_vecs: list[np.ndarray] = []
        all_edge_logits: list[np.ndarray] = []
        edge_snapshots: list = []
        if len(reps) >= 2:
            real_n = n_total_clauses if n_total_clauses is not None else len(reps)
            _edge_pairs = (
                [(i, j) for i in range(len(reps)) for j in range(i + 1, len(reps))]
                if self.all_pairs
                else [(i, i + 1) for i in range(len(reps) - 1)]
            )
            # Softmax des logits nœuds comme features d'arête
            node_type_probs = _softmax(node_logits)
            for src_i, dst_i in _edge_pairs:
                real_src = clause_positions[src_i] if clause_positions else src_i
                real_dst = clause_positions[dst_i] if clause_positions else dst_i
                connector = (connector_reps[src_i]
                             if connector_reps and src_i < len(connector_reps) and dst_i == src_i + 1
                             else None)
                # Vectorizer avec les reps originales pour les features syntaxiques
                edge_vec_base = vectorize_edge(
                    reps[src_i], reps[dst_i], connector,
                    real_src, real_dst, real_n,
                    self.vocabulary, self.word_embedding,
                )
                # Enrichir avec les representations R-GCN + node type predictions
                enriched_edge = np.concatenate([
                    edge_vec_base,
                    enriched[src_i],      # representation R-GCN du source
                    enriched[dst_i],      # representation R-GCN du destination
                    node_type_probs[src_i],  # proba types nœud source (7 dims)
                    node_type_probs[dst_i],  # proba types nœud destination (7 dims)
                ])
                edge_vecs.append(enriched_edge)
                edge_logit = self.encoder.forward_edge(enriched_edge)
                all_edge_logits.append(edge_logit)
                if _snap:
                    edge_snapshots.append(self.encoder.snapshot_edge_cache())
                rel_idx = int(np.argmax(edge_logit))
                rel_conf = float(_softmax((edge_logit / self.temperature).reshape(1, -1))[0, rel_idx])
                marker_tok_id = connector.token_span[0] if connector is not None else None
                negated = _detect_negation(reps[src_i], reps[dst_i], connector)
                edge_triples.append((src_i, dst_i, self.relation_types[rel_idx], rel_conf, negated, marker_tok_id))

        if edge_vecs:
            self._cached_edge_vecs = np.stack(edge_vecs)
            self._cached_edge_logits = np.stack(all_edge_logits)
            if _snap:
                self._cached_edge_snapshots = edge_snapshots

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

        return emit(text, node_types, node_labels, token_spans,
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

    def get_enriched_vectors(self) -> np.ndarray | None:
        """
        Retourne les vecteurs enrichis du dernier forward(), ou None si aucun.

        H1 correction : utilisé pour le workflow decode() après forward() :
          cir_json = pipeline.forward(reps, text)
          surface = decoder.decode(pipeline.get_enriched_vectors())
        """
        return self._cached_enriched_vecs

    def analyze(
        self,
        text: str,
        gcn_bin: str = "gcn",
        taxonomy_dir=None,
        text_parser=None,
    ) -> dict:
        """
        Texte brut → CausalIR dict (bridge + forward en une opération).

        text_parser (optionnel) : tout objet implémentant le Protocol TextParser
          (layer0/interface.py). Si None, utilise GCNBridgeParser(gcn_bin, taxonomy_dir).
        Qualité approximative si text_parser=None — voir frontend.bridge.

        Raises:
            GCNBridgeError: si gcn-cli est absent ou l'appel échoue (quand text_parser=None).
        """
        import warnings
        from ..frontend.bridge import GCNBridgeParser
        if text_parser is None:
            warnings.warn(
                "CGNPipeline.analyze() produit des UDRepresentation approximatifs. "
                "Voir frontend.bridge pour les limitations de qualité.",
                UserWarning,
                stacklevel=2,
            )
            text_parser = GCNBridgeParser(gcn_bin, taxonomy_dir)
        reps, connector_reps = text_parser.parse(text)
        return self.forward(reps, text, connector_reps=connector_reps)

    def analyze_or_skip(
        self,
        text: str,
        gcn_bin: str = "gcn",
        taxonomy_dir=None,
    ) -> dict | None:
        """
        Comme analyze(), retourne None si gcn est absent ou l'appel échoue.

        Usage recommandé pour les pipelines CI/CD sans gcn-cli installé.
        Vérifie via shutil.which() avant d'appeler le subprocess.
        Aucun UserWarning émis (l'appelant connaît les limitations).

        Returns:
            CausalIR dict, ou None si gcn_bin introuvable ou erreur.
        """
        import shutil
        from ..frontend.bridge import _call_gcn_analyze, _cir_to_reps_and_connectors, GCNBridgeError
        if shutil.which(gcn_bin) is None:
            return None
        try:
            cir = _call_gcn_analyze(text, gcn_bin, taxonomy_dir)
            reps, connector_reps = _cir_to_reps_and_connectors(cir)
            return self.forward(reps, text, connector_reps=connector_reps)
        except GCNBridgeError:
            return None

    def loss(
        self,
        node_logits: np.ndarray,    # (N, 7)  — logits nœuds du forward
        edge_logits: np.ndarray | None,  # (E, 11) — logits arêtes du forward, ou None
        gold_node: np.ndarray,      # (N,) int — indices dans NODE_TYPES
        gold_edge: np.ndarray | None = None,  # (E,) int — indices dans RELATION_TYPES
        edge_loss_weight: float = 1.0,  # pondération relative edge_loss / node_loss
        gold_surface: np.ndarray | None = None,  # (T,) int — tokens gold pour le décodeur
        node_class_weights: np.ndarray | None = None,  # (7,) float — poids par classe nœud
        edge_class_weights: np.ndarray | None = None,  # (11,) float — poids par classe arête
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
        node_loss, d_node = _cross_entropy(node_logits, gold_node, node_class_weights)

        if edge_logits is not None and gold_edge is not None and len(edge_logits) > 0:
            if len(edge_logits) != len(gold_edge):
                raise ValueError(
                    f"Désalignement edge_logits/gold_edge : {len(edge_logits)} logits vs {len(gold_edge)} labels"
                )
            edge_loss, d_edge = _cross_entropy(edge_logits, gold_edge, edge_class_weights)
        else:
            edge_loss = 0.0
            d_edge = np.zeros((0, len(self.relation_types)), dtype=np.float32)

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
            # S11 : propager le gradient décodeur vers le R-GCN
            if (d_node_embs is not None
                    and d_enriched is not None
                    and d_node_embs.shape == d_enriched.shape):
                d_enriched += d_node_embs

        # --- Rétropropagation R-GCN (S5 : boucle sur _graph_layers en ordre inverse) ---
        if (self._cached_edge_index is not None
                and self._cached_enriched_vecs is not None):
            d_curr = d_enriched
            for _layer in reversed(self._graph_layers):
                if hasattr(_layer, 'backward_message_pass'):
                    d_curr, graph_grads = _layer.backward_message_pass(d_curr)
                    _layer.update(graph_grads, lr)

        # --- Word embedding backward (S2) ---
        if (self.word_embedding is not None
                and self._cached_reps is not None
                and d_enriched is not None
                and d_enriched.shape[1] > self.vocabulary.d_clause):
            d_emb_slice = d_enriched[:, self.vocabulary.d_clause:]
            for i in range(min(len(self._cached_reps), len(d_emb_slice))):
                self.word_embedding.backward(d_emb_slice[i], self._cached_reps[i].root_lemma)
            self.word_embedding.update(lr)

    def backward_accumulate(
        self,
        d_node_logits: np.ndarray,
        d_edge_logits: np.ndarray,
    ) -> None:
        """S10 : accumule les gradients sans appeler update. Utiliser avec apply_accumulated_gradients()."""
        if not hasattr(self.encoder, 'backward_node_dx'):
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

        all_node_grads = None
        d_enriched = np.zeros((N, vecs.shape[1]), dtype=np.float32)
        for i in range(n):
            if _has_node_snap:
                self.encoder.restore_node_cache(self._cached_node_snapshots[i])
            else:
                self.encoder.forward_node(vecs[i])
            grads_i, dx_i = self.encoder.backward_node_dx(d_node_logits[i])
            d_enriched[i] = dx_i
            if all_node_grads is None:
                all_node_grads = [(dW.copy(), db.copy()) for dW, db in grads_i]
            else:
                for j, (dW_i, db_i) in enumerate(grads_i):
                    all_node_grads[j] = (all_node_grads[j][0] + dW_i, all_node_grads[j][1] + db_i)

        all_edge_grads = None
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
                        all_edge_grads[j] = (all_edge_grads[j][0] + dW_i, all_edge_grads[j][1] + db_i)

        # Accumulation (liste de tuples → somme)
        def _acc(existing, new):
            if existing is None:
                return new
            if new is None:
                return existing
            return [(e[0] + n[0], e[1] + n[1]) for e, n in zip(existing, new)]

        self._accum_node_grads = _acc(self._accum_node_grads, all_node_grads)
        self._accum_edge_grads = _acc(self._accum_edge_grads, all_edge_grads)

        # R-GCN gradients
        if (self._cached_edge_index is not None
                and self._cached_enriched_vecs is not None):
            d_curr = d_enriched
            layer_grads_list = []
            for _layer in reversed(self._graph_layers):
                if hasattr(_layer, 'backward_message_pass'):
                    d_curr, g = _layer.backward_message_pass(d_curr)
                    layer_grads_list.append((_layer, g))
            if self._accum_rgcn_grads is None:
                self._accum_rgcn_grads = [(lyr, [gg.copy() for gg in g]) for lyr, g in layer_grads_list]
            else:
                for (_, acc_g), (_, new_g) in zip(self._accum_rgcn_grads, layer_grads_list):
                    for i in range(len(acc_g)):
                        acc_g[i] += new_g[i]

        # --- Décodeur backward accumulation (B3) ---
        if (self.decoder is not None
                and self._cached_decode_gradient is not None
                and hasattr(self.decoder, 'backward_decode')):
            _d_node_embs, dec_grads, d_attn_vec = self.decoder.backward_decode(
                self._cached_decode_gradient
            )
            if self._accum_dec_grads is None:
                self._accum_dec_grads = [g.copy() for g in dec_grads]
                self._accum_dec_attn = d_attn_vec.copy()
            else:
                for i in range(len(self._accum_dec_grads)):
                    self._accum_dec_grads[i] += dec_grads[i]
                self._accum_dec_attn += d_attn_vec

        # --- Word embedding backward accumulation (S2) ---
        if (self.word_embedding is not None
                and self._cached_reps is not None
                and d_enriched.shape[1] > self.vocabulary.d_clause):
            d_emb_slice = d_enriched[:, self.vocabulary.d_clause:]
            for i in range(min(len(self._cached_reps), len(d_emb_slice))):
                self.word_embedding.backward(d_emb_slice[i], self._cached_reps[i].root_lemma)
            # update() appelé dans apply_accumulated_gradients() avec normalisation

    def apply_accumulated_gradients(self, lr: float, n_samples: int = 1) -> None:
        """S10 : applique les gradients accumulés normalisés par n_samples."""
        if self._accum_node_grads is not None and hasattr(self.encoder, 'update_node'):
            norm = [(dW / n_samples, db / n_samples) for dW, db in self._accum_node_grads]
            self.encoder.update_node(norm, lr)
        if self._accum_edge_grads is not None and hasattr(self.encoder, 'update_edge'):
            norm = [(dW / n_samples, db / n_samples) for dW, db in self._accum_edge_grads]
            self.encoder.update_edge(norm, lr)
        if self._accum_rgcn_grads is not None:
            for _layer, acc_g in self._accum_rgcn_grads:
                _layer.update([g / n_samples for g in acc_g], lr)
        if self._accum_dec_grads is not None and self.decoder is not None:
            norm_grads = [g / n_samples for g in self._accum_dec_grads]
            norm_attn = self._accum_dec_attn / n_samples
            self.decoder.update(norm_grads, norm_attn, lr)
        self._accum_node_grads = None
        self._accum_edge_grads = None
        self._accum_rgcn_grads = None
        self._accum_dec_grads = None
        self._accum_dec_attn = None

        # --- Word embedding update (S2) ---
        if self.word_embedding is not None:
            self.word_embedding.update(lr / max(n_samples, 1))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _cross_entropy(
    logits: np.ndarray,   # (N, C)
    labels: np.ndarray,   # (N,) int
    class_weights: np.ndarray | None = None,  # (C,) float — poids par classe
) -> tuple[float, np.ndarray]:
    """Cross-entropie NumPy. Retourne (loss, d_logits) normalisés par N.

    Si class_weights est fourni, pondère la loss par le poids de la classe gold.
    Utile pour rééquilibrer les classes rares (ex: edge classification).

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
    per_sample_loss = -np.log(probs[np.arange(N), labels] + 1e-9)
    if class_weights is not None:
        weights = class_weights[labels]                     # (N,)
        per_sample_loss *= weights
    loss = float(per_sample_loss.mean())
    d_logits = probs.copy()
    d_logits[np.arange(N), labels] -= 1.0
    if class_weights is not None:
        d_logits *= class_weights[labels][:, np.newaxis]    # (N, 1) — poids gold par sample
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
