# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
GCNEngine — moteur de requêtes causales sur corpus.

Charge un checkpoint et expose l'extraction causale sur un corpus
de textes ou de code. Chaque résultat est tracé jusqu'à sa source.

Ce que GCN fait :
  - Analyser une phrase pré-segmentée → CausalIR
  - Analyser un corpus (batch) → liste de CausalIR
  - Charger un checkpoint sans connaître l'architecture

Ce que GCN ne fait PAS :
  - Segmenter du texte brut (→ frontends Rust via gcn-cli)
  - Générer du texte libre (→ LLMs)
  - Suivre des instructions générales (→ LLMs)

Usage :
    from gcn_python import GCNEngine

    engine = GCNEngine.from_pretrained("model.npz")

    # Une phrase pré-segmentée
    cir = engine.analyze("The vulnerability enables remote code execution.")

    # Un corpus
    cirs = engine.analyze_batch(corpus_lines)
    graph = CausalGraph.from_cirs(cirs)
"""
from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from pathlib import Path

# Imports locaux tardifs (inside methods) pour éviter les imports circulaires


class GCNEngine:
    """
    Interface du moteur GCN Causal Engine.

    Charge un checkpoint et expose l'extraction causale sur des phrases
    pré-segmentées. La segmentation du texte brut est la responsabilité
    des frontends (gcn-cli Rust) ou de l'appelant.

    Exemples
    --------
    >>> engine = GCNEngine.from_pretrained("model.npz")
    >>> cir = engine.analyze("The auth bypass enables data exfiltration.")
    >>> cir["edges"][0][2]["relation"]
    'enable'
    >>> cirs = engine.analyze_batch(open("corpus.txt").readlines())
    """

    def __init__(self, pipeline, text_parser=None):
        self._pipeline = pipeline
        self._text_parser = text_parser
        self._pipeline.encoder.training = False
        for layer in getattr(self._pipeline, "_graph_layers", []):
            if hasattr(layer, "training"):
                layer.training = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        gcn_bin: str = "gcn",
        device: str = "cpu",
        trusted: bool = False,
        taxonomy_dir=None,
    ) -> GCNEngine:
        """
        Charge un modèle depuis un checkpoint .npz et retourne un GCNEngine prêt.

        Le checkpoint encode toute l'architecture (dimensions, bidirectionnel,
        word embeddings…) — pas besoin de spécifier l'architecture manuellement.

        Sécurité : `trusted=False` par défaut refuse le chargement (le .npz
        exige `allow_pickle=True` → exécution de pickle). Passez `trusted=True`
        uniquement pour un checkpoint local de confiance.

        Parameters
        ----------
        checkpoint : chemin vers un fichier .npz produit par gcn-train
        gcn_bin    : chemin vers le binaire gcn-cli Rust (pour le parsing UD)
        device     : "cpu" ou "cuda" (pour RGCNLayerGAT PyTorch)
        trusted    : opt-in explicite pour un fichier local de confiance
        """
        from .constants import NODE_TYPES, RELATION_TYPES
        from .layer1.features import FeatureVocabulary
        from .layer2.reference import MLPEncoder
        from .layer3.reference import RGCNLayer
        from .pipeline.cgnp import CGNPipeline
        from .security import guarded_np_load
        from .training.checkpoint import load_checkpoint

        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint}")
        if not trusted:
            raise RuntimeError(
                f"Refus de charger {checkpoint.name} : checkpoint .npz non fiable par défaut "
                "(allow_pickle requis → exécution de pickle). Relancez avec trusted=True "
                "pour un fichier local de confiance."
            )

        # Garde anti-RCE : audit pickle (allowlist numpy) AVANT toute
        # désérialisation, même quand l'appelant a posé trusted=True.
        data = guarded_np_load(checkpoint)

        # Lire les métadonnées d'architecture sauvegardées par save_checkpoint
        if "_arch_json" not in data:
            # C1.1 : aligné sur load_checkpoint/eval_runner (warn, pas raise).
            # Warn fort car défauts = métriques potentiellement trompeuses.
            warnings.warn(
                f"Checkpoint {checkpoint.name} sans _arch_json — repli sur défauts. "
                "Re-entraîner avec gcn-train >= 2.1.0 pour supprimer cet avertissement.",
                UserWarning,
                stacklevel=2,
            )
            arch: dict = {}
        else:
            # M5 : checkpoints .npz = artefacts locaux de confiance (allow_pickle requis
            # pour _arch_json/_vocab_json). Ne jamais charger un .npz non fiable.
            arch = json.loads(str(data["_arch_json"][0]))
        d_eff         = int(arch.get("d_eff", FeatureVocabulary().d_clause))
        d_emb         = int(arch.get("d_emb", 0))
        n_rel         = int(arch.get("n_relations", len(RELATION_TYPES)))
        bidirectional = bool(arch.get("bidirectional", arch.get("bidi_flag", False)))
        all_pairs = bool(arch.get("all_pairs", False))
        n_rgcn_layers = int(arch.get("n_rgcn_layers", 1))
        graph_class  = arch.get("graph_class", "RGCNLayer")
        # §1 (amelioration_v3) — défauts = comportement historique
        clause_pooling = str(arch.get("clause_pooling", "root"))
        subject_object_emb = bool(arch.get("subject_object_emb", False))
        gat_residual = bool(arch.get("gat_residual", False))
        n_gat_heads = int(arch.get("n_gat_heads", 1))
        gat_layernorm = bool(arch.get("gat_layernorm", False))
        gat_output_activation = str(arch.get("gat_output_activation", "sigmoid"))
        rgcn_output_activation = str(arch.get("rgcn_output_activation", "sigmoid"))
        rgcn_layernorm = bool(arch.get("rgcn_layernorm", False))
        mlp_hidden = int(arch.get("mlp_hidden", 128))
        freeze_embeddings = bool(arch.get("freeze_embeddings", False))

        vocab = FeatureVocabulary()
        if "_vocab_json" in data:
            vocab = FeatureVocabulary.from_json(str(data["_vocab_json"][0]))

        d_edge = vocab.d_edge_closed_loop(d_eff, len(NODE_TYPES), d_emb,
                                          subject_object_emb)

        # Phase C : substitution pour from_pretrained — TransformerMLPEncoder
        # si arch.get("global_attention", False). d_clause = D_effective (d_eff),
        # jamais vocabulary.d_clause brut.
        _global_attention = bool(arch.get("global_attention", False))
        _mha_heads = int(arch.get("n_gat_heads_mha", 4))
        if _global_attention:
            from .layer2.reference import TransformerMLPEncoder
            encoder = TransformerMLPEncoder(d_clause=d_eff, d_edge=d_edge,
                                            mlp_hidden=mlp_hidden,
                                            n_heads=_mha_heads)
        else:
            encoder = MLPEncoder(d_clause=d_eff, d_edge=d_edge, mlp_hidden=mlp_hidden)

        # Phases B/D : flags RGCNLayerPT persistés (défauts = comportement historique).
        _pairnorm = bool(arch.get("pairnorm", False))
        _drop_edge = float(arch.get("drop_edge", 0.0))
        _use_compgcn = bool(arch.get("use_compgcn", False))
        _d_rel_emb = int(arch.get("d_rel_emb", 32))

        # M5 : RGCNLayerPT accepte aussi device (comme GAT).
        try:
            from .layer3.pytorch_rgcn import RGCNLayerPT
            _has_pt = True
        except ImportError:
            _has_pt = False
        if graph_class == "RGCNLayerGAT":
            try:
                from .layer3.gat import RGCNLayerGAT
                graph = RGCNLayerGAT(d_in=d_eff, d_out=d_eff, n_relations=n_rel,
                                     device=device, n_heads=n_gat_heads,
                                     output_activation=gat_output_activation,
                                     use_layernorm=gat_layernorm)
            except ImportError:
                warnings.warn("PyTorch absent — repli sur RGCNLayer (NumPy).", UserWarning,
                              stacklevel=2)
                graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=n_rel,
                                  output_activation=rgcn_output_activation,
                                  use_layernorm=rgcn_layernorm)
        elif graph_class == "RGCNLayerPT" and _has_pt:
            from .layer3.pytorch_rgcn import RGCNLayerPT
            graph = RGCNLayerPT(d_in=d_eff, d_out=d_eff, n_relations=n_rel,
                                device=device, pairnorm=_pairnorm,
                                drop_edge=_drop_edge,
                                use_compgcn=_use_compgcn, d_rel_emb=_d_rel_emb)
        else:
            graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=n_rel,
                              output_activation=rgcn_output_activation,
                              use_layernorm=rgcn_layernorm)

        word_embedding = None
        if d_emb > 0:
            from .layer1.embedding import WordEmbedding
            word_embedding = WordEmbedding(d_emb=d_emb)
            if freeze_embeddings:
                word_embedding.frozen = True

        # Restaurer les hyperparamètres d'inférence depuis l'arch — sans ça,
        # analyze() utilise les défauts (seuil 0.0, morph actif, temp 1.0) même
        # si le modèle a été entraîné avec d'autres valeurs.
        edge_threshold = float(arch.get("edge_threshold", 0.0))
        drop_morph     = bool(arch.get("drop_morph", False))
        temperature    = float(arch.get("temperature", 1.0))
        bfs_depth      = arch.get("bfs_depth")
        if bfs_depth is not None:
            bfs_depth = int(bfs_depth)

        # M5 : reconstruit les couches R-GCN extra (n_rgcn_layers>1).
        pipeline = CGNPipeline(
            encoder=encoder, graph=graph, vocabulary=vocab,
            word_embedding=word_embedding, bidirectional=bidirectional,
            all_pairs=all_pairs, n_rgcn_layers=n_rgcn_layers,
            edge_threshold=edge_threshold, drop_morph=drop_morph,
            temperature=temperature, bfs_depth=bfs_depth,
            clause_pooling=clause_pooling,
            subject_object_emb=subject_object_emb,
            gat_residual=gat_residual,
        )
        # Phases B/C/D : attrs d'arch lus par load_checkpoint (validation
        # global_attention/use_compgcn) — posés avant load_checkpoint.
        pipeline.global_attention = _global_attention
        pipeline.mha_heads = _mha_heads
        pipeline.pairnorm = _pairnorm
        pipeline.drop_edge = _drop_edge
        pipeline.use_compgcn = _use_compgcn
        pipeline.d_rel_emb = _d_rel_emb
        pipeline.two_pass_val = bool(arch.get("two_pass_val", True))
        load_checkpoint(pipeline, checkpoint, trusted=True)

        # Text parser : gcn-cli si disponible, sinon bridge heuristique
        import shutil
        text_parser = None
        if shutil.which(gcn_bin):
            from .frontend.bridge import GCNBridgeParser
            text_parser = GCNBridgeParser(gcn_bin, taxonomy_dir)

        return cls(pipeline, text_parser=text_parser)

    # ------------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> dict:
        """
        Une phrase pré-segmentée → CausalIR dict.

        La segmentation du texte brut en phrases est la responsabilité
        des frontends (gcn-cli) ou de gcn-discuss. Le moteur reçoit
        une phrase et produit un CIR.

        Returns
        -------
        dict CausalIR :
          {
            "source_text": str,
            "nodes": [{"node_type": str, "label": str, ...}],
            "edges": [[src_id, dst_id, {"relation": str, "confidence": float, ...}]],
          }
        """
        return self._pipeline.analyze(
            text,
            text_parser=self._text_parser,
        )

    def verbalize(self, text: str, use_neural: bool = False) -> dict:
        """
        Texte brut → CIR prédit + texte verbalisé.

        use_neural=False (défaut) : ReferenceDecoder (templates statiques).
        use_neural=True : TrainableDecoder si chargé depuis checkpoint, sinon repli template.

        Requiert gcn-cli ou text_parser passé au constructeur.
        Retourne {"cir": <CausalIR dict>, "text": <str>}.

        Note : le TrainableDecoder est un RNN piloté par pooling attentionnel des nœuds
        sans réinjection de tokens — qualité de génération limitée (vrai teacher forcing = Phase 3).
        """
        cir = self._pipeline.analyze(text, text_parser=self._text_parser)
        if use_neural and self._pipeline.decoder is not None:
            enriched = self._pipeline.get_enriched_vectors()
            if enriched is not None:
                return {"cir": cir, "text": self._pipeline.decoder.decode(enriched)}
        from .verbalizer.decoder import ReferenceDecoder
        return {"cir": cir, "text": ReferenceDecoder().decode_cir(cir) or ""}

    def predict_links(self, text: str, threshold: float = 0.5,
                        bfs_depth: int | None = None) -> list[tuple[int, int, float]]:
        """Prédit les arêtes manquantes d'une phrase (tête LinkPredHead).

        Lève RuntimeError si aucune tête attachée (voir CGNPipeline.predict_links).
        bfs_depth limite les candidats aux nœuds à ≤ depth sauts.
        None = valeur du pipeline (config d'entraînement via --bfs-depth),
        elle-même None = tous les candidats (illimité).
        """
        if bfs_depth is None:
            bfs_depth = getattr(self._pipeline, "bfs_depth", None)
        cir = self.analyze(text)
        vecs = self._pipeline.get_enriched_vectors()
        if vecs is None:
            raise RuntimeError("predict_links : aucun vecteur (analyze sans forward?).")
        existing = set()
        for e in cir.get("edges", []) or []:
            if isinstance(e, (list, tuple)) and len(e) == 3:
                existing.add((int(e[0]), int(e[1])))
        n = len(vecs)
        if bfs_depth is not None:
            from .layer3.link_pred import candidates_within_depth
            adj: dict[int, list[int]] = {}
            for s, d in existing:
                adj.setdefault(s, []).append(d)
            cand_set = set()
            for s in range(n):
                for t in candidates_within_depth(adj, s, int(bfs_depth)):
                    if (s, t) not in existing:
                        cand_set.add((s, t))
            candidates = sorted(cand_set)
        else:
            candidates = [(i, j) for i in range(n) for j in range(n)
                          if i != j and (i, j) not in existing]
        scored = self._pipeline.predict_links(candidates, vecs)
        return [(s, d, c) for s, d, c in scored if c >= threshold]

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """
        Liste de phrases pré-segmentées → liste de CausalIR.

        Chaque élément est une phrase individuelle (pas un paragraphe).
        La segmentation est la responsabilité de l'appelant.
        """
        return [self.analyze(t) for t in texts if t.strip()]

    def stream(self, source) -> Iterator[dict]:
        """
        Itérateur paresseux — une ligne = une phrase pré-segmentée.

        Convient aux grands corpus. Ne charge pas tout en mémoire.

        Usage :
            for cir in engine.stream(open("corpus.txt")):
                if cir["edges"]:
                    process(cir)
        """
        for line in source:
            text = line.strip() if hasattr(line, "strip") else str(line).strip()
            if text:
                yield self.analyze(text)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def node_types(self) -> list[str]:
        """Les 7 types de nœuds que le modèle peut prédire."""
        return list(self._pipeline.node_types)

    @property
    def relation_types(self) -> list[str]:
        """Les 11 types de relations que le modèle peut prédire."""
        return list(self._pipeline.relation_types)

    def __repr__(self) -> str:
        _we = getattr(self._pipeline, 'word_embedding', None)
        d_eff = self._pipeline.vocabulary.d_clause_effective(
            _we.d_emb if _we is not None else 0,
            bool(getattr(self._pipeline, 'subject_object_emb', False)))
        n_rel = len(self._pipeline.relation_types)
        parser = "gcn-cli" if self._text_parser else "bridge heuristique"
        return (
            f"GCNEngine(d_eff={d_eff}, n_relations={n_rel}, "
            f"node_types={len(self.node_types)}, parser={parser!r})"
        )
