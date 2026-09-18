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
from pathlib import Path
from typing import Iterator

import numpy as np

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
    ) -> "GCNEngine":
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
        from .training.checkpoint import load_checkpoint
        from .layer1.features import FeatureVocabulary
        from .layer2.reference import MLPEncoder
        from .layer3.reference import RGCNLayer
        from .pipeline.cgnp import CGNPipeline
        from .constants import NODE_TYPES, RELATION_TYPES

        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint}")
        if not trusted:
            raise RuntimeError(
                f"Refus de charger {checkpoint.name} : checkpoint .npz non fiable par défaut "
                "(allow_pickle requis → exécution de pickle). Relancez avec trusted=True "
                "pour un fichier local de confiance."
            )

        data = np.load(checkpoint, allow_pickle=True)

        # Lire les métadonnées d'architecture sauvegardées par save_checkpoint
        if "_arch_json" not in data:
            raise ValueError(
                f"Checkpoint {checkpoint.name} ne contient pas _arch_json. "
                "Re-entraîner avec gcn-train >= 2.1.0 pour générer ce champ."
            )
        # M5 : checkpoints .npz = artefacts locaux de confiance (allow_pickle requis
        # pour _arch_json/_vocab_json). Ne jamais charger un .npz non fiable.
        arch = json.loads(str(data["_arch_json"][0]))
        d_eff        = arch["d_eff"]
        d_emb        = arch["d_emb"]
        n_rel        = arch["n_relations"]
        bidirectional = arch["bidirectional"]
        n_rgcn_layers = int(arch.get("n_rgcn_layers", 1))
        graph_class  = arch.get("graph_class", "RGCNLayer")

        vocab = FeatureVocabulary()
        if "_vocab_json" in data:
            vocab = FeatureVocabulary.from_json(str(data["_vocab_json"][0]))

        from .constants import NODE_TYPES
        d_edge = vocab.d_edge_closed_loop(d_eff, len(NODE_TYPES), d_emb)

        encoder = MLPEncoder(d_clause=d_eff, d_edge=d_edge)

        # M5 : RGCNLayerPT accepte aussi device (comme GAT).
        try:
            from .layer3.pytorch_rgcn import RGCNLayerPT  # noqa: F401
            _has_pt = True
        except ImportError:
            _has_pt = False
        if graph_class == "RGCNLayerGAT":
            try:
                from .layer3.gat import RGCNLayerGAT
                graph = RGCNLayerGAT(d_in=d_eff, d_out=d_eff, n_relations=n_rel,
                                     device=device)
            except ImportError:
                warnings.warn("PyTorch absent — repli sur RGCNLayer (NumPy).", UserWarning)
                graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=n_rel)
        elif graph_class == "RGCNLayerPT" and _has_pt:
            from .layer3.pytorch_rgcn import RGCNLayerPT
            graph = RGCNLayerPT(d_in=d_eff, d_out=d_eff, n_relations=n_rel,
                                device=device)
        else:
            graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=n_rel)

        word_embedding = None
        if d_emb > 0:
            from .layer1.embedding import WordEmbedding
            word_embedding = WordEmbedding(d_emb=d_emb)

        # M5 : reconstruit les couches R-GCN extra (n_rgcn_layers>1).
        pipeline = CGNPipeline(
            encoder=encoder, graph=graph, vocabulary=vocab,
            word_embedding=word_embedding, bidirectional=bidirectional,
            n_rgcn_layers=n_rgcn_layers,
        )
        load_checkpoint(pipeline, checkpoint, trusted=True)

        # Text parser : gcn-cli si disponible, sinon bridge heuristique
        import shutil
        text_parser = None
        if shutil.which(gcn_bin):
            from .frontend.bridge import GCNBridgeParser
            text_parser = GCNBridgeParser(gcn_bin)

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
        d_eff = self._pipeline.vocabulary.d_clause
        n_rel = len(self._pipeline.relation_types)
        parser = "gcn-cli" if self._text_parser else "bridge heuristique"
        return (
            f"GCNEngine(d_clause={d_eff}, n_relations={n_rel}, "
            f"node_types={len(self.node_types)}, parser={parser!r})"
        )
