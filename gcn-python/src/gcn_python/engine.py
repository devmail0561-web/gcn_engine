"""
GCNEngine — API haut niveau, même usage qu'un LLM.

Usage :
    from gcn_python import GCNEngine

    engine = GCNEngine.from_pretrained("model.npz")
    cir    = engine.analyze("Les ventes baissent car la demande recule.")
    cirs   = engine.analyze_batch(["phrase 1", "phrase 2"])
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
    Interface haut niveau du moteur GCN Causal Engine.

    Équivalent de `pipeline("text")` chez HuggingFace :
      - Un seul point d'entrée : texte brut → CausalIR
      - Le checkpoint encode l'architecture — pas besoin de la connaître
      - Mode évaluation activé automatiquement

    Exemples
    --------
    >>> engine = GCNEngine.from_pretrained("model.npz")
    >>> cir = engine.analyze("Les ventes baissent car la demande recule.")
    >>> cir["edges"][0][2]["relation"]
    'cause'

    >>> for cir in engine.stream(open("corpus.txt")):
    ...     print(cir["source_text"], "→", len(cir["edges"]), "relations")
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
    ) -> "GCNEngine":
        """
        Charge un modèle depuis un checkpoint .npz et retourne un GCNEngine prêt.

        Le checkpoint encode toute l'architecture (dimensions, bidirectionnel,
        word embeddings…) — pas besoin de spécifier l'architecture manuellement.

        Parameters
        ----------
        checkpoint : chemin vers un fichier .npz produit par gcn-train
        gcn_bin    : chemin vers le binaire gcn-cli Rust (pour le parsing UD)
        device     : "cpu" ou "cuda" (pour RGCNLayerGAT PyTorch)
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

        data = np.load(checkpoint, allow_pickle=True)

        # Lire les métadonnées d'architecture sauvegardées par save_checkpoint
        if "_arch_json" not in data:
            raise ValueError(
                f"Checkpoint {checkpoint.name} ne contient pas _arch_json. "
                "Re-entraîner avec gcn-train >= 2.1.0 pour générer ce champ."
            )
        arch = json.loads(str(data["_arch_json"][0]))
        d_eff        = arch["d_eff"]
        d_emb        = arch["d_emb"]
        n_rel        = arch["n_relations"]
        bidirectional = arch["bidirectional"]
        graph_class  = arch.get("graph_class", "RGCNLayer")

        vocab = FeatureVocabulary()
        if "_vocab_json" in data:
            vocab = FeatureVocabulary.from_json(str(data["_vocab_json"][0]))

        # Dimension edge : lire depuis le checkpoint via le vocab
        d_edge = vocab.d_edge_closed_loop(d_eff, len(vocab.upos_tags) - 11, d_emb)
        # Recalcul propre depuis le vocab et les dimensions connues
        from .constants import NODE_TYPES
        d_edge = vocab.d_edge_closed_loop(d_eff, len(NODE_TYPES), d_emb)

        encoder = MLPEncoder(d_clause=d_eff, d_edge=d_edge)

        if graph_class == "RGCNLayerGAT":
            try:
                from .layer3.gat import RGCNLayerGAT
                graph = RGCNLayerGAT(d_in=d_eff, d_out=d_eff, n_relations=n_rel,
                                     device=device)
            except ImportError:
                warnings.warn("PyTorch absent — repli sur RGCNLayer (NumPy).", UserWarning)
                graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=n_rel)
        else:
            graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=n_rel)

        word_embedding = None
        if d_emb > 0:
            from .layer1.embedding import WordEmbedding
            word_embedding = WordEmbedding(d_emb=d_emb)

        pipeline = CGNPipeline(
            encoder=encoder, graph=graph, vocabulary=vocab,
            word_embedding=word_embedding, bidirectional=bidirectional,
        )
        load_checkpoint(pipeline, checkpoint)

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
    # Segmentation en phrases (interne)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Segmente un texte en phrases individuelles.

        Utilise une règle simple : découpe sur '.', '!', '?' suivis d'un
        espace et d'une majuscule, ou en fin de texte. Suffisant pour la
        majorité des textes FR/EN. Chaque phrase est nettoyée et filtrée
        si elle est trop courte (< 3 mots) pour être analysable.
        """
        import re
        # Découpe sur ponctuation forte suivie d'espace + majuscule ou fin
        raw = re.split(r'(?<=[.!?])\s+(?=[A-ZÀÂÄÉÈÊËÏÎÔÙÛÜÇ])', text.strip())
        sentences = []
        for s in raw:
            s = s.strip()
            if s and len(s.split()) >= 3:
                sentences.append(s)
        return sentences if sentences else [text.strip()]

    # ------------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> dict:
        """
        Une phrase → CausalIR dict.

        Traite **une seule phrase**. Pour un paragraphe ou un document,
        utiliser analyze_document() qui segmente automatiquement.

        Parameters
        ----------
        text : une phrase en langage naturel (FR ou EN)

        Returns
        -------
        dict CausalIR :
          {
            "source_text": str,
            "nodes": [{"node_type": str, "label": str, ...}],
            "edges": [[src_id, dst_id, {"relation": str, "confidence": float, ...}]],
            "cycles": [],
            "unresolved": [],
          }
        """
        return self._pipeline.analyze(
            text,
            text_parser=self._text_parser,
        )

    def analyze_document(self, text: str) -> list[dict]:
        """
        Paragraphe ou document multi-phrases → liste de CausalIR, une par phrase.

        Segmente automatiquement le texte en phrases, analyse chacune
        indépendamment et retourne la liste des CIR dans l'ordre du texte.
        Les phrases sans relation causale retournent un CIR avec edges=[].

        Parameters
        ----------
        text : paragraphe ou document (plusieurs phrases)

        Returns
        -------
        Liste de dicts CausalIR — un par phrase détectée.

        Exemple
        -------
        >>> cirs = engine.analyze_document(
        ...     "Les ventes baissent car la demande recule. "
        ...     "Si les coûts augmentent, les marges s'effondrent. "
        ...     "La direction a décidé de réduire les effectifs."
        ... )
        >>> len(cirs)
        3
        >>> cirs[0]["edges"][0][2]["relation"]
        'cause'
        >>> cirs[1]["edges"][0][2]["relation"]
        'condition'
        >>> cirs[2]["edges"]   # pas de relation causale détectée
        []
        """
        sentences = self._split_sentences(text)
        return [self.analyze(s) for s in sentences]

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """
        Liste de phrases → liste de CausalIR (une phrase par élément).

        Pour des documents multi-phrases, utiliser analyze_document() sur
        chaque document ou stream() sur un fichier.

        Parameters
        ----------
        texts : liste de phrases (une phrase par élément, pas des paragraphes)

        Returns
        -------
        Liste de dicts CausalIR dans le même ordre que texts.
        """
        return [self.analyze(t) for t in texts]

    def stream(self, source) -> Iterator[dict]:
        """
        Itérateur paresseux sur un fichier ou une liste de textes.

        Chaque ligne est traitée comme une phrase. Pour des fichiers dont
        chaque ligne est un paragraphe, utiliser stream_documents().

        Usage :
            for cir in engine.stream(open("corpus.txt")):
                if cir["edges"]:
                    process(cir)

        Ne charge pas tout en mémoire — convient aux grands corpus.
        """
        for line in source:
            text = line.strip() if hasattr(line, "strip") else str(line).strip()
            if text:
                yield self.analyze(text)

    def stream_documents(self, source) -> Iterator[list[dict]]:
        """
        Itérateur paresseux sur un fichier de paragraphes (une par ligne).

        Chaque ligne est segmentée en phrases et retourne une liste de CIR.

        Usage :
            for sentence_cirs in engine.stream_documents(open("docs.txt")):
                for cir in sentence_cirs:
                    if cir["edges"]:
                        store(cir)
        """
        for line in source:
            text = line.strip() if hasattr(line, "strip") else str(line).strip()
            if text:
                yield self.analyze_document(text)

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
