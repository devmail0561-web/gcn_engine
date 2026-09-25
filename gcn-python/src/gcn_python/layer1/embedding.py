# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import numpy as np


class WordEmbedding:
    """
    Table d'embeddings apprenables pour les lemmes racines (S1/S2/S9).

    Mappe les chaînes root_lemma vers des vecteurs denses de dimension d_emb.
    Supporte le chargement de vecteurs pré-entraînés format GloVe/FastText (S9).

    Usage avec CGNPipeline :
        we = WordEmbedding(d_emb=50)
        pipeline = CGNPipeline(encoder, graph, lang, vocab, word_embedding=we)

    Checkpoint : sauvegardé via save_checkpoint (clés word_emb_E et _word_emb_vocab_json).
    """

    def __init__(self, d_emb: int = 50, seed: int = 42, frozen: bool = False) -> None:
        self.d_emb = d_emb
        self._seed = seed
        self.frozen = frozen  # Amélioration C : gel des pré-entraînés (voir load_from_file)
        self._vocab: dict[str, int] = {"_unk": 0}
        self._lemmas: list[str] = ["_unk"]
        self._rng = np.random.default_rng(seed)
        self._E: np.ndarray = self._rng.normal(
            0, np.sqrt(1.0 / d_emb), (1, d_emb)
        ).astype(np.float32)
        self._grad_accum: np.ndarray | None = None
        # Plage d'indices pré-entraînés (gelée si frozen) — fixée par load_from_file().
        # Les lemmes spéciaux _subj_absent/_obj_absent (indices 1-2) et les lemmes
        # ajoutés après le chargement restent toujours entraînables.
        self._pretrained_start: int | None = None
        self._pretrained_end: int | None = None
        # Amélioration B : vecteurs d'absence appris (comme _unk)
        self.add_lemma('_subj_absent')   # index 1
        self.add_lemma('_obj_absent')    # index 2

    # ── gestion du vocabulaire ────────────────────────────────────────

    def add_lemma(self, lemma: str) -> int:
        """Ajoute un lemme au vocabulaire si absent ; retourne son indice."""
        if lemma not in self._vocab:
            idx = len(self._lemmas)
            self._vocab[lemma] = idx
            self._lemmas.append(lemma)
            new_row = self._rng.normal(
                0, np.sqrt(1.0 / self.d_emb), (1, self.d_emb)
            ).astype(np.float32)
            self._E = np.vstack([self._E, new_row])
        return self._vocab[lemma]

    def build_vocab(self, lemmas: list[str]) -> None:
        """Ajoute tous les lemmes de la liste."""
        for lemma in lemmas:
            self.add_lemma(lemma)

    # ── forward / backward ───────────────────────────────────────────

    def lookup(self, lemma: str) -> np.ndarray:
        """Retourne une copie de l'embedding pour lemma (ou _unk si absent)."""
        idx = self._vocab.get(lemma, 0)
        return self._E[idx].copy()

    def _is_frozen_idx(self, idx: int) -> bool:
        """True si l'indice est gelé (C) : frozen ET dans la plage pré-entraînée."""
        return (
            bool(self.frozen)
            and self._pretrained_start is not None
            and self._pretrained_end is not None
            and self._pretrained_start <= idx < self._pretrained_end
        )

    def backward(self, d_emb: np.ndarray, lemma: str) -> None:
        """Accumule le gradient pour l'embedding de lemma (no-op si gelé)."""
        idx = self._vocab.get(lemma, 0)
        if self._is_frozen_idx(idx):
            return  # accumuler rien
        if self._grad_accum is None:
            self._grad_accum = np.zeros_like(self._E)
        self._grad_accum[idx] += d_emb

    def update(self, lr: float) -> None:
        """Étape SGD ; vide les gradients accumulés.

        Si frozen, seules les lignes hors plage pré-entraînée sont mises à jour
        (_subj_absent/_obj_absent et lemmes ajoutés après le chargement).
        """
        if self._grad_accum is not None:
            if self.frozen and self._pretrained_start is not None:
                grads = self._grad_accum.copy()
                grads[self._pretrained_start:self._pretrained_end] = 0.0
                self._E -= lr * grads
            else:
                self._E -= lr * self._grad_accum
            self._grad_accum = None

    # ── chargement pré-entraîné (S9) ─────────────────────────────────

    def load_from_file(self, path: str, encoding: str = "utf-8") -> int:
        """
        Charge des embeddings pré-entraînés format GloVe/FastText (texte).

        Format attendu : 'mot d1 d2 ... dn' par ligne.
        La première ligne peut être un en-tête 'V d_emb' — ignorée automatiquement.
        Ne charge que les vecteurs dont la dimension correspond à self.d_emb.

        Retourne le nombre de vecteurs chargés avec succès.
        La plage d'indices chargés est mémorisée : avec frozen=True, seuls
        ces indices sont gelés (les spéciaux _subj_absent/_obj_absent et les
        lemmes ajoutés ensuite restent entraînables).
        """
        loaded = 0
        self._pretrained_start = len(self._lemmas)
        loaded = 0
        _first_real_dim: int | None = None
        with open(path, encoding=encoding) as f:
            for line in f:
                parts = line.rstrip().split(" ")
                dim = len(parts) - 1
                if dim <= 1:
                    continue  # en-tête "V d_emb" ou ligne vide
                if _first_real_dim is None:
                    _first_real_dim = dim
                    if _first_real_dim != self.d_emb:
                        raise ValueError(
                            f"Dimension du fichier ({_first_real_dim}) ≠ d_emb ({self.d_emb}). "
                            f"Utilisez --embedding-dim {_first_real_dim} pour ce fichier, "
                            f"ou omettez --embedding-file pour des embeddings aléatoires "
                            f"{self.d_emb}-dim."
                        )
                if dim != self.d_emb:
                    continue  # dimension incompatible (ne devrait pas arriver après la vérif)
                word = parts[0]
                try:
                    vec = np.array(parts[1:], dtype=np.float32)
                except ValueError:
                    continue
                idx = self.add_lemma(word)
                self._E[idx] = vec
                loaded += 1
        self._pretrained_end = len(self._lemmas)
        return loaded

    @classmethod
    def load_from_fasttext(cls, ft_model, vocab: list[str], d_emb: int = 300,
                           frozen: bool = True) -> WordEmbedding:
        """Construit un WordEmbedding depuis un modèle fastText multilingue.

        Convertit fastText → format word2vec texte temporaire → délègue à
        load_from_file() (qui gère _pretrained_start/_pretrained_end).
        Contourne le fait que lookup() ne fait pas d'auto-add.

        Parameters
        ----------
        ft_model : modèle fastText (ou fasttext-wheel) exposant get_word_vector(str).
        vocab : lemmes à extraire.
        d_emb : dimension (300 pour cc.XX.300.bin).
        frozen : gel des pré-entraînés (défaut True).
        """
        try:
            import fasttext  # noqa: F401
        except ImportError:
            try:
                import fasttext_wheel  # noqa: F401
            except ImportError as _e:
                if not hasattr(ft_model, "get_word_vector"):
                    raise ImportError(
                        "load_from_fasttext requiert fasttext-wheel (ou fasttext) : "
                        "pip install fasttext-wheel"
                    ) from _e
                # ft_model duck-typé (mock/test) exposant get_word_vector :
                # on procède sans le package binaire.
        # Contrat du modèle : vérifié INDÉPENDAMMENT de la présence du package.
        # Sans ce contrôle, un modèle invalide déclenchait un AttributeError
        # au milieu de la boucle d'écriture, ou un ImportError trompeur selon
        # que fasttext est installé ou non (test non déterministe).
        if not hasattr(ft_model, "get_word_vector"):
            raise TypeError(
                "load_from_fasttext : ft_model doit exposer get_word_vector(lemma) "
                f"— type reçu : {type(ft_model).__name__}"
            )
        import pathlib
        import tempfile
        obj = cls(d_emb=d_emb, frozen=frozen)
        with tempfile.TemporaryDirectory() as _tmpdir:  # FIX-2 : nettoyage garanti
            tmp = pathlib.Path(_tmpdir) / "_ft_vecs.txt"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(f"{len(vocab)} {d_emb}\n")
                for lemma in vocab:
                    vec = ft_model.get_word_vector(lemma)
                    f.write(lemma + " " + " ".join(f"{v:.6f}" for v in vec) + "\n")
            obj.load_from_file(str(tmp))
        return obj

    # ── paramètres / checkpoint ───────────────────────────────────────

    def parameters(self) -> list[np.ndarray]:
        return [self._E]

    def to_json(self) -> str:
        """Sérialise les lemmes et la plage pré-entraînée (pour from_json)."""
        return json.dumps({
            "lemmas": self._lemmas,
            "__pretrained_start": self._pretrained_start,
            "__pretrained_end": self._pretrained_end,
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str, d_emb: int = 50) -> WordEmbedding:
        """Recrée le vocabulaire depuis to_json(). Les poids sont à restaurer séparément."""
        obj = cls(d_emb=d_emb)
        data = json.loads(s)
        # Compatibilité : ancien format = liste de lemmes directement
        lemmas = data if isinstance(data, list) else data.get("lemmas", [])
        for lemma in lemmas[1:]:  # index 0 = _unk, déjà présent
            obj.add_lemma(lemma)
        if isinstance(data, dict):
            obj._pretrained_start = data.get("__pretrained_start")
            obj._pretrained_end = data.get("__pretrained_end")
        else:
            # C1.4 : format ancien (liste) — plage non restaurée, gel silencieusement inopérant.
            import warnings as _w_emb
            _w_emb.warn(
                "WordEmbedding.from_json : format ancien (liste) — "
                "_pretrained_start/_pretrained_end non restaurés. "
                "freeze_embeddings=True sans effet sur ce vocabulaire.",
                UserWarning,
                stacklevel=2,
            )
        return obj
