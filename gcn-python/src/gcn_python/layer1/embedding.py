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

    def __init__(self, d_emb: int = 50, seed: int = 42) -> None:
        self.d_emb = d_emb
        self._seed = seed
        self._vocab: dict[str, int] = {"_unk": 0}
        self._lemmas: list[str] = ["_unk"]
        self._rng = np.random.default_rng(seed)
        self._E: np.ndarray = self._rng.normal(
            0, np.sqrt(1.0 / d_emb), (1, d_emb)
        ).astype(np.float32)
        self._grad_accum: np.ndarray | None = None

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

    def backward(self, d_emb: np.ndarray, lemma: str) -> None:
        """Accumule le gradient pour l'embedding de lemma."""
        idx = self._vocab.get(lemma, 0)
        if self._grad_accum is None:
            self._grad_accum = np.zeros_like(self._E)
        self._grad_accum[idx] += d_emb

    def update(self, lr: float) -> None:
        """Étape SGD ; vide les gradients accumulés."""
        if self._grad_accum is not None:
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
        """
        loaded = 0
        with open(path, encoding=encoding) as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) - 1 != self.d_emb:
                    continue  # en-tête ou dimension incompatible
                word = parts[0]
                try:
                    vec = np.array(parts[1:], dtype=np.float32)
                except ValueError:
                    continue
                idx = self.add_lemma(word)
                self._E[idx] = vec
                loaded += 1
        return loaded

    # ── paramètres / checkpoint ───────────────────────────────────────

    def parameters(self) -> list[np.ndarray]:
        return [self._E]

    def to_json(self) -> str:
        """Sérialise la liste des lemmes (ordre = indices)."""
        return json.dumps(self._lemmas, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str, d_emb: int = 50) -> "WordEmbedding":
        """Recrée le vocabulaire depuis to_json(). Les poids sont à restaurer séparément."""
        obj = cls(d_emb=d_emb)
        lemmas = json.loads(s)
        for lemma in lemmas[1:]:  # index 0 = _unk, déjà présent
            obj.add_lemma(lemma)
        return obj
