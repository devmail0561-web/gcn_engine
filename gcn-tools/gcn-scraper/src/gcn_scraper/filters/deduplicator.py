"""Déduplication exacte + near-dedup par shingling."""
from __future__ import annotations
import hashlib


class Deduplicator:
    """Détecte les doublons exacts (hash MD5) et near-doublons (shingling Jaccard)."""

    def __init__(self, shingle_size: int = 4, jaccard_threshold: float = 0.7):
        self._seen_hashes: set[str] = set()
        self._seen_shingles: list[frozenset] = []
        self.shingle_size = shingle_size
        self.jaccard_threshold = jaccard_threshold

    def is_duplicate(self, text: str) -> bool:
        """True si la phrase est un doublon exact ou near-doublon."""
        normalized = text.strip().lower()
        h = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        if h in self._seen_hashes:
            return True

        tokens = normalized.split()
        if len(tokens) >= self.shingle_size:
            shingles = frozenset(
                tuple(tokens[i:i + self.shingle_size])
                for i in range(len(tokens) - self.shingle_size + 1)
            )
            for seen in self._seen_shingles:
                intersection = len(shingles & seen)
                union = len(shingles | seen)
                if union > 0 and intersection / union > self.jaccard_threshold:
                    self._seen_hashes.add(h)
                    return True
            self._seen_shingles.append(shingles)

        self._seen_hashes.add(h)
        return False

    def reset(self) -> None:
        self._seen_hashes.clear()
        self._seen_shingles.clear()

    @property
    def seen_count(self) -> int:
        return len(self._seen_hashes)
