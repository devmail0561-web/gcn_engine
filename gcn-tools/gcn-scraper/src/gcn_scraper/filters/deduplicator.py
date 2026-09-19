"""Déduplication exacte + near-dedup par shingling."""
from __future__ import annotations
from collections import deque
import hashlib


class Deduplicator:
    """Détecte les doublons exacts (hash MD5) et near-doublons (shingling Jaccard).

    La mémoire near-dedup est bornée (LRU) : au-delà de `shingle_memory`
    phrases, les plus anciennes sont oubliées. Un index inversé
    (shingle → ids) évite le scan O(N) complet : seuls les candidats
    partageant au moins un shingle sont comparés.
    """

    def __init__(
        self,
        shingle_size: int = 4,
        jaccard_threshold: float = 0.7,
        shingle_memory: int = 2000,
    ):
        self._seen_hashes: set[str] = set()
        self._seen_shingles: deque = deque()  # ids, ordre d'arrivée (borné)
        self._entries: dict[int, frozenset] = {}  # id -> shingles
        self._shingle_index: dict[tuple, set[int]] = {}  # shingle -> ids
        self._next_id = 0
        self.shingle_size = shingle_size
        self.jaccard_threshold = jaccard_threshold
        self.shingle_memory = shingle_memory

    def _remember(self, shingles: frozenset) -> None:
        entry_id = self._next_id
        self._next_id += 1
        self._seen_shingles.append(entry_id)
        self._entries[entry_id] = shingles
        for sh in shingles:
            self._shingle_index.setdefault(sh, set()).add(entry_id)
        # Éviction LRU au-delà du cap.
        while len(self._seen_shingles) > self.shingle_memory:
            old_id = self._seen_shingles.popleft()
            old_shingles = self._entries.pop(old_id, frozenset())
            for sh in old_shingles:
                ids = self._shingle_index.get(sh)
                if ids is not None:
                    ids.discard(old_id)
                    if not ids:
                        del self._shingle_index[sh]

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
            candidates: set[int] = set()
            for sh in shingles:
                candidates.update(self._shingle_index.get(sh, ()))
            for cid in candidates:
                seen = self._entries.get(cid)
                if seen is None:
                    continue
                intersection = len(shingles & seen)
                union = len(shingles | seen)
                if union > 0 and intersection / union > self.jaccard_threshold:
                    self._seen_hashes.add(h)
                    self._remember(shingles)
                    return True
            self._remember(shingles)

        self._seen_hashes.add(h)
        return False

    def reset(self) -> None:
        self._seen_hashes.clear()
        self._seen_shingles.clear()
        self._entries.clear()
        self._shingle_index.clear()

    @property
    def seen_count(self) -> int:
        """Nombre de hash enregistrés (phrases vues, y compris near-doublons rejetés)."""
        return len(self._seen_hashes)
