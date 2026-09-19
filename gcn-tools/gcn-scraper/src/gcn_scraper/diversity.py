"""Diversité du scraping — évite de re-scraper les mêmes URLs/queries à chaque run.

Constat : toutes les sources itéraient des listes fixes dans un ordre fixe,
avec pagination démarrant à 0. Deux runs produisaient donc les mêmes données
(sauf contenu frais des flux RSS).

Fix : ordre mélangé de façon déterministe par `seed` + rotation de l'offset
de pagination. Même seed → reproductible (y compris entre processus :
le hachage est stable, pas le hash() Python salé par PYTHONHASHSEED).
Seeds différents (ou seed=None → tirage aléatoire OS) → sous-ensembles
différents quand le budget coupe tôt. seed=0 est une graine valide.
"""
from __future__ import annotations

import hashlib
import random


def _stable_int(seed: int, salt: str) -> int:
    """Entier déterministe inter-processus (hash() Python est salé par run)."""
    digest = hashlib.sha256(f"{int(seed)}:{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def make_rng(seed: int | None) -> tuple[random.Random, int]:
    """Retourne (rng, seed_effective). seed=None → tirage aléatoire OS."""
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**31 - 1)
    return random.Random(int(seed)), int(seed)


def shuffled(items: list, seed: int | None, salt: str = "") -> list:
    """Copie mélangée de façon déterministe (seed + salt). seed=None → ordre fixe."""
    rng = random.Random(_stable_int(int(seed) if seed is not None else 0, salt))
    out = list(items)
    rng.shuffle(out)
    return out


def page_offset(seed: int | None, salt: str, per_page: int, depth: int = 4) -> int:
    """Offset de pagination rotatif : 0, per_page, ..., (depth-1)*per_page."""
    if seed is None:
        return 0
    rng = random.Random(_stable_int(int(seed), salt))
    return rng.randint(0, max(0, depth - 1)) * max(1, per_page)
