"""Diversité du scraping — évite de re-scraper les mêmes URLs/queries à chaque run.

Constat : toutes les sources itéraient des listes fixes dans un ordre fixe,
avec pagination démarrant à 0. Deux runs produisaient donc les mêmes données
(sauf contenu frais des flux RSS).

Fix : ordre mélangé de façon déterministe par `seed` + rotation de l'offset
de pagination. Même seed → reproductible. Seeds différents (ou seed=None →
tirage horodaté) → sous-ensembles différents quand le budget coupe tôt.
"""
from __future__ import annotations

import random


def make_rng(seed: int | None) -> tuple[random.Random, int]:
    """Retourne (rng, seed_effective). seed=None → tirage horodaté."""
    if seed is None:
        seed = random.SystemRandom().randint(0, 2**31 - 1)
    return random.Random(int(seed)), int(seed)


def shuffled(items: list, seed: int | None, salt: str = "") -> list:
    """Copie mélangée de façon déterministe (seed + salt)."""
    rng = random.Random(hash((int(seed or 0), salt)) & 0xFFFFFFFF)
    out = list(items)
    rng.shuffle(out)
    return out


def page_offset(seed: int | None, salt: str, per_page: int, depth: int = 4) -> int:
    """Offset de pagination rotatif : 0, per_page, ..., (depth-1)*per_page."""
    if not seed:
        return 0
    rng = random.Random(hash((int(seed), salt)) & 0xFFFFFFFF)
    return rng.randint(0, max(0, depth - 1)) * max(1, per_page)
