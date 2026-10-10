# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Compteur global de mutations de paramètres (invalidation cache M1a).

Portée : paramètres du chemin forward déterminant le CIR (encodeur,
embeddings, R-GCN/GAT), `CGNPipeline.backward/apply_accumulated_gradients`,
`load_checkpoint` et chargements d'embeddings — tous appellent `bump()`.
Hors portée (post-forward, jamais le CIR) : decoder, link_predictor —
leurs update() ne bumpent volontairement pas (bump inutile : tag inchangé).
Le tag modèle du cache analyze est recalculé (hash plein)
uniquement quand l'epoch a bougé — sinon réutilisé en O(1).

Zéro import : aucun risque de cycle (feuilles bas niveau autorisées).
Convention obligatoire : toute future méthode mutant des paramètres du
forward DOIT appeler `bump()`, sinon le cache analyze devient faux.
Mutations invisibles (optimizer PyTorch externe) : appeler clear_cache().
"""
from __future__ import annotations

_EPOCH = 0


def current() -> int:
    """Epoch actuel (ne touche à rien)."""
    return _EPOCH


def bump() -> int:
    """Signale une mutation de paramètres. Retourne le nouvel epoch."""
    global _EPOCH
    _EPOCH += 1
    return _EPOCH
