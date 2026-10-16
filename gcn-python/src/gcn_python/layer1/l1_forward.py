# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Forward L1 — ``forward(ids, mask)`` + pooling fixe (refacto L1b).

Références : ``plan-remodelage-tout-appris.md`` (étape 2, [T3] flux découplé,
[R1] pooling figé + caches ``.copy()``), ``plan-remodelage-inference.md``
§[3] (``vectorize_*`` → ``forward(ids, mask)``).

``forward_ids`` : lookup ``E_tok`` + masque (zéros hors masque), copies
défensives. ``pool_clauses`` : reps de clauses par MOYENNE fixe sur les
tokens de chaque span (jamais d'argmax différencié, jamais de routage appris :
[T3] « pooling fixe + routage explicite »). Clause vide → ligne zéro
(documenté, pas silencieux : comptée dans ``PoolMap.vides``).
``PoolMap`` = miroir explicite lignes → indices tokens ([R1] backward_ids).
Legacy cohabite (aucune suppression — bascule plus tard).
"""

from __future__ import annotations

import numpy as np

from .tok_embedding import TokEmbedding


def forward_ids(
    emb: TokEmbedding,
    ids,
    mask=None,
) -> np.ndarray:
    """``(N, d_model)`` : lignes ``E_tok`` dans l'ordre des ids, zéros hors masque.

    Copies défensives ([R1]) : le retour ne partage rien avec ``E``.
    ``mask`` : 0/1 par position (``None`` = tout visible).
    """
    lignes = emb.lookup_batch(ids).copy()
    if mask is None:
        return lignes
    drapeau = np.asarray(mask, dtype=np.float32).ravel()
    if drapeau.shape[0] != lignes.shape[0]:
        raise ValueError(
            f"mask de longueur {drapeau.shape[0]} pour {lignes.shape[0]} ids."
        )
    return (lignes * drapeau[:, None]).astype(np.float32)


class PoolMap:
    """Miroir lignes → tokens ([R1] backward_ids) : ``rows[c]`` = indices des
    tokens poolés dans la clause ``c`` (copies) ; ``vides`` = clauses sans token.
    """

    def __init__(self, rows: list[list[int]]) -> None:
        self.rows: list[list[int]] = [list(r) for r in rows]
        self.vides: int = sum(1 for r in self.rows if not r)

    def __len__(self) -> int:
        return len(self.rows)


def align_spans(
    offsets: list[tuple[int, int]], spans: list[tuple[int, int]]
) -> list[list[int]]:
    """Mappe des spans caractères (clauses B / ``token_span`` gold) vers des
    indices tokens : token retenu ssi son DÉBUT est dans ``[s, e)`` (règle
    unique, déterministe ; tokens à cheval = clause de leur début).
    Retourne des listes neuves (pas d'alias sur ``offsets``).
    """
    alignes: list[list[int]] = []
    for debut, fin in spans:
        if not (0 <= debut <= fin):
            raise ValueError(f"span invalide : {(debut, fin)}.")
        alignes.append([i for i, (a, _) in enumerate(offsets) if debut <= a < fin])
    return alignes


def pool_clauses(
    E: np.ndarray, alignes: list[list[int]]
) -> tuple[np.ndarray, PoolMap]:
    """Moyenne fixe par clause → ``(C, d_model)`` + carte miroir.

    Clause vide → ligne zéro (comptée, ``PoolMap.vides``). Aucun paramètre
    appris, aucun routage : positions explicites ([T3]).
    """
    matrice = np.asarray(E, dtype=np.float32)
    if matrice.ndim != 2:
        raise ValueError(f"E doit être (N, d), reçue {matrice.shape}.")
    miroir = PoolMap(alignes)
    n = matrice.shape[0]
    for c, membres in enumerate(miroir.rows):
        for i in membres:
            if not (0 <= i < n):
                raise ValueError(
                    f"clause {c} : indice token {i} hors E ({n} lignes)."
                )
    reps = np.zeros((len(miroir), matrice.shape[1]), dtype=np.float32)
    for c, membres in enumerate(miroir.rows):
        if membres:
            reps[c] = matrice[membres].mean(axis=0)
    return reps, miroir
