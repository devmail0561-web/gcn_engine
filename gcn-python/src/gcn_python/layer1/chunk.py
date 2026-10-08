# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Chunks + manifest — ``stable_key → chunk_hash`` (étape 5, [C]).

Références : ``plan-remodelage-inference.md`` §[C] (``stable_key→chunk_hash``,
manifest ``{tokenizer_hash, d_model}``, ``assert vec_dim==manifest``).

- ``chunk_hash`` : sha256 du ``tokenizer_hash`` + ids du chunk (ordre signifiant)
  + bornes — identifie un chunk de façon reproductible (dédup, cache, traçabilité).
- Manifest : ``{"tokenizer_hash", "d_model"}`` EXACTEMENT ([C]) — pas de version
  (``format_refacto`` vit dans les payloads s0..s3, pas dans le manifest).
- ``assert_compatible`` : ``vec_dim == manifest`` ([C]) + hash — ``ValueError``
  sinon (fail-fast, [R5] raise).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


def chunk_hash(tokenizer_hash_str: str, ids, debut: int = 0, fin: int | None = None) -> str:
    """Empreinte d'un chunk : ``sha256(tokenizer_hash + ids + bornes)`` (hex).

    ``ids`` : ids stables du chunk (ordre signifiant) ; ``debut``/``fin`` :
    bornes caractères optionnelles (désambiguïsation de position).
    """
    if not isinstance(tokenizer_hash_str, str) or not tokenizer_hash_str:
        raise ValueError("tokenizer_hash requis (chaîne non vide).")
    arr = np.asarray(ids, dtype=np.uint64).ravel()
    h = hashlib.sha256()
    h.update(tokenizer_hash_str.encode("utf-8"))
    h.update(b"\n")
    h.update(arr.tobytes())
    h.update(b"\n")
    fin = arr.shape[0] if fin is None else fin
    h.update(f"{int(debut)}:{int(fin)}".encode())
    return h.hexdigest()


def write_manifest(path: str, tokenizer_hash_str: str, d_model: int) -> str:
    """Écrit le manifest ``{tokenizer_hash, d_model}`` ([C]). Retourne ``path``."""
    if not isinstance(tokenizer_hash_str, str) or not tokenizer_hash_str:
        raise ValueError("tokenizer_hash requis (chaîne non vide).")
    if not isinstance(d_model, int) or d_model < 1:
        raise ValueError(f"d_model invalide : {d_model!r}.")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"tokenizer_hash": tokenizer_hash_str, "d_model": d_model}, f,
                  ensure_ascii=False)
    return path


def read_manifest(path: str) -> dict:
    """Lit un manifest (clés ``tokenizer_hash`` + ``d_model`` exigées)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("tokenizer_hash"):
        raise ValueError(f"manifest sans tokenizer_hash : {path}.")
    if not isinstance(data.get("d_model"), int) or data["d_model"] < 1:
        raise ValueError(f"manifest sans d_model valide : {path}.")
    return {"tokenizer_hash": data["tokenizer_hash"], "d_model": data["d_model"]}


def assert_compatible(manifest: dict, d_model: int, tokenizer_hash_str: str) -> None:
    """``vec_dim == manifest`` ([C]) + hash ([R5]) : ``ValueError`` sinon."""
    if manifest.get("d_model") != d_model:
        raise ValueError(
            f"vec_dim={d_model} ≠ manifest d_model={manifest.get('d_model')}."
        )
    if manifest.get("tokenizer_hash") != tokenizer_hash_str:
        raise ValueError("tokenizer_hash ≠ manifest — re-geler le vocab ([R5]).")


__all__ = [
    "assert_compatible",
    "chunk_hash",
    "read_manifest",
    "write_manifest",
]
