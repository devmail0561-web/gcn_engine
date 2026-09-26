# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Vecteurs de graphe séparés graph_vecs.npz + manifest (Étape 3 plan v5).

- Clé stable : sha256(sentence_text utf-8) + "_" + position nœud dans cir["nodes"].
- Manifest JSON : checkpoint_hash (poids triés), d_eff, created_at, entries.
- Hash canonique : sha256(concat(poids triés)) — stable même si ZIP varie.
- NpzFile FIFO (oldest-first eviction) : cache de handles np.load(..., allow_pickle=False, mmap_mode='r').
- Fallback _minimal_reps : qualité dégradée documentée.
"""
from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

import threading as _threading
_HANDLES: dict[str, object] = {}
_HANDLES_LOCK = _threading.Lock()
_MAX_HANDLES = 8


def stable_key(sentence_text: str, node_pos: int) -> str:
    h = hashlib.sha256(sentence_text.encode("utf-8")).hexdigest()
    return f"{h}_{int(node_pos)}"


def checkpoint_hash(arrays: dict) -> str:
    parts = []
    for k in sorted(arrays.keys()):
        # inclure quand même les poids, exclure les JSON d'arch/vocab
        if k.startswith(("_", "link_pred", "hyperedge")) and (
            k.endswith("_json") or k.startswith("_")
        ):
            continue
        try:
            parts.append(np.ascontiguousarray(arrays[k]).tobytes())
        except Exception:  # noqa: S112, BLE001  # poids non sérialisables : on saute cet élément
            continue
    return "sha256:" + hashlib.sha256(b"".join(parts)).hexdigest()


def save_graph_vecs(path: Path, vecs: dict[str, np.ndarray],
                     checkpoint_hash_str: str, d_eff: int,
                     meta: dict[str, dict] | None = None) -> Path:
    path = Path(path)
    tmp = path.with_suffix(".tmp.npz")
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in vecs.items()}
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    manifest = {
        "checkpoint_hash": checkpoint_hash_str,
        "d_eff": int(d_eff),
        "created_at": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(),
        "entries": {
            k: {"shape": list(np.asarray(v).shape),
                **(meta.get(k, {}) if meta else {})}
            for k, v in vecs.items()
        },
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _get_handle(path: Path):
    key = str(path)
    with _HANDLES_LOCK:
        if key in _HANDLES:
            return _HANDLES[key]
        handle = np.load(path, allow_pickle=False, mmap_mode="r")
        if len(_HANDLES) >= _MAX_HANDLES:
            oldest = next(iter(_HANDLES))
            with contextlib.suppress(Exception):
                _HANDLES[oldest].close()
            del _HANDLES[oldest]
        _HANDLES[key] = handle
        return handle


def load_graph_vec(path: Path, key: str,
                   expected_checkpoint_hash: str | None = None) -> np.ndarray | None:
    """Charge un vecteur par clé stable. Retourne None si absent/stale (warn)."""
    path = Path(path)
    manifest_p = path.with_suffix(".manifest.json")
    if manifest_p.exists() and expected_checkpoint_hash:
        try:
            manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
            if manifest.get("checkpoint_hash") != expected_checkpoint_hash:
                warnings.warn(
                    f"graph_vecs : checkpoint changé (manifest {manifest.get('checkpoint_hash')} "
                    f"≠ attendu) — fallback.",
                    UserWarning, stacklevel=2,
                )
                return None
        except Exception:  # noqa: S110, BLE001  # manifest corrompu : ignoré, fallback dégradé
            pass
    elif not manifest_p.exists():
        warnings.warn("graph_vecs : manifest absent — qualité dégradée (fallback).",
                      UserWarning, stacklevel=2)
    try:
        handle = _get_handle(path)
        if key not in handle.files:
            return None
        return np.array(handle[key])
    except Exception as exc:  # noqa: BLE001  # npz illisible/corrompu : warn + fallback None
        warnings.warn(f"graph_vecs : lecture impossible ({exc}) — fallback.",
                      UserWarning, stacklevel=2)
        return None
