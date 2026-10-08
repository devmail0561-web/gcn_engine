# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""E_tok appris — embeddings de tokens stables (refacto L1a).

Références : ``plan-remodelage-tout-appris.md`` (étape 2 : L1 + forward_ids +
d_model), ``plan-remodelage-inference.md`` §[6] (``WordEmbedding`` lemmes →
``Embedding(vocab_tok, d_model)`` + R5), [R4] (amendements s1/s2), [T6]
(différences apprises dans ``E_tok``, pas codées).

Vocab = ids stables TRIÉS (déterminisme d'ordre, insertion indépendante) ;
OOV → vecteur zéro (Option A, comme ``WordEmbedding.lookup`` : ``_unk``
reste propre, pas d'absorption du bruit). Taux OOV exposé pour la garde
R5 (raise train/val si > 5 %, backoff ``_unk`` prod — câblée au loader).

Format s1/s2 ([R4]) : ``{"format_refacto": 20261007, "tok_vocab": [...],
"tokenizer_hash": ..., "rules": ..., "d_model": ..., "E": [[...]]}``.
``from_dict`` vérifie ``d_model`` + hash (échec = ``ValueError``, fail-fast).
Legacy cohabite (aucune suppression — bascule plus tard).
"""

from __future__ import annotations

import json

import numpy as np

from ..layer0.tokenizer import RULES_VERSION, tokenizer_hash

#: Marqueur de format refacto ([R4] : s0 de base, s1/s2 apportent vocab + d_model).
FORMAT_REFACTO = 20261007


class TokEmbedding:
    """Table d'embeddings apprenable sur vocabulaire de tokens stables.

    ``build_vocab`` accepte des ids dans n'importe quel ordre (triés en
    interne) ; ``lookup``/``lookup_batch`` retournent des COPIES (R1 : caches
    ``.copy()``) ; OOV → zéros + compteur (garde R5 au loader).
    """

    def __init__(self, d_model: int = 128, seed: int = 42) -> None:
        if d_model < 1:
            raise ValueError(f"d_model doit être ≥ 1 (reçu {d_model!r}).")
        self.d_model = d_model
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._vocab: list[int] = []
        self._index: dict[int, int] = {}
        self._E = np.zeros((0, d_model), dtype=np.float32)
        self._grad_accum: np.ndarray | None = None
        self._oov_count = 0
        self._lookup_count = 0

    # ── vocabulaire ────────────────────────────────────────────────

    def build_vocab(self, ids) -> None:
        """Ajoute des ids stables (ordre d'insertion indifférent : triés)."""
        nouveaux = sorted(set(ids) - set(self._index))
        if not nouveaux:
            return
        base = len(self._vocab)
        self._vocab.extend(nouveaux)
        for rang, tok_id in enumerate(nouveaux, start=base):
            self._index[tok_id] = rang
        scale = np.sqrt(1.0 / self.d_model)
        lignes = self._rng.normal(0, scale, (len(nouveaux), self.d_model))
        self._E = np.vstack([self._E, lignes.astype(np.float32)])

    @property
    def vocab_size(self) -> int:
        """Nombre de lignes de ``E`` (= ``len(tok_vocab)``)."""
        return len(self._vocab)

    @property
    def tok_vocab(self) -> list[int]:
        """Vocabulaire trié (copie)."""
        return list(self._vocab)

    @property
    def oov_rate(self) -> float:
        """Taux OOV depuis le dernier ``reset_oov()`` (garde R5)."""
        if self._lookup_count == 0:
            return 0.0
        return self._oov_count / self._lookup_count

    def reset_oov(self) -> None:
        """Remet à zéro le compteur OOV."""
        self._oov_count = 0
        self._lookup_count = 0

    # ── forward / backward ─────────────────────────────────────────

    def row_index(self, tok_id: int) -> int | None:
        """Indice de ligne ou ``None`` si OOV."""
        return self._index.get(tok_id)

    def lookup(self, tok_id: int) -> np.ndarray:
        """Copie de la ligne (zéros si OOV, compteur incrémenté)."""
        self._lookup_count += 1
        idx = self._index.get(tok_id)
        if idx is None:
            self._oov_count += 1
            return np.zeros(self.d_model, dtype=np.float32)
        return self._E[idx].copy()

    def lookup_batch(self, ids) -> np.ndarray:
        """``(N, d_model)`` : lignes dans l'ordre des ids (OOV → zéros)."""
        arr = np.asarray(ids, dtype=np.uint64).ravel()
        out = np.zeros((arr.shape[0], self.d_model), dtype=np.float32)
        self._lookup_count += arr.shape[0]
        for pos, tok_id in enumerate(arr.tolist()):
            idx = self._index.get(tok_id)
            if idx is None:
                self._oov_count += 1
            else:
                out[pos] = self._E[idx]
        return out

    def backward(self, d_out: np.ndarray, ids) -> None:
        """Accumule le gradient (no-op sur OOV — ``_unk`` reste propre)."""
        arr = np.asarray(ids, dtype=np.uint64).ravel()
        grads = np.asarray(d_out, dtype=np.float32).reshape(arr.shape[0], self.d_model)
        if self._grad_accum is None:
            self._grad_accum = np.zeros_like(self._E)
        for pos, tok_id in enumerate(arr.tolist()):
            idx = self._index.get(tok_id)
            if idx is not None:
                self._grad_accum[idx] += grads[pos]

    def update(self, lr: float) -> None:
        """Étape SGD ; vide les gradients (ignore les non-finis, vecteurs préservés)."""
        if self._grad_accum is None:
            return
        if not np.isfinite(self._grad_accum).all():
            import warnings as _w_nan

            _w_nan.warn(
                "TokEmbedding.update : gradients non finis — mise à jour ignorée.",
                UserWarning,
                stacklevel=2,
            )
            self._grad_accum = None
            return
        self._E -= lr * self._grad_accum
        self._grad_accum = None

    def parameters(self) -> list[np.ndarray]:
        return [self._E]

    # ── persistance s1/s2 ──────────────────────────────────────────

    def to_dict(self) -> dict:
        """Dictionnaire s1/s2 : vocab trié + hash + d_model + E."""
        return {
            "format_refacto": FORMAT_REFACTO,
            "tok_vocab": list(self._vocab),
            "tokenizer_hash": tokenizer_hash(RULES_VERSION, [str(v) for v in self._vocab]),
            "rules": RULES_VERSION,
            "d_model": self.d_model,
            "E": self._E.tolist(),
        }

    def to_json(self) -> str:
        """Sérialise s1/s2 (poids inclus)."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> TokEmbedding:
        """Reconstruit depuis s1/s2 ; vérifie format + d_model + hash (fail-fast)."""
        if data.get("format_refacto") != FORMAT_REFACTO:
            raise ValueError(
                f"format_refacto={data.get('format_refacto')} ≠ {FORMAT_REFACTO} "
                "(legacy-UD : convertir ou ré-entraîner)."
            )
        d_model = data.get("d_model")
        if not isinstance(d_model, int) or d_model < 1:
            raise ValueError(f"d_model invalide : {d_model!r}.")
        vocab = data.get("tok_vocab", [])
        attendu = tokenizer_hash(
            data.get("rules", RULES_VERSION), [str(v) for v in vocab]
        )
        if data.get("tokenizer_hash") != attendu:
            raise ValueError(
                "tokenizer_hash incohérent (vocab ou règles modifiés) — "
                "re-geler le vocab avant de charger."
            )
        obj = cls(d_model=d_model)
        obj.build_vocab(vocab)
        lignes = np.asarray(data.get("E", []), dtype=np.float32)
        if lignes.shape != (len(vocab), d_model):
            raise ValueError(
                f"E de forme {lignes.shape}, attendue ({len(vocab)}, {d_model})."
            )
        obj._E = lignes
        return obj

    @classmethod
    def from_json(cls, s: str) -> TokEmbedding:
        """Recrée depuis ``to_json()`` (mêmes vérifications)."""
        return cls.from_dict(json.loads(s))

    def save_npz(self, path: str) -> str:
        """Sauvegarde `.npz` s1/s2 (``E``, ``vocab``, ``hash``, ``d_model``)."""
        np.savez(
            path,
            E=self._E,
            tok_vocab=np.asarray(self._vocab, dtype=np.uint64),
            tokenizer_hash=np.asarray(
                tokenizer_hash(RULES_VERSION, [str(v) for v in self._vocab])
            ),
            d_model=np.asarray(self.d_model, dtype=np.int64),
            format_refacto=np.asarray(FORMAT_REFACTO, dtype=np.int64),
        )
        return path

    @classmethod
    def load_npz(cls, path: str) -> TokEmbedding:
        """Charge un `.npz` s1/s2 (vérifie format + hash, fail-fast)."""
        with np.load(path, allow_pickle=False) as z:
            if int(z["format_refacto"]) != FORMAT_REFACTO:
                raise ValueError("npz legacy-UD : convertir ou ré-entraîner.")
            d_model = int(z["d_model"])
            vocab = [int(v) for v in z["tok_vocab"].tolist()]
            if str(z["tokenizer_hash"]) != tokenizer_hash(
                RULES_VERSION, [str(v) for v in vocab]
            ):
                raise ValueError("tokenizer_hash incohérent — re-geler le vocab.")
            obj = cls(d_model=d_model)
            obj.build_vocab(vocab)
            lignes = np.asarray(z["E"], dtype=np.float32)
            if lignes.shape != (len(vocab), d_model):
                raise ValueError(f"E de forme {lignes.shape}, incohérente avec le vocab.")
            obj._E = lignes
            return obj


__all__ = [
    "FORMAT_REFACTO",
    "TokEmbedding",
]
