# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Garde anti-RCE pour les checkpoints ``.npz`` à tableaux ``dtype=object``.

Un ``.npz`` produit par ``gcn-train`` contient les métadonnées ``_arch_json`` et
``_vocab_json`` sous forme de tableaux ``object``, ce qui impose
``np.load(..., allow_pickle=True)`` : la lecture d'un tel membre **exécute du
pickle**. Un fichier .npz piégé peut donc exécuter du code arbitraire.

La seule porte d'entrée sûre consiste à valider le pickle **avant** toute
désérialisation non contrôlée. Ce module le fait en deux temps :

1. ``assert_npz_pickle_safe`` ouvre l'archive, lit chaque membre ``.npy``, et —
   uniquement pour les membres ``dtype=object``, seuls à impliquer du pickle —
   les désérialise avec un ``Unpickler`` restreint dont ``find_class`` n'accepte
   que les classes numpy nécessaires à la reconstruction d'un tableau.
   Toute autre classe (``os.system``, ``builtins.eval``, ``__reduce__``
   arbitraire, …) lève ``UnsafeCheckpointError`` **sans avoir été exécutée**.
2. ``guarded_np_load`` n'appelle ensuite ``np.load(allow_pickle=True)`` que si
   l'étape 1 a réussi.

Les membres à données continues (poids du modèle) ne contiennent jamais de
pickle : leur lecture ne peut pas exécuter de code et n'est pas auditée.

Les points d'appel (``load_checkpoint``, ``GCNEngine.from_pretrained``,
``run_eval``) passent obligatoirement par cette garde — le contrôle est donc
réalisé même quand l'appelant a déjà posé ``trusted=True``.
"""
from __future__ import annotations

import io
import pickle
import zipfile
from pathlib import Path

import numpy as np

__all__ = ["UnsafeCheckpointError", "assert_npz_pickle_safe", "guarded_np_load"]


class UnsafeCheckpointError(ValueError):
    """Le checkpoint contient un pickle qui référence une classe non autorisée."""


# Seules ces classes peuvent apparaître dans le pickle d'un tableau numpy
# (protocoles 2-5, numpy 1.x et 2.x). Tout le reste est rejeté.
_ALLOWED_GLOBALS = frozenset(
    {
        ("numpy", "ndarray"),
        ("numpy", "matrix"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("copyreg", "_reconstructor"),
        ("collections", "OrderedDict"),
    }
)


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler qui refuse toute classe hors allowlist numpy."""

    def find_class(self, module: str, name: str):
        if (module, name) not in _ALLOWED_GLOBALS:
            raise UnsafeCheckpointError(
                f"pickle interdit dans le checkpoint : {module}.{name}"
            )
        return super().find_class(module, name)


def _read_header(fp: io.BytesIO) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = np.lib.format.read_magic(fp)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(fp)
    if version in ((2, 0), (3, 0)):
        return np.lib.format.read_array_header_2_0(fp)
    raise UnsafeCheckpointError(f"version .npy non prise en charge : {version!r}")


def assert_npz_pickle_safe(path: Path | str) -> None:
    """Audit le pickle de chaque membre ``object`` avant toute désérialisation.

    Lève ``UnsafeCheckpointError`` si un membre référence une classe hors
    allowlist, ``ValueError`` si l'archive est illisible en tant qu'``.npz``.
    """
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.endswith(".npy")]
            if not members:
                return
            for name in members:
                buf = io.BytesIO(zf.read(name))
                try:
                    _shape, _fortran, dtype = _read_header(buf)
                except (ValueError, OSError, EOFError) as exc:
                    raise UnsafeCheckpointError(
                        f"{path.name}:{name} — en-tête .npy illisible ({exc})"
                    ) from exc
                if not dtype.hasobject:
                    # Membre à données continues : pas de pickle, donc pas de
                    # code exécutable à la lecture.
                    continue
                try:
                    _RestrictedUnpickler(buf).load()
                except UnsafeCheckpointError:
                    raise
                except Exception as exc:  # pickle corrompu / inattendu
                    raise UnsafeCheckpointError(
                        f"{path.name}:{name} — pickle illisible ({exc})"
                    ) from exc
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{path.name} n'est pas une archive .npz valide ({exc})") from exc


def guarded_np_load(path: Path | str, **kwargs):
    """``np.load(allow_pickle=True)`` après audit pickle préalable du fichier."""
    assert_npz_pickle_safe(path)
    kwargs.setdefault("allow_pickle", True)
    if kwargs["allow_pickle"] is not True:
        raise ValueError("guarded_np_load attend allow_pickle=True explicite")
    return np.load(path, **kwargs)
