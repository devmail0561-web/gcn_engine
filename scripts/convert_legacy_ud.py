#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Convertisseur legacy-UD → refacto, one-shot (R4, étape 4).

Legacy = tout .npz SANS ``format_refacto`` (E lemmes, dims 106/234/247/1115,
têtes 22) ; refacto = seul format écrit (``format_refacto=20261007``,
``tok_E``/vocab trié/hash, ``d_model``, ``act_types:[3]`` — s0..s3).

Conversion = table E (lexique) : chaque lemme legacy → id stable calculé
EXACTEMENT comme le runtime (``token_family`` + ``normalize_token`` +
``stable_id``), lignes réordonnées en vocab trié. Têtes MLP/RGCN legacy
(dims incompatibles) : NON converties → ré-entraîner (rapporté).
Spéciaux ``_unk/_subj_absent/_obj_absent`` : abandonnés (OOV → zéros,
Option A ; aucun appelant explicite ``lookup("_unk")`` dans le code).
Collision (2 lemmes → même id) : premier en ordre trié gagne, comptée.

- ``--dry-run`` : calcule tout, N'ÉCRIT RIEN (mtime/hash d'entrée vérifiés
  inchangés, sortie non créée) ; rapport JSON sur stdout ; sortie 0.
- réel : écrit le .npz refacto (via ``TokEmbedding.save_npz``) ; rapport
  avec sha256 entrée/sortie. Déterminisme : 2 runs → même sha256
  (« sortie==gold hash ») ; équivalence : ``lookup`` legacy == ``lookup``
  converti par lemme (vérifiée en test, pas ici).

Usage :
    python3 scripts/convert_legacy_ud.py ENTREE.npz --output SORTIE.npz [--d-model N] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gcn-python" / "src"))

from gcn_python.layer0.tokenizer import (
    normalize_token,
    stable_id,
    token_family,
)
from gcn_python.layer1.tok_embedding import TokEmbedding
from gcn_python.security import guarded_np_load
from gcn_python.training.checkpoint import (
    LegacyUDError,
    checkpoint_format_kind,
)

#: Lemmes spéciaux legacy abandonnés (OOV → zéros, Option A).
SPECIALES_ABANDONNEES = frozenset({"_unk", "_subj_absent", "_obj_absent"})


def sha256_fichier(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def etat_fichier(path: Path) -> tuple[float, str]:
    st = os.stat(path)
    return (st.st_mtime_ns, sha256_fichier(path))


def charger_legacy(path: Path) -> tuple[list[str], object]:
    """E + lemmes d'un legacy-UD (garde pickle ; refuse le refacto)."""
    if checkpoint_format_kind(path) != "legacy-ud":
        raise LegacyUDError(
            f"{path.name} : pas un legacy-UD (déjà refacto ou inconnu) — rien à convertir."
        )
    data = guarded_np_load(path)
    brut = json.loads(str(data["_word_emb_vocab_json"][0]))
    lemmes = brut if isinstance(brut, list) else brut.get("lemmas", [])
    return [str(l) for l in lemmes], data["word_emb_E"]


def convertir(
    lemmes: list[str], E: object, d_model: int | None = None
) -> tuple[TokEmbedding, dict]:
    """Construit l'E_tok refacto + rapport (aucune écriture)."""
    import numpy as np

    matrice = np.asarray(E, dtype=np.float32)
    if matrice.ndim != 2:
        raise ValueError(f"E legacy de forme {matrice.shape}, attendue (V, d).")
    d_emb = matrice.shape[1]
    if d_model is None:
        d_model = d_emb
    if d_model != d_emb:
        raise ValueError(
            f"d_model={d_model} ≠ d_emb legacy={d_emb} : têtes incompatibles — "
            "ré-entraîner (la conversion ne porte que la table E)."
        )
    if len(lemmes) != matrice.shape[0]:
        raise ValueError(
            f"{len(lemmes)} lemmes pour {matrice.shape[0]} lignes E — incohérent."
        )
    # Ordre trié = canonique (premier gagne en cas de collision).
    attribution: dict[int, tuple[str, int]] = {}
    collisions = 0
    abandonnes = 0
    for rang, lemme in sorted(enumerate(lemmes), key=lambda p: p[1]):
        if lemme in SPECIALES_ABANDONNEES:
            abandonnes += 1
            continue
        tok_id = stable_id(normalize_token(lemme, token_family(lemme)))
        if tok_id in attribution:
            collisions += 1
            continue
        attribution[tok_id] = (lemme, rang)
    emb = TokEmbedding(d_model=d_model, seed=42)
    ids_tries = sorted(attribution)
    emb.build_vocab(ids_tries)
    for pos, tok_id in enumerate(ids_tries):
        _, rang = attribution[tok_id]
        emb._E[pos] = matrice[rang]
    rapport = {
        "n_lemmes": len(lemmes),
        "n_mappees": len(ids_tries),
        "n_collisions": collisions,
        "n_abandonnes": abandonnes,
        "d_model": d_model,
        "note_tetes": "MLP/RGCN legacy (dims 106/234/247/1115, 22 intents) NON convertis : ré-entraîner.",
    }
    return emb, rapport


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Convertit un checkpoint legacy-UD en format refacto (one-shot).")
    ap.add_argument("entree", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--d-model", type=int, default=None,
                    help="d_model exigé (défaut : d_emb du fichier)")
    ap.add_argument("--dry-run", action="store_true",
                    help="calcule sans écrire (mtime/hash d'entrée inchangés)")
    args = ap.parse_args(argv)

    avant = etat_fichier(args.entree)
    lemmes, E = charger_legacy(args.entree)
    emb, rapport = convertir(lemmes, E, args.d_model)
    rapport.update({
        "entree": str(args.entree),
        "sha256_entree": avant[1],
        "dry_run": args.dry_run,
    })
    if args.dry_run:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            temoin = str(Path(tmp) / "temoin.npz")
            emb.save_npz(temoin)
            rapport["sha256_sortie_simulee"] = sha256_fichier(Path(temoin))
        apres = etat_fichier(args.entree)
        if apres != avant:
            print(json.dumps({"erreur": "dry-run a modifié l'entrée !",
                              "avant": avant, "apres": apres}))
            return 1
        if args.output.exists():
            print(json.dumps({"erreur": "dry-run : la sortie existe déjà ?! "
                              "(rien n'aurait dû être écrit)"}))
            return 1
        rapport["entree_inchangee"] = True
        print(json.dumps(rapport, ensure_ascii=False, indent=1))
        return 0
    emb.save_npz(str(args.output))
    rapport.update({
        "sortie": str(args.output),
        "sha256_sortie": sha256_fichier(args.output),
    })
    print(json.dumps(rapport, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
