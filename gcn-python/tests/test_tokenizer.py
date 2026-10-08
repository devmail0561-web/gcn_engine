# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests du tokenizer L0 — split fixe + ids stables ([T6], [R5], [R1]).

Exécution : ``PYTHONPATH=gcn-python/src pytest gcn-python/tests/test_tokenizer.py -q``
"""

from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path

from gcn_python.layer0.tokenizer import (
    RULES_VERSION,
    normalize_token,
    split,
    stable_id,
    token_family,
    tokenizer_hash,
)

RACINE_DEPOT = Path(__file__).resolve().parents[2]
TEXTE_FR = "Café ÉTÉ : l'homme reste, vraiment ?"


def test_split_fr_forme_et_offsets():
    ids, offsets, tokens = split(TEXTE_FR)
    assert len(ids) == len(offsets) == len(tokens) > 0
    assert tokens[0] == "cafe" and tokens[1] == "ete"  # minuscule + sans-accent
    assert "?" in tokens  # ponctuation gardée (actes)
    assert any(t == "l'homme" for t in tokens)  # apostrophe interne
    for debut, fin in offsets:  # offsets valides dans l'original
        assert 0 <= debut < fin <= len(TEXTE_FR)
        assert TEXTE_FR[debut:fin].strip()
    plats = [o for span in offsets for o in span]
    assert plats == sorted(plats)  # ordonnés
    for (a, _), (c, _) in itertools.pairwise(offsets):
        assert a < c  # pas de recouvrement au départ


def test_determinisme_meme_processus():
    assert split(TEXTE_FR) == split(TEXTE_FR)


def test_ids_stables_sans_vocab_et_sans_hash_random():
    # Id = sha256 (pas hash()) : identique sous PYTHONHASHSEED 0 et 1.
    code = (
        "import sys; sys.path.insert(0, 'gcn-python/src');"
        "from gcn_python.layer0.tokenizer import split;"
        f"print(split({TEXTE_FR!r})[0])"
    )
    sorties = set()
    for seed in ("0", "1"):
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
            cwd=RACINE_DEPOT,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        )
        assert proc.returncode == 0, proc.stderr
        sorties.add(proc.stdout)
    assert len(sorties) == 1, "ids instables entre seeds"
    assert all(isinstance(i, int) and 0 <= i < 2**64 for i in split(TEXTE_FR)[0])


def test_cjk_caractere_par_caractere():
    ids, offsets, tokens = split("日本語")
    assert tokens == ["日本語"[0], "日本語"[1], "日本語"[2]]
    assert offsets == [(0, 1), (1, 2), (2, 3)]
    assert len(set(ids)) == 3


def test_arabe_rtl_mots_et_ponctuation():
    ids, offsets, tokens = split("هل تأتي؟")
    assert tokens[-1] == "؟"  # ponctuation arabe séparée
    assert len(ids) == len(offsets) == len(tokens) == 3
    assert token_family("تأتي") == "arabic"
    assert normalize_token("تأتي", "arabic") == "تأتي"  # NFC, diacritiques gardés


def test_mixte_latin_cjk():
    _, _, tokens = split("Hello世界!")
    assert tokens == ["hello", "世", "界", "!"]


def test_vide_et_blancs():
    assert split("") == ([], [], [])
    assert split("   \t\n") == ([], [], [])


def test_hash_regles_vocab():
    h1 = tokenizer_hash(RULES_VERSION, ["b", "a"])
    h2 = tokenizer_hash(RULES_VERSION, ["a", "b"])
    assert h1 == h2 and len(h1) == 64  # ordre vocab indifférent
    assert tokenizer_hash(RULES_VERSION) != tokenizer_hash(RULES_VERSION, ["x"])
    assert tokenizer_hash("autre-version", ["a"]) != h1  # règles versionnées


def test_ids_distincts_par_norme():
    assert stable_id("cafe") != stable_id("café")
    assert stable_id("cafe") == stable_id("cafe")


def test_pas_de_tiers_ni_random():
    # stdlib seule + jamais de hash() builtin (randomisé) : ids stables.
    import ast

    src = RACINE_DEPOT / "gcn-python/src/gcn_python/layer0/tokenizer.py"
    arbre = ast.parse(src.read_text(encoding="utf-8"))
    autorises = {"hashlib", "unicodedata", "collections", "__future__"}
    for n in ast.walk(arbre):
        if isinstance(n, ast.Import):
            assert all(a.name.split(".")[0] in autorises for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            assert (n.module or "").split(".")[0] in autorises
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            assert n.func.id != "hash", "hash() builtin interdit (PYTHONHASHSEED)"
