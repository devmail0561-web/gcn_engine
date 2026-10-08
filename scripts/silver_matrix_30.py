#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Matrice silver 30 ex AVANT tête (R9) : règles d'actes vs gold sentence_type.

30 phrases gold_mini stratifiées (5 exclamatives + 3 interrogatives + 22
déclaratives, seed 42) : ``detect_act`` vs mapping attendu
(déclarative→assert, interrogative→question, exclamative→assert+saillant).
Moitié « intent » de R9 (l'erreur silver complète viendra des campagnes
d'annotation ; weight silver 0.0 par défaut en attendant).

Sortie : JSON {n, exacts, precision, matrice, erreurs} sur stdout.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gcn-python" / "src"))

from gcn_python.layer2.act import detect_act

SEED = 42
ATTENDU = {
    "declarative": ("assert", False),
    "interrogative": ("question", False),
    "exclamative": ("assert", True),
}


def main() -> int:
    src = REPO / "gcn-datasets" / "DATA" / "supervision" / "gold_mini" / "gold_mini.jsonl"
    phrases = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    par_type: dict[str, list[dict]] = {}
    for p in phrases:
        par_type.setdefault(p.get("sentence_type", "declarative"), []).append(p)
    rng = random.Random(SEED)
    choisis = list(par_type.get("exclamative", [])) + list(par_type.get("interrogative", []))
    declaratives = par_type.get("declarative", [])[:]
    rng.shuffle(declaratives)
    choisis += declaratives[: 30 - len(choisis)]
    matrice: dict[str, dict[str, int]] = {}
    erreurs = []
    for p in choisis:
        stype = p.get("sentence_type", "declarative")
        attendu = ATTENDU.get(stype, ("assert", False))
        got = detect_act(p["text"])
        cle_att = f"{attendu[0]}:{attendu[1]}"
        cle_got = f"{got['act']}:{got['saillant']}"
        matrice.setdefault(cle_att, {}).setdefault(cle_got, 0)
        matrice[cle_att][cle_got] += 1
        if (got["act"], got["saillant"]) != attendu:
            erreurs.append({"id": p.get("id"), "type": stype, "texte": p["text"][:80],
                            "attendu": cle_att, "obtenu": cle_got})
    exacts = sum(v.get(k, 0) for k, v in matrice.items())
    print(json.dumps({"n": len(choisis), "exacts": exacts,
                      "precision": round(exacts / len(choisis), 4) if choisis else 0.0,
                      "matrice": matrice, "erreurs": erreurs}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
