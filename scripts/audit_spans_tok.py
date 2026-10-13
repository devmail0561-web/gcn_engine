# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Audit pré-training tok : reconstructibilité des char-spans gold + cohérence CIR.

Usage : python3 scripts/audit_spans_tok.py <split.json> [autres...]
Sortie 0 ssi : taux de rejet ≤ 5 % ET 0 erreur structurelle
(spans hors bornes, arête pendante, marker hors bornes).
Méthode : formes UD matchées en ordre sur le texte (pliage apostrophes
1:1 — les offsets trouvés valent dans l'original). Tout échec = rejet
explicite avec motif, jamais deviné.
"""

from __future__ import annotations

import json
import sys
from collections import Counter

SEUIL_REJET = 0.05
APOS = {
    "\u2019": "'",
    "\u2018": "'",
    "\u201b": "'",
    "\u02bc": "'",
    "\u0060": "'",
    "\u00b4": "'",
}


def plier(s: str) -> str:
    for a, b in APOS.items():
        s = s.replace(a, b)
    return s


def reconstruit(texte: str, tokens: list) -> tuple[dict | None, str]:
    fond = plier(texte)
    if len(fond) != len(texte):
        return None, "pliage-non-1:1"
    offs: dict = {}
    pos = 0
    for t in tokens:
        f = plier(t.get("form", ""))
        if not f:
            return None, "forme-vide"
        i = fond.find(f, pos)
        if i < 0:
            return None, f"introuvable:{f[:20]}"
        offs[t["id"]] = (i, i + len(f))
        pos = i + len(f)
    return offs, "ok"


def audite(fichier: str) -> dict:
    with open(fichier, encoding="utf-8") as fh:
        doc = json.load(fh)
    sents = doc["document"]["sentences"]
    rejet = Counter()
    bornes = pendantes = markers = 0
    total = len(sents)
    for s in sents:
        offs, motif = reconstruit(s["text"], s["tokens"])
        if offs is None:
            rejet[motif] += 1
            continue
        ntoks = len(s["tokens"])
        ids = {n["id"] for n in s["cir"]["nodes"]}
        for n in s["cir"]["nodes"]:
            a, b = n["token_span"]
            if not (1 <= a <= b <= ntoks):
                bornes += 1
        for e in s["cir"]["edges"]:
            srcs = e.get("sources") or ([e["source"]] if "source" in e else [])
            if any(x not in ids for x in srcs) or e["target"] not in ids:
                pendantes += 1
            m = e.get("marker_token")
            if m is not None and not (1 <= m <= ntoks):
                markers += 1
    return {
        "fichier": fichier,
        "phrases": total,
        "rejets": dict(rejet),
        "bornes": bornes,
        "pendantes": pendantes,
        "markers": markers,
    }


def main() -> int:
    fichiers = sys.argv[1:]
    if not fichiers:
        print("usage: audit_spans_tok.py <split.json> [...]", file=sys.stderr)
        return 2
    echec = False
    for f in fichiers:
        r = audite(f)
        n_rejet = sum(r["rejets"].values())
        taux = n_rejet / r["phrases"] if r["phrases"] else 1.0
        struct = r["bornes"] + r["pendantes"] + r["markers"]
        statut = "OK" if (taux <= SEUIL_REJET and struct == 0) else "STOP"
        if statut == "STOP":
            echec = True
        print(
            f"[{statut}] {r['fichier']}: {r['phrases']} phrases, "
            f"rejet {n_rejet} ({taux:.1%}) {r['rejets']}, "
            f"bornes={r['bornes']} pendantes={r['pendantes']} markers={r['markers']}"
        )
    return 1 if echec else 0


if __name__ == "__main__":
    sys.exit(main())
