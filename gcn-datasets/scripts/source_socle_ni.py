# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Sourcing lot socle NI — candidats depuis g0/fr vers 4 sous-lots.

Protocole : `docs/refactory/plan-socle-ni.md` §2. Déterministe (seed 42).
Le minage PROPOSE (déclencheurs larges, frontières de mots obligatoires),
l'humain DISPOSE (double annotation aveugle + arbitrage).

Usage :
    python3 gcn-datasets/scripts/source_socle_ni.py [--out-dir DIR] [--marge 3]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "gcn-tools" / "gcn-scraper" / "src"))

from gcn_scraper.filters.causal_scorer import RELATION_KEYWORDS

SEED = 42
MARGE_DEFAUT = 3  # candidats proposés par phrase à annoter (choix humain)
QUOTAS = {
    "concept": 10,
    "evenement": 10,
    "contrainte": 10,
    "etat_local": 5,
    "condition": 5,
    "assert": 10,
    "question": 10,
    "request": 10,
    "mixte": 10,
}

DECLENCHEURS_NOEUDS = {
    "concept": [r"concept", r"notion", r"idée", r"théorie"],
    "evenement": [r"événement", r"survenue", r"accident", r"crise", r"incident"],
    "contrainte": [
        r"contrainte",
        r"limite",
        r"interdiction",
        r"obligation",
        r"restriction",
    ],
    "etat_local": [r"état", r"situation", r"position"],
    "condition": [r"condition", r"prérequis", r"exigence"],
}
RE_NOEUDS = {
    k: re.compile(r"\b(?:" + "|".join(v) + r")\b", re.IGNORECASE)
    for k, v in DECLENCHEURS_NOEUDS.items()
}

CONNECTEURS_FR = [m for rel in RELATION_KEYWORDS.get("fr", {}).values() for m in rel]
RE_CONNECTEURS = re.compile(
    "|".join(
        sorted(
            (re.escape(m) for m in CONNECTEURS_FR if len(m) >= 3), key=len, reverse=True
        )
    ),
    re.IGNORECASE,
)
RE_IMPERATIF = re.compile(
    r"^(?:[A-ZÉÈÊ][a-zéèê]+(?:ez|ons|e|ez-vous|ons-nous)\b|Il faut\b|Prière de\b)"
)
RE_EXCLUS = re.compile(r"\bse\s+limit|\blimit(er|era|ait|ant)\b", re.IGNORECASE)


def md5(t: str) -> str:
    return hashlib.md5(t.strip().encode("utf-8")).hexdigest()


def charger_interdits() -> set[str]:
    """Empreintes des textes déjà en train/val/test ou annotés (zéro fuite)."""
    empreintes: set[str] = set()
    for split in ("train", "val", "test"):
        p = (
            RACINE
            / "gcn-datasets"
            / "DATA"
            / "supervision"
            / "v4"
            / "final2"
            / split
            / f"{split}.json"
        )
        for s in json.loads(p.read_text(encoding="utf-8"))["document"]["sentences"]:
            empreintes.add(md5(s["text"]))
    rapport = json.loads(
        (
            RACINE / "gcn-datasets" / "DATA" / "supervision" / "g0" / "g0_report.json"
        ).read_text(encoding="utf-8")
    )
    _ = rapport  # deja_annote compté globalement ; la dédup texte couvre
    return empreintes


def pre_classe_intent(texte: str) -> str:
    t = texte.rstrip()
    if t.endswith("?"):
        return "question"
    if t.endswith("!") or RE_IMPERATIF.search(t):
        return "request"
    return "assert"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="gcn-datasets/DATA/supervision/socle_ni")
    ap.add_argument("--marge", type=int, default=MARGE_DEFAUT)
    args = ap.parse_args()
    rng = random.Random(SEED)

    src = RACINE / "gcn-datasets" / "DATA" / "supervision" / "g0" / "fr_a_annoter.jsonl"
    interdits = charger_interdits()
    phrases = []
    for ligne in src.read_text(encoding="utf-8").splitlines():
        if not ligne.strip():
            continue
        e = json.loads(ligne)
        t = e["text"].strip()
        if not (30 <= len(t) <= 600):
            continue
        if md5(t) in interdits:
            continue
        if RE_EXCLUS.search(t):
            continue  # formes verbales (« se limiter »), pas le nom contrainte
        if not RE_CONNECTEURS.search(t):
            continue  # quarantaine sans_connecteur (règle plan-3M §4)
        phrases.append(e)
    rng.shuffle(phrases)

    pris: set[str] = set()
    candidats: dict[str, list] = {
        k: []
        for k in [
            "concept",
            "evenement",
            "contrainte",
            "etat_local",
            "condition",
            "question",
            "request",
            "assert",
            "mixte",
        ]
    }
    for e in phrases:
        t = e["text"].strip()
        h = md5(t)
        if h in pris:
            continue
        for nt, rx in RE_NOEUDS.items():
            if rx.search(t) and len(candidats[nt]) < QUOTAS[nt] * args.marge:
                candidats[nt].append(
                    {
                        **e,
                        "sous_lot": "N-noeuds"
                        if nt in ("concept", "evenement", "contrainte")
                        else "N-renfort",
                        "declencheur": nt,
                    }
                )
                pris.add(h)
                break
        else:
            acte = pre_classe_intent(t)
            if len(candidats[acte]) < QUOTAS[acte] * args.marge:
                candidats[acte].append(
                    {**e, "sous_lot": "N-intent", "pre_classe": acte}
                )
                pris.add(h)

    # N-mixte : 2-clauses (≥2 segments forts) parmi les restantes, avec nœud cible
    deux = [
        e
        for e in phrases
        if md5(e["text"].strip()) not in pris
        and len(re.split(r"[.!?;:]", e["text"])) >= 3
        and any(rx.search(e["text"]) for rx in RE_NOEUDS.values())
    ]
    for e in deux[: QUOTAS["mixte"] * args.marge]:
        candidats["mixte"].append({**e, "sous_lot": "N-mixte"})

    out = RACINE / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": SEED,
        "quotas": QUOTAS,
        "marge": args.marge,
        "source": "g0/fr_a_annoter.jsonl",
        "filtre": "30-600c + connecteur FR",
        "dedup": "md5 vs final2 train/val/test",
        "sous_lots": {},
    }
    for k, lst in candidats.items():
        p = out / f"candidats_{k}.jsonl"
        p.write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in lst), encoding="utf-8"
        )
        manifest["sous_lots"][k] = {
            "candidats": len(lst),
            "quota": QUOTAS.get(k, QUOTAS["mixte"] if k == "mixte" else 0),
            "couvert": len(lst) >= QUOTAS.get(k, 0) * args.marge,
        }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    manquants = [k for k, v in manifest["sous_lots"].items() if not v["couvert"]]
    print(json.dumps(manifest["sous_lots"], ensure_ascii=False, indent=1))
    if manquants:
        print(
            f"SOUS-QUOTA(marge x{args.marge}) : {manquants} — élargir déclencheurs ou marge"
        )
        return 1
    print("OK — tous quotas couverts avec marge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
