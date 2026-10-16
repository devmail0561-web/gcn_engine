# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Assemble le lot socle NI — passe aveugle assistant.

Protocole : `docs/refactory/plan-socle-ni.md` §3. Annotations compactes
fournies par l'annotateur (fichiers /tmp/ann*.json en session, recopiés ici
en dur via SELECTION) + textes sources (g0 + elicitées taggées).
Déterministe : ordre et ids ni001-ni080 figés.

Usage : python3 gcn-datasets/scripts/build_socle_ni.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]

TYPES_NOEUDS = {
    "processus",
    "etat_local",
    "etat_global",
    "entite",
    "condition",
    "concept",
    "evenement",
    "contrainte",
}
RELATIONS = {
    "cause",
    "enable",
    "prevent",
    "condition",
    "concession",
    "sequence",
    "motivation",
    "filter",
    "opposition",
    "data_dependency",
    "control_dependency",
    "conditional_cause",
    "mediated_cause",
    "joint_cause",
    "conditional_prevent",
    "mediated_prevent",
    "joint_prevent",
    "analogy",
    "counterfactual",
}
ACTES = {"assert", "question", "request"}

# clé -> (fichier_candidats, index, annotation_compacte). Index = ordre fichier.
# Échanges documentés : q8/q9 (fragments citation -> vraies questions),
# a3/a4 (fragments multilingues -> asserts FR), requêtes minées (0 authentique,
# regex impératif corrigée) -> 10 elicitées taggées (QUARANTAINE motivée).
SELECTION: dict[str, tuple[str, int]] = {
    **{f"concept{i}": ("concept", i) for i in range(10)},
    **{f"evenement{i}": ("evenement", i) for i in range(10)},
    **{f"contrainte{i}": ("contrainte", i) for i in range(7)},
    "contrainte7": ("contrainte", 11),  # échange : idx 7 = énumération sans
    # contrainte réelle (2 concepts, 0 contrainte) ; idx 11 = obligation nette
    **{f"contrainte{i}": ("contrainte", i) for i in range(8, 10)},
    **{f"etat_local{i}": ("etat_local", i) for i in range(5)},
    **{f"condition{i}": ("condition", i) for i in range(5)},
    **{f"question{i}": ("question", i) for i in range(8)},
    "question8": ("question", 10),
    "question9": ("question", 11),
    **{f"assert{i}": ("assert", i) for i in range(3)},
    "assert3": ("assert", 11),
    "assert4": ("assert", 14),
    **{f"assert{i}": ("assert", i) for i in range(5, 10)},
    **{f"mixte{i}": ("mixte", i) for i in range(10)},
}


def tokens_de(texte: str) -> list[dict]:
    return [
        {
            "id": i + 1,
            "form": w,
            "lemma": w.lower(),
            "pos": "other",
            "dep_rel": "other",
            "dep_head": -1,
            "morph": {},
        }
        for i, w in enumerate(texte.split(" "))
    ]


def main() -> int:
    ann: dict = {}
    for f in ("/tmp/ann1.json", "/tmp/ann2.json", "/tmp/ann3.json", "/tmp/ann4.json"):
        ann.update(json.loads(Path(f).read_text(encoding="utf-8")))
    socle = RACINE / "gcn-datasets" / "DATA" / "supervision" / "socle_ni"
    cand = {
        k: [
            json.loads(l)
            for l in (socle / f"candidats_{k}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if l.strip()
        ]
        for k in [
            "concept",
            "evenement",
            "contrainte",
            "etat_local",
            "condition",
            "question",
            "assert",
            "mixte",
        ]
    }

    phrases = []
    for n, cle in enumerate(
        sorted(
            SELECTION,
            key=lambda c: (
                [
                    "concept",
                    "evenement",
                    "contrainte",
                    "etat_local",
                    "condition",
                    "question",
                    "request",
                    "assert",
                    "mixte",
                ].index(c.rstrip("0123456789")),
                int(c[len(c.rstrip("0123456789")) :]),
            ),
        ),
        start=1,
    ):
        fichier, idx = SELECTION[cle]
        e = cand[fichier][idx]
        texte = e["text"].strip()
        a = ann[cle]
        toks = tokens_de(texte)
        assert a["intent"] in ACTES, cle
        nodes = []
        for j, (typ, label, s, p) in enumerate(a["nodes"], start=1):
            assert typ in TYPES_NOEUDS, (cle, typ)
            assert 1 <= s <= p <= len(toks) and p - s + 1 <= 15, (cle, s, p)
            nodes.append(
                {
                    "id": f"n{j}",
                    "type": typ,
                    "label": label,
                    "token_span": [s, p],
                    "scope": "specific",
                    "temporal_index": j - 1,
                    "origin": "explicit",
                    "span_source": "manual",
                }
            )
        ids_noeuds = {x["id"] for x in nodes}
        edges = []
        for srcs, tgt, rel, expl, conf, marker in a["edges"]:
            assert rel in RELATIONS, (cle, rel)
            assert set(srcs) <= ids_noeuds and tgt in ids_noeuds, cle
            assert 0.5 <= conf <= 0.9, (cle, conf)  # pas de 1.0 (guide §2)
            assert marker is None or 1 <= marker <= len(toks), (cle, marker)
            edges.append(
                {
                    "sources": srcs,
                    "target": tgt,
                    "relation": rel,
                    "explicit": expl,
                    "confidence": conf,
                    "third": None,
                    "marker_token": marker,
                }
            )
        phrases.append(
            {
                "id": f"ni{n:03d}",
                "text": texte,
                "tokens": toks,
                "sentence_type": a["stype"],
                "registre": "g0_fr",
                "intent": a["intent"],
                "cir": {"nodes": nodes, "edges": edges},
                "lot": "socle_ni",
                "annotateur": "assistant-aveugle",
                "source_g0": {k: e.get(k) for k in ("lang", "source", "url")},
            }
        )

    # Requêtes elicitées (registre taggé, jamais mélangées sans tag).
    for i in range(10):
        cle = f"request{i}"
        a = ann[cle]
        texte = a["text"]
        toks = tokens_de(texte)
        nodes = [
            {
                "id": f"n{j}",
                "type": t,
                "label": lb,
                "token_span": [s, p],
                "scope": "specific",
                "temporal_index": j - 1,
                "origin": "explicit",
                "span_source": "manual",
            }
            for j, (t, lb, s, p) in enumerate(a["nodes"], start=1)
        ]
        assert all(1 <= s <= p <= len(toks) for _, _, s, p in a["nodes"]), cle
        edges = [
            {
                "sources": s,
                "target": t,
                "relation": r,
                "explicit": x,
                "confidence": c,
                "third": None,
                "marker_token": m,
            }
            for s, t, r, x, c, m in a["edges"]
        ]
        phrases.append(
            {
                "id": f"ni{70 + i + 1:03d}",
                "text": texte,
                "tokens": toks,
                "sentence_type": a["stype"],
                "registre": "elicited",
                "intent": a["intent"],
                "cir": {"nodes": nodes, "edges": edges},
                "lot": "socle_ni",
                "annotateur": "assistant-aveugle",
            }
        )

    assert len(phrases) == 80, len(phrases)
    # Zéro fuite : aucun texte dans final2.
    interleak = set()
    for split in ("train", "val", "test"):
        d = json.loads(
            (
                RACINE
                / "gcn-datasets"
                / "DATA"
                / "supervision"
                / "v4"
                / "final2"
                / split
                / f"{split}.json"
            ).read_text(encoding="utf-8")
        )["document"]["sentences"]
        interleak.update(hashlib.md5(s["text"].strip().encode()).hexdigest() for s in d)
    for p in phrases:
        assert hashlib.md5(p["text"].strip().encode()).hexdigest() not in interleak, p[
            "id"
        ]

    (socle / "lot_socle_ni_assistant.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in phrases), encoding="utf-8"
    )
    from collections import Counter

    ct, ci, cr = Counter(), Counter(), Counter()
    for p in phrases:
        for x in p["cir"]["nodes"]:
            ct[x["type"]] += 1
        for x in p["cir"]["edges"]:
            cr[x["relation"]] += 1
        ci[p["intent"]] += 1
    manifest = {
        "lot": "socle_ni",
        "passe": "assistant-aveugle",
        "n": 80,
        "quarantaine": (
            "10 requêtes minées (0 authentique : FP regex "
            "impératif sur noms en -e, puis ! exclamatifs "
            "≠ request R9) — motif conservé, tombstones"
        ),
        "noeuds": dict(ct),
        "intents": dict(ci),
        "relations": dict(cr),
    }
    (socle / "manifest_assistant.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
