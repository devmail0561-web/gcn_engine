#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Adapte le CIR code (analyze-code) au schéma d'entraînement (C0).

Chaque fichier source -> 1 phrase : tokens = nœuds AST (1 token/nœud),
lemme = kind grammatical (vocab fini, cf. kinds.rs), clauses ponctuelles,
arêtes listes -> dicts. Les labels source bruts (~78k uniques) ne sont
JAMAIS utilisés comme lemmes (pas de généralisation possible).

Règle pos (jetons structurels, pas linguistiques — documentée, pas devinée
en silence) : NOUN si le kind définit/nomme (definition/declaration/import),
VERB sinon. Les deux sont dans CONTENT_POS (features.py) donc les lemmes
sont poolés vers les embeddings ; tout autre pos tuerait le signal lexical.

Usage :
    python3 scripts/code_to_train.py <code_ast.jsonl> <out_train.json> [max_nodes=150]
    -> {"schema_version": "4.0", "document": {"id": ..., "sentences": [...]}}

Fichiers > max_nodes nœuds : quarantaines (motif compté, tombstone) —
all_pairs explose en n² (2762 nœuds = 3.8M paires ≈ 17 Go : OOM vérifié
le 2026-10-02). Granularité par bloc pour les géants = chantier ultérieur.
"""
import json
import sys
from collections import Counter
from pathlib import Path

NOUN_HINTS = ("definition", "declaration", "import")


def kind_pos(kind: str) -> str:
    k = (kind or "").lower()
    return "NOUN" if any(h in k for h in NOUN_HINTS) else "VERB"


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    max_nodes = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    sentences, quar, skipped_kind, backward, rels = [], [], 0, 0, Counter()
    for idx, line in enumerate(open(src, encoding="utf-8", errors="replace")):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        cir = r.get("cir", {})
        nodes = cir.get("nodes", [])
        if not nodes:
            continue
        if len(nodes) > max_nodes:
            quar.append({"id": r.get("file", f"fichier-{idx}"),
                         "motif": f"trop gros: {len(nodes)} noeuds > {max_nodes} "
                                  f"(all_pairs en n2)",
                         "tombstone": True})
            continue
        lang = str(r.get("lang", "?"))
        order = sorted(range(len(nodes)), key=lambda i: nodes[i].get("id", i))
        new_id = {nodes[i].get("id"): f"n{k + 1}" for k, i in enumerate(order)}
        toks, clauses = [], []
        for k, i in enumerate(order):
            n = nodes[i]
            kind = n.get("kind")
            if not kind:
                skipped_kind += 1
                continue
            tid = k + 1
            label = str(n.get("label", ""))
            toks.append({
                "id": tid,
                "form": label.split("\n")[0][:64] or kind,
                "lemma": str(kind),
                "pos": kind_pos(str(kind)),
                "dep_rel": "root",
                "dep_head": 0,
                "morph": {},
            })
            clauses.append({
                "id": f"n{tid}",
                "type": str(n.get("node_type", "processus")),
                "label": label[:256],
                "token_span": [tid, tid],
                "scope": str(n.get("scope", "specific")),
                "temporal_index": int(n.get("temporal_index") or 0),
                "origin": str(n.get("origin", "explicit")),
            })
        if not clauses:
            continue
        kept = {f"n{k + 1}" for k, i in enumerate(order) if nodes[i].get("kind")}
        edges = []
        for e in cir.get("edges", []):
            if not isinstance(e, list) or len(e) < 3:
                continue
            s, t, a = new_id.get(e[0]), new_id.get(e[1]), e[2] or {}
            if not s or not t or s not in kept or t not in kept:
                continue
            if s > t:
                # Vers l'arrière : transmis tel quel — le loader décide
                # (ignore cause/enable/prevent, remappe les autres, loader.py:206-230).
                backward += 1
            rels[a.get("relation", "?")] += 1
            edges.append({
                "sources": [s], "target": t,
                "relation": str(a.get("relation", "")),
                "confidence": 1.0, "explicit": True,
            })
        stem = Path(r.get("file", f"fichier-{idx}")).stem.replace(" ", "_")[:40]
        sentences.append({
            "id": f"code-{stem}-{idx}",
            "text": str(cir.get("source_text", r.get("file", "")))[:4000],
            "tokens": toks,
            "sentence_type": "declarative",
            "registre": f"code/{lang}",
            "cir": {"nodes": clauses, "edges": edges},
        })
    doc = {"schema_version": "4.0",
           "document": {"id": Path(dst).stem, "sentences": sentences}}
    json.dump(doc, open(dst, "w"), ensure_ascii=False)
    qpath = str(Path(dst).with_name(Path(dst).stem + "_quar.json"))
    json.dump(quar, open(qpath, "w"), ensure_ascii=False)
    print(f"phrases: {len(sentences)} | quarantaines: {len(quar)} "
          f"| sans-kind ignorés: {skipped_kind} | inverses transmises: {backward} (loader décide)")
    print("REL:", dict(rels))


main()
