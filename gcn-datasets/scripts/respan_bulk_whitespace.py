# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Re-spanning lot37 (bulk TOK-regex → whitespace).

Motif (2026-10-16, ablation) : les spans style TOK-regex (« L ' ICA »)
dégradent le test -0.08 vs exclusion ; re-spannés whitespace : écart au
bruit (0.147 vs 0.174, Δ<0.05). Les labels PV1 sont intacts (mêmes jugements),
seuls tokens/spans/marqueurs changent (mappage par offsets caractères,
inclusion stricte puis repli chevauchement, assert par nœud).

Entrée : annotated/lot37_bulk.json → sortie : annotated/lot39_bulk_respan.json
(id `bulk*` → `rsp*`, 126/126, loader 0-rejet).
"""
from __future__ import annotations

import argparse
import json


def tok_offsets(text: str, forms: list[str]) -> list[tuple[int, int]]:
    offs, p = [], 0
    for f in forms:
        i = text.find(f, p)
        assert i >= 0, (f, text[:60])
        offs.append((i, i + len(f)))
        p = i + len(f)
    return offs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Re-span bulk whitespace.")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args(argv)
    d = json.load(open(args.src, encoding="utf-8"))
    out = []
    for s in d["document"]["sentences"]:
        t = s["text"]
        to = tok_offsets(t, [x["form"] for x in s["tokens"]])
        wo = tok_offsets(t, t.split(" "))
        wtoks = [{"id": k + 1, "form": f} for k, f in enumerate(t.split(" "))]
        nodes = []
        for n in s["cir"]["nodes"]:
            a, b = n["token_span"]
            cs, ce = to[a - 1][0], to[b - 1][1]
            ids = [k + 1 for k, (ws, we) in enumerate(wo) if ws >= cs and we <= ce]
            if not ids:
                ids = [k + 1 for k, (ws, we) in enumerate(wo) if ws < ce and we > cs]
            assert ids, (s["id"], n["id"])
            nn = dict(n)
            nn["token_span"] = [ids[0], ids[-1]]
            nn["span_source"] = "respan-whitespace"
            nodes.append(nn)
        edges = []
        for e in s["cir"]["edges"]:
            ne = dict(e)
            mt = e.get("marker_token")
            if mt:
                ms, me = to[mt - 1]
                mid = (ms + me) // 2
                ne["marker_token"] = next(
                    k + 1 for k, (ws, we) in enumerate(wo) if ws <= mid < we
                )
            edges.append(ne)
        out.append({"id": s["id"].replace("bulk", "rsp"), "text": t,
                    "tokens": wtoks, "cir": {"nodes": nodes, "edges": edges}})
    doc = {"schema_version": "4.0",
           "document": {"id": "lot39-bulk-respan", "lang": "fr",
                        "weight": 1.0, "sentences": out}}
    json.dump(doc, open(args.dst, "w", encoding="utf-8"), ensure_ascii=False)
    print(json.dumps({"phrases": len(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
