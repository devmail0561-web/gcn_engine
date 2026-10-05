#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Construit le dataset d'entraînement de l'étage-2 (deux-étages).

Étage-1 (gelé, gate 0.854) : `balanced_auto.annotate` (gcn-tools — c'est
là que vit la langue). Étage-2 : classifieur 12 classes (11 relations +
REJECT) sur features AGNOSTIQUES calculées ici et stockées numériques :
`gcn-python` ne voit jamais de lemme ni de langue.

Features (36 dims, ordre gelé, cf. FEATURE_SCHEMA) :
  0-10  prior one-hot (11 relations, ordre RELS)
  11    confiance proposeur (1.0/0.9/0.8)
  12    direction fwd=1 / bwd=0
  13    n_markers (len(find_markers))
  14    marker_at_head (match du connecteur utilisé en tête de phrase)
  15    conflit_marqueurs (≥2 relations distinctes parmi find_markers)
  16-17 longueur spans src/dst (tokens regex)
  18-19 content-counts src/dst (content_words sur spans reconstruits)
  20-27 type nœud src one-hot (7 + _other)
  28-35 type nœud dst one-hot (7 + _other)

Labels : 0-10 gold (phrases final2) / 11 REJECT (quarantaines déclenchées).
Sans-proposition → bucket `abstain` (perte de couverture étage-1,
inatteignable à l'inférence), exclu du train.

Usage :
    python3 scripts/build_stage2.py --out gcn-datasets/DATA/supervision/v4/stage2 [--seed 42]
    python3 scripts/build_stage2.py --out ... --check   # vérifie un build existant
"""
import argparse
import glob
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gcn-tools" / "gcn-annotate" / "src"))
from gcn_annotate.balanced_auto import (  # noqa: E402
    annotate,
    content_words,
    find_markers,
    tokenize,
)

V4 = ROOT / "gcn-datasets" / "DATA" / "supervision" / "v4"

RELS = ["cause", "concession", "condition", "control_dependency",
        "data_dependency", "enable", "filter", "motivation", "opposition",
        "prevent", "sequence"]
REJECT = 11
NTYPES = ["condition", "entite", "etat_systemique", "transition",
          "etat", "action", "processus", "_other"]
N_FEATURES = 36
FEATURE_SCHEMA = ("prior_onehot[11],confidence,direction,n_markers,"
                  "marker_at_head,conflit_marqueurs,span_len[2],"
                  "content[2],ntype_src[8],ntype_dst[8]")


def propose(text):
    """(proposal_dict, lang) — fr puis repli en, comme gate_accuracy."""
    for lang in ("fr", "en"):
        try:
            r = annotate(text, lang)
        except Exception:
            r = None
        if r and r.get("edges"):
            try:
                ms = find_markers(text, lang)
            except Exception:
                ms = []
            return r, lang, ms
    return None, None, []


def featurize(text, r, lang, markers):
    """proposal -> vecteur 36 dims (tool-side : le lexique vit ici)."""
    edge = r["edges"][0]
    silv = r.get("_silver", {})
    prior = edge.get("relation")
    vec = [0.0] * N_FEATURES
    if prior in RELS:
        vec[RELS.index(prior)] = 1.0
    vec[11] = float(edge.get("attributes", {}).get("confidence", 0.0))
    vec[12] = 1.0 if silv.get("direction") == "fwd" else 0.0
    vec[13] = float(len(markers))
    conn = silv.get("connector", "")
    head = 0.0
    for c, _rel, _d, m in markers:
        if c == conn and m.start() == 0:
            head = 1.0
            break
    vec[14] = head
    vec[15] = 1.0 if len({rel for _, rel, _, _ in markers}) >= 2 else 0.0
    toks = tokenize(text)
    nodes = r.get("nodes", [])
    for k in (0, 1):
        if k < len(nodes):
            a, b = nodes[k].get("token_span", [1, 1])
            a, b = max(1, int(a)), max(int(b), int(a))
            vec[16 + k] = float(b - a + 1)
            clause = " ".join(toks[a - 1:b])
            vec[18 + k] = float(len(content_words(clause, lang)))
            t = nodes[k].get("type", "_other")
            vec[20 + k * 8 + NTYPES.index(t if t in NTYPES else "_other")] = 1.0
    return vec


def sha_texts(items):
    h = hashlib.sha256()
    for it in sorted(items, key=lambda x: x["id"]):
        h.update((it["id"] + "\x00" + it["text"]).encode("utf-8"))
    return h.hexdigest()[:16]


def build(out_dir, seed):
    rng = random.Random(seed)
    out = Path(out_dir)
    # --- gold final2 (splits hérités) ---
    splits = {}
    for split in ("train", "val", "test"):
        d = json.load(open(V4 / f"final2/{split}/{split}.json"))
        splits[split] = d["document"]["sentences"]
    rows = {s: [] for s in ("train", "val", "test")}
    abstain = {s: [] for s in ("train", "val", "test")}
    for split, ss in splits.items():
        for s in ss:
            text = s.get("text", "")
            es = s.get("cir", {}).get("edges", [])
            gold = es[0].get("relation") if es else None
            if gold not in RELS:
                continue
            r, lang, markers = propose(text)
            if r is None:
                abstain[split].append({"id": s["id"], "gold": gold,
                                       "text": text})
                continue
            rows[split].append({
                "id": s["id"], "text": text, "split": split,
                "label": RELS.index(gold), "gold": gold,
                "proposed": r["edges"][0].get("relation"),
                "lang": lang, "features": featurize(text, r, lang, markers),
            })
    # --- quarantaines dédupliquées, split seedé 100/25/24 ---
    seen = {}
    for f in sorted(glob.glob(str(V4 / "annotated" / "quarantaine*.json"))):
        for q in json.load(open(f)):
            if isinstance(q, dict) and q.get("id") and q.get("text"):
                seen.setdefault(q["id"], q)
    qs = list(seen.values())
    rng.shuffle(qs)
    qparts = {"train": qs[:100], "val": qs[100:125], "test": qs[125:]}
    qfire = {s: [] for s in ("train", "val", "test")}
    qsilent = []
    for part, items in qparts.items():
        for q in items:
            r, lang, markers = propose(q["text"])
            if r is None:
                qsilent.append(q["id"])
                continue
            qfire[part].append({
                "id": q["id"], "text": q["text"], "split": f"quar_{part}",
                "label": REJECT, "gold": "REJECT",
                "proposed": r["edges"][0].get("relation"),
                "lang": lang, "features": featurize(q["text"], r, lang, markers),
                "motif": (q.get("motif") or "")[:120],
            })
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": seed, "n_features": N_FEATURES,
        "feature_schema": FEATURE_SCHEMA, "relations": RELS,
        "node_types": NTYPES, "reject_label": REJECT,
        "langs": ["fr", "en"],
        "splits": {},
        "abstain": {s: len(abstain[s]) for s in abstain},
        "abstain_ids": {s: [a["id"] for a in abstain[s]] for s in abstain},
        "quar_dedup": len(seen),
        "quar_silent_true_negatives": qsilent,
    }
    for split in ("train", "val", "test"):
        data = rows[split] + qfire[split]
        rng.shuffle(data)
        json.dump(data, open(out / f"{split}.json", "w"),
                  ensure_ascii=False)
        dist = dict(sorted(Counter(d["label"] for d in data).items()))
        manifest["splits"][split] = {
            "n": len(data), "gold_n": len(rows[split]),
            "reject_n": len(qfire[split]), "label_dist": dist,
            "sha256": sha_texts(data),
        }
    json.dump(abstain, open(out / "abstain.json", "w"), ensure_ascii=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                       encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items()
                      if k != "abstain_ids"}, indent=1, ensure_ascii=False))
    return manifest


def check(out_dir):
    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_features"] == N_FEATURES, "schema features modifié"
    assert manifest["relations"] == RELS, "ordre relations modifié"
    assert manifest["node_types"] == NTYPES, "types nœuds modifiés"
    seen_ids = set()
    for split in ("train", "val", "test"):
        data = json.load(open(out / f"{split}.json"))
        for d in data:
            assert len(d["features"]) == N_FEATURES, d["id"]
            assert 0 <= d["label"] <= REJECT, d["id"]
            assert all(isinstance(x, (int, float)) for x in d["features"])
            assert d["id"] not in seen_ids, f"doublon inter-splits {d['id']}"
            seen_ids.add(d["id"])
        m = manifest["splits"][split]
        assert m["n"] == len(data), split
        assert m["sha256"] == sha_texts(data), f"sha {split}"
    print(f"CHECK OK: {len(seen_ids)} échantillons uniques, "
          f"dims={N_FEATURES}, labels⊂[0,11]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check(args.out)
    else:
        build(args.out, args.seed)


if __name__ == "__main__":
    main()
