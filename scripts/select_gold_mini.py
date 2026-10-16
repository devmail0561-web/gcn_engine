#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Sélection gold_mini 60-100 + gel held-out — socle 0c (phase 0).

Extraction seule (zéro re-annotation) depuis les lots v4 arbitrés 08-24.
Stratification : relation primaire × registre, seed 42, déterministe.
Held-out : snapshot gelé (copie + manifest sha256) de mixte_final/test
(seed 42) et v4/final2/test, HORS docs/.

Sorties (gcn-datasets/DATA/supervision/, hors git) :
    gold_mini/gold_mini.jsonl + gold_mini_manifest.json
    heldout/heldout_*.json + heldout_manifest.json
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SEED = 42
PER_REL = 7  # ~7 × 11 relations ≈ 77 + non-déclaratives → dans 60-100
V4_TYPES = {"processus", "etat_local", "etat_global", "entite",
            "condition", "concept", "evenement", "contrainte"}
V4_RELS = {"cause", "enable", "prevent", "condition", "concession", "sequence",
           "motivation", "filter", "opposition", "data_dependency",
           "control_dependency", "analogy", "counterfactual",
           "conditional_cause", "mediated_cause", "joint_cause",
           "conditional_prevent", "mediated_prevent", "joint_prevent"}

SUP = Path("gcn-datasets/DATA/supervision")
LOT_RE = re.compile(r"lot(\d+).*\.json$")


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def load_lot(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(d, dict) and "document" in d:
        return d["document"].get("sentences", [])
    return d if isinstance(d, list) else []


def primary_rel(sent: dict) -> str | None:
    cir = sent.get("cir")
    if not isinstance(cir, dict):
        return None
    for e in cir.get("edges", []) or []:
        if isinstance(e, dict) and e.get("relation"):
            return e["relation"]
    return None


def valid_v4(sent: dict) -> bool:
    cir = sent.get("cir")
    if not isinstance(cir, dict):
        return False
    for nd in cir.get("nodes", []) or []:
        if isinstance(nd, dict) and nd.get("type") not in V4_TYPES:
            return False
    for e in cir.get("edges", []) or []:
        if isinstance(e, dict) and e.get("relation") not in V4_RELS:
            return False
    return True


def main() -> int:
    rng = random.Random(SEED)
    lots = sorted((SUP / "v4" / "annotated").glob("lot*.json"))
    lots = [p for p in lots if (m := LOT_RE.match(p.name)) and 8 <= int(m.group(1)) <= 24]
    if not lots:
        print("ERREUR : aucun lot 08-24 trouvé", file=sys.stderr)
        return 1
    pool: dict[str, list[dict]] = {}
    special: list[dict] = []
    seen_hashes: set[str] = set()
    file_hashes = {}
    for p in lots:
        file_hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        for s in load_lot(p):
            if not valid_v4(s):
                continue
            h = sha256_text(s.get("text", ""))
            if h in seen_hashes:  # dédup inter-lots
                continue
            seen_hashes.add(h)
            rec = {"source_file": p.name, "sentence_id": s.get("id"),
                   "text_sha256": h, **s}
            rel = primary_rel(s)
            if s.get("sentence_type", "declarative") != "declarative":
                special.append(rec)
            elif rel:
                pool.setdefault(rel, []).append(rec)
    # Tri déterministe avant shuffle seedé (reproductible)
    for v in pool.values():
        v.sort(key=lambda r: (r["source_file"], str(r["sentence_id"])))
    chosen: list[dict] = []
    for rel in sorted(pool):
        cands = pool[rel][:]
        rng.shuffle(cands)
        chosen.extend(cands[:PER_REL])
    special.sort(key=lambda r: (r["source_file"], str(r["sentence_id"])))
    chosen.extend(special)  # exclamatives/interrogatives : précieuses (futurs actes)
    chosen.sort(key=lambda r: (r["source_file"], str(r["sentence_id"])))

    outdir = SUP / "gold_mini"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "gold_mini.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen)
    counts_rel: dict[str, int] = {}
    counts_reg: dict[str, int] = {}
    counts_st: dict[str, int] = {}
    for r in chosen:
        counts_rel[primary_rel(r) or "?"] = counts_rel.get(primary_rel(r) or "?", 0) + 1
        counts_reg[r.get("registre")] = counts_reg.get(r.get("registre"), 0) + 1
        counts_st[r.get("sentence_type")] = counts_st.get(r.get("sentence_type"), 0) + 1
    manifest = {
        "id": "gold-mini-v1", "seed": SEED,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n": len(chosen), "target": "60-100 (socle refacto 0c)",
        "source": "lots v4 arbitrés 08-24, extraction seule (zéro re-annotation)",
        "source_files": file_hashes,
        "counts_relation": counts_rel, "counts_registre": counts_reg,
        "counts_sentence_type": counts_st,
        "note": "Jamais vu au train (acceptation C). Quasi-FR ; EN mince — extension gold-3000.",
    }
    (outdir / "gold_mini_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    # Gel held-out (snapshot + manifest, hors docs/)
    hodir = SUP / "heldout"
    hodir.mkdir(parents=True, exist_ok=True)
    frozen = []
    for src, dst in [("mixte_final/test/test.json", "heldout_mixte_final_test.json"),
                     ("v4/final2/test/test.json", "heldout_v4_final2_test.json")]:
        sp = SUP / src
        if not sp.exists():
            print(f"AVERTISSEMENT : {src} absent, ignoré")
            continue
        dp = hodir / dst
        shutil.copyfile(sp, dp)
        frozen.append({"dst": dst, "src": src,
                       "sha256": hashlib.sha256(dp.read_bytes()).hexdigest(),
                       "bytes": dp.stat().st_size})
    (hodir / "heldout_manifest.json").write_text(json.dumps(
        {"id": "heldout-v1", "seed": SEED,
         "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "note": "Splits existants re-gelés (mixte_final seed 42). Ne jamais entraîner dessus.",
         "files": frozen}, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"gold_mini : n={len(chosen)} "
          f"(dans 60-100 : {60 <= len(chosen) <= 100})")
    print(f"  relations : {dict(sorted(counts_rel.items()))}")
    print(f"  registres : {dict(sorted(counts_reg.items(), key=lambda kv: -kv[1]))}")
    print(f"  sentence_types : {counts_st}")
    print(f"held-out gelés : {[f['dst'] for f in frozen]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
