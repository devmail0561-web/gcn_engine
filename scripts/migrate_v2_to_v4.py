#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""migrate_v2_to_v4.py — Migration schéma v2 → v4 pour les datasets GCN-NL.

Transformations :
  - edge.source (str) → edge.sources ([str])
  - node.type : etat→etat_local, action→processus, transition→processus,
                etat_systemique→etat_global
  - Ajout edge.third = null si absent
  - Ajout edge.confidence = 1.0 si absent (gold)
  - Ajout sentence.intent = "" si absent
  - Ajout document.schema_version = "4.0"

Usage :
    python scripts/migrate_v2_to_v4.py                      # tous les fichiers
    python scripts/migrate_v2_to_v4.py --dry-run             # simulation
    python scripts/migrate_v2_to_v4.py --file <path>         # fichier unique
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "gcn-datasets"

NODE_TYPE_ALIASES = {
    "etat": "etat_local",
    "etat_systemique": "etat_global",
    "action": "processus",
    "transition": "processus",
    # déjà valides — identité
    "etat_local": "etat_local",
    "etat_global": "etat_global",
    "processus": "processus",
    "entite": "entite",
    "condition": "condition",
    "concept": "concept",
    "evenement": "evenement",
    "contrainte": "contrainte",
}

TARGET_FILES = [
    DATASETS_DIR / "real" / "train" / "train.json",
    DATASETS_DIR / "real" / "val"   / "val.json",
    DATASETS_DIR / "real" / "test"  / "test.json",
    DATASETS_DIR / "splits" / "train" / "train.json",
    DATASETS_DIR / "splits" / "val"   / "val.json",
    DATASETS_DIR / "splits" / "test"  / "test.json",
    DATASETS_DIR / "DATA" / "supervision" / "combined_v2" / "train.json",
    DATASETS_DIR / "DATA" / "supervision" / "combined_v2" / "val.json",
    DATASETS_DIR / "DATA" / "supervision" / "combined_v2" / "test.json",
]


def migrate_node(node: dict) -> dict:
    t = node.get("type", "")
    node["type"] = NODE_TYPE_ALIASES.get(t, t)
    return node


def migrate_edge(edge: dict) -> dict:
    # Aplatir attributes imbriqués → niveau arête
    attrs = edge.pop("attributes", {}) or {}
    for key in ("confidence", "explicit", "negated", "marker_token", "temporal_gap"):
        if key in attrs and key not in edge:
            edge[key] = attrs[key]

    # source → sources
    if "source" in edge and "sources" not in edge:
        edge["sources"] = [edge.pop("source")]
    elif "source" in edge and "sources" in edge:
        edge.pop("source")

    # third absent → null
    if "third" not in edge:
        edge["third"] = None

    # confidence absent → 1.0 (gold)
    if "confidence" not in edge:
        edge["confidence"] = 1.0

    return edge


def migrate_sentence(sentence: dict) -> dict:
    cir = sentence.get("cir", {})

    nodes = [migrate_node(n) for n in cir.get("nodes", [])]
    edges = [migrate_edge(e) for e in cir.get("edges", [])]
    cir["nodes"] = nodes
    cir["edges"] = edges
    sentence["cir"] = cir

    if "intent" not in sentence:
        sentence["intent"] = ""

    return sentence


def migrate_document(data: dict) -> tuple[dict, int, int]:
    """Retourne (data_migrée, n_nodes_migrés, n_edges_migrés)."""
    data["schema_version"] = "4.0"

    doc = data.get("document", data)
    sentences = doc.get("sentences", [])

    n_nodes = n_edges = 0
    migrated = []
    for s in sentences:
        before_nodes = [n.get("type") for n in s.get("cir", {}).get("nodes", [])]
        before_edges = [e.get("source") for e in s.get("cir", {}).get("edges", [])]
        s = migrate_sentence(s)
        n_nodes += sum(1 for old, new in zip(before_nodes,
                       [n["type"] for n in s["cir"]["nodes"]])
                       if old != new)
        n_edges += sum(1 for e in before_edges if e is not None)
        migrated.append(s)

    doc["sentences"] = migrated
    if "document" in data:
        data["document"] = doc
    return data, n_nodes, n_edges


def migrate_file(path: Path, dry_run: bool = False) -> None:
    if not path.exists():
        print(f"  SKIP {path.name} — fichier absent")
        return

    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)

    data, n_nodes, n_edges = migrate_document(data)

    new_json = json.dumps(data, ensure_ascii=False, indent=2)

    # Compter les phrases
    doc = data.get("document", data)
    n_sentences = len(doc.get("sentences", []))

    print(f"  {'[DRY]' if dry_run else 'OK  '} {path.relative_to(REPO_ROOT)} "
          f"— {n_sentences} phrases, {n_nodes} nœuds migrés, {n_edges} arêtes source→sources")

    if not dry_run:
        # Backup
        backup = path.with_suffix(".json.v2bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(new_json, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate GCN datasets v2 → v4")
    parser.add_argument("--dry-run", action="store_true", help="Simulation sans écriture")
    parser.add_argument("--file", type=Path, default=None, help="Fichier unique")
    args = parser.parse_args()

    files = [args.file] if args.file else TARGET_FILES

    print(f"Migration v2→v4 {'[DRY RUN]' if args.dry_run else ''}")
    print("=" * 60)
    for f in files:
        migrate_file(f, dry_run=args.dry_run)
    print("=" * 60)
    if not args.dry_run:
        print("Backups .json.v2bak créés. Migration terminée.")
    else:
        print("DRY RUN terminé — aucune modification.")


if __name__ == "__main__":
    main()
