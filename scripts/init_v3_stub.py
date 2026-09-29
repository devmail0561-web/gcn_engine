#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""init_v3_stub.py — Initialise un stub v3.0 depuis un checkpoint v2.

NE PAS utiliser ce stub directement en production.
Ce stub est un point de départ pour réentraîner sur le schéma v3.0 (d_clause=106,
19 relations, 8 nœuds). Tous les poids shape-dépendants sont réinitialisés par
He-init (cohérent avec le reste du code) ; les poids shape-stables (embeddings,
couches internes) sont préservés.

Usage :
    python scripts/init_v3_stub.py model_train_v2.npz
    python scripts/init_v3_stub.py model_train_v2.npz --out checkpoints/stub_v3.npz
    python scripts/init_v3_stub.py model_train_v2.npz --d-emb 128

La décomposition des formes v2 → v3 est calculée automatiquement depuis l'arch_json.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "gcn-python" / "src"))

from gcn_python.security.npz_guard import guarded_np_load  # noqa: E402

# v3.0 constants
N_RELATIONS_V3 = 19
N_NODE_TYPES_V3 = 8
D_CLAUSE_V3 = 106  # 18+38+5+5+4+5+3+7+1+3+12+5


def _he_init(shape: tuple, fan_in: int, rng: np.random.Generator) -> np.ndarray:
    """He-init : N(0, sqrt(2/fan_in)) — cohérent avec le reste du code."""
    std = np.sqrt(2.0 / max(fan_in, 1))
    return rng.standard_normal(shape).astype(np.float32) * std


def _zero_bias(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float32)


def stub_v3(
    v2_path: Path,
    out_path: Path,
    d_emb_override: int | None = None,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)

    # --- Charger le checkpoint v2 ---
    print(f"Lecture checkpoint v2 : {v2_path}")
    data = guarded_np_load(v2_path)
    arch_v2 = json.loads(str(data["_arch_json"][0]))

    # --- Architecture v2 ---
    d_emb = d_emb_override if d_emb_override is not None else int(arch_v2.get("d_emb", 128))
    d_eff_v2 = int(arch_v2.get("d_eff", 207))
    mlp_hidden = int(arch_v2.get("mlp_hidden", 128))
    d_hidden = int(arch_v2.get("d_hidden", d_eff_v2))
    n_relations_v2 = int(arch_v2.get("n_relations", 22))
    bidi = bool(arch_v2.get("bidirectional", True))
    subject_object_emb = bool(arch_v2.get("subject_object_emb", False))
    n_rgcn_layers = int(arch_v2.get("n_rgcn_layers", 1))
    graph_class = arch_v2.get("graph_class", "RGCNLayerPT")

    # --- Architecture v3 ---
    d_eff_v3 = D_CLAUSE_V3 + d_emb + (2 * d_emb if subject_object_emb else 0)
    n_all_rel_v2 = n_relations_v2  # already bidi in checkpoint
    n_all_rel_v3 = N_RELATIONS_V3 * 2 if bidi else N_RELATIONS_V3

    # Recalcule d_conn depuis la forme encoder_6 v2
    # encoder_6 = (mlp_hidden_edge, d_edge_closed_loop)
    # d_edge_closed_loop = d_edge + 2*d_emb_per_clause + 2*d_eff + 2*n_node_types_v2
    # d_emb_per_clause = 3*d_emb si subject_object_emb else d_emb
    d_emb_per_clause = 3 * d_emb if subject_object_emb else d_emb
    n_node_types_v2 = int(arch_v2.get("n_node_types", 7))
    enc6_shape = data["encoder_6"].shape  # (mlp_hidden_edge, d_ecl_v2)
    d_ecl_v2 = enc6_shape[1]
    d_edge_v2 = d_ecl_v2 - 2 * d_emb_per_clause - 2 * d_eff_v2 - 2 * n_node_types_v2
    # d_edge = 2*d_clause + d_conn + 4
    d_clause_v2 = d_eff_v2 - d_emb - (2 * d_emb if subject_object_emb else 0)
    d_conn = d_edge_v2 - 2 * d_clause_v2 - 4

    d_edge_v3 = 2 * D_CLAUSE_V3 + d_conn + 4
    d_ecl_v3 = d_edge_v3 + 2 * d_emb_per_clause + 2 * d_eff_v3 + 2 * N_NODE_TYPES_V3

    print(f"  v2 : d_clause={d_clause_v2}, d_eff={d_eff_v2}, n_rel={n_relations_v2}, "
          f"n_nodes={n_node_types_v2}, d_ecl={d_ecl_v2}")
    print(f"  v3 : d_clause={D_CLAUSE_V3}, d_eff={d_eff_v3}, n_rel={N_RELATIONS_V3}, "
          f"n_nodes={N_NODE_TYPES_V3}, d_ecl={d_ecl_v3}")
    print(f"  d_conn (inchangé)={d_conn}, d_emb={d_emb}")

    # --- Construire les nouveaux arrays ---
    arrays: dict[str, np.ndarray] = {}

    # Encoder node MLP : couche 0 dépend de d_eff
    arrays["encoder_0"] = _he_init((mlp_hidden, d_eff_v3), d_eff_v3, rng)
    arrays["encoder_1"] = _zero_bias(mlp_hidden)
    # Couches intermédiaires préservées (shape stable)
    for k in ("encoder_2", "encoder_3"):
        arrays[k] = data[k].copy()
    # Node classifier output : 7 → 8 nœuds
    arrays["encoder_4"] = _he_init((N_NODE_TYPES_V3, data["encoder_4"].shape[1]),
                                   data["encoder_4"].shape[1], rng)
    arrays["encoder_5"] = _zero_bias(N_NODE_TYPES_V3)

    # Edge MLP : couche 0 dépend de d_ecl
    n_edge_hidden = data["encoder_6"].shape[0]
    arrays["encoder_6"] = _he_init((n_edge_hidden, d_ecl_v3), d_ecl_v3, rng)
    arrays["encoder_7"] = _zero_bias(n_edge_hidden)
    # Couches intermédiaires préservées
    for k in ("encoder_8", "encoder_9", "encoder_10", "encoder_11"):
        if k in data.files:
            arrays[k] = data[k].copy()
    # Edge classifier output : 11 → 19 relations
    arrays["encoder_12"] = _he_init((N_RELATIONS_V3, data["encoder_12"].shape[1]),
                                    data["encoder_12"].shape[1], rng)
    arrays["encoder_13"] = _zero_bias(N_RELATIONS_V3)

    # R-GCN graph layers : (n_all_rel, d_in, d_out) → (n_all_rel_v3, d_eff_v3, d_eff_v3)
    for layer_i in range(n_rgcn_layers):
        prefix = "graph" if layer_i == 0 else f"graph_extra_{layer_i}"
        g0 = f"{prefix}_0"
        g1 = f"{prefix}_1"
        g2 = f"{prefix}_2"
        if g0 in data.files:
            arrays[g0] = _he_init((n_all_rel_v3, d_eff_v3, d_eff_v3), d_eff_v3, rng)
        if g1 in data.files:
            arrays[g1] = _he_init((d_eff_v3, d_eff_v3), d_eff_v3, rng)
        if g2 in data.files:
            arrays[g2] = _he_init((1, n_all_rel_v3, 2 * d_eff_v3), 2 * d_eff_v3, rng)

    # Embeddings préservés (shape stable : vocab × d_emb)
    if "word_emb_E" in data.files:
        arrays["word_emb_E"] = data["word_emb_E"].copy()
    for k in ("word_emb_pretrained_start", "word_emb_pretrained_end"):
        if k in data.files:
            arrays[k] = data[k].copy()

    # Clés JSON metadata
    arrays["_vocab_json"] = data["_vocab_json"].copy()  # vocab inchangé (tokens)

    # Arch v3 — mise à jour des dimensions
    arch_v3 = dict(arch_v2)
    arch_v3.update({
        "d_eff":           d_eff_v3,
        "n_relations":     n_all_rel_v3,
        "schema":          "3.0",
        "stub_from":       str(v2_path.name),
        "n_node_types":    N_NODE_TYPES_V3,
        "training_seed":   None,
        "training_timestamp": None,
        "training_n_epochs": 0,
    })
    arrays["_arch_json"] = np.array([json.dumps(arch_v3)], dtype=object)

    # Clés supplémentaires optionnelles
    for k in data.files:
        if k not in arrays and not k.startswith("encoder_") and not k.startswith("graph"):
            arrays[k] = data[k].copy()

    # --- Archiver v2 ---
    archive_dir = v2_path.parent / "archive_v2"
    archive_dir.mkdir(exist_ok=True)
    archive_dst = archive_dir / v2_path.name
    if not archive_dst.exists():
        shutil.copy2(v2_path, archive_dst)
        print(f"  v2 archivé → {archive_dst}")
    else:
        print(f"  v2 déjà archivé : {archive_dst}")

    # --- Sauvegarder stub v3 ---
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(out_path)
    print(f"\nStub v3 sauvegardé : {out_path}")
    print("AVERTISSEMENT : stub non-entraîné — réentraînement obligatoire avec gcn-train.")
    print(f"  Nouvelles dimensions : d_clause={D_CLAUSE_V3}, d_eff={d_eff_v3}, "
          f"n_relations={N_RELATIONS_V3}, n_nodes={N_NODE_TYPES_V3}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Checkpoint v2 source (.npz)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Chemin de sortie (défaut: stub_v3.npz à côté du source)")
    parser.add_argument("--d-emb", type=int, default=None,
                        help="Override d_emb (défaut: valeur du checkpoint v2)")
    parser.add_argument("--seed", type=int, default=42, help="Seed He-init (défaut: 42)")
    args = parser.parse_args()

    v2_path = args.checkpoint.resolve()
    if not v2_path.exists():
        sys.exit(f"Erreur : checkpoint introuvable : {v2_path}")
    out_path = args.out.resolve() if args.out else v2_path.parent / "stub_v3.npz"

    stub_v3(v2_path, out_path, d_emb_override=args.d_emb, seed=args.seed)


if __name__ == "__main__":
    main()
