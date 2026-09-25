#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
Exemple d'entraînement avec gcn-transformers (XLM-RoBERTa).

Usage:
    python example_train.py --data-dir gcn-datasets/real/train \
                            --val-dir gcn-datasets/real/val \
                            --epochs 20 \
                            --output model_xlmroberta.npz
"""
import argparse
from pathlib import Path

import numpy as np
from gcn_python.constants import NODE_TYPES, RELATION_TYPES
from gcn_python.data.loader import GCNDataLoader, reps_from_sentence
from gcn_python.evaluation.metrics import edge_accuracy, edge_macro_f1, node_accuracy
from gcn_python.layer1.embedding import WordEmbedding
from gcn_python.layer1.features import FeatureVocabulary
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.pipeline.cgnp import CGNPipeline
from gcn_python.training.checkpoint import save_checkpoint

from gcn_transformers import XLMRobertaEncoder


def train(args):
    """Entraînement complet avec XLM-RoBERTa."""
    print("=== Initialisation ===")

    # Vocabulaire + embeddings NON OPTIONNELS (REMEDIATION-DIAGNOSTIC.md §2/§11) :
    # même défaut que gcn-train (--embedding-dim 128). 0 = désactivé (déconseillé).
    vocab = FeatureVocabulary()
    d_emb = args.embedding_dim
    d_eff = vocab.d_clause_effective(d_emb=d_emb, subject_object_emb=False)
    d_edge = vocab.d_edge_closed_loop(d_eff, n_node_types=len(NODE_TYPES), d_emb=d_emb,
                                      subject_object_emb=False)
    print(f"Dimensions : d_clause={d_eff} (dont d_emb={d_emb}), d_edge={d_edge}")

    # Encodeur XLM-RoBERTa
    print(f"Chargement XLM-RoBERTa (freeze_layers={args.freeze_layers})...")
    encoder = XLMRobertaEncoder(
        d_clause=d_eff,
        d_edge=d_edge,
        freeze_layers=args.freeze_layers,
        learning_rate=args.lr,
        device=args.device,
    )

    # Graphe R-GCN
    graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=11)

    # Pipeline (avec embeddings lexicaux sauf --embedding-dim 0)
    word_embedding = WordEmbedding(d_emb=d_emb) if d_emb > 0 else None
    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab,
                           word_embedding=word_embedding)
    print(f"Pipeline initialisé (embeddings : {'activés d_emb=%d' % d_emb if word_embedding else 'DÉSACTIVÉS — déconseillé'})\n")

    # Charger données
    print(f"Chargement données train : {args.data_dir}")
    train_loader = GCNDataLoader(args.data_dir)
    train_samples = list(train_loader)
    print(f"  → {len(train_samples)} samples train")

    val_samples = []
    if args.val_dir:
        print(f"Chargement données val : {args.val_dir}")
        val_loader = GCNDataLoader(args.val_dir)
        val_samples = list(val_loader)
        print(f"  → {len(val_samples)} samples val")

    print()

    # Vocabulaire d'embeddings : lemmes racines du train (cf train.py:511-559)
    if word_embedding is not None:
        _lemmas = []
        for s in train_samples:
            _reps, _, _ = reps_from_sentence(s.sentence)
            _lemmas.extend(r.root_lemma for r in _reps)
        word_embedding.build_vocab(_lemmas)
        print(f"  → vocab embeddings : {len(word_embedding._lemmas)} lemmes\n")

    # Entraînement
    print("=== Entraînement ===")
    best_val_f1 = 0.0

    for epoch in range(args.epochs):
        # Train
        encoder.train()
        total_loss = 0.0
        for sample in train_samples:
            # Forward (boucle conforme à gcn-python/training/train.py:775-860)
            reps, valid_clause_idxs, connector_reps = reps_from_sentence(sample.sentence)
            if not reps:
                continue
            pipeline.forward(
                reps, sample.sentence.text,
                clause_positions=valid_clause_idxs,
                n_total_clauses=len(sample.sentence.clauses),
                connector_reps=connector_reps,
                gold_edge_map=sample.edge_map,
            )

            # Récupérer logits
            node_logits = pipeline._cached_node_logits
            edge_logits = pipeline._cached_edge_logits
            if node_logits is None or len(node_logits) == 0:
                continue

            # Aligner le gold sur les clauses valides (train.py:903-908)
            if valid_clause_idxs:
                gold_node = sample.gold_node_labels[np.array(valid_clause_idxs, dtype=np.int64)]
            else:
                gold_node = sample.gold_node_labels

            # Filtrer arêtes gold (paires consécutives, indices ORIGINAUX des clauses)
            gold_edge = None
            edge_logits_filtered = None
            valid_mask = None
            if (len(valid_clause_idxs) >= 2 and sample.edge_map
                    and edge_logits is not None and len(edge_logits) > 0):
                pairs = [(valid_clause_idxs[k], valid_clause_idxs[k + 1])
                         for k in range(len(valid_clause_idxs) - 1)]
                gold_edge_full = np.array([sample.edge_map.get(p, -1) for p in pairs],
                                          dtype=np.int64)
                valid_mask = gold_edge_full >= 0
                if valid_mask.any():
                    gold_edge = gold_edge_full[valid_mask]
                    edge_logits_filtered = edge_logits[np.where(valid_mask)[0]]

            # Loss
            loss, d_node, d_edge_filtered = pipeline.loss(
                node_logits, edge_logits_filtered,
                gold_node, gold_edge
            )
            total_loss += loss

            # Backward
            d_edge_full = None
            if d_edge_filtered is not None and edge_logits is not None:
                d_edge_full = np.zeros_like(edge_logits)
                d_edge_full[np.where(valid_mask)[0]] = d_edge_filtered
            pipeline.backward(d_node, d_edge_full, lr=args.lr)

        avg_loss = total_loss / len(train_samples)

        # Validation
        val_metrics = ""
        if val_samples:
            encoder.eval()
            val_node_acc = []
            val_edge_acc = []
            val_edge_f1_scores = []

            for sample in val_samples:
                reps, valid_clause_idxs, connector_reps = reps_from_sentence(sample.sentence)
                if not reps:
                    continue
                pipeline.forward(
                    reps, sample.sentence.text,
                    clause_positions=valid_clause_idxs,
                    n_total_clauses=len(sample.sentence.clauses),
                    connector_reps=connector_reps,
                    gold_edge_map=sample.edge_map,
                )
                node_logits = pipeline._cached_node_logits
                edge_logits = pipeline._cached_edge_logits
                if node_logits is None or len(node_logits) == 0:
                    continue
                if valid_clause_idxs:
                    gold_node = sample.gold_node_labels[np.array(valid_clause_idxs, dtype=np.int64)]
                else:
                    gold_node = sample.gold_node_labels

                # Métriques nœuds : listes de labels (metrics.py attend list[str], ordre (pred, gold))
                pred_nodes = [NODE_TYPES[i] for i in np.argmax(node_logits, axis=1)]
                gold_nodes = [NODE_TYPES[i] for i in gold_node]
                val_node_acc.append(node_accuracy(pred_nodes, gold_nodes))

                # Métriques arêtes : idem, sur paires consécutives valides
                if (len(valid_clause_idxs) >= 2 and sample.edge_map
                        and edge_logits is not None and len(edge_logits) > 0):
                    pairs = [(valid_clause_idxs[k], valid_clause_idxs[k + 1])
                             for k in range(len(valid_clause_idxs) - 1)]
                    gold_edge_full = np.array([sample.edge_map.get(p, -1) for p in pairs],
                                              dtype=np.int64)
                    mask = gold_edge_full >= 0
                    if mask.any():
                        idx = np.where(mask)[0]
                        pred_edges = [RELATION_TYPES[i]
                                      for i in np.argmax(edge_logits[idx], axis=1)]
                        gold_edges = [RELATION_TYPES[i] for i in gold_edge_full[idx]]
                        val_edge_acc.append(edge_accuracy(pred_edges, gold_edges))
                        val_edge_f1_scores.append(edge_macro_f1(pred_edges, gold_edges))

            avg_node_acc = np.mean(val_node_acc)
            avg_edge_acc = np.mean(val_edge_acc)
            avg_edge_f1 = np.mean(val_edge_f1_scores)

            val_metrics = (f" | Val: node_acc={avg_node_acc:.3f} "
                          f"edge_acc={avg_edge_acc:.3f} edge_f1={avg_edge_f1:.3f}")

            # Sauvegarder meilleur modèle
            if avg_edge_f1 > best_val_f1:
                best_val_f1 = avg_edge_f1
                save_checkpoint(pipeline, args.output)
                val_metrics += " [SAVED]"

        print(f"Epoch {epoch+1:3d}/{args.epochs} — Loss: {avg_loss:.4f}{val_metrics}")

    print("\n=== Entraînement terminé ===")
    print(f"Meilleur val_edge_f1 : {best_val_f1:.3f}")
    print(f"Checkpoint sauvegardé : {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Entraînement gcn-transformers")
    parser.add_argument("--data-dir", type=Path, required=True,
                       help="Répertoire contenant les fichiers JSON d'entraînement")
    parser.add_argument("--val-dir", type=Path, default=None,
                       help="Répertoire contenant les fichiers JSON de validation")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Nombre d'epochs (défaut: 20)")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate AdamW (défaut: 1e-5)")
    parser.add_argument("--freeze-layers", type=int, default=10,
                       help="Nombre de couches Transformer à geler (défaut: 10/12)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device PyTorch : 'cuda', 'cpu', ou None (auto)")
    parser.add_argument("--embedding-dim", type=int, default=128,
                       help="Dimension embeddings lexicaux apprenables, ACTIVÉS PAR DÉFAUT "
                            "(défaut: 128, comme gcn-train). 0 = désactivé (déconseillé)")
    parser.add_argument("--output", type=Path, default="model_xlmroberta.npz",
                       help="Chemin du checkpoint de sortie (défaut: model_xlmroberta.npz)")

    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"ERREUR : {args.data_dir} n'existe pas")
        return 1

    if args.val_dir and not args.val_dir.exists():
        print(f"ERREUR : {args.val_dir} n'existe pas")
        return 1

    train(args)
    return 0


if __name__ == "__main__":
    exit(main())
