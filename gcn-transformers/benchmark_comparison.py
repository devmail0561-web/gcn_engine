#!/usr/bin/env python3
# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark Comparatif : MLPEncoder vs gcn-transformers

Compare performance, vitesse, mémoire des encodeurs.
"""
import argparse
import os
import time

import numpy as np

try:
    import psutil
except ImportError:  # dépendance optionnelle : mémoire rapportée à 0.0
    psutil = None
from dataclasses import dataclass
from pathlib import Path

from gcn_python.constants import NODE_TYPES, RELATION_TYPES
from gcn_python.data.loader import GCNDataLoader, reps_from_sentence
from gcn_python.evaluation.metrics import edge_accuracy, edge_macro_f1, node_accuracy
from gcn_python.layer1.embedding import WordEmbedding
from gcn_python.layer1.features import FeatureVocabulary
from gcn_python.layer2.reference import MLPEncoder
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.pipeline.cgnp import CGNPipeline


@dataclass
class BenchmarkResult:
    """Résultats benchmark pour un encodeur."""
    encoder_name: str
    # Métriques performance
    train_node_acc: float
    train_edge_acc: float
    train_edge_f1: float
    val_node_acc: float
    val_edge_acc: float
    val_edge_f1: float
    # Métriques temps
    init_time: float
    train_time_per_epoch: float
    inference_time_per_sample: float
    # Métriques mémoire
    memory_mb: float
    # Métriques entraînement
    epochs_trained: int
    final_train_loss: float


def get_memory_mb():
    """Retourne mémoire RSS utilisée en MB (0.0 si psutil absent)."""
    if psutil is None:
        return 0.0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def _forward_sample(pipeline, sample):
    """Forward conforme à gcn-python/training/train.py.

    Retourne (node_logits, edge_logits_filtrés, gold_node, gold_edge) ou None
    si le sample est vide. Les paires d'arêtes utilisent les indices ORIGINAUX
    des clauses (valid_clause_idxs) et le gold nœuds est aligné dessus.
    """
    reps, valid_clause_idxs, connector_reps = reps_from_sentence(sample.sentence)
    if not reps:
        return None
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
        return None
    if valid_clause_idxs:
        gold_node = sample.gold_node_labels[np.array(valid_clause_idxs, dtype=np.int64)]
    else:
        gold_node = sample.gold_node_labels
    gold_edge = None
    edge_logits_filtered = None
    valid_idx = None
    if (len(valid_clause_idxs) >= 2 and sample.edge_map
            and edge_logits is not None and len(edge_logits) > 0):
        pairs = [(valid_clause_idxs[k], valid_clause_idxs[k + 1])
                 for k in range(len(valid_clause_idxs) - 1)]
        gold_edge_full = np.array([sample.edge_map.get(p, -1) for p in pairs],
                                  dtype=np.int64)
        mask = gold_edge_full >= 0
        if mask.any():
            valid_idx = np.where(mask)[0]
            gold_edge = gold_edge_full[valid_idx]
            edge_logits_filtered = edge_logits[valid_idx]
    return node_logits, edge_logits, edge_logits_filtered, gold_node, gold_edge, valid_idx


def _sample_metrics(fwd):
    """Métriques (list[str], ordre (pred, gold)) depuis le résultat de _forward_sample."""
    node_logits, _, edge_logits_filtered, gold_node, gold_edge, _ = fwd
    pred_nodes = [NODE_TYPES[i] for i in np.argmax(node_logits, axis=1)]
    gold_nodes = [NODE_TYPES[i] for i in gold_node]
    node_acc = node_accuracy(pred_nodes, gold_nodes)
    edge_acc, edge_f1 = None, None
    if gold_edge is not None and edge_logits_filtered is not None:
        pred_edges = [RELATION_TYPES[i] for i in np.argmax(edge_logits_filtered, axis=1)]
        gold_edges = [RELATION_TYPES[i] for i in gold_edge]
        edge_acc = edge_accuracy(pred_edges, gold_edges)
        edge_f1 = edge_macro_f1(pred_edges, gold_edges)
    return node_acc, edge_acc, edge_f1


def benchmark_encoder(
    encoder_name: str,
    encoder,
    vocab: FeatureVocabulary,
    train_samples: list[dict],
    val_samples: list[dict],
    epochs: int = 5,
    lr: float = 0.001,
    d_emb: int = 0,
    word_embedding=None,
) -> BenchmarkResult:
    """
    Benchmark un encodeur.

    Args:
        encoder_name: Nom de l'encodeur
        encoder: Instance de l'encodeur
        vocab: FeatureVocabulary
        train_samples: Samples d'entraînement
        val_samples: Samples de validation
        epochs: Nombre d'epochs
        lr: Learning rate
        d_emb: Dimension embeddings lexicaux (défaut 0 = désactivés, déconseillé)
        word_embedding: Instance WordEmbedding partagée (ou None si d_emb=0)

    Returns:
        BenchmarkResult avec toutes les métriques
    """
    print(f"\n{'='*60}")
    print(f"Benchmark : {encoder_name}")
    print(f"{'='*60}")

    # Mesure mémoire avant
    mem_before = get_memory_mb()

    # Initialisation pipeline
    d_eff = vocab.d_clause_effective(d_emb=d_emb, subject_object_emb=False)
    graph = RGCNLayer(d_in=d_eff, d_out=d_eff, n_relations=11)

    init_start = time.time()
    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab,
                           word_embedding=word_embedding)
    init_time = time.time() - init_start

    # Mesure mémoire après init
    mem_after = get_memory_mb()
    memory_mb = mem_after - mem_before

    print(f"  Initialisation : {init_time:.2f}s")
    print(f"  Mémoire : {memory_mb:.1f} MB")

    # Entraînement
    print(f"\n  Entraînement ({epochs} epochs)...")
    train_times = []
    final_train_loss = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0

        for sample in train_samples:
            # Forward (boucle conforme à gcn-python/training/train.py)
            fwd = _forward_sample(pipeline, sample)
            if fwd is None:
                continue
            node_logits, edge_logits, edge_logits_filtered, gold_node, gold_edge, valid_idx = fwd

            # Loss
            loss, d_node, d_edge_filtered = pipeline.loss(
                node_logits, edge_logits_filtered,
                gold_node, gold_edge
            )
            epoch_loss += loss

            # Backward
            d_edge_full = None
            if d_edge_filtered is not None and edge_logits is not None:
                d_edge_full = np.zeros_like(edge_logits)
                d_edge_full[valid_idx] = d_edge_filtered
            pipeline.backward(d_node, d_edge_full, lr=lr)

        epoch_time = time.time() - epoch_start
        train_times.append(epoch_time)
        avg_loss = epoch_loss / len(train_samples)
        final_train_loss = avg_loss

        print(f"    Epoch {epoch+1}/{epochs} : {epoch_time:.1f}s, loss={avg_loss:.4f}")

    avg_train_time = np.mean(train_times)

    # Évaluation train
    print("\n  Évaluation train...")
    if hasattr(encoder, "eval"):
        encoder.eval()  # MLPEncoder (NumPy) n'a pas de mode eval : rien à faire

    train_node_accs = []
    train_edge_accs = []
    train_edge_f1s = []

    for sample in train_samples:
        fwd = _forward_sample(pipeline, sample)
        if fwd is None:
            continue
        node_acc, edge_acc, edge_f1 = _sample_metrics(fwd)
        train_node_accs.append(node_acc)
        if edge_acc is not None:
            train_edge_accs.append(edge_acc)
            train_edge_f1s.append(edge_f1)

    train_node_acc = float(np.mean(train_node_accs)) if train_node_accs else 0.0
    train_edge_acc = float(np.mean(train_edge_accs)) if train_edge_accs else 0.0
    train_edge_f1 = float(np.mean(train_edge_f1s)) if train_edge_f1s else 0.0

    # Évaluation validation
    print("  Évaluation validation...")

    val_node_accs = []
    val_edge_accs = []
    val_edge_f1s = []
    inference_times = []

    for sample in val_samples:
        inf_start = time.time()
        fwd = _forward_sample(pipeline, sample)
        inf_time = time.time() - inf_start
        inference_times.append(inf_time)
        if fwd is None:
            continue

        node_acc, edge_acc, edge_f1 = _sample_metrics(fwd)
        val_node_accs.append(node_acc)
        if edge_acc is not None:
            val_edge_accs.append(edge_acc)
            val_edge_f1s.append(edge_f1)

    val_node_acc = float(np.mean(val_node_accs)) if val_node_accs else 0.0
    val_edge_acc = float(np.mean(val_edge_accs)) if val_edge_accs else 0.0
    val_edge_f1 = float(np.mean(val_edge_f1s)) if val_edge_f1s else 0.0
    avg_inference_time = float(np.mean(inference_times)) * 1000 if inference_times else 0.0  # en ms

    # Résultats
    print("\n  Résultats :")
    print(f"    Train : node_acc={train_node_acc:.3f}, edge_acc={train_edge_acc:.3f}, edge_f1={train_edge_f1:.3f}")
    print(f"    Val   : node_acc={val_node_acc:.3f}, edge_acc={val_edge_acc:.3f}, edge_f1={val_edge_f1:.3f}")
    print(f"    Temps : {avg_train_time:.1f}s/epoch, {avg_inference_time:.1f}ms/sample")

    return BenchmarkResult(
        encoder_name=encoder_name,
        train_node_acc=train_node_acc,
        train_edge_acc=train_edge_acc,
        train_edge_f1=train_edge_f1,
        val_node_acc=val_node_acc,
        val_edge_acc=val_edge_acc,
        val_edge_f1=val_edge_f1,
        init_time=init_time,
        train_time_per_epoch=avg_train_time,
        inference_time_per_sample=avg_inference_time,
        memory_mb=memory_mb,
        epochs_trained=epochs,
        final_train_loss=final_train_loss
    )


def print_comparison(results: list[BenchmarkResult]):
    """Affiche tableau comparatif."""
    print(f"\n{'='*80}")
    print("COMPARAISON FINALE")
    print(f"{'='*80}\n")

    # Tableau métriques
    print("Métriques de Performance :")
    print("-" * 80)
    print(f"{'Encodeur':<30} | {'Val Edge F1':<12} | {'Val Node Acc':<12} | {'Val Edge Acc':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r.encoder_name:<30} | {r.val_edge_f1:<12.3f} | {r.val_node_acc:<12.3f} | {r.val_edge_acc:<12.3f}")
    print()

    # Tableau vitesse
    print("Métriques de Vitesse :")
    print("-" * 80)
    print(f"{'Encodeur':<30} | {'Init (s)':<12} | {'Train (s/epoch)':<15} | {'Inférence (ms)':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r.encoder_name:<30} | {r.init_time:<12.2f} | {r.train_time_per_epoch:<15.1f} | {r.inference_time_per_sample:<15.1f}")
    print()

    # Tableau mémoire
    print("Métriques de Mémoire :")
    print("-" * 80)
    print(f"{'Encodeur':<30} | {'Mémoire (MB)':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r.encoder_name:<30} | {r.memory_mb:<15.1f}")
    print()

    # Recommandations
    print("Recommandations :")
    print("-" * 80)

    best_perf = max(results, key=lambda r: r.val_edge_f1)
    fastest = min(results, key=lambda r: r.inference_time_per_sample)
    lightest = min(results, key=lambda r: r.memory_mb)

    print(f"  🏆 Meilleure performance : {best_perf.encoder_name} (edge_f1={best_perf.val_edge_f1:.3f})")
    print(f"  ⚡ Plus rapide          : {fastest.encoder_name} ({fastest.inference_time_per_sample:.1f}ms/sample)")
    print(f"  💾 Plus léger          : {lightest.encoder_name} ({lightest.memory_mb:.1f} MB)")
    print()

    # Cas d'usage
    print("Cas d'Usage :")
    print("-" * 80)
    print(f"  • Prototypage rapide    → {lightest.encoder_name}")
    print(f"  • Production temps réel → {fastest.encoder_name}")
    print(f"  • Recherche SOTA        → {best_perf.encoder_name}")
    print(f"  • Ressources limitées   → {lightest.encoder_name}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark MLPEncoder vs gcn-transformers")
    parser.add_argument("--train-dir", type=Path, required=True,
                       help="Répertoire données train")
    parser.add_argument("--val-dir", type=Path, required=True,
                       help="Répertoire données validation")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Nombre d'epochs (défaut: 5)")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate (défaut: 0.001)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Nombre max de samples (défaut: tous)")
    parser.add_argument("--encoders", type=str, default="mlp,xlmroberta",
                       help="Encodeurs à tester (mlp,xlmroberta,camembert,codebert)")
    parser.add_argument("--embedding-dim", type=int, default=128,
                       help="Dimension embeddings lexicaux, ACTIVÉS PAR DÉFAUT "
                            "(défaut: 128, comme gcn-train). 0 = désactivé (déconseillé)")

    args = parser.parse_args()

    # Vérifier répertoires
    if not args.train_dir.exists():
        print(f"ERREUR : {args.train_dir} n'existe pas")
        return 1
    if not args.val_dir.exists():
        print(f"ERREUR : {args.val_dir} n'existe pas")
        return 1

    print("="*80)
    print("BENCHMARK COMPARATIF : MLPEncoder vs gcn-transformers")
    print("="*80)
    print("\nConfiguration :")
    print(f"  Train dir : {args.train_dir}")
    print(f"  Val dir   : {args.val_dir}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  LR        : {args.lr}")
    print(f"  Encodeurs : {args.encoders}")
    print(f"  Embeddings lexicaux : {'d_emb=%d (activés)' % args.embedding_dim if args.embedding_dim > 0 else 'DÉSACTIVÉS (déconseillé)'}")

    # Charger données
    print("\nChargement données...")
    train_loader = GCNDataLoader(args.train_dir)
    train_samples = list(train_loader)

    val_loader = GCNDataLoader(args.val_dir)
    val_samples = list(val_loader)

    if args.max_samples:
        train_samples = train_samples[:args.max_samples]
        val_samples = val_samples[:args.max_samples]

    print(f"  Train : {len(train_samples)} samples")
    print(f"  Val   : {len(val_samples)} samples")

    # Vocabulaire + embeddings partagés (même défaut que gcn-train)
    vocab = FeatureVocabulary()
    d_emb = args.embedding_dim
    d_eff = vocab.d_clause_effective(d_emb=d_emb, subject_object_emb=False)
    d_edge = vocab.d_edge_closed_loop(d_eff, n_node_types=len(NODE_TYPES), d_emb=d_emb,
                                      subject_object_emb=False)
    word_embedding = WordEmbedding(d_emb=d_emb) if d_emb > 0 else None
    if word_embedding is not None:
        _lemmas = []
        for s in train_samples:
            _reps, _, _ = reps_from_sentence(s.sentence)
            _lemmas.extend(r.root_lemma for r in _reps)
        word_embedding.build_vocab(_lemmas)
        print(f"  Vocab embeddings : {len(word_embedding._lemmas)} lemmes")

    print(f"  Dimensions : d_clause={d_eff} (dont d_emb={d_emb}), d_edge={d_edge}")

    # Encodeurs à tester
    encoders_to_test = args.encoders.split(',')
    results = []

    # MLPEncoder
    if 'mlp' in encoders_to_test:
        encoder = MLPEncoder(d_clause=d_eff, d_edge=d_edge)
        result = benchmark_encoder(
            "MLPEncoder",
            encoder,
            vocab,
            train_samples,
            val_samples,
            epochs=args.epochs,
            lr=args.lr,
            d_emb=d_emb,
            word_embedding=word_embedding
        )
        results.append(result)

    # XLMRobertaEncoder
    if 'xlmroberta' in encoders_to_test:
        try:
            from gcn_transformers import XLMRobertaEncoder
            encoder = XLMRobertaEncoder(d_clause=d_eff, d_edge=d_edge, learning_rate=1e-5)
            result = benchmark_encoder(
                "XLMRobertaEncoder",
                encoder,
                vocab,
                train_samples,
                val_samples,
                epochs=args.epochs,
                lr=1e-5,  # AdamW lr
                d_emb=d_emb,
                word_embedding=word_embedding
            )
            results.append(result)
        except ImportError:
            print("\n⚠️  XLMRobertaEncoder non disponible (pip install gcn-transformers)")

    # CamembertEncoder
    if 'camembert' in encoders_to_test:
        try:
            from gcn_transformers import CamembertEncoder
            encoder = CamembertEncoder(d_clause=d_eff, d_edge=d_edge, learning_rate=1e-5)
            result = benchmark_encoder(
                "CamembertEncoder",
                encoder,
                vocab,
                train_samples,
                val_samples,
                epochs=args.epochs,
                lr=1e-5,
                d_emb=d_emb,
                word_embedding=word_embedding
            )
            results.append(result)
        except ImportError:
            print("\n⚠️  CamembertEncoder non disponible (pip install gcn-transformers)")

    # CodeBERTEncoder
    if 'codebert' in encoders_to_test:
        try:
            from gcn_transformers import CodeBERTEncoder
            encoder = CodeBERTEncoder(d_clause=d_eff, d_edge=d_edge, learning_rate=1e-5)
            result = benchmark_encoder(
                "CodeBERTEncoder",
                encoder,
                vocab,
                train_samples,
                val_samples,
                epochs=args.epochs,
                lr=1e-5,
                d_emb=d_emb,
                word_embedding=word_embedding
            )
            results.append(result)
        except ImportError:
            print("\n⚠️  CodeBERTEncoder non disponible (pip install gcn-transformers)")

    # Comparaison finale
    if len(results) > 1:
        print_comparison(results)
    elif len(results) == 1:
        print(f"\n✅ Benchmark terminé : {results[0].encoder_name}")
    else:
        print("\n❌ Aucun encodeur testé")

    return 0


if __name__ == "__main__":
    exit(main())
