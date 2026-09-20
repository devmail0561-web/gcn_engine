# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import json
import warnings
from pathlib import Path

import click
import numpy as np

from ..data.loader import GCNDataLoader, reps_from_sentence
from ..pipeline.cgnp import CGNPipeline
from ..layer1.features import FeatureVocabulary
from ..layer2.reference import MLPEncoder
from ..layer3.reference import RGCNLayer
from ..training.checkpoint import load_checkpoint
from ..evaluation.metrics import (
    node_accuracy, node_macro_f1,
    edge_accuracy, edge_macro_f1,
    causal_graph_similarity,
)
from ..constants import NODE_TYPES, RELATION_TYPES


def run_eval(data_dir: Path, model_path: Path) -> dict:
    """Évalue le pipeline CGNP sur toutes les sentences d'un répertoire.

    Retourne un dict avec : n_samples, n_skipped, node_accuracy,
    node_macro_f1, edge_accuracy, edge_macro_f1.
    """
    vocab = FeatureVocabulary()
    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)
    load_checkpoint(pipeline, model_path, trusted=True)

    loader = GCNDataLoader(data_dir)

    all_node_preds: list[str] = []
    all_node_gold: list[str] = []
    all_edge_preds: list[str] = []
    all_edge_gold: list[str] = []
    causal_sim_scores: list[float] = []
    n_samples = 0
    n_skipped = 0

    for sample in loader:
        if not sample.sentence.clauses:
            n_skipped += 1
            continue
        try:
            reps, valid_clause_idxs, connector_reps = reps_from_sentence(sample.sentence)
            if not reps:
                n_skipped += 1
                continue
            pred_cir = pipeline.forward(
                reps, sample.sentence.text,
                clause_positions=valid_clause_idxs,
                n_total_clauses=len(sample.sentence.clauses),
                connector_reps=connector_reps,
            )
        except Exception as exc:
            warnings.warn(f"[{sample.sentence.id}] forward ignoré : {exc}", stacklevel=2)
            n_skipped += 1
            continue

        node_logits = pipeline._cached_node_logits
        edge_logits = pipeline._cached_edge_logits
        if node_logits is None or len(node_logits) == 0:
            n_skipped += 1
            continue

        if valid_clause_idxs:
            gold_node = sample.gold_node_labels[np.array(valid_clause_idxs, dtype=np.int64)]
        else:
            gold_node = sample.gold_node_labels

        node_pred_idxs = np.argmax(node_logits, axis=1)
        all_node_preds.extend(NODE_TYPES[i] for i in node_pred_idxs)
        all_node_gold.extend(NODE_TYPES[i] for i in gold_node)

        if (valid_clause_idxs and len(valid_clause_idxs) >= 2
                and sample.edge_map and edge_logits is not None and len(edge_logits) > 0):
            pairs = [
                (valid_clause_idxs[k], valid_clause_idxs[k + 1])
                for k in range(len(valid_clause_idxs) - 1)
            ]
            gold_edge_full = np.array(
                [sample.edge_map.get(p, -1) for p in pairs], dtype=np.int64
            )
            valid_mask = gold_edge_full >= 0
            if valid_mask.any():
                valid_idxs = np.where(valid_mask)[0]
                gold_edge = gold_edge_full[valid_idxs]
                edge_pred_idxs = np.argmax(edge_logits[valid_idxs], axis=1)
                all_edge_preds.extend(RELATION_TYPES[i] for i in edge_pred_idxs)
                all_edge_gold.extend(RELATION_TYPES[i] for i in gold_edge)

        # causal_graph_similarity: compare pred_cir vs gold CIR reconstructed from sample
        if pred_cir is not None:
            gold_nodes = [
                {"node_type": NODE_TYPES[int(lbl)] if int(lbl) < len(NODE_TYPES) else "action"}
                for lbl in sample.gold_node_labels
            ]
            gold_edges = [
                [src, tgt, {"relation": RELATION_TYPES[int(rel)]}]
                for (src, tgt), rel in sample.edge_map.items()
                if 0 <= int(rel) < len(RELATION_TYPES)
            ]
            gold_cir = {"nodes": gold_nodes, "edges": gold_edges}
            sim = causal_graph_similarity(pred_cir, gold_cir)
            causal_sim_scores.append(sim["overall"])

        n_samples += 1

    # BLEU and cross_modal_consistency require decoder surface output — not available here
    if n_samples > 0:
        warnings.warn(
            "run_eval: generation_bleu et cross_modal_consistency ne peuvent pas être calculés "
            "sans sortie décodeur (surface text). Utilisez un pipeline avec décodeur attaché.",
            UserWarning,
            stacklevel=2,
        )

    return {
        "n_samples": n_samples,
        "n_skipped": n_skipped,
        "node_accuracy": node_accuracy(all_node_preds, all_node_gold),
        "node_macro_f1": node_macro_f1(all_node_preds, all_node_gold),
        "edge_accuracy": edge_accuracy(all_edge_preds, all_edge_gold),
        "edge_macro_f1": edge_macro_f1(all_edge_preds, all_edge_gold),
        "causal_graph_similarity": (
            round(float(np.mean(causal_sim_scores)), 4) if causal_sim_scores else None
        ),
    }


@click.command("gcn-eval")
@click.option("--data-dir", required=True, type=click.Path(path_type=Path))
@click.option("--model-path", required=True, type=click.Path(path_type=Path))
@click.option("--output", default=None, type=click.Path(path_type=Path),
              help="Chemin JSON du rapport (optionnel, sinon stdout)")
def eval_cmd(
    data_dir: Path, model_path: Path, output: Path | None
) -> None:
    """Évalue le pipeline CGNP sur un répertoire de données annotées."""
    report = run_eval(data_dir, model_path)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if output:
        Path(output).write_text(text, encoding="utf-8")
        click.echo(f"Rapport écrit : {output}")
    else:
        click.echo(text)
