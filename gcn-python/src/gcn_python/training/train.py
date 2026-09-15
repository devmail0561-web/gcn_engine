from __future__ import annotations
import csv
import json
from pathlib import Path

import click
import numpy as np

from ..layer1.features import FeatureVocabulary
from ..layer2.reference import MLPEncoder
from ..layer3.reference import RGCNLayer
from ..pipeline.cgnp import CGNPipeline
from ..taxonomy.loader import TaxonomyIndex
from ..data.loader import GCNDataLoader, reps_from_sentence
from ..constants import NODE_TYPES, RELATION_TYPES
from .checkpoint import save_checkpoint


DEFAULT_TAXONOMY_DIR = Path(__file__).parents[5] / "gcn-core" / "data" / "taxonomies"


@click.command("gcn-train")
@click.option("--data-dir", required=True, type=click.Path(path_type=Path),
              help="Répertoire contenant les fichiers YAML d'entraînement")
@click.option("--taxonomy-dir", type=click.Path(path_type=Path), default=None,
              envvar="GCN_TAXONOMY_DIR", help="Répertoire des taxonomies")
@click.option("--lang", default="fr", show_default=True)
@click.option("--epochs", default=50, show_default=True, type=int)
@click.option("--lr", default=0.001, show_default=True, type=float)
@click.option("--output", default="model.npz", show_default=True,
              type=click.Path(path_type=Path), help="Chemin du checkpoint de sortie")
@click.option("--log-csv", default=None, type=click.Path(path_type=Path),
              help="CSV des métriques par epoch (optionnel)")
def train_cmd(
    data_dir: Path,
    taxonomy_dir: Path | None,
    lang: str,
    epochs: int,
    lr: float,
    output: Path,
    log_csv: Path | None,
) -> None:
    """Entraîne le pipeline CGNP (NumPy référence) par descente de gradient."""
    if taxonomy_dir is None:
        taxonomy_dir = DEFAULT_TAXONOMY_DIR
    if not taxonomy_dir.is_dir():
        raise click.ClickException(f"Répertoire taxonomies introuvable : {taxonomy_dir}")

    tax = TaxonomyIndex.load(taxonomy_dir, lang)
    vocab = FeatureVocabulary.build(tax)
    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge)
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipeline = CGNPipeline(
        encoder=encoder, graph=graph,
        taxonomy_dir=taxonomy_dir, lang=lang, vocabulary=vocab,
    )

    loader = GCNDataLoader(data_dir, lang=lang)
    if len(loader) == 0:
        raise click.ClickException(f"Aucune sentence dans {data_dir}")

    click.echo(f"Données : {len(loader)} sentences | epochs={epochs} lr={lr}")

    history: list[dict] = []
    csv_writer = None
    csv_file = None
    if log_csv:
        csv_file = open(log_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=["epoch", "loss"])
        csv_writer.writeheader()

    try:
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_samples = 0

            for sample in loader:
                if not sample.sentence.clauses:
                    continue

                try:
                    reps, valid_clause_idxs = reps_from_sentence(sample.sentence)
                    if not reps:
                        continue
                    pipeline.forward(reps, sample.sentence.text)
                except Exception:
                    continue

                node_logits = pipeline._cached_node_logits
                edge_logits = pipeline._cached_edge_logits
                if node_logits is None or len(node_logits) == 0:
                    continue

                # Aligner les gold labels sur les seules clauses converties en reps
                if valid_clause_idxs:
                    gold_node = sample.gold_node_labels[
                        np.array(valid_clause_idxs, dtype=np.int64)
                    ]
                else:
                    gold_node = sample.gold_node_labels
                # Aligner les gold edges sur les paires consécutives prédites via edge_map
                if (valid_clause_idxs and len(valid_clause_idxs) >= 2
                        and sample.edge_map
                        and edge_logits is not None and len(edge_logits) > 0):
                    pairs = [
                        (valid_clause_idxs[k], valid_clause_idxs[k + 1])
                        for k in range(len(valid_clause_idxs) - 1)
                    ]
                    gold_edge_full = np.array(
                        [sample.edge_map.get(p, -1) for p in pairs], dtype=np.int64
                    )
                    valid_edge_mask = gold_edge_full >= 0
                    if valid_edge_mask.any():
                        valid_edge_idxs = np.where(valid_edge_mask)[0]
                        gold_edge = gold_edge_full[valid_edge_idxs]
                        edge_logits_arg = edge_logits[valid_edge_idxs]
                        pipeline.filter_edge_cache(valid_edge_idxs)
                    else:
                        gold_edge = None
                        edge_logits_arg = None
                else:
                    # Fallback : paper_examples sans tokens ou pas de valid_clause_idxs
                    gold_edge = sample.gold_edge_labels if len(sample.gold_edge_labels) > 0 else None
                    edge_logits_arg = edge_logits if edge_logits is not None and len(edge_logits) > 0 else None

                loss_val, d_node, d_edge = pipeline.loss(
                    node_logits, edge_logits_arg, gold_node, gold_edge
                )

                # Backward
                pipeline.backward(d_node, d_edge, lr=lr)

                epoch_loss += loss_val
                n_samples += 1

            avg_loss = epoch_loss / max(n_samples, 1)
            history.append({"epoch": epoch, "loss": avg_loss})

            if csv_writer:
                csv_writer.writerow({"epoch": epoch, "loss": avg_loss})

            if epoch % max(1, epochs // 10) == 0 or epoch == 1:
                click.echo(f"Epoch {epoch:4d}/{epochs}  loss={avg_loss:.4f}")
    finally:
        if csv_file:
            csv_file.close()

    save_checkpoint(pipeline, output)
    click.echo(f"Checkpoint sauvegardé : {output}")

    # Vérification : la loss doit décroître sur les 10 dernières epochs
    if len(history) >= 10:
        first = sum(r["loss"] for r in history[:5]) / 5
        last = sum(r["loss"] for r in history[-5:]) / 5
        if last < first:
            click.echo(f"Loss décroissante : {first:.4f} → {last:.4f}  ✓")
        else:
            click.echo(f"Avertissement : loss non décroissante ({first:.4f} → {last:.4f})")
