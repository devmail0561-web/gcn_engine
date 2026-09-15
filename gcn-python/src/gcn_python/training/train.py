from __future__ import annotations
import csv
import json
import warnings
from pathlib import Path

import click
import numpy as np

from ..layer1.features import FeatureVocabulary
from ..layer2.reference import MLPEncoder
from ..layer3.reference import RGCNLayer
from ..pipeline.cgnp import CGNPipeline
from ..data.loader import GCNDataLoader, reps_from_sentence
from ..constants import NODE_TYPES, RELATION_TYPES
from ..evaluation.metrics import node_accuracy, node_macro_f1, edge_accuracy, edge_macro_f1
from ..evaluation.recorder import TrainingRecorder
from .checkpoint import save_checkpoint


@click.command("gcn-train")
@click.option("--data-dir", required=True, type=click.Path(path_type=Path),
              help="Répertoire contenant les fichiers JSON d'entraînement")
@click.option("--lang", default="fr", show_default=True)
@click.option("--epochs", default=50, show_default=True, type=int)
@click.option("--lr", default=0.001, show_default=True, type=float)
@click.option("--output", default="model.npz", show_default=True,
              type=click.Path(path_type=Path), help="Chemin du checkpoint de sortie")
@click.option("--log-csv", default=None, type=click.Path(path_type=Path),
              help="CSV des métriques par epoch (optionnel)")
@click.option("--verbalize-dir", default=None, type=click.Path(path_type=Path),
              help="Répertoire contenant les paires verbalize JSON (optionnel, active l'entraînement conjoint)")
def train_cmd(
    data_dir: Path,
    lang: str,
    epochs: int,
    lr: float,
    output: Path,
    log_csv: Path | None,
    verbalize_dir: Path | None,
) -> None:
    """Entraîne le pipeline CGNP (NumPy référence) par descente de gradient."""
    from ..data.verbalize_loader import VerbalizerDataLoader
    from ..verbalizer.trainable import TrainableDecoder

    vocab = FeatureVocabulary()
    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge)
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)

    verb_loader: VerbalizerDataLoader | None = None
    verb_source_map: dict[str, list] = {}
    decoder: TrainableDecoder | None = None
    if verbalize_dir is not None:
        verb_loader = VerbalizerDataLoader(verbalize_dir)
        if len(verb_loader) == 0:
            raise click.ClickException(f"Aucune paire verbalize dans {verbalize_dir}")
        decoder = TrainableDecoder(verb_loader.vocab)
        verb_source_map = verb_loader.source_text_map()
        click.echo(f"Verbalize : {len(verb_loader)} paires | vocab={len(verb_loader.vocab)} tokens")

    pipeline = CGNPipeline(encoder=encoder, graph=graph, lang=lang, vocabulary=vocab,
                           decoder=decoder)

    loader = GCNDataLoader(data_dir, lang=lang)
    if len(loader) == 0:
        raise click.ClickException(f"Aucune sentence dans {data_dir}")

    click.echo(f"Données : {len(loader)} sentences | epochs={epochs} lr={lr}")

    recorder = TrainingRecorder()
    history: list[dict] = []
    csv_writer = None
    csv_file = None
    if log_csv:
        csv_file = open(log_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=["epoch", "loss", "node_accuracy", "node_macro_f1",
                        "edge_accuracy", "edge_macro_f1"],
        )
        csv_writer.writeheader()

    try:
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_samples = 0
            epoch_node_preds: list[str] = []
            epoch_node_gold: list[str] = []
            epoch_edge_preds: list[str] = []
            epoch_edge_gold: list[str] = []

            for sample in loader:
                if not sample.sentence.clauses:
                    continue

                try:
                    reps, valid_clause_idxs, connector_reps = reps_from_sentence(sample.sentence)
                    if not reps:
                        continue
                    pipeline.forward(
                        reps, sample.sentence.text,
                        clause_positions=valid_clause_idxs,
                        n_total_clauses=len(sample.sentence.clauses),
                        connector_reps=connector_reps,
                    )
                except ValueError:
                    raise  # misconfiguration (d_out, clause_positions…) — non ignorable
                except Exception as exc:
                    warnings.warn(
                        f"[{sample.sentence.id}] forward ignoré : "
                        f"{type(exc).__name__}: {exc}",
                        UserWarning, stacklevel=2,
                    )
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
                    gold_edge = None
                    edge_logits_arg = None

                # Joint training : chercher une surface gold pour ce sample
                _gold_surface = None
                if verb_source_map:
                    _surfaces = verb_source_map.get(sample.sentence.text, [])
                    if _surfaces:
                        _gold_surface = _surfaces[0]

                loss_val, d_node, d_edge = pipeline.loss(
                    node_logits, edge_logits_arg, gold_node, gold_edge,
                    gold_surface=_gold_surface,
                )

                # Backward
                pipeline.backward(d_node, d_edge, lr=lr)

                # Accumuler les prédictions pour les métriques de l'époque
                node_pred_idxs = np.argmax(node_logits, axis=1)
                if valid_clause_idxs:
                    gold_node_aligned = sample.gold_node_labels[
                        np.array(valid_clause_idxs, dtype=np.int64)
                    ]
                else:
                    gold_node_aligned = sample.gold_node_labels
                epoch_node_preds.extend(NODE_TYPES[i] for i in node_pred_idxs)
                epoch_node_gold.extend(NODE_TYPES[i] for i in gold_node_aligned)

                if gold_edge is not None and edge_logits_arg is not None and len(edge_logits_arg) > 0:
                    edge_pred_idxs = np.argmax(edge_logits_arg, axis=1)
                    epoch_edge_preds.extend(RELATION_TYPES[i] for i in edge_pred_idxs)
                    epoch_edge_gold.extend(RELATION_TYPES[i] for i in gold_edge)

                epoch_loss += loss_val
                n_samples += 1

            # Entraînement standalone du décodeur sur les paires verbalize
            if verb_loader is not None and pipeline.decoder is not None:
                for vsample in verb_loader:
                    if len(vsample.gold_tokens) == 0:
                        continue
                    dec_logits = pipeline.decoder.forward_decode(vsample.node_type_embeddings)
                    dec_loss, d_dec = pipeline.decoder.loss_decode(dec_logits, vsample.gold_tokens)
                    if not np.isfinite(dec_loss):
                        continue
                    _, dec_grads = pipeline.decoder.backward_decode(d_dec)
                    pipeline.decoder.update(dec_grads, lr)
                    epoch_loss += dec_loss
                    n_samples += 1

            avg_loss = epoch_loss / max(n_samples, 1)
            metrics = {
                "node_accuracy": node_accuracy(epoch_node_preds, epoch_node_gold),
                "node_macro_f1": node_macro_f1(epoch_node_preds, epoch_node_gold),
                "edge_accuracy": edge_accuracy(epoch_edge_preds, epoch_edge_gold),
                "edge_macro_f1": edge_macro_f1(epoch_edge_preds, epoch_edge_gold),
            }
            recorder.record(epoch, avg_loss, metrics)
            history.append({"epoch": epoch, "loss": avg_loss, **metrics})

            if csv_writer:
                csv_writer.writerow({"epoch": epoch, "loss": avg_loss, **metrics})

            if epoch % max(1, epochs // 10) == 0 or epoch == 1:
                click.echo(
                    f"Epoch {epoch:4d}/{epochs}  loss={avg_loss:.4f}"
                    f"  node_acc={metrics['node_accuracy']:.3f}"
                    f"  edge_acc={metrics['edge_accuracy']:.3f}"
                )
    finally:
        if csv_file:
            csv_file.close()

    save_checkpoint(pipeline, output)
    click.echo(f"Checkpoint sauvegardé : {output}")

    if log_csv:
        json_path = Path(log_csv).with_suffix(".json")
        recorder.to_json(json_path)
        click.echo(f"Courbe d'entraînement : {json_path}")

    # Vérification : la loss doit décroître sur les 10 dernières epochs
    if len(history) >= 10:
        first = sum(r["loss"] for r in history[:5]) / 5
        last = sum(r["loss"] for r in history[-5:]) / 5
        if last < first:
            click.echo(f"Loss décroissante : {first:.4f} → {last:.4f}  ✓")
        else:
            click.echo(f"Avertissement : loss non décroissante ({first:.4f} → {last:.4f})")
