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
from ..evaluation.metrics import (
    node_accuracy, node_macro_f1, edge_accuracy, edge_macro_f1,
    graph_exact_match as _gem,
)
from ..evaluation.recorder import TrainingRecorder
from .checkpoint import save_checkpoint


@click.command("gcn-train")
@click.option("--data-dir", required=True, type=click.Path(path_type=Path),
              help="Répertoire contenant les fichiers JSON d'entraînement")
@click.option("--epochs", default=50, show_default=True, type=int)
@click.option("--lr", default=0.001, show_default=True, type=float)
@click.option("--output", default="model.npz", show_default=True,
              type=click.Path(path_type=Path), help="Chemin du checkpoint de sortie")
@click.option("--log-csv", default=None, type=click.Path(path_type=Path),
              help="CSV des métriques par epoch (optionnel)")
@click.option("--verbalize-dir", default=None, type=click.Path(path_type=Path),
              help="Répertoire contenant les paires verbalize JSON (optionnel, active l'entraînement conjoint)")
@click.option("--decoder-only", is_flag=True, default=False,
              help="Entraîne uniquement le décodeur (l'encodeur et le R-GCN sont gelés)")
@click.option("--encoder-checkpoint", default=None, type=click.Path(path_type=Path),
              help="Checkpoint à charger pour initialiser l'encodeur (utilisé avec --decoder-only)")
@click.option("--all-pairs/--no-all-pairs", default=False, show_default=True,
              help="Superviser toutes les paires (i,j) avec i<j, pas seulement consécutives. "
                   "Requiert dataset re-annoté avec des arêtes gap>1.")
@click.option("--embedding-dim", default=0, show_default=True, type=int,
              help="Activer les word embeddings apprenables (S1/S2). 0 = désactivé.")
@click.option("--embedding-file", default=None, type=click.Path(path_type=Path),
              help="Fichier GloVe/FastText pour initialiser les embeddings (S9). "
                   "Active automatiquement --embedding-dim si non précisé.")
@click.option("--mini-batch-size", default=1, show_default=True, type=int,
              help="Taille du mini-batch pour accumulation de gradients (S10). 1 = SGD standard.")
@click.option("--use-attention/--no-attention", default=False, show_default=True,
              help="Activer la couche R-GCN+GAT avec attention par relation.")
@click.option("--bidirectional/--no-bidirectional", default=False, show_default=True,
              help="Activer le message passing bidirectionnel (arêtes inverses, 22 types de relations).")
@click.option("--edge-loss-weight", default=1.0, show_default=True, type=float,
              help="Pondération relative de la loss arêtes dans la loss totale.")
@click.option("--weighted-loss/--no-weighted-loss", default=False, show_default=True,
              help="Activer les class weights inversement proportionnels à la fréquence (rééquilibre les classes rares).")
@click.option("--val-dir", default=None, type=click.Path(path_type=Path),
              help="Répertoire val JSON (optionnel). Métriques val calculées à chaque epoch.")
@click.option("--patience", default=0, show_default=True, type=int,
              help="Epochs sans amélioration de val_node_macro_f1 avant arrêt. "
                   "0 = désactivé (défaut). Requiert --val-dir.")
@click.option("--weight-decay", default=0.0, show_default=True, type=float,
              help="Coefficient de régularisation L2 sur les poids MLP. 0 = désactivé.")
@click.option("--rgcn-dropout", default=0.0, show_default=True, type=float,
              help="Dropout sur les features d'entrée des couches R-GCN/GAT. 0 = désactivé.")
@click.option("--label-smoothing", default=0.0, show_default=True, type=float,
              help="Lissage des labels [0, 1]. 0 = one-hot strict. Recommandé : 0.05–0.1.")
def train_cmd(
    data_dir: Path,
    epochs: int,
    lr: float,
    output: Path,
    log_csv: Path | None,
    verbalize_dir: Path | None,
    decoder_only: bool,
    encoder_checkpoint: Path | None,
    all_pairs: bool,
    embedding_dim: int,
    embedding_file: Path | None,
    mini_batch_size: int,
    use_attention: bool,
    bidirectional: bool,
    edge_loss_weight: float,
    weighted_loss: bool,
    val_dir: Path | None,
    patience: int,
    weight_decay: float,
    rgcn_dropout: float,
    label_smoothing: float,
) -> None:
    """Entraîne le pipeline CGNP (NumPy référence) par descente de gradient."""
    from ..data.verbalize_loader import VerbalizerDataLoader
    from ..verbalizer.trainable import TrainableDecoder

    vocab = FeatureVocabulary()

    # S1/S2/S9 : word embeddings optionnels
    word_embedding = None
    d_emb = 0
    if embedding_file is not None or embedding_dim > 0:
        from ..layer1.embedding import WordEmbedding
        if embedding_file is not None and embedding_dim == 0:
            # Détecter la dimension depuis la première ligne du fichier
            with open(embedding_file, encoding="utf-8") as ef:
                for line in ef:
                    parts = line.strip().split()
                    if len(parts) > 2:
                        d_emb = len(parts) - 1
                        break
        else:
            d_emb = embedding_dim
        if d_emb > 0:
            word_embedding = WordEmbedding(d_emb=d_emb)
            if embedding_file is not None:
                n_loaded = word_embedding.load_from_file(str(embedding_file))
                click.echo(f"Embeddings : {n_loaded} vecteurs chargés (d_emb={d_emb})")

    d_effective = vocab.d_clause + d_emb
    # Closed-loop : edge MLP reçoit features + enriched vectors + node probs
    n_node_types = len(NODE_TYPES)
    d_edge_closed = vocab.d_edge_closed_loop(d_effective, n_node_types, d_emb)
    encoder = MLPEncoder(d_clause=d_effective, d_edge=d_edge_closed, weight_decay=weight_decay)

    # Couche 3 : choix du graph selon les flags
    n_rel = 22 if bidirectional else len(RELATION_TYPES)
    if use_attention:
        from ..layer3.gat import RGCNLayerGAT
        graph = RGCNLayerGAT(d_in=d_effective, d_out=d_effective, n_relations=n_rel,
                             dropout=rgcn_dropout)
    else:
        graph = RGCNLayer(d_in=d_effective, d_out=d_effective, n_relations=n_rel,
                          dropout=rgcn_dropout)

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

    if decoder_only and decoder is None:
        raise click.ClickException("--decoder-only requiert --verbalize-dir")

    if patience > 0 and val_dir is None:
        raise click.ClickException("--patience requiert --val-dir")

    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab,
                           decoder=decoder, all_pairs=all_pairs, word_embedding=word_embedding,
                           bidirectional=bidirectional)

    if encoder_checkpoint is not None:
        from .checkpoint import load_checkpoint
        load_checkpoint(pipeline, encoder_checkpoint)
        click.echo(f"Checkpoint encodeur chargé : {encoder_checkpoint}")

    loader = GCNDataLoader(data_dir, all_pairs=all_pairs, shuffle=True)
    if len(loader) == 0:
        raise click.ClickException(f"Aucune sentence dans {data_dir}")

    val_loader = None
    if val_dir is not None:
        val_loader = GCNDataLoader(val_dir, all_pairs=all_pairs, shuffle=False)
        if len(val_loader) == 0:
            raise click.ClickException(f"Aucune sentence dans {val_dir}")
        click.echo(f"Val : {len(val_loader)} sentences")

    # Calcul des class weights si --weighted-loss (inversement proportionnel à la fréquence)
    node_class_weights = None
    edge_class_weights = None
    if weighted_loss:
        from collections import Counter
        node_counts = Counter()
        edge_counts = Counter()
        for sample in loader:
            for label in sample.gold_node_labels:
                node_counts[int(label)] += 1
            for (src, tgt), rel in sample.edge_map.items():
                edge_counts[int(rel)] += 1
        if node_counts:
            total_nodes = sum(node_counts.values())
            n_node_classes = len(NODE_TYPES)
            node_class_weights = np.zeros(n_node_classes, dtype=np.float32)
            for c in range(n_node_classes):
                count = node_counts.get(c, 1)
                node_class_weights[c] = total_nodes / (n_node_classes * count)
            click.echo(f"Node class weights : {dict(zip(NODE_TYPES, node_class_weights.round(3)))}")
        if edge_counts:
            total_edges = sum(edge_counts.values())
            n_edge_classes = len(RELATION_TYPES)
            edge_class_weights = np.zeros(n_edge_classes, dtype=np.float32)
            for c in range(n_edge_classes):
                count = edge_counts.get(c, 1)
                edge_class_weights[c] = total_edges / (n_edge_classes * count)
            click.echo(f"Edge class weights : {dict(zip(RELATION_TYPES, edge_class_weights.round(3)))}")

    # S2 : pré-remplir le vocabulaire des embeddings depuis toutes les lemmes
    if word_embedding is not None:
        all_lemmas = [
            r.root_lemma
            for sample in loader
            for r in reps_from_sentence(sample.sentence)[0]
        ]
        word_embedding.build_vocab(all_lemmas)
        click.echo(f"Embeddings vocab : {len(all_lemmas)} lemmes ({len(set(all_lemmas))} uniques)")

    click.echo(f"Données : {len(loader)} sentences | epochs={epochs} lr={lr}")

    recorder = TrainingRecorder()
    history: list[dict] = []
    csv_writer = None
    csv_file = None
    if log_csv:
        csv_fieldnames = [
            "epoch", "loss", "node_accuracy", "node_macro_f1",
            "edge_accuracy", "edge_macro_f1", "graph_exact_match",
        ]
        if val_loader is not None:
            csv_fieldnames.extend([
                "val_loss", "val_node_accuracy", "val_node_macro_f1",
                "val_edge_accuracy", "val_edge_macro_f1", "val_graph_exact_match",
            ])
        csv_file = open(log_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
        csv_writer.writeheader()

    best_val_f1 = -1.0
    best_epoch_num = 0
    best_checkpoint_path = str(output) + ".best.npz"
    _patience_counter = 0

    def _set_training_mode(pipeline: CGNPipeline, training: bool) -> None:
        """Bascule TOUS les composants avec dropout en mode eval ou train."""
        if hasattr(pipeline.encoder, 'training'):
            pipeline.encoder.training = training
        for layer in pipeline._graph_layers:
            if hasattr(layer, 'training'):
                layer.training = training

    def _run_eval_pass(pipeline, loader, epoch_node_preds, epoch_node_gold,
                       epoch_edge_preds, epoch_edge_gold,
                       epoch_sent_node_preds, epoch_sent_node_gold,
                       epoch_sent_edge_preds, epoch_sent_edge_gold):
        """Exécute un pass forward sur le val set et retourne les métriques."""
        total_loss = 0.0
        n = 0
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
            except (ValueError, RuntimeError, IndexError, KeyError, TypeError):
                continue

            node_logits = pipeline._cached_node_logits
            edge_logits = pipeline._cached_edge_logits
            if node_logits is None or len(node_logits) == 0:
                continue

            if valid_clause_idxs:
                gold_node = sample.gold_node_labels[
                    np.array(valid_clause_idxs, dtype=np.int64)
                ]
            else:
                gold_node = sample.gold_node_labels

            gold_edge = None
            edge_logits_arg = None
            if (valid_clause_idxs and len(valid_clause_idxs) >= 2
                    and sample.edge_map
                    and edge_logits is not None and len(edge_logits) > 0):
                if all_pairs:
                    pairs = [
                        (valid_clause_idxs[i], valid_clause_idxs[j])
                        for i in range(len(valid_clause_idxs))
                        for j in range(i + 1, len(valid_clause_idxs))
                    ]
                else:
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

            loss_val, _, _ = pipeline.loss(
                node_logits, edge_logits_arg, gold_node, gold_edge,
                edge_loss_weight=edge_loss_weight,
                node_class_weights=node_class_weights,
                edge_class_weights=edge_class_weights,
            )
            total_loss += loss_val
            n += 1

            node_pred_idxs = np.argmax(node_logits, axis=1)
            epoch_node_preds.extend(NODE_TYPES[i] for i in node_pred_idxs)
            epoch_node_gold.extend(NODE_TYPES[i] for i in gold_node)

            sent_node_pred = [NODE_TYPES[i] for i in node_pred_idxs]
            sent_node_gold = [NODE_TYPES[i] for i in gold_node]
            epoch_sent_node_preds.append(sent_node_pred)
            epoch_sent_node_gold.append(sent_node_gold)

            if gold_edge is not None and edge_logits_arg is not None and len(edge_logits_arg) > 0:
                edge_pred_idxs = np.argmax(edge_logits_arg, axis=1)
                epoch_edge_preds.extend(RELATION_TYPES[i] for i in edge_pred_idxs)
                epoch_edge_gold.extend(RELATION_TYPES[i] for i in gold_edge)
                epoch_sent_edge_preds.append([RELATION_TYPES[i] for i in edge_pred_idxs])
                epoch_sent_edge_gold.append([RELATION_TYPES[i] for i in gold_edge])
            else:
                epoch_sent_edge_preds.append([])
                epoch_sent_edge_gold.append([])

        return total_loss / max(n, 1)

    try:
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_samples = 0
            batch_step_count = 0
            epoch_node_preds: list[str] = []
            epoch_node_gold: list[str] = []
            epoch_edge_preds: list[str] = []
            epoch_edge_gold: list[str] = []
            epoch_sent_node_preds: list[list[str]] = []
            epoch_sent_node_gold: list[list[str]] = []
            epoch_sent_edge_preds: list[list[str]] = []
            epoch_sent_edge_gold: list[list[str]] = []

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
                    raise
                except (RuntimeError, IndexError, KeyError, TypeError) as exc:
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

                if valid_clause_idxs:
                    gold_node = sample.gold_node_labels[
                        np.array(valid_clause_idxs, dtype=np.int64)
                    ]
                else:
                    gold_node = sample.gold_node_labels
                if (valid_clause_idxs and len(valid_clause_idxs) >= 2
                        and sample.edge_map
                        and edge_logits is not None and len(edge_logits) > 0):
                    if all_pairs:
                        pairs = [
                            (valid_clause_idxs[i], valid_clause_idxs[j])
                            for i in range(len(valid_clause_idxs))
                            for j in range(i + 1, len(valid_clause_idxs))
                        ]
                    else:
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

                _gold_surface = None
                if verb_source_map:
                    _surfaces = verb_source_map.get(sample.sentence.text, [])
                    if _surfaces:
                        _gold_surface = _surfaces[0]

                loss_val, d_node, d_edge = pipeline.loss(
                    node_logits, edge_logits_arg, gold_node, gold_edge,
                    edge_loss_weight=edge_loss_weight,
                    gold_surface=_gold_surface,
                    node_class_weights=node_class_weights,
                    edge_class_weights=edge_class_weights,
                    label_smoothing=label_smoothing,
                )

                if not decoder_only:
                    if mini_batch_size <= 1:
                        pipeline.backward(d_node, d_edge, lr=lr)
                    else:
                        pipeline.backward_accumulate(d_node, d_edge)
                        batch_step_count += 1
                        if batch_step_count >= mini_batch_size:
                            pipeline.apply_accumulated_gradients(lr, n_samples=batch_step_count)
                            batch_step_count = 0

                node_pred_idxs = np.argmax(node_logits, axis=1)
                if valid_clause_idxs:
                    gold_node_aligned = sample.gold_node_labels[
                        np.array(valid_clause_idxs, dtype=np.int64)
                    ]
                else:
                    gold_node_aligned = sample.gold_node_labels
                epoch_node_preds.extend(NODE_TYPES[i] for i in node_pred_idxs)
                epoch_node_gold.extend(NODE_TYPES[i] for i in gold_node_aligned)

                sent_node_pred = [NODE_TYPES[i] for i in node_pred_idxs]
                sent_node_gold = [NODE_TYPES[i] for i in gold_node_aligned]
                epoch_sent_node_preds.append(sent_node_pred)
                epoch_sent_node_gold.append(sent_node_gold)

                if gold_edge is not None and edge_logits_arg is not None and len(edge_logits_arg) > 0:
                    edge_pred_idxs = np.argmax(edge_logits_arg, axis=1)
                    epoch_edge_preds.extend(RELATION_TYPES[i] for i in edge_pred_idxs)
                    epoch_edge_gold.extend(RELATION_TYPES[i] for i in gold_edge)
                    epoch_sent_edge_preds.append([RELATION_TYPES[i] for i in edge_pred_idxs])
                    epoch_sent_edge_gold.append([RELATION_TYPES[i] for i in gold_edge])
                else:
                    epoch_sent_edge_preds.append([])
                    epoch_sent_edge_gold.append([])

                epoch_loss += loss_val
                n_samples += 1

            if mini_batch_size > 1 and batch_step_count > 0 and not decoder_only:
                pipeline.apply_accumulated_gradients(lr, n_samples=batch_step_count)
                batch_step_count = 0

            if verb_loader is not None and pipeline.decoder is not None:
                for vsample in verb_loader:
                    if len(vsample.gold_tokens) == 0:
                        continue
                    dec_logits = pipeline.decoder.forward_decode(vsample.node_type_embeddings)
                    dec_loss, d_dec = pipeline.decoder.loss_decode(dec_logits, vsample.gold_tokens)
                    if not np.isfinite(dec_loss):
                        continue
                    _, dec_grads, d_attn_vec = pipeline.decoder.backward_decode(d_dec)
                    pipeline.decoder.update(dec_grads, d_attn_vec, lr)
                    epoch_loss += dec_loss
                    n_samples += 1

            avg_loss = epoch_loss / max(n_samples, 1)
            metrics = {
                "node_accuracy": node_accuracy(epoch_node_preds, epoch_node_gold),
                "node_macro_f1": node_macro_f1(epoch_node_preds, epoch_node_gold),
                "edge_accuracy": edge_accuracy(epoch_edge_preds, epoch_edge_gold),
                "edge_macro_f1": edge_macro_f1(epoch_edge_preds, epoch_edge_gold),
                "graph_exact_match": _gem(
                    epoch_sent_node_preds, epoch_sent_node_gold,
                    epoch_sent_edge_preds, epoch_sent_edge_gold,
                ),
            }

            # Val pass
            if val_loader is not None:
                _set_training_mode(pipeline, False)
                val_node_preds, val_node_gold = [], []
                val_edge_preds, val_edge_gold = [], []
                val_sent_node_preds, val_sent_node_gold = [], []
                val_sent_edge_preds, val_sent_edge_gold = [], []
                val_loss = _run_eval_pass(
                    pipeline, val_loader,
                    val_node_preds, val_node_gold,
                    val_edge_preds, val_edge_gold,
                    val_sent_node_preds, val_sent_node_gold,
                    val_sent_edge_preds, val_sent_edge_gold,
                )
                _set_training_mode(pipeline, True)
                metrics["val_loss"] = val_loss
                metrics["val_node_accuracy"] = node_accuracy(val_node_preds, val_node_gold)
                metrics["val_node_macro_f1"] = node_macro_f1(val_node_preds, val_node_gold)
                metrics["val_edge_accuracy"] = edge_accuracy(val_edge_preds, val_edge_gold)
                metrics["val_edge_macro_f1"] = edge_macro_f1(val_edge_preds, val_edge_gold)
                metrics["val_graph_exact_match"] = _gem(
                    val_sent_node_preds, val_sent_node_gold,
                    val_sent_edge_preds, val_sent_edge_gold,
                )

            recorder.record(epoch, avg_loss, metrics)
            history.append({"epoch": epoch, "loss": avg_loss, **metrics})

            if csv_writer:
                csv_writer.writerow({"epoch": epoch, "loss": avg_loss, **metrics})

            if epoch % max(1, epochs // 10) == 0 or epoch == 1:
                msg = (
                    f"Epoch {epoch:4d}/{epochs}  loss={avg_loss:.4f}"
                    f"  node_acc={metrics['node_accuracy']:.3f}"
                    f"  edge_acc={metrics['edge_accuracy']:.3f}"
                    f"  gem={metrics['graph_exact_match']:.3f}"
                )
                if val_loader is not None:
                    msg += (
                        f"  val_loss={metrics['val_loss']:.4f}"
                        f"  val_node_f1={metrics['val_node_macro_f1']:.3f}"
                    )
                click.echo(msg)

            # Early stopping
            if patience > 0 and val_loader is not None:
                val_f1 = metrics.get("val_node_macro_f1", 0.0)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_epoch_num = epoch
                    _patience_counter = 0
                    save_checkpoint(pipeline, Path(best_checkpoint_path))
                else:
                    _patience_counter += 1
                    if _patience_counter >= patience:
                        click.echo(
                            f"Early stopping : {patience} epochs sans amélioration "
                            f"(best val_node_macro_f1={best_val_f1:.4f} à epoch {best_epoch_num})"
                        )
                        break
    finally:
        if csv_file:
            csv_file.close()

    # Early stopping : restaurer le meilleur checkpoint
    if patience > 0 and val_loader is not None:
        import shutil
        if best_epoch_num > 0:
            shutil.copy2(best_checkpoint_path, str(output))
            Path(best_checkpoint_path).unlink(missing_ok=True)
            click.echo(f"Best checkpoint restauré (epoch {best_epoch_num}, val_f1={best_val_f1:.4f})")
        else:
            save_checkpoint(pipeline, output)
            click.echo("Aucune amélioration val — checkpoint final sauvegardé")
    else:
        save_checkpoint(pipeline, output)

    click.echo(f"Checkpoint sauvegardé : {output}")

    if log_csv:
        json_path = Path(log_csv).with_suffix(".json")
        recorder.to_json(json_path)
        click.echo(f"Courbe d'entraînement : {json_path}")

    if len(history) >= 10:
        first = sum(r["loss"] for r in history[:5]) / 5
        last = sum(r["loss"] for r in history[-5:]) / 5
        if last < first:
            click.echo(f"Loss décroissante : {first:.4f} → {last:.4f}  ✓")
        else:
            click.echo(f"Avertissement : loss non décroissante ({first:.4f} → {last:.4f})")
