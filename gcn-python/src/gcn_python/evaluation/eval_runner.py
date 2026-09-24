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


def run_eval(
    data_dir: Path,
    model_path: Path,
    *,
    edge_threshold_override: float | None = None,
) -> dict:
    """Évalue le pipeline CGNP sur toutes les sentences d'un répertoire.

    Retourne un dict avec : n_samples, n_skipped, node_accuracy,
    node_macro_f1, edge_accuracy, edge_macro_f1.

    edge_threshold_override : surcharge le seuil stocké dans _arch_json.
    """
    # Reconstruire le pipeline depuis l'arch du checkpoint (miroir de
    # GCNEngine.from_pretrained) : sinon tout modèle bidirectional / embeddings /
    # all_pairs / multi-couches crashe au load (shapes) ou est évalué dans le
    # mauvais mode (métriques fausses silencieusement).
    _arch: dict = {}
    _raw = None
    try:
        _raw = np.load(model_path, allow_pickle=True)
        if "_arch_json" not in _raw:
            warnings.warn(
                "run_eval : checkpoint sans _arch_json — hyperparamètres d'inférence "
                "(edge_threshold, temperature, all_pairs…) inconnus, pipeline par défaut. "
                "Re-entraîner avec gcn-train >= 2.1.0 pour générer ce champ.",
                UserWarning, stacklevel=2,
            )
        else:
            _arch = json.loads(str(_raw["_arch_json"][0]))
    except Exception as exc:
        warnings.warn(f"run_eval : arch illisible ({exc}) — pipeline par défaut.",
                      UserWarning, stacklevel=2)
    # trusted=False si l'arch est absente/illisible → hyperparamètres inconnus,
    # métriques potentiellement non représentatives du modèle réel.
    _arch_trusted = bool(_arch)
    _d_eff = int(_arch.get("d_eff", FeatureVocabulary().d_clause))
    _d_emb = int(_arch.get("d_emb", 0))
    _n_rel = int(_arch.get("n_relations", len(RELATION_TYPES)))
    _bidi = bool(_arch.get("bidirectional", _arch.get("bidi_flag", False)))
    _all_pairs = bool(_arch.get("all_pairs", False))
    _n_layers = int(_arch.get("n_rgcn_layers", 1))
    _gclass = _arch.get("graph_class", "RGCNLayer")
    # Restaurer edge_threshold, drop_morph et temperature depuis l'arch —
    # sans ça run_eval utilise les défauts même si le modèle a été entraîné
    # avec d'autres valeurs → métriques non représentatives de la prod.
    _edge_threshold = float(_arch.get("edge_threshold", 0.0))
    _drop_morph = bool(_arch.get("drop_morph", False))
    _temperature = float(_arch.get("temperature", 1.0))
    _bfs_depth = _arch.get("bfs_depth")
    if _bfs_depth is not None:
        _bfs_depth = int(_bfs_depth)
    if edge_threshold_override is not None:
        _edge_threshold = float(edge_threshold_override)
    # V2 : charger le vocab depuis le checkpoint AVANT de construire l'encodeur,
    # identique à GCNEngine.from_pretrained — évite un crash shape mismatch si
    # le modèle a été entraîné avec connector_lemmas (d_edge différent du défaut).
    vocab = FeatureVocabulary()
    if _raw is not None and "_vocab_json" in _raw:
        try:
            vocab = FeatureVocabulary.from_json(str(_raw["_vocab_json"][0]))
        except Exception as exc:
            warnings.warn(f"run_eval : vocab illisible ({exc}) — vocab par défaut.",
                          UserWarning, stacklevel=2)
    _word_embedding = None
    if _d_emb > 0:
        from ..layer1.embedding import WordEmbedding
        _word_embedding = WordEmbedding(d_emb=_d_emb)
        if bool(_arch.get("freeze_embeddings", False)):
            _word_embedding.frozen = True
    _mlp_hidden = int(_arch.get("mlp_hidden", 128))
    _sob_eval = bool(_arch.get("subject_object_emb", False))
    encoder = MLPEncoder(d_clause=_d_eff,
                         d_edge=vocab.d_edge_closed_loop(_d_eff, len(NODE_TYPES), _d_emb,
                                                         _sob_eval),
                         mlp_hidden=_mlp_hidden)
    if _gclass == "RGCNLayerGAT":
        try:
            from ..layer3.gat import RGCNLayerGAT
            graph = RGCNLayerGAT(d_in=_d_eff, d_out=int(_arch.get("d_hidden", _d_eff)),
                                 n_relations=_n_rel,
                                 n_heads=int(_arch.get("n_gat_heads", 1)),
                                 output_activation=str(_arch.get("gat_output_activation", "sigmoid")),
                                 use_layernorm=bool(_arch.get("gat_layernorm", False)))
        except ImportError:
            warnings.warn("run_eval : PyTorch absent — repli sur RGCNLayer (NumPy).",
                          UserWarning, stacklevel=2)
            graph = RGCNLayer(d_in=_d_eff, d_out=int(_arch.get("d_hidden", _d_eff)),
                              n_relations=_n_rel)
    else:
        graph = RGCNLayer(d_in=_d_eff, d_out=int(_arch.get("d_hidden", _d_eff)),
                          n_relations=_n_rel)
    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab,
                           word_embedding=_word_embedding, bidirectional=_bidi,
                           all_pairs=_all_pairs, n_rgcn_layers=_n_layers,
                           edge_threshold=_edge_threshold, drop_morph=_drop_morph,
                           temperature=_temperature, bfs_depth=_bfs_depth,
                           clause_pooling=str(_arch.get("clause_pooling", "root")),
                           subject_object_emb=bool(_arch.get("subject_object_emb", False)),
                           gat_residual=bool(_arch.get("gat_residual", False)))
    load_checkpoint(pipeline, model_path, trusted=True)
    # D1 : load_checkpoint restaure les hyperparamètres depuis l'arch — re-appliquer
    # l'override après, sinon la valeur arch écrase l'override passé explicitement.
    if edge_threshold_override is not None:
        pipeline.edge_threshold = float(edge_threshold_override)
    # V1 : mode évaluation — désactiver le dropout pour des métriques déterministes.
    pipeline.encoder.training = False
    for _layer in pipeline._graph_layers:
        if hasattr(_layer, "training"):
            _layer.training = False

    loader = GCNDataLoader(data_dir, all_pairs=_all_pairs)

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
            if pipeline.all_pairs:
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
        "trusted": _arch_trusted,
    }


@click.command("gcn-eval")
@click.option("--data-dir", required=True, type=click.Path(path_type=Path),
              help="Répertoire val JSON (requis).")
@click.option("--model-path", required=True, type=click.Path(path_type=Path),
              help="Checkpoint .npz du modèle (requis).")
@click.option("--test-dir", default=None, type=click.Path(path_type=Path),
              help="Répertoire test JSON (held-out). Métriques reportées sous la clé 'test'.")
@click.option("--output", default=None, type=click.Path(path_type=Path),
              help="Chemin JSON du rapport (optionnel, sinon stdout)")
@click.option("--gate", default=None, type=float,
              help="Ajoute un booléen 'gate' au rapport (val_node_macro_f1 >= seuil). "
                   "Par défaut : pas de contrôle. N'active pas la tête de liens en "
                   "production ; avec --on-fail error, exit 1 si sous le seuil.")
@click.option("--on-fail", default="warn", type=click.Choice(["warn", "error"]), show_default=True,
              help="warn = jamais fail-closed (défaut) ; error = exit 1 si sous le seuil.")
@click.option("--edge-threshold", default=None, type=float,
              help="Surcharge le seuil d'arête stocké dans le checkpoint. "
                   "Par défaut : valeur enregistrée dans _arch_json (reproduit la config d'entraînement). "
                   "Sans _arch_json : repli sur 0.0 + avertissement.")
@click.option("--quiet", is_flag=True, default=False,
              help="Supprime les avertissements (warnings) pendant l'évaluation : "
                   "stdout = JSON pur, machine-readable même avec -W default. "
                   "Sans --quiet, les warnings partent sur stderr (mélangés à stdout "
                   "si les flux sont combinés).")
def eval_cmd(
    data_dir: Path, model_path: Path, test_dir: Path | None, output: Path | None,
    gate: float | None, on_fail: str, edge_threshold: float | None,
    quiet: bool,
) -> None:
    """Évalue le pipeline CGNP sur un répertoire de données annotées.

    Utilisez --test-dir pour obtenir les métriques held-out honnêtes (test set).
    Les métriques --data-dir sont reportées à la racine du JSON ;
    les métriques --test-dir sont reportées sous la clé "test".
    """
    def _build_report() -> dict:
        report = run_eval(data_dir, model_path, edge_threshold_override=edge_threshold)

        if test_dir is not None:
            test_report = run_eval(
                Path(test_dir), model_path, edge_threshold_override=edge_threshold
            )
            report["test"] = {
                "data_dir": str(test_dir),
                "n_samples": test_report["n_samples"],
                "n_skipped": test_report["n_skipped"],
                "node_accuracy": test_report["node_accuracy"],
                "node_macro_f1": test_report["node_macro_f1"],
                "edge_accuracy": test_report["edge_accuracy"],
                "edge_macro_f1": test_report["edge_macro_f1"],
                "causal_graph_similarity": test_report.get("causal_graph_similarity"),
            }

        if gate is not None:
            f1 = report.get("node_macro_f1")
            report["gate"] = {"threshold": gate, "val_node_macro_f1": f1,
                              "passed": (f1 is not None and f1 >= gate)}
            if not report["gate"]["passed"]:
                msg = (f"gcn-eval gate : val_node_macro_f1={f1} < seuil {gate} "
                       f"(bloquant données, pas code — tête de liens non activable en prod).")
                if on_fail == "error":
                    raise click.ClickException(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)
        return report

    try:
        if quiet:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                report = _build_report()
        else:
            report = _build_report()
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f"Erreur d'évaluation : {exc}") from exc
    # F1 : répertoire vide ou toutes les sentences ignorées → diagnostic stderr
    if report.get("n_samples", 0) == 0:
        click.echo(
            "ATTENTION : n_samples=0 — aucune sentence évaluée "
            "(répertoire vide, sentences sans clauses, ou toutes ignorées).",
            err=True,
        )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if output:
        Path(output).write_text(text, encoding="utf-8")
        click.echo(f"Rapport écrit : {output}")
    else:
        click.echo(text)
