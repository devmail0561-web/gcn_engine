# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
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
from ..constants import NODE_TYPES, RELATION_TYPES, ALL_RELATION_TYPES
from ..evaluation.metrics import (
    node_accuracy, node_macro_f1, edge_accuracy, edge_macro_f1,
    graph_exact_match as _gem,
)
from ..evaluation.recorder import TrainingRecorder
from .checkpoint import save_checkpoint


def _minimal_reps_from_labels(node_labels: list[str], node_types: list[str]) -> list:
    """UDRepresentation minimaux depuis labels CIR pour le verbalizer (enriched_vecs réels)."""
    from ..layer1.representation import UDRepresentation
    reps = []
    for label, ntype in zip(node_labels, node_types):
        lemma = label.split()[0] if label.strip() else ntype
        reps.append(UDRepresentation(
            tokens=[{"lemma": lemma, "pos": "VERB", "dep_rel": "root", "morph": {}}],
            root_lemma=lemma, root_pos="VERB", root_dep_rel="root",
            root_morph={}, subject_pos=None,
            has_object=False, has_advcl=False, has_temporal_obl=False,
            token_span=(0, max(1, len(label.split()))),
        ))
    return reps


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
@click.option("--fasttext", default=None, type=click.Path(path_type=Path),
               help="Modèle fastText multilingue .bin (cc.XX.300.bin, 157 langues, même API). "
                    "Mutuellement exclusif avec --embedding-file. Impose d_emb=300, "
                    "frozen=True par défaut. Préserve l'agnosticisme langue (pas de CamemBERT).")
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
@click.option("--link-pred/--no-link-pred", default=False, show_default=True,
              help="Entraîne la tête LinkPredHead (BCE auxiliaire positifs/négatifs).")
@click.option("--neg-ratio", default=1.0, show_default=True, type=float,
              help="Négatifs par positif pour la tête de liens.")
@click.option("--src-aggregation", default="mean", show_default=True, type=click.Choice(["mean", "max"]),
              help="Agrégation multi-sources de la tête de liens.")
@click.option("--bfs-depth", default=None, show_default=True, type=int,
              help="Profondeur BFS des candidats de predict_links (engine). "
                   "None = tous les candidats (défaut, illimité). "
                   "Stocké dans _arch_json et repris par défaut en prédiction.")
@click.option("--n-rgcn-layers", default=1, show_default=True, type=int,
              help="Nombre de couches R-GCN empilées (≥1). Avec n>1 et GAT, "
                   "recommander --rgcn-dropout 0.2 --weight-decay 1e-4 pour limiter "
                   "l'overfitting sur petits datasets (avertissement, pas de contrainte).")
@click.option("--clause-pooling", default="root", show_default=True,
              type=click.Choice(["root", "mean", "max"]),
              help="Pooling des embeddings de clause (A) : root (défaut), mean ou max "
                   "sur les tokens de contenu. Requiert --embedding-dim > 0.")
@click.option("--subject-object-emb/--no-subject-object-emb", default=False, show_default=True,
              help="Concaténer les embeddings sujet/objet (B, +2*d_emb, _absent appris). "
                   "Requiert --embedding-dim > 0.")
@click.option("--mlp-hidden", default=128, show_default=True, type=int,
              help="Première couche cachée du MLP nœuds. Recommandé : 256 avec "
                   "--subject-object-emb (d_effective=226).")
@click.option("--freeze-embeddings/--no-freeze-embeddings", default=False, show_default=True,
              help="Geler les embeddings pré-entraînés (C). Requiert --embedding-file. "
                   "Les spéciaux _subj_absent/_obj_absent restent entraînables.")
@click.option("--silver-weight", default=1.0, show_default=True, type=float,
              help="Poids des phrases silver dans la loss arêtes (F). 1.0 = aucun effet. "
                   "Variante immédiate : 0.7.")
@click.option("--n-gat-heads", default=1, show_default=True, type=int,
              help="Nombre de têtes d'attention GAT (D). Requiert --use-attention. "
                   "d_out doit être divisible (d_emb=49 → d_out=128 → {1,2,4,8}).")
@click.option("--gat-residual/--no-gat-residual", default=False, show_default=True,
              help="Connexions résiduelles entre couches graphe (E3).")
@click.option("--gat-layernorm/--no-gat-layernorm", default=False, show_default=True,
              help="LayerNorm après concaténation des têtes GAT (E2).")
@click.option("--legacy-decoder", is_flag=True, default=False,
              help="Choix explicite du TrainableDecoder déprécié (G). Défaut : legacy "
                   "tant que --connectors-file est absent.")
@click.option("--connectors-file", default=None, type=click.Path(path_type=Path),
              help="Vocabulaire de connecteurs JSON (G2, connectors_fr.json). "
                   "Active l'entraînement de l'assembleur lexical.")
@click.option("--verbalize-source-dir", default=None, type=click.Path(path_type=Path),
              help="Dataset source des paires verbalize (G1, défaut : --data-dir).")
@click.option("--edge-threshold", default=0.0, show_default=True, type=float,
              help="Seuil de confiance minimum pour émettre une arête [0, 1[. 0 = tout émettre (défaut).")
@click.option("--drop-morph/--no-drop-morph", default=False, show_default=True,
               help="Zéroter les features morphologiques (Tense/Aspect/Mood/Polarity) à l'entraînement "
                    "pour simuler le bridge heuristique (parité train/inférence).")
@click.option("--global-attention/--no-global-attention", default=False, show_default=True,
               help="Phase C : MHA globale sur les nœuds UD avant le MLP nœuds "
                    "(TransformerMLPEncoder, poids fixes + résidu). Capte les arcs "
                    "distants au-delà du voisinage k-hop R-GCN.")
@click.option("--mha-heads", default=4, show_default=True, type=int,
               help="Nombre de têtes de la MHA globale (Phase C). Requiert --global-attention. "
                    "d_effective doit être divisible par cette valeur.")
@click.option("--pairnorm/--no-pairnorm", default=False, show_default=True,
               help="Phase B : PairNorm anti-over-smoothing dans RGCNLayerPT "
                    "(recentrage + normalisation L2 avant activation).")
@click.option("--drop-edge", default=0.0, show_default=True, type=float,
               help="Phase B : probabilité de suppression d'arête en train (DropEdge, "
                    "RGCNLayerPT uniquement). 0.0 = désactivé.")
@click.option("--use-compgcn/--no-compgcn", default=False, show_default=True,
               help="Phase D : embeddings relationnels continus (CompGCN, RGCNLayerPT). "
                    "W_effective = Linear(E_r). Généralisation entre relations proches.")
@click.option("--d-rel-emb", default=32, show_default=True, type=int,
               help="Dimension des embeddings de relations CompGCN (Phase D). "
                    "Requiert --use-compgcn.")
@click.option("--seed", default=None, type=int,
              help="Graine pour la reproductibilité (numpy + torch si disponible).")
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
    fasttext: Path | None,
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
    link_pred: bool,
    neg_ratio: float,
    src_aggregation: str,
    bfs_depth: int | None,
    n_rgcn_layers: int,
    edge_threshold: float,
    drop_morph: bool,
    seed: int | None,
    clause_pooling: str,
    subject_object_emb: bool,
    mlp_hidden: int,
    freeze_embeddings: bool,
    silver_weight: float,
    n_gat_heads: int,
    gat_residual: bool,
    gat_layernorm: bool,
    legacy_decoder: bool,
    connectors_file: Path | None,
    verbalize_source_dir: Path | None,
    global_attention: bool,
    mha_heads: int,
    pairnorm: bool,
    drop_edge: float,
    use_compgcn: bool,
    d_rel_emb: int,
) -> None:
    """Entraîne le pipeline CGNP (NumPy référence) par descente de gradient."""
    from ..data.verbalize_loader import VerbalizerDataLoader
    from ..verbalizer.trainable import TrainableDecoder

    # R9 : reproductibilité — seed numpy + torch si disponible
    if seed is not None:
        np.random.seed(seed)
        try:
            import torch as _torch
            _torch.manual_seed(seed)
        except ImportError:
            pass
        click.echo(f"Seed : {seed}")

    vocab = FeatureVocabulary()

    # Préconditions A/B/C/F (amelioration_v3)
    _has_emb = embedding_file is not None or embedding_dim > 0 or fasttext is not None
    if fasttext is not None and embedding_file is not None:
        raise click.ClickException("--fasttext et --embedding-file sont mutuellement exclusifs.")
    if fasttext is not None and embedding_dim not in (0, 300):
        raise click.ClickException("--fasttext impose d_emb=300 (pas besoin de --embedding-dim).")
    if clause_pooling != "root" and not _has_emb:
        raise click.ClickException("--clause-pooling != root requiert --embedding-dim > 0.")
    if subject_object_emb and not _has_emb:
        raise click.ClickException("--subject-object-emb requiert --embedding-dim > 0.")
    if freeze_embeddings and embedding_file is None:
        raise click.ClickException("--freeze-embeddings requiert --embedding-file.")
    if not (0.0 < silver_weight <= 1.0):
        raise click.ClickException(f"--silver-weight doit être dans ]0, 1] (reçu {silver_weight}).")
    if mlp_hidden < 1:
        raise click.ClickException(f"--mlp-hidden doit être ≥ 1 (reçu {mlp_hidden}).")

    # S1/S2/S9 + Phase A (fastText multilingue) : word embeddings optionnels
    word_embedding = None
    d_emb = 0
    _fasttext_path: Path | None = fasttext
    if _has_emb:
        from ..layer1.embedding import WordEmbedding
        if _fasttext_path is not None:
            # Phase A : d_emb=300 imposé, frozen=True par défaut.
            # Les vecteurs sont remplis après build_vocab (lemmes connus).
            d_emb = 300
            word_embedding = WordEmbedding(d_emb=d_emb, frozen=True)
            click.echo(f"Embeddings fastText : {fasttext} (d_emb=300, frozen)")
        elif embedding_file is not None and embedding_dim == 0:
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
            if freeze_embeddings:
                # C : gel des pré-entraînés (spéciaux _absent toujours entraînables)
                word_embedding.frozen = True
                click.echo("Embeddings pré-entraînés gelés (--freeze-embeddings)")

    # B : point unique de vérité pour la dimension effective
    d_effective = vocab.d_clause_effective(d_emb, subject_object_emb)
    # Closed-loop : edge MLP reçoit features + enriched vectors + node probs
    n_node_types = len(NODE_TYPES)
    d_edge_closed = vocab.d_edge_closed_loop(d_effective, n_node_types, d_emb,
                                                 subject_object_emb)
    # Phase C : substitution MLPEncoder → TransformerMLPEncoder (MHA globale).
    # d_clause = D_effective (inclut déjà d_emb), jamais vocabulary.d_clause brut.
    if global_attention:
        from ..layer2.reference import TransformerMLPEncoder
        try:
            encoder = TransformerMLPEncoder(
                d_clause=d_effective, d_edge=d_edge_closed,
                weight_decay=weight_decay, mlp_hidden=mlp_hidden,
                n_heads=mha_heads)
        except ValueError as _e:
            raise click.ClickException(str(_e)) from _e
        click.echo(f"MHA globale : n_heads={mha_heads} (TransformerMLPEncoder, poids fixes)")
    else:
        if mha_heads != 4:
            raise click.ClickException("--mha-heads requiert --global-attention.")
        encoder = MLPEncoder(d_clause=d_effective, d_edge=d_edge_closed,
                             weight_decay=weight_decay, mlp_hidden=mlp_hidden)

    # Couche 3 : choix du graph selon les flags
    # Phase B/D : PairNorm/DropEdge/CompGCN vivent dans RGCNLayerPT uniquement.
    if use_attention and (pairnorm or drop_edge > 0.0 or use_compgcn):
        raise click.ClickException(
            "--pairnorm/--drop-edge/--use-compgcn requièrent le backend RGCNLayerPT "
            "(incompatibles avec --use-attention/GAT).")
    if not (0.0 <= drop_edge < 1.0):
        raise click.ClickException(f"--drop-edge doit être dans [0, 1[ (reçu {drop_edge}).")
    if d_rel_emb < 1:
        raise click.ClickException(f"--d-rel-emb doit être ≥ 1 (reçu {d_rel_emb}).")
    n_rel = len(ALL_RELATION_TYPES) if bidirectional else len(RELATION_TYPES)  # L-6
    if use_attention:
        from ..layer3.gat import RGCNLayerGAT
        graph = RGCNLayerGAT(d_in=d_effective, d_out=d_effective, n_relations=n_rel,
                             dropout=rgcn_dropout, n_heads=n_gat_heads,
                             use_layernorm=gat_layernorm)
        # output_activation=sigmoid par défaut ; le pipeline passe les
        # intermédiaires en relu quand n_rgcn_layers > 1 (E4)
        click.echo(f"GAT : n_heads={n_gat_heads} layernorm={gat_layernorm}")
    else:
        if n_gat_heads != 1:
            raise click.ClickException("--n-gat-heads requiert --use-attention.")
        if gat_layernorm:
            raise click.ClickException("--gat-layernorm requiert --use-attention.")
        if pairnorm or drop_edge > 0.0 or use_compgcn:
            # Phase B/D : backend PyTorch (seul à supporter ces options).
            from ..layer3.pytorch_rgcn import RGCNLayerPT
            graph = RGCNLayerPT(d_in=d_effective, d_out=d_effective, n_relations=n_rel,
                                pairnorm=pairnorm, drop_edge=drop_edge,
                                use_compgcn=use_compgcn, d_rel_emb=d_rel_emb)
            click.echo(f"RGCNLayerPT : pairnorm={pairnorm} drop_edge={drop_edge} "
                       f"use_compgcn={use_compgcn} d_rel_emb={d_rel_emb}")
        else:
            graph = RGCNLayer(d_in=d_effective, d_out=d_effective, n_relations=n_rel,
                              dropout=rgcn_dropout)

    verb_loader: VerbalizerDataLoader | None = None
    verb_source_map: dict[str, list] = {}
    decoder: TrainableDecoder | None = None
    assembler = None
    verbalize_mode = "legacy"
    connector_vocab: list[str] | None = None
    if connectors_file is not None:
        _conn_doc = json.loads(Path(connectors_file).read_text(encoding="utf-8"))
        connector_vocab = list(_conn_doc.get("connectors", []))
        if not connector_vocab:
            raise click.ClickException(f"Aucun connecteur dans {connectors_file}")
    if verbalize_dir is not None:
        _verb_source = verbalize_source_dir if verbalize_source_dir is not None else data_dir
        verb_loader = VerbalizerDataLoader(verbalize_dir, source_json_dir=_verb_source,
                                           connector_vocab=connector_vocab)
        if len(verb_loader) == 0:
            raise click.ClickException(f"Aucune paire verbalize dans {verbalize_dir}")
        decoder = TrainableDecoder(verb_loader.vocab)
        verb_source_map = verb_loader.source_text_map()
        click.echo(f"Verbalize : {len(verb_loader)} paires | vocab={len(verb_loader.vocab)} tokens")
        if connector_vocab is not None:
            from ..verbalizer.trainable import LexicalConnectorAssembler
            assembler = LexicalConnectorAssembler(connector_vocab)
            verbalize_mode = "lexical"
            click.echo(f"Assembleur lexical : {len(connector_vocab)} connecteurs (mode lexical)")
        elif legacy_decoder:
            click.echo("Décodeur legacy explicite (--legacy-decoder)")

    if decoder_only and decoder is None:
        raise click.ClickException("--decoder-only requiert --verbalize-dir")

    if patience > 0 and val_dir is None:
        raise click.ClickException("--patience requiert --val-dir")

    # B4 : validation --n-rgcn-layers (restriction GAT levée — E4)
    if n_rgcn_layers < 1:
        raise click.ClickException(f"--n-rgcn-layers doit être ≥ 1 (reçu {n_rgcn_layers}).")
    if bfs_depth is not None and bfs_depth < 1:
        raise click.ClickException(f"--bfs-depth doit être ≥ 1 (reçu {bfs_depth}).")

    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab,
                           decoder=decoder, all_pairs=all_pairs, word_embedding=word_embedding,
                           bidirectional=bidirectional, n_rgcn_layers=n_rgcn_layers,
                           edge_threshold=edge_threshold, drop_morph=drop_morph,
                           bfs_depth=bfs_depth, clause_pooling=clause_pooling,
                           subject_object_emb=subject_object_emb,
                           gat_residual=gat_residual, assembler=assembler)
    # §1 : métadonnées d'arch (checkpoint) — silver_weight / verbalize_mode
    pipeline.silver_weight = silver_weight
    pipeline.verbalize_mode = verbalize_mode
    # Phases B/C/D : flags d'arch persistés dans _arch_json (checkpoint.py).
    pipeline.global_attention = global_attention
    pipeline.mha_heads = mha_heads
    pipeline.pairnorm = pairnorm
    pipeline.drop_edge = drop_edge
    pipeline.use_compgcn = use_compgcn
    pipeline.d_rel_emb = d_rel_emb

    link_pred_head = None
    if link_pred:
        from ..layer3.link_pred import LinkPredHead
        link_pred_head = LinkPredHead(d_in=d_effective, src_aggregation=src_aggregation)
        pipeline.link_predictor = link_pred_head
        click.echo(f"LinkPredHead : d_in={d_effective} src_agg={src_aggregation} "
                   f"neg_ratio={neg_ratio} bfs_depth={bfs_depth}")

    if encoder_checkpoint is not None:
        from .checkpoint import load_checkpoint
        load_checkpoint(pipeline, encoder_checkpoint, trusted=True)
        click.echo(f"Checkpoint encodeur chargé : {encoder_checkpoint}")

    loader = GCNDataLoader(data_dir, all_pairs=all_pairs, shuffle=True,
                           silver_weight=silver_weight)
    if len(loader) == 0:
        raise click.ClickException(f"Aucune sentence dans {data_dir}")
    if silver_weight < 1.0:
        click.echo(f"Silver-weight : {silver_weight} (F — loss arêtes pondérée)")
    # R6 : détecter les N-arêtes (hyperedge_map) non supervisées
    _n_hyper = sum(1 for s in loader if s.hyperedge_map)
    if _n_hyper:
        warnings.warn(
            f"{_n_hyper} phrase(s) avec des N-arêtes (sources multiples) dans {data_dir}. "
            "Ces arêtes sont collectées dans hyperedge_map mais jamais supervisées "
            "par l'entraînement (gradient 0). Utilisez des arêtes binaires ou "
            "implémentez une tête N-aire.",
            UserWarning, stacklevel=2,
        )

    val_loader = None
    if val_dir is not None:
        val_loader = GCNDataLoader(val_dir, all_pairs=all_pairs, shuffle=False,
                                   silver_weight=silver_weight)
        if len(val_loader) == 0:
            raise click.ClickException(f"Aucune sentence dans {val_dir}")
        click.echo(f"Val : {len(val_loader)} sentences")

    # Passe unique : class weights + vocab embeddings (évite deux itérations sur le dataset)
    node_class_weights = None
    edge_class_weights = None
    if weighted_loss or word_embedding is not None:
        from collections import Counter
        node_counts: Counter = Counter() if weighted_loss else Counter()
        edge_counts: Counter = Counter() if weighted_loss else Counter()
        all_lemmas: list[str] = [] if word_embedding is not None else []
        for sample in loader:
            if weighted_loss:
                for label in sample.gold_node_labels:
                    node_counts[int(label)] += 1
                for (src, tgt), rel in sample.edge_map.items():
                    edge_counts[int(rel)] += 1
            if word_embedding is not None:
                reps_s, _, _ = reps_from_sentence(sample.sentence)
                # A : avec pooling != root, élargir le vocab aux lemmes de contenu ;
                # B : ajouter les lemmes sujet/objet (les _absent sont pré-enregistrés)
                if clause_pooling == "root" and not subject_object_emb:
                    all_lemmas.extend(r.root_lemma for r in reps_s)
                else:
                    from ..layer1.features import _pool_lemmas, _find_subj_obj_lemmas
                    for r in reps_s:
                        if clause_pooling == "root":
                            all_lemmas.append(r.root_lemma)
                        else:
                            all_lemmas.extend(_pool_lemmas(r, clause_pooling))
                        if subject_object_emb:
                            all_lemmas.extend(_find_subj_obj_lemmas(r))
        if weighted_loss:
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
                n_edge_classes = encoder.n_relation_types
                edge_class_weights = np.zeros(n_edge_classes, dtype=np.float32)
                for c in range(n_edge_classes):
                    count = edge_counts.get(c, 1)
                    edge_class_weights[c] = total_edges / (n_edge_classes * count)
                click.echo(f"Edge class weights : {dict(zip(RELATION_TYPES, edge_class_weights.round(3)))}")
        if word_embedding is not None and all_lemmas:
            word_embedding.build_vocab(all_lemmas)
            click.echo(f"Embeddings vocab : {len(all_lemmas)} lemmes ({len(set(all_lemmas))} uniques)")
            if _fasttext_path is not None:
                # Phase A : remplissage fastText après build_vocab (vocab connu).
                # fastText subword → get_word_vector répond pour tout lemme.
                try:
                    import fasttext as _ft
                    _ft_model = _ft.load_model(str(_fasttext_path))
                except ImportError:
                    try:
                        import fasttext_wheel as _ft
                        _ft_model = _ft.load_model(str(_fasttext_path))
                    except ImportError as _e:
                        raise click.ClickException(
                            "load_from_fasttext requiert fasttext-wheel (ou fasttext) : "
                            "pip install fasttext-wheel"
                        ) from _e
                _n_ft = 0
                for _lemma in dict.fromkeys(all_lemmas):
                    _vec = np.asarray(
                        _ft_model.get_word_vector(_lemma), dtype=np.float32)
                    if _vec.shape != (300,):
                        continue
                    _idx = word_embedding._vocab.get(_lemma)
                    if _idx is not None:
                        word_embedding._E[_idx] = _vec
                        _n_ft += 1
                # Plage pré-entraînée = tous les non-spéciaux (comme load_from_file
                # appelé juste après __init__ : start=3). Spéciaux _absent (1-2)
                # restent entraînables même avec frozen=True.
                word_embedding._pretrained_start = 3
                word_embedding._pretrained_end = len(word_embedding._lemmas)
                word_embedding.frozen = True
                click.echo(f"Embeddings fastText : {_n_ft} vecteurs remplis (d_emb=300, frozen)")

    click.echo(f"Données : {len(loader)} sentences | epochs={epochs} lr={lr}")

    recorder = TrainingRecorder()
    history: list[dict] = []
    # B2 : csv_file/csv_writer initialisés ici pour être visibles dans le finally.
    # L'ouverture réelle du fichier est faite à l'intérieur du try (ci-dessous)
    # pour garantir la fermeture en cas d'exception pendant le setup.
    csv_writer = None
    csv_file = None

    best_val_f1 = -1.0
    best_epoch_num = 0
    best_checkpoint_path = str(output) + ".best.npz"
    _patience_counter = 0

    def _set_training_mode(pipeline: CGNPipeline, training: bool) -> None:
        """Bascule TOUS les composants avec dropout en mode eval ou train."""
        def _toggle(obj, mode):
            # nn.Module : appeler .train() pour propager récursivement aux sous-modules
            if hasattr(obj, 'train') and callable(obj.train):
                obj.train(mode)
            elif hasattr(obj, 'training'):
                obj.training = mode
        _toggle(pipeline.encoder, training)
        for layer in pipeline._graph_layers:
            _toggle(layer, training)
        # B9 : couvrir aussi decoder et link_predictor (omis avant)
        if pipeline.decoder is not None:
            _toggle(pipeline.decoder, training)
        if getattr(pipeline, 'link_predictor', None) is not None:
            _toggle(pipeline.link_predictor, training)

    def _run_eval_pass(pipeline, loader, epoch_node_preds, epoch_node_gold,
                       epoch_edge_preds, epoch_edge_gold,
                       epoch_sent_node_preds, epoch_sent_node_gold,
                       epoch_sent_edge_preds, epoch_sent_edge_gold):
        """Exécute un pass forward sur le val set et retourne les métriques."""
        total_loss = 0.0
        n = 0
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
                # C2.3-doc : gold_edge_map intentionnellement absent ici —
                # évaluation sans oracle (asymétrie teacher-forcing BUG-1).
                # Ne pas symétriser par le gold : ferait passer le R-GCN val
                # sur types réels → F1 val optimiste → early stopping biaisé.
                pipeline.forward(
                    reps, sample.sentence.text,
                    clause_positions=valid_clause_idxs,
                    n_total_clauses=len(sample.sentence.clauses),
                    connector_reps=connector_reps,
                )
            except (ValueError, RuntimeError, IndexError, KeyError, TypeError) as _eval_exc:
                n_skipped += 1
                import warnings as _wv
                _wv.warn(f"_run_eval_pass : phrase {sample.sentence.id!r} ignorée — {_eval_exc}",
                         UserWarning, stacklevel=2)
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
                    _val_edge_sw = (
                        np.array([sample.edge_conf_map.get(pairs[i], 1.0)
                                  for i in valid_edge_idxs], dtype=np.float32)
                        if sample.edge_conf_map else None
                    )
                else:
                    _val_edge_sw = None

            loss_val, _, _ = pipeline.loss(
                node_logits, edge_logits_arg, gold_node, gold_edge,
                edge_loss_weight=edge_loss_weight,
                node_class_weights=node_class_weights,
                edge_class_weights=edge_class_weights,
                sample_weight=sample.sentence.weight,
                edge_sample_weights=_val_edge_sw,  # S-5 cohérence val
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

        if n_skipped > 0:
            import warnings as _wvs
            _wvs.warn(f"_run_eval_pass : {n_skipped} phrase(s) ignorée(s) sur {n + n_skipped} — "
                      f"val_f1 calculée sur {n} phrase(s) seulement.",
                      UserWarning, stacklevel=3)
        return total_loss / max(n, 1)

    try:
        # B2 : ouverture du CSV ici (dans le try) pour garantir la fermeture
        # en cas d'exception pendant le setup ultérieur (class weights, etc.).
        if log_csv:
            csv_fieldnames = [
                "epoch", "loss", "node_accuracy", "node_macro_f1",
                "edge_accuracy", "edge_macro_f1", "graph_exact_match",
            ]
            if pipeline.decoder is not None:
                csv_fieldnames.append("decoder_loss")  # L-3 : séparé de loss
            if assembler is not None:
                csv_fieldnames.append("verbalize_connector_prec1")
                csv_fieldnames.append("assembler_avg_loss")  # C2.4 : loss assembleur
            if val_loader is not None:
                csv_fieldnames.extend([
                    "val_loss", "val_node_accuracy", "val_node_macro_f1",
                    "val_edge_accuracy", "val_edge_macro_f1", "val_graph_exact_match",
                ])
            csv_file = open(log_csv, "w", newline="", encoding="utf-8")
            # C1.5 : extrasaction='ignore' — un resume avec headers différents
            # (decoder/assembler apparu-disparu) ne doit pas crasher writerow.
            csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames,
                                        extrasaction='ignore')
            csv_writer.writeheader()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_dec_loss = 0.0   # L-3 : loss décodeur séparée
            epoch_asm_loss = 0.0   # C2.4 : loss assembleur séparée (par arête)
            n_dec_samples = 0
            n_asm_samples = 0      # C2.4 : compteur assembleur séparé (par arête)
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
            _asm_pred_idxs: list[int] = []  # G3 : connecteurs prédits (par arête)
            _asm_gold_idxs: list = []       # G3 : connecteurs gold (None exclus)

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
                        gold_edge_map=sample.edge_map,  # BUG-1 teacher-forcing : types réels en train uniquement
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
                        # S-5 : confidence par arête depuis edge_conf_map
                        _edge_confs = [
                            sample.edge_conf_map.get(pairs[i], 1.0)
                            for i in valid_edge_idxs
                        ]
                        _edge_sw = np.array(_edge_confs, dtype=np.float32) if any(
                            p in sample.edge_conf_map for p in pairs) else None
                    else:
                        gold_edge = None
                        edge_logits_arg = None
                        _edge_sw = None
                else:
                    gold_edge = None
                    edge_logits_arg = None
                    _edge_sw = None

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
                    sample_weight=sample.sentence.weight,
                    edge_sample_weights=_edge_sw,  # S-5 : confidence par arête
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

                # Tête de liens : BCE auxiliaire (positifs gold + négatifs échantillonnés).
                # Ne touche qu'à la tête (pas au backbone) — désactivé par défaut.
                if link_pred_head is not None and not decoder_only:
                    _ev = pipeline._cached_enriched_vecs
                    if _ev is not None and len(_ev) >= 2:
                        from ..layer3.link_pred import sample_negatives
                        if valid_clause_idxs:
                            _pos_of = {c: k for k, c in enumerate(valid_clause_idxs)}
                            _true = [(_pos_of[a], _pos_of[b]) for (a, b) in sample.edge_map
                                     if a in _pos_of and b in _pos_of]
                        else:
                            _true = list(sample.edge_map.keys())
                        if _true:
                            _neg = sample_negatives(_true, len(_ev),
                                                    neg_ratio=neg_ratio, seed=epoch * 1000 + n_samples)
                            _gW = np.zeros_like(link_pred_head.W)
                            _gb = np.zeros_like(link_pred_head.b)
                            _n_lp = 0
                            for (_a, _b) in _true:
                                _, _g = link_pred_head.loss_and_grad(_ev[_a], _ev[_b], 1)
                                _gW += _g[0]; _gb += _g[1]; _n_lp += 1
                            for (_a, _b) in _neg:
                                _, _g = link_pred_head.loss_and_grad(_ev[_a], _ev[_b], 0)
                                _gW += _g[0]; _gb += _g[1]; _n_lp += 1
                            if _n_lp:
                                link_pred_head.update([_gW / _n_lp, _gb / _n_lp], lr)

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
                from ..verbalizer.trainable import SurfaceVocabulary as _SV
                from ..data.verbalize_loader import _node_type_embeddings
                for vsample in verb_loader:
                    if len(vsample.gold_tokens) == 0:
                        continue
                    import json as _json
                    _cir = _json.loads(vsample.ir_json)
                    _ntypes = [n.get("node_type", "entite") for n in _cir.get("nodes", [])]
                    _labels = vsample.node_labels if vsample.node_labels else _ntypes
                    _reps = _minimal_reps_from_labels(_labels, _ntypes)
                    if not _reps:
                        continue
                    pipeline.forward(_reps, vsample.source_text)
                    # G1 : vecteurs lexicaux réels depuis clause_texts si disponibles,
                    # sinon repli sur les one-hot de types (comportement historique)
                    if word_embedding is not None and vsample.clause_texts:
                        _lex_vecs = np.stack([
                            np.mean([word_embedding.lookup(w)
                                     for w in ct.lower().split() or ['_unk']],
                                    axis=0).astype(np.float32)
                            for ct in vsample.clause_texts
                        ])
                    else:
                        _lex_vecs = _node_type_embeddings(
                            _json.loads(vsample.ir_json)['nodes'])
                    _eos = pipeline.decoder.vocab._t2i.get(_SV.EOS, -1)
                    _gold = (np.append(vsample.gold_tokens, _eos).astype(np.int64)
                             if _eos >= 0 else vsample.gold_tokens)
                    dec_logits = pipeline.decoder.forward_decode(_lex_vecs, _gold)
                    dec_loss, d_dec = pipeline.decoder.loss_decode(dec_logits, _gold)
                    if not np.isfinite(dec_loss):
                        continue
                    _, dec_grads, d_attn_vec = pipeline.decoder.backward_decode(d_dec)
                    pipeline.decoder.update(dec_grads, d_attn_vec, lr)
                    epoch_dec_loss += dec_loss   # L-3 : séparé de epoch_loss
                    n_dec_samples += 1
                    # G2/G3 : entraînement assembleur + collecte connector_prec@1
                    if assembler is not None and vsample.edge_triples:
                        _golds = vsample.connector_gold_idx or []
                        for _k, (_s, _d, _r) in enumerate(vsample.edge_triples):
                            _g = _golds[_k] if _k < len(_golds) else None
                            _pred = assembler.predict(_r)
                            if _g is None:
                                continue
                            _asm_pred_idxs.append(_pred)
                            _asm_gold_idxs.append(_g)
                            _a_loss, _a_grads = assembler.loss_and_grad(_r, _g)
                            assembler.update(_a_grads, lr)
                            epoch_asm_loss += _a_loss   # C2.4 : séparé de epoch_loss
                            n_asm_samples += 1          # C2.4 : n_samples reste par phrase

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
            if pipeline.decoder is not None:
                metrics["decoder_loss"] = (  # L-3 : séparé, non mélangé dans loss
                    epoch_dec_loss / max(n_dec_samples, 1)
                )
            # G3 : connector_precision@1 (gold None exclus) — remplace le BLEU invalide
            if assembler is not None:
                from ..evaluation.metrics import connector_precision_at_1
                metrics["verbalize_connector_prec1"] = connector_precision_at_1(
                    _asm_pred_idxs, _asm_gold_idxs)
                # C2.4 : loss assembleur rapportée séparément (jamais dans epoch_loss)
                metrics["assembler_avg_loss"] = (
                    epoch_asm_loss / max(n_asm_samples, 1)
                )

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
            import tempfile as _tf
            _tmp = Path(str(output) + ".tmp.restore")
            shutil.copy2(best_checkpoint_path, str(_tmp))
            Path(_tmp).replace(output)  # atomique POSIX
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
