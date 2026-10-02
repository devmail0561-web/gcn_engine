# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import csv
import json
import warnings
from pathlib import Path

import click
import numpy as np

from ..constants import (
    INTENT_TYPES,
    NODE_TYPES,
    QUALIFIERS,
    RELATION_TYPES,
    SENTENCE_TYPES,
    coarse_node,
    coarse_relation,
    rgcn_n_relations,
)
from ..data.loader import GCNDataLoader, reps_from_sentence
from ..evaluation.metrics import (
    edge_accuracy,
    edge_macro_f1,
    node_accuracy,
    node_macro_f1,
    per_class_report,
)
from ..evaluation.metrics import (
    graph_exact_match as _gem,
)
from ..evaluation.recorder import TrainingRecorder
from ..layer1.features import FeatureVocabulary
from ..layer2.reference import MLPEncoder
from ..layer3.reference import RGCNLayer
from ..pipeline.cgnp import CGNPipeline
from .checkpoint import save_checkpoint


def collect_val_logits_for_calibration(val_loader, pipeline, all_pairs: bool = True):
    """Collecte (logits arêtes, labels gold) sur le val set pour T5-min (B2 audit).

    Chemin forward direct via `reps_from_sentence` (même chemin que
    `_run_eval_pass`) + alignement logits↔gold via `edge_map`. Les phrases en
    échec émettent `UserWarning` (jamais de `except: pass` silencieux).

    Retourne `(_val_logits, _val_labels, _t5_skipped)` — la décision
    (gate N>=10, `optimize_temperature`, messages) reste à l'appelant.
    """
    _val_logits, _val_labels = [], []
    _t5_skipped = 0
    for _vs in val_loader:
        try:
            _reps, _valid_idxs, _conn_reps = reps_from_sentence(_vs.sentence)
            if not _reps or len(_reps) < 2:
                continue
            pipeline.forward(
                _reps, _vs.sentence.text,
                clause_positions=_valid_idxs,
                n_total_clauses=len(_vs.sentence.clauses),
                connector_reps=_conn_reps,
            )
            _logits = pipeline._cached_edge_logits
            if _logits is None or len(_logits) == 0 or not _vs.edge_map:
                continue
            # Aligner logits ↔ gold comme _run_eval_pass : les paires
            # sont ordonnées sur les reps, les clés edge_map sur les
            # indices de clauses originaux.
            if all_pairs:
                _pairs = [
                    (_valid_idxs[i], _valid_idxs[j])
                    for i in range(len(_valid_idxs))
                    for j in range(i + 1, len(_valid_idxs))
                ]
            else:
                _pairs = [
                    (_valid_idxs[k], _valid_idxs[k + 1])
                    for k in range(len(_valid_idxs) - 1)
                ]
            _gold_full = np.array(
                [_vs.edge_map.get(p, -1) for p in _pairs], dtype=np.int64
            )
            _mask = _gold_full >= 0
            if _mask.any():
                _idxs = np.where(_mask)[0]
                # Garde : le forward peut filtrer (seuil) — ne garder
                # que les indices valides dans la plage des logits.
                _idxs = _idxs[_idxs < len(_logits)]
                if len(_idxs) > 0:
                    _val_logits.append(_logits[_idxs])
                    _val_labels.extend(_gold_full[_idxs].tolist())
        except Exception as _e:  # noqa: BLE001
            _t5_skipped += 1
            warnings.warn(f"T5-min : phrase {_vs.sentence.id!r} ignorée — {_e}",
                          UserWarning, stacklevel=2)
            continue
    return _val_logits, _val_labels, _t5_skipped


def _minimal_reps_from_labels(node_labels: list[str], node_types: list[str]) -> list:
    """UDRepresentation minimaux depuis labels CIR pour le verbalizer (enriched_vecs réels)."""
    from ..layer1.representation import UDRepresentation
    reps = []
    for label, ntype in zip(node_labels, node_types, strict=False):
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
@click.option("--all-pairs/--no-all-pairs", default=True, show_default=True,
              help="Superviser toutes les paires (i,j) avec i<j, pas seulement consécutives. "
                   "Requiert dataset re-annoté avec des arêtes gap>1.")
@click.option("--embedding-dim", default=128, show_default=True, type=int,
              help="Dimension des word embeddings apprenables (S1/S2), ACTIVÉS PAR DÉFAUT. "
                   "0 = désactivé (DÉCONSEILLÉ : aucun signal lexical, "
                   "cf REMEDIATION-DIAGNOSTIC.md).")
@click.option("--embedding-file", default=None, type=click.Path(path_type=Path),
               help="Fichier GloVe/FastText pour initialiser les embeddings (S9). "
                    "Active automatiquement --embedding-dim si non précisé.")
@click.option("--fasttext", default=None, type=click.Path(path_type=Path),
               help="Modèle fastText multilingue .bin (cc.XX.300.bin, 157 langues, même API). "
                    "Mutuellement exclusif avec --embedding-file. Impose d_emb=300, "
                    "frozen=True par défaut. Préserve l'agnosticisme langue (pas de CamemBERT).")
@click.option("--mini-batch-size", default=4, show_default=True, type=int,
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
@click.option("--weight-decay", default=1e-4, show_default=True, type=float,
              help="Coefficient de régularisation L2 sur les poids MLP. 0 = désactivé.")
@click.option("--rgcn-dropout", default=0.1, show_default=True, type=float,
              help="Dropout sur les features d'entrée des couches R-GCN/GAT. 0 = désactivé.")
@click.option("--edge-dropout", default=0.3, show_default=True, type=float,
              help="Dropout des couches cachées de la tête edge (MLPEncoder). "
                   "0 = désactivé (recommandé en few-shot : variance >> signal).")
@click.option("--label-smoothing", default=0.05, show_default=True, type=float,
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
@click.option("--no-positional", is_flag=True, default=False, show_default=True,
              help="v5.5 : zéroter les 12 features positionnelles Éq.9 (ablation structure).")
@click.option("--no-ternary", is_flag=True, default=False, show_default=True,
              help="v5.5 : zéroter les 5 features ternaires (ablation).")
@click.option("--no-mood", "no_mood_flag", is_flag=True, default=False, show_default=True,
              help="v5.5 : masquer Mood sans changer la granularité (contrairement à --coarse-phase).")
@click.option("--no-tense", "no_tense_flag", is_flag=True, default=False, show_default=True,
              help="v5.5 : masquer Tense sans changer la granularité (contrairement à --coarse-phase).")
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
@click.option("--two-pass-val/--no-two-pass-val", default=True, show_default=True,
              help="Passage préliminaire d'arêtes en val/inférence pour prédire les types "
                   "avant le R-GCN (corrige l'asymétrie teacher forcing BUG-1).")
@click.option("--rgcn-activation", default="sigmoid", show_default=True,
              type=click.Choice(["sigmoid", "relu", "none"]),
              help="Activation de sortie du R-GCN NumPy. relu débloque les gradients "
                   "(sigmoid max 0.25). GAT utilise --gat-output-activation séparément.")
@click.option("--rgcn-layernorm/--no-rgcn-layernorm", default=False, show_default=True,
              help="LayerNorm après l'activation dans le R-GCN NumPy. "
                   "Stabilise les magnitudes des features entre couches.")
@click.option("--max-grad-norm", default=5.0, show_default=True, type=float,
              help="Norme maximale des gradients (gradient clipping). 0 = désactivé.")
@click.option("--max-class-weight", default=5.0, show_default=True, type=float,
              help="Plafond des class weights avec --weighted-loss. Évite qu'une classe "
                   "ultra-rare domine la loss. 0 = pas de plafond.")
@click.option("--min-class-count", default=30, show_default=True, type=int,
              help="N_min v5 : une classe d'arête avec 0 < N < N_min est coupée du "
                   "softmax (ni apprise ni prédite) — le moteur ne prédit que ce "
                   "qu'il peut apprendre. Annotez jusqu'à N_min pour réactiver.")
@click.option("--scheduled-sampling/--no-scheduled-sampling", default=False, show_default=True,
              help="Scheduled sampling : réduit linéairement la probabilité d'injecter "
                   "gold_edge_map en entraînement (1.0 → 0.0 sur ss-final-epoch epochs). "
                   "Réduit l'asymétrie train/val due au teacher forcing R-GCN. "
                   "Requiert --two-pass-val (actif par défaut).")
@click.option("--ss-final-epoch", default=50, show_default=True, type=int,
              help="Epoch (incluse) où p_gold atteint 0.0 avec --scheduled-sampling. "
                   "Avant : p_gold = 1 - (epoch-1)/ss_final_epoch. Après : p_gold = 0.0.")
@click.option("--n-intent-types", default=0, show_default=True, type=int,
              help="Éq.6 : active la tête d'intention (0=désactivé). "
                   "Doit correspondre à len(INTENT_TYPES) si > 0. "
                   "Requiert des phrases annotées avec SentenceRecord.intent.")
@click.option("--n-sentence-types", default=0, show_default=True, type=int,
              help="P4a : active la tête type de phrase (0=désactivé). "
                   "Doit correspondre à len(SENTENCE_TYPES)=4 si > 0. "
                   "Golds dérivés par règle (ponctuation) sauf annotation.")
@click.option("--qual-heads/--no-qual-heads", default=False, show_default=True,
              help="P4 quals : active les 3 têtes edge-qualifiers "
                   "(polarity/2, voice/2, modality/4) sur vecteurs enrichis. "
                   "Chaque (qual, classe) sous N_min est masquée.")
@click.option("--coarse-phase/--no-coarse-phase", default=False, show_default=True,
              help="D4/§15 ETUDE — entraînement au niveau coarse (5 relations, 4 nœuds). "
                   "Masque Mood et Tense. Les types ayant N≥coarse-n-min restent au niveau fin.")
@click.option("--coarse-n-min", default=400, show_default=True, type=int,
              help="Nombre minimum d'exemples par type fin pour promotion coarse→fine.")
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
    edge_dropout: float,
    label_smoothing: float,
    link_pred: bool,
    neg_ratio: float,
    src_aggregation: str,
    bfs_depth: int | None,
    n_rgcn_layers: int,
    edge_threshold: float,
    drop_morph: bool,
    no_positional: bool,
    no_ternary: bool,
    no_mood_flag: bool,
    no_tense_flag: bool,
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
    two_pass_val: bool,
    rgcn_activation: str,
    rgcn_layernorm: bool,
    max_grad_norm: float,
    max_class_weight: float,
    min_class_count: int,
    use_compgcn: bool,
    d_rel_emb: int,
    scheduled_sampling: bool,
    ss_final_epoch: int,
    n_intent_types: int,
    n_sentence_types: int,
    qual_heads: bool,
    coarse_phase: bool,
    coarse_n_min: int,
) -> None:
    """Entraîne le pipeline CGNP (NumPy référence) par descente de gradient."""
    from ..data.verbalize_loader import VerbalizerDataLoader
    from ..verbalizer.trainable import TrainableDecoder

    # R9 : reproductibilité — seed numpy + torch si disponible
    _init_seed = seed if seed is not None else 42
    if seed is not None:
        np.random.seed(seed)
        try:
            import torch as _torch
            _torch.manual_seed(seed)
        except ImportError:
            pass
        click.echo(f"Seed : {seed}")

    vocab = FeatureVocabulary()

    # Embeddings NON OPTIONNELS (REMEDIATION-DIAGNOSTIC.md §2/§11) : activés par
    # défaut (--embedding-dim 128). _dim_from_cli distingue le défaut d'une valeur
    # passée explicitement (pour --fasttext/--embedding-file seuls).
    try:
        _dim_source = click.get_current_context().get_parameter_source("embedding_dim")
        _dim_from_cli = _dim_source != click.core.ParameterSource.DEFAULT
    except RuntimeError:  # appel direct de la fonction (tests) : pas de contexte click
        _dim_from_cli = embedding_dim != 128

    # Préconditions A/B/C/F (amelioration_v3)
    _has_emb = embedding_file is not None or embedding_dim > 0 or fasttext is not None
    if fasttext is not None and embedding_file is not None:
        raise click.ClickException("--fasttext et --embedding-file sont mutuellement exclusifs.")
    if fasttext is not None and _dim_from_cli and embedding_dim not in (0, 300):
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
    if not _has_emb:
        raise click.ClickException(
            "word_embedding obligatoire (D10 ETUDE). "
            "Utilisez --embedding-dim 128 (défaut) ou --embedding-file chemin.bin"
        )

    # S1/S2/S9 + Phase A (fastText multilingue) : word embeddings activés par défaut
    word_embedding = None
    d_emb = 0
    _fasttext_path: Path | None = fasttext
    if _has_emb:
        from ..layer1.embedding import WordEmbedding
        if _fasttext_path is not None:
            # Phase A : d_emb=300 imposé, frozen=True par défaut.
            # Les vecteurs sont remplis après build_vocab (lemmes connus).
            d_emb = 300
            word_embedding = WordEmbedding(d_emb=d_emb, frozen=True, seed=_init_seed)
            click.echo(f"Embeddings fastText : {fasttext} (d_emb=300, frozen)")
        elif embedding_file is not None and (embedding_dim == 0 or not _dim_from_cli):
            # Détecter la dimension depuis la première ligne du fichier
            # (cas --embedding-file seul : le défaut 128 ne doit pas écraser le fichier)
            with open(embedding_file, encoding="utf-8") as ef:
                for line in ef:
                    parts = line.strip().split()
                    if len(parts) > 2:
                        d_emb = len(parts) - 1
                        break
        else:
            d_emb = embedding_dim
        if d_emb > 0:
            word_embedding = WordEmbedding(d_emb=d_emb, seed=_init_seed)
            if embedding_file is not None:
                n_loaded = word_embedding.load_from_file(str(embedding_file))
                click.echo(f"Embeddings : {n_loaded} vecteurs chargés (d_emb={d_emb})")
            if freeze_embeddings:
                # C : gel des pré-entraînés (spéciaux _absent toujours entraînables)
                word_embedding.frozen = True
                click.echo("Embeddings pré-entraînés gelés (--freeze-embeddings)")

    # v5.8 (P3) : le chargeur et la passe de comptage précèdent la construction —
    # le remap coarse doit être connu AVANT de dimensionner encodeur/graphe.
    loader = GCNDataLoader(data_dir, all_pairs=all_pairs, shuffle=True,
                           silver_weight=silver_weight, seed=_init_seed)
    if len(loader) == 0:
        raise click.ClickException(f"Aucune sentence dans {data_dir}")
    val_loader = None
    if val_dir is not None:
        val_loader = GCNDataLoader(val_dir, all_pairs=all_pairs, shuffle=False,
                                   silver_weight=silver_weight)
        if len(val_loader) == 0:
            raise click.ClickException(f"Aucune sentence dans {val_dir}")
        click.echo(f"Val : {len(val_loader)} sentences")

    # Passe unique sur le dataset : edge logit mask + optional weights/vocab
    node_class_weights = None
    edge_class_weights = None
    _edge_logit_mask: np.ndarray | None = None
    from collections import Counter
    node_counts: Counter = Counter()
    edge_counts: Counter = Counter()
    intent_counts: Counter = Counter()
    sentence_counts: Counter = Counter()
    all_lemmas: list[str] = []
    for sample in loader:
        for rel in sample.edge_map.values():
            edge_counts[int(rel)] += 1
        # v5.8 (P3) : node_counts toujours rempli — le gate weighted_loss
        # laissait _promoted_node vide et basculait tout en coarse.
        for label in sample.gold_node_labels:
            node_counts[int(label)] += 1
        # Éq.6 : comptage des labels d'intention (pour le masque et l'activation N_min)
        if n_intent_types > 0 and sample.sentence.intent:
            with contextlib.suppress(ValueError):
                intent_counts[INTENT_TYPES.index(sample.sentence.intent)] += 1
        # P4a : comptage des types de phrases (annotés ou règle bootstrap).
        if n_sentence_types > 0 and sample.sentence_type:
            with contextlib.suppress(ValueError):
                sentence_counts[SENTENCE_TYPES.index(sample.sentence_type)] += 1
        if word_embedding is not None:
            reps_s, _, conn_s = reps_from_sentence(sample.sentence)
            # A : avec pooling != root, élargir le vocab aux lemmes de contenu ;
            # B : ajouter les lemmes sujet/objet (les _absent sont pré-enregistrés)
            if clause_pooling == "root" and not subject_object_emb:
                all_lemmas.extend(r.root_lemma for r in reps_s)
            else:
                from ..layer1.features import _find_subj_obj_lemmas, _pool_lemmas
                for r in reps_s:
                    if clause_pooling == "root":
                        all_lemmas.append(r.root_lemma)
                    else:
                        all_lemmas.extend(_pool_lemmas(r, clause_pooling))
                    if subject_object_emb:
                        all_lemmas.extend(_find_subj_obj_lemmas(r))
            # P4a : marques illocutoires "?" / "!" au vocabulaire — sinon
            # le pooling mean de la tête phrase les résout en zéros OOV et
            # l'interrogative reste invisible (vecteurs figés à l'init si le
            # routage embeddings est en mode root : séparables mais non appris —
            # même statut assumé que les connecteurs orphelins, cf. garde).
            from ..layer1.features import SENTENCE_MARK_FORMS as _SENT_MARKS
            all_lemmas.extend(
                t.get("lemma", "") for r in reps_s for t in r.tokens
                if t.get("form") in _SENT_MARKS and t.get("lemma")
            )
            # Marqueurs : les lemmes de connecteurs (gold ou redécouverts) entrent
            # au vocabulaire pour que WordEmbedding apprenne leurs vecteurs.
            # Sans connecteur annoté ni gap syntaxique, rien n'est ajouté.
            # v5.3 : ces vecteurs s'entraînent par le chemin nœuds (vocab partagé),
            # pas par le chemin arête (gradient edge_vec_base non routé — assumé :
            # routage dédié seulement si un lemme exclusif apparaît, cf. garde).
            _conn_lemmas = [
                _c.root_lemma for _c in conn_s
                if _c is not None and getattr(_c, "root_lemma", "") not in ("", "_unknown")
            ]
            all_lemmas.extend(_conn_lemmas)
            _clause_lemmas = {t.get("lemma", "") for r in reps_s for t in r.tokens}
            _orphans = sorted(set(_conn_lemmas) - _clause_lemmas - {""})
            if _orphans:
                import warnings as _w_orph
                _w_orph.warn(
                    f"Lemmes connecteurs sans chemin de gradient (hors vocab clauses) : "
                    f"{_orphans} — vecteurs gelés tant que le routage edge n'existe pas.",
                    UserWarning, stacklevel=2,
                )

    # D4/§15 ETUDE — mode coarse : promotion des types ayant N≥coarse_n_min.
    # v5.8 (P3) : calculé ici (comptes connus) pour dimensionner encodeur/graphe.
    # Types promus gardent leur label fin ; les autres sont remappés vers le groupe coarse.
    _active_node_types = NODE_TYPES
    _active_relation_types = RELATION_TYPES
    _node_remap: dict[int, int] | None = None
    _edge_remap: dict[int, int] | None = None
    if coarse_phase:
        _promoted_rel = {RELATION_TYPES[c] for c, n in edge_counts.items() if n >= coarse_n_min}
        _promoted_node = {NODE_TYPES[c] for c, n in node_counts.items() if n >= coarse_n_min}
        # Types actifs = promoted (fine) + groupes coarse des non-promoted
        _fine_to_active_rel: dict[str, str] = {}
        _fine_to_active_node: dict[str, str] = {}
        for r in RELATION_TYPES:
            _fine_to_active_rel[r] = r if r in _promoted_rel else coarse_relation(r)
        for n in NODE_TYPES:
            _fine_to_active_node[n] = n if n in _promoted_node else coarse_node(n)
        _active_relation_types = sorted(set(_fine_to_active_rel.values()),
                                        key=lambda x: (x not in RELATION_TYPES, x))
        _active_node_types = sorted(set(_fine_to_active_node.values()),
                                    key=lambda x: (x not in NODE_TYPES, x))
        # Tables de remapping (index fin → index actif)
        _active_rel_idx = {r: i for i, r in enumerate(_active_relation_types)}
        _active_node_idx = {n: i for i, n in enumerate(_active_node_types)}
        _edge_remap = {i: _active_rel_idx[_fine_to_active_rel[r]]
                       for i, r in enumerate(RELATION_TYPES)}
        _node_remap = {i: _active_node_idx[_fine_to_active_node[n]]
                       for i, n in enumerate(NODE_TYPES)}
        click.echo(
            f"  [coarse] {len(_active_relation_types)} relations actives, "
            f"{len(_active_node_types)} nœuds actifs. "
            f"Promoted fine: rel={sorted(_promoted_rel)}, node={sorted(_promoted_node)}"
        )

    # B : point unique de vérité pour la dimension effective
    d_effective = vocab.d_clause_effective(d_emb, subject_object_emb)
    # Closed-loop : edge MLP reçoit features + enriched vectors + node probs.
    # v5.8 (P3) : dims sur types ACTIFS (post-remap), pas sur 19/8.
    n_node_types = len(_active_node_types)
    n_relation_types = len(_active_relation_types)
    d_edge_closed = vocab.d_edge_closed_loop(d_effective, n_node_types, d_emb,
                                                 subject_object_emb)
    # P4b : couplage intent←sentence automatique quand les deux têtes sont
    # actives (distribution détachée en entrée intent). Persisté en arch.
    _intent_conditioned = bool(n_intent_types > 0 and n_sentence_types > 0)
    if _intent_conditioned:
        click.echo("  [P4b] intent conditionnée par sentence_type (concat distribution).")
    # Phase C : substitution MLPEncoder → TransformerMLPEncoder (MHA globale).
    # d_clause = D_effective (inclut déjà d_emb), jamais vocabulary.d_clause brut.
    if not (0.0 <= edge_dropout < 1.0):
        raise click.ClickException(f"--edge-dropout doit être dans [0, 1[ (reçu {edge_dropout}).")
    if global_attention:
        from ..layer2.reference import TransformerMLPEncoder
        try:
            encoder = TransformerMLPEncoder(
                d_clause=d_effective, d_edge=d_edge_closed,
                weight_decay=weight_decay, mlp_hidden=mlp_hidden,
                edge_dropout=edge_dropout,
                n_node_types=n_node_types,
                n_relation_types=n_relation_types,
                n_sentence_types=n_sentence_types,
                n_intent_types=n_intent_types,
                intent_conditioned=_intent_conditioned,
                qual_heads=qual_heads,
                n_heads=mha_heads, seed=_init_seed)
        except ValueError as _e:
            raise click.ClickException(str(_e)) from _e
        click.echo(f"MHA globale : n_heads={mha_heads} (TransformerMLPEncoder, poids fixes)")
    else:
        if mha_heads != 4:
            raise click.ClickException("--mha-heads requiert --global-attention.")
        encoder = MLPEncoder(d_clause=d_effective, d_edge=d_edge_closed,
                             weight_decay=weight_decay, mlp_hidden=mlp_hidden,
                             edge_dropout=edge_dropout,
                             n_node_types=n_node_types,
                             n_relation_types=n_relation_types,
                             n_sentence_types=n_sentence_types,
                             n_intent_types=n_intent_types,
                             intent_conditioned=_intent_conditioned,
                             qual_heads=qual_heads,
                             seed=_init_seed)

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
    # v5.2 : +1 type no-edge R-GCN (jamais prédit, message passing seul).
    # v5.8 (P3) : sur types actifs (coarse) — pas 19/8 figés.
    # Anciens checkpoints (38/19) refusés bruyamment au chargement (triplet).
    n_rel = rgcn_n_relations(n_relation_types, bidirectional)  # L-6
    if use_attention:
        from ..layer3.gat import RGCNLayerGAT
        graph = RGCNLayerGAT(d_in=d_effective, d_out=d_effective, n_relations=n_rel,
                             dropout=rgcn_dropout, n_heads=n_gat_heads,
                             use_layernorm=gat_layernorm, seed=_init_seed)
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
                                use_compgcn=use_compgcn, d_rel_emb=d_rel_emb,
                                seed=_init_seed)
            click.echo(f"RGCNLayerPT : pairnorm={pairnorm} drop_edge={drop_edge} "
                       f"use_compgcn={use_compgcn} d_rel_emb={d_rel_emb}")
            if not hasattr(graph, "backward_message_pass"):
                raise click.ClickException(
                    "RGCNLayerPT : backward_message_pass absent — matrices W_r/E_r/W_comp "
                    "seraient gelées à l'initialisation (seul le MLP serait entraîné). "
                    "Pour entraîner le R-GCN, utiliser un optimizer PyTorch externe via "
                    "graph.torch_parameters()."
                )
        else:
            graph = RGCNLayer(d_in=d_effective, d_out=d_effective, n_relations=n_rel,
                              dropout=rgcn_dropout, output_activation=rgcn_activation,
                              use_layernorm=rgcn_layernorm, seed=_init_seed)

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
            assembler = LexicalConnectorAssembler(connector_vocab, n_relations=len(RELATION_TYPES))
            verbalize_mode = "lexical"
            click.echo(f"Assembleur lexical : {len(connector_vocab)} connecteurs (mode lexical)")
        elif legacy_decoder:
            click.echo("Décodeur legacy explicite (--legacy-decoder)")

    if decoder_only and decoder is None:
        raise click.ClickException("--decoder-only requiert --verbalize-dir")

    if scheduled_sampling and not two_pass_val:
        raise click.ClickException(
            "--scheduled-sampling requiert --two-pass-val (actif par défaut). "
            "Sans two-pass, l'inférence utilise des types 0 partout."
        )
    if ss_final_epoch < 1:
        raise click.ClickException(f"--ss-final-epoch doit être ≥ 1 (reçu {ss_final_epoch}).")

    if patience > 0 and val_dir is None:
        raise click.ClickException("--patience requiert --val-dir")
    if val_dir is not None and patience == 0:
        click.echo("Avertissement : --val-dir sans --patience — pas d'early stopping. "
                   "Ajoutez --patience 15 pour activer l'arrêt anticipé.")

    # B4 : validation --n-rgcn-layers (restriction GAT levée — E4)
    if n_rgcn_layers < 0:
        raise click.ClickException(f"--n-rgcn-layers doit être ≥ 0 (reçu {n_rgcn_layers}).")
    if n_rgcn_layers == 0:
        click.echo("R-GCN désactivé (--n-rgcn-layers 0) — pipeline MLP seul.")
    if bfs_depth is not None and bfs_depth < 1:
        raise click.ClickException(f"--bfs-depth doit être ≥ 1 (reçu {bfs_depth}).")

    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab,
                           decoder=decoder, all_pairs=all_pairs, word_embedding=word_embedding,
                           bidirectional=bidirectional, n_rgcn_layers=n_rgcn_layers,
                           edge_threshold=edge_threshold, drop_morph=drop_morph,
                           bfs_depth=bfs_depth, clause_pooling=clause_pooling,
                           subject_object_emb=subject_object_emb,
                           node_types=_active_node_types,
                           relation_types=_active_relation_types,
                            gat_residual=gat_residual, assembler=assembler,
                            no_mood=(coarse_phase or no_mood_flag),
                            no_tense=(coarse_phase or no_tense_flag),
                            no_positional=no_positional,
                            no_ternary=no_ternary,
                           n_intent_types=n_intent_types,
                           n_sentence_types=n_sentence_types, seed=_init_seed)
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
    pipeline.two_pass_val = two_pass_val

    link_pred_head = None
    if link_pred:
        from ..layer3.link_pred import LinkPredHead
        link_pred_head = LinkPredHead(d_in=d_effective, seed=_init_seed,
                                      src_aggregation=src_aggregation)
        pipeline.link_predictor = link_pred_head
        click.echo(f"LinkPredHead : d_in={d_effective} src_agg={src_aggregation} "
                   f"neg_ratio={neg_ratio} bfs_depth={bfs_depth}")

    if encoder_checkpoint is not None:
        from .checkpoint import load_checkpoint
        load_checkpoint(pipeline, encoder_checkpoint, trusted=True)
        click.echo(f"Checkpoint encodeur chargé : {encoder_checkpoint}")

    # (Chargeur déjà construit avant l'encodeur — v5.8 P3 : le remap coarse
    # doit précéder le dimensionnement.)
    # Provenance — enregistrée dans _arch_json pour traçabilité complète.
    import datetime as _dt
    import hashlib as _hl
    _h = _hl.sha256()
    for _p in sorted(data_dir.glob("*.json")):
        _h.update(_p.read_bytes())
    pipeline.training_seed = _init_seed
    pipeline.training_data_hash = _h.hexdigest()[:16]
    pipeline.training_n_epochs = epochs
    pipeline.training_timestamp = _dt.datetime.now(_dt.timezone.utc).isoformat()
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

    # (Chargeurs déjà construits avant l'encodeur — v5.8 P3.)
    # (Comptes déjà calculés avant construction — v5.8 P3. Le masque suit.)
    if edge_counts:
        # v5.8 (P3) : comptes en espace ACTIF (fines remappées si coarse).
        _active_edge_counts: dict[int, int] = {}
        if _edge_remap is not None:
            for _fc, _n in edge_counts.items():
                _ac = _edge_remap.get(int(_fc), int(_fc))
                _active_edge_counts[_ac] = _active_edge_counts.get(_ac, 0) + _n
        else:
            _active_edge_counts = {int(c): int(n) for c, n in edge_counts.items()}
        n_edge_classes = encoder.n_relation_types
        assert n_edge_classes == len(_active_relation_types), (
            f"encodeur {n_edge_classes} ≠ {len(_active_relation_types)} types actifs — "
            "construction incohérente (P3).")
        _edge_logit_mask = np.zeros(n_edge_classes, dtype=bool)
        for c in range(n_edge_classes):
            if _active_edge_counts.get(c, 0) > 0:
                _edge_logit_mask[c] = True
        n_active = int(_edge_logit_mask.sum())
        n_inactive = n_edge_classes - n_active
        if n_inactive > 0:
            click.echo(
                f"  [v3.0] {n_inactive} classe(s) arête vide(s) masquées du softmax "
                f"(N=0 dans le dataset) — masquées tant que N=0."
            )
        # v5 : coupe aux prouvées — 0 < N < N_min ni apprise ni prédite.
        # Le moteur ne prédit que ce qu'il peut apprendre ; annotez jusqu'à
        # N_min pour réactiver (0 = désactive la coupe, comportement historique).
        if min_class_count > 0:
            _rare = [(_active_relation_types[c], _active_edge_counts.get(c, 0))
                     for c in range(n_edge_classes)
                     if _edge_logit_mask[c] and _active_edge_counts.get(c, 0) < min_class_count]
            for _rel, _n in _rare:
                _edge_logit_mask[_active_relation_types.index(_rel)] = False
            if _rare:
                _names = ", ".join(f"{r} (N={n})" for r, n in _rare)
                click.echo(
                    f"  [v5] {len(_rare)} classe(s) sous N_min={min_class_count} "
                    f"coupée(s) du softmax : {_names}."
                )
            if not _edge_logit_mask.any():
                raise click.ClickException(
                    "Aucune classe d'arête ne survit (vides + sous N_min="
                    f"{min_class_count}) : annotez au moins {min_class_count} "
                    "exemples d'une relation, ou baissez --min-class-count "
                    "(0 = désactive la coupe)."
                )
        # Couverture marqueurs (guide réannotation v5) : gold annotés vs
        # connecteurs retrouvés vs gaps sans connecteur.
        _n_gold = _n_found = _n_gaps = 0
        for _s in loader:
            _rs, _, _cs = reps_from_sentence(_s.sentence)
            _n_gaps += max(0, len(_rs) - 1)
            _n_gold += sum(1 for _e in _s.sentence.edges
                           if getattr(_e, "marker_token", None) is not None)
            _n_found += sum(1 for _c in _cs if _c is not None)
        click.echo(
            f"  [v5] marqueurs : {_n_gold} gold annotés, {_n_found}/{_n_gaps} "
            f"gaps avec connecteur."
        )
        # Masque forward+loss : le pipeline émet et apprend uniquement sur les
        # classes vues (persisté au checkpoint, appliqué à l'inférence).
        pipeline.edge_logit_mask = _edge_logit_mask
    # Éq.6 — masque intent logits (même logique que _edge_logit_mask)
    _intent_logit_mask: np.ndarray | None = None
    if n_intent_types > 0 and intent_counts:
        _intent_logit_mask = np.zeros(n_intent_types, dtype=bool)
        for c in range(n_intent_types):
            if intent_counts.get(c, 0) > 0:
                _intent_logit_mask[c] = True
        n_active_intent = int(_intent_logit_mask.sum())
        click.echo(f"  [Éq.6] {n_active_intent}/{n_intent_types} types d'intention actifs.")
    # P4a — masque types de phrases (même logique que _intent_logit_mask ;
    # N_min v5 s'applique aussi : pas d'apprentissage sous le seuil).
    _sentence_logit_mask: np.ndarray | None = None
    if n_sentence_types > 0 and sentence_counts:
        _sentence_logit_mask = np.zeros(n_sentence_types, dtype=bool)
        for c in range(n_sentence_types):
            if sentence_counts.get(c, 0) > 0:
                _sentence_logit_mask[c] = True
        n_active_sent = int(_sentence_logit_mask.sum())
        click.echo(f"  [P4a] {n_active_sent}/{n_sentence_types} types de phrases actifs.")
    # P4 quals — comptage par (qual, classe) et masques N_min.
    # Sans annotations variées, tout est masqué : dortoir, pas de loss.
    qual_counts: dict[str, Counter] = {q: Counter() for q in QUALIFIERS}
    if qual_heads:
        for _smp in loader:
            for _qm in _smp.qual_map.values():
                for _qn, _qi in _qm.items():
                    if _qn in qual_counts:
                        qual_counts[_qn][int(_qi)] += 1
    _qual_logit_masks: dict[str, np.ndarray] | None = None
    if qual_heads:
        _qual_logit_masks = {}
        for _qn, _classes in QUALIFIERS.items():
            _m = np.zeros(len(_classes), dtype=bool)
            for c in range(len(_classes)):
                if qual_counts[_qn].get(c, 0) >= min_class_count > 0 or (min_class_count <= 0 and qual_counts[_qn].get(c, 0) > 0):
                    _m[c] = True
            _qual_logit_masks[_qn] = _m
        click.echo("  [P4 quals] actives : " + ", ".join(
            f"{_qn}={int(_qual_logit_masks[_qn].sum())}/{len(_c)}"
            for _qn, _c in QUALIFIERS.items()))

    # (Remap coarse déjà calculé avant construction — v5.8 P3. L'encodeur
    # est dimensionné sur types actifs ; le masque suit en espace actif.
    # Remaps exposés sur le pipeline pour la passe éval.)
    pipeline._coarse_node_remap = _node_remap
    pipeline._coarse_edge_remap = _edge_remap
    pipeline._coarse_phase = bool(coarse_phase)
    pipeline._coarse_n_min = int(coarse_n_min)

    if weighted_loss:
        # v5.8 (P3) : poids en espace ACTIF (coarse remappé), noms actifs.
        _active_node_counts: dict[int, int] = {}
        if _node_remap is not None:
            for _fc, _n in node_counts.items():
                _ac = _node_remap.get(int(_fc), int(_fc))
                _active_node_counts[_ac] = _active_node_counts.get(_ac, 0) + _n
        else:
            _active_node_counts = {int(c): int(n) for c, n in node_counts.items()}
        if _active_node_counts:
            total_nodes = sum(_active_node_counts.values())
            n_node_classes = len(_active_node_types)
            node_class_weights = np.zeros(n_node_classes, dtype=np.float32)
            for c in range(n_node_classes):
                count = _active_node_counts.get(c, 1)
                node_class_weights[c] = total_nodes / (n_node_classes * count)
            if max_class_weight > 0:
                node_class_weights = np.clip(node_class_weights, 0, max_class_weight)
            click.echo(f"Node class weights : {dict(zip(_active_node_types, node_class_weights.round(3), strict=False))}")
        if edge_counts:
            total_edges = sum(_active_edge_counts.values())
            edge_class_weights = np.zeros(n_edge_classes, dtype=np.float32)
            for c in range(n_edge_classes):
                count = _active_edge_counts.get(c, 0)
                if count > 0:
                    edge_class_weights[c] = total_edges / (n_edge_classes * count)
            if max_class_weight > 0:
                edge_class_weights = np.clip(edge_class_weights, 0, max_class_weight)
            click.echo(f"Edge class weights : {dict(zip(_active_relation_types, edge_class_weights.round(3), strict=False))}")
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

    # Rng local pour scheduled sampling — séquence déterministe sans polluer le global.
    _ss_rng = np.random.default_rng(_init_seed + 1)

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
    best_val_edge_f1 = -1.0
    best_edge_epoch = 0
    best_edge_checkpoint_path = str(output) + ".best_edge.npz"

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
        _n_cut_val = 0  # v5.1 : exemples val de classes coupées, hors mesure
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
            # v5.8 (P3) : remap coarse éval — mêmes tables que le train.
            _ev_nremap = getattr(pipeline, '_coarse_node_remap', None)
            if _ev_nremap is not None and gold_node is not None:
                gold_node = np.array([_ev_nremap.get(int(i), int(i)) for i in gold_node],
                                     dtype=np.int64)

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
                if _edge_logit_mask is not None and valid_edge_mask.any():
                    # v5.1 : classes coupées hors supervision — ni apprises,
                    # ni prédites, ni punies (le masque seul punissait via -log(~0)).
                    _active = np.asarray(_edge_logit_mask, dtype=bool)
                    _keep = _active[np.clip(gold_edge_full, 0, len(_active) - 1)]
                    _n_cut_val += int((valid_edge_mask & ~_keep).sum())
                    valid_edge_mask = valid_edge_mask & _keep
                if valid_edge_mask.any():
                    valid_edge_idxs = np.where(valid_edge_mask)[0]
                    gold_edge = gold_edge_full[valid_edge_idxs]
                    _ev_eremap = getattr(pipeline, '_coarse_edge_remap', None)
                    if _ev_eremap is not None:
                        gold_edge = np.array(
                            [_ev_eremap.get(int(i), int(i)) for i in gold_edge],
                            dtype=np.int64)
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
                edge_logit_mask=_edge_logit_mask if edge_counts else None,
                sample_weight=sample.sentence.weight,
                edge_sample_weights=_val_edge_sw,  # S-5 cohérence val
            )
            total_loss += loss_val
            n += 1

            node_pred_idxs = np.argmax(node_logits, axis=1)
            epoch_node_preds.extend(pipeline.node_types[i] for i in node_pred_idxs)
            epoch_node_gold.extend(pipeline.node_types[i] for i in gold_node)

            sent_node_pred = [pipeline.node_types[i] for i in node_pred_idxs]
            sent_node_gold = [pipeline.node_types[i] for i in gold_node]
            epoch_sent_node_preds.append(sent_node_pred)
            epoch_sent_node_gold.append(sent_node_gold)

            if gold_edge is not None and edge_logits_arg is not None and len(edge_logits_arg) > 0:
                edge_pred_idxs = np.argmax(edge_logits_arg, axis=1)
                epoch_edge_preds.extend(pipeline.relation_types[i] for i in edge_pred_idxs)
                epoch_edge_gold.extend(pipeline.relation_types[i] for i in gold_edge)
                epoch_sent_edge_preds.append([pipeline.relation_types[i] for i in edge_pred_idxs])
                epoch_sent_edge_gold.append([pipeline.relation_types[i] for i in gold_edge])
            else:
                epoch_sent_edge_preds.append([])
                epoch_sent_edge_gold.append([])

        if n_skipped > 0:
            import warnings as _wvs
            _wvs.warn(f"_run_eval_pass : {n_skipped} phrase(s) ignorée(s) sur {n + n_skipped} — "
                      f"val_f1 calculée sur {n} phrase(s) seulement.",
                      UserWarning, stacklevel=3)
        return total_loss / max(n, 1), _n_cut_val

    try:
        # B2 : ouverture du CSV ici (dans le try) pour garantir la fermeture
        # en cas d'exception pendant le setup ultérieur (class weights, etc.).
        if log_csv:
            csv_fieldnames = [
                "epoch", "loss", "node_accuracy", "node_macro_f1",
                "edge_accuracy", "edge_macro_f1", "graph_exact_match",
                "edge_cut_train", "sentence_accuracy",
                "qual_polarity_accuracy", "qual_voice_accuracy",
                "qual_modality_accuracy",
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
                    "edge_cut_val",
                ])
            csv_file = open(log_csv, "w", newline="", encoding="utf-8")  # noqa: SIM115  # handle référencé puis fermé dans le finally de train()
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
            epoch_sent_type_preds: list[str] = []  # P4a
            epoch_sent_type_gold: list[str] = []  # P4a
            epoch_intent_preds: list[str] = []  # P4b (noms, "(none)" si sans gold)
            epoch_intent_gold: list[str] = []  # P4b
            epoch_sent_intent_gold: list[tuple[str, str]] = []  # P4b co-occurrences
            epoch_qual_preds: dict[str, list[str]] = {}  # P4 quals (noms par tête)
            epoch_qual_gold: dict[str, list[str]] = {}  # P4 quals
            epoch_sent_node_preds: list[list[str]] = []
            epoch_sent_node_gold: list[list[str]] = []
            epoch_sent_edge_preds: list[list[str]] = []
            epoch_sent_edge_gold: list[list[str]] = []
            _asm_pred_idxs: list[int] = []  # G3 : connecteurs prédits (par arête)
            _asm_gold_idxs: list = []       # G3 : connecteurs gold (None exclus)
            # P0-6 : compteur de samples sautés SANS warn (clauses vides, reps
            # vides, logits vides) — une epoch à 0 sample ne doit pas être silencieuse.
            _n_skip_silent = 0
            epoch_n_cut = 0  # v5.1 : exemples train de classes coupées, hors loss

            for sample in loader:
                if not sample.sentence.clauses:
                    _n_skip_silent += 1
                    continue

                try:
                    reps, valid_clause_idxs, connector_reps = reps_from_sentence(sample.sentence)
                    if not reps:
                        _n_skip_silent += 1
                        continue
                    if scheduled_sampling and ss_final_epoch > 0:
                        _p_gold = max(0.0, 1.0 - (epoch - 1) / ss_final_epoch)
                        _gold_map = sample.edge_map if (_ss_rng.random() < _p_gold) else None
                    else:
                        _gold_map = sample.edge_map  # teacher forcing standard
                    pipeline.forward(
                        reps, sample.sentence.text,
                        clause_positions=valid_clause_idxs,
                        n_total_clauses=len(sample.sentence.clauses),
                        connector_reps=connector_reps,
                        gold_edge_map=_gold_map,
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
                    _n_skip_silent += 1
                    continue
                _gold_quals = None  # P4 quals : défini dans la branche paires ci-dessous

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
                    if _edge_logit_mask is not None and valid_edge_mask.any():
                        # v5.1 : voir passe éval — classes coupées hors loss.
                        _active = np.asarray(_edge_logit_mask, dtype=bool)
                        _keep = _active[np.clip(gold_edge_full, 0, len(_active) - 1)]
                        epoch_n_cut += int((valid_edge_mask & ~_keep).sum())
                        valid_edge_mask = valid_edge_mask & _keep
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
                        # P4 quals — golds alignés sur les paires valides.
                        _gold_quals = None
                        if qual_heads and sample.qual_map:
                            _gold_quals = {}
                            for _qn in QUALIFIERS:
                                _gq = np.array([
                                    sample.qual_map.get(pairs[i], {}).get(_qn, 0)
                                    for i in valid_edge_idxs
                                ], dtype=np.int64)
                                if len(_gq):
                                    _gold_quals[_qn] = _gq
                    else:
                        gold_edge = None
                        edge_logits_arg = None
                        _edge_sw = None
                        _gold_quals = None
                else:
                    gold_edge = None
                    edge_logits_arg = None
                    _edge_sw = None
                    _gold_quals = None

                # D4 — remapping labels coarse si actif
                if _node_remap is not None and gold_node is not None:
                    gold_node = np.array([_node_remap.get(int(i), int(i)) for i in gold_node],
                                         dtype=np.int64)
                if _edge_remap is not None and gold_edge is not None:
                    gold_edge = np.array([_edge_remap.get(int(i), int(i)) for i in gold_edge],
                                         dtype=np.int64)

                _gold_surface = None
                if verb_source_map:
                    _surfaces = verb_source_map.get(sample.sentence.text, [])
                    if _surfaces:
                        _gold_surface = _surfaces[0]

                # Éq.6 — label d'intention (si annoté et tête active)
                _gold_intent = None
                if n_intent_types > 0 and sample.sentence.intent:
                    with contextlib.suppress(ValueError):
                        _gold_intent = np.array(
                            [INTENT_TYPES.index(sample.sentence.intent)], dtype=np.int64
                        )

                # P4a — label type de phrase (annoté > règle bootstrap, cf. loader).
                _gold_sentence = None
                if n_sentence_types > 0 and sample.sentence_type:
                    with contextlib.suppress(ValueError):
                        _gold_sentence = np.array(
                            [SENTENCE_TYPES.index(sample.sentence_type)], dtype=np.int64
                        )

                loss_val, d_node, d_edge = pipeline.loss(
                    node_logits, edge_logits_arg, gold_node, gold_edge,
                    edge_loss_weight=edge_loss_weight,
                    gold_surface=_gold_surface,
                    node_class_weights=node_class_weights,
                    edge_class_weights=edge_class_weights,
                    edge_logit_mask=_edge_logit_mask if edge_counts else None,
                    label_smoothing=label_smoothing,
                    sample_weight=sample.sentence.weight,
                    edge_sample_weights=_edge_sw,  # S-5 : confidence par arête
                    gold_intent=_gold_intent,
                    intent_logit_mask=_intent_logit_mask,
                    gold_sentence=_gold_sentence,
                    sentence_logit_mask=_sentence_logit_mask,
                    gold_quals=_gold_quals if qual_heads else None,
                    qual_logit_masks=_qual_logit_masks,
                )
                # P4a : exactitude type de phrase (tête supervisée si gold).
                # P4 quals : exactitudes par tête (dortoir sans annotations variées).
                _ql_cached = pipeline._cached_qual_logits or {}
                for _qn, _classes in QUALIFIERS.items():
                    _gq_all = (_gold_quals or {}).get(_qn)
                    _ql_all = _ql_cached.get(_qn)
                    if _gq_all is None or _ql_all is None or len(_gq_all) == 0:
                        continue
                    _nq = min(len(_gq_all), len(_ql_all))
                    for _gi, _li in zip(_gq_all[:_nq], _ql_all[:_nq], strict=False):
                        _gi, _li = int(_gi), int(np.argmax(_li))
                        if 0 <= _gi < len(_classes) and 0 <= _li < len(_classes):
                            epoch_qual_preds.setdefault(_qn, []).append(_classes[_li])
                            epoch_qual_gold.setdefault(_qn, []).append(_classes[_gi])
                _sent_logits = pipeline._cached_sentence_logits
                if (_gold_sentence is not None and _sent_logits is not None
                        and len(_sent_logits) > 0):
                    _sp = int(np.argmax(_sent_logits[0]))
                    _sg = int(_gold_sentence[0])
                    if 0 <= _sp < len(SENTENCE_TYPES) and 0 <= _sg < len(SENTENCE_TYPES):
                        epoch_sent_type_preds.append(SENTENCE_TYPES[_sp])
                        epoch_sent_type_gold.append(SENTENCE_TYPES[_sg])
                # P4b : prédictions intent + co-occurrence gold (sentence × intent).
                _intent_logits = pipeline._cached_intent_logits
                if (_gold_intent is not None and _intent_logits is not None
                        and len(_intent_logits) > 0):
                    _ip = int(np.argmax(_intent_logits[0]))
                    _ig = int(_gold_intent[0])
                    if 0 <= _ip < len(INTENT_TYPES) and 0 <= _ig < len(INTENT_TYPES):
                        epoch_intent_preds.append(INTENT_TYPES[_ip])
                        epoch_intent_gold.append(INTENT_TYPES[_ig])
                        if _gold_sentence is not None:
                            _sg = int(_gold_sentence[0])
                            if 0 <= _sg < len(SENTENCE_TYPES):
                                epoch_sent_intent_gold.append(
                                    (SENTENCE_TYPES[_sg], INTENT_TYPES[_ig]))
                # P0-7 : loss non finie → STOP avec contexte (pas de training
                # continué sur gradients corrompus ; loss() ne fait que warner).
                if not np.isfinite(loss_val):
                    raise ValueError(
                        f"[epoch {epoch}] loss non finie ({loss_val!r}) sur phrase "
                        f"{sample.sentence.id!r} — arrêt (overflow softmax, labels "
                        "corrompus ?). Corrigez les données ou l'architecture."
                    )

                if not decoder_only:
                    _mgn = max_grad_norm if max_grad_norm > 0 else None
                    if mini_batch_size <= 1:
                        pipeline.backward(d_node, d_edge, lr=lr,
                                          weight_decay=weight_decay,
                                          max_grad_norm=_mgn)
                    else:
                        pipeline.backward_accumulate(d_node, d_edge,
                                                    max_grad_norm=_mgn)
                        batch_step_count += 1
                        if batch_step_count >= mini_batch_size:
                            pipeline.apply_accumulated_gradients(
                                lr, n_samples=batch_step_count,
                                weight_decay=weight_decay)
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
                                _gW += _g[0]
                                _gb += _g[1]
                                _n_lp += 1
                            for (_a, _b) in _neg:
                                _, _g = link_pred_head.loss_and_grad(_ev[_a], _ev[_b], 0)
                                _gW += _g[0]
                                _gb += _g[1]
                                _n_lp += 1
                            if _n_lp:
                                link_pred_head.update([_gW / _n_lp, _gb / _n_lp], lr)

                node_pred_idxs = np.argmax(node_logits, axis=1)
                if valid_clause_idxs:
                    gold_node_aligned = sample.gold_node_labels[
                        np.array(valid_clause_idxs, dtype=np.int64)
                    ]
                else:
                    gold_node_aligned = sample.gold_node_labels
                epoch_node_preds.extend(pipeline.node_types[i] for i in node_pred_idxs)
                epoch_node_gold.extend(pipeline.node_types[i] for i in gold_node_aligned)

                sent_node_pred = [pipeline.node_types[i] for i in node_pred_idxs]
                sent_node_gold = [pipeline.node_types[i] for i in gold_node_aligned]
                epoch_sent_node_preds.append(sent_node_pred)
                epoch_sent_node_gold.append(sent_node_gold)

                if gold_edge is not None and edge_logits_arg is not None and len(edge_logits_arg) > 0:
                    edge_pred_idxs = np.argmax(edge_logits_arg, axis=1)
                    epoch_edge_preds.extend(pipeline.relation_types[i] for i in edge_pred_idxs)
                    epoch_edge_gold.extend(pipeline.relation_types[i] for i in gold_edge)
                    epoch_sent_edge_preds.append([pipeline.relation_types[i] for i in edge_pred_idxs])
                    epoch_sent_edge_gold.append([pipeline.relation_types[i] for i in gold_edge])
                else:
                    epoch_sent_edge_preds.append([])
                    epoch_sent_edge_gold.append([])

                epoch_loss += loss_val
                n_samples += 1

            if mini_batch_size > 1 and batch_step_count > 0 and not decoder_only:
                pipeline.apply_accumulated_gradients(
                    lr, n_samples=batch_step_count, weight_decay=weight_decay)
                batch_step_count = 0

            if verb_loader is not None and pipeline.decoder is not None:
                from ..data.verbalize_loader import _node_type_embeddings
                from ..verbalizer.trainable import SurfaceVocabulary as _SV
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

            # P0-6 : skips silencieux visibles par epoch ; epoch vide = STOP
            # (avant : avg_loss = 0.0 enregistré sans aucun signal).
            if _n_skip_silent:
                warnings.warn(
                    f"[epoch {epoch}] {_n_skip_silent} sample(s) ignoré(s) sans signal "
                    "(clauses vides, reps vides ou logits vides).",
                    UserWarning, stacklevel=2,
                )
            if n_samples == 0:
                raise ValueError(
                    f"[epoch {epoch}] 0 sample entraîné ({_n_skip_silent} ignoré(s)) — "
                    "arrêt : données vides ou 100% filtrées. Vérifiez --data-dir."
                )
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
                "edge_cut_train": epoch_n_cut,  # v5.1 : hors loss, comptés pas punis
                # P4a : exactitude type de phrase (vide si tête inactive).
                "sentence_accuracy": (
                    sum(p == g for p, g in zip(epoch_sent_type_preds,
                                               epoch_sent_type_gold, strict=False))
                    / max(len(epoch_sent_type_gold), 1)
                    if epoch_sent_type_gold else 0.0),
                # P4 quals : une exactitude par tête (0.0 si inactive/sans gold).
                **{f"qual_{_qn}_accuracy": (
                    sum(p == g for p, g in zip(epoch_qual_preds.get(_qn, []),
                                               epoch_qual_gold.get(_qn, []), strict=False))
                    / max(len(epoch_qual_gold.get(_qn, [])), 1)
                    if epoch_qual_gold.get(_qn) else 0.0)
                   for _qn in QUALIFIERS},
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
                val_loss, _val_n_cut = _run_eval_pass(
                    pipeline, val_loader,
                    val_node_preds, val_node_gold,
                    val_edge_preds, val_edge_gold,
                    val_sent_node_preds, val_sent_node_gold,
                    val_sent_edge_preds, val_sent_edge_gold,
                )
                metrics["edge_cut_val"] = _val_n_cut
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

            # Sauver le meilleur checkpoint val_edge_macro_f1 (indépendant de l'early stopping)
            if val_loader is not None:
                _cur_edge_f1 = metrics.get("val_edge_macro_f1", 0.0)
                if _cur_edge_f1 > best_val_edge_f1:
                    best_val_edge_f1 = _cur_edge_f1
                    best_edge_epoch = epoch
                    save_checkpoint(pipeline, Path(best_edge_checkpoint_path))

            # Early stopping (surveille val_node_macro_f1 pour la patience)
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

    # T5-min : optimiser la température sur le val set si disponible (C5 fix)
    # §2.7 ETUDE : chemin forward direct via reps (pas de passage str via le
    # bridge — _cached_edge_logits restait vide et `except: pass` masquait tout).
    if val_loader is not None:
        try:
            from ..evaluation.calibration import optimize_temperature
            _val_logits, _val_labels, _t5_skipped = collect_val_logits_for_calibration(
                val_loader, pipeline, all_pairs)
            if _t5_skipped > 0:
                click.echo(f"T5-min : {_t5_skipped} phrase(s) ignorée(s) (voir warnings).")
            if not _val_logits:
                click.echo("T5-min : aucun logit collecté — chemin forward incorrect ou val sans arêtes")
            if _val_logits and len(_val_labels) >= 10:
                import numpy as _np
                _all_logits = _np.vstack(_val_logits)[:len(_val_labels)]
                _all_labels = _np.array(_val_labels[:len(_all_logits)], dtype=_np.int64)
                best_t, best_ece = optimize_temperature(_all_logits, _all_labels)
                pipeline.temperature = best_t
                click.echo(f"T5-min : température optimisée = {best_t:.3f} (ECE={best_ece:.4f})")
                save_checkpoint(pipeline, output)
        except Exception as _t5_err:  # noqa: BLE001
            click.echo(f"T5-min : échec ({_t5_err}) — température inchangée")

    click.echo(f"Checkpoint sauvegardé : {output}")
    if val_loader is not None and best_edge_epoch > 0:
        click.echo(
            f"Best val_edge_macro_f1={best_val_edge_f1:.4f} (epoch {best_edge_epoch}) "
            f"→ {best_edge_checkpoint_path}"
        )

    # v5.6 : visibilité par classe — plus aucune classe en échec silencieux.
    # Dernière epoch (pas best) : honnête sur l'état final, supports visibles.
    if epoch_edge_gold:
        click.echo("Rapport edge/train (dernière epoch) :\n"
                   + per_class_report(epoch_edge_preds, epoch_edge_gold,
                                      pipeline.relation_types))
    if val_loader is not None and val_edge_gold:
        click.echo("Rapport edge/val (dernière epoch) :\n"
                   + per_class_report(val_edge_preds, val_edge_gold,
                                      pipeline.relation_types))
    # P4a/b : confusion type de phrase + co-occurrences gold (sentence × intent).
    # Dernière epoch : matrice des relations supposées, pas des preuves.
    if epoch_sent_type_gold:
        click.echo("Confusion sentence/train (dernière epoch) :\n"
                   + per_class_report(epoch_sent_type_preds, epoch_sent_type_gold,
                                      SENTENCE_TYPES))
    if epoch_sent_intent_gold:
        from collections import Counter as _Counter
        _co = _Counter(epoch_sent_intent_gold)
        _rows = ["sentence × intent (gold, dernière epoch) :"]
        for (_s, _i) in sorted(_co):
            _rows.append(f"  {_s:<14s} × {_i:<14s} : {_co[(_s, _i)]}")
        click.echo("\n".join(_rows))

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
