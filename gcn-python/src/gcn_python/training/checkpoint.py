# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from ..pipeline.cgnp import CGNPipeline
from ..layer1.features import FeatureVocabulary


def _check_path_safe(path: Path, *, allow_symlink: bool = False) -> None:
    """Refuse les chemins suspects : symlinks dans l'arbre, hardlinks, FIFO.

    Appelé avant lecture (load_checkpoint) ET avant écriture (save_checkpoint)
    pour éliminer les vecteurs de substitution silencieuse.
    """
    import os
    import stat as _stat
    # Vérifier chaque composant du chemin (pas seulement la feuille)
    check = path if not path.exists() else path
    p = path.resolve().parent  # on vérifie les parents d'abord
    # Vérifier la feuille ET chaque parent jusqu'à la racine
    parts_to_check = [path] + list(path.parents)
    for part in parts_to_check:
        if not part.exists():
            continue
        if part.is_symlink():
            if not allow_symlink:
                raise RuntimeError(
                    f"Chemin suspect (symlink) : {part}. "
                    "Passez allow_symlink=True si vous avez vérifié la cible."
                )
    # Vérifier la feuille si elle existe : hardlink et FIFO
    if path.exists() and not path.is_symlink():
        st = path.stat()
        if _stat.S_ISFIFO(st.st_mode):
            raise RuntimeError(
                f"Refus de lire/écrire {path.name} : c'est un FIFO (named pipe)."
            )
        if st.st_nlink > 1 and not allow_symlink:
            raise RuntimeError(
                f"Refus de lire/écrire {path.name} : {st.st_nlink} liens durs détectés "
                "(vecteur de substitution silencieuse). Passez allow_symlink=True pour forcer."
            )


def save_checkpoint(pipeline: CGNPipeline, path: Path) -> None:
    """Sérialise tous les poids du pipeline dans un fichier .npz."""
    path = Path(path)
    # Refuser les chemins suspects avant d'écrire
    _check_path_safe(path, allow_symlink=False)

    arrays: dict[str, np.ndarray] = {}

    # Métadonnées d'architecture — lues par GCNEngine.from_pretrained()
    # pour reconstruire le pipeline sans hardcoder les indices de shapes.
    we = getattr(pipeline, 'word_embedding', None)
    d_emb = we.d_emb if we is not None else 0
    _sob = bool(getattr(pipeline, 'subject_object_emb', False))
    d_eff = pipeline.vocabulary.d_clause_effective(d_emb, _sob)
    graph0 = pipeline._graph_layers[0]
    n_rel = getattr(graph0, 'n_relations', len(pipeline.relation_types))
    d_hidden = getattr(graph0, 'd_out', d_eff)
    try:
        vocab_size = len(pipeline.vocabulary)
    except TypeError:
        vocab_size = getattr(pipeline.vocabulary, 'vocab_size', 0)
    arch = {
        "d_eff":        d_eff,
        "d_emb":        d_emb,
        "d_hidden":     int(d_hidden),
        "vocab_size":   int(vocab_size),
        "n_relations":  n_rel,
        "bidirectional": pipeline.bidirectional,
        "bidi_flag":    bool(pipeline.bidirectional),
        "all_pairs": pipeline.all_pairs,
        "n_rgcn_layers": pipeline.n_rgcn_layers,
        "graph_class":  type(graph0).__name__,
        "edge_threshold": float(getattr(pipeline, 'edge_threshold', 0.0)),
        "drop_morph":   bool(getattr(pipeline, 'drop_morph', False)),
        "temperature":  float(getattr(pipeline, 'temperature', 1.0)),
        "bfs_depth":    (None if getattr(pipeline, 'bfs_depth', None) is None
                         else int(pipeline.bfs_depth)),
        # §1 (amelioration_v3) — clés absentes = défauts (vieux checkpoints lisibles)
        "clause_pooling": str(getattr(pipeline, 'clause_pooling', 'root')),
        "subject_object_emb": _sob,
        "freeze_embeddings": bool(getattr(we, 'frozen', False)) if we is not None else False,
        "n_gat_heads": int(getattr(graph0, 'n_heads', 1)),
        "gat_residual": bool(getattr(pipeline, 'gat_residual', False)),
        "gat_layernorm": bool(getattr(graph0, 'norm', None) is not None
                              and type(graph0).__name__ == "RGCNLayerGAT"),
        "gat_output_activation": str(getattr(graph0, 'output_activation', 'sigmoid')),
        "rgcn_output_activation": str(getattr(graph0, 'output_activation', 'sigmoid')),
        "silver_weight": float(getattr(pipeline, 'silver_weight', 1.0)),
        "verbalize_mode": str(getattr(pipeline, 'verbalize_mode', 'legacy')),
        "mlp_hidden": int(getattr(pipeline.encoder, 'mlp_hidden', 128)),
        # Phase C : MHA globale — n_gat_heads_mha distinct de n_gat_heads (GAT).
        "global_attention": bool(getattr(pipeline, 'global_attention',
                                        type(pipeline.encoder).__name__ == "TransformerMLPEncoder")),
        "n_gat_heads_mha": int(getattr(pipeline, 'mha_heads',
                                      getattr(pipeline.encoder, 'n_heads', 4)
                                      if type(pipeline.encoder).__name__ == "TransformerMLPEncoder"
                                      else 4)),
        # Phase B : anti-over-smoothing (RGCNLayerPT uniquement).
        "pairnorm": bool(getattr(pipeline, 'pairnorm',
                                getattr(graph0, 'pairnorm', False))),
        "drop_edge": float(getattr(pipeline, 'drop_edge',
                                  getattr(graph0, 'drop_edge', 0.0))),
        # Phase D : CompGCN (RGCNLayerPT uniquement).
        "use_compgcn": bool(getattr(pipeline, 'use_compgcn',
                                   getattr(graph0, 'use_compgcn', False))),
        "d_rel_emb": int(getattr(pipeline, 'd_rel_emb',
                                getattr(graph0, 'd_rel_emb', 32))),
        "two_pass_val": bool(getattr(pipeline, 'two_pass_val', True)),
        "rgcn_layernorm": bool(getattr(graph0, 'use_layernorm', False)),
    }
    arrays["_arch_json"] = np.array([json.dumps(arch)], dtype=object)

    encoder_params = pipeline.encoder.parameters()
    for i, p in enumerate(encoder_params):
        arrays[f"encoder_{i}"] = p

    # S5 : sauvegarder toutes les couches R-GCN (_graph_layers)
    for layer_i, layer in enumerate(getattr(pipeline, '_graph_layers', [pipeline.graph])):
        prefix = "graph" if layer_i == 0 else f"graph_extra_{layer_i}"
        for j, p in enumerate(layer.parameters()):
            arrays[f"{prefix}_{j}"] = p

    vocab_json = pipeline.vocabulary.to_json()
    arrays["_vocab_json"] = np.array([vocab_json], dtype=object)

    if pipeline.decoder is not None and hasattr(pipeline.decoder, 'parameters'):
        decoder_params = pipeline.decoder.parameters()
        for i, p in enumerate(decoder_params):
            arrays[f"decoder_{i}"] = p
        if hasattr(pipeline.decoder, 'to_json'):
            arrays["_decoder_meta_json"] = np.array([pipeline.decoder.to_json()], dtype=object)

    # LinkPredHead : clés optionnelles (warn si absentes au chargement, pas ValueError)
    link_pred = getattr(pipeline, 'link_predictor', None)
    if link_pred is not None and hasattr(link_pred, 'parameters'):
        for i, p in enumerate(link_pred.parameters()):
            arrays[f"link_pred_{i}"] = np.asarray(p)
        if hasattr(link_pred, 'to_json'):
            arrays["_link_pred_meta_json"] = np.array([link_pred.to_json()], dtype=object)

    # S9 : sauvegarder word_embedding si présent
    if getattr(pipeline, 'word_embedding', None) is not None:
        we = pipeline.word_embedding
        arrays["_word_emb_vocab_json"] = np.array([we.to_json()], dtype=object)
        arrays["word_emb_E"] = we._E
        # C : plage pré-entraînée (gel partiel) — None si jamais chargé
        if we._pretrained_start is not None and we._pretrained_end is not None:
            arrays["word_emb_pretrained_start"] = np.array([we._pretrained_start])
            arrays["word_emb_pretrained_end"] = np.array([we._pretrained_end])

    # Phase C : sauvegarder les poids MHA (fixes mais différents selon seed)
    # Nécessaire pour la reproductibilité exacte à la reprise d'un checkpoint.
    if type(pipeline.encoder).__name__ == "TransformerMLPEncoder":
        try:
            import torch as _pt
            for _k, _v in pipeline.encoder._mha.state_dict().items():
                safe_k = _k.replace('.', '__')
                arrays[f"_mha_{safe_k}"] = _v.detach().cpu().numpy()
        except Exception:
            pass  # PyTorch absent ou erreur — checkpoint reste valide sans MHA

    # G2 : sauvegarder l'assembleur lexical si présent
    _asm = getattr(pipeline, 'assembler', None)
    if _asm is not None and hasattr(_asm, 'parameters'):
        for i, p in enumerate(_asm.parameters()):
            arrays[f"assembler_{i}"] = np.asarray(p)
        if hasattr(_asm, 'to_json'):
            arrays["_assembler_meta_json"] = np.array([_asm.to_json()], dtype=object)

    # Sauvegarde atomique : tmp + replace POSIX
    path = Path(path)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _gat_ar_2d_compatible(layer, j: int, arr) -> bool:
    """Cas D (§1) : a_r 2-D d'un vieux checkpoint mono-tête, convertible par load_state.

    True si la couche est GAT (n_heads==1), clé = a_r (index 2) et shape (R, 2*D).
    """
    if j != 2 or getattr(layer, 'n_heads', None) != 1:
        return False
    try:
        _shape = tuple(arr.shape)
    except (AttributeError, TypeError):
        return False
    n_rel = getattr(layer, 'n_relations', None)
    d_out = getattr(layer, 'd_out', None)
    return len(_shape) == 2 and _shape == (n_rel, 2 * d_out)


def load_checkpoint(
    pipeline: CGNPipeline, path: Path,
    *, trusted: bool = False, allow_symlink: bool = False,
    _skip_path_check: bool = False,
) -> None:
    """Restaure les poids depuis un fichier .npz produit par save_checkpoint.

    Atomique : toutes les shapes sont validées avant toute mutation du pipeline.
    Un ValueError laisse le pipeline intact (vocabulary, encoder et graph inchangés).

    Sécurité : le .npz contient des tableaux `object` (JSON d'architecture/vocab)
    qui exigent `allow_pickle=True`, soit une exécution de pickle au chargement.
    `trusted=False` par défaut refuse le chargement ; passez `trusted=True`
    uniquement pour un checkpoint local de confiance (produit par votre
    `gcn-train`). Les CLI (`gcn-train --encoder-checkpoint`, `gcn-eval`,
    `gcn-discuss`, `gcn-index`) le passent explicitement — l'invocation vaut opt-in.

    Symlinks : refusés par défaut (R4 — vecteur de substitution silencieuse).
    Passez `allow_symlink=True` si vous êtes sûr de la cible.
    """
    if not trusted:
        raise RuntimeError(
            f"Refus de charger {Path(path).name} : checkpoint .npz non fiable par défaut "
            "(allow_pickle requis → exécution de pickle). Relancez avec trusted=True "
            "pour un fichier local de confiance."
        )
    if not _skip_path_check:
        try:
            _check_path_safe(Path(path), allow_symlink=allow_symlink)
        except RuntimeError:
            raise
        if allow_symlink and Path(path).is_symlink():
            import warnings as _w
            _w.warn(
                f"Checkpoint {Path(path).name} est un symlink — allow_symlink=True "
                "passé explicitement.",
                UserWarning,
                stacklevel=2,
            )
    data = np.load(path, allow_pickle=True)

    # Clés attendues : inconnues -> warn+ignore (forward-compat v2.5 dans code v2.0)
    _VALID_PREFIXES = ("encoder_", "graph_", "graph_extra_", "decoder_",
                       "link_pred_", "hyperedge_", "assembler_", "_mha_")
    _VALID_EXACT = {"_vocab_json", "_decoder_meta_json", "_word_emb_vocab_json", "word_emb_E",
                    "_arch_json", "_link_pred_meta_json", "_assembler_meta_json",
                    "word_emb_pretrained_start", "word_emb_pretrained_end"}
    unexpected = set(data.files) - _VALID_EXACT
    unexpected = {k for k in unexpected if not any(k.startswith(p) for p in _VALID_PREFIXES)}
    if unexpected:
        import warnings as _w2
        _w2.warn(
            f"Checkpoint {Path(path).name} contient des clés inconnues ignorées : "
            f"{sorted(unexpected)}.",
            UserWarning,
            stacklevel=2,
        )

    # Validation sémantique du triplet (d_eff, d_hidden, |V|, n_relations, bidi)
    _arch: dict = {}
    if "_arch_json" in data:
        import json as _json
        try:
            _arch = _json.loads(str(data["_arch_json"][0]))
        except Exception as _exc:
            import warnings as _w_arch
            _w_arch.warn(
                f"Checkpoint {Path(path).name} : _arch_json illisible ({_exc}) — "
                "validation architecture désactivée.",
                UserWarning, stacklevel=2,
            )
            _arch = {}
    else:
        import warnings as _w_noarch
        _w_noarch.warn(
            f"Checkpoint {Path(path).name} sans _arch_json — validation architecture "
            "désactivée (re-entraîner avec gcn-train >= 2.1.0).",
            UserWarning, stacklevel=2,
        )
    if isinstance(_arch, dict):
        _we = getattr(pipeline, 'word_embedding', None)
        _d_emb = _we.d_emb if _we is not None else 0
        _d_eff = pipeline.vocabulary.d_clause_effective(
            _d_emb, bool(getattr(pipeline, 'subject_object_emb', False)))
        _g0 = pipeline._graph_layers[0] if getattr(pipeline, '_graph_layers', None) else pipeline.graph
        _d_hid = getattr(_g0, 'd_out', _d_eff)
        _n_rel = getattr(_g0, 'n_relations', len(pipeline.relation_types))
        for _k, _exp, _found in (
            ("d_eff", _d_eff, _arch.get("d_eff")),
            ("d_hidden", int(_d_hid), _arch.get("d_hidden")),
            ("n_relations", _n_rel, _arch.get("n_relations")),
            ("bidirectional", bool(pipeline.bidirectional),
             _arch.get("bidirectional", _arch.get("bidi_flag"))),
            ("all_pairs", bool(pipeline.all_pairs), _arch.get("all_pairs")),
            ("n_rgcn_layers", pipeline.n_rgcn_layers, _arch.get("n_rgcn_layers")),
            # Phases C/D : mismatch silencieux = métriques trompeuses (MHA aléatoire
            # différente, shapes CompGCN incompatibles). Vieux checkpoints (clés
            # absentes) → skip silencieux (backward compat).
            ("global_attention", bool(getattr(pipeline, 'global_attention', False)),
             _arch.get("global_attention")),
            ("use_compgcn", bool(getattr(pipeline, 'use_compgcn', False)),
             _arch.get("use_compgcn")),
        ):
            if _found is not None and _found != _exp:
                raise ValueError(
                    f"Incompatibilité de dimension pour {_k} : "
                    f"checkpoint={_found!r} ≠ pipeline={_exp!r}."
                )
        # Vieux checkpoint sans n_rgcn_layers : clé absente → skip silencieux,
        # mais si le pipeline a > 1 couche, la couche extra resterait aléatoire.
        if isinstance(_arch, dict) and _arch.get("n_rgcn_layers") is None and pipeline.n_rgcn_layers > 1:
            import warnings as _w_nl
            _w_nl.warn(
                f"Checkpoint {Path(path).name} : n_rgcn_layers absent en arch — "
                f"pipeline {pipeline.n_rgcn_layers} couches, validation couches désactivée "
                "(vieux checkpoint pré-2.1.0).",
                UserWarning, stacklevel=2,
            )

    # Double-absent policy : decoder
    import logging as _log3
    _ckpt_log = _log3.getLogger(__name__)
    _has_ckpt_decoder = any(k.startswith("decoder_") for k in data.files)
    if pipeline.decoder is None and not _has_ckpt_decoder:
        _ckpt_log.debug("Checkpoint sans decoder_* et pipeline sans decoder — usage normal.")
    elif pipeline.decoder is not None and not _has_ckpt_decoder:
        import warnings as _w3
        _w3.warn("Pipeline avec decoder mais checkpoint sans decoder_* "
                 "— poids non restaurés (init aléatoire).", UserWarning, stacklevel=2)

    # Double-absent policy : link_predictor (optionnel, jamais bloquant)
    _has_ckpt_lp = any(k.startswith("link_pred_") for k in data.files)
    _has_pipe_lp = getattr(pipeline, 'link_predictor', None) is not None
    if _has_pipe_lp and not _has_ckpt_lp:
        import warnings as _w3lp
        _w3lp.warn("Pipeline avec link_predictor mais checkpoint sans link_pred_* "
                   "— tête non restaurée (init aléatoire).", UserWarning, stacklevel=2)
    elif not _has_pipe_lp and not _has_ckpt_lp:
        _ckpt_log.debug("Checkpoint sans link_pred_* et pipeline sans link_predictor — usage normal.")

    new_vocab = None
    if "_vocab_json" in data:
        new_vocab = FeatureVocabulary.from_json(str(data["_vocab_json"][0]))

    # Validation de toutes les formes avant toute mutation
    encoder_params = pipeline.encoder.parameters()
    for i, p in enumerate(encoder_params):
        key = f"encoder_{i}"
        if key in data and data[key].shape != p.shape:
            raise ValueError(
                f"Incompatibilité de dimension pour encoder_{i} : "
                f"checkpoint={data[key].shape} ≠ pipeline={p.shape}. "
                f"Reconstruisez le pipeline avec la même taxonomie que le checkpoint."
            )

    # S5 : valider toutes les couches R-GCN
    for layer_i, layer in enumerate(getattr(pipeline, '_graph_layers', [pipeline.graph])):
        prefix = "graph" if layer_i == 0 else f"graph_extra_{layer_i}"
        for j, p in enumerate(layer.parameters()):
            key = f"{prefix}_{j}"
            if key in data and data[key].shape != p.shape:
                if _gat_ar_2d_compatible(layer, j, data[key]):
                    continue  # Cas D : load_state() convertit (R,2D) → (1,R,2D)
                raise ValueError(
                    f"Incompatibilité de dimension pour {key} : "
                    f"checkpoint={data[key].shape} ≠ pipeline={p.shape}."
                )

    # Toutes les formes validées — mutation sûre
    if new_vocab is not None:
        pipeline.vocabulary = new_vocab

    # D1 : restaurer les hyperparamètres d'inférence depuis l'arch — cohérent avec vocab.
    # from_pretrained et run_eval le font au constructeur ; load_checkpoint doit le faire
    # ici pour que --encoder-checkpoint produise le même comportement d'inférence
    # que le checkpoint d'origine (seuil, morph, température).
    if isinstance(_arch, dict):
        _et = _arch.get("edge_threshold")
        _dm = _arch.get("drop_morph")
        _tp = _arch.get("temperature")
        if _et is not None:
            pipeline.edge_threshold = float(_et)
        if _dm is not None:
            pipeline.drop_morph = bool(_dm)
        if _tp is not None:
            pipeline.temperature = float(_tp)
        _bd = _arch.get("bfs_depth")
        if _bd is not None:
            pipeline.bfs_depth = int(_bd)
        # §1 (amelioration_v3) : restaurer les hyperparamètres d'inférence —
        # clés absentes (vieux checkpoints) → défauts, jamais d'erreur.
        _cp = _arch.get("clause_pooling")
        if _cp is not None:
            pipeline.clause_pooling = str(_cp)
        _so = _arch.get("subject_object_emb")
        if _so is not None:
            pipeline.subject_object_emb = bool(_so)
        _gr = _arch.get("gat_residual")
        if _gr is not None:
            pipeline.gat_residual = bool(_gr)
        _vm = _arch.get("verbalize_mode")
        if _vm is not None:
            pipeline.verbalize_mode = str(_vm)
        _sw = _arch.get("silver_weight")
        if _sw is not None:
            pipeline.silver_weight = float(_sw)
        _tpv = _arch.get("two_pass_val")
        if _tpv is not None:
            pipeline.two_pass_val = bool(_tpv)
        _mh = _arch.get("mlp_hidden")
        if _mh is not None and hasattr(pipeline.encoder, 'mlp_hidden'):
            _mh_int = int(_mh)
            if _mh_int != pipeline.encoder.mlp_hidden:
                # C1.3 : ne pas écraser — les matrices du pipeline ont été
                # construites avec la valeur du pipeline, pas avec l'arch.
                import warnings as _w_mh
                _w_mh.warn(
                    f"mlp_hidden archivé={_mh_int} != pipeline={pipeline.encoder.mlp_hidden} "
                    "— attribut non mis à jour (matrices incompatibles). "
                    f"Reconstruire avec mlp_hidden={_mh_int}.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                pipeline.encoder.mlp_hidden = _mh_int

    for i, p in enumerate(encoder_params):
        key = f"encoder_{i}"
        if key in data:
            p[:] = data[key]

    # S5 : restaurer toutes les couches R-GCN
    for layer_i, layer in enumerate(getattr(pipeline, '_graph_layers', [pipeline.graph])):
        prefix = "graph" if layer_i == 0 else f"graph_extra_{layer_i}"
        layer_params = layer.parameters()
        # H5 correction : utiliser load_state() pour PyTorch RGCNLayerPT
        if hasattr(layer, 'load_state'):
            ckpt_arrays = [data[f"{prefix}_{j}"] for j in range(len(layer_params))
                           if f"{prefix}_{j}" in data]
            if ckpt_arrays:
                layer.load_state(ckpt_arrays)
        else:
            for j, p in enumerate(layer_params):
                key = f"{prefix}_{j}"
                if key in data:
                    p[:] = data[key]

    # S9 : restaurer word_embedding si présent dans le checkpoint
    if "_word_emb_vocab_json" in data:
        from ..layer1.embedding import WordEmbedding
        d_emb = int(data["word_emb_E"].shape[1]) if "word_emb_E" in data else 50
        we = WordEmbedding.from_json(str(data["_word_emb_vocab_json"][0]), d_emb=d_emb)
        if "word_emb_E" in data:
            we._E = data["word_emb_E"].astype(np.float32)
        # C : plage pré-entraînée + flag frozen depuis l'arch (§1)
        if "word_emb_pretrained_start" in data and "word_emb_pretrained_end" in data:
            we._pretrained_start = int(data["word_emb_pretrained_start"][0])
            we._pretrained_end = int(data["word_emb_pretrained_end"][0])
        elif isinstance(_arch, dict) and _arch.get("freeze_embeddings"):
            import warnings as _wfro
            _wfro.warn(
                f"Checkpoint {Path(path).name} : freeze_embeddings demandé mais plage "
                "pré-entraînée absente (vieux checkpoint) — gel total appliqué.",
                UserWarning, stacklevel=2,
            )
        if isinstance(_arch, dict) and _arch.get("freeze_embeddings") is not None:
            we.frozen = bool(_arch.get("freeze_embeddings"))
        pipeline.word_embedding = we

    if "_decoder_meta_json" in data:
        from ..verbalizer.trainable import TrainableDecoder
        decoder = TrainableDecoder.from_json(str(data["_decoder_meta_json"][0]))
        decoder_params = decoder.parameters()
        # B7 : pré-validation complète de TOUTES les shapes avant toute mutation
        # (identique au pattern encodeur/RGCN lignes 183-202).
        for i, p in enumerate(decoder_params):
            key = f"decoder_{i}"
            if key in data and data[key].shape != p.shape:
                raise ValueError(
                    f"Incompatibilité de dimension pour decoder_{i} : "
                    f"checkpoint={data[key].shape} ≠ decoder={p.shape}."
                )
        # Toutes les shapes validées — mutation sûre
        for i, p in enumerate(decoder_params):
            key = f"decoder_{i}"
            if key in data:
                p[:] = data[key]
        pipeline.decoder = decoder

    # Restaurer LexicalConnectorAssembler (G2) si présent dans le checkpoint
    if "_assembler_meta_json" in data:
        from ..verbalizer.trainable import LexicalConnectorAssembler
        asm = LexicalConnectorAssembler.from_json(str(data["_assembler_meta_json"][0]))
        for i, p in enumerate(asm.parameters()):
            key = f"assembler_{i}"
            if key in data:
                if p.ndim == 0:
                    p[()] = data[key].item() if hasattr(data[key], "item") else data[key]
                elif data[key].shape != p.shape:
                    raise ValueError(
                        f"Incompatibilité de dimension pour {key} : "
                        f"checkpoint={data[key].shape} ≠ assembler={p.shape}."
                    )
                else:
                    p[:] = data[key]
        pipeline.assembler = asm

    # Phase C : restaurer les poids MHA si présents
    _mha_keys = [k for k in data.files if k.startswith("_mha_")]
    if _mha_keys and type(pipeline.encoder).__name__ == "TransformerMLPEncoder":
        try:
            import torch as _pt
            sd = {}
            for k in _mha_keys:
                orig_k = k[5:].replace('__', '.')
                sd[orig_k] = _pt.as_tensor(np.asarray(data[k]))
            pipeline.encoder._mha.load_state_dict(sd, strict=False)
        except Exception as _mha_exc:
            import warnings as _w_mha
            _w_mha.warn(
                f"Checkpoint : restauration MHA échouée ({_mha_exc}) — "
                "poids MHA ré-initialisés (run non reproductible exactement).",
                UserWarning, stacklevel=2,
            )

    # Restaurer LinkPredHead si présent dans le checkpoint
    if "_link_pred_meta_json" in data:
        from ..layer3.link_pred import LinkPredHead
        head = LinkPredHead.from_json(str(data["_link_pred_meta_json"][0]))
        # B7 parity : pré-validation de TOUTES les shapes avant toute mutation
        for i, p in enumerate(head.parameters()):
            key = f"link_pred_{i}"
            if key in data and data[key].shape != p.shape:
                raise ValueError(
                    f"Incompatibilité de dimension pour link_pred_{i} : "
                    f"checkpoint={data[key].shape} ≠ head={p.shape}."
                )
        # Toutes les shapes validées — mutation sûre
        for i, p in enumerate(head.parameters()):
            key = f"link_pred_{i}"
            if key in data:
                if p.ndim == 0:
                    p[()] = data[key].item() if hasattr(data[key], "item") else data[key]
                else:
                    p[:] = data[key]
        pipeline.link_predictor = head
