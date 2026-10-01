# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..constants import NODE_TYPE_ALIASES, NODE_TYPES, RELATION_TYPES
from ..layer1.representation import UDRepresentation
from .json_reader import load_all_sentences
from .schema import ClauseRecord, SentenceRecord, TokenRecord


def _node_type_idx(node_type: str, sentence_id: str) -> int:
    # Migration transparente v2→D5
    node_type = NODE_TYPE_ALIASES.get(node_type, node_type)
    if node_type not in NODE_TYPES:
        raise ValueError(
            f"[sentence {sentence_id}] node_type inconnu : {node_type!r}. "
            f"Valeurs autorisées : {NODE_TYPES}"
        )
    return NODE_TYPES.index(node_type)


def _relation_idx(relation: str, sentence_id: str) -> int:
    if relation not in RELATION_TYPES:
        raise ValueError(
            f"[sentence {sentence_id}] relation inconnue : {relation!r}. "
            f"Valeurs autorisées : {RELATION_TYPES}"
        )
    return RELATION_TYPES.index(relation)


@dataclass
class TrainingSample:
    sentence: SentenceRecord
    gold_node_labels: np.ndarray  # (N,) int — indices dans NODE_TYPES
    edge_map: dict  # {(src_clause_idx, tgt_clause_idx): rel_idx} — seule source de vérité pour les arêtes
    hyperedge_map: dict = field(default_factory=dict)  # {(frozenset(sources_str), tgt_idx): rel_idx} N-aires
    edge_conf_map: dict = field(default_factory=dict)  # {(src, tgt): float} — confidence par arête (S-5)
    # Ternaire (§11.5) : {(src_idx, tgt_idx): (role, node_id_str)} depuis EdgeRecord.third.
    # Propagé jusqu'ici (pas encore supervisé — loss BCE quand N_min=20) pour que
    # le moteur voie le tiers dès que les données existent. Vide sur données actuelles.
    third_map: dict = field(default_factory=dict)


class GCNDataLoader:
    """Itère sur les sentences JSON d'un répertoire et produit des TrainingSample."""

    def __init__(self, data_dir: Path, repeat: bool = False,
                  all_pairs: bool = True, shuffle: bool = False, seed: int = 42,
                  silver_weight: float = 1.0):
        self.data_dir = data_dir
        self.repeat = repeat
        self.all_pairs = all_pairs  # True par défaut — toutes les paires supervisées
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed) if shuffle else None
        self.silver_weight = silver_weight  # Amélioration F (1.0 = aucun effet)
        self._records = load_all_sentences(data_dir, silver_weight)
        self._warned_total = False
        # Compteur agrégé pour arêtes asymétriques en direction inverse
        self._total_backward_asymmetric = 0
        self._warned_backward_asymmetric = False
        # Tripwire modifiers (plan v2) : le champ existe au schéma mais aucun
        # consommateur moteur — si des modifiers apparaissent, le signaler au
        # lieu de les ignorer silencieusement.
        self._total_modifiers = 0
        self._warned_modifiers = False

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        while True:
            records = self._records
            if getattr(self, '_shuffle', False):
                idxs = self._rng.permutation(len(records))
                records = [records[i] for i in idxs]
            for rec in records:
                try:
                    yield self._to_sample(rec)
                except ValueError as exc:
                    warnings.warn(f"[{rec.id}] sample ignoré : {exc}", UserWarning, stacklevel=2)
            if (not self.repeat
                    and hasattr(self, '_total_backward_asymmetric')
                    and self._total_backward_asymmetric > 0
                    and hasattr(self, '_warned_backward_asymmetric')
                    and not self._warned_backward_asymmetric):
                warnings.warn(
                    f"Total : {self._total_backward_asymmetric} arête(s) asymétrique(s) "
                    "(cause/enable/prevent) en direction inverse sur l'ensemble du dataset — "
                    "supervision potentiellement incorrecte (inversion cause/effet). "
                    "Annoter dans la direction correcte (src < tgt) pour éliminer ce biais.",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_backward_asymmetric = True
            if (not self.repeat
                    and hasattr(self, '_total_modifiers')
                    and self._total_modifiers > 0
                    and hasattr(self, '_warned_modifiers')
                    and not self._warned_modifiers):
                warnings.warn(
                    f"Total : {self._total_modifiers} modifieur(s) annoté(s) sans "
                    f"consommateur moteur — inventaire requis avant feature/tête.",
                    UserWarning,
                    stacklevel=2,
                )
                self._warned_modifiers = True
            if not self.repeat:
                break

    def _to_sample(self, rec: SentenceRecord) -> TrainingSample:
        # _relation_idx est appelé après les guards gap>1 et backward (src>tgt)
        # pour éviter de rejeter toute la phrase sur une arête non-supervisable
        # dont la relation serait inconnue.
        node_id_to_idx = {c.node_id: i for i, c in enumerate(rec.clauses)}
        # Tripwire modifiers : compteur robuste même si _to_sample est appelé
        # hors __init__ (tests) — pas d'AttributeError.
        self._total_modifiers = getattr(self, '_total_modifiers', 0) + sum(
            1 for c in rec.clauses if getattr(c, 'modifiers', None))
        node_labels = np.array(
            [_node_type_idx(c.node_type, rec.id) for c in rec.clauses],
            dtype=np.int64,
        )
        edge_map: dict[tuple[int, int], int] = {}
        edge_conf_map: dict[tuple[int, int], float] = {}
        hyperedge_map: dict[tuple[frozenset, int], int] = {}
        third_map: dict[tuple[int, int], tuple[str, str]] = {}
        n_backward = 0
        for e in rec.edges:
            src_list = list(getattr(e, "sources", None) or ([e.source] if e.source else []))
            tgt = getattr(e, "target", "")
            if len(src_list) > 1:
                # N-arête : bypass gap>1, route vers hyperedge_map (non-ordonné v2.0, cf ADR).
                src_idxs = [node_id_to_idx.get(s) for s in src_list]
                tgt_idx = node_id_to_idx.get(tgt)
                if any(i is None for i in src_idxs) or tgt_idx is None:
                    warnings.warn(
                        f"[{rec.id}] hyperarête {src_list}→{tgt} : node_id inconnu — ignorée.",
                        UserWarning, stacklevel=2,
                    )
                    continue
                if not src_list:
                    warnings.warn(f"[{rec.id}] hyperarête sources vide — ignorée.",
                                  UserWarning, stacklevel=2)
                    continue
                rel_idx = _relation_idx(e.relation, rec.id)
                hyperedge_map[(frozenset(str(s) for s in src_list), tgt_idx)] = rel_idx
                continue
            src_id = src_list[0] if src_list else ""
            src_idx = node_id_to_idx.get(src_id)
            tgt_idx = node_id_to_idx.get(tgt)
            if src_idx is None or tgt_idx is None:
                warnings.warn(
                    f"[{rec.id}] arête {src_id}→{tgt} : node_id inconnu — arête ignorée.",
                    UserWarning, stacklevel=2,
                )
                continue
            rel_idx = _relation_idx(e.relation, rec.id)
            if src_idx > tgt_idx:
                n_backward += 1
                if e.relation in {"cause", "enable", "prevent"}:
                    warnings.warn(
                        f"[{rec.id}] arête asymétrique ignorée : {e.relation} "
                        f"({src_id}→{tgt}) src={src_idx} > tgt={tgt_idx}. "
                        "Annoter dans la direction src < tgt pour éviter l'inversion cause/effet.",
                        UserWarning, stacklevel=2,
                    )
                    if hasattr(self, '_total_backward_asymmetric'):
                        self._total_backward_asymmetric += 1
                    continue
                key = (tgt_idx, src_idx)
                if key in edge_map:
                    warnings.warn(
                        f"[{rec.id}] conflit arête anti-parallèle {src_id}→{tgt} "
                        f"(clé {key} déjà présente, relation ignorée).",
                        UserWarning, stacklevel=2,
                    )
                else:
                    edge_map[key] = rel_idx
                    if e.confidence is not None:
                        edge_conf_map[key] = float(e.confidence)
                    _third = getattr(e, "third", None) or {}
                    if isinstance(_third, dict) and _third.get("role") in ("condition", "mediator") \
                            and _third.get("node") is not None:
                        third_map[key] = (_third["role"], str(_third["node"]))
            else:
                key = (src_idx, tgt_idx)
                edge_map[key] = rel_idx
                if e.confidence is not None:
                    edge_conf_map[key] = float(e.confidence)
                third = getattr(e, "third", None) or {}
                role = third.get("role") if isinstance(third, dict) else None
                tnode = third.get("node") if isinstance(third, dict) else None
                if role in ("condition", "mediator") and tnode is not None:
                    third_map[key] = (role, str(tnode))
        if n_backward:
            warnings.warn(
                f"[{rec.id}] {n_backward} arête(s) gold en direction inverse (src > tgt). "
                "Les relations asymétriques (cause, enable, prevent) sont ignorées. "
                "Les relations symétriques sont remappées (tgt→src). "
                "Annoter dans la direction src < tgt pour éviter toute perte.",
                UserWarning,
                stacklevel=2,
            )
        return TrainingSample(rec, node_labels, edge_map, hyperedge_map, edge_conf_map,
                              third_map)


def reps_from_sentence(
    rec: SentenceRecord,
) -> tuple[list[UDRepresentation], list[int], list[UDRepresentation | None]]:
    """Une UDRepresentation par ClauseRecord non-vide, construite depuis les tokens JSON.

    Bypass spaCy : garantit l'alignement exact features ↔ gold labels.
    Retourne ([], [], []) si le SentenceRecord n'a pas de tokens annotés.

    Retourne un triplet :
    - reps : UDRepresentation par clause valide
    - valid_indices : indices des clauses converties dans rec.clauses
    - connector_reps : UDRepresentation du connecteur entre reps[k] et reps[k+1],
      ou None si aucun connecteur trouvé (longueur = len(reps) - 1)
    """
    if not rec.tokens or not rec.clauses:
        return [], [], []
    result: list[UDRepresentation] = []
    valid_indices: list[int] = []
    for i, clause in enumerate(rec.clauses):
        rep = _rep_from_clause(clause, rec.tokens)
        if rep is not None:
            result.append(rep)
            valid_indices.append(i)
    connector_reps: list[UDRepresentation | None] = []
    # Marqueurs gold : index (src_clause_idx, tgt_clause_idx) -> marker_token.
    # Le marker annoté prime sur la redécouverte syntaxique (moteur apprend
    # depuis l'annotation ; fallback _connector_between sinon).
    node_id_to_idx = {c.node_id: i for i, c in enumerate(rec.clauses)}
    gold_markers: dict[tuple[int, int], int] = {}
    for e in rec.edges:
        srcs = list(getattr(e, "sources", None) or ([e.source] if e.source else []))
        if len(srcs) != 1 or e.marker_token is None:
            continue
        a = node_id_to_idx.get(srcs[0])
        b = node_id_to_idx.get(getattr(e, "target", ""))
        if a is not None and b is not None:
            try:
                gold_markers[(a, b)] = int(e.marker_token)
            except (TypeError, ValueError):
                warnings.warn(
                    f"[{rec.id}] marker_token invalide {e.marker_token!r} — redécouverte syntaxique.",
                    UserWarning, stacklevel=2,
                )
    for k in range(len(result) - 1):
        a, b = valid_indices[k], valid_indices[k + 1]
        connector_reps.append(
            _connector_between(
                rec.clauses[a], rec.clauses[b], rec.tokens,
                marker_token=gold_markers.get((a, b)),
            )
        )
    return result, valid_indices, connector_reps


def _rep_from_token(tok) -> UDRepresentation:
    """UDRepresentation ponctuelle depuis un token (connecteur gold ou redécouvert)."""
    return UDRepresentation(
        tokens=[{"lemma": tok.lemma, "pos": tok.pos, "dep_rel": tok.dep_rel, "morph": tok.morph,
                 "id": tok.id, "dep_head": tok.dep_head, "form": tok.form}],
        root_lemma=tok.lemma,
        root_pos=tok.pos,
        root_dep_rel=tok.dep_rel,
        root_morph=tok.morph,
        subject_pos=None,
        has_object=False,
        has_advcl=False,
        has_temporal_obl=False,
        token_span=(tok.id, tok.id),
    )


def _connector_between(
    clause_a: ClauseRecord,
    clause_b: ClauseRecord,
    all_tokens: list[TokenRecord],
    marker_token: int | None = None,
) -> UDRepresentation | None:
    """Token connecteur entre deux spans consécutives.

    marker_token (annotation gold) prime : le token désigné est pris tel quel,
    sans filtre POS — l'annotateur sait. Sinon redécouverte syntaxique
    (SCONJ/CCONJ/ADP du gap). None si rien dans les deux cas.
    """
    if marker_token is not None:
        tok = next((t for t in all_tokens if t.id == marker_token), None)
        if tok is not None:
            return _rep_from_token(tok)
        warnings.warn(
            f"marker_token={marker_token} sans token correspondant — redécouverte syntaxique.",
            UserWarning, stacklevel=3,
        )
    end_a = clause_a.token_span[1]
    start_b = clause_b.token_span[0]
    gap_toks = [t for t in all_tokens if end_a < t.id < start_b]
    if not gap_toks:
        return None
    tok = next((t for t in gap_toks if t.pos in {"SCONJ", "CCONJ", "ADP"}), None)
    if tok is None:
        return None
    return _rep_from_token(tok)


def _rep_from_clause(
    clause: ClauseRecord,
    all_tokens: list[TokenRecord],
) -> UDRepresentation | None:
    span_start, span_end = clause.token_span
    if span_start > span_end:
        warnings.warn(
            f"Span inversée dans {clause.node_id} : ({span_start}, {span_end}) — clause ignorée.",
            UserWarning, stacklevel=3,
        )
        return None
    span_toks = [t for t in all_tokens if span_start <= t.id <= span_end]
    if not span_toks:
        return None

    # Priorité : VERB/AUX, sinon NOUN/PROPN, sinon premier token
    root_tok = (
        next((t for t in span_toks if t.pos in {"VERB", "AUX"}), None)
        or next((t for t in span_toks if t.pos in {"NOUN", "PROPN"}), None)
        or span_toks[0]
    )

    subject = next((t for t in span_toks if t.dep_rel in {"nsubj", "nsubj:pass"}), None)

    return UDRepresentation(
        tokens=[
            {"lemma": t.lemma, "pos": t.pos, "dep_rel": t.dep_rel, "morph": t.morph,
             "id": t.id, "dep_head": t.dep_head, "form": t.form}
            for t in span_toks
        ],
        root_lemma=root_tok.lemma,
        root_pos=root_tok.pos,
        root_dep_rel=root_tok.dep_rel,
        root_morph=root_tok.morph,
        subject_pos=subject.pos if subject else None,
        # L3 : retirer "nobj" (relation UD invalide)
        has_object=any(t.dep_rel in {"obj", "iobj"} for t in span_toks),
        has_advcl=any(t.dep_rel == "advcl" for t in span_toks),
        has_temporal_obl=any(t.dep_rel in {"obl", "obl:tmod"} for t in span_toks),
        token_span=clause.token_span,
    )
