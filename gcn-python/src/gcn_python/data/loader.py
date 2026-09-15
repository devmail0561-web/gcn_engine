from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .yaml_reader import load_all_sentences
from .schema import SentenceRecord, TokenRecord, ClauseRecord
from ..constants import NODE_TYPES, RELATION_TYPES
from ..layer1.representation import UDRepresentation


def _node_type_idx(node_type: str, sentence_id: str) -> int:
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
    gold_edge_labels: np.ndarray  # (E,) int — indices dans RELATION_TYPES
    edge_map: dict  # {(src_clause_idx, tgt_clause_idx): rel_idx} — alignement sémantique


class GCNDataLoader:
    """Itère sur les sentences YAML d'un répertoire et produit des TrainingSample."""

    def __init__(self, data_dir: Path, lang: str = "fr", repeat: bool = False):
        self.data_dir = data_dir
        self.lang = lang
        self.repeat = repeat
        self._records = load_all_sentences(data_dir, lang)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        while True:
            for rec in self._records:
                yield self._to_sample(rec)
            if not self.repeat:
                break

    def _to_sample(self, rec: SentenceRecord) -> TrainingSample:
        node_id_to_idx = {c.node_id: i for i, c in enumerate(rec.clauses)}
        node_labels = np.array(
            [_node_type_idx(c.node_type, rec.id) for c in rec.clauses],
            dtype=np.int64,
        )
        edge_labels = np.array(
            [_relation_idx(e.relation, rec.id) for e in rec.edges],
            dtype=np.int64,
        )
        edge_map: dict[tuple[int, int], int] = {}
        for e in rec.edges:
            src_idx = node_id_to_idx.get(e.source)
            tgt_idx = node_id_to_idx.get(e.target)
            if src_idx is not None and tgt_idx is not None:
                edge_map[(src_idx, tgt_idx)] = _relation_idx(e.relation, rec.id)
        return TrainingSample(rec, node_labels, edge_labels, edge_map)


def reps_from_sentence(
    rec: SentenceRecord,
) -> tuple[list[UDRepresentation], list[int]]:
    """Une UDRepresentation par ClauseRecord non-vide, construite depuis les tokens YAML.

    Bypass spaCy : garantit l'alignement exact features ↔ gold labels.
    Retourne ([], []) si le SentenceRecord n'a pas de tokens annotés (format paper_examples).

    Le second élément est la liste des indices de clause (dans rec.clauses) effectivement
    convertis — nécessaire pour aligner les gold labels avec les logits du forward.
    """
    if not rec.tokens or not rec.clauses:
        return [], []
    result: list[UDRepresentation] = []
    valid_indices: list[int] = []
    for i, clause in enumerate(rec.clauses):
        rep = _rep_from_clause(clause, rec.tokens, rec.lang)
        if rep is not None:
            result.append(rep)
            valid_indices.append(i)
    return result, valid_indices


def _rep_from_clause(
    clause: ClauseRecord,
    all_tokens: list[TokenRecord],
    lang: str,
) -> UDRepresentation | None:
    span_start, span_end = clause.token_span
    span_toks = [t for t in all_tokens if span_start <= t.id <= span_end]
    if not span_toks:
        return None

    # Priorité : VERB annoté gcn_causal_type="verbe", sinon premier VERB/AUX, sinon premier token
    root_tok = (
        next((t for t in span_toks
              if t.gcn_causal_type == "verbe" and t.pos in {"VERB", "AUX"}), None)
        or next((t for t in span_toks if t.pos in {"VERB", "AUX"}), span_toks[0])
    )

    subject = next((t for t in span_toks if t.dep_rel in {"nsubj", "nsubj:pass"}), None)

    return UDRepresentation(
        tokens=[
            {"lemma": t.lemma, "pos": t.pos, "dep_rel": t.dep_rel, "morph": t.morph}
            for t in span_toks
        ],
        root_lemma=root_tok.lemma,
        root_pos=root_tok.pos,
        root_dep_rel=root_tok.dep_rel,
        root_morph=root_tok.morph,
        subject_pos=subject.pos if subject else None,
        has_object=any(t.dep_rel in {"obj", "iobj", "nobj"} for t in span_toks),
        has_advcl=any(t.dep_rel == "advcl" for t in span_toks),
        has_temporal_obl=any(t.dep_rel in {"obl", "obl:tmod"} for t in span_toks),
        token_span=clause.token_span,
        lang=lang,
    )
