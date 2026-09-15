from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import warnings
import numpy as np

from .json_reader import load_all_sentences
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
    edge_map: dict  # {(src_clause_idx, tgt_clause_idx): rel_idx} — seule source de vérité pour les arêtes


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
                try:
                    yield self._to_sample(rec)
                except ValueError as exc:
                    warnings.warn(f"[{rec.id}] sample ignoré : {exc}", UserWarning, stacklevel=2)
            if not self.repeat:
                break

    def _to_sample(self, rec: SentenceRecord) -> TrainingSample:
        node_id_to_idx = {c.node_id: i for i, c in enumerate(rec.clauses)}
        node_labels = np.array(
            [_node_type_idx(c.node_type, rec.id) for c in rec.clauses],
            dtype=np.int64,
        )
        edge_map: dict[tuple[int, int], int] = {}
        n_backward = 0
        for e in rec.edges:
            src_idx = node_id_to_idx.get(e.source)
            tgt_idx = node_id_to_idx.get(e.target)
            if src_idx is None or tgt_idx is None:
                warnings.warn(
                    f"[{rec.id}] arête {e.source}→{e.target} : node_id inconnu — arête ignorée.",
                    UserWarning, stacklevel=2,
                )
                continue
            rel_idx = _relation_idx(e.relation, rec.id)
            gap = abs(tgt_idx - src_idx)
            if gap > 1:
                warnings.warn(
                    f"[{rec.id}] arête longue distance {e.source}→{e.target} "
                    f"(gap={gap}) : aucune supervision d'arête possible (forward prédit "
                    f"uniquement les paires consécutives).",
                    UserWarning, stacklevel=2,
                )
                continue  # arête non-supervisable, ne pas insérer dans edge_map
            edge_map[(src_idx, tgt_idx)] = rel_idx
            if src_idx > tgt_idx:
                n_backward += 1
        if n_backward:
            warnings.warn(
                f"[{rec.id}] {n_backward} arête(s) gold en direction inverse "
                f"(src > tgt) — non supervisées (le forward prédit uniquement la "
                f"direction consécutive croissante).",
                UserWarning, stacklevel=2,
            )
        return TrainingSample(rec, node_labels, edge_map)


def reps_from_sentence(
    rec: SentenceRecord,
) -> tuple[list[UDRepresentation], list[int], list[UDRepresentation | None]]:
    """Une UDRepresentation par ClauseRecord non-vide, construite depuis les tokens YAML.

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
        rep = _rep_from_clause(clause, rec.tokens, rec.lang)
        if rep is not None:
            result.append(rep)
            valid_indices.append(i)
    connector_reps: list[UDRepresentation | None] = [
        _connector_between(
            rec.clauses[valid_indices[k]],
            rec.clauses[valid_indices[k + 1]],
            rec.tokens,
            rec.lang,
        )
        for k in range(len(result) - 1)
    ]
    return result, valid_indices, connector_reps


def _connector_between(
    clause_a: ClauseRecord,
    clause_b: ClauseRecord,
    all_tokens: list[TokenRecord],
    lang: str,
) -> UDRepresentation | None:
    """Retourne une UDRepresentation pour le token connecteur entre deux spans consécutives."""
    end_a = clause_a.token_span[1]
    start_b = clause_b.token_span[0]
    gap_toks = [t for t in all_tokens if end_a < t.id < start_b]
    if not gap_toks:
        return None
    tok = (
        next((t for t in gap_toks if t.gcn_causal_type == "conjonction"), None)
        or next((t for t in gap_toks if t.pos in {"SCONJ", "CCONJ", "ADP"}), None)
    )
    if tok is None:
        return None
    return UDRepresentation(
        tokens=[{"lemma": tok.lemma, "pos": tok.pos, "dep_rel": tok.dep_rel, "morph": tok.morph}],
        root_lemma=tok.lemma,
        root_pos=tok.pos,
        root_dep_rel=tok.dep_rel,
        root_morph=tok.morph,
        subject_pos=None,
        has_object=False,
        has_advcl=False,
        has_temporal_obl=False,
        token_span=(tok.id, tok.id),
        lang=lang,
    )


def _rep_from_clause(
    clause: ClauseRecord,
    all_tokens: list[TokenRecord],
    lang: str,
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
