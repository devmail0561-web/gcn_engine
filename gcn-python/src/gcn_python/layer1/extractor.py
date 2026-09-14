from __future__ import annotations
import spacy
from spacy.language import Language
from spacy.tokens import Doc

from ..constants import SPACY_MODELS
from .representation import UDRepresentation


_nlp_cache: dict[str, Language] = {}


def get_nlp(lang: str) -> Language:
    if lang not in _nlp_cache:
        model = SPACY_MODELS.get(lang)
        if model is None:
            raise ValueError(
                f"No spaCy model configured for lang='{lang}'. "
                f"Available: {list(SPACY_MODELS.keys())}"
            )
        try:
            _nlp_cache[lang] = spacy.load(model)
        except OSError:
            raise OSError(
                f"spaCy model '{model}' not installed. "
                f"Run: python -m spacy download {model}"
            )
    return _nlp_cache[lang]


def extract(text: str, lang: str) -> list[UDRepresentation]:
    """text → spaCy Doc → List[UDRepresentation] via UD dependency tree."""
    if not text.strip():
        return []
    nlp = get_nlp(lang)
    doc = nlp(text)
    return _segment_to_representations(doc, lang)


def _segment_to_representations(doc: Doc, lang: str) -> list[UDRepresentation]:
    """Segmente le Doc en clauses via les relations UD."""
    clause_dep_rels = {"root", "advcl", "relcl", "csubj", "ccomp", "xcomp", "parataxis"}
    roots = [tok for tok in doc if tok.dep_ in clause_dep_rels and tok.pos_ == "VERB"]

    if not roots:
        # Fallback: find any root token
        fallback = next((t for t in doc if t.dep_ == "ROOT"), None)
        if fallback:
            roots = [fallback]
        else:
            return []

    representations = []
    for root in roots:
        subtree_tokens = sorted(root.subtree, key=lambda t: t.i)
        if not subtree_tokens:
            continue

        start_idx = subtree_tokens[0].i + 1   # 1-based
        end_idx = subtree_tokens[-1].i + 1

        tokens_data = [
            {
                "lemma": t.lemma_.lower(),
                "pos": t.pos_,
                "dep_rel": t.dep_,
                "morph": dict(t.morph),
            }
            for t in subtree_tokens
        ]

        subject = next(
            (c for c in root.children if c.dep_ in {"nsubj", "nsubj:pass"}), None
        )
        has_object = any(c.dep_ in {"obj", "iobj", "nobj"} for c in root.children)
        has_advcl = any(c.dep_ == "advcl" for c in root.children)
        has_temporal_obl = any(
            c.dep_ in {"obl", "obl:tmod"} and (
                "Tense" in dict(c.morph) or any(gc.dep_ == "case" for gc in c.children)
            )
            for c in root.children
        )

        representations.append(UDRepresentation(
            tokens=tokens_data,
            root_lemma=root.lemma_.lower(),
            root_pos=root.pos_,
            root_dep_rel=root.dep_,
            root_morph=dict(root.morph),
            subject_pos=subject.pos_ if subject else None,
            has_object=has_object,
            has_advcl=has_advcl,
            has_temporal_obl=has_temporal_obl,
            token_span=(start_idx, end_idx),
            lang=lang,
        ))

    return representations
