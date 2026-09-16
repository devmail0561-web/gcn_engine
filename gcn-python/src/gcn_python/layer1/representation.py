from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class UDRepresentation:
    """
    Abstraction UD-agnostique d'une clause construite depuis des tokens YAML annotés.
    Aucune heuristique linguistique — uniquement des features UD universelles.
    """
    tokens: list[dict]          # [{lemma, pos(UPOS), dep_rel(UD), morph: dict}]
    root_lemma: str
    root_pos: str               # UPOS
    root_dep_rel: str           # UD dep_rel
    root_morph: dict[str, str]  # {Tense: "Past", Aspect: "Imp", Mood: "Ind", …}
    subject_pos: str | None     # UPOS du nsubj, ou None
    has_object: bool            # dep obj/iobj existe
    has_advcl: bool             # dep advcl existe
    has_temporal_obl: bool      # dep obl avec morph temporel
    token_span: tuple[int, int]

    @property
    def tense(self) -> str:
        return self.root_morph.get("Tense", "_absent")

    @property
    def aspect(self) -> str:
        return self.root_morph.get("Aspect", "_absent")

    @property
    def mood(self) -> str:
        return self.root_morph.get("Mood", "_absent")

    @property
    def is_negative(self) -> bool:
        return self.root_morph.get("Polarity", "") == "Neg"
