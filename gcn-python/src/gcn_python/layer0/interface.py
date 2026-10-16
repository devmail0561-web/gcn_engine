# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..layer1.representation import UDRepresentation


@runtime_checkable
class TextParser(Protocol):
    """
    Protocol Layer 0 : texte brut → liste de UDRepresentation.

    Toute implémentation DOIT remplir root_lemma, root_pos, root_dep_rel.
    Les implémentations haute qualité remplissent aussi root_morph
    (Tense/Aspect/Mood/Polarity) depuis un vrai parser UD.
    Les implémentations bridge laissent ces champs à _absent / False.

    Contrainte (amendée 2026-10-08, D-SPACY) : import spaCy autorisé UNIQUEMENT en lazy
    et optionnel (moteur B, T1 : pin + timeout + fallback + test sans-spacy vert) ;
    jamais d'import au top-level ni de dépendance dure dans `gcn_python`.
    spaCy peut aussi implémenter ce Protocol en code utilisateur.
    """

    def parse(
        self,
        text: str,
    ) -> tuple[list[UDRepresentation], list[UDRepresentation | None]]:
        """
        text → (clause_reps, connector_reps)

        clause_reps     : une UDRepresentation par clause détectée
        connector_reps  : len(clause_reps) - 1 éléments, un par paire consécutive
                          (UDRepresentation du connecteur ou None si absent)
        """
        ...
