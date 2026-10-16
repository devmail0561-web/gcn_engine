# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Flags --use-tok des CLIs batch (discuss/index) — résolution unique.

Règle : auto-force si le checkpoint chargé est tok-only (pipeline None) ;
gate explicite si --use-tok sans moteur tok fondé (E_tok vide refusé —
même doctrine que le gate capacitaire de ``engine.analyze``).
Défaut inchangé (legacy) dans tous les autres cas.
"""

from __future__ import annotations

import click


def resolve_use_tok(engine, use_tok: bool, outil: str) -> bool:
    """Retourne le flag effectif (auto-force tok-only, gate sinon)."""
    if getattr(engine, "_pipeline", None) is None:
        if not use_tok:
            click.echo(f"{outil} : checkpoint tok détecté — chemin tok activé.")
        return True
    if use_tok:
        tok = getattr(engine, "_tok_engine", None)
        if tok is None or getattr(tok, "vocab_size", 0) == 0:
            raise click.ClickException(
                f"{outil} : --use-tok exige un moteur tok fondé (checkpoint s1+ "
                "ou tok_engine avec vocabulaire) — E_tok vide refusé."
            )
    return use_tok
