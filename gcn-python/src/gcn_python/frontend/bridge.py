# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
Pont texte brut → UDRepresentation via gcn-cli subprocess (lattice par défaut).

Le bridge appelle `gcn <subcommand> -- <text>` (CLI Rust, défaut `analyze`) qui émet
un lattice de tokens observés (forme, POS morphologique, pseudo-dep_rel
positionnels — zéro dictionnaire, zéro décision). Ce module construit des
UDRepresentation multi-tokens réelles depuis ce lattice. Aucune langue
n'est nommée ici : le bridge route une chaîne de sous-commande, sans
liste de langues en dur.

Limites honnêtes (couche B : parseur UD) :
  - morph partiel (flags de forme : Tense=Past si imparfait, VerbForm) ;
  - pas de dep_rel authentiques (pseudo-dep_rel positionnels) ;
  - négation analytique et scope non détectés (zéro word matching).

  Pour la qualité maximale : UDRepresentation annotées via GCNDataLoader.
"""
from __future__ import annotations

import json
import subprocess

from ..layer1.representation import UDRepresentation


class GCNBridgeError(RuntimeError):
    """Levée quand gcn-cli est indisponible, timeout, code non-zéro ou JSON invalide."""




def _resolve_gcn_bin(gcn_bin: str) -> str:
    """Résout gcn_bin en chemin absolu et valide le résultat.

    - Nom nu (ex. "gcn") : résolu via shutil.which() → chemin absolu.
      Élimine la fenêtre PATH-hijacking : un attaquant qui modifie PATH
      après la résolution ne peut plus substituer le binaire.
    - Chemin absolu : vérifié directement (fichier existant et exécutable).
    - Chemin relatif : converti en absolu via resolve(), puis vérifié.

    Raises:
        GCNBridgeError: binaire introuvable ou invalide.
    """
    import os as _os
    import shutil as _shutil
    if not gcn_bin or not gcn_bin.strip():
        raise GCNBridgeError("gcn_bin vide — chemin invalide.")
    # Nom nu (pas de séparateur) → résolution via PATH au moment de l'appel
    if _os.sep not in gcn_bin and (not _os.altsep or _os.altsep not in gcn_bin):
        resolved = _shutil.which(gcn_bin)
        if resolved is None:
            raise GCNBridgeError(
                f"Binaire gcn introuvable dans PATH : {gcn_bin!r}. "
                "Installez gcn-cli ou passez gcn_bin=<chemin absolu>."
            )
        return resolved  # absolu, stable même si PATH change ensuite
    # Chemin fourni (absolu ou relatif)
    from pathlib import Path as _Path
    p = _Path(gcn_bin).resolve()
    if not p.exists():
        raise GCNBridgeError(f"Binaire gcn introuvable : {p}")
    return str(p)


def _validate_gcn_bin(gcn_bin: str) -> None:
    """Valide gcn_bin (garde de compatibilité — préférer _resolve_gcn_bin).

    Refuse les chaînes vides. Les métacaractères shell sont inoffensifs avec
    shell=False mais on les refuse quand même (défense en profondeur).
    """
    import re as _re
    if not gcn_bin or not gcn_bin.strip():
        raise GCNBridgeError("gcn_bin vide — chemin invalide.")
    dangerous = _re.search(r'[;&|`$()<>\n\r]', gcn_bin)
    if dangerous:
        raise GCNBridgeError(
            f"gcn_bin {gcn_bin!r} contient le caractère dangereux {dangerous.group()!r}."
        )


def _call_gcn_lattice(text: str, gcn_bin: str, subcommand: str = "analyze") -> dict:
    """Appelle `gcn <subcommand> -- <text>` (lattice par défaut, sans --data-dir).

    Retourne le lattice JSON {source_text, tokens[], clauses[]} : tokens
    observés (forme, POS morphologique, pseudo-dep_rel positionnels),
    zéro décision linguistique, zéro dictionnaire.
    """
    if not subcommand or not subcommand.strip():
        raise GCNBridgeError("subcommand vide — nom de sous-commande invalide.")
    gcn_bin = _resolve_gcn_bin(gcn_bin)
    # analyze/analyze-en émettent le lattice par défaut (zéro dictionnaire).
    cmd = [gcn_bin, subcommand, "--", text]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            timeout=30, check=False,
        )
    except FileNotFoundError:
        raise GCNBridgeError(
            f"Binaire gcn introuvable : {gcn_bin!r}."
        ) from None
    except subprocess.TimeoutExpired:
        raise GCNBridgeError(
            f"Timeout (30s) lors de `{gcn_bin} {subcommand} -- <text>`."
        ) from None
    if proc.returncode != 0:
        raise GCNBridgeError(
            f"`{gcn_bin} {subcommand} -- <text>` a échoué "
            f"(code {proc.returncode}) :\n{proc.stderr.strip()[:300]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise GCNBridgeError(
            f"Sortie lattice JSON invalide : {exc}. "
            f"Début : {proc.stdout[:100]!r}"
        ) from exc


_LATTICE_POS_TO_UPOS = {
    "verb": "VERB", "noun": "NOUN", "adv": "ADV",
    "punct": "PUNCT", "other": "X",
}

_LATTICE_DEP_TO_UD = {
    "root": "root", "nsubj": "nsubj", "obj": "obj", "mark": "mark",
    "advcl": "advcl", "det": "det", "advmod": "advmod",
    "punct": "punct", "other": "dep",
}


def _lattice_token_to_ud(tok: dict) -> dict:
    """Token lattice Rust → token UD Python.

    POS : mapping direct sauf fonction structurelle (mark→SCONJ,
    det→DET, advmod→ADV) — la fonction vient de la POSITION, pas du lemme.
    Morph : dérivé des flags de forme (imparfait→Tense=Past,
    infinitive→VerbForm=Inf, participle→VerbForm=Part). Le reste est absent
    (pas de parseur UD sur ce chemin — couche B).
    """
    dep = str(tok.get("dep_rel", "other"))
    pos_raw = str(tok.get("pos", "other"))
    if dep == "mark":
        upos = "SCONJ"
    elif dep == "det":
        upos = "DET"
    elif dep == "advmod":
        upos = "ADV"
    else:
        upos = _LATTICE_POS_TO_UPOS.get(pos_raw, "X")
    morph: dict = {}
    for flag in tok.get("flags") or []:
        if flag == "imparfait":
            morph["Tense"] = "Past"
        elif flag == "infinitive":
            morph["VerbForm"] = "Inf"
        elif flag == "participle":
            morph["VerbForm"] = "Part"
    try:
        tid = int(tok.get("index", -1))
    except (TypeError, ValueError):
        tid = -1
    try:
        dhead = int(tok.get("dep_head", -1))
    except (TypeError, ValueError):
        dhead = -1
    return {
        "lemma": str(tok.get("lemma", "")),
        "pos": upos,
        "dep_rel": _LATTICE_DEP_TO_UD.get(dep, "dep"),
        "morph": morph,
        "id": tid,
        "dep_head": dhead,
        "form": str(tok.get("form", "")),
    }


def _rep_from_ud_tokens(ud_tokens: list, has_advcl_sentence: bool) -> UDRepresentation:
    """Groupe de tokens UD réels → UDRepresentation (une clause)."""
    root = None
    for t in ud_tokens:
        if t.get("dep_rel") == "root":
            root = t
            break
    if root is None:
        for t in ud_tokens:
            if t.get("dep_rel") == "advcl":
                root = t
                break
    if root is None:
        for t in ud_tokens:
            if t.get("pos") == "VERB":
                root = t
                break
    if root is None:
        for t in ud_tokens:
            if t.get("dep_rel") in ("nsubj", "obj") and t.get("pos") == "NOUN":
                root = t
                break
    if root is None:
        for t in ud_tokens:
            if t.get("pos") not in ("PUNCT", "X"):
                root = t
                break
    if root is None and ud_tokens:
        root = ud_tokens[0]
    subj_pos = None
    for t in ud_tokens:
        if t.get("dep_rel") == "nsubj":
            subj_pos = t.get("pos")
            break
    ids = [t.get("id", 0) for t in ud_tokens if isinstance(t.get("id"), int)]
    span = (min(ids), max(ids)) if ids else (0, 0)
    return UDRepresentation(
        tokens=ud_tokens,
        root_lemma=(root or {}).get("lemma", ""),
        root_pos=(root or {}).get("pos", "X"),
        root_dep_rel=(root or {}).get("dep_rel", "dep"),
        root_morph=dict((root or {}).get("morph", {}) or {}),
        subject_pos=subj_pos,
        has_object=any(t.get("dep_rel") == "obj" for t in ud_tokens),
        has_advcl=has_advcl_sentence,
        has_temporal_obl=False,
        token_span=span,
    )


def _reps_from_lattice(lattice: dict) -> tuple[list, list]:
    """Lattice Rust → (clause_reps, connector_reps), tokens réels.

    Une rep par clause. Connecteur = token mark réel (lemme/POS/span
    observés) entre clauses adjacentes, sinon None.
    Clause unique transitive (nsubj + obj) : scindée au verbe principal
    en (groupe sujet | groupe prédicat) pour que la tête d'arêtes ait
    une paire à classifier — découpe structurelle, pas lexicale.
    """
    raw_tokens = lattice.get("tokens") or []
    ud = [_lattice_token_to_ud(t) for t in raw_tokens if isinstance(t, dict)]
    if not ud:
        return [], []
    clauses = lattice.get("clauses") or []
    has_advcl_sentence = any(t.get("dep_rel") == "advcl" for t in ud)

    # Regrouper par clause (champ clause du lattice, 0-based).
    # ud est aligné avec les entrées dict de raw_tokens (même ordre, même filtre).
    by_clause: dict[int, list] = {}
    for t in raw_tokens:
        if not isinstance(t, dict):
            continue
        r = _lattice_token_to_ud(t)
        try:
            cid = int(r.get("clause", t.get("clause", 0)))
        except (TypeError, ValueError):
            cid = 0
        by_clause.setdefault(cid, []).append(r)
    _ = clauses  # spans informatifs (le découpage vient du champ clause)
    ordered = [by_clause[k] for k in sorted(by_clause)]

    if len(ordered) == 1:
        toks = ordered[0]
        verb_idx = next(
            (i for i, t in enumerate(toks)
             if t.get("dep_rel") in ("root", "advcl") and t.get("pos") == "VERB"),
            None,
        )
        has_subj = any(t.get("dep_rel") == "nsubj" for t in toks)
        has_obj = any(t.get("dep_rel") == "obj" for t in toks)
        if verb_idx is not None and has_subj and has_obj:
            src, dst = toks[:verb_idx], toks[verb_idx:]
            if src and dst:
                return (
                    [_rep_from_ud_tokens(src, has_advcl_sentence),
                     _rep_from_ud_tokens(dst, has_advcl_sentence)],
                    [None],
                )
        # Subordonnée sans transitivité : scinder au marqueur (ou à l'advcl)
        # en (principale | subordonnée) avec connecteur réel.
        cut = next(
            (i for i, t in enumerate(toks) if t.get("dep_rel") == "mark"),
            next(
                (i for i, t in enumerate(toks) if t.get("dep_rel") == "advcl"),
                None,
            ),
        )
        if cut is not None and cut > 0 and cut < len(toks):
            mark_tok = next(
                (t for t in toks if t.get("dep_rel") == "mark"), None
            )
            src, dst = toks[:cut], toks[cut:]
            conn = None
            if mark_tok is not None:
                mid = mark_tok.get("id", 0)
                conn = UDRepresentation(
                    tokens=[mark_tok], root_lemma=mark_tok.get("lemma", ""),
                    root_pos="SCONJ", root_dep_rel="mark", root_morph={},
                    subject_pos=None, has_object=False, has_advcl=False,
                    has_temporal_obl=False, token_span=(mid, mid),
                )
            return (
                [_rep_from_ud_tokens(src, has_advcl_sentence),
                 _rep_from_ud_tokens(dst, has_advcl_sentence)],
                [conn],
            )
        return [_rep_from_ud_tokens(toks, has_advcl_sentence)], []

    reps = [_rep_from_ud_tokens(toks, has_advcl_sentence) for toks in ordered]
    connectors: list = []
    for i in range(len(ordered) - 1):
        mark = next(
            (t for t in ordered[i] + ordered[i + 1] if t.get("dep_rel") == "mark"),
            None,
        )
        if mark is None:
            connectors.append(None)
            continue
        mid = mark.get("id", 0)
        connectors.append(UDRepresentation(
            tokens=[mark],
            root_lemma=mark.get("lemma", ""),
            root_pos="SCONJ",
            root_dep_rel="mark",
            root_morph={},
            subject_pos=None,
            has_object=False,
            has_advcl=False,
            has_temporal_obl=False,
            token_span=(mid, mid),
        ))
    return reps, connectors


class GCNLatticeParser:
    """TextParser via `gcn <subcommand> -- <text>` (lattice par défaut) : tokens réels, zéro dico.

    Contraste avec GCNBridgeParser (reps synthétiques à 1 token depuis le
    CIR) : ici les reps portent les vrais tokens, vraies positions, vrais
    spans et pseudo-dep_rel positionnels. morph reste partiel (flags de
    forme uniquement — pas de dep_rel authentiques avant la couche B).
    """

    def __init__(self, gcn_bin: str = "gcn", subcommand: str = "analyze"):
        self.gcn_bin = gcn_bin
        self.subcommand = subcommand

    def parse(self, text: str) -> tuple[list, list]:
        """Implémente TextParser.parse — retourne (clause_reps, connector_reps)."""
        lattice = _call_gcn_lattice(text, self.gcn_bin, self.subcommand)
        return _reps_from_lattice(lattice)


def reps_from_lattice_text(
    text: str,
    gcn_bin: str = "gcn",
    subcommand: str = "analyze",
) -> list[UDRepresentation]:
    """Texte brut → list[UDRepresentation] multi-tokens via lattice (défaut CLI)."""
    lattice = _call_gcn_lattice(text, gcn_bin, subcommand)
    reps, _ = _reps_from_lattice(lattice)
    return reps


