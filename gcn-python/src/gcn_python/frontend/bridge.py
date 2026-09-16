"""
P1 — Pont texte brut → UDRepresentation via gcn-cli subprocess.

Ce module permet au moteur ML Python de traiter du texte brut sans dépendance
NLP externe (spaCy interdit, pas de PyO3). Le bridge appelle `gcn analyze`
(CLI Rust) et construit des UDRepresentation heuristiques depuis le CIR produit.

QUALITÉ DE L'INFÉRENCE :
  Les UDRepresentation produits sont APPROXIMATIFS — les champs UD (root_pos,
  root_dep_rel, root_morph) sont dérivés du node_type CausalIR, pas d'une
  analyse syntaxique réelle.

  | Feature ML      | Source                                  | Qualité    |
  |-----------------|------------------------------------------|------------|
  | root_pos (UPOS) | NODE_TYPE_TO_POS[node_type]              | ~85%       |
  | root_dep_rel    | NODE_TYPE_TO_DEP[node_type]              | ~80%       |
  | root_morph      | toujours {} → _absent pour Tense/Mood    | dégradé    |
  | is_negative     | toujours False (morph={})                | 0%         |
  | has_object      | attributes.patient is not None           | ~90%       |
  | has_advcl       | toujours False (CIR ne porte pas l'info) | conservatif|
  | has_temporal_obl| temporal_ref not in (None, "unresolved") | ~85%       |
  | subject_pos     | attributes.agent présent → NOUN          | ~70%       |

  Pour la qualité maximale : fournir des UDRepresentation annotées manuellement
  ou via les datasets gcn-nl (GCNDataLoader).
"""
from __future__ import annotations

import json
import re
import subprocess
import warnings
from pathlib import Path

from ..layer1.representation import UDRepresentation


class GCNBridgeError(RuntimeError):
    """Levée quand gcn-cli est indisponible, timeout, code non-zéro ou JSON invalide."""


# Mappings heuristiques node_type CIR → UPOS / dep_rel UD
# Publics pour tests directs.

NODE_TYPE_TO_POS: dict[str, str] = {
    "action":          "VERB",
    "transition":      "VERB",
    "processus":       "NOUN",
    "etat":            "NOUN",
    "etat_systemique": "NOUN",
    "entite":          "NOUN",
    "condition":       "SCONJ",
}
_DEFAULT_POS = "NOUN"

NODE_TYPE_TO_DEP: dict[str, str] = {
    "action":          "root",
    "transition":      "root",
    "processus":       "root",
    "etat":            "nsubj",
    "etat_systemique": "nsubj",
    "entite":          "nsubj",
    "condition":       "advcl",
}
_DEFAULT_DEP = "root"

_LABEL_RE = re.compile(r'^([^(?\s]+)')


def _extract_lemma(label: str) -> str:
    """
    Extrait le lemme depuis un label CIR.

    "décroissance(ventes)"  → "décroissance"
    "cause_cachée(?)"       → "cause_cachée"
    "hausse"                → "hausse"
    ""  / whitespace        → "_unknown"
    """
    stripped = label.strip() if label else ""
    if not stripped:
        return "_unknown"
    m = _LABEL_RE.match(stripped)
    return m.group(1).strip() if m else stripped


def _extract_span(node: dict) -> tuple[int, int]:
    """
    Extrait (start, end) depuis un nœud CIR.

    Supporte :
      - source_span.token_span.{start, end}  ← format Rust serde
      - token_span: [start, end]             ← format flat legacy
    Fallback : (0, 0).
    """
    src = node.get("source_span")
    if isinstance(src, dict):
        ts = src.get("token_span", {})
        if isinstance(ts, dict):
            s = ts.get("start")
            e = ts.get("end")
            return (0 if s is None else int(s), 0 if e is None else int(e))
    flat = node.get("token_span")
    if isinstance(flat, (list, tuple)) and len(flat) >= 2:
        return (int(flat[0]), int(flat[1]))
    return (0, 0)


def _parse_edges(edges: list) -> list[tuple[int, int, dict]]:
    """
    Normalise les edges CIR vers (src_id: int, dst_id: int, edge_obj: dict).

    Supporte :
      - format tuple [src, dst, obj]  ← sortie ir_emitter.py et gcn analyze Rust
      - format dict {"source": ..., "target": ..., ...}
    Les entrées invalides sont silencieusement ignorées.
    """
    result: list[tuple[int, int, dict]] = []
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) == 3:
            src_id, dst_id, edge_obj = e
            if isinstance(edge_obj, dict):
                try:
                    result.append((int(src_id), int(dst_id), edge_obj))
                except (TypeError, ValueError):
                    pass
        elif isinstance(e, dict):
            src = e.get("source")
            dst = e.get("target")
            if src is not None and dst is not None:
                try:
                    result.append((int(src), int(dst), e))
                except (TypeError, ValueError):
                    pass
    return result


def _rep_from_cir_node(node: dict, lang: str) -> UDRepresentation:
    """
    Construit une UDRepresentation heuristique depuis un nœud CIR.

    root_pos et root_dep_rel sont dérivés de node_type via les mappings
    NODE_TYPE_TO_POS / NODE_TYPE_TO_DEP. root_morph est toujours {}.
    """
    node_type = (node.get("node_type") or "action").lower()
    label = node.get("label") or ""
    root_lemma = _extract_lemma(label)
    root_pos = NODE_TYPE_TO_POS.get(node_type, _DEFAULT_POS)
    root_dep_rel = NODE_TYPE_TO_DEP.get(node_type, _DEFAULT_DEP)

    if node_type not in NODE_TYPE_TO_POS:
        warnings.warn(
            f"gcn bridge : node_type inconnu {node_type!r} — fallback NOUN/root.",
            UserWarning,
            stacklevel=3,
        )

    attrs = node.get("attributes") or {}
    subject_pos = "NOUN" if attrs.get("agent") else None
    has_object = attrs.get("patient") is not None
    temporal_ref = node.get("temporal_ref")
    has_temporal_obl = temporal_ref not in (None, "unresolved")

    return UDRepresentation(
        tokens=[{"lemma": root_lemma, "pos": root_pos, "dep_rel": root_dep_rel, "morph": {}}],
        root_lemma=root_lemma,
        root_pos=root_pos,
        root_dep_rel=root_dep_rel,
        root_morph={},  # → Tense/Aspect/Mood = _absent, is_negative = False
        subject_pos=subject_pos,
        has_object=has_object,
        has_advcl=False,  # CIR ne porte pas l'info sur les sous-clauses advcl internes
        has_temporal_obl=has_temporal_obl,
        token_span=_extract_span(node),
        lang=lang,
    )


def _build_connector_rep(marker_token_id: int, lang: str) -> UDRepresentation:
    """UDRepresentation synthétique pour un token connecteur entre deux clauses."""
    return UDRepresentation(
        tokens=[{"lemma": "_connector", "pos": "SCONJ", "dep_rel": "mark", "morph": {}}],
        root_lemma="_connector",
        root_pos="SCONJ",
        root_dep_rel="mark",
        root_morph={},
        subject_pos=None,
        has_object=False,
        has_advcl=False,
        has_temporal_obl=False,
        token_span=(marker_token_id, marker_token_id),
        lang=lang,
    )


def _cir_to_reps_and_connectors(
    cir: dict,
    lang: str,
) -> tuple[list[UDRepresentation], list[UDRepresentation | None]]:
    """
    CIR dict → (clause_reps, connector_reps).

    connector_reps a len(clause_reps) - 1 éléments. Chaque élément est une
    UDRepresentation synthétique SCONJ si marker_token présent dans l'arête
    entre nodes[i] et nodes[i+1], sinon None.
    """
    nodes = cir.get("nodes", [])
    if not nodes:
        return [], []

    parsed_edges = _parse_edges(cir.get("edges", []))

    # Construire index (src_node_id, dst_node_id) → marker_token
    # Ignorer marker_token <= 0 (token fictif ou absent)
    pair_to_marker: dict[tuple[int, int], int] = {}
    for src_id, dst_id, edge_obj in parsed_edges:
        marker = edge_obj.get("marker_token")
        if marker is not None:
            try:
                m = int(marker)
                if m > 0:
                    pair_to_marker[(src_id, dst_id)] = m
            except (TypeError, ValueError):
                pass

    clause_reps = [_rep_from_cir_node(n, lang) for n in nodes]

    connector_reps: list[UDRepresentation | None] = []
    for i in range(len(nodes) - 1):
        src_node_id = nodes[i].get("id", i)
        dst_node_id = nodes[i + 1].get("id", i + 1)
        marker = pair_to_marker.get((src_node_id, dst_node_id))
        connector_reps.append(_build_connector_rep(marker, lang) if marker else None)

    return clause_reps, connector_reps


def _call_gcn_analyze(
    text: str,
    gcn_bin: str,
    taxonomy_dir: Path | None,
) -> dict:
    """
    Appelle `gcn analyze <text>` et retourne le CIR parsé.

    Raises:
        GCNBridgeError: binaire absent, timeout, code non-zéro, JSON invalide.
    """
    cmd = [gcn_bin, "analyze"]
    if taxonomy_dir is not None:
        cmd += ["--data-dir", str(taxonomy_dir)]
    cmd.append(text)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
    except FileNotFoundError:
        raise GCNBridgeError(
            f"Binaire gcn introuvable : {gcn_bin!r}. "
            "Vérifiez que gcn-cli est installé dans PATH "
            "ou spécifiez gcn_bin=<chemin absolu>."
        ) from None
    except subprocess.TimeoutExpired:
        raise GCNBridgeError(
            f"Timeout (30s) lors de `{gcn_bin} analyze`. "
            f"Texte (80 premiers chars) : {text[:80]!r}"
        ) from None

    if proc.returncode != 0:
        raise GCNBridgeError(
            f"`{gcn_bin} analyze` a échoué (code {proc.returncode}) :\n"
            f"{proc.stderr.strip()[:300]}"
        )

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise GCNBridgeError(
            f"Sortie JSON invalide de `{gcn_bin} analyze` : {exc}. "
            f"Début de la sortie : {proc.stdout[:100]!r}"
        ) from exc


class GCNBridgeParser:
    """
    Implémentation de layer0.interface.TextParser via le subprocess gcn-cli.

    Qualité approximative — root_morph toujours {}, donc Tense/Aspect/Mood
    et Polarity toujours _absent/False (14 dimensions de features à zéro).
    Voir module docstring pour la table de qualité complète.

    Implémente TextParser (layer0/interface.py) par duck typing (pas d'import
    direct du Protocol pour éviter les dépendances circulaires).
    """

    def __init__(self, gcn_bin: str = "gcn", taxonomy_dir=None):
        self.gcn_bin = gcn_bin
        self.taxonomy_dir = taxonomy_dir

    def parse(
        self,
        text: str,
        lang: str = "fr",
    ) -> tuple[list, list]:
        """Implémente TextParser.parse — retourne (clause_reps, connector_reps)."""
        cir = _call_gcn_analyze(text, self.gcn_bin, self.taxonomy_dir)
        return _cir_to_reps_and_connectors(cir, lang)


def reps_from_text(
    text: str,
    lang: str = "fr",
    gcn_bin: str = "gcn",
    taxonomy_dir: Path | None = None,
) -> list[UDRepresentation]:
    """
    Texte brut → list[UDRepresentation] via `gcn analyze` (subprocess).

    QUALITÉ APPROXIMATIVE : voir module docstring pour les limitations.

    Args:
        text: texte brut à analyser.
        lang: code langue ISO 639-1 ("fr", "en").
        gcn_bin: chemin vers le binaire gcn-cli (défaut : "gcn" dans PATH).
        taxonomy_dir: répertoire des taxonomies causales (optionnel).

    Returns:
        Liste de UDRepresentation, une par nœud CIR produit par gcn analyze.

    Raises:
        GCNBridgeError: binaire absent, timeout, code non-zéro ou JSON invalide.
    """
    warnings.warn(
        "reps_from_text() produit des UDRepresentation APPROXIMATIFS depuis le CIR "
        "(root_pos/dep_rel heuristiques, root_morph={}, is_negative=False). "
        "Pour la qualité maximale, utiliser des données annotées via GCNDataLoader.",
        UserWarning,
        stacklevel=2,
    )
    cir = _call_gcn_analyze(text, gcn_bin, taxonomy_dir)
    reps, _ = _cir_to_reps_and_connectors(cir, lang)
    return reps
