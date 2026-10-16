# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
from pathlib import Path

import click

from ..constants import NODE_ORIGIN_VALUES, NODE_TYPES, RELATION_TYPES, SCOPE_VALUES

# Mappings node_type → UPOS / dep_rel pour la tokenisation synthétique.
# Identiques à frontend/bridge.py : NODE_TYPE_TO_POS / NODE_TYPE_TO_DEP.
_SYNTH_POS: dict[str, str] = {
    "action": "VERB", "transition": "VERB", "processus": "NOUN",
    "etat": "NOUN", "etat_systemique": "NOUN", "entite": "NOUN", "condition": "SCONJ",
}
_SYNTH_DEP: dict[str, str] = {
    "action": "root", "transition": "root", "processus": "root",
    "etat": "nsubj", "etat_systemique": "nsubj", "entite": "nsubj", "condition": "advcl",
}


def _synthetic_tokens(nodes: list) -> list:
    """Génère un token synthétique par nœud CIR pour rendre le document entraînable.

    B1 correctif : _cir_to_doc produisait tokens=[] → reps_from_sentence retournait
    ([],[],[]) → toutes les phrases bootstrappées étaient ignorées à l'entraînement.

    QUALITÉ APPROXIMATIVE : root_morph={}, Tense/Aspect/Mood=_absent, is_negative=False.
    Les features morphologiques (14 dims) seront nulles — analogue au bridge heuristique.
    Pour la qualité maximale, annoter manuellement les tokens UD.
    """
    tokens = []
    for i, n in enumerate(nodes):
        label = (n.get("label") or "").strip()
        # Accepte "node_type" (CIR Rust/Python) et "type" (format doc gcn-nl)
        ntype = str(n.get("node_type") or n.get("type") or "action").lower()
        lemma = label.split()[0] if label else ntype
        tokens.append({
            "id": i + 1,
            "form": lemma,
            "lemma": lemma,
            "pos": _SYNTH_POS.get(ntype, "NOUN"),
            "dep_rel": _SYNTH_DEP.get(ntype, "root"),
            "dep_head": 0,
            "morph": {},
        })
    return tokens


@click.command("gcn-bootstrap")
@click.option("--input", "input_file", required=True, type=click.Path(path_type=Path),
              help="Fichier texte (.txt) — une phrase par ligne")
@click.option("--out-dir", required=True, type=click.Path(path_type=Path),
              help="Répertoire de sortie pour les fichiers JSON générés")
@click.option("--gcn-bin", default="gcn", show_default=True,
              help="Chemin vers le binaire gcn-cli Rust")
def bootstrap_cmd(
    input_file: Path, out_dir: Path,
    gcn_bin: str,
) -> None:
    """Génère des données d'entraînement JSON depuis des phrases brutes via gcn-cli Rust.

    `gcn analyze` émet désormais un lattice de tokens (zéro dictionnaire),
    pas un CausalIR : le bootstrap symbolique texte→CIR est retiré.
    Annotez via le pipeline ML (GCNEngine) puis réinjectez les CIR.
    """
    raise click.ClickException(
        "gcn-bootstrap texte→CIR est retiré : `gcn analyze` émet un lattice "
        "(tokens observés, zéro décision), pas un CausalIR. "
        "Produisez les CIR via GCNEngine (modèle entraîné) ou les frontends "
        "code/graph/table, puis annotez les phrases manuellement."
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    # Corps symbolique texte→CIR supprimé : `gcn analyze` émet un lattice,
    # pas un CausalIR (plus de frontend symbolique FR/EN). `_cir_to_doc` et
    # `_extract_token_span` restent pour les sources CIR (code/graph/table).


def _extract_token_span(node: dict) -> list[int]:
    """Extrait [start, end] depuis un nœud CausalIR.

    Supporte le format Python (source_span.token_span.{start,end})
    et le format Rust (token_span flat [start, end]).
    """
    src = node.get("source_span")
    if isinstance(src, dict):
        ts = src.get("token_span", {})
        if isinstance(ts, dict):
            # Issue #3 CRITICAL : fallback pour null (ts.get retourne None si clé présente)
            start = ts.get("start")
            end = ts.get("end")
            start = 0 if start is None else int(start)
            end = 0 if end is None else int(end)
            return [start, end]
    flat = node.get("token_span", [0, 0])
    if not flat:
        return [0, 0]
    try:
        result = [int(v) for v in flat]
    except (TypeError, ValueError):
        return [0, 0]
    if len(result) < 2:
        return [result[0], result[0]]
    return result[:2]


def _normalize_edge(e) -> dict | list[dict] | None:
    """
    Normalise une arête CIR vers le format doc gcn-nl.

    Supporte :
      - format tuple/list : [src_id, dst_id, edge_obj]  ← sortie gcn analyze (Rust)
      - format dict       : {"source": ..., "target": ..., "relation": ...}
    Retourne None si le format est invalide ou incomplet.
    Pour JOINT_CAUSE/JointPrevent avec sources=[A, B] : retourne list[dict] (2 arêtes).
    """
    import hashlib

    from ..data.edge_norm import normalize_node_id

    if isinstance(e, (list, tuple)) and len(e) == 3:
        src_id, dst_id, edge_obj = e
        if not isinstance(edge_obj, dict):
            return None
        try:
            source, target = normalize_node_id(src_id), normalize_node_id(dst_id)
        except ValueError:
            return None
        sources_list = [source]
    elif isinstance(e, dict):
        edge_obj = e
        try:
            src_raw = e.get("sources", e.get("source", ""))
            target = normalize_node_id(e.get("target", e.get("dst", "")))
            if isinstance(src_raw, (list, tuple)):
                sources_list = []
                for s in src_raw:
                    with contextlib.suppress(ValueError):
                        sources_list.append(normalize_node_id(s))
                if not sources_list:
                    return None
            else:
                sources_list = [normalize_node_id(src_raw)]
        except ValueError:
            return None
    else:
        return None

    relation = edge_obj.get("relation_type", edge_obj.get("relation", RELATION_TYPES[0]))
    extra = {k: edge_obj[k] for k in
             ("provenance", "temporal_gap", "in_cycle", "derivation", "modifiers", "marker_token")
             if k in edge_obj}

    def _build(src: str, jgid: str | None = None) -> dict:
        r = {
            "source": src, "target": target,
            "relation": relation,
            "confidence": float(edge_obj["confidence"]) if "confidence" in edge_obj else None,
            "explicit": bool(edge_obj.get("explicit", True)),
            "negated": bool(edge_obj["negated"]) if "negated" in edge_obj else None,
            **extra,
        }
        if jgid is not None:
            r["joint_group_id"] = jgid
        return r

    if len(sources_list) == 1:
        return _build(sources_list[0])

    if len(sources_list) == 2:
        if relation not in ("joint_cause", "joint_prevent"):
            import warnings
            warnings.warn(
                f"bootstrap : arête multi-source avec relation '{relation}' — "
                "edge_norm forcera joint_cause à l'ingestion moteur.",
                UserWarning,
                stacklevel=2,
            )
        key = f"{target}|{'|'.join(sorted(sources_list))}"
        jgid = hashlib.sha256(key.encode()).hexdigest()[:16]
        return [_build(sources_list[0], jgid), _build(sources_list[1], jgid)]

    return _build(sources_list[0])


def _canonical_node_id(raw, pos: int) -> str:
    """Id CIR brut -> id canonique 'nNNN' (cohérent avec les arêtes normalisées).

    Sans ça, _cir_to_doc émettait des ids int (0, 1) côté nœuds et str ("0", "1")
    côté arêtes — le loader ignorait alors 100 % des arêtes (node_id inconnu).
    """
    from ..data.edge_norm import normalize_node_id
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return f"n{pos + 1:03d}"
    try:
        return normalize_node_id(raw)
    except ValueError:
        return f"n{pos + 1:03d}"


def _cir_to_doc(text: str, cir: dict) -> dict:
    """Convertit un CausalIR dict (format Rust ou Python) en document JSON gcn-nl.

    TOKENS SYNTHÉTIQUES (B1) : un token par nœud, dérivé du label et du node_type.
    token_span=(i+1, i+1) pour chaque nœud — aligné sur l'id du token synthétique.
    Qualité approximative : root_morph={} → Tense/Aspect/Mood=_absent, is_negative=False.
    Pour la qualité maximale, remplacer les tokens par des annotations UD réelles.
    """
    nodes = cir.get("nodes", [])
    edges = cir.get("edges", [])

    # Assigner token_span=(i+1, i+1) cohérent avec les tokens synthétiques (id=i+1)
    doc_nodes = []
    for i, n in enumerate(nodes):
        nd = {
            "id": _canonical_node_id(n.get("id"), i),
            "type": n.get("node_type", NODE_TYPES[1]),
            "label": n.get("label", ""),
            "token_span": [i + 1, i + 1],
            "scope": n.get("scope", SCOPE_VALUES[4]),
            "temporal_index": n.get("temporal_index", 0),
            "origin": n.get("origin", NODE_ORIGIN_VALUES[0]),
        }
        for key in ("parent", "temporal_ref", "attributes"):
            if key in n:
                nd[key] = n[key]
        doc_nodes.append(nd)

    _raw_edges = [_normalize_edge(e) for e in edges]
    doc_edges = []
    for r in _raw_edges:
        if r is None:
            continue
        if isinstance(r, list):
            doc_edges.extend(r)  # JOINT_CAUSE → 2 arêtes
        else:
            doc_edges.append(r)

    return {
        "document": {
            "sentences": [
                {
                    "id": "s001",
                    "text": text,
                    "tokens": _synthetic_tokens(nodes),
                    "cir": {
                        "nodes": doc_nodes,
                        "edges": doc_edges,
                    },
                }
            ],
        }
    }
