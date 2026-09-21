# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import click

from ..constants import NODE_TYPES, RELATION_TYPES, SCOPE_VALUES, NODE_ORIGIN_VALUES

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
@click.option("--taxonomy-dir", default=None, type=click.Path(path_type=Path),
              envvar="GCN_TAXONOMY_DIR",
              help="Répertoire des taxonomies (ou env GCN_TAXONOMY_DIR)")
@click.option("--gcn-bin", default="gcn", show_default=True,
              help="Chemin vers le binaire gcn-cli Rust")
def bootstrap_cmd(
    input_file: Path, out_dir: Path,
    taxonomy_dir: Path | None, gcn_bin: str,
) -> None:
    """Génère des données d'entraînement JSON depuis des phrases brutes via gcn-cli Rust.

    Appelle `gcn analyze` (texte via stdin) pour chaque ligne, convertit le CausalIR JSON
    produit au format gcn-nl (document.sentences), et écrit les fichiers dans out-dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = [l.strip() for l in input_file.read_text("utf-8").splitlines() if l.strip()]

    if not texts:
        raise click.ClickException(f"Aucune phrase dans {input_file}")

    click.echo(f"Génération de {len(texts)} exemples vers {out_dir} ...")
    success = 0
    errors = 0

    # Résoudre gcn_bin une seule fois (évite la résolution PATH répétée + cohérence)
    from ..frontend.bridge import _resolve_gcn_bin, GCNBridgeError as _GCNBridgeError
    try:
        gcn_bin_resolved = _resolve_gcn_bin(gcn_bin)
    except _GCNBridgeError as exc:
        raise click.ClickException(str(exc)) from exc

    for i, text in enumerate(texts):
        try:
            cmd_args = [gcn_bin_resolved, "analyze"]
            if taxonomy_dir:
                cmd_args += ["--data-dir", str(taxonomy_dir)]
            # -- sépare explicitement les options du texte (évite "--option" parsé comme flag)
            cmd_args += ["--", text]
            # Issue #2 CRITICAL : encodage UTF-8 explicite
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=30,
            )
            if result.returncode != 0:
                click.echo(f"  [{i+1}] Erreur gcn-cli : {result.stderr.strip()}", err=True)
                errors += 1
                continue

            cir = json.loads(result.stdout)
            doc = _cir_to_doc(text, cir)
            out_path = out_dir / f"generated_{i+1:04d}.json"
            out_path.write_text(
                json.dumps(doc, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            success += 1
        except subprocess.TimeoutExpired:
            click.echo(f"  [{i+1}] Timeout", err=True)
            errors += 1
        except (json.JSONDecodeError, KeyError) as exc:
            click.echo(f"  [{i+1}] Parse error: {exc}", err=True)
            errors += 1

    click.echo(f"Terminé : {success} succès, {errors} erreurs.")
    if success == 0 and errors > 0:
        click.echo(
            f"ATTENTION : 0 document généré sur {errors} tentative(s) — "
            "répertoire de sortie probablement vide. Vérifiez gcn-cli et le format d'entrée.",
            err=True,
        )


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


def _normalize_edge(e) -> dict | None:
    """
    Normalise une arête CIR vers le format doc gcn-nl.

    Supporte :
      - format tuple/list : [src_id, dst_id, edge_obj]  ← sortie gcn analyze (Rust)
      - format dict       : {"source": ..., "target": ..., "relation": ...}
    Retourne None si le format est invalide ou incomplet.
    """
    if isinstance(e, (list, tuple)) and len(e) == 3:
        src_id, dst_id, edge_obj = e
        if not isinstance(edge_obj, dict):
            return None
        try:
            from ..data.edge_norm import normalize_node_id
            source, target = normalize_node_id(src_id), normalize_node_id(dst_id)
        except ValueError:
            return None
    elif isinstance(e, dict):
        edge_obj = e
        try:
            from ..data.edge_norm import normalize_node_id
            source = normalize_node_id(e.get("source", ""))
            target = normalize_node_id(e.get("target", ""))
        except ValueError:
            return None
    else:
        return None
    relation = edge_obj.get("relation_type", edge_obj.get("relation", RELATION_TYPES[0]))
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "confidence": float(edge_obj["confidence"]) if "confidence" in edge_obj else None,
        "explicit": bool(edge_obj.get("explicit", True)),
        "negated": bool(edge_obj["negated"]) if "negated" in edge_obj else None,
    }


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
    doc_nodes = [
        {
            "id": _canonical_node_id(n.get("id"), i),
            "type": n.get("node_type", NODE_TYPES[1]),
            "label": n.get("label", ""),
            "token_span": [i + 1, i + 1],
            "scope": n.get("scope", SCOPE_VALUES[4]),
            "temporal_index": n.get("temporal_index", 0),
            "origin": n.get("origin", NODE_ORIGIN_VALUES[0]),
        }
        for i, n in enumerate(nodes)
    ]

    doc_edges = [d for e in edges if (d := _normalize_edge(e)) is not None]

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
