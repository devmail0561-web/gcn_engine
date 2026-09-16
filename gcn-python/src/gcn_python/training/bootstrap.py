from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import click


@click.command("gcn-bootstrap")
@click.option("--input", "input_file", required=True, type=click.Path(path_type=Path),
              help="Fichier texte (.txt) — une phrase par ligne")
@click.option("--lang", default="fr", show_default=True)
@click.option("--out-dir", required=True, type=click.Path(path_type=Path),
              help="Répertoire de sortie pour les fichiers JSON générés")
@click.option("--taxonomy-dir", default=None, type=click.Path(path_type=Path),
              envvar="GCN_TAXONOMY_DIR",
              help="Répertoire des taxonomies (ou env GCN_TAXONOMY_DIR)")
@click.option("--gcn-bin", default="gcn", show_default=True,
              help="Chemin vers le binaire gcn-cli Rust")
def bootstrap_cmd(
    input_file: Path, lang: str, out_dir: Path,
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

    for i, text in enumerate(texts):
        try:
            cmd_args = [gcn_bin, "analyze"]
            if taxonomy_dir:
                cmd_args += ["--data-dir", str(taxonomy_dir)]
            # Issue #1 CRITICAL : text comme argument positionnel, pas stdin
            cmd_args.append(text)
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
            doc = _cir_to_doc(text, lang, cir)
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
    return list(flat) if flat else [0, 0]


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
        source, target = str(src_id), str(dst_id)
    elif isinstance(e, dict):
        edge_obj = e
        source = str(e.get("source", ""))
        target = str(e.get("target", ""))
    else:
        return None
    relation = edge_obj.get("relation_type", edge_obj.get("relation", "cause"))
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "confidence": float(edge_obj.get("confidence", 1.0)),
        "explicit": bool(edge_obj.get("explicit", True)),
        "negated": bool(edge_obj.get("negated", False)),
    }


def _cir_to_doc(text: str, lang: str, cir: dict) -> dict:
    """Convertit un CausalIR dict (format Rust ou Python) en document JSON gcn-nl."""
    nodes = cir.get("nodes", [])
    edges = cir.get("edges", [])

    doc_nodes = [
        {
            "id": n.get("id", f"n{i+1:03d}"),
            "type": n.get("node_type", "action"),
            "label": n.get("label", ""),
            "token_span": _extract_token_span(n),
            "scope": n.get("scope", "specific"),
            "temporal_index": n.get("temporal_index", 0),
            "origin": n.get("origin", "explicit"),
        }
        for i, n in enumerate(nodes)
    ]

    doc_edges = [d for e in edges if (d := _normalize_edge(e)) is not None]

    return {
        "document": {
            "lang": lang,
            "sentences": [
                {
                    "id": "s001",
                    "text": text,
                    "tokens": [],
                    "cir": {
                        "nodes": doc_nodes,
                        "edges": doc_edges,
                    },
                }
            ],
        }
    }
