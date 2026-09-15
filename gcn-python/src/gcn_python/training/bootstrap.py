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
@click.option("--gcn-bin", default="gcn", show_default=True,
              help="Chemin vers le binaire gcn-cli Rust")
def bootstrap_cmd(input_file: Path, lang: str, out_dir: Path, gcn_bin: str) -> None:
    """Génère des données d'entraînement JSON depuis des phrases brutes via gcn-cli Rust.

    Appelle `gcn analyze <texte>` pour chaque ligne, convertit le CausalIR JSON
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
            result = subprocess.run(
                [gcn_bin, "analyze", text, "--lang", lang, "--compact"],
                capture_output=True, text=True, timeout=30,
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


def _cir_to_doc(text: str, lang: str, cir: dict) -> dict:
    """Convertit un CausalIR dict (format Rust/JSON) en document JSON gcn-nl."""
    nodes = cir.get("nodes", [])
    edges = cir.get("edges", [])

    doc_nodes = [
        {
            "id": n.get("id", f"n{i+1:03d}"),
            "type": n.get("node_type", "action"),
            "label": n.get("label", ""),
            "token_span": list(n.get("token_span", [0, 0])),
            "scope": n.get("scope", "specific"),
            "temporal_index": n.get("temporal_index", 0),
            "origin": n.get("origin", "explicit"),
        }
        for i, n in enumerate(nodes)
    ]

    doc_edges = [
        {
            "source": e.get("source", ""),
            "target": e.get("target", ""),
            "relation": e.get("relation_type", e.get("relation", "cause")),
            "confidence": float(e.get("confidence", 1.0)),
            "explicit": bool(e.get("explicit", True)),
            "negated": bool(e.get("negated", False)),
        }
        for e in edges
    ]

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
