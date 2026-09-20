# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
gcn-index — indexation batch d'un corpus → graphe causal persistant.

Analyse tous les fichiers texte d'un répertoire (ou un fichier unique)
et construit un graphe causal JSON sans interaction utilisateur.
Utile pour pré-indexer un grand corpus avant gcn-discuss.

Usage :
    gcn-index --corpus rapports/ --checkpoint model.npz --output graph.json
    gcn-discuss --checkpoint model.npz --graph graph.json
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from .discuss import _read_texts, _split_lines
from .verbalizer.instructions import CausalGraph


@click.command("gcn-index")
@click.option("--corpus", required=True, type=click.Path(path_type=Path),
              help="Fichier texte ou répertoire à indexer.")
@click.option("--checkpoint", required=True, type=click.Path(path_type=Path),
              help="Checkpoint .npz produit par gcn-train.")
@click.option("--output", required=True, type=click.Path(path_type=Path),
              help="Fichier JSON de sortie pour le graphe causal.")
@click.option("--gcn-bin", default="gcn", show_default=True,
              help="Chemin vers le binaire gcn-cli Rust.")
@click.option("--append/--overwrite", default=False, show_default=True,
              help="Enrichir un graphe existant (--append) ou le remplacer.")
def index_cmd(
    corpus: Path,
    checkpoint: Path,
    output: Path,
    gcn_bin: str,
    append: bool,
) -> None:
    """Indexe un corpus de fichiers texte → graphe causal JSON."""
    from .engine import GCNEngine

    if not corpus.exists():
        raise click.ClickException(f"Corpus introuvable : {corpus}")
    if not checkpoint.exists():
        raise click.ClickException(f"Checkpoint introuvable : {checkpoint}")

    # Charger le moteur
    click.echo(f"Chargement du modèle depuis {checkpoint.name}...")
    engine = GCNEngine.from_pretrained(checkpoint, gcn_bin=gcn_bin, trusted=True)
    engine._pipeline.encoder.training = False

    # Charger ou créer le graphe
    graph = CausalGraph()
    if append and output.exists():
        graph = CausalGraph.load(output)
        click.echo(f"Graphe existant chargé : {len(graph.edges)} relations")

    # Lire les fichiers
    texts = _read_texts(corpus)
    if not texts:
        raise click.ClickException("Aucun fichier texte trouvé.")

    click.echo(f"Indexation de {len(texts)} fichier(s)...")
    total = 0
    for filename, content in texts:
        lines = _split_lines(content)
        n_new = 0
        for line in lines:
            try:
                cir = engine.analyze(line)
                if cir.get("edges"):
                    graph.add_cir(cir)
                    n_new += len(cir["edges"])
            except Exception:
                pass
        total += n_new
        click.echo(f"  {filename:<40} {n_new:4d} relation(s)")

    graph.save(output)
    click.echo(f"\nGraphe sauvegardé : {output}  ({len(graph.edges)} relations au total)")
