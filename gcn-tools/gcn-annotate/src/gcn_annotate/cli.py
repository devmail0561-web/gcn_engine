"""CLI pour gcn-annotate — outil d'annotation LLM."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click


@click.group()
def main() -> None:
    """Outil d'annotation LLM pour le projet GCN-NL."""
    pass


@main.command("annotate")
@click.option("--input", "input_path", required=True, type=click.Path(path_type=Path, exists=True),
              help="Fichier texte avec une phrase par ligne")
@click.option("--output", "output_dir", required=True, type=click.Path(path_type=Path),
              help="Répertoire de sortie pour les JSON annotés")
@click.option("--lang", default="fr", show_default=True, help="Langue des phrases")
@click.option("--llm-backend", default="anthropic", show_default=True,
              type=click.Choice(["anthropic", "openai"]),
              help="Backend LLM à utiliser")
@click.option("--model", default=None, help="Nom du modèle (défaut: dépend du backend)")
@click.option("--batch-size", default=10, show_default=True, type=int,
              help="Nombre de phrases par appel API")
def annotate_cmd(
    input_path: Path,
    output_dir: Path,
    lang: str,
    llm_backend: str,
    model: str | None,
    batch_size: int,
) -> None:
    """Annote un fichier de phrases et produit des JSON GCN-NL."""
    sentences = [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    if not sentences:
        raise click.ClickException(f"Aucune phrase dans {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Créer l'annotateur
    if llm_backend == "anthropic":
        from .annotator import AnthropicAnnotator
        kwargs = {}
        if model:
            kwargs["model"] = model
        annotator = AnthropicAnnotator(**kwargs)
    else:
        from .annotator import OpenAIAnnotator
        kwargs = {}
        if model:
            kwargs["model"] = model
        annotator = OpenAIAnnotator(**kwargs)

    click.echo(f"Annotateur : {llm_backend} | batch_size={batch_size} | {len(sentences)} phrases")

    all_sentences = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        batch_id = i // batch_size + 1
        click.echo(f"  Batch {batch_id} ({len(batch)} phrases)...", nl=False)
        try:
            batch_results = annotator.annotate(batch, lang=lang)
            all_sentences.extend(batch_results)
            click.echo(f" OK ({len(batch_results)} résultats)")
        except Exception as e:
            click.echo(f" ERREUR: {e}")
            continue

    if not all_sentences:
        raise click.ClickException("Aucune annotation produite")

    # Écrire le document final
    doc = {
        "document": {
            "id": f"llm-{input_path.stem}",
            "lang": lang,
            "sentences": all_sentences,
        }
    }
    output_path = output_dir / f"{input_path.stem}_llm.json"
    output_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    click.echo(f"Sortie : {output_path} ({len(all_sentences)} sentences)")


@main.command("eval")
@click.option("--gold", "gold_path", required=True, type=click.Path(path_type=Path, exists=True),
              help="Fichier JSON gold (annotations humaines)")
@click.option("--pred", "pred_path", required=True, type=click.Path(path_type=Path, exists=True),
              help="Fichier JSON prédit (annotations LLM)")
def eval_cmd(gold_path: Path, pred_path: Path) -> None:
    """Évalue la qualité des annotations LLM contre un gold standard."""
    gold_doc = json.loads(gold_path.read_text(encoding="utf-8"))
    pred_doc = json.loads(pred_path.read_text(encoding="utf-8"))

    gold_sents = gold_doc.get("document", gold_doc).get("sentences", [])
    pred_sents = pred_doc.get("document", pred_doc).get("sentences", [])

    if not gold_sents or not pred_sents:
        raise click.ClickException("Fichiers JSON vides ou format invalide")

    # Extraire les types de nœuds et relations
    gold_nodes = []
    pred_nodes = []
    gold_edges = []
    pred_edges = []

    for gs in gold_sents:
        for node in gs.get("cir", {}).get("nodes", []):
            gold_nodes.append(node.get("type", ""))
        for edge in gs.get("cir", {}).get("edges", []):
            gold_edges.append(edge.get("relation", ""))

    for ps in pred_sents:
        for node in ps.get("cir", {}).get("nodes", []):
            pred_nodes.append(node.get("type", ""))
        for edge in ps.get("cir", {}).get("edges", []):
            pred_edges.append(edge.get("relation", ""))

    # Calculer les métriques
    min_node_len = min(len(gold_nodes), len(pred_nodes))
    node_correct = sum(1 for i in range(min_node_len) if gold_nodes[i] == pred_nodes[i])
    node_acc = node_correct / max(min_node_len, 1)

    min_edge_len = min(len(gold_edges), len(pred_edges))
    edge_correct = sum(1 for i in range(min_edge_len) if gold_edges[i] == pred_edges[i])
    edge_acc = edge_correct / max(min_edge_len, 1)

    click.echo(f"Nodes : {min_node_len} alignés, accuracy = {node_acc:.3f}")
    click.echo(f"Edges : {min_edge_len} alignés, accuracy = {edge_acc:.3f}")
    click.echo(f"Gold nodes: {len(gold_nodes)} | Pred nodes: {len(pred_nodes)}")
    click.echo(f"Gold edges: {len(gold_edges)} | Pred edges: {len(pred_edges)}")


if __name__ == "__main__":
    main()
