from __future__ import annotations
import json
import sys
from pathlib import Path
import click

from ..taxonomy.loader import TaxonomyIndex
from ..layer1.features import FeatureVocabulary
from ..layer2.reference import MLPEncoder
from ..layer3.reference import RGCNLayer
from .cgnp import CGNPipeline


# Default: sibling gcn-core directory
DEFAULT_TAXONOMY_DIR = Path(__file__).parents[5] / "gcn-core" / "data" / "taxonomies"


@click.command("gcn-forward")
@click.argument("text")
@click.option("--lang", default="fr", show_default=True, help="Language code (fr, en, …)")
@click.option(
    "--taxonomy-dir", type=click.Path(path_type=Path),
    default=None, envvar="GCN_TAXONOMY_DIR",
    help="Path to taxonomies directory",
)
@click.option("--pretty/--compact", default=True, help="Pretty-print JSON output")
@click.option(
    "--model-path", type=click.Path(path_type=Path), default=None,
    help="Checkpoint .npz (produit par gcn-train). Sans ce flag : poids aléatoires.",
)
def forward_cmd(text: str, lang: str, taxonomy_dir: Path | None, pretty: bool,
                model_path: Path | None):
    """
    Run the CGNP forward pass: text → CausalIR JSON → stdout.

    Uses reference implementations (NumPy) for Layers 2 and 3.
    Replace with trained models via the CausalEncoder / CausalGraph protocols.
    """
    if taxonomy_dir is None:
        taxonomy_dir = DEFAULT_TAXONOMY_DIR

    if not taxonomy_dir.is_dir():
        raise click.ClickException(
            f"Taxonomy directory not found: {taxonomy_dir}\n"
            f"Set GCN_TAXONOMY_DIR or pass --taxonomy-dir"
        )

    tax = TaxonomyIndex.load(taxonomy_dir, lang)
    vocab = FeatureVocabulary.build(tax)
    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge)
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)

    pipeline = CGNPipeline(
        encoder=encoder,
        graph=graph,
        taxonomy_dir=taxonomy_dir,
        lang=lang,
        vocabulary=vocab,
    )

    if model_path is not None:
        from ..training.checkpoint import load_checkpoint
        if not model_path.exists():
            raise click.ClickException(f"Checkpoint introuvable : {model_path}")
        load_checkpoint(pipeline, model_path)
    else:
        click.echo("Avertissement : poids aléatoires (pas de --model-path)", file=sys.stderr)

    result = pipeline.forward(text)
    click.echo(json.dumps(result, ensure_ascii=False, indent=2 if pretty else None))
