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
from ..data.json_reader import load_sentences
from ..data.loader import reps_from_sentence


DEFAULT_TAXONOMY_DIR = Path(__file__).parents[5] / "gcn-core" / "data" / "taxonomies"


@click.command("gcn-forward")
@click.argument("dataset_path", type=click.Path(path_type=Path, exists=True))
@click.option("--sentence-id", default=None,
              help="ID de la sentence dans le fichier (défaut : première)")
@click.option("--lang", default="fr", show_default=True, help="Code langue (fr, en, …)")
@click.option(
    "--taxonomy-dir", type=click.Path(path_type=Path),
    default=None, envvar="GCN_TAXONOMY_DIR",
    help="Chemin du répertoire de taxonomies",
)
@click.option("--pretty/--compact", default=True, help="JSON indenté ou compact")
@click.option(
    "--model-path", type=click.Path(path_type=Path), default=None,
    help="Checkpoint .npz (produit par gcn-train). Sans ce flag : poids aléatoires.",
)
def forward_cmd(
    dataset_path: Path,
    sentence_id: str | None,
    lang: str,
    taxonomy_dir: Path | None,
    pretty: bool,
    model_path: Path | None,
) -> None:
    """
    Run the CGNP forward pass: dataset JSON annoté → CausalIR JSON → stdout.

    DATASET_PATH doit être un fichier au format dataset GCN-NL (tokens + cir).
    """
    if taxonomy_dir is None:
        taxonomy_dir = DEFAULT_TAXONOMY_DIR
    if not taxonomy_dir.is_dir():
        raise click.ClickException(
            f"Répertoire taxonomies introuvable : {taxonomy_dir}\n"
            f"Définir GCN_TAXONOMY_DIR ou passer --taxonomy-dir"
        )

    records = load_sentences(dataset_path, lang)
    if not records:
        raise click.ClickException(f"Aucune sentence chargée depuis {dataset_path}")

    if sentence_id:
        rec = next((r for r in records if r.id == sentence_id), None)
        if rec is None:
            raise click.ClickException(
                f"Sentence {sentence_id!r} introuvable dans {dataset_path}. "
                f"IDs disponibles : {[r.id for r in records]}"
            )
    else:
        rec = records[0]

    reps, valid_clause_idxs, connector_reps = reps_from_sentence(rec)
    if not reps:
        raise click.ClickException(
            f"La sentence {rec.id!r} ne contient pas de tokens annotés "
            f"(format paper_examples non supporté ici — utiliser un fichier dataset avec tokens)."
        )

    tax = TaxonomyIndex.load(taxonomy_dir, lang)
    vocab = FeatureVocabulary.build(tax)
    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge)
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipeline = CGNPipeline(
        encoder=encoder, graph=graph,
        taxonomy_dir=taxonomy_dir, lang=lang, vocabulary=vocab,
    )

    if model_path is not None:
        from ..training.checkpoint import load_checkpoint
        if not model_path.exists():
            raise click.ClickException(f"Checkpoint introuvable : {model_path}")
        load_checkpoint(pipeline, model_path)
    else:
        click.echo("Avertissement : poids aléatoires (pas de --model-path)", file=sys.stderr)

    result = pipeline.forward(
        reps, rec.text,
        clause_positions=valid_clause_idxs,
        n_total_clauses=len(rec.clauses),
        connector_reps=connector_reps,
    )
    click.echo(json.dumps(result, ensure_ascii=False, indent=2 if pretty else None))
