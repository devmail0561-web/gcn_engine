from __future__ import annotations
import json
import sys
from pathlib import Path
import click

from ..engine import GCNEngine
from ..data.json_reader import load_sentences
from ..data.loader import reps_from_sentence


@click.command("gcn-forward")
@click.option("--text", default=None,
              help="[DEBUG] Analyser une seule phrase. Pour les corpus, utiliser --file.")
@click.option("--file", "input_file", default=None,
              type=click.Path(path_type=Path, exists=True),
              help="Fichier texte : une phrase par ligne (mode batch).")
@click.option("--dataset", "dataset_path", default=None,
              type=click.Path(path_type=Path, exists=True),
              help="Fichier JSON annoté au format gcn-nl (mode dataset).")
@click.option("--checkpoint", "model_path", default=None,
              type=click.Path(path_type=Path),
              help="Checkpoint .npz produit par gcn-train.")
@click.option("--gcn-bin", default="gcn", show_default=True,
              help="Chemin vers le binaire gcn-cli Rust.")
@click.option("--pretty/--compact", default=True, show_default=True,
              help="JSON indenté ou compact.")
def forward_cmd(
    text: str | None,
    input_file: Path | None,
    dataset_path: Path | None,
    model_path: Path | None,
    gcn_bin: str,
    pretty: bool,
) -> None:
    """
    Analyse du texte et produit un CausalIR JSON sur stdout.

    Trois modes d'entrée :

    \b
      Texte brut  : gcn-forward --text "phrase" --checkpoint model.npz
      Fichier     : gcn-forward --file corpus.txt --checkpoint model.npz
      Dataset JSON: gcn-forward --dataset data.json --checkpoint model.npz
    """
    indent = 2 if pretty else None

    # --- Mode texte brut ou fichier : utilise GCNEngine ---
    if text is not None or input_file is not None:
        if model_path is None:
            raise click.ClickException(
                "--checkpoint requis en mode texte. "
                "Exemple : gcn-forward --text \"...\" --checkpoint model.npz"
            )
        engine = GCNEngine.from_pretrained(model_path, gcn_bin=gcn_bin)

        if text is not None:
            result = engine.analyze(text)
            click.echo(json.dumps(result, ensure_ascii=False, indent=indent))

        else:
            lines = [l.strip() for l in input_file.read_text("utf-8").splitlines() if l.strip()]
            results = []
            for line in lines:
                cir = engine.analyze(line)
                results.append(cir)
                if not pretty:
                    click.echo(json.dumps(cir, ensure_ascii=False))
            if pretty:
                click.echo(json.dumps(results, ensure_ascii=False, indent=indent))
        return

    # --- Mode dataset JSON annoté (rétrocompatibilité) ---
    if dataset_path is not None:
        from ..layer1.features import FeatureVocabulary
        from ..layer2.reference import MLPEncoder
        from ..layer3.reference import RGCNLayer
        from ..pipeline.cgnp import CGNPipeline

        records = load_sentences(dataset_path)
        if not records:
            raise click.ClickException(f"Aucune sentence chargée depuis {dataset_path}")

        rec = records[0]
        reps, valid_clause_idxs, connector_reps = reps_from_sentence(rec)
        if not reps:
            raise click.ClickException(
                f"La sentence {rec.id!r} ne contient pas de tokens annotés."
            )

        vocab = FeatureVocabulary()
        encoder = MLPEncoder(d_clause=vocab.d_clause,
                             d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
        graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
        pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)

        if model_path is not None:
            from ..training.checkpoint import load_checkpoint
            load_checkpoint(pipeline, model_path)
        else:
            click.echo("Avertissement : poids aléatoires (pas de --checkpoint)", file=sys.stderr)

        result = pipeline.forward(
            reps, rec.text,
            clause_positions=valid_clause_idxs,
            n_total_clauses=len(rec.clauses),
            connector_reps=connector_reps,
        )
        click.echo(json.dumps(result, ensure_ascii=False, indent=indent))
        return

    raise click.UsageError(
        "Fournir --text, --file, ou --dataset.\n"
        "Exemple : gcn-forward --text \"Les ventes baissent car la demande recule.\" "
        "--checkpoint model.npz"
    )
