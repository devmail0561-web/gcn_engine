# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import sys
import warnings
import click
from .decoder import ReferenceDecoder


@click.command("gcn-verbalize")
@click.argument("ir", type=click.File("r"), default="-")
@click.option("--quiet", is_flag=True, default=False,
              help="Supprime les avertissements : stdout = texte verbalisé pur, "
                   "consommable par le pont Rust même avec -W default.")
def verbalize_cmd(ir: click.File, quiet: bool) -> None:
    """
    Decode a CausalIR JSON to a surface form.

    IR: path to a CausalIR JSON file, or '-' to read from stdin.

    Uses ReferenceDecoder (deterministic templates) — always available, no
    training needed. The neural TrainableDecoder is NOT used here, even if
    a checkpoint contains one.
    """
    ir_json = ir.read()
    decoder = ReferenceDecoder()
    try:
        if quiet:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = decoder.decode(ir_json)
        else:
            result = decoder.decode(ir_json)
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)
    click.echo(result)
