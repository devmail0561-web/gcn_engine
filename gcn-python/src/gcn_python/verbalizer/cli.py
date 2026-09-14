from __future__ import annotations
import sys
import click
from .decoder import ReferenceDecoder


@click.command("gcn-verbalize")
@click.argument("ir", type=click.File("r"), default="-")
def verbalize_cmd(ir: click.File) -> None:
    """
    Decode a CausalIR JSON to a surface form.

    IR: path to a CausalIR JSON file, or '-' to read from stdin.

    What the decoder produces depends on its training data.
    No explicit parameter needed.
    """
    ir_json = ir.read()
    decoder = ReferenceDecoder()
    try:
        result = decoder.decode(ir_json)
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)
    click.echo(result)
