"""
gcn-chat — Session interactive du moteur GCN Causal Engine.

Commandes :
    <texte>                     → analyse causale + CIR
    explain: <concept>          → causes de ce concept
    effects: <concept>          → effets de ce concept
    chain: <concept_a> <concept_b>  → chemin causal
    counterfactual: <concept>   → Pearl do-calculus
    summarize                   → aperçu de session
    verbalize                   → CIR → texte structuré
    help                        → aide
    clear                       → vider la session
    quit                        → quitter

Les commandes sont en anglais (fixes). Le contenu analysé peut être
en n'importe quelle langue — les labels des nœuds viennent du texte.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click

from .verbalizer.instructions import InstructionHandler, parse_command


# ---------------------------------------------------------------------------
# Formatage CIR → affichage lisible
# ---------------------------------------------------------------------------

def _format_cir(cir: dict) -> str:
    nodes = cir.get("nodes", [])
    edges = cir.get("edges", [])
    if not nodes:
        return "  (no structure)"
    node_map = {}
    lines = []
    for n in nodes:
        if isinstance(n, dict):
            nid   = n.get("id")
            ntype = n.get("node_type", "?")
            label = n.get("label", "")
            node_map[nid] = (ntype, label)
            lines.append(f"  [{ntype}]  {label}")
    if edges:
        lines.append("")
        for e in edges:
            if isinstance(e, (list, tuple)) and len(e) == 3:
                src, dst, attrs = e
                rel  = attrs.get("relation", "?")
                conf = attrs.get("confidence", 0.0)
                neg  = " [negated]" if attrs.get("negated") else ""
                src_lbl = node_map.get(src, ("?", str(src)))[1]
                dst_lbl = node_map.get(dst, ("?", str(dst)))[1]
                lines.append(
                    f"  {src_lbl}  --[{rel}{neg}]-->  {dst_lbl}  ({conf:.0%})"
                )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Segmentation de texte (heuristique, sans dépendance linguistique)
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """
    Segmente un texte en phrases.
    Découpe sur ponctuation forte suivie d'espace + majuscule (universel).
    """
    import re
    raw = re.split(r'(?<=[.!?])\s+(?=\S)', text.strip())
    return [s.strip() for s in raw if len(s.split()) >= 3]


# ---------------------------------------------------------------------------
# Session principale
# ---------------------------------------------------------------------------

def _print_welcome():
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║     GCN Causal Engine — Interactive Session      ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()
    print("  Type text to analyze, or a command (type 'help').")
    print()


def run_chat(checkpoint: Optional[Path] = None, gcn_bin: str = "gcn") -> None:
    from .engine import GCNEngine

    _print_welcome()

    engine = None
    if checkpoint and checkpoint.exists():
        print(f"  Loading model from {checkpoint.name}...", end="", flush=True)
        try:
            engine = GCNEngine.from_pretrained(checkpoint, gcn_bin=gcn_bin)
            print(" ready.\n")
        except Exception as e:
            print(f"\n  Load error: {e}\n")
    else:
        print("  No checkpoint — analysis unavailable.")
        print("  Start with: gcn-chat --checkpoint model.npz\n")

    handler = InstructionHandler()

    while True:
        try:
            user_input = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  bye.")
            break

        if not user_input:
            continue

        cmd, arg = parse_command(user_input)

        # Commandes de contrôle
        if cmd == "quit":
            print("  bye.")
            break

        if cmd == "clear":
            handler.clear()
            print("  session cleared.\n")
            continue

        if cmd == "help":
            print()
            print(handler.execute("help"))
            print()
            continue

        # Commandes sur le graphe causal
        if cmd != "text":
            response = handler.execute(user_input)
            if response:
                print()
                print(response)
                print()
            continue

        # Texte à analyser
        if engine is None:
            print("  no model loaded — cannot analyze.")
            continue

        sentences = _split_sentences(user_input) or [user_input]
        print()
        for sent in sentences:
            try:
                cir = engine.analyze(sent)
                handler.add_cir(cir)
                print(f"  \"{sent[:90]}{'...' if len(sent) > 90 else ''}\"")
                formatted = _format_cir(cir)
                if "(no structure)" in formatted:
                    print("  (no causal relation detected)")
                else:
                    print(formatted)
            except Exception as e:
                print(f"  error: {e}")
            print()


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

@click.command("gcn-chat")
@click.option("--checkpoint", "ckpt", default=None, type=click.Path(path_type=Path),
              help="Checkpoint .npz from gcn-train.")
@click.option("--gcn-bin", default="gcn", show_default=True,
              help="Path to gcn-cli Rust binary.")
def _chat_cmd(ckpt: Optional[Path], gcn_bin: str) -> None:
    """Interactive causal analysis session."""
    run_chat(checkpoint=ckpt, gcn_bin=gcn_bin)
