"""
gcn-discuss — Session de discussion causale sur corpus.

Commandes (préfixe /) :
  /analyze <fichier_ou_répertoire>  — analyser un fichier ou un répertoire entier
  /save    <path.json>              — persister le graphe de session
  /load    <path.json>              — charger un graphe existant
  /summarize                        — résumé de la session
  /help                             — aide
  /quit                             — quitter

Questions libres (sans préfixe /) :
  Toute entrée sans / est traitée comme une question sur le graphe accumulé.
  Si aucune structure causale n'est trouvée sur le sujet : message explicite.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from .verbalizer.instructions import InstructionHandler, CausalGraph, parse_command
from .verbalizer.query_report import QueryVerbalizer


# ---------------------------------------------------------------------------
# Lecture des fichiers texte (raw text — pas de JSON annoté)
# ---------------------------------------------------------------------------

def _read_texts(path: Path) -> list[tuple[str, str]]:
    """
    Lit un fichier ou tous les fichiers texte d'un répertoire.
    Retourne une liste de (nom_fichier, contenu_texte).
    Formats supportés : .txt, .md, .log, .csv (première colonne).
    Les fichiers .json sont ignorés — ce sont des données annotées, pas du texte brut.
    """
    TEXT_EXTENSIONS = {".txt", ".md", ".log", ".text", ".rst"}
    results = []

    if path.is_file():
        if path.suffix.lower() in TEXT_EXTENSIONS:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                results.append((path.name, content))
            except Exception as e:
                print(f"  ⚠ Impossible de lire {path.name} : {e}")
        else:
            print(f"  ⚠ Format non supporté : {path.suffix}. "
                  f"Formats acceptés : {', '.join(sorted(TEXT_EXTENSIONS))}")
    elif path.is_dir():
        files = sorted(p for p in path.iterdir()
                       if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS)
        if not files:
            print(f"  ⚠ Aucun fichier texte trouvé dans {path}")
        for f in files:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                results.append((f.name, content))
            except Exception as e:
                print(f"  ⚠ {f.name} : {e}")
    else:
        print(f"  ⚠ Chemin introuvable : {path}")

    return results


def _split_lines(text: str) -> list[str]:
    """
    Retourne les lignes non vides d'un texte.
    Chaque ligne est traitée comme une unité d'analyse.
    """
    return [l.strip() for l in text.splitlines() if len(l.strip()) >= 10]


# ---------------------------------------------------------------------------
# Formatage des rapports de réponse
# ---------------------------------------------------------------------------

def _format_response(handler: InstructionHandler, question: str) -> str:
    """
    Répond à une question sur le graphe accumulé via QueryVerbalizer.
    Détecte le type de requête depuis la structure (pas la langue).
    """
    vb = QueryVerbalizer(handler.graph)
    q  = question.lower().strip().rstrip("?.,!")

    # Commandes formelles (explain:, effects:, chain:, counterfactual:)
    cmd, arg = parse_command(question)
    if cmd == "explain" and arg:
        return vb.causes(arg)
    if cmd == "effects" and arg:
        return vb.effects(arg)
    if cmd == "chain" and arg:
        parts = arg.split()
        if len(parts) >= 2:
            return vb.path(parts[0], parts[-1])
    if cmd == "counterfactual" and arg:
        return vb.counterfactual(arg)
    if cmd == "summarize":
        return vb.summarize()
    if cmd == "verbalize":
        return handler.execute("verbalize")

    # Heuristique structurelle : dernier mot non-vide = sujet probable
    words = [w.strip("?.,!()") for w in question.split() if len(w.strip("?.,!()")) > 2]
    if not words:
        return vb.summarize()

    subject = words[-1]

    # Chercher dans le graphe
    causes  = handler.graph.find_causes(subject)
    effects = handler.graph.find_effects(subject)

    if causes and effects:
        return vb.causes(subject) + "\n\n" + vb.effects(subject)
    if causes:
        return vb.causes(subject)
    if effects:
        return vb.effects(subject)

    # Rien trouvé
    return (
        f"  No causal structure found for {subject!r} in the submitted corpus.\n"
        f"  Use /analyze <file> to add documents."
    )


# ---------------------------------------------------------------------------
# Session principale
# ---------------------------------------------------------------------------

def _print_header(n_relations: int = 0, n_sources: int = 0) -> None:
    print()
    print("  GCN Causal Engine")
    print("  ─────────────────────────────────────────────────────")
    if n_relations:
        print(f"  Corpus : {n_relations} relation(s), {n_sources} source(s)")
    else:
        print("  Corpus : vide  —  utilisez /analyze pour charger des documents")
    print()


HELP_TEXT = """
  Commandes (préfixe /) :
    /analyze <fichier_ou_répertoire>  — analyser un fichier texte ou un répertoire
    /save    <path.json>              — sauvegarder le graphe de session
    /load    <path.json>              — charger un graphe existant
    /summarize                        — résumé du corpus courant
    /help                             — ce message
    /quit                             — quitter

  Questions (sans préfixe /) :
    Toute entrée sans / est traitée comme une question sur le corpus analysé.
    Exemples :
      What causes data exfiltration?
      How does phishing lead to ransomware?
      explain: authentication_bypass
      effects: CVE-2024-1234
      chain: phishing ransomware
      counterfactual: firewall_disabled
"""


def run_discuss(
    checkpoint: Optional[Path] = None,
    graph_path: Optional[Path] = None,
    gcn_bin: str = "gcn",
) -> None:
    """Lance la session de discussion."""
    from .engine import GCNEngine

    # Charger le moteur
    engine = None
    if checkpoint and checkpoint.exists():
        try:
            engine = GCNEngine.from_pretrained(checkpoint, gcn_bin=gcn_bin)
            engine._pipeline.encoder.training = False
        except Exception as e:
            print(f"\n  Erreur de chargement du checkpoint : {e}")
            print("  Impossible d'analyser sans checkpoint.")

    # Charger ou créer le graphe de session
    handler = InstructionHandler()
    if graph_path and graph_path.exists():
        try:
            handler.graph = CausalGraph.load(graph_path)
            print(f"\n  Graphe chargé : {graph_path.name}")
        except Exception as e:
            print(f"\n  Erreur de chargement du graphe : {e}")

    _print_header(
        len(handler.graph.edges),
        len({text for _, _, _, text in handler.graph.edges}),
    )

    while True:
        try:
            user_input = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  bye.")
            break

        if not user_input:
            continue

        # ── Commandes /commande ──────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input[1:].split(None, 1)
            cmd   = parts[0].lower() if parts else ""
            arg   = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("quit", "exit", "q"):
                print("  bye.")
                break

            elif cmd == "help":
                print(HELP_TEXT)

            elif cmd == "summarize":
                print()
                print(handler.execute("summarize"))
                print()

            elif cmd == "analyze":
                if not arg:
                    print("  Usage : /analyze <fichier_ou_répertoire>")
                    continue
                if engine is None:
                    print("  Aucun checkpoint chargé — impossible d'analyser.")
                    continue
                target = Path(arg)
                texts  = _read_texts(target)
                if not texts:
                    continue
                print()
                total_new = 0
                for filename, content in texts:
                    lines = _split_lines(content)
                    n_new = 0
                    for line in lines:
                        try:
                            cir = engine.analyze(line)
                            if cir.get("edges"):
                                handler.add_cir(cir)
                                n_new += len(cir["edges"])
                        except Exception:
                            pass
                    total_new += n_new
                    print(f"  {filename} : {n_new} relation(s) extraite(s)")
                n_total = len(handler.graph.edges)
                print(f"  Total session : {n_total} relation(s)")
                print()

            elif cmd == "save":
                if not arg:
                    print("  Usage : /save <path.json>")
                    continue
                try:
                    handler.graph.save(Path(arg))
                    print(f"  Graphe sauvegardé : {arg}  ({len(handler.graph.edges)} relations)")
                except Exception as e:
                    print(f"  Erreur de sauvegarde : {e}")

            elif cmd == "load":
                if not arg:
                    print("  Usage : /load <path.json>")
                    continue
                try:
                    handler.graph = CausalGraph.load(Path(arg))
                    n = len(handler.graph.edges)
                    print(f"  Graphe chargé : {arg}  ({n} relations)")
                except Exception as e:
                    print(f"  Erreur de chargement : {e}")

            else:
                print(f"  Commande inconnue : /{cmd}   (tapez /help)")

        # ── Question sur le corpus ───────────────────────────────────────
        else:
            if not handler.graph.edges:
                print("  Corpus vide. Utilisez /analyze <fichier> pour charger des documents.")
                continue
            print()
            response = _format_response(handler, user_input)
            print(response)
            print()


# ---------------------------------------------------------------------------
# Point d'entrée CLI
# ---------------------------------------------------------------------------

@click.command("gcn-discuss")
@click.option("--checkpoint", "ckpt", default=None, type=click.Path(path_type=Path),
              help="Checkpoint .npz produit par gcn-train.")
@click.option("--graph", "graph_path", default=None, type=click.Path(path_type=Path),
              help="Graphe de session JSON à charger (produit par /save).")
@click.option("--gcn-bin", default="gcn", show_default=True,
              help="Chemin vers le binaire gcn-cli Rust.")
def discuss_cmd(
    ckpt: Optional[Path],
    graph_path: Optional[Path],
    gcn_bin: str,
) -> None:
    """Session de discussion causale sur corpus — /analyze, questions libres, /save."""
    run_discuss(checkpoint=ckpt, graph_path=graph_path, gcn_bin=gcn_bin)
