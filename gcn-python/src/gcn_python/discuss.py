# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
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
from .verbalizer.decoder import ReferenceDecoder


# ---------------------------------------------------------------------------
# Lecture des fichiers texte (raw text — pas de JSON annoté)
# ---------------------------------------------------------------------------

def _read_texts(path: Path) -> list[tuple[str, str]]:
    """
    Lit un fichier ou tous les fichiers texte d'un répertoire.
    Retourne une liste de (nom_fichier, contenu_texte).
    Formats supportés : .txt, .md, .log, .text, .rst.
    Les fichiers .json sont ignorés — ce sont des données annotées, pas du texte brut.
    Les diagnostics partent sur stderr (click.echo err=True) pour ne jamais
    polluer une sortie standard consommée en aval (ex. gcn-index).
    """
    TEXT_EXTENSIONS = {".txt", ".md", ".log", ".text", ".rst"}
    results = []

    if path.is_file():
        if path.suffix.lower() in TEXT_EXTENSIONS:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                results.append((path.name, content))
            except Exception as e:
                click.echo(f"  ⚠ Impossible de lire {path.name} : {e}", err=True)
        else:
            click.echo(f"  ⚠ Format non supporté : {path.suffix}. "
                       f"Formats acceptés : {', '.join(sorted(TEXT_EXTENSIONS))}", err=True)
    elif path.is_dir():
        files = sorted(p for p in path.iterdir()
                       if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS)
        if not files:
            click.echo(f"  ⚠ Aucun fichier texte trouvé dans {path}", err=True)
        for f in files:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                results.append((f.name, content))
            except Exception as e:
                click.echo(f"  ⚠ {f.name} : {e}", err=True)
    else:
        click.echo(f"  ⚠ Chemin introuvable : {path}", err=True)

    return results


def _split_lines(text: str, min_line_len: int = 10) -> list[str]:
    """
    Retourne les lignes non vides d'un texte.
    Chaque ligne est traitée comme une unité d'analyse.
    min_line_len remplace l'ancien magic number 10 (défaut rétrocompat).
    """
    return [l.strip() for l in text.splitlines() if len(l.strip()) >= min_line_len]


def extract_concepts_from_cir(cir: dict) -> list[str]:
    """Extrait des concepts (lemmes) depuis cir['nodes'] via la regex bridge.

    Réutilise frontend.bridge._LABEL_RE (^([^(?\\s]+), avec ? littéral).
    Strip ?.,!(), longueur ≥ 3.
    """
    try:
        from .frontend.bridge import _LABEL_RE
    except Exception:
        import re as _re
        _LABEL_RE = _re.compile(r'^([^(?\s]+)')
    concepts = []
    for n in cir.get("nodes", []) or []:
        label = (n.get("label") or "").strip()
        if not label:
            continue
        m = _LABEL_RE.match(label)
        lemma = (m.group(1).strip() if m else label).strip("?.,!()")
        if len(lemma) >= 3 and lemma.lower() not in concepts:
            concepts.append(lemma.lower())
    return concepts


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


def _write_log(log_path: Optional[Path], entry: dict) -> None:
    """Ajoute une entrée au fichier de log JSON (une entrée par ligne)."""
    if log_path is None:
        return
    import json, datetime
    entry["ts"] = datetime.datetime.utcnow().isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_discuss(
    checkpoint: Optional[Path] = None,
    graph_path: Optional[Path] = None,
    gcn_bin: str = "gcn",
    log_path: Optional[Path] = None,
    session_dir: Optional[Path] = None,
) -> None:
    """Lance la session de discussion."""
    from .engine import GCNEngine
    from .cli.session import SessionStore

    # Session persistante : graphe + vecs + historique (anti-perte)
    session = SessionStore(session_dir)
    restored = {"graph_edges": 0, "vecs": 0, "history": 0}
    if session_dir is not None:
        restored = session.load()
        if restored["graph_edges"] or restored["vecs"]:
            print(f"\n  Session restaurée : {restored['graph_edges']} relation(s), "
                  f"{restored['vecs']} vecteur(s), {restored['history']} événement(s).")

    # Charger le moteur
    engine = None
    if checkpoint and checkpoint.exists():
        try:
            engine = GCNEngine.from_pretrained(checkpoint, gcn_bin=gcn_bin, trusted=True)
            engine._pipeline.encoder.training = False
        except Exception as e:
            print(f"\n  Erreur de chargement du checkpoint : {e}")
            print("  Impossible d'analyser sans checkpoint.")

    # Charger ou créer le graphe de session
    handler = InstructionHandler()
    if restored["graph_edges"]:
        handler.graph = session.graph
    decoder = ReferenceDecoder()   # CIR → texte, flux direct sans fichier
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
            if session.session_dir is not None:
                session.graph = handler.graph
                session.save()
                print(f"  Session sauvegardée : {session.session_dir}")
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
                if session.session_dir is not None:
                    session.graph = handler.graph
                    session.save()
                    print(f"  Session sauvegardée : {session.session_dir}")
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
                                # Collecte session : vecteurs persistés (anti-perte)
                                if session.session_dir is not None:
                                    session.collect(engine, line, cir)
                                # CIR → verbalizer directement, sans fichier
                                verbalized = decoder.decode_cir(cir)
                                if verbalized:
                                    print(verbalized)
                                # Aussi stocké dans le graphe pour les requêtes
                                handler.add_cir(cir)
                                n_new += len(cir["edges"])
                        except Exception as _e:
                            import warnings as _dw
                            _dw.warn(
                                f"discuss: erreur analyse/verbalisation CIR : {_e}",
                                UserWarning, stacklevel=2,
                            )
                    total_new += n_new
                    print(f"  {filename} : {n_new} relation(s) extraite(s)")
                    _write_log(log_path, {
                        "event": "analyze",
                        "file": filename,
                        "relations_extracted": n_new,
                        "session_total": len(handler.graph.edges),
                    })
                n_total = len(handler.graph.edges)
                print(f"  Total session : {n_total} relation(s)")
                print()
                if session.session_dir is not None:
                    session.graph = handler.graph
                    session.save()
                    print(f"  Session sauvegardée : {session.session_dir}")
                    if n_total == 0:
                        import sys as _sys
                        print(
                            "  ATTENTION : graphe vide persisté (0 relation) — "
                            "aucune arête causale extraite dans cette session.",
                            file=_sys.stderr,
                        )

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
            # Fallback entrée libre : si rien trouvé et moteur dispo, analyser la
            # question elle-même (Python→Rust sens unique) puis réessayer une fois.
            if "No causal structure" in response and engine is not None:
                try:
                    cir = engine.analyze(user_input)
                    concepts = extract_concepts_from_cir(cir)
                    if cir.get("edges"):
                        # Répondre depuis la question uniquement — ne pas modifier le graphe corpus.
                        tmp_handler = InstructionHandler()
                        tmp_handler.add_cir(cir)
                        tmp_response = _format_response(tmp_handler, user_input)
                        if "No causal structure" not in tmp_response:
                            response = tmp_response
                        if session.session_dir is not None:
                            session.collect(engine, user_input, cir)
                    elif concepts:
                        response += f"\n  (concepts détectés dans la question : {', '.join(concepts)} — source=question, non corpus)"
                except Exception as _e:
                    response += f"\n  (fallback analyze impossible : {_e})"
            print(response)
            print()
            _answered = "No causal structure" not in response
            if session.session_dir is not None:
                session.graph = handler.graph
                session.record_exchange(user_input, _answered)
                session.save()
            _write_log(log_path, {
                "event": "query",
                "question": user_input,
                "session_relations": len(handler.graph.edges),
                "answered": _answered,
            })


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
@click.option("--log", "log_path", default=None, type=click.Path(path_type=Path),
              help="Fichier de log JSON pour le monitoring (optionnel).")
@click.option("--session-dir", default=None, type=click.Path(path_type=Path),
              help="Répertoire de session persistante (graphe + vecs + historique). "
                   "Restauré au démarrage, sauvegardé à chaque analyse/question et à la sortie.")
def discuss_cmd(
    ckpt: Optional[Path],
    graph_path: Optional[Path],
    gcn_bin: str,
    log_path: Optional[Path],
    session_dir: Optional[Path],
) -> None:
    """Session de discussion causale sur corpus — /analyze, questions libres, /save."""
    run_discuss(checkpoint=ckpt, graph_path=graph_path, gcn_bin=gcn_bin,
               log_path=log_path, session_dir=session_dir)
