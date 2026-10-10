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

from pathlib import Path

import click

from .dialogue import DialogueSession
from .dialogue import format_response as _format_response  # re-export (compat)
from .pipeline.nlu_routing import nlu_route
from .verbalizer.decoder import ReferenceDecoder
from .verbalizer.instructions import CausalGraph, InstructionHandler, parse_command
from .verbalizer.query_report import QueryVerbalizer

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
            except OSError as e:  # read_text(errors="replace") : seul un échec d'E/S est possible
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
            except OSError as e:  # read_text(errors="replace") : seul un échec d'E/S est possible
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
    return [line.strip() for line in text.splitlines() if len(line.strip()) >= min_line_len]


def extract_concepts_from_cir(cir: dict) -> list[str]:
    """Extrait des concepts (lemmes) depuis cir['nodes'] via la regex bridge.

    Réutilise frontend.bridge._LABEL_RE (^([^(?\\s]+), avec ? littéral).
    Strip ?.,!(), longueur ≥ 3.
    """
    try:
        from .frontend.bridge import _LABEL_RE
    except Exception:  # noqa: BLE001  # repli sur regex locale si l'import interne échoue
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
# Formatage des rapports de réponse : voir dialogue.format_response
# (re-exporté ci-dessus pour compatibilité). L'heuristique « dernier mot »
# y est conservée en repli ; DialogueSession fournit le sujet résolu (9b).
# ---------------------------------------------------------------------------


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


def _write_log(log_path: Path | None, entry: dict) -> None:
    """Ajoute une entrée au fichier de log JSON (une entrée par ligne)."""
    if log_path is None:
        return
    import datetime
    import json
    entry["ts"] = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_discuss(
    checkpoint: Path | None = None,
    graph_path: Path | None = None,
    gcn_bin: str = "gcn",
    log_path: Path | None = None,
    session_dir: Path | None = None,
    vecs_index: Path | None = None,
) -> None:
    """Lance la session de discussion."""
    from .cli.session import SessionStore
    from .data.graph_vecs import load_graph_vecs_index
    from .engine import GCNEngine

    # Session persistante : graphe + vecs + historique (anti-perte)
    session = SessionStore(session_dir)
    # F4 : compteur de blocs session pour éviter les collisions d'ids inter-phrases
    _sess_blk = [0]
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
            engine = GCNEngine.from_pretrained(
                checkpoint, gcn_bin=gcn_bin, trusted=True
            )
            engine._pipeline.encoder.training = False
        except Exception as e:  # noqa: BLE001  # checkpoint illisible : message d'erreur CLI, sans moteur
            print(f"\n  Erreur de chargement du checkpoint : {e}")
            print("  Impossible d'analyser sans checkpoint.")

    # Index de vecteurs de graphe pour résolution sémantique (--vecs-index)
    _graph_vec_index: list = []
    if vecs_index is not None and vecs_index.exists():
        _graph_vec_index = load_graph_vecs_index(vecs_index)
        if _graph_vec_index:
            print(f"\n  Index sémantique chargé : {len(_graph_vec_index)} nœuds vectorisés")

    # Charger ou créer le graphe de session
    handler = InstructionHandler()
    if restored["graph_edges"]:
        handler.graph = session.graph
    decoder = ReferenceDecoder()   # CIR → texte, flux direct sans fichier
    if graph_path and graph_path.exists():
        try:
            handler.graph = CausalGraph.load(graph_path)
            print(f"\n  Graphe chargé : {graph_path.name}")
        except Exception as e:  # noqa: BLE001  # graphe JSON invalide : message d'erreur CLI, graphe conservé
            print(f"\n  Erreur de chargement du graphe : {e}")
    # F4c : reprendre la numérotation après les blocs déjà présents (session
    # restaurée ou --graph préchargé) — sinon d00001_ est réutilisé et les
    # nœuds d'une session précédente sont silencieusement écrasés.
    import re as _blk_re
    for _nid in handler.graph.nodes:
        _m = _blk_re.match(r"^d(\d+)_", str(_nid))
        if _m:
            _sess_blk[0] = max(_sess_blk[0], int(_m.group(1)))

    # État de dialogue 9b-e (sujets, clarification, langue, journal).
    # Reconstruit depuis l'historique persisté si la session est restaurée.
    dialogue = DialogueSession()
    # q-blocks 9a : même reseed que F4c — sinon q00001_ réutilisé, nœuds
    # question écrasés + arêtes dupliquées après restauration.
    dialogue.reseed_qblocks(handler.graph.nodes)
    if restored["history"]:
        dialogue.from_history(session.history)

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
                                # F4 : préfixe par bloc (is not None : id=0 est falsy)
                                _sess_blk[0] += 1
                                _pfx = f"d{_sess_blk[0]:05d}_"
                                for _n in cir.get("nodes", []) or []:
                                    if isinstance(_n, dict) and _n.get("id") is not None:
                                        _n["id"] = _pfx + str(_n["id"])
                                _re = []
                                for _e in cir.get("edges", []) or []:
                                    if isinstance(_e, (list, tuple)) and len(_e) == 3:
                                        _re.append([_pfx + str(_e[0]), _pfx + str(_e[1]), _e[2]])
                                    elif isinstance(_e, dict):
                                        _e = dict(_e)
                                        if _e.get("source") is not None:
                                            _e["source"] = _pfx + str(_e["source"])
                                        if _e.get("target") is not None:
                                            _e["target"] = _pfx + str(_e["target"])
                                        _re.append(_e)
                                cir["edges"] = _re
                                # Aussi stocké dans le graphe pour les requêtes
                                # (origine 9a : document — texte analysé du corpus)
                                handler.add_cir(cir, session_origin="document")
                                n_new += len(cir["edges"])
                        except Exception as _e:  # noqa: BLE001  # phrase invalide : warn + phrase suivante
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
                except Exception as e:  # noqa: BLE001  # sauvegarde impossible : message d'erreur CLI
                    print(f"  Erreur de sauvegarde : {e}")

            elif cmd == "load":
                if not arg:
                    print("  Usage : /load <path.json>")
                    continue
                try:
                    handler.graph = CausalGraph.load(Path(arg))
                    n = len(handler.graph.edges)
                    print(f"  Graphe chargé : {arg}  ({n} relations)")
                except Exception as e:  # noqa: BLE001  # chargement impossible : message d'erreur CLI
                    print(f"  Erreur de chargement : {e}")

            else:
                print(f"  Commande inconnue : /{cmd}   (tapez /help)")

        # ── Question sur le corpus ───────────────────────────────────────
        else:
            if not handler.graph.edges:
                print("  Corpus vide. Utilisez /analyze <fichier> pour charger des documents.")
                continue
            print()
            # Tour de dialogue 9b-e : résolution sujet, clarification, langue,
            # enrichissement question (9a), routage NLU, journal.
            response, info = dialogue.handle(
                user_input, handler, engine=engine,
                vec_index=_graph_vec_index or None,
            )
            # Collecte session : vecteurs de la question (anti-perte, best-effort)
            if engine is not None and session.session_dir is not None:
                try:
                    _qcir = engine.analyze(user_input)
                    if _qcir.get("edges"):
                        session.collect(engine, user_input, _qcir)
                except Exception:  # noqa: S110, BLE001
                    pass
            print(response)
            print()
            _answered = "No causal structure" not in response
            if session.session_dir is not None:
                session.graph = handler.graph
                session.record_exchange(
                    user_input, _answered, turn=info["n"], subject=info["subject"],
                    summary=info["summary"], status=info["status"],
                    lang=info["lang"], added=info["added"],
                )
                if info.get("summarized") is not None:
                    session.record_summary(info["summarized"])
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
@click.option("--vecs-index", "vecs_index", default=None, type=click.Path(path_type=Path),
              help="Fichier graph_vecs.npz produit par gcn-index --vecs-out. "
                   "Active la résolution sémantique des requêtes : le concept de la question "
                   "est résolu vers le nœud du graphe le plus proche (espace R-GCN).")
def discuss_cmd(
    ckpt: Path | None,
    graph_path: Path | None,
    gcn_bin: str,
    log_path: Path | None,
    session_dir: Path | None,
    vecs_index: Path | None,
) -> None:
    """Session de discussion causale sur corpus — /analyze, questions libres, /save."""
    run_discuss(checkpoint=ckpt, graph_path=graph_path, gcn_bin=gcn_bin,
               log_path=log_path, session_dir=session_dir,
               vecs_index=vecs_index)
