"""
gcn-chat — Interface conversationnelle du moteur GCN Causal Engine.

Permet d'analyser du texte et d'interroger la structure causale extraite
dans une session interactive persistante, comme un LLM.

Usage :
    gcn-chat --checkpoint model.npz

Session :
    > Les ventes baissent car la demande recule.
    > Qu'est-ce qui cause la baisse des ventes ?
    > Et si la demande augmentait ?
    > clear    — vider la session
    > quit     — quitter
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Formattage de la réponse CIR → texte lisible
# ---------------------------------------------------------------------------

def _format_cir(cir: dict) -> str:
    """Produit un résumé lisible d'un CIR."""
    nodes = cir.get("nodes", [])
    edges = cir.get("edges", [])

    if not nodes:
        return "  (aucune structure causale détectée)"

    lines = []

    # Nœuds
    node_map = {}
    for n in nodes:
        nid  = n.get("id") if isinstance(n, dict) else n
        ntype = n.get("node_type", "?") if isinstance(n, dict) else "?"
        label = n.get("label", "") if isinstance(n, dict) else ""
        node_map[nid] = (ntype, label)
        lines.append(f"  [{ntype}] {label}")

    # Arêtes
    if edges:
        lines.append("")
        for e in edges:
            if isinstance(e, (list, tuple)) and len(e) == 3:
                src_id, dst_id, attrs = e
                relation   = attrs.get("relation", "?")
                confidence = attrs.get("confidence", 0.0)
                negated    = attrs.get("negated", False)
                src_lbl = node_map.get(src_id, ("?", str(src_id)))[1] or str(src_id)
                dst_lbl = node_map.get(dst_id, ("?", str(dst_id)))[1] or str(dst_id)
                neg_mark = " [NÉGATION]" if negated else ""
                lines.append(
                    f"  {src_lbl}  --{relation}--> {dst_lbl}"
                    f"  (conf={confidence:.2f}){neg_mark}"
                )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Réponses aux questions sur le CIR accumulé
# ---------------------------------------------------------------------------

_CAUSE_PATTERN    = re.compile(r"(cause|provoque|entraîne|ce qui cause|pourquoi)\b", re.I)
_EFFECT_PATTERN   = re.compile(r"(effet|conséquence|résultat|qu.*arrive|qu.*se passe)\b", re.I)
_RELATION_PATTERN = re.compile(r"(relation|lien|connexion|comment .* lié)\b", re.I)
_LIST_PATTERN     = re.compile(r"(liste|montre|affiche|rappelle|résume)\b", re.I)
_CLEAR_PATTERN    = re.compile(r"^(clear|reset|vider|recommencer|nouveau)\s*$", re.I)
_QUIT_PATTERN     = re.compile(r"^(quit|exit|bye|au revoir|q)\s*$", re.I)
_HELP_PATTERN     = re.compile(r"^(aide|help|\?)\s*$", re.I)


def _answer_question(question: str, session_cirs: list[dict]) -> str:
    """Répond à une question en langage naturel sur les CIR accumulés."""

    all_edges = []
    all_nodes = {}
    for cir in session_cirs:
        for n in cir.get("nodes", []):
            if isinstance(n, dict):
                all_nodes[n.get("id")] = n
        for e in cir.get("edges", []):
            if isinstance(e, (list, tuple)) and len(e) == 3:
                all_edges.append((cir.get("source_text", ""), e))

    if not all_edges:
        return "Je n'ai pas encore analysé de relation causale dans cette session."

    q = question.lower()

    if _LIST_PATTERN.search(q):
        lines = ["Relations causales identifiées dans cette session :"]
        for text, (src, dst, attrs) in all_edges:
            src_n = all_nodes.get(src, {})
            dst_n = all_nodes.get(dst, {})
            lines.append(
                f"  [{src_n.get('node_type','?')}] {src_n.get('label',src)}"
                f"  --{attrs.get('relation','?')}--> "
                f"  [{dst_n.get('node_type','?')}] {dst_n.get('label',dst)}"
                f"  | \"{text[:60]}...\""
            )
        return "\n".join(lines)

    if _CAUSE_PATTERN.search(q):
        lines = ["Causes identifiées :"]
        found = False
        for _, (src, dst, attrs) in all_edges:
            if attrs.get("relation") in {"cause", "enable", "motivation"}:
                src_n = all_nodes.get(src, {})
                dst_n = all_nodes.get(dst, {})
                lines.append(
                    f"  {src_n.get('label', src)}"
                    f"  cause/permet  {dst_n.get('label', dst)}"
                    f"  (relation: {attrs.get('relation')})"
                )
                found = True
        if not found:
            return "Aucune relation de cause/enable/motivation détectée dans la session."
        return "\n".join(lines)

    if _EFFECT_PATTERN.search(q):
        lines = ["Effets / conséquences identifiés :"]
        found = False
        for _, (src, dst, attrs) in all_edges:
            src_n = all_nodes.get(src, {})
            dst_n = all_nodes.get(dst, {})
            lines.append(
                f"  {src_n.get('label', src)}"
                f"  --> [{attrs.get('relation')}] --> "
                f"  {dst_n.get('label', dst)}"
            )
            found = True
        if not found:
            return "Aucune conséquence détectée."
        return "\n".join(lines)

    # Question générale → résumé de la session
    n_phrases = len(session_cirs)
    n_rel     = len(all_edges)
    from collections import Counter
    rel_counts = Counter(e[1][2].get("relation", "?") for e in all_edges)
    top = rel_counts.most_common(3)
    summary = (
        f"Session actuelle : {n_phrases} phrase(s) analysée(s), "
        f"{n_rel} relation(s) causale(s).\n"
        f"Relations dominantes : {', '.join(f'{r}({c})' for r,c in top)}.\n"
        f"Tapez 'liste' pour voir toutes les relations, ou analysez plus de texte."
    )
    return summary


# ---------------------------------------------------------------------------
# Session principale
# ---------------------------------------------------------------------------

def _print_welcome():
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║       GCN Causal Engine — Session interactive        ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()
    print("  Entrez du texte pour l'analyser, ou posez une question")
    print("  sur les relations causales identifiées.")
    print()
    print("  Commandes : clear | quit | aide")
    print()


def _is_question(text: str) -> bool:
    """Détermine si l'entrée est une question ou du texte à analyser."""
    t = text.strip()
    if t.endswith("?"):
        return True
    q_words = {"qu'", "que ", "quoi", "qui ", "comment", "pourquoi",
               "cause", "effet", "relation", "liste", "affiche", "montre",
               "résume", "rappelle", "what", "why", "how", "show", "list"}
    t_lower = t.lower()
    return any(t_lower.startswith(w) or (" " + w) in t_lower for w in q_words)


import click

@click.command("gcn-chat")
@click.option("--checkpoint", "ckpt", default=None, type=click.Path(path_type=Path),
              help="Checkpoint .npz produit par gcn-train.")
@click.option("--gcn-bin", default="gcn", show_default=True,
              help="Chemin vers le binaire gcn-cli Rust.")
def _chat_cmd(ckpt: Optional[Path], gcn_bin: str) -> None:
    """Session interactive — analysez du texte et interrogez la structure causale."""
    run_chat(checkpoint=ckpt, gcn_bin=gcn_bin)


def run_chat(checkpoint: Optional[Path] = None, gcn_bin: str = "gcn") -> None:
    """Lance la session interactive."""
    from .engine import GCNEngine

    _print_welcome()

    if checkpoint is None or not checkpoint.exists():
        print("  ⚠  Aucun checkpoint — poids aléatoires (résultats non significatifs).")
        print("     Lancez avec : gcn-chat --checkpoint model.npz")
        print()

    try:
        if checkpoint and checkpoint.exists():
            print(f"  Chargement du modèle depuis {checkpoint.name}...", end="", flush=True)
            engine = GCNEngine.from_pretrained(checkpoint, gcn_bin=gcn_bin)
            print(" prêt.\n")
        else:
            engine = None
    except Exception as e:
        print(f"\n  Erreur de chargement : {e}")
        engine = None

    session_cirs: list[dict] = []
    history: list[tuple[str, str]] = []   # (rôle, texte)

    while True:
        try:
            user_input = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Au revoir.")
            break

        if not user_input:
            continue

        history.append(("user", user_input))

        # Commandes spéciales
        if _QUIT_PATTERN.match(user_input):
            print("  Au revoir.")
            break

        if _CLEAR_PATTERN.match(user_input):
            session_cirs.clear()
            history.clear()
            print("  Session vidée.\n")
            continue

        if _HELP_PATTERN.match(user_input):
            print()
            print("  Entrez du texte → le moteur extrait la structure causale.")
            print("  Posez une question → le moteur répond sur les CIR de la session.")
            print()
            print("  Exemples de texte à analyser :")
            print('    "Les ventes baissent car la demande recule."')
            print('    "Si les coûts augmentent, les marges s\'effondrent."')
            print()
            print("  Exemples de questions :")
            print('    "Qu\'est-ce qui cause la baisse ?"')
            print('    "Liste toutes les relations."')
            print('    "Quels sont les effets identifiés ?"')
            print()
            print("  Commandes : clear  quit  aide")
            print()
            continue

        # Question sur la session
        if _is_question(user_input):
            response = _answer_question(user_input, session_cirs)
            print()
            print(response)
            print()
            history.append(("assistant", response))
            continue

        # Texte à analyser
        if engine is None:
            print("  (Pas de modèle chargé — impossible d'analyser.)")
            continue

        sentences = GCNEngine._split_sentences(user_input)
        print()
        for sent in sentences:
            try:
                cir = engine.analyze(sent)
                session_cirs.append(cir)
                has_edges = bool(cir.get("edges"))
                print(f"  Phrase : \"{sent[:80]}{'...' if len(sent) > 80 else ''}\"")
                print(_format_cir(cir))
                if not has_edges:
                    print("  (aucune relation causale détectée)")
            except Exception as e:
                print(f"  Erreur d'analyse : {e}")
            print()

        n_total = sum(len(c.get("edges", [])) for c in session_cirs)
        history.append(("assistant", f"{len(sentences)} phrase(s) analysée(s), "
                                     f"{n_total} relation(s) au total dans la session."))
