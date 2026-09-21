# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
gcn-index — indexation batch d'un corpus → graphe causal persistant.

Analyse tous les fichiers texte d'un répertoire (ou un fichier unique)
et construit un graphe causal JSON sans interaction utilisateur.
Utile pour pré-indexer un grand corpus avant gcn-discuss.

Usage :
    gcn-index --corpus rapports/ --checkpoint model.npz --output graph.json
    gcn-discuss --checkpoint model.npz --graph graph.json
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import click

from .discuss import _read_texts, _split_lines
from .verbalizer.instructions import CausalGraph

log = logging.getLogger(__name__)

# Segmentation externe naïve : fin .!? suivie d'espace ou fin de ligne.
# Limitation documentée : casse "M.", "e.g.", "..." (segmentation externe
# injectable à l'avenir, pas de hardcoding linguistique supplémentaire).
_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def segment_sentences(line: str) -> list[str]:
    parts = [p.strip() for p in _SENT_RE.split(line or "") if p.strip()]
    return parts or ([line.strip()] if line.strip() else [])


@click.command("gcn-index")
@click.option("--corpus", required=True, type=click.Path(path_type=Path),
              help="Fichier texte ou répertoire à indexer.")
@click.option("--checkpoint", required=True, type=click.Path(path_type=Path),
              help="Checkpoint .npz produit par gcn-train.")
@click.option("--output", required=True, type=click.Path(path_type=Path),
              help="Fichier JSON de sortie pour le graphe causal.")
@click.option("--gcn-bin", default="gcn", show_default=True,
              help="Chemin vers le binaire gcn-cli Rust.")
@click.option("--append/--overwrite", default=False, show_default=True,
              help="Enrichir un graphe existant (--append) ou le remplacer.")
@click.option("--min-line-len", default=10, show_default=True, type=int,
              help="Longueur minimale de ligne (remplace le magic number 10).")
@click.option("--vecs-out", default=None, type=click.Path(path_type=Path),
              help="Fichier .npz de sortie pour les vecteurs enrichis (+ manifest JSON). "
                   "Anti-perte : les vecteurs calculés pendant l'indexation sont persistés "
                   "par clé stable et réutilisables en session.")
@click.option("--taxonomy-dir", default=None, type=click.Path(path_type=Path),
              help="Répertoire des taxonomies causales (transmis à gcn-cli --data-dir). "
                   "Parité avec gcn-bootstrap.")
def index_cmd(
    corpus: Path,
    checkpoint: Path,
    output: Path,
    gcn_bin: str,
    append: bool,
    min_line_len: int,
    vecs_out: Optional[Path],
    taxonomy_dir: Optional[Path],
) -> None:
    """Indexe un corpus de fichiers texte → graphe causal JSON."""
    from .engine import GCNEngine
    from .data.graph_vecs import save_graph_vecs, stable_key, checkpoint_hash as _ckpt_hash

    import numpy as _np

    if not corpus.exists():
        raise click.ClickException(f"Corpus introuvable : {corpus}")
    if not checkpoint.exists():
        raise click.ClickException(f"Checkpoint introuvable : {checkpoint}")

    # Charger le moteur
    click.echo(f"Chargement du modèle depuis {checkpoint.name}...")
    try:
        engine = GCNEngine.from_pretrained(
            checkpoint, gcn_bin=gcn_bin, trusted=True, taxonomy_dir=taxonomy_dir
        )
    except Exception as exc:
        raise click.ClickException(f"Erreur de chargement du checkpoint : {exc}") from exc
    engine._pipeline.encoder.training = False

    # Charger ou créer le graphe
    graph = CausalGraph()
    if append and output.exists():
        try:
            graph = CausalGraph.load(output)
            click.echo(f"Graphe existant chargé : {len(graph.edges)} relations")
        except Exception as exc:
            raise click.ClickException(f"Graphe existant illisible : {exc}") from exc

    # Lire les fichiers
    texts = _read_texts(corpus)
    if not texts:
        raise click.ClickException("Aucun fichier texte trouvé.")

    click.echo(f"Indexation de {len(texts)} fichier(s)...")
    total = 0
    collected: dict[str, object] = {}
    collected_meta: dict[str, dict] = {}
    # F4 : préfixe par bloc (sentence avec arêtes), pas par fichier — évite les
    # collisions d'ids quand deux phrases du même fichier produisent les mêmes ids.
    block_idx = 0
    for file_idx, (filename, content) in enumerate(texts, 1):
        lines = _split_lines(content, min_line_len=min_line_len)
        n_new = 0
        for line in lines:
            # Séquence correcte : segmenter en phrases, un analyze() par phrase,
            # puis bloc de discours avec ids préfixés (bBBBBB_nMMM).
            for sent in segment_sentences(line):
                try:
                    cir = engine.analyze(sent)
                except Exception as exc:
                    log.warning("index: analyze impossible (%s) : %s", filename, exc)
                    continue
                if cir.get("edges"):
                    block_idx += 1
                    # Collecte vecs AVANT préfixage (positions = ordre cir["nodes"])
                    if vecs_out is not None:
                        try:
                            _ev = engine._pipeline.get_enriched_vectors()
                            if _ev is not None:
                                for _pos, _node in enumerate(cir.get("nodes", []) or []):
                                    if _pos >= len(_ev):
                                        break
                                    _key = stable_key(sent, _pos)
                                    collected[_key] = _np.asarray(_ev[_pos], dtype=_np.float32)
                                    collected_meta[_key] = {
                                        "node_id": str(_node.get("id", _pos)),
                                        "node_label": str(_node.get("label", "")),
                                        "shape": [int(v) for v in _np.asarray(_ev[_pos]).shape],
                                        "source_text": sent[:200],
                                    }
                        except Exception as exc:
                            log.warning("index: vecs non collectés (%s) : %s", filename, exc)
                    prefix = f"b{block_idx:05d}_"
                    for n in cir.get("nodes", []):
                        # is not None : l'id 0 (entier) est falsy — n.get("id") le sautait
                        if isinstance(n, dict) and n.get("id") is not None:
                            n["id"] = prefix + str(n["id"])
                    remapped = []
                    for e in cir.get("edges", []):
                        if isinstance(e, (list, tuple)) and len(e) == 3:
                            s, d, a = e
                            remapped.append([prefix + str(s), prefix + str(d), a])
                        elif isinstance(e, dict):
                            e = dict(e)
                            if e.get("source") is not None:
                                e["source"] = prefix + str(e["source"])
                            if e.get("target") is not None:
                                e["target"] = prefix + str(e["target"])
                            remapped.append(e)
                    cir["edges"] = remapped
                    try:
                        graph.add_discourse_block(cir)
                    except Exception as exc:
                        log.warning("index: bloc ignoré (%s) : %s", filename, exc)
                        continue
                    n_new += len(cir["edges"])
        total += n_new
        click.echo(f"  {filename:<40} {n_new:4d} relation(s)")

    graph.save(output)
    click.echo(f"\nGraphe sauvegardé : {output}  ({len(graph.edges)} relations au total)")
    if total == 0:
        click.echo(
            "ATTENTION : aucune relation extraite — graphe vide persisté. "
            "Vérifiez le corpus, le checkpoint et la disponibilité de gcn-bin.",
            err=True,
        )
    if vecs_out is not None and collected:
        try:
            _params = {}
            for _i, _p in enumerate(engine._pipeline.encoder.parameters()):
                _params[f"encoder_{_i}"] = _np.asarray(_p)
            _d = int(next(iter(collected.values())).shape[-1])
            save_graph_vecs(vecs_out, collected, _ckpt_hash(_params), _d, collected_meta)
            click.echo(f"Vecteurs sauvegardés : {vecs_out}  ({len(collected)} vecteur(s) + manifest)")
        except Exception as exc:
            log.warning("index: sauvegarde vecs impossible : %s", exc)
