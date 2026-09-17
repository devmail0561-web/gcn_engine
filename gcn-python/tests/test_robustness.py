"""
Tests de robustesse prod — valident que le modèle discrimine les connecteurs
et la direction des arêtes. Nécessitent un checkpoint entraîné sur données réelles.
Sautés si le checkpoint n'existe pas.
"""
import pytest
import numpy as np
from pathlib import Path
from gcn_python.layer1.representation import UDRepresentation
from gcn_python.layer1.features import FeatureVocabulary, vectorize_edge, CONNECTOR_LEMMAS

CHECKPOINT = Path("checkpoints/prod_v1.npz")


def _make_rep(lemma, pos="VERB", dep_rel="root", morph=None) -> UDRepresentation:
    return UDRepresentation(
        tokens=[{"lemma": lemma, "pos": pos, "dep_rel": dep_rel, "morph": morph or {}}],
        root_lemma=lemma, root_pos=pos, root_dep_rel=dep_rel,
        root_morph=morph or {}, subject_pos=None,
        has_object=False, has_advcl=False, has_temporal_obl=False,
        token_span=(1, 1),
    )


def _make_connector(lemma, pos="SCONJ") -> UDRepresentation:
    return _make_rep(lemma, pos=pos, dep_rel="mark")


# ---------------------------------------------------------------------------
# Test 1 — Le connecteur change le vecteur edge
# ---------------------------------------------------------------------------

def test_connecteur_si_vs_bien_que_change_vecteur():
    """
    Avec un lexique de connecteurs fourni, deux connecteurs différents
    produisent des vecteurs edge différents.
    Sans lexique (défaut language-agnostic), les lemmes ne sont pas encodés.
    """
    # Fournir le lexique FR — la différence "si" vs "bien" est encodée
    vocab = FeatureVocabulary(connector_lemmas=CONNECTOR_LEMMAS)
    src = _make_rep("baisser", morph={"Tense": "Pres"})
    dst = _make_rep("réduire", morph={"Tense": "Pres"})
    conn_si   = _make_connector("si")
    conn_bien = _make_connector("bien")

    vec_si   = vectorize_edge(src, dst, conn_si,   1, 2, 3, vocab)
    vec_bien = vectorize_edge(src, dst, conn_bien,  1, 2, 3, vocab)

    assert not np.array_equal(vec_si, vec_bien), (
        "Vecteurs identiques malgré des connecteurs différents — "
        "le connecteur n'est pas encodé dans vectorize_edge."
    )


# ---------------------------------------------------------------------------
# Test 2 — L'inversion src/dst change le vecteur edge
# ---------------------------------------------------------------------------

def test_inversion_src_dst_change_vecteur():
    """vectorize_edge(A, B) ≠ vectorize_edge(B, A) — direction encodée."""
    vocab = FeatureVocabulary()
    rep_a = _make_rep("baisser")
    rep_b = _make_rep("réduire")
    conn  = _make_connector("parce")

    vec_ab = vectorize_edge(rep_a, rep_b, conn, 1, 2, 3, vocab)
    vec_ba = vectorize_edge(rep_b, rep_a, conn, 2, 1, 3, vocab)

    assert not np.array_equal(vec_ab, vec_ba), (
        "Vecteurs A→B et B→A identiques — direction de l'arête non encodée."
    )


# ---------------------------------------------------------------------------
# Test 3 — Le modèle prédit "concession" pour "bien que" (si checkpoint dispo)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Checkpoint prod_v1.npz absent")
def test_bien_que_predit_concession():
    """
    Avec checkpoint prod, le connecteur "bien que" doit déclencher "concession",
    pas "cause" (classe dominante à 60% dans le dataset réel).
    """
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import load_checkpoint
    from gcn_python.constants import NODE_TYPES

    vocab = FeatureVocabulary()
    d_eff  = vocab.d_clause
    d_edge = vocab.d_edge_closed_loop(d_eff, len(NODE_TYPES))
    encoder  = MLPEncoder(d_clause=d_eff, d_edge=d_edge)
    graph    = RGCNLayer(d_in=d_eff, d_out=d_eff)
    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)
    load_checkpoint(pipeline, CHECKPOINT)
    pipeline.encoder.training = False

    src  = _make_rep("baisser", morph={"Tense": "Pres"})
    dst  = _make_rep("réduire", morph={"Tense": "Pres"})
    conn = _make_connector("bien")

    cir = pipeline.forward([src, dst], "Les ventes baissent bien qu'on réduise les coûts.",
                           connector_reps=[conn])
    assert cir["edges"], "Aucune arête produite par le forward"
    predicted = cir["edges"][0][2]["relation"]
    assert predicted == "concession", (
        f"Attendu 'concession' pour le connecteur 'bien que', obtenu '{predicted}'. "
        f"Le modèle est probablement biaisé vers 'cause' — entraîner davantage ou "
        f"augmenter les données de concession."
    )


# ---------------------------------------------------------------------------
# Test 4 — Phrase hors-template (lemmes non vus à l'entraînement)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Checkpoint prod_v1.npz absent")
def test_phrase_hors_template_produit_cir_valide():
    """
    Des lemmes hors-vocabulaire d'entraînement doivent quand même produire
    un CIR structurellement valide (2 nœuds, 1 arête, types connus).
    Prouve que le modèle généralise via les features syntaxiques, pas les lemmes.
    """
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import load_checkpoint
    from gcn_python.constants import NODE_TYPES

    vocab = FeatureVocabulary()
    d_eff  = vocab.d_clause
    d_edge = vocab.d_edge_closed_loop(d_eff, len(NODE_TYPES))
    encoder  = MLPEncoder(d_clause=d_eff, d_edge=d_edge)
    graph    = RGCNLayer(d_in=d_eff, d_out=d_eff)
    pipeline = CGNPipeline(encoder=encoder, graph=graph, vocabulary=vocab)
    load_checkpoint(pipeline, CHECKPOINT)
    pipeline.encoder.training = False

    src  = _make_rep("désinhiber", pos="VERB", morph={"Tense": "Pres"})
    dst  = _make_rep("exacerber",  pos="VERB", morph={"Tense": "Fut"})
    conn = _make_connector("parce")

    cir = pipeline.forward([src, dst], "...", connector_reps=[conn])
    assert len(cir["nodes"]) == 2, f"Attendu 2 nœuds, obtenu {len(cir['nodes'])}"
    assert len(cir["edges"]) == 1, f"Attendu 1 arête, obtenu {len(cir['edges'])}"
    for node in cir["nodes"]:
        assert node["node_type"] in NODE_TYPES, (
            f"Type de nœud inconnu en production : {node['node_type']!r}"
        )
