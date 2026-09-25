# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""
Tests de robustesse prod — valident que le modèle discrimine les connecteurs
et la direction des arêtes. Nécessitent un checkpoint entraîné sur données réelles.
Sautés si le checkpoint n'existe pas.
"""
from pathlib import Path

import numpy as np
import pytest

from gcn_python.layer1.features import FeatureVocabulary, vectorize_edge
from gcn_python.layer1.representation import UDRepresentation

# Lemmes de connecteurs FR définis localement — données tests, pas données moteur
CONNECTOR_LEMMAS = [
    "parce", "car", "puisque", "comme", "si", "bien", "quoique",
    "malgré", "pour", "afin", "donc", "alors", "ensuite", "puis",
    "mais", "or", "pourtant", "cependant", "néanmoins",
]

# Checkpoint prod : chemin absolu depuis la racine du dépôt (gcn-datasets/ est
# git-ignoré — le test saute proprement sur un clone vierge).
REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO_ROOT / "gcn-datasets" / "checkpoints" / "prod_v1.npz"

# Lexique de connecteurs du checkpoint prod_v1 (commit bcbb319, 2026-09-17 17:23,
# liste brute de 54 entrées — doublons compris — avant la déduplication de 61836c3).
# Il donne d_conn = 18 + 54 + 11 + 2 = 85 → d_edge = 247 →
# d_edge_closed_loop(129, 7, 50) = 619, la forme exacte de encoder_6 du checkpoint.
_LEGACY_CONNECTOR_LEMMAS = [
    "parce", "car", "puisque", "comme", "si", "bien", "quoique",
    "quoique", "malgré", "pour", "afin", "donc", "alors", "ensuite",
    "puis", "mais", "or", "pourtant", "cependant", "néanmoins",
    "cependant", "pourvu", "seulement", "lorsque", "dès", "tant",
    "si", "non", "jamais", "ni", "plutôt", "au lieu",
    "en revanche", "en raison", "grâce", "sous", "condition",
    "contrairement", "selon", "à cause", "devant", "chez",
    "vers", "après", "avant", "depuis", "pendant", "durant",
    "chez", "entre", "parmi", "hors", "outre", "faute",
]


def _pipeline_from_checkpoint():
    """Reconstruit le pipeline depuis _arch_json/_vocab_json du checkpoint.

    Miroir de run_eval()/GCNEngine.from_pretrained : construire un pipeline aux
    défauts (d_clause=79) puis charger un checkpoint d_eff=129/d_emb=50 lèverait
    « Incompatibilité de dimension ». L'archive passe d'abord par la garde
    anti-pickle de gcn_python.security.
    """
    import json

    from gcn_python.constants import NODE_TYPES, RELATION_TYPES
    from gcn_python.layer1.embedding import WordEmbedding
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.security import guarded_np_load
    from gcn_python.training.checkpoint import load_checkpoint

    raw = guarded_np_load(CHECKPOINT)
    if "_arch_json" not in raw:
        raise AssertionError(f"{CHECKPOINT.name} sans _arch_json")
    arch = json.loads(str(raw["_arch_json"][0]))
    vocab_data = json.loads(str(raw["_vocab_json"][0])) if "_vocab_json" in raw else {}
    # _vocab_json (2026-09-17) ne contient PAS connector_lemmas : sans le lexique
    # d'entraînement, d_conn = 31 au lieu de 85, d_edge = 565 au lieu de 619, et
    # load_checkpoint lève « Incompatibilité de dimension pour encoder_6 ».
    vocab_data["connector_lemmas"] = list(_LEGACY_CONNECTOR_LEMMAS)
    vocab = FeatureVocabulary(**vocab_data)

    d_eff = int(arch.get("d_eff", vocab.d_clause))
    d_emb = int(arch.get("d_emb", 0))
    d_edge = vocab.d_edge_closed_loop(
        d_eff, len(NODE_TYPES), d_emb, bool(arch.get("subject_object_emb", False))
    )
    encoder = MLPEncoder(
        d_clause=d_eff,
        d_edge=d_edge,
        mlp_hidden=int(arch.get("mlp_hidden", 128)),
    )
    graph = RGCNLayer(
        d_in=d_eff,
        d_out=int(arch.get("d_hidden", d_eff)),
        n_relations=int(arch.get("n_relations", len(RELATION_TYPES))),
    )
    pipeline = CGNPipeline(
        encoder=encoder,
        graph=graph,
        vocabulary=vocab,
        word_embedding=WordEmbedding(d_emb=d_emb) if d_emb > 0 else None,
        bidirectional=bool(arch.get("bidirectional", False)),
        all_pairs=bool(arch.get("all_pairs", False)),
        n_rgcn_layers=int(arch.get("n_rgcn_layers", 1)),
        edge_threshold=float(arch.get("edge_threshold", 0.0)),
        drop_morph=bool(arch.get("drop_morph", False)),
        temperature=float(arch.get("temperature", 1.0)),
        subject_object_emb=bool(arch.get("subject_object_emb", False)),
        bfs_depth=(None if arch.get("bfs_depth") is None else int(arch["bfs_depth"])),
    )
    load_checkpoint(pipeline, CHECKPOINT, trusted=True)
    pipeline.encoder.training = False
    return pipeline


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
@pytest.mark.xfail(
    strict=True,
    reason="prod_v1 prédit 'cause' pour les 50 connecteurs du lexique legacy "
           "(confiance 0.533–0.546) sur ce harnais 2 nœuds : biais de classe "
           "cause ≈ 60 % dans le jeu d'entraînement. Ré-entraînement requis — "
           "le jour où 'bien que' → 'concession', ce test passe et le xfail "
           "strict signale la fin du biais.",
)
def test_bien_que_predit_concession():
    """
    Avec checkpoint prod, le connecteur "bien que" doit déclencher "concession",
    pas "cause" (classe dominante à 60% dans le dataset réel).
    """
    pipeline = _pipeline_from_checkpoint()

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


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Checkpoint prod_v1.npz absent")
def test_lexique_legacy_reellement_utilise():
    """
    prod_v1 a été entraîné avec un lexique de connecteurs de 54 entrées qui n'est
    pas dans _vocab_json. Si on reconstruit mal ce lexique (565 dims au lieu de
    619), le chargement casse ; si on l'ignore au forward, les connecteurs ne
    changent plus rien à la décision. Ici les deux doivent être vrais.
    """
    pipeline = _pipeline_from_checkpoint()

    confs = {}
    for lemma in ("bien", "parce"):
        cir = pipeline.forward(
            [_make_rep("baisser", morph={"Tense": "Pres"}),
             _make_rep("réduire", morph={"Tense": "Pres"})],
            "Les ventes baissent bien qu'on réduise les coûts.",
            connector_reps=[_make_connector(lemma)],
        )
        assert cir["edges"], "Aucune arête produite par le forward"
        confs[lemma] = cir["edges"][0][2]["confidence"]

    assert confs["bien"] != confs["parce"], (
        f"Confiances identiques ({confs['bien']}) quel que soit le connecteur : "
        f"le bloc connector_lemmas du lexique legacy n'atteint pas le MLP d'arête."
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
    from gcn_python.constants import NODE_TYPES

    pipeline = _pipeline_from_checkpoint()

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
