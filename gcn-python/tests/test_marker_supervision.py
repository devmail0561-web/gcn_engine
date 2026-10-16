# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Marqueurs appris, pas chargés — câblage moteur (sans données nouvelles).

- marker_token annoté prime sur la redécouverte syntaxique ;
- third propagé jusqu'au TrainingSample ;
- lemmes de connecteurs au vocabulaire d'embeddings.
"""
import warnings

from gcn_python.data.loader import GCNDataLoader, reps_from_sentence
from gcn_python.data.schema import ClauseRecord, EdgeRecord, SentenceRecord, TokenRecord


def _tok(i, form, lemma, pos, dep_rel, head):
    return TokenRecord(id=i, form=form, lemma=lemma, pos=pos,
                       dep_rel=dep_rel, dep_head=head, morph={})


def _cl(nid, span, ntype="processus"):
    return ClauseRecord(node_id=nid, node_type=ntype, label=nid,
                        token_span=span, scope="specific",
                        temporal_index=0, origin="explicit")


def _rec(tokens, clauses, edges):
    return SentenceRecord(id="s-mk", text="t", tokens=tokens,
                          clauses=clauses, edges=edges)


def _toks_gap_sconj():
    # Clauses [1,2] et [4,5], gap id=3 SCONJ "car".
    return [
        _tok(1, "ventes", "vente", "NOUN", "nsubj", 2),
        _tok(2, "baissent", "baisser", "VERB", "root", 0),
        _tok(3, "car", "car", "SCONJ", "mark", 2),
        _tok(4, "prix", "prix", "NOUN", "nsubj", 5),
        _tok(5, "montent", "monter", "VERB", "advcl", 2),
    ]


def test_gold_marker_priority_over_syntax():
    """marker_token=3 (ADV, pas SCONJ) : le gold gagne, la syntaxe aurait rendu None."""
    toks = _toks_gap_sconj()
    toks[2] = _tok(3, "donc", "donc", "ADV", "advmod", 2)
    rec = _rec(toks, [_cl("n1", (1, 2)), _cl("n2", (4, 5))],
               [EdgeRecord(source="n1", target="n2", relation="cause", marker_token=3)])
    reps, _valid, conns = reps_from_sentence(rec)
    assert len(reps) == 2 and len(conns) == 1
    assert conns[0] is not None, "gold marker ignoré : connector None"
    assert conns[0].root_lemma == "donc", f"gold non utilisé : {conns[0].root_lemma!r}"


def test_gold_marker_missing_falls_back_with_warn():
    """marker_token sans token : warn + redécouverte syntaxique (car)."""
    rec = _rec(_toks_gap_sconj(), [_cl("n1", (1, 2)), _cl("n2", (4, 5))],
               [EdgeRecord(source="n1", target="n2", relation="cause", marker_token=99)])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _, _, conns = reps_from_sentence(rec)
    assert conns[0] is not None and conns[0].root_lemma == "car"
    assert any("marker_token" in str(x.message) for x in w)


def test_no_marker_syntactic_unchanged():
    """Sans marker_token : comportement historique (SCONJ du gap)."""
    rec = _rec(_toks_gap_sconj(), [_cl("n1", (1, 2)), _cl("n2", (4, 5))],
               [EdgeRecord(source="n1", target="n2", relation="cause")])
    _, _, conns = reps_from_sentence(rec)
    assert conns[0] is not None and conns[0].root_lemma == "car"


def test_third_map_propagated():
    """EdgeRecord.third arrive dans TrainingSample.third_map."""
    import tempfile
    from pathlib import Path
    rec = _rec(_toks_gap_sconj(),
               [_cl("n1", (1, 2)), _cl("n2", (4, 5)), _cl("n3", (4, 5), "condition")],
               [EdgeRecord(source="n1", target="n2", relation="conditional_cause",
                           third={"role": "condition", "node": "n3"})])
    with tempfile.TemporaryDirectory() as d:
        sample = GCNDataLoader(Path(d))._to_sample(rec)
    assert sample.third_map.get((0, 1)) == ("condition", "n3"), sample.third_map


def test_third_map_empty_without_third():
    """Données actuelles (sans third) : third_map vide, rien ne change."""
    import tempfile
    from pathlib import Path
    rec = _rec(_toks_gap_sconj(), [_cl("n1", (1, 2)), _cl("n2", (4, 5))],
               [EdgeRecord(source="n1", target="n2", relation="cause")])
    with tempfile.TemporaryDirectory() as d:
        sample = GCNDataLoader(Path(d))._to_sample(rec)
    assert sample.third_map == {}
    assert len(sample.edge_map) == 1


def test_connector_lemma_in_embedding_vocab(tmp_path):
    """Mini-train : le lemme du connecteur ('car') entre au vocabulaire WordEmbedding."""
    import json

    from click.testing import CliRunner

    from gcn_python.training.train import train_cmd

    toks = [
        {"id": 1, "form": "ventes", "lemma": "vente", "pos": "NOUN",
         "dep_rel": "nsubj", "dep_head": 2, "morph": {}},
        {"id": 2, "form": "baissent", "lemma": "baisser", "pos": "VERB",
         "dep_rel": "root", "dep_head": 0, "morph": {}},
        {"id": 3, "form": "car", "lemma": "car", "pos": "SCONJ",
         "dep_rel": "mark", "dep_head": 2, "morph": {}},
        {"id": 4, "form": "prix", "lemma": "prix", "pos": "NOUN",
         "dep_rel": "nsubj", "dep_head": 5, "morph": {}},
        {"id": 5, "form": "montent", "lemma": "monter", "pos": "VERB",
         "dep_rel": "advcl", "dep_head": 2, "morph": {}},
    ]
    dataset = {"document": {"lang": "fr", "sentences": [{
        "id": "s1", "text": "Les ventes baissent car les prix montent.", "tokens": toks,
        "cir": {"nodes": [
            {"id": "n1", "type": "processus", "label": "baisse ventes", "token_span": [1, 2]},
            {"id": "n2", "type": "etat_local", "label": "hausse prix", "token_span": [4, 5]}],
            "edges": [{"source": "n1", "target": "n2", "relation": "cause",
                       "attributes": {"confidence": 1.0}}]}}]}}
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.json").write_text(json.dumps(dataset), encoding="utf-8")
    out = tmp_path / "model.npz"
    result = CliRunner().invoke(train_cmd, [
        "--data-dir", str(data_dir), "--epochs", "1", "--embedding-dim", "8",
        # Données jouet N=1 < N_min=30 : opt-out explicite du gate v5.
        "--min-class-count", "1", "--output", str(out)])
    assert result.exit_code == 0, f"train échoué :\n{result.output}\n{result.exception}"
    import numpy as np
    data = np.load(out, allow_pickle=True)
    vocab = json.loads(str(data["_word_emb_vocab_json"][0]))
    assert "car" in vocab["lemmas"], f"'car' absent du vocab : {vocab['lemmas']}"


def _mini_pipeline(mask=None):
    import numpy as np

    from conftest import make_test_pipeline
    pipe = make_test_pipeline()
    if mask is not None:
        pipe.edge_logit_mask = np.asarray(mask, dtype=bool)
    return pipe


def test_edge_logit_mask_restricts_emit_argmax():
    """Masque forward : classes inactives à -1e9, argmax restreint aux vues."""
    import numpy as np

    from gcn_python.constants import RELATION_TYPES
    pipe = _mini_pipeline()
    n = len(pipe.relation_types)
    mask = np.zeros(n, dtype=bool)
    mask[RELATION_TYPES.index("cause")] = True
    mask[RELATION_TYPES.index("condition")] = True
    pipe.edge_logit_mask = mask
    # Forward manuel via reps minimales : on vérifie le cache après forward.
    from gcn_python.data.loader import reps_from_sentence
    from gcn_python.data.schema import ClauseRecord, EdgeRecord, SentenceRecord, TokenRecord

    def tk(i, lemma, pos, dep, head):
        return TokenRecord(id=i, form=lemma, lemma=lemma, pos=pos,
                           dep_rel=dep, dep_head=head, morph={})
    rec = SentenceRecord(
        id="s", text="t",
        tokens=[tk(1, "vente", "NOUN", "nsubj", 2),
                tk(2, "baisser", "VERB", "root", 0),
                tk(3, "car", "SCONJ", "mark", 2),
                tk(4, "prix", "NOUN", "nsubj", 5),
                tk(5, "monter", "VERB", "advcl", 2)],
        clauses=[ClauseRecord(node_id="n1", node_type="processus", label="a",
                              token_span=(1, 2), scope="specific",
                              temporal_index=0, origin="explicit"),
                 ClauseRecord(node_id="n2", node_type="etat_local", label="b",
                              token_span=(4, 5), scope="specific",
                              temporal_index=1, origin="explicit")],
        edges=[EdgeRecord(source="n1", target="n2", relation="cause")])
    reps, _valid, conn = reps_from_sentence(rec)
    pipe.forward(reps, rec.text, connector_reps=conn)
    cached = np.asarray(pipe._cached_edge_logits)
    assert cached.shape[1] == n
    assert np.all(cached[:, ~mask] == -1e9), "classes masquées non neutralisées"
    assert np.all(np.isfinite(cached[:, mask])), "classes actives corrompues"


def test_edge_logit_mask_checkpoint_roundtrip(tmp_path):
    """Masque persisté au checkpoint et restauré au chargement."""
    import json

    import numpy as np

    from gcn_python.training.checkpoint import load_checkpoint, save_checkpoint
    pipe = _mini_pipeline()
    n = len(pipe.relation_types)
    mask = np.zeros(n, dtype=bool)
    mask[0] = True
    mask[3] = True
    pipe.edge_logit_mask = mask
    ckpt = tmp_path / "m.npz"
    save_checkpoint(pipe, ckpt)
    arch = json.loads(str(np.load(ckpt, allow_pickle=True)["_arch_json"][0]))
    assert arch["edge_logit_mask"] == [bool(x) for x in mask]
    pipe2 = _mini_pipeline()
    assert pipe2.edge_logit_mask is None
    load_checkpoint(pipe2, ckpt, trusted=True)
    assert pipe2.edge_logit_mask is not None
    assert list(pipe2.edge_logit_mask) == list(mask)


def test_edge_logit_mask_validation():
    """Masque invalide refusé à la construction."""
    import numpy as np
    import pytest
    pipe = _mini_pipeline()
    with pytest.raises(ValueError, match="incompatible"):
        from gcn_python.pipeline.cgnp import CGNPipeline
        CGNPipeline(encoder=pipe.encoder, graph=pipe.graph, vocabulary=pipe.vocabulary,
                    word_embedding=pipe.word_embedding,
                    edge_logit_mask=np.zeros(3, dtype=bool))
    with pytest.raises(ValueError, match="aucune classe active"):
        from gcn_python.pipeline.cgnp import CGNPipeline
        CGNPipeline(encoder=pipe.encoder, graph=pipe.graph, vocabulary=pipe.vocabulary,
                    word_embedding=pipe.word_embedding,
                    edge_logit_mask=np.zeros(len(pipe.relation_types), dtype=bool))


def _mkexp_dataset():
    """4 phrases, clauses identiques, seul le connecteur change (car→cause, si→condition)."""
    def sent(i, lemma, form, rel):
        toks = [
            {"id": 1, "form": "ventes", "lemma": "vente", "pos": "NOUN",
             "dep_rel": "nsubj", "dep_head": 2, "morph": {}},
            {"id": 2, "form": "baissent", "lemma": "baisser", "pos": "VERB",
             "dep_rel": "root", "dep_head": 0, "morph": {}},
            {"id": 3, "form": form, "lemma": lemma, "pos": "SCONJ",
             "dep_rel": "mark", "dep_head": 2, "morph": {}},
            {"id": 4, "form": "prix", "lemma": "prix", "pos": "NOUN",
             "dep_rel": "nsubj", "dep_head": 5, "morph": {}},
            {"id": 5, "form": "montent", "lemma": "monter", "pos": "VERB",
             "dep_rel": "advcl", "dep_head": 2, "morph": {}},
        ]
        return {"id": f"s{i}", "text": f"ventes baissent {form} prix montent",
                "tokens": toks,
                "cir": {"nodes": [
                    {"id": "n1", "type": "processus", "label": "baisse ventes",
                     "token_span": [1, 2]},
                    {"id": "n2", "type": "etat_local", "label": "hausse prix",
                     "token_span": [4, 5]}],
                    "edges": [{"source": "n1", "target": "n2", "relation": rel,
                               "marker_token": 3,
                               "attributes": {"confidence": 1.0}}]}}
    return {"document": {"lang": "fr", "sentences": [
        sent(1, "car", "car", "cause"), sent(2, "car", "car", "cause"),
        sent(3, "si", "si", "condition"), sent(4, "si", "si", "condition")]}}


def test_marker_learned_end_to_end_real():
    """ÉTAT RÉEL, zéro mock : vrai train (60 epochs, ~3 s) puis prédictions.

    Les 4 phrases ont des clauses identiques — seul le marqueur change.
    4/4 prouve que le moteur a appris car→cause et si→condition depuis
    les annotations, pas depuis du code ou des mocks.
    """
    import json

    import numpy as np
    from click.testing import CliRunner

    from gcn_python.constants import RELATION_TYPES
    from gcn_python.data.loader import GCNDataLoader, reps_from_sentence
    from gcn_python.training.train import train_cmd

    data_dir = tmp_path = None
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.json").write_text(json.dumps(_mkexp_dataset()), encoding="utf-8")
        out = tmp_path / "model.npz"
        result = CliRunner().invoke(train_cmd, [
            "--data-dir", str(data_dir), "--epochs", "60", "--embedding-dim", "8",
            "--seed", "7", "--lr", "0.005", "--output", str(out),
            # Données jouet N=2/classe < N_min=30 : opt-out explicite du gate v5.
            "--min-class-count", "2"])
        assert result.exit_code == 0, f"train réel échoué :\n{result.output}\n{result.exception}"

        from gcn_python import GCNEngine
        eng = GCNEngine.from_pretrained(out, trusted=True)
        mask = eng._pipeline.edge_logit_mask
        assert mask is not None and int(mask.sum()) == 2, "masque non restauré"
        ok = 0
        for s in GCNDataLoader(data_dir):
            reps, _valid, conn = reps_from_sentence(s.sentence)
            assert conn and conn[0] is not None and conn[0].root_lemma in ("car", "si")
            eng._pipeline.forward(reps, s.sentence.text, connector_reps=conn)
            el = np.asarray(eng._pipeline._cached_edge_logits)
            pred = RELATION_TYPES[int(np.argmax(el[0]))]
            gold = RELATION_TYPES[int(next(iter(s.edge_map.values())))]
            assert pred == gold, f"{s.sentence.id} (marqueur {conn[0].root_lemma}) : {pred} != {gold}"
            ok += 1
        assert ok == 4


def test_bridge_real_binary_no_mock():
    """ÉTAT RÉEL : vrai binaire gcn, lattice sans dictionnaire (skip si absent)."""
    import shutil
    from pathlib import Path

    import pytest

    from gcn_python.frontend.bridge import GCNLatticeParser
    gcn_bin = (shutil.which("gcn") or
               "/home/virus-one/Bureau/projet_CNM/gcn-core/target/debug/gcn")
    if not Path(gcn_bin).exists():
        pytest.skip("binaire gcn absent")
    reps, _connectors = GCNLatticeParser(gcn_bin=gcn_bin).parse(
        "Si les ventes baissent, on reduit les couts.")
    assert len(reps) == 2, f"2 clauses attendues du vrai binaire : {reps}"
    assert all(r.root_lemma not in ("", "_unknown") for r in reps)
