# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests mise-à-niveau v5 (10 familles -> cas essentiels non-régressifs)."""
import warnings
from pathlib import Path

import numpy as np


def test_edge_norm_ids():
    from gcn_python.data.edge_norm import normalize_edge
    e = normalize_edge([1, 2, {"relation": "cause", "confidence": 0.9}])
    assert e["src"] == "n001" and e["dst"] == "n002"
    e = normalize_edge({"source": "n1", "target": "n2", "relation": "cause", "confidence": 0.5})
    assert e["src"] == "n001"
    e = normalize_edge({"source": "n001", "target": "n002", "relation": "cause", "confidence": 0.5})
    assert e["src"] == "n001"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        e2 = normalize_edge({"source": "n1", "target": "n2", "relation": "cause"})
        assert e2.get("confidence") is None
        assert any("confidence" in str(x.message) for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        e3 = normalize_edge({"source": "n1", "target": "n2"})
        assert e3 is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e4 = normalize_edge({"source": "n1", "target": "n2", "relation": "cause",
                             "confidence": 0.5, "explicit": True})
        assert e4.get("negated") is None
        e5 = normalize_edge({"source": "n1", "target": "n2", "relation": "cause",
                             "confidence": 0.5, "negated": False})
        assert e5["negated"] is False


def test_edge_norm_sanitize():
    from gcn_python.data.edge_norm import normalize_edge
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = normalize_edge({"source": "n1", "target": "n2", "relation": "cause\ninjected",
                            "confidence": 0.5})
        assert "\n" not in e["relation"]


def test_hyperedge_bypass():
    from gcn_python.data.schema import SentenceRecord, ClauseRecord, EdgeRecord
    from gcn_python.data.loader import GCNDataLoader
    def _cl(nid):
        return ClauseRecord(node_id=nid, node_type="action", label=nid,
                            token_span=(0, 0), scope="specific",
                            temporal_index=0, origin="explicit")
    rec = SentenceRecord(id="s1", text="t", tokens=[], clauses=[_cl("n001"), _cl("n002"), _cl("n003")],
                         edges=[EdgeRecord(source="", target="n003", relation="cause",
                                           sources=["n001", "n003"])])
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        loader = GCNDataLoader(Path(d))
        sample = loader._to_sample(rec)
        assert len(sample.hyperedge_map) == 1
        assert len(sample.edge_map) == 0


def test_find_match_level_order():
    from gcn_python.verbalizer.instructions import CausalGraph
    g = CausalGraph()
    g.nodes = {"n1": {"label": "bateau"}, "n2": {"label": "eau", "node_type": "entite"}}
    g.add_cir({"nodes": [{"id": "n1", "label": "bateau"}, {"id": "n2", "label": "eau"}],
               "edges": [["n1", "n2", {"relation": "cause", "confidence": 0.9}]],
               "source_text": "t"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # "eau" ne doit PAS matcher "bateau" au niveau word-boundary (défaut min_level=2)
        assert g.find_effects("eau", min_level=1) == []   # exact seulement
        assert g.find_effects("eau", min_level=2) == []   # word-boundary — "eau" n'est pas un mot dans "bateau"
        # Au niveau 4 (substring explicite), le match est toujours possible
        res = g.find_effects("eau", min_level=4)
        assert len(res) == 1 and res[0][2]["_match_level"] == 4


def test_query_truncation_word_aware():
    from gcn_python.verbalizer.instructions import CausalGraph
    from gcn_python.verbalizer.query_report import QueryVerbalizer
    g = CausalGraph()
    long_text = "mot " * 100 + "\nINJECTÉ"
    g.add_cir({"nodes": [{"id": "n1", "label": "a"}, {"id": "n2", "label": "b"}],
               "edges": [["n1", "n2", {"relation": "cause", "confidence": None}]],
               "source_text": long_text})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = QueryVerbalizer(g, max_source_chars=200).causes("b")
    assert "\nINJECTÉ" not in out and "(conf=?)" in out


def test_graph_vecs_roundtrip(tmp_path):
    from gcn_python.data.graph_vecs import save_graph_vecs, load_graph_vec, stable_key, checkpoint_hash
    vecs = {"k1": np.ones(4, dtype=np.float32)}
    h = checkpoint_hash({"w": np.ones((2, 2))})
    p = save_graph_vecs(tmp_path / "v.npz", vecs, h, 4, {"k1": {"node_id": "n001"}})
    assert load_graph_vec(p, "k1", expected_checkpoint_hash=h) is not None
    assert load_graph_vec(p, "missing", expected_checkpoint_hash=h) is None


def test_checkpoint_atomic_and_arch(tmp_path):
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import save_checkpoint, load_checkpoint
    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    pipe = CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab)
    ckpt = tmp_path / "m.npz"
    save_checkpoint(pipe, ckpt)
    assert not (tmp_path / "m.tmp.npz").exists()
    import json
    arch = json.loads(str(__import__("numpy").load(ckpt, allow_pickle=True)["_arch_json"][0]))
    assert {"d_eff", "d_hidden", "vocab_size", "n_relations", "bidi_flag"} <= set(arch)
    enc2 = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr2 = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    p2 = CGNPipeline(encoder=enc2, graph=gr2, vocabulary=FeatureVocabulary())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_checkpoint(p2, ckpt, trusted=True)
    # double-absent warn attendu (pas de decoder des deux côtés)


def test_migrate_no_loss(tmp_path):
    import json
    import subprocess
    import sys
    doc = {"document": {"sentences": [{"id": "s1", "text": "t", "cir": {
        "nodes": [], "edges": [{"source": "n001", "target": "n002", "relation": "cause"}]}}]}}
    src = tmp_path / "in.json"
    dst = tmp_path / "out.json"
    src.write_text(json.dumps(doc), encoding="utf-8")
    script = Path(__file__).parents[2] / "gcn-datasets" / "scripts" / "migrate_v1_v2.py"
    r = subprocess.run([sys.executable, str(script),
                        str(src), str(dst)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads(dst.read_text(encoding="utf-8"))
    assert out["schema_version"] == "2.0"
    assert out["document"]["sentences"][0]["cir"]["edges"][0]["sources"] == ["n001"]


def _make_pipeline(d=16):
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    return CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab), vocab


def test_session_save_load_roundtrip(tmp_path):
    import numpy as np
    from gcn_python.cli.session import SessionStore
    from gcn_python.data.graph_vecs import stable_key
    sdir = tmp_path / "sess"
    s = SessionStore(sdir)
    s.graph.add_cir({"nodes": [{"id": "n1", "label": "a"}, {"id": "n2", "label": "b"}],
                     "edges": [["n1", "n2", {"relation": "cause", "confidence": 0.9}]],
                     "source_text": "a cause b"})
    s.vecs[stable_key("a cause b", 0)] = np.ones(4, dtype=np.float32)
    s.vecs_meta[stable_key("a cause b", 0)] = {"node_id": "n1"}
    s.record_exchange("pourquoi b ?", True)
    assert s.save() == sdir
    s2 = SessionStore(sdir)
    rep = s2.load()
    assert rep["graph_edges"] == 1 and rep["vecs"] == 1 and rep["history"] == 1
    assert s2.lookup("a cause b", 0) is not None
    assert s2.lookup("introuvable", 0) is None


def test_linkpred_score_and_checkpoint(tmp_path):
    import numpy as np
    from gcn_python.layer3.link_pred import (LinkPredHead, sample_negatives,
                                             candidates_within_depth)
    head = LinkPredHead(d_in=8, src_aggregation="mean")
    u = np.ones(8, dtype=np.float32)
    v = np.ones(8, dtype=np.float32)
    p = head.score(u, v)
    assert 0.0 <= p <= 1.0
    loss, grads = head.loss_and_grad(u, v, 1)
    assert loss >= 0.0 and len(grads) == 2
    head.update([g * 0.01 for g in grads], 0.01)
    negs = sample_negatives([(0, 1)], 4, neg_ratio=1.0, seed=0)
    assert len(negs) == 1 and negs[0] != (0, 1)
    assert candidates_within_depth({0: [1], 1: [2]}, 0, 2) == {1, 2}
    # round-trip checkpoint
    pipe, vocab = _make_pipeline()
    pipe.link_predictor = head
    from gcn_python.training.checkpoint import save_checkpoint, load_checkpoint
    ckpt = tmp_path / "lp.npz"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        save_checkpoint(pipe, ckpt)
        pipe2, _ = _make_pipeline()
        load_checkpoint(pipe2, ckpt, trusted=True)
    assert getattr(pipe2, "link_predictor", None) is not None
    assert np.allclose(pipe2.link_predictor.W, head.W)


def test_predict_links_requires_head():
    pipe, _ = _make_pipeline()
    try:
        pipe.predict_links([(0, 1)], node_vecs=np.ones((2, 4), dtype=np.float32))
    except RuntimeError as e:
        assert "LinkPredictor" in str(e)
    else:
        raise AssertionError("RuntimeError attendu sans tête")


def test_run_eval_respects_checkpoint_arch(tmp_path):
    """run_eval reconstruit bidi/all_pairs depuis l'arch (pas de crash ni mode faux)."""
    import json
    import numpy as np
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder
    from gcn_python.layer3.reference import RGCNLayer
    from gcn_python.pipeline.cgnp import CGNPipeline
    from gcn_python.training.checkpoint import save_checkpoint
    from gcn_python.evaluation.eval_runner import run_eval
    vocab = FeatureVocabulary()
    enc = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge_closed_loop(vocab.d_clause, 7))
    gr = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause, n_relations=22)
    pipe = CGNPipeline(encoder=enc, graph=gr, vocabulary=vocab,
                       all_pairs=True, bidirectional=True)
    ckpt = tmp_path / "arch.npz"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        save_checkpoint(pipe, ckpt)
    doc = {"document": {"sentences": [{
        "id": "s1", "text": "A cause B.",
        "tokens": [
            {"id": 1, "form": "A", "lemma": "a", "pos": "NOUN", "dep_rel": "nsubj", "dep_head": 2},
            {"id": 2, "form": "cause", "lemma": "causer", "pos": "VERB", "dep_rel": "root", "dep_head": 0},
            {"id": 3, "form": "B", "lemma": "b", "pos": "NOUN", "dep_rel": "obj", "dep_head": 2},
        ],
        "cir": {"nodes": [
            {"id": "n001", "type": "action", "label": "causer(a)", "token_span": [1, 2]},
            {"id": "n002", "type": "etat", "label": "b", "token_span": [3, 3]},
        ], "edges": [
            {"sources": ["n001"], "target": "n002", "relation": "cause", "confidence": 0.9},
        ]},
    }]}}
    ddir = tmp_path / "data"
    ddir.mkdir()
    (ddir / "s.json").write_text(json.dumps(doc), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = run_eval(ddir, ckpt)
    assert report["n_samples"] == 1, report
    assert report["node_macro_f1"] is not None


def test_decoder_source_bias_and_is_inferred():
    import numpy as np
    from gcn_python.verbalizer.trainable import TrainableDecoder, SurfaceVocabulary
    from gcn_python.pipeline.ir_emitter import emit
    vocab = SurfaceVocabulary()
    vocab.build(["les ventes baissent"])
    dec = TrainableDecoder(vocab, d_hidden=8, d_in=6)
    rng = np.random.default_rng(0)
    embs = rng.normal(size=(3, 6)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out_plain = dec.decode(embs)
        out_bias = dec.decode(embs, source_bias=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert isinstance(out_plain, str) and isinstance(out_bias, str)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dec.decode(embs, source_bias=np.ones(2, dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError attendu (biais de mauvaise taille)")
    cir = emit("t", ["action", "etat"], ["a", "b"], [(1, 1), (2, 2)],
               ["specific", "specific"], [(0, 1, "cause", 0.9, False, None)],
               node_origins=["explicit", "inferred"],
               node_inferred=[False, True])
    assert cir["nodes"][0]["is_inferred"] is False
    assert cir["nodes"][1]["is_inferred"] is True
