# Copyright 2026 Michel Tendeng
# SPDX-License-Identifier: Apache-2.0
"""Tests P4 quals — têtes polarity/voice/modality en dortoir N_min."""
import numpy as np

from gcn_python.constants import QUALIFIERS


def test_qualifiers_values():
    assert QUALIFIERS["polarity"] == ["positive", "negative"]
    assert QUALIFIERS["voice"] == ["active", "passive"]
    assert QUALIFIERS["modality"] == ["indicative", "subjunctive", "conditional", "imperative"]


def test_qual_heads_off_by_default():
    """Sans --qual-heads : pas de couches, pas de logits, pas de loss."""
    from conftest import make_test_pipeline
    pipe = make_test_pipeline()
    assert getattr(pipe.encoder, 'qual_heads', False) is False
    assert not hasattr(pipe.encoder, '_qual_layers')


def test_qual_heads_shapes_and_update():
    """Têtes quals : forward/backward/update cohérents."""
    from conftest import make_word_embedding
    from gcn_python.constants import NODE_TYPES
    from gcn_python.layer1.features import FeatureVocabulary
    from gcn_python.layer2.reference import MLPEncoder

    vocab = FeatureVocabulary()
    we = make_word_embedding()
    d_eff = vocab.d_clause_effective(we.d_emb)
    enc = MLPEncoder(d_clause=d_eff,
                     d_edge=vocab.d_edge_closed_loop(d_eff, len(NODE_TYPES), we.d_emb),
                     qual_heads=True, seed=7)
    rng = np.random.RandomState(7)
    x = rng.randn(enc._qual_layers["polarity"][0].W.shape[1]).astype(np.float32)
    out = enc.forward_quals(x)
    assert out["polarity"].shape == (2,)
    assert out["voice"].shape == (2,)
    assert out["modality"].shape == (4,)
    before = {k: [lay.W.copy() for lay in v] for k, v in enc._qual_layers.items()}
    grads, dxs = enc.backward_quals(
        {k: np.zeros_like(v) for k, v in out.items()})
    assert set(dxs) == {"polarity", "voice", "modality"}
    enc.update_quals(grads, lr=0.01)
    # gradient nul → poids inchangés (pas de mouvement fantôme)
    for k, old_ws in before.items():
        for a, b in zip(old_ws, [lay.W for lay in enc._qual_layers[k]], strict=False):
            assert np.allclose(a, b)


def _mkexp_quals():
    def sent(i, neg, voice, modal, morph2, morph5):
        toks = [
            {"id": 1, "form": "ventes", "lemma": "vente", "pos": "NOUN",
             "dep_rel": "nsubj", "dep_head": 2, "morph": {}},
            {"id": 2, "form": "baissent", "lemma": "baisser", "pos": "VERB",
             "dep_rel": "root", "dep_head": 0, "morph": morph2},
            {"id": 3, "form": "car", "lemma": "car", "pos": "SCONJ",
             "dep_rel": "mark", "dep_head": 2, "morph": {}},
            {"id": 4, "form": "prix", "lemma": "prix", "pos": "NOUN",
             "dep_rel": "nsubj", "dep_head": 5, "morph": {}},
            {"id": 5, "form": "montent", "lemma": "monter", "pos": "VERB",
             "dep_rel": "advcl", "dep_head": 2, "morph": morph5},
        ]
        return {"id": f"s{i}", "text": "t", "tokens": toks,
                "cir": {"nodes": [
                    {"id": "n1", "type": "processus", "label": "a", "token_span": [1, 2]},
                    {"id": "n2", "type": "etat_local", "label": "b", "token_span": [4, 5]}],
                    "edges": [{"source": "n1", "target": "n2", "relation": "cause",
                               "marker_token": 3, "polarity": neg, "voice": voice,
                               "modality": modal, "attributes": {"confidence": 1.0}}]}}
    return {"document": {"sentences": [
        sent(1, "positive", "active", "indicative", {}, {}),
        sent(2, "positive", "active", "indicative", {}, {}),
        sent(3, "negative", "passive", "conditional",
             {"Polarity": "Neg", "Voice": "Pass", "Mood": "Cnd"}, {"Voice": "Pass"}),
        sent(4, "negative", "passive", "conditional",
             {"Polarity": "Neg", "Voice": "Pass", "Mood": "Cnd"}, {"Voice": "Pass"})]}}


def test_quals_learned_end_to_end_real():
    """ÉTAT RÉEL : les 3 têtes apprennent (morph distinctif) — backward ET accumulate."""
    import json
    import tempfile
    from pathlib import Path

    from click.testing import CliRunner

    from gcn_python.training.train import train_cmd

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.json").write_text(json.dumps(_mkexp_quals()), encoding="utf-8")
        for _mb in (4, 1):  # v5.9 : les deux chemins entraînent les têtes
            out = tmp_path / f"model_mb{_mb}.npz"
            result = CliRunner().invoke(train_cmd, [
                "--data-dir", str(data_dir), "--epochs", "60", "--embedding-dim", "8",
                "--seed", "7", "--lr", "0.005", "--output", str(out),
                "--min-class-count", "1", "--qual-heads",
                "--mini-batch-size", str(_mb)])
            assert result.exit_code == 0, f"train échoué :\n{result.output}\n{result.exception}"

            from gcn_python import GCNEngine
            eng = GCNEngine.from_pretrained(out, trusted=True)
            assert eng._pipeline.encoder.qual_heads is True
            ok = tot = 0
            from gcn_python.data.loader import GCNDataLoader, reps_from_sentence
            for smp in GCNDataLoader(data_dir):
                reps, _, conn = reps_from_sentence(smp.sentence)
                eng._pipeline.forward(reps, smp.sentence.text, connector_reps=conn)
                for qn, gold in smp.qual_map[(0, 1)].items():
                    pred = QUALIFIERS[qn][int(np.argmax(
                        np.asarray(eng._pipeline._cached_qual_logits[qn][0])))]
                    tot += 1
                    ok += (pred == QUALIFIERS[qn][gold])
                    assert pred == QUALIFIERS[qn][gold], (
                        f"[mb={_mb}] {smp.sentence.id}/{qn} : {pred} != {QUALIFIERS[qn][gold]}")
            assert (ok, tot) == (12, 12)
