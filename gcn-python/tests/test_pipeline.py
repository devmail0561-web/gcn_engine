import numpy as np
import pytest
from gcn_python.layer1.features import FeatureVocabulary, vectorize_edge
from gcn_python.layer1.representation import UDRepresentation
from gcn_python.layer2.reference import MLPEncoder
from gcn_python.layer3.reference import RGCNLayer
from gcn_python.pipeline.cgnp import CGNPipeline


def make_rep(lang: str = "fr") -> UDRepresentation:
    return UDRepresentation(
        tokens=[{"lemma": "baisser", "pos": "VERB", "dep_rel": "root", "morph": {}}],
        root_lemma="baisser",
        root_pos="VERB",
        root_dep_rel="root",
        root_morph={"Tense": "Pres"},
        subject_pos="NOUN",
        has_object=False,
        has_advcl=False,
        has_temporal_obl=False,
        token_span=(1, 2),
        lang=lang,
    )


def make_pipeline(lang: str = "fr") -> CGNPipeline:
    vocab = FeatureVocabulary()
    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge)
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause)
    return CGNPipeline(encoder=encoder, graph=graph, lang=lang, vocabulary=vocab)


def test_forward_returns_cir():
    pipeline = make_pipeline()
    rep = make_rep()
    result = pipeline.forward([rep], "Si les ventes baissent, on réduit les coûts.")
    assert "source_lang" in result
    assert "nodes" in result
    assert "edges" in result
    assert result["source_text"] != ""


def test_forward_empty_reps():
    pipeline = make_pipeline()
    result = pipeline.forward([], "")
    assert result["nodes"] == []


def test_forward_single_rep():
    pipeline = make_pipeline()
    result = pipeline.forward([make_rep()], "Il court.")
    assert len(result["nodes"]) >= 1


def test_metadata_pipeline_field():
    pipeline = make_pipeline()
    result = pipeline.forward([make_rep()], "Il travaille.")
    assert "pipeline" in result["metadata"]
    assert any(
        "layer" in p.lower() or "cgnp" in p.lower()
        for p in result["metadata"]["pipeline"]
    )
    assert not any("spacy" in p.lower() for p in result["metadata"]["pipeline"])


def test_forward_two_reps():
    pipeline = make_pipeline()
    rep1, rep2 = make_rep(), make_rep()
    result = pipeline.forward([rep1, rep2], "Les ventes baissent puis on réduit.")
    assert len(result["nodes"]) == 2
    assert len(result["edges"]) >= 1


def test_forward_connector_slot_nonzero():
    """Avec un connector_rep, les features UPOS du connecteur ne sont pas toutes à zéro."""
    vocab = FeatureVocabulary()

    rep1 = make_rep()
    rep2 = make_rep()
    connector = UDRepresentation(
        tokens=[{"lemma": "parce", "pos": "SCONJ", "dep_rel": "mark", "morph": {}}],
        root_lemma="parce", root_pos="SCONJ", root_dep_rel="mark",
        root_morph={}, subject_pos=None,
        has_object=False, has_advcl=False, has_temporal_obl=False,
        token_span=(3, 3), lang="fr",
    )

    vec_with = vectorize_edge(rep1, rep2, connector, 0, 1, 2, vocab)
    vec_without = vectorize_edge(rep1, rep2, None, 0, 1, 2, vocab)

    n_upos = len(vocab.upos_tags)
    d_conn = vocab.d_conn
    upos_with = vec_with[-d_conn: -d_conn + n_upos]
    upos_without = vec_without[-d_conn: -d_conn + n_upos]

    assert not np.all(upos_with == 0), "La UPOS du connecteur doit être encodée"
    assert np.all(upos_without == 0), "Sans connecteur, la UPOS doit être nulle"


def test_forward_rgcn_dout_mismatch_raises():
    """Un RGCNLayer avec d_out ≠ d_clause doit lever ValueError dès la construction."""
    vocab = FeatureVocabulary()
    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge)
    graph = RGCNLayer(d_in=vocab.d_clause, d_out=vocab.d_clause + 1)
    with pytest.raises(ValueError, match="d_out=.*≠.*d_clause"):
        CGNPipeline(encoder=encoder, graph=graph, lang="fr", vocabulary=vocab)


def test_forward_position_features_noncontiguous():
    """clause_positions corrige les features de distance pour clauses non-contiguës."""
    vocab = FeatureVocabulary()
    rep1 = make_rep()
    rep2 = make_rep()

    vec_filtered = vectorize_edge(rep1, rep2, None, 0, 1, 2, vocab)
    vec_original = vectorize_edge(rep1, rep2, None, 0, 2, 3, vocab)

    assert not np.allclose(vec_filtered[-2:], vec_original[-2:]), \
        "Les features de position doivent différer selon les positions originales"


# ─── T-1 : pipeline lang="en" ────────────────────────────────────────────────

def test_forward_en_returns_valid_cir():
    """T-1 : le pipeline EN produit un CausalIR valide."""
    pipeline = make_pipeline(lang="en")
    rep = make_rep(lang="en")
    result = pipeline.forward([rep], "Sales fall because costs rise.")
    assert "source_lang" in result
    assert "nodes" in result
    assert len(result["nodes"]) >= 1


def test_forward_en_condition_label():
    """T-1 : le label de condition EN est 'hidden_cause(?)' et non 'cause_cachée(?)'."""
    from gcn_python.pipeline.label_builder import build_label
    rep = make_rep(lang="en")
    label = build_label(rep, "condition")
    assert label == "hidden_cause(?)", f"Attendu 'hidden_cause(?)', obtenu {label!r}"


# ─── T-2 : backward() sans backward_message_pass ────────────────────────────

def test_backward_with_pt_graph_does_not_crash():
    """T-2 : backward() ne crashe pas quand le graph n'a pas de backward_message_pass.

    Utilise RGCNLayerPT (après fix) qui n'a plus la méthode. Si torch n'est pas
    installé, teste via un stub minimal satisfaisant le Protocol.
    """
    try:
        from gcn_python.layer3.pytorch_rgcn import RGCNLayerPT
        torch_available = True
    except ImportError:
        torch_available = False

    vocab = FeatureVocabulary()

    if torch_available:
        graph = RGCNLayerPT(d_in=vocab.d_clause, d_out=vocab.d_clause,
                            n_relations=11, device="cpu")
        assert not hasattr(graph, 'backward_message_pass'), \
            "RGCNLayerPT ne doit plus avoir backward_message_pass"
    else:
        # Stub minimal sans backward_message_pass
        class _StubGraph:
            d_in = vocab.d_clause
            d_out = vocab.d_clause

            def message_pass(self, node_features, edge_index, edge_types):
                return node_features

            def parameters(self):
                return []

            def update(self, grads, lr):
                pass

        graph = _StubGraph()

    encoder = MLPEncoder(d_clause=vocab.d_clause, d_edge=vocab.d_edge, seed=0)
    pipeline = CGNPipeline(encoder=encoder, graph=graph, lang="fr", vocabulary=vocab)

    rep1, rep2 = make_rep(), make_rep()
    pipeline.forward([rep1, rep2], "test")
    d_node = np.zeros((2, 7), dtype=np.float32)
    d_edge = np.zeros((1, 11), dtype=np.float32)
    pipeline.backward(d_node, d_edge, lr=0.01)
